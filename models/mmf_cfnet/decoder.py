import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import Decoder, modules, Attention
from ..swinTransformer import SwinTransformer


class PSPBlock(nn.Module):

    def __init__(self, in_channels, out_channels, pool_size, use_bathcnorm=True):
        super().__init__()
        if pool_size == 1:
            use_bathcnorm = False  # PyTorch does not support BatchNorm for 1x1 shape
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(pool_size, pool_size)),
            modules.Conv2dReLU(in_channels, out_channels, (1, 1), use_batchnorm=use_bathcnorm)
        )

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        x = self.pool(x)
        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)
        return x


class PSPModule(nn.Module):
    def __init__(self, in_channels, out_channels, sizes=(1, 2, 3, 6), use_bathcnorm=True):
        super().__init__()

        self.blocks = nn.ModuleList([
            PSPBlock(in_channels, out_channels, size, use_bathcnorm=use_bathcnorm) for size in sizes
        ])

    def forward(self, x):
        xs = [block(x) for block in self.blocks] + [x]
        x = torch.cat(xs, dim=1)
        return x


class FPNBlock(nn.Module):
    def __init__(self, pyramid_channels, skip_channels):
        super().__init__()
        self.skip_conv = nn.Sequential(
                nn.Conv2d(skip_channels, pyramid_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(pyramid_channels),   # adjust to "SynchronizedBatchNorm2d" if you need.
                nn.ReLU(inplace=True)
            )

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        skip = self.skip_conv(skip)
        x = x + skip
        return x


class MergeBlock(nn.Module):
    def __init__(self, policy):
        super().__init__()
        if policy not in ["add", "cat"]:
            raise ValueError(
                "`merge_policy` must be one of: ['add', 'cat'], got {}".format(
                    policy
                )
            )
        self.policy = policy

    def forward(self, x):
        if self.policy == 'add':
            return sum(x)
        elif self.policy == 'cat':
            return torch.cat(x, dim=1)
        else:
            raise ValueError(
                "`merge_policy` must be one of: ['add', 'cat'], got {}".format(self.policy)
            )


class MMF_CFNetDecoder(Decoder):
    def __init__(
            self,
            encoder_channels,
            encoder_channels1=None,
            encoder_channels2=None,
            encoder_channels3=None,
            encoder_depth=5,
            embed_dim = 768,
            psp_channels=512,
            pyramid_channels=256,
            segmentation_channels=128,
            dropout=0.2,
            merge_policy="cat",
            fusion_form="concat",
    ):
        super().__init__()

        self.out_channels = segmentation_channels if merge_policy == "add" else segmentation_channels * 4
        if encoder_depth < 3:
            raise ValueError("Encoder depth for UPerNet decoder cannot be less than 3, got {}.".format(encoder_depth))

        encoder_channels = encoder_channels[::-1]
        encoder_channels = encoder_channels[:encoder_depth + 1]

        encoder_channels1 = encoder_channels1[::-1]
        encoder_channels1 = encoder_channels1[:encoder_depth + 1]

        encoder_channels2 = encoder_channels2[::-1]
        encoder_channels2 = encoder_channels2[:encoder_depth + 1]

        encoder_channels3 = encoder_channels3[::-1]
        encoder_channels3 = encoder_channels3[:encoder_depth + 1]

        zipped = zip(encoder_channels, encoder_channels1, encoder_channels2, encoder_channels3)
        mapped = map(sum, zipped)
        mapped = list(mapped)
        mapped[0] = encoder_channels[0]
        encoder_channels = tuple(mapped)

        self.tiaozheng1 = nn.Conv2d(encoder_channels1[0], encoder_channels[0], kernel_size=1)
        self.tiaozheng2 = nn.Conv2d(encoder_channels2[0], encoder_channels[0], kernel_size=1)
        self.tiaozheng3 = nn.Conv2d(encoder_channels3[0], encoder_channels[0], kernel_size=1)

        # adjust encoder channels according to fusion form
        self.fusion_form = fusion_form


        self.p4 = FPNBlock(pyramid_channels, encoder_channels[1])
        self.p3 = FPNBlock(pyramid_channels, encoder_channels[2])
        self.p2 = FPNBlock(pyramid_channels, encoder_channels[3])

        self.sa4 = Attention("sha", channels=encoder_channels[1])
        self.sa3 = Attention("sha", channels=encoder_channels[2])
        self.sa2 = Attention("sha", channels=encoder_channels[3])

        self.cbam_s4 = Attention("cbam_spatial", in_channels=pyramid_channels)
        self.cbam_s3 = Attention("cbam_spatial", in_channels=pyramid_channels)
        self.cbam_s2 = Attention("cbam_spatial", in_channels=pyramid_channels)

        self.merge = MergeBlock(merge_policy)

        self.conv_last = modules.Conv2dReLU(self.out_channels, pyramid_channels, 1)
        self.dropout = nn.Dropout2d(p=dropout, inplace=True)

        # self.conv_chenge = nn.Conv2d(in_channels=encoder_channels[0], out_channels=embed_dim, kernel_size=1, stride=1, padding=0)

        self.conv_chenge = modules.Conv2dReLU(in_channels=encoder_channels[0], out_channels=embed_dim, kernel_size=1, use_batchnorm=True,)

        self.swin_transform = SwinTransformer(patch_size=1, in_chans=embed_dim, num_classes=2,
                     embed_dim=embed_dim, depths=(2,), num_heads=(3,),
                     window_size=7, mlp_ratio=4., qkv_bias=True,
                     drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, patch_norm=True,
                     use_checkpoint=False)
        self.conv_end = modules.Conv2dReLU(in_channels=embed_dim * 2, out_channels=pyramid_channels, kernel_size=1, use_batchnorm=True,)


    def forward(self, features):

        features[1][-1] = self.tiaozheng1(features[1][-1])
        features[2][-1] = self.tiaozheng2(features[2][-1])
        features[3][-1] = self.tiaozheng1(features[3][-1])

        feature1 = torch.cat((features[0][-1], features[1][-1]), dim=3)
        feature2 = torch.cat((features[2][-1], features[3][-1]), dim=3)
        feature = torch.cat((feature1, feature2), dim=2)


        feature = self.conv_chenge(feature)
        swin_features = self.swin_transform(feature)


        features = self.aggregation_layer(features[0], features[1], features[2], features[3],
                                          self.fusion_form, ignore_original_img=True)

        c2, c3, c4, c5 = features[-4:]



        p5 = self.conv_end(swin_features)

        c4 = self.sa4(c4)
        p4 = self.p4(p5, c4)
        p4 = self.cbam_s4(p4)

        c3 = self.sa3(c3)
        p3 = self.p3(p4, c3)
        p3 = self.cbam_s3(p3)

        c2 = self.sa2(c2)
        p2 = self.p2(p3, c2)
        p2 = self.cbam_s2(p2)

        output_size = p2.size()[2:]
        feature_pyramid = [nn.functional.interpolate(p, output_size,
                                                     mode='bilinear', align_corners=False) for p in [p5, p4, p3, p2]]
        x = self.merge(feature_pyramid)
        x = self.conv_last(x)
        x = self.dropout(x)

        return x
