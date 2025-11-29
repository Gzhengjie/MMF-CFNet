import torch
import torch.nn as nn
from torch.nn import init
from torch.nn.parameter import Parameter

try:
    from inplace_abn import InPlaceABN
except ImportError:
    InPlaceABN = None


class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):

        if use_batchnorm == "inplace" and InPlaceABN is None:
            raise RuntimeError(
                "In order to use `use_batchnorm='inplace'` inplace_abn package must be installed. "
                + "To install see: https://github.com/mapillary/inplace_abn"
            )

        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        if use_batchnorm == "inplace":
            bn = InPlaceABN(out_channels, activation="leaky_relu", activation_param=0.0)
            relu = nn.Identity()

        elif use_batchnorm and use_batchnorm != "inplace":
            bn = nn.BatchNorm2d(out_channels)

        else:
            bn = nn.Identity()

        super(Conv2dReLU, self).__init__(conv, bn, relu)


class SCSEModule(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid(),
        )
        self.sSE = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())

    def forward(self, x):
        # print("在注意力机制内部", x.shape)
        return x * self.cSE(x) + x * self.sSE(x)


class CBAMChannel(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(CBAMChannel, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)


class CBAMSpatial(nn.Module):
    def __init__(self, in_channels, kernel_size=7):
        super(CBAMSpatial, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(out)
        return x * self.sigmoid(out)


class CBAM(nn.Module):
    """
    Woo S, Park J, Lee J Y, et al. Cbam: Convolutional block attention module[C]
    //Proceedings of the European conference on computer vision (ECCV).
    """
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ChannelGate = CBAMChannel(in_channels, reduction)
        self.SpatialGate = CBAMSpatial(kernel_size)

    def forward(self, x):
        x = self.ChannelGate(x)
        x = self.SpatialGate(x)
        return x


class ECAM(nn.Module):
    """
    Ensemble Channel Attention Module for UNetPlusPlus.
    Fang S, Li K, Shao J, et al. SNUNet-CD: A Densely Connected Siamese Network for Change Detection of VHR Images[J].
    IEEE Geoscience and Remote Sensing Letters, 2021.
    Not completely consistent, to be improved.
    """
    def __init__(self, in_channels, out_channels, map_num=4):
        super(ECAM, self).__init__()
        self.ca1 = CBAMChannel(in_channels * map_num, reduction=16)
        self.ca2 = CBAMChannel(in_channels, reduction=16 // 4)
        self.up = nn.ConvTranspose2d(in_channels * map_num, in_channels * map_num, 2, stride=2)
        self.conv_final = nn.Conv2d(in_channels * map_num, out_channels, kernel_size=1)

    def forward(self, x):
        """
        x (list[tensor] or tuple(tensor))
        """
        out = torch.cat(x, 1)
        intra = torch.sum(torch.stack(x), dim=0)
        ca2 = self.ca2(intra)
        out = self.ca1(out) * (out + ca2.repeat(1, 4, 1, 1))
        out = self.up(out)
        out = self.conv_final(out)
        return out


class SEModule(nn.Module):
    """
    Hu J, Shen L, Sun G. Squeeze-and-excitation networks[C]
    //Proceedings of the IEEE conference on computer vision and pattern recognition. 2018: 7132-7141.
    """
    def __init__(self, in_channels, reduction=16):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # print("x.shape--->", x.shape)
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ArgMax(nn.Module):

    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.argmax(x, dim=self.dim)


class Clamp(nn.Module):
    def __init__(self, min=0, max=1):
        super().__init__()
        self.min, self.max = min, max

    def forward(self, x):
        return torch.clamp(x, self.min, self.max)


class ShuffleAttention(nn.Module):
    def __init__(self, channels, reduction=8, num_groups=4):
        """
        Shuffle Attention 模块。

        参数:
            channels (int): 输入通道数。
            reduction (int): 通道缩减比例。
            num_groups (int): 通道分组的组数。
        """
        super(ShuffleAttention, self).__init__()
        self.num_groups = num_groups
        self.group_channels = channels // num_groups  # 每组的通道数

        # 通道注意力的 squeeze 操作（通过全局平均池化）
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        # 注意力权重生成网络
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False)

        # 激活函数
        self.sigmoid = nn.Sigmoid()

    def channel_shuffle(self, x):
        """
        通道打乱操作（Channel Shuffle）。

        参数:
            x (Tensor): 输入张量，形状为 (B, C, H, W)。

        返回:
            Tensor: 通道打乱后的张量，形状与输入一致。
        """
        B, C, H, W = x.size()
        x = x.view(B, self.num_groups, self.group_channels, H, W)  # 分组
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # 在组间进行 shuffle
        return x.view(B, C, H, W)  # 恢复原始形状

    def forward(self, x):
        # 通道打乱
        shuffled_x = self.channel_shuffle(x)

        # 全局平均池化（Squeeze 操作）
        attention = self.global_avg_pool(shuffled_x)

        # MLP（多层感知机）生成注意力权重
        attention = self.fc1(attention)
        attention = self.relu(attention)
        attention = self.fc2(attention)

        # 通过 Sigmoid 激活获得注意力权重
        attention = self.sigmoid(attention)

        # 应用注意力权重到输入特征
        out = x * attention
        return out


class Activation(nn.Module):

    def __init__(self, name, **params):

        super().__init__()

        if name is None or name == 'identity':
            self.activation = nn.Identity(**params)
        elif name == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif name == 'softmax2d':
            self.activation = nn.Softmax(dim=1, **params)
        elif name == 'softmax':
            self.activation = nn.Softmax(**params)
        elif name == 'logsoftmax':
            self.activation = nn.LogSoftmax(**params)
        elif name == 'tanh':
            self.activation = nn.Tanh()
        elif name == 'argmax':
            self.activation = ArgMax(**params)
        elif name == 'argmax2d':
            self.activation = ArgMax(dim=1, **params)
        elif name == 'clamp':
            self.activation = Clamp(**params)
        elif callable(name):
            self.activation = name(**params)
        else:
            raise ValueError('Activation should be callable/sigmoid/softmax/logsoftmax/tanh/None; got {}'.format(name))

    def forward(self, x):
        return self.activation(x)


class Attention(nn.Module):

    def __init__(self, name, **params):
        super().__init__()

        if name is None:
            self.attention = nn.Identity(**params)
        elif name == 'scse':
            self.attention = SCSEModule(**params)
        elif name == 'cbam_channel':
            self.attention = CBAMChannel(**params)
        elif name == 'cbam_spatial':
            self.attention = CBAMSpatial(**params)
        elif name == 'cbam':
            self.attention = CBAM(**params)
        elif name == 'se':
            self.attention = SEModule(**params)
        elif name == 'sha':
            self.attention = ShuffleAttention(**params)
        else:
            raise ValueError("Attention {} is not implemented".format(name))

    def forward(self, x):
        return self.attention(x)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)
