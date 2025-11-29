from typing import Optional, Union

from ..base import ClassificationHead, SegmentationHead, SegmentationModel
from ..encoders import get_encoder
from .decoder import MMF_CFNetDecoder


class MMF_CFNet(SegmentationModel):
    def __init__(
        self,
        encoder_name: str = "resnet18",
        encoder_name1: str = "resnet18",
        encoder_name2: str = "resnet18",
        encoder_name3: str = "resnet18",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_psp_channels: int = 512,
        decoder_pyramid_channels: int = 256,
        decoder_segmentation_channels: int = 256,
        decoder_merge_policy: str = "add",
        decoder_dropout: float = 0.2,
        in_channels: int = 1,
        in_channels1: int = 2,
        in_channels2: int = 3,
        in_channels3: int = 4,
        classes: int = 1,
        activation: Optional[str] = None,
        upsampling: int = 4,
        aux_params: Optional[dict] = None,
        siam_encoder: bool = True,
        fusion_form: str = "concat",
        **kwargs
    ):
        super().__init__()

        self.siam_encoder = siam_encoder

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )

        if not self.siam_encoder:
            self.encoder_non_siam1 = get_encoder(
                encoder_name1,
                in_channels=in_channels1,
                depth=encoder_depth,
                weights=encoder_weights,
            )

        if not self.siam_encoder:
            self.encoder_non_siam2 = get_encoder(
                encoder_name2,
                in_channels=in_channels2,
                depth=encoder_depth,
                weights=encoder_weights,
            )

        if not self.siam_encoder:
            self.encoder_non_siam3 = get_encoder(
                encoder_name3,
                in_channels=in_channels3,
                depth=encoder_depth,
                weights=encoder_weights,
            )

        self.decoder = MMF_CFNetDecoder(
            encoder_channels=self.encoder.out_channels,
            encoder_depth=encoder_depth,
            psp_channels=decoder_psp_channels,
            pyramid_channels=decoder_pyramid_channels,
            segmentation_channels=decoder_segmentation_channels,
            dropout=decoder_dropout,
            merge_policy=decoder_merge_policy,
            fusion_form=fusion_form,
        )

        if not self.siam_encoder:
            self.decoder = MMF_CFNetDecoder(
                encoder_channels=self.encoder.out_channels,
                encoder_channels1=self.encoder_non_siam1.out_channels,
                encoder_channels2=self.encoder_non_siam2.out_channels,
                encoder_channels3=self.encoder_non_siam3.out_channels,
                encoder_depth=encoder_depth,
                psp_channels=decoder_psp_channels,
                pyramid_channels=decoder_pyramid_channels,
                segmentation_channels=decoder_segmentation_channels,
                dropout=decoder_dropout,
                merge_policy=decoder_merge_policy,
                fusion_form=fusion_form,
            )

        self.segmentation_head = SegmentationHead(
            in_channels=self.decoder.out_channels,
            out_channels=classes,
            activation=activation,
            kernel_size=1,
            upsampling=upsampling,
            align_corners=False,
        )

        if aux_params is not None:
            self.classification_head = ClassificationHead(
                in_channels=self.encoder.out_channels[-1], **aux_params
            )
        else:
            self.classification_head = None

        self.name = "MMF-CFNet-{}".format(encoder_name)
        self.initialize()

if __name__ == "__main__":


    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    input1 = torch.randn(1, 11, 256, 256).to(device)
    net = MMF_CFNet(
        encoder_name="resnet18",
        encoder_name1="resnet18",
        encoder_name2="resnet18",
        encoder_name3="resnet18",
        encoder_depth=5,
        in_channels=5,
        in_channels1=2,
        in_channels2=1,
        in_channels3=3,
        encoder_weights="imagenet",
        fusion_form="concat",
        siam_encoder=False
    ).to(device)
    # res = net.forward(input1, input2)
    res = net.forward(input1)
    print(res.shape)