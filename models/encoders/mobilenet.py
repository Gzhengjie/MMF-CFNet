""" Each encoder should have following attributes and methods and be inherited from `_base.EncoderMixin`

Attributes:

    _out_channels (list of int): specify number of channels for each encoder feature tensor
    _depth (int): specify number of stages in decoder (in other words number of downsampling operations)
    _in_channels (int): default number of input channels in first Conv2d layer for encoder (usually 3)

Methods:

    forward(self, x: torch.Tensor)
        produce list of features of different spatial resolutions, each feature is a 4D torch.tensor of
        shape NCHW (features should be sorted in descending order according to spatial resolution, starting
        with resolution same as input `x` tensor).

        Input: `x` with shape (1, 3, 64, 64)
        Output: [f0, f1, f2, f3, f4, f5] - features with corresponding shapes
                [(1, 3, 64, 64), (1, 64, 32, 32), (1, 128, 16, 16), (1, 256, 8, 8),
                (1, 512, 4, 4), (1, 1024, 2, 2)] (C - dim may differ)

        also should support number of features according to specified depth, e.g. if depth = 5,
        number of feature tensors = 6 (one with same resolution as input and 5 downsampled),
        depth = 3 -> number of feature tensors = 4 (one with same resolution as input and 3 downsampled).
"""

import torchvision
import torch.nn as nn
from torchvision.models import mobilenet_v3_large, mobilenet_v3_small

from ._base import EncoderMixin
from .mobilenet_transformer import TransformerEncoder


class MobileNetV2Encoder(torchvision.models.MobileNetV2, EncoderMixin):

    def __init__(self, out_channels, depth=5, **kwargs):
        super().__init__(**kwargs)
        self._depth = depth
        self._out_channels = out_channels
        self._in_channels = 3
        del self.classifier

    def get_stages(self):
        return [
            nn.Identity(),
            self.features[:2],
            self.features[2:4],
            self.features[4:7],
            self.features[7:14],
            self.features[14:],
        ]

    def forward(self, x):
        stages = self.get_stages()

        features = []
        for i in range(self._depth + 1):
            x = stages[i](x)
            features.append(x)

        return features

    def load_state_dict(self, state_dict, **kwargs):
        state_dict.pop("classifier.1.bias", None)
        state_dict.pop("classifier.1.weight", None)
        super().load_state_dict(state_dict, **kwargs)


class MobileNetV3Encoder(nn.Module, EncoderMixin):
    def __init__(self, version="large", pretrained=True, out_channels=None, depth=5):
        """
        MobileNetV3 Encoder.

        Args:
            version (str): "large" or "small" to specify the version of MobileNetV3.
            pretrained (bool): Whether to load pretrained weights from torchvision.
            out_channels (list): Output channels for each stage.
            depth (int): Number of stages to return features for.
        """
        super().__init__()

        self.version = version
        if version == "large":
            self.model = mobilenet_v3_large(pretrained=pretrained)
            default_out_channels = [16, 24, 40, 80, 112, 160]
        elif version == "small":
            self.model = mobilenet_v3_small(pretrained=pretrained)
            default_out_channels = [16, 16, 16, 24, 48, 576]
        else:
            raise ValueError("Version must be 'large' or 'small'")

        self._depth = depth
        self._in_channels = 3
        self._out_channels = out_channels or default_out_channels

        # Get the feature extractor (removing the classifier)
        self.features = self.model.features
        del self.model.classifier

    def get_stages(self):
        if self.version == "large":
            return [
                nn.Identity(),
                self.features[:2],
                self.features[2:4],
                self.features[4:7],
                self.features[7:12],
                self.features[12:],
            ]
        return [
            nn.Identity(),
            self.features[:1],
            self.features[1:2],
            self.features[2:4],
            self.features[4:9],
            self.features[9:],
        ]


    def forward(self, x):
        """
        Forward pass of the encoder.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            List[torch.Tensor]: Features from the specified stages.
        """
        stages = self.get_stages()

        features = []
        for i in range(self._depth + 1):
            x = stages[i](x)
            features.append(x)

        return features

    def load_state_dict(self, state_dict, **kwargs):
        state_dict.pop("classifier.1.bias", None)
        state_dict.pop("classifier.1.weight", None)
        super().load_state_dict(state_dict, **kwargs)


from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_config import MODEL_SPECS, get_config


def make_divisible(
        value: float,
        divisor: int,
        min_value: Optional[float] = None,
        round_down_protect: bool = True,
) -> int:
    """
    This function is copied from here
    "https://github.com/tensorflow/models/blob/master/official/vision/modeling/layers/nn_layers.py"

    This is to ensure that all layers have channels that are divisible by 8.

    Args:
        value: A `float` of original value.
        divisor: An `int` of the divisor that need to be checked upon.
        min_value: A `float` of  minimum value threshold.
        round_down_protect: A `bool` indicating whether round down more than 10%
        will be allowed.

    Returns:
        The adjusted value in `int` that is divisible against divisor.
    """
    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if round_down_protect and new_value < 0.9 * value:
        new_value += divisor
    return int(new_value)


def conv2d(in_channels, out_channels, kernel_size=3, stride=1, groups=1, bias=False, norm=True, act=True):
    # conv = nn.Sequential()
    # padding = (kernel_size - 1) // 2
    # conv.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias, groups=groups))
    # if norm:
    #     conv.append(nn.BatchNorm2d(out_channels))
    # if act:
    #     conv.append(nn.ReLU6())

    conv = []
    padding = (kernel_size - 1) // 2
    conv.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias, groups=groups))
    if norm:
        conv.append(nn.BatchNorm2d(out_channels))
    if act:
        conv.append(nn.ReLU6())

    return nn.Sequential(*conv)


class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio, act=False, squeeze_exactation=False):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        hidden_dim = int(round(in_channels * expand_ratio))
        self.block = nn.Sequential()
        if expand_ratio != 1:
            self.block.add_module("exp_1x1", conv2d(in_channels, hidden_dim, kernel_size=3, stride=stride))
        if squeeze_exactation:
            self.block.add_module("conv_3x3", conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, groups=hidden_dim))
        self.block.add_module("res_1x1", conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, act=act))
        self.use_res_connect = self.stride == 1 and in_channels == out_channels

    def forward(self, x):
        if self.use_res_connect:
            return x + self.block(x)
        else:
            return self.block(x)


class UniversalInvertedBottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, start_dw_kernel_size, middle_dw_kernel_size, middle_dw_downsample,
                 stride, expand_ratio):
        """An inverted bottleneck block with optional depthwises.
        Referenced from here https://github.com/tensorflow/models/blob/master/official/vision/modeling/layers/nn_blocks.py
        """
        super(UniversalInvertedBottleneckBlock, self).__init__()
        # starting depthwise conv
        self.start_dw_kernel_size = start_dw_kernel_size
        if self.start_dw_kernel_size:
            stride_ = stride if not middle_dw_downsample else 1
            self._start_dw_ = conv2d(in_channels, in_channels, kernel_size=start_dw_kernel_size, stride=stride_, groups=in_channels, act=False)
        # expansion with 1x1 convs
        expand_filters = make_divisible(in_channels * expand_ratio, 8)
        self._expand_conv = conv2d(in_channels, expand_filters, kernel_size=1)
        # middle depthwise conv
        self.middle_dw_kernel_size = middle_dw_kernel_size
        if self.middle_dw_kernel_size:
            stride_ = stride if middle_dw_downsample else 1
            self._middle_dw = conv2d(expand_filters, expand_filters, kernel_size=middle_dw_kernel_size, stride=stride_, groups=expand_filters)
        # projection with 1x1 convs
        self._proj_conv = conv2d(expand_filters, out_channels, kernel_size=1, stride=1, act=False)

        # expand depthwise conv (not used)
        # _end_dw_kernel_size = 0
        # self._end_dw = conv2d(out_channels, out_channels, kernel_size=_end_dw_kernel_size, stride=stride, groups=in_channels, act=False)

    def forward(self, x):
        if self.start_dw_kernel_size:
            x = self._start_dw_(x)
            # print("_start_dw_", x.shape)
        x = self._expand_conv(x)
        # print("_expand_conv", x.shape)
        if self.middle_dw_kernel_size:
            x = self._middle_dw(x)
            # print("_middle_dw", x.shape)
        x = self._proj_conv(x)
        # print("_proj_conv", x.shape)
        return x


class MultiQueryAttentionLayerWithDownSampling(nn.Module):
    def __init__(self, in_channels, num_heads, key_dim, value_dim, query_h_strides, query_w_strides, kv_strides, dw_kernel_size=3, dropout=0.0):
        """Multi Query Attention with spatial downsampling.
        Referenced from here https://github.com/tensorflow/models/blob/master/official/vision/modeling/layers/nn_blocks.py

        3 parameters are introduced for the spatial downsampling:
        1. kv_strides: downsampling factor on Key and Values only.
        2. query_h_strides: vertical strides on Query only.
        3. query_w_strides: horizontal strides on Query only.

        This is an optimized version.
        1. Projections in Attention is explict written out as 1x1 Conv2D.
        2. Additional reshapes are introduced to bring a up to 3x speed up.
        """
        super(MultiQueryAttentionLayerWithDownSampling, self).__init__()
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.query_h_strides = query_h_strides
        self.query_w_strides = query_w_strides
        self.kv_strides = kv_strides
        self.dw_kernel_size = dw_kernel_size
        self.dropout = dropout

        self.head_dim = self.key_dim // num_heads

        if self.query_h_strides > 1 or self.query_w_strides > 1:
            self._query_downsampling_norm = nn.BatchNorm2d(in_channels)
        self._query_proj = conv2d(in_channels, self.num_heads * self.key_dim, 1, 1, norm=False, act=False)

        if self.kv_strides > 1:
            self._key_dw_conv = conv2d(in_channels, in_channels, dw_kernel_size, kv_strides, groups=in_channels,
                                       norm=True, act=False)
            self._value_dw_conv = conv2d(in_channels, in_channels, dw_kernel_size, kv_strides, groups=in_channels,
                                         norm=True, act=False)
        self._key_proj = conv2d(in_channels, key_dim, 1, 1, norm=False, act=False)
        self._value_proj = conv2d(in_channels, key_dim, 1, 1, norm=False, act=False)
        self._output_proj = conv2d(num_heads * key_dim, in_channels, 1, 1, norm=False, act=False)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        bs, seq_len, _, _ = x.size()
        # print(x.size())
        if self.query_h_strides > 1 or self.query_w_strides > 1:
            q = F.avg_pool2d(self.query_h_strides, self.query_w_strides)
            q = self._query_downsampling_norm(q)
            q = self._query_proj(q)
        else:
            q = self._query_proj(x)
        px = q.size(2)
        q = q.view(bs, self.num_heads, -1, self.key_dim)  # [batch_size, num_heads, seq_len, key_dim]

        if self.kv_strides > 1:
            k = self._key_dw_conv(x)
            k = self._key_proj(k)
            v = self._value_dw_conv(x)
            v = self._value_proj(v)
        else:
            k = self._key_proj(x)
            v = self._value_proj(x)
        k = k.view(bs, 1, self.key_dim, -1)   # [batch_size, 1, key_dim, seq_length]
        v = v.view(bs, 1, -1, self.key_dim)    # [batch_size, 1, seq_length, key_dim]

        # calculate attention score
        # print(q.shape, k.shape, v.shape)
        attn_score = torch.matmul(q, k) / (self.head_dim ** 0.5)
        attn_score = self.dropout(attn_score)
        attn_score = F.softmax(attn_score, dim=-1)

        # context = torch.einsum('bnhm,bmv->bnhv', attn_score, v)
        # print(attn_score.shape, v.shape)
        context = torch.matmul(attn_score, v)
        context = context.view(bs, self.num_heads * self.key_dim, px, px)
        output = self._output_proj(context)
        # print(output.shape)
        return output


class MNV4layerScale(nn.Module):
    def __init__(self, init_value):
        """LayerScale as introduced in CaiT: https://arxiv.org/abs/2103.17239
        Referenced from here https://github.com/tensorflow/models/blob/master/official/vision/modeling/layers/nn_blocks.py

        As used in MobileNetV4.

        Attributes:
            init_value (float): value to initialize the diagonal matrix of LayerScale.
        """
        super(MNV4layerScale, self).__init__()
        self.init_value = init_value

    def forward(self, x):
        gamma = self.init_value * torch.ones(x.size(-1), dtype=x.dtype, device=x.device)
        return x * gamma


class MultiHeadSelfAttentionBlock(nn.Module):
    def __init__(self, in_channels, num_heads, key_dim, value_dim, query_h_strides, query_w_strides,
                 kv_strides, use_layer_scale, use_multi_query, use_residual=True):
        super(MultiHeadSelfAttentionBlock, self).__init__()
        self.query_h_strides = query_h_strides
        self.query_w_strides = query_w_strides
        self.kv_strides = kv_strides
        self.use_layer_scale = use_layer_scale
        self.use_multi_query = use_multi_query
        self.use_residual = use_residual
        self._input_norm = nn.BatchNorm2d(in_channels)

        if self.use_multi_query:
            self.multi_query_attention = MultiQueryAttentionLayerWithDownSampling(
                in_channels, num_heads, key_dim, value_dim, query_h_strides, query_w_strides, kv_strides
            )
        else:
            self.multi_head_attention = nn.MultiheadAttention(in_channels, num_heads, kdim=key_dim)

        if use_layer_scale:
            self.layer_scale_init_value = 1e-5
            self.layer_scale = MNV4layerScale(self.layer_scale_init_value)

    def forward(self, x):
        # Not using CPE, skipped
        # input norm
        shortcut = x
        x = self._input_norm(x)
        # multi query
        if self.use_multi_query:
            # print(x.size())
            x = self.multi_query_attention(x)
            # print(x.size())
        else:
            x = self.multi_head_attention(x, x)
        # layer scale
        if self.use_layer_scale:
            x = self.layer_scale(x)
        # use residual
        if self.use_residual:
            x = x + shortcut
        return x


def build_blocks(layer_spec):
    global msha
    if not layer_spec.get("block_name"):
        return nn.Sequential()
    block_names = layer_spec["block_name"]
    layers = nn.Sequential()
    if block_names == "convbn":
        schema_ = ["in_channels", "out_channels", "kernel_size", "stride"]
        for i in range(layer_spec["num_blocks"]):
            args = dict(zip(schema_, layer_spec["block_specs"][i]))
            layers.add_module(f"convbn_{i}", conv2d(**args))
    elif block_names == "uib":
        schema_ = ["in_channels", "out_channels", "start_dw_kernel_size", "middle_dw_kernel_size", "middle_dw_downsample",
                   "stride", "expand_ratio", "msha"]
        for i in range(layer_spec["num_blocks"]):
            args = dict(zip(schema_, layer_spec["block_specs"][i]))
            msha = args.pop("msha") if "msha" in args else 0
            layers.add_module(f"uib_{i}", UniversalInvertedBottleneckBlock(**args))
            if msha:
                msha_schema_ = [
                    "in_channels", "num_heads", "key_dim", "value_dim", "query_h_strides", "query_w_strides", "kv_strides",
                    "use_layer_scale", "use_multi_query", "use_residual"
                ]
                args = dict(zip(msha_schema_, [args["out_channels"]] + (msha)))
                layers.add_module(
                    f"msha_{i}", MultiHeadSelfAttentionBlock(**args)
                )
    elif block_names == "fused_ib":
        schema_ = ["in_channels", "out_channels", "stride", "expand_ratio", "act"]
        for i in range(layer_spec["num_blocks"]):
            args = dict(zip(schema_, layer_spec["block_specs"][i]))
            layers.add_module(f"fused_ib_{i}", InvertedResidual(**args))
    else:
        raise NotImplementedError
    return layers


class MobileNetV4Encoder(nn.Module, EncoderMixin):
    def __init__(self, model, pretrained=True, out_channels=None, depth=5):
        # MobileNetV4ConvSmall  MobileNetV4ConvMedium  MobileNetV4ConvLarge
        # MobileNetV4HybridMedium  MobileNetV4HybridLarge
        """Params to initiate MobilenNetV4
        Args:
            model : support 5 types of models as indicated in
            "https://github.com/tensorflow/models/blob/master/official/vision/modeling/backbones/mobilenet.py"
        """
        super(MobileNetV4Encoder, self).__init__()
        # print(MODEL_SPECS.keys(), model not in MODEL_SPECS.keys())
        assert model in MODEL_SPECS.keys()
        self.model = model
        self._depth = depth
        self._in_channels = 3
        self._out_channels = out_channels
        # self.num_classes = num_classes
        self.spec = MODEL_SPECS[self.model]

        self.conv00 = nn.Identity()
        # conv0
        self.conv0 = build_blocks(self.spec["conv0"])
        # layer1
        self.layer1 = build_blocks(self.spec["layer1"])
        # layer2
        self.layer2 = build_blocks(self.spec["layer2"])
        # layer3
        self.layer3 = build_blocks(self.spec["layer3"])
        # layer4
        self.layer4 = build_blocks(self.spec["layer4"])
        # layer5
        self.layer5 = build_blocks(self.spec["layer5"])
        # classify [optional]
        # self.fc = nn.Linear(1280, num_classes)

    def forward(self, x):
        x0 = self.conv00(x)
        x1 = self.conv0(x0)
        x2 = self.layer1(x1)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)
        x6 = self.layer5(x5)
        # x5 = F.adaptive_avg_pool2d(x5, 1)
        # out = self.fc(x5.flatten(1))

        return [x0, x1, x2, x3, x4, x6]



# mobilevit
# ！/usr/bin/python
# -*- coding: utf-8 -*-
# @Time: 2022/12/16 14:40
# @Author: 'IReverser'
# @ FileName: mobilevit.py
# MobileViT
"""
    original code from apple:
    https://github.com/apple/ml-cvnets/blob/main/cvnets/models/classification/mobilevit.py
"""
from typing import Optional, Tuple, Union, Dict, List
import math
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
# from mobilenet_transformer import TransformerEncoder
# from model_config import get_config


def make_divisible(v: Union[float, int],
                   divisor: Optional[int]=8,
                   min_value: Optional[Union[float, int]]=None) -> Union[float, int]:
    """ Optional[X] == Union[X, None]
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_value < 0.9 * v:
        new_value += divisor
    return new_value


class ConvLayer(nn.Module):
    """
    Applies a 2D convolution over an input
    Args:
        in_channels (int): :math:`C_{in}` from an expected input of size :math:`(N, C_{in}, H_{in}, W_{in})`
        out_channels (int): :math:`C_{out}` from an expected output of size :math:`(N, C_{out}, H_{out}, W_{out})`
        kernel_size (Union[int, Tuple[int, int]]): Kernel size for convolution.
        stride (Union[int, Tuple[int, int]]): Stride for convolution. Default: 1
        groups (Optional[int]): Number of groups in convolution. Default: 1
        bias (Optional[bool]): Use bias. Default: ``False``
        use_norm (Optional[bool]): Use normalization layer after convolution. Default: ``True``
        use_act (Optional[bool]): Use activation layer after convolution (or convolution and normalization). Default: ``True``
    Shape:
        - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, H_{out}, W_{out})`
    .. note::
        For depth-wise convolution, `groups=C_{in}=C_{out}`.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 stride: Optional[Union[int, Tuple[int, int]]] = 1,
                 groups: Optional[int] = 1,
                 bias: Optional[bool] = False,
                 use_norm: Optional[bool] = True,
                 use_act: Optional[bool] = True,
                 ) -> None:
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        assert isinstance(kernel_size, Tuple)

        if isinstance(stride, int):
            stride = (stride, stride)
        assert isinstance(stride, Tuple)

        padding = (int((kernel_size[0] - 1) / 2),
                   int((kernel_size[0] - 1) / 2))

        block = nn.Sequential()

        conv_layer = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            groups=groups,
            padding=padding,
            bias=bias
        )

        block.add_module(name="conv", module=conv_layer)

        if use_norm:
            norm_layer = nn.BatchNorm2d(num_features=out_channels, momentum=0.1)
            block.add_module(name="norm", module=norm_layer)
        if use_act:
            act_layer = nn.SiLU()
            block.add_module(name="act", module=act_layer)

        self.block = block

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class InvertedResidual(nn.Module):
    """
    This class implements the inverted residual block, as described in `MobileNetv2 <https://arxiv.org/abs/1801.04381>`_ paper
    https://github.com/apple/ml-cvnets/blob/6acab5e446357cc25842a90e0a109d5aeeda002f/cvnets/modules/mobilenetv2.py
    Args:
        in_channels (int): :math:`C_{in}` from an expected input of size :math:`(N, C_{in}, H_{in}, W_{in})`
        out_channels (int): :math:`C_{out}` from an expected output of size :math:`(N, C_{out}, H_{out}, W_{out)`
        stride (int): Use convolutions with a stride. Default: 1
        expand_ratio (Union[int, float]): Expand the input channels by this factor in depth-wise conv
        skip_connection (Optional[bool]): Use skip-connection. Default: True
    Shape:
        - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, H_{out}, W_{out})`
    .. note::
        If `in_channels =! out_channels` and `stride > 1`, we set `skip_connection=False`
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: int,
                 expand_ratio: Union[int, float],
                 skip_connection: Optional[bool] = True) -> None:
        assert stride in [1, 2]
        hidden_dim = make_divisible(int(round(in_channels * expand_ratio)), 8)

        super(InvertedResidual, self).__init__()

        block = nn.Sequential()
        if expand_ratio != 1:
            block.add_module(name="exp_1x1", module=ConvLayer(in_channels=in_channels, out_channels=hidden_dim, kernel_size=1),)

        block.add_module(name="conv_3x3", module=ConvLayer(in_channels=hidden_dim, out_channels=hidden_dim, stride=stride, kernel_size=3, groups=hidden_dim),)

        block.add_module(name="res_1x1", module=ConvLayer(in_channels=hidden_dim, out_channels=out_channels, kernel_size=1, use_act=False, use_norm=True),)

        self.block = block
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.exp = expand_ratio
        self.stride = stride
        self.use_res_connect = (self.stride == 1 and in_channels == out_channels and skip_connection)

    def forward(self, x: Tensor) -> Tensor:
        if self.use_res_connect:
            return x + self.block(x)
        else:
            return self.block(x)


class MobileViTBlock(nn.Module):
    """
    This class defines the `MobileViT block <https://arxiv.org/abs/2110.02178?context=cs.LG>`_
    Args:
        opts: command line arguments
        in_channels (int): :math:`C_{in}` from an expected input of size :math:`(N, C_{in}, H, W)`
        transformer_dim (int): Input dimension to the transformer unit
        ffn_dim (int): Dimension of the FFN block
        n_transformer_blocks (int): Number of transformer blocks. Default: 2
        head_dim (int): Head dimension in the multi-head attention. Default: 32
        attn_dropout (float): Dropout in multi-head attention. Default: 0.0
        dropout (float): Dropout rate. Default: 0.0
        ffn_dropout (float): Dropout between FFN layers in transformer. Default: 0.0
        patch_h (int): Patch height for unfolding operation. Default: 8
        patch_w (int): Patch width for unfolding operation. Default: 8
        transformer_norm_layer (Optional[str]): Normalization layer in the transformer block. Default: layer_norm
        conv_ksize (int): Kernel size to learn local representations in MobileViT block. Default: 3
        no_fusion (Optional[bool]): Do not combine the input and output feature maps. Default: False
    """
    def __init__(self,
                 in_channels: int,
                 transformer_dim: int,
                 ffn_dim: int,
                 n_transformer_blocks: int = 2,
                 head_dim: int = 32,
                 attn_dropout: float = 0.0,
                 dropout: float = 0.0,
                 ffn_dropout: float = 0.0,
                 patch_h: int = 8,
                 patch_w: int = 8,
                 conv_ksize: Optional[int] = 3,
                 *args,
                 **kwargs) -> None:
        super(MobileViTBlock, self).__init__()
        conv_3x3_in = ConvLayer(in_channels=in_channels,
                                out_channels=in_channels,
                                kernel_size=1,
                                stride=1,
                                use_norm=False,
                                use_act=False)

        conv_1x1_in = ConvLayer(in_channels=in_channels,
                                out_channels=transformer_dim,
                                kernel_size=1,
                                stride=1,
                                use_norm=False,
                                use_act=False)

        conv_1x1_out = ConvLayer(in_channels=transformer_dim,
                                 out_channels=in_channels,
                                 kernel_size=1,
                                 stride=1)

        conv_3x3_out = ConvLayer(in_channels=2*in_channels,
                                 out_channels=in_channels,
                                 kernel_size=conv_ksize,
                                 stride=1)

        self.local_rep = nn.Sequential()
        self.local_rep.add_module(name="conv_3x3", module=conv_3x3_in)
        self.local_rep.add_module(name="conv_1x1", module=conv_1x1_in)

        assert transformer_dim % head_dim == 0
        num_heads = transformer_dim // head_dim

        global_rep = [
            TransformerEncoder(embed_dim=transformer_dim,
                               ffn_latent_dim=ffn_dim,
                               num_heads=num_heads,
                               attn_dropout=attn_dropout,
                               dropout=dropout,
                               ffn_dropout=ffn_dropout)
            for _ in range(n_transformer_blocks)
        ]

        global_rep.append(nn.LayerNorm(transformer_dim))
        self.global_rep = nn.Sequential(*global_rep)

        self.conv_proj = conv_1x1_out
        self.fusion = conv_3x3_out

        self.patch_h = patch_h
        self.patch_w = patch_w
        self.patch_area = self.patch_h * self.patch_w

        self.cnn_in_dim = in_channels
        self.cnn_out_dim = transformer_dim
        self.n_heads = num_heads
        self.ffn_dim = ffn_dim
        self.attn_dropout = attn_dropout
        self.ffn_dropout = ffn_dropout
        self.n_blocks = n_transformer_blocks
        self.conv_ksize = conv_ksize

    def unfolding(self, x: Tensor) -> Tuple[Tensor, Dict]:
        patch_h, patch_w = self.patch_h, self.patch_w
        patch_area = patch_h * patch_w
        batch_size, in_channels, orig_h, orig_w = x.shape

        new_h = int(math.ceil(orig_h / self.patch_h) * self.patch_h)
        new_w = int(math.ceil(orig_w / self.patch_w) * patch_w)

        interpolate = False
        if new_h != orig_h or new_w != orig_w:
            # Note: Padding can be done, but then it needs to be handled in attention function.
            x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)
            interpolate = True

        # number of patches along height and width
        num_patch_h = new_h // patch_h  # n_h
        num_patch_w = new_w // patch_w  # n_w
        num_patches = num_patch_h * num_patch_w  # N

        # [B, C, H, W] -> [B * C * n_h, p_h, n_w, p_w]
        x = x.reshape(batch_size * in_channels * num_patch_h, patch_h, num_patch_w, patch_w)
        # [B * C * n_h, p_h, n_w, p_w] -> [B * C * n_h, n_w, p_h, p_w]
        x = x.transpose(1, 2)
        # [B * C * n_h, n_w, p_h, p_w] -> [B, C, N, P] where N = n_h * n_w and P = p_h * p_w
        x = x.reshape(batch_size, in_channels, num_patches, patch_area)
        # [B, C, N, P] -> [B, P, N, C]
        x = x.transpose(1, 3)
        # [B, P, N, C] -> [BP, N, C]
        x = x.reshape(batch_size * patch_area, num_patches, -1)

        info_dict = {
            "orig_size": (orig_h, orig_w),
            "batch_size": batch_size,
            "interpolate": interpolate,
            "total_patches": num_patches,
            "num_patches_h": num_patch_h,
            "num_patches_w": num_patch_w,
        }
        return x, info_dict

    def folding(self, x: Tensor, info_dict: Dict) -> Tensor:
        n_dim = x.dim()
        assert n_dim == 3, "Tensor should be of shape BPxNxC. Got: {}".format(x.shape)

        # [BP, N, C] -> [B, P, N, C]
        x = x.contiguous().view(
            info_dict["batch_size"], self.patch_area, info_dict["total_patches"], -1)

        batch_size, pixels, num_patches, channels = x.size()
        num_patch_h = info_dict["num_patches_h"]
        num_patch_w = info_dict["num_patches_w"]

        x = x.transpose(1, 3)
        # [B, P, N, C] -> [B*C*n_h, n_w, p_h, p_w]
        x = x.reshape(batch_size * channels * num_patch_h, num_patch_w, self.patch_h, self.patch_w)
        # [B*C*n_h, n_w, p_h, p_w] -> [B*C*n_h, p_h, n_w, p_w]
        x = x.transpose(1, 2)
        # [B*C*n_h, p_h, n_w, p_w] -> [B, C, H, W]
        x = x.reshape(batch_size, channels, num_patch_h * self.patch_h, num_patch_w * self.patch_w)

        if info_dict["interpolate"]:
            x = F.interpolate(x, size=info_dict["orig_size"], mode="bilinear", align_corners=False)
        return x

    def forward(self, x: Tensor) -> Tensor:
        res = x

        fm = self.local_rep(x)

        # convert feature map to patches
        patches, info_dict = self.unfolding(fm)

        # learn global representation
        for transformer_layer in self.global_rep:
            patches = transformer_layer(patches)

        # [B * Patch * Patches * C] -> [B * C * Pathes * Patch]
        fm = self.folding(x=patches, info_dict=info_dict)

        fm = self.conv_proj(fm)

        fm = self.fusion(torch.cat((res, fm), dim=1))

        return fm


class MobileViT(nn.Module, EncoderMixin):
    """
    This class implements the `MobileViT architecture <https://arxiv.org/abs/2110.02178?context=cs.LG>`_
    """
    def __init__(self, model_cfg: Dict, out_channels=None, depth=5):
        super(MobileViT, self).__init__()

        image_channels = 3
        self._out_channels = out_channels
        self._depth = depth

        out_channels = 16
        self.conv_1 = ConvLayer(in_channels=image_channels, out_channels=out_channels, kernel_size=3, stride=2)

        self.layer_1, out_channels = self._make_layer(input_channel=out_channels, cfg=model_cfg["layer1"])
        self.layer_2, out_channels = self._make_layer(input_channel=out_channels, cfg=model_cfg["layer2"])
        self.layer_3, out_channels = self._make_layer(input_channel=out_channels, cfg=model_cfg["layer3"])
        self.layer_4, out_channels = self._make_layer(input_channel=out_channels, cfg=model_cfg["layer4"])
        self.layer_5, out_channels = self._make_layer(input_channel=out_channels, cfg=model_cfg["layer5"])

        exp_channels = min(model_cfg["last_layer_exp_factor"] * out_channels, 960)
        self.conv_1x1_exp = ConvLayer(in_channels=out_channels,
                                      out_channels=exp_channels,
                                      kernel_size=1)

        self.classifier = nn.Sequential()
        self.classifier.add_module(name="global_pool", module=nn.AdaptiveAvgPool2d(1))
        self.classifier.add_module(name="flatten", module=nn.Flatten())
        if 0.0 < model_cfg["cls_dropout"] < 1.0:
            self.classifier.add_module(name="dropout", module=nn.Dropout(p=model_cfg["cls_dropout"]))
        # self.classifier.add_module(name="fc", module=nn.Linear(in_features=exp_channels,
        #                                                        out_features=num_classes))

        # weight initialize
        self.apply(self.init_parameters)

    def _make_layer(self, input_channel, cfg: Dict) -> Tuple[nn.Sequential, int]:
        block_type = cfg.get("block_type", "mobilevit")
        if block_type.lower() == "mobilevit":
            return self._make_vit_layer(input_channel=input_channel, cfg=cfg)
        else:
            return self._make_mobilenet_layer(input_channel=input_channel, cfg=cfg)

    @staticmethod
    def _make_mobilenet_layer(input_channel: int, cfg: Dict) -> Tuple[nn.Sequential, int]:
        out_channels = cfg.get("out_channels")
        num_blocks = cfg.get("num_blocks", 2)
        expand_ratio = cfg.get("expand_ratio", 4)
        block = []

        for i in range(num_blocks):
            stride = cfg.get("stride", 1) if i == 0 else 1

            layer = InvertedResidual(in_channels=input_channel,
                                     out_channels=out_channels,
                                     stride=stride,
                                     expand_ratio=expand_ratio)

            block.append(layer)
            input_channel = out_channels

        return nn.Sequential(*block), input_channel

    @staticmethod
    def _make_vit_layer(input_channel: int, cfg: Dict) -> [nn.Sequential, int]:
        stride = cfg.get("stride", 1)
        block = []

        if stride == 2:
            layer = InvertedResidual(in_channels=input_channel,
                                     out_channels=cfg.get("out_channels"),
                                     stride=stride,
                                     expand_ratio=cfg.get("mv_expand_ratio", 4))

            block.append(layer)
            input_channel = cfg.get("out_channels")

        transformer_dim = cfg["transformer_channels"]
        ffn_dim = cfg.get("ffn_dim")
        num_heads = cfg.get("num_heads", 4)
        head_dim = transformer_dim // num_heads

        if transformer_dim % head_dim != 0:
            raise ValueError("Transformer input dimension should be divisible by head dimension. "
                             "Got {} and {}.".format(transformer_dim, head_dim))

        block.append(MobileViTBlock(in_channels=input_channel,
                                    transformer_dim=transformer_dim,
                                    ffn_dim=ffn_dim,
                                    n_transformer_blocks=cfg.get("transformer_blocks", 1),
                                    patch_h=cfg.get("patch_h", 2),
                                    patch_w=cfg.get("patch_w", 2),
                                    ffn_dropout=cfg.get("ffn_dropout", 0.0),
                                    attn_dropout=cfg.get("attn_dropout", 0.1),
                                    head_dim=head_dim,
                                    conv_ksize=3))
        return nn.Sequential(*block), input_channel

    @staticmethod
    def init_parameters(m):
        if isinstance(m, nn.Conv2d):
            if m.weight is not None:
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            if m.weight is not None:
                nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.Linear, )):
            if m.weight is not None:
                nn.init.trunc_normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        else:
            pass

    def forward(self, x: Tensor):
        x_ = x
        x0 = self.conv_1(x)
        x1 = self.layer_1(x0)
        x2 = self.layer_2(x1)
        x3 = self.layer_3(x2)
        x4 = self.layer_4(x3)
        x5 = self.layer_5(x4)
        x6 = self.conv_1x1_exp(x5)
        # feas = x
        # x = self.classifier(x)
        # if is_feas:
        #     return feas, x
        # else:
        #     return x
        return [x_, x1, x2, x3, x4, x6]



# def mobilevit_xxs(num_classes: int = 1000):
#     # pretrain weight link
#     # https://docs-assets.developer.apple.com/ml-research/models/cvnets/classification/mobilevit_xxs.pt
#     config = get_config("xxs")
#     m = MobileViT(config, num_classes=num_classes)
#     return m
#
#
# def mobilevit_xs(num_classes: int = 1000):
#     # pretrain weight link
#     # https://docs-assets.developer.apple.com/ml-research/models/cvnets/classification/mobilevit_xs.pt
#     config = get_config("xs")
#     m = MobileViT(config, num_classes=num_classes)
#     return m
#
#
# def mobilevit_s(num_classes: int = 1000):
#     # pretrain weight link
#     # https://docs-assets.developer.apple.com/ml-research/models/cvnets/classification/mobilevit_s.pt
#     config = get_config("s")
#     m = MobileViT(config, num_classes=num_classes)
#     return m

#
# model_dict = {
#     "mobilevit_xxs": mobilevit_xxs,
#     "mobilevit_xs": mobilevit_xs,
#     "mobilevit_s": mobilevit_s
# }


# def create_mobilevit(model_name="mobilevit_xxs", num_classes=1000):
#     model = model_dict[model_name](num_classes=num_classes)
#     return model

mobilenet_encoders = {
    "mobilenet_v2": {
        "encoder": MobileNetV2Encoder,
        "params": {
            "out_channels": (3, 16, 24, 32, 96, 1280),
        },
    },
    "mobilenet_v3_large": {
        "encoder": MobileNetV3Encoder,
        "params": {
            "out_channels": (3, 16, 24, 40, 112, 960),
            "version": "large",
        },
    },
    "mobilenet_v3_small": {
        "encoder": MobileNetV3Encoder,
        "params": {
            "out_channels": (3, 16, 16, 24, 48, 576),
            "version": "small",
        },
    },
    "mobilenet_v4_conv_small": {
        "encoder": MobileNetV4Encoder,
        "params": {
            "model": "MNV4ConvSmall",
            "out_channels": (3, 32, 32, 64, 96, 1280),
        },
    },
    "mobilenet_v4_conv_medium": {
        "encoder": MobileNetV4Encoder,
        "params": {
            "model": "MNV4ConvMedium",
            "out_channels": (3, 32, 48, 80, 160, 1280),
        },
    },
    "mobilenet_v4_conv_large": {
        "encoder": MobileNetV4Encoder,
        "params": {
            "model": "MNV4ConvLarge",
            "out_channels": (3, 24, 48, 96, 192, 1280),
        },
    },
    "mobilenet_v4_hybrid_medium": {
        "encoder": MobileNetV4Encoder,
        "params": {
            "model": "MNV4HybridMedium",
            "out_channels": (3, 32, 48, 80, 160, 1280),
        },
    },
    "mobilenet_v4_hybrid_large": {
        "encoder": MobileNetV4Encoder,
        "params": {
            "model": "MNV4HybridLarge",
            "out_channels": (3, 24, 48, 96, 192, 1280),
        },
    },
    "mobilevit_xxs":{
        "encoder": MobileViT,
        "params": {
            "model_cfg": get_config("xxs"),
            "out_channels": (3, 16, 24, 48, 64, 320),
        },
    },
    "mobilevit_xs":{
        "encoder": MobileViT,
        "params": {
            "model_cfg": get_config("xs"),
            "out_channels": (3, 32, 48, 64, 80, 384),
        },
    },
    "mobilevit_s":{
        "encoder": MobileViT,
        "params": {
            "model_cfg": get_config("s"),
            "out_channels": (3, 32, 64, 96, 128, 640),
        },
    },
}
