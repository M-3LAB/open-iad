# adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Type, Any, Callable, Union, List, Optional, Tuple


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3transpose(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1,  dilation: int = 1) -> nn.ConvTranspose2d:
    """3x3 transpose convolution with padding"""
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation,
                     output_padding=dilation, groups=groups, bias=False, dilation=dilation)
    

def conv1x1transpose(in_planes: int, out_planes: int, stride: int = 1) -> nn.ConvTranspose2d:
    """1x1 transpose convolution"""
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False, output_padding=1)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        dropout_rate: float = 0,
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        if self.dropout is not None:
            out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class PreActBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        dropout_rate: float = 0,
    ) -> None:
        super(PreActBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('PreActBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in PreActBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.bn1 = norm_layer(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn2 = norm_layer(planes)
        self.conv2 = conv3x3(planes, planes)
        self.downsample = downsample
        self.stride = stride
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None

    def forward(self, x: Tensor) -> Tensor:
        out = self.bn1(x)
        out = self.relu(out)

        identity = out
        if self.downsample is not None:
            identity = self.downsample(identity)

        out = self.conv1(out)

        if self.dropout is not None:
            out = self.dropout(out)
        
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out += identity

        return out


class TransposeBasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        upsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        dropout_rate: float = 0,
    ) -> None:
        super(TransposeBasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3transpose(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.upsample = upsample
        self.stride = stride
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None


    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        if self.dropout is not None:
            out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.upsample is not None:
            identity = self.upsample(x)

        out += identity
        out = self.relu(out)

        return out


class TransposePreActBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        upsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        dropout_rate: float = 0,
    ) -> None:
        super(TransposePreActBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('TransposePreActBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicTransposePreActBlockBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.bn1 = norm_layer(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv3x3transpose(inplanes, planes, stride)
        self.bn2 = norm_layer(planes)
        self.conv2 = conv3x3(planes, planes)
        self.upsample = upsample
        self.stride = stride
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None


    def forward(self, x: Tensor) -> Tensor:
        out = self.bn1(x)
        out = self.relu(out)

        identity = out
        if self.upsample is not None:
            identity = self.upsample(identity)

        out = self.conv1(out)

        if self.dropout is not None:
            out = self.dropout(out)
        
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out += identity

        return out


class ResNetEncDec(nn.Module):

    def __init__(
        self,
        layers: List[int],
        uplayers: List[int],
        preact: bool = False,
        pool: bool = True,
        in_channels: int = 3,
        num_classes: int = 1,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        final_activation: str = 'identity',
        dropout_rate: float = 0
    ) -> None:
        super(ResNetEncDec, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.pool = pool
        self.preact = preact
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.dropout_rate = dropout_rate
        self.conv1 = nn.Conv2d(in_channels, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        if not preact:
            self.bn1 = norm_layer(self.inplanes)
            self.relu = nn.ReLU(inplace=True)
        if pool: 
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        block = PreActBlock if preact else BasicBlock
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        
        if preact:
            self.nin = nn.Sequential(
                nn.ReLU(inplace=True),
                conv1x1(512, 256),
                nn.ReLU(inplace=True),
                conv1x1(256, 128),
            )
        else:
            self.nin = nn.Sequential(
                conv1x1(512, 256),
                nn.ReLU(inplace=True),
                conv1x1(256, 128),
                nn.ReLU(inplace=True),
            )
        self.inplanes = 128
        block = TransposePreActBlock if preact else TransposeBasicBlock
        self.uplayer1 = self._make_layer(block, 64, uplayers[0], stride=2,
                                       dilate=replace_stride_with_dilation[2], transpose=True)
        self.uplayer2 = self._make_layer(block, 32, uplayers[1], stride=2,
                                       dilate=replace_stride_with_dilation[1], transpose=True)
        self.uplayer3 = self._make_layer(block, 16, uplayers[2], stride=2,
                                       dilate=replace_stride_with_dilation[0], transpose=True)
        if preact:
            self.bn1 = norm_layer(16)
            self.relu = nn.ReLU(inplace=True)
        if pool:
            self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.convtranspose1 = nn.ConvTranspose2d(16, num_classes, kernel_size=7, stride=2, padding=3,
                               bias=False, output_padding=1)
        if final_activation =='sigmoid':
            self.final_activation = nn.Sigmoid()
        elif final_activation =='softmax':
            self.final_activation = nn.Softmax(dim=1)
        elif final_activation =='relu':
            self.final_activation = nn.ReLU(inplace=True)
        else:
            self.final_activation = nn.Identity()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]
                if isinstance(m, TransposeBasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block: Type[Union[BasicBlock, TransposeBasicBlock]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False, transpose: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        resample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            if transpose:
                resample = nn.Sequential(
                    conv1x1transpose(self.inplanes, planes * block.expansion, stride),
                    norm_layer(planes * block.expansion),
                )
            else:
                resample = nn.Sequential(
                  conv1x1(self.inplanes, planes * block.expansion, stride),
                  norm_layer(planes * block.expansion),
                )

        layers = []
        layers.append(block(self.inplanes, planes, stride, resample, self.groups,
                            self.base_width, previous_dilation, norm_layer, dropout_rate=self.dropout_rate))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, dropout_rate=self.dropout_rate))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        if not self.preact:
            x = self.bn1(x)
            x = self.relu(x)
        if self.pool:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.nin(x)

        x = self.uplayer1(x)
        x = self.uplayer2(x)
        x = self.uplayer3(x)

        if self.preact:
            x = self.bn1(x)
            x = self.relu(x)
        if self.pool:
            x = self.upsample(x)
        x = self.convtranspose1(x)

        x = self.final_activation(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x) 


def resnet18_enc_dec(**kwargs: Any) -> ResNetEncDec:
    return ResNetEncDec([2, 2, 2, 2], [1, 1, 1], **kwargs)
