"""
Minimal BiSeNet face parsing model for inference.
Based on https://github.com/yakhyo/face-parsing (MIT License).
"""

from typing import List, Optional, Type, Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F


def conv3x3(in_channels: int, out_channels: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = norm_layer(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[BasicBlock],
        layers: List[int],
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.in_channels = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                f"replace_stride_with_dilation should be None or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(
        self,
        block: Type[BasicBlock],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.in_channels != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.in_channels, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.in_channels,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        self.in_channels = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.in_channels,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        feat8 = self.layer2(x)  # 1/8
        feat16 = self.layer3(feat8)  # 1/16
        feat32 = self.layer4(feat16)  # 1/32

        return feat8, feat16, feat32


def resnet18() -> ResNet:
    return ResNet(BasicBlock, [2, 2, 2, 2])


def resnet34() -> ResNet:
    return ResNet(BasicBlock, [3, 4, 6, 3])


class ConvBNReLU(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        groups: int = 1,
        dilation: int = 1,
        inplace: bool = True,
        bias: bool = False,
    ) -> None:
        super().__init__()

        if padding is None:
            padding = kernel_size // 2

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.norm = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        return x


class BiSeNetOutput(nn.Module):
    def __init__(self, in_channels: int, mid_channels: int, num_classes: int) -> None:
        super().__init__()
        self.conv_block = ConvBNReLU(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=3,
            stride=1,
        )
        self.conv = nn.Conv2d(
            in_channels=mid_channels,
            out_channels=num_classes,
            kernel_size=1,
            bias=False,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv_block(x)
        x = self.conv(x)
        return x


class AttentionRefinementModule(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv_block = ConvBNReLU(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1)
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        feat = self.conv_block(x)
        feat_shape = [int(t) for t in feat.size()[2:]]
        pool = F.avg_pool2d(feat, feat_shape)
        attention = self.attention(pool)
        out = torch.mul(feat, attention)
        return out


class ContextPath(nn.Module):
    def __init__(self, backbone_name: str = "resnet18") -> None:
        super().__init__()
        if backbone_name == "resnet18":
            self.backbone = resnet18()
        elif backbone_name == "resnet34":
            self.backbone = resnet34()
        else:
            raise ValueError("Available backbone modules: resnet18, resnet34")

        self.arm16 = AttentionRefinementModule(in_channels=256, out_channels=128)
        self.arm32 = AttentionRefinementModule(in_channels=512, out_channels=128)
        self.conv_head32 = ConvBNReLU(in_channels=128, out_channels=128, kernel_size=3, stride=1)
        self.conv_head16 = ConvBNReLU(in_channels=128, out_channels=128, kernel_size=3, stride=1)
        self.conv_avg = ConvBNReLU(in_channels=512, out_channels=128, kernel_size=1, stride=1)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        feat8, feat16, feat32 = self.backbone(x)

        h8, w8 = feat8.size()[2:]
        h16, w16 = feat16.size()[2:]
        h32, w32 = feat32.size()[2:]

        feat32_shape = [int(t) for t in feat32.size()[2:]]
        avg = F.avg_pool2d(feat32, feat32_shape)
        avg = self.conv_avg(avg)
        avg_up = F.interpolate(avg, (h32, w32), mode="nearest")

        feat32_arm = self.arm32(feat32)
        feat32_sum = feat32_arm + avg_up
        feat32_up = F.interpolate(feat32_sum, (h16, w16), mode="nearest")
        feat32_up = self.conv_head32(feat32_up)

        feat16_arm = self.arm16(feat16)
        feat16_sum = feat16_arm + feat32_up
        feat16_up = F.interpolate(feat16_sum, (h8, w8), mode="nearest")
        feat16_up = self.conv_head16(feat16_up)

        return feat8, feat16_up, feat32_up


class FeatureFusionModule(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        self.conv_block = ConvBNReLU(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1)
        self.conv1 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels // 4,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.conv2 = nn.Conv2d(
            in_channels=out_channels // 4,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, fsp: Tensor, fcp: Tensor) -> Tensor:
        fcat = torch.cat([fsp, fcp], dim=1)
        feat = self.conv_block(fcat)

        feat_shape = [int(t) for t in feat.size()[2:]]
        attention = F.avg_pool2d(feat, feat_shape)
        attention = self.conv1(attention)
        attention = self.relu(attention)
        attention = self.conv2(attention)
        attention = self.sigmoid(attention)
        feat_attention = torch.mul(feat, attention)
        feat_out = feat_attention + feat
        return feat_out


class BiSeNet(nn.Module):
    def __init__(self, num_classes: int, backbone_name: str = "resnet18") -> None:
        super().__init__()
        self.fpn = ContextPath(backbone_name=backbone_name)
        self.ffm = FeatureFusionModule(in_channels=256, out_channels=256)

        self.conv_out = BiSeNetOutput(in_channels=256, mid_channels=256, num_classes=num_classes)
        self.conv_out16 = BiSeNetOutput(in_channels=128, mid_channels=64, num_classes=num_classes)
        self.conv_out32 = BiSeNetOutput(in_channels=128, mid_channels=64, num_classes=num_classes)

    def forward(self, x: Tensor):
        h, w = x.size()[2:]
        feat_res8, feat_cp8, feat_cp16 = self.fpn(x)
        feat_fuse = self.ffm(feat_res8, feat_cp8)

        feat_out = self.conv_out(feat_fuse)
        feat_out16 = self.conv_out16(feat_cp8)
        feat_out32 = self.conv_out32(feat_cp16)

        feat_out = F.interpolate(feat_out, (h, w), mode="bilinear", align_corners=True)
        feat_out16 = F.interpolate(feat_out16, (h, w), mode="bilinear", align_corners=True)
        feat_out32 = F.interpolate(feat_out32, (h, w), mode="bilinear", align_corners=True)

        return feat_out, feat_out16, feat_out32
