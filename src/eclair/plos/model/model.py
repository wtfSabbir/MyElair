"""Modele Minkowski."""

import math
from typing import Any

import MinkowskiEngine as ME  # noqa: N817
import torch
from torch import nn


class SelfAttention(nn.Module):
    """
    Self-Attention layer for 3D sparse tensors using MinkowskiEngine.

    :param in_dim: Dimension of the input features.
    """

    def __init__(self, in_dim: int) -> None:
        super().__init__()

        self.linear_query = ME.MinkowskiLinear(in_dim, in_dim // 8)
        self.linear_key = ME.MinkowskiLinear(in_dim, in_dim // 8)
        self.linear_value = ME.MinkowskiLinear(in_dim, in_dim)
        self.pooling = ME.MinkowskiGlobalAvgPooling()
        self.normalized = nn.Softmax(dim=-1)

    def forward(self, x: ME.SparseTensor) -> ME.SparseTensor:
        """
        Forward pass of the SelfAttention layer.

        :param x: Input sparse tensor.
        :return: Output sparse tensor after applying self-attention.
        """
        identity = x
        q = self.linear_query(x)
        k = self.linear_key(x)
        v = self.linear_value(x)
        k_feat_t = k.features.T

        attmap = torch.matmul(q.F, k_feat_t) / (math.sqrt(q.F.size(1)))
        out = self.normalized(attmap)
        out = torch.matmul(out, v.F)
        out = ME.SparseTensor(
            features=out,
            tensor_stride=x.tensor_stride,
            device=x.device,
            coordinate_map_key=x.coordinate_map_key,
            coordinate_manager=x.coordinate_manager,
        )
        return out + identity


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention layer for 3D sparse tensors using MinkowskiEngine.

    :param in_dim: Dimension of the input features.
    :param num_heads: Number of attention heads.
    """

    def __init__(self, in_dim: int, num_heads: int) -> None:
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = in_dim // num_heads

        self.linear_query = nn.ModuleList([ME.MinkowskiLinear(in_dim, self.head_dim) for _ in range(num_heads)])
        self.linear_key = nn.ModuleList([ME.MinkowskiLinear(in_dim, self.head_dim) for _ in range(num_heads)])
        self.linear_value = nn.ModuleList([ME.MinkowskiLinear(in_dim, self.head_dim) for _ in range(num_heads)])

        self.normalized = nn.Softmax(dim=-1)
        self.linear_out = ME.MinkowskiLinear(in_dim, in_dim)

    def forward(self, x: ME.SparseTensor) -> ME.SparseTensor:
        """
        Forward pass of the MultiHeadAttention layer.

        :param x: Input sparse tensor.
        :return: Output sparse tensor after applying multi-head attention.
        """
        identity = x
        all_heads: list[torch.Tensor] = []

        for i in range(self.num_heads):
            q = self.linear_query[i](x)  # Query for head i
            k = self.linear_key[i](x)  # Key for head i
            v = self.linear_value[i](x)  # Value for head i
            k_feat_t = k.features.T
            attmap = torch.matmul(q.F, k_feat_t) / (math.sqrt(q.F.size(1)))
            out = self.normalized(attmap)
            out = torch.matmul(out, v.F)
            all_heads.append(out)

        all_heads_concat = torch.cat(all_heads, dim=-1)
        out = ME.SparseTensor(
            features=all_heads_concat,
            tensor_stride=x.tensor_stride,
            device=x.device,
            coordinate_map_key=x.coordinate_map_key,
            coordinate_manager=x.coordinate_manager,
        )
        out = self.linear_out(out)

        return out + identity


class BasicBlock(nn.Module):
    """
    Basic residual block for ResNet with MinkowskiEngine convolutions.

    :param inplanes: Number of input planes.
    :param planes: Number of output planes.
    :param stride: Stride of the convolution. Default is 1.
    :param dilation: Dilation rate of the convolution. Default is 1.
    :param downsample: Downsample layer. Default is None.
    :param bn_momentum: Batch normalization momentum. Default is 0.1.
    :param dimension: Dimension of the convolution (e.g., 3 for 3D). Must be greater than 0.
    """

    expansion = 1

    def __init__(  # noqa: PLR0913, PLR0917
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        dilation: int = 1,
        downsample: nn.Module | None = None,
        bn_momentum: float = 0.1,
        dimension: int = -1,
    ) -> None:
        super().__init__()

        self.conv1 = ME.MinkowskiConvolution(
            inplanes,
            planes,
            kernel_size=3,
            stride=stride,
            dilation=dilation,
            dimension=dimension,
        )
        self.norm1 = ME.MinkowskiBatchNorm(planes, momentum=bn_momentum)
        self.conv2 = ME.MinkowskiConvolution(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            dilation=dilation,
            dimension=dimension,
        )
        self.norm2 = ME.MinkowskiBatchNorm(planes, momentum=bn_momentum)
        self.relu = ME.MinkowskiReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: ME.SparseTensor) -> ME.SparseTensor:
        """
        Forward pass of the BasicBlock.

        :param x: Input sparse tensor.
        :return: Output sparse tensor after passing through the BasicBlock.
        """
        residual = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return self.relu(out)


class Bottleneck(nn.Module):
    """
    Bottleneck residual block for ResNet with MinkowskiEngine convolutions.

    :param inplanes: Number of input planes.
    :param planes: Number of output planes.
    :param stride: Stride of the convolution. Default is 1.
    :param dilation: Dilation rate of the convolution. Default is 1.
    :param downsample: Downsample layer. Default is None.
    :param bn_momentum: Batch normalization momentum. Default is 0.1.
    :param dimension: Dimension of the convolution (e.g., 3 for 3D). Must be greater than 0.
    """

    expansion = 4

    def __init__(  # noqa: PLR0913, PLR0917
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        dilation: int = 1,
        downsample: nn.Module | None = None,
        bn_momentum: float = 0.1,
        dimension: int = -1,
    ) -> None:
        super().__init__()

        self.conv1 = ME.MinkowskiConvolution(inplanes, planes, kernel_size=1, dimension=dimension)
        self.norm1 = ME.MinkowskiBatchNorm(planes, momentum=bn_momentum)

        self.conv2 = ME.MinkowskiConvolution(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            dilation=dilation,
            dimension=dimension,
        )
        self.norm2 = ME.MinkowskiBatchNorm(planes, momentum=bn_momentum)

        self.conv3 = ME.MinkowskiConvolution(planes, planes * self.expansion, kernel_size=1, dimension=dimension)
        self.norm3 = ME.MinkowskiBatchNorm(planes * self.expansion, momentum=bn_momentum)

        self.relu = ME.MinkowskiReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: ME.SparseTensor) -> ME.SparseTensor:
        """
        Forward pass of the Bottleneck block.

        :param x: Input sparse tensor.
        :return: Output sparse tensor after passing through the Bottleneck block.
        """
        residual = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = out.float()  # remet en torch.float32

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.norm3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return self.relu(out)


class ResNetBase(nn.Module):
    """
    Base class for ResNet with MinkowskiEngine convolutions.

    :param in_channels: Number of input channels.
    :param out_channels: Number of output channels.
    :param d: Dimension of the convolutions (e.g., 3 for 3D). Default is 3.
    """

    BLOCK: type[BasicBlock | Bottleneck] | None = None
    LAYERS: tuple[int, ...] = (2, 2, 2, 2)  # Default for ResNetBase
    INIT_DIM = 64
    PLANES: tuple[int, ...] = (64, 128, 256, 512)

    def __init__(self, in_channels: int, out_channels: int, d: int = 3) -> None:
        nn.Module.__init__(self)
        self.D = d

        self.network_initialization(in_channels, out_channels, d)
        self.weight_initialization()

    def network_initialization(self, in_channels: int, out_channels: int, d: int) -> None:
        """
        Initialize the network layers.

        :param in_channels: Number of input channels.
        :param out_channels: Number of output channels.
        :param d: Dimension of the Minkowski space.
        """
        self.inplanes = self.INIT_DIM
        self.conv1 = nn.Sequential(
            ME.MinkowskiConvolution(in_channels, self.inplanes, kernel_size=3, stride=2, dimension=d),
            ME.MinkowskiInstanceNorm(self.inplanes),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiMaxPooling(kernel_size=2, stride=2, dimension=d),
        )

        self.layer1 = self._make_layer(self.BLOCK, self.PLANES[0], self.LAYERS[0], stride=2)  # type: ignore[arg-type]
        self.layer2 = self._make_layer(self.BLOCK, self.PLANES[1], self.LAYERS[1], stride=2)  # type: ignore[arg-type]
        self.layer3 = self._make_layer(self.BLOCK, self.PLANES[2], self.LAYERS[2], stride=2)  # type: ignore[arg-type]
        self.layer4 = self._make_layer(self.BLOCK, self.PLANES[3], self.LAYERS[3], stride=2)  # type: ignore[arg-type]

        self.conv5 = nn.Sequential(
            ME.MinkowskiDropout(),
            ME.MinkowskiConvolution(self.inplanes, self.inplanes, kernel_size=3, stride=3, dimension=d),
            ME.MinkowskiInstanceNorm(self.inplanes),
            ME.MinkowskiGELU(),
        )

        self.glob_pool = ME.MinkowskiGlobalMaxPooling()

        self.final = ME.MinkowskiLinear(self.inplanes, out_channels, bias=True)

    def weight_initialization(self) -> None:
        """Initialize weights for the convolution layers using Kaiming normalization."""
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(m.kernel, mode="fan_out", nonlinearity="relu")

            if isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)

    def _make_layer(  # noqa: PLR0913, PLR0917
        self,
        block: type[nn.Module],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilation: int = 1,
        bn_momentum: float = 0.1,
    ) -> nn.Sequential:
        """
        Create a layer in the ResNet model.

        :param block: Block type to be used in the layer.
        :param planes: Number of output planes.
        :param blocks: Number of blocks.
        :param stride: Stride of the convolution. Default is 1.
        :param dilation: Dilation rate of the convolution. Default is 1.
        :param bn_momentum: Batch normalization momentum. Default is 0.1.
        :return: A sequential container with the blocks for the layer.
        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:  # type: ignore[attr-defined]
            downsample = nn.Sequential(
                ME.MinkowskiConvolution(
                    self.inplanes,
                    planes * block.expansion,  # type: ignore[attr-defined]
                    kernel_size=1,
                    stride=stride,
                    dimension=self.D,
                ),
                ME.MinkowskiBatchNorm(planes * block.expansion, momentum=bn_momentum),  # type: ignore[attr-defined]
            )
        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride=stride,
                dilation=dilation,
                downsample=downsample,
                dimension=self.D,
            )
        )
        self.inplanes = planes * block.expansion  # type: ignore[attr-defined]
        for _i in range(1, blocks):
            layers.append(block(self.inplanes, planes, stride=1, dilation=dilation, dimension=self.D))  # noqa: PERF401

        return nn.Sequential(*layers)

    def forward(self, x: ME.SparseTensor) -> Any:  # noqa: ANN401
        """
        Forward pass of the ResNetBase model.

        :param x: Input sparse tensor.
        :return: Output tensor after passing through the model.
        """
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.conv5(x)
        x = self.glob_pool(x)
        return self.final(x)


class MinkUNetBase(ResNetBase):
    """MinkUNetBase."""

    BLOCK: type[BasicBlock | Bottleneck] | None = None
    DILATIONS: tuple[int, ...] = (1, 1, 1, 1, 1, 1, 1, 1)
    LAYERS: tuple[int, ...] = (2, 2, 2, 2, 2, 2, 2, 2)
    PLANES: tuple[int, ...] = (32, 64, 128, 256, 256, 128, 96, 96)
    INIT_DIM: int = 32
    OUT_TENSOR_STRIDE: int = 1

    def __init__(self, in_channels: int, out_channels: int, d: int = 3) -> None:
        """
        Initialize the MinkUNetBase model.

        :param in_channels: Number of input channels.
        :param out_channels: Number of output channels.
        :param d: Dimension of the Minkowski space.
        """
        ResNetBase.__init__(self, in_channels, out_channels, d)

    def network_initialization(self, in_channels: int, out_channels: int, d: int) -> None:
        """
        Initialize the network layers.

        :param in_channels: Number of input channels.
        :param out_channels: Number of output channels.
        :param d: Dimension of the Minkowski space.
        """
        if self.BLOCK is None:  # Pour que mypy arrete de m'embeter
            message = "BLOCK is None"
            raise ValueError(message)
        # Output of the first conv concated to conv6
        self.inplanes = self.INIT_DIM
        self.conv0p1s1 = ME.MinkowskiConvolution(in_channels, self.inplanes, kernel_size=2, dimension=d)

        bn_momentum = 0.02
        self.bn0 = ME.MinkowskiBatchNorm(self.inplanes, momentum=bn_momentum)

        self.conv1p1s2 = ME.MinkowskiConvolution(self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=d)
        self.bn1 = ME.MinkowskiBatchNorm(self.inplanes, momentum=bn_momentum)

        self.block1 = self._make_layer(self.BLOCK, self.PLANES[0], self.LAYERS[0], bn_momentum=bn_momentum)

        self.conv2p2s2 = ME.MinkowskiConvolution(self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=d)
        self.bn2 = ME.MinkowskiBatchNorm(self.inplanes, momentum=bn_momentum)

        self.block2 = self._make_layer(self.BLOCK, self.PLANES[1], self.LAYERS[1], bn_momentum=bn_momentum)

        self.conv3p4s2 = ME.MinkowskiConvolution(self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=d)

        self.bn3 = ME.MinkowskiBatchNorm(self.inplanes, momentum=bn_momentum)
        self.block3 = self._make_layer(self.BLOCK, self.PLANES[2], self.LAYERS[2], bn_momentum=bn_momentum)

        self.conv4p8s2 = ME.MinkowskiConvolution(self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=d)
        self.bn4 = ME.MinkowskiBatchNorm(self.inplanes, momentum=bn_momentum)
        self.block4 = self._make_layer(self.BLOCK, self.PLANES[3], self.LAYERS[3], bn_momentum=bn_momentum)

        self.att_1 = MultiHeadAttention(self.inplanes, num_heads=4)

        self.convtr4p16s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[4], kernel_size=2, stride=2, dimension=d
        )
        self.bntr4 = ME.MinkowskiBatchNorm(self.PLANES[4], momentum=bn_momentum)

        self.inplanes = self.PLANES[4] + self.PLANES[2] * self.BLOCK.expansion
        self.block5 = self._make_layer(self.BLOCK, self.PLANES[4], self.LAYERS[4], bn_momentum=bn_momentum)

        self.att_2 = MultiHeadAttention(self.inplanes, num_heads=4)

        self.convtr5p8s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[5], kernel_size=2, stride=2, dimension=d
        )
        self.bntr5 = ME.MinkowskiBatchNorm(self.PLANES[5], momentum=bn_momentum)

        self.inplanes = self.PLANES[5] + self.PLANES[1] * self.BLOCK.expansion
        self.block6 = self._make_layer(self.BLOCK, self.PLANES[5], self.LAYERS[5], bn_momentum=bn_momentum)

        self.convtr6p4s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[6], kernel_size=2, stride=2, dimension=d
        )
        self.bntr6 = ME.MinkowskiBatchNorm(self.PLANES[6], momentum=bn_momentum)

        self.inplanes = self.PLANES[6] + self.PLANES[0] * self.BLOCK.expansion
        self.block7 = self._make_layer(self.BLOCK, self.PLANES[6], self.LAYERS[6], bn_momentum=bn_momentum)

        self.convtr7p2s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[7], kernel_size=2, stride=2, dimension=d
        )
        self.bntr7 = ME.MinkowskiBatchNorm(self.PLANES[7], momentum=bn_momentum)

        self.inplanes = self.PLANES[7] + self.INIT_DIM
        self.block8 = self._make_layer(self.BLOCK, self.PLANES[7], self.LAYERS[7], bn_momentum=bn_momentum)

        self.final = ME.MinkowskiConvolution(
            self.PLANES[7] * self.BLOCK.expansion,
            out_channels,
            kernel_size=1,
            bias=True,
            dimension=d,
        )
        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, x: ME.TensorField) -> ME.TensorField:
        """
        Forward pass through the network.

        :param x: Input tensor.
        :return: Output tensor.
        """
        out = self.conv0p1s1(x)
        out = self.bn0(out)
        out_p1 = self.relu(out)

        out = self.conv1p1s2(out_p1)
        out = self.bn1(out)
        out = self.relu(out)
        out_b1p2 = self.block1(out)

        out_b1p2 = out_b1p2.float()  # remet en torch.float32

        out = self.conv2p2s2(out_b1p2)
        out = self.bn2(out)
        out = self.relu(out)
        out_b2p4 = self.block2(out)

        out_b2p4 = out_b2p4.float()  # remet en torch.float32

        out = self.conv3p4s2(out_b2p4)
        out = self.bn3(out)
        out = self.relu(out)
        out_b3p8 = self.block3(out)

        out_b3p8 = out_b3p8.float()  # remet en torch.float32

        # tensor_stride 16
        out = self.conv4p8s2(out_b3p8)
        out = self.bn4(out)
        out = self.relu(out)
        out = self.block4(out)

        out = out.float()  # remet en torch.float32

        # tensor_stride 8
        out = self.convtr4p16s2(out)
        out = self.bntr4(out)
        out = self.relu(out)

        out = ME.cat(out, out_b3p8)
        out = self.block5(out)

        out = out.float()  # remet en torch.float32

        # tensor_stride 4
        out = self.convtr5p8s2(out)
        out = self.bntr5(out)
        out = self.relu(out)

        out = out.float()  # remet en torch.float32

        out = ME.cat(out, out_b2p4)
        out = self.block6(out)

        out = out.float()  # remet en torch.float32

        # tensor_stride 2
        out = self.convtr6p4s2(out)
        out = self.bntr6(out)
        out = self.relu(out)

        out = ME.cat(out, out_b1p2)
        out = self.block7(out)

        out = out.float()  # remet en torch.float32

        # tensor_stride 1
        out = self.convtr7p2s2(out)
        out = self.bntr7(out)
        out = self.relu(out)

        out = out.float()  # remet en torch.float32

        out = ME.cat(out, out_p1)
        out = self.block8(out)

        return self.final(out)


class MinkUNet18(MinkUNetBase):
    """MinkUNet18."""

    BLOCK = BasicBlock
    LAYERS = (2, 2, 2, 2, 2, 2, 2, 2)


class MinkUNet34(MinkUNetBase):
    """MinkUNet34."""

    BLOCK = BasicBlock
    LAYERS = (2, 3, 4, 6, 2, 2, 2, 2)


class MinkUNet14(MinkUNetBase):
    """MinkUNet14."""

    BLOCK = BasicBlock
    LAYERS = (1, 1, 1, 1, 1, 1, 1, 1)


class MinkUNet14C(MinkUNet14):
    """MinkUNet14C."""

    PLANES = (32, 64, 128, 256, 192, 192, 128, 128)


class MinkUNet18A(MinkUNet18):
    """MinkUNet18A."""

    PLANES = (32, 64, 128, 256, 128, 128, 96, 96)


class MinkUNet18B(MinkUNet18):
    """MinkUNet18B."""

    PLANES = (32, 64, 128, 256, 128, 128, 128, 128)


class MinkUNet18D(MinkUNet18):
    """MinkUNet18D."""

    PLANES = (32, 64, 128, 256, 384, 384, 384, 384)


class MinkUNet50(MinkUNetBase):
    """MinkUNet50."""

    BLOCK = Bottleneck
    LAYERS = (2, 3, 4, 6, 2, 2, 2, 2)


class MinkUNet101(MinkUNetBase):
    """MinkUNet101."""

    BLOCK = Bottleneck
    LAYERS = (2, 3, 4, 23, 2, 2, 2, 2)


class MinkUNet34A(MinkUNet34):
    """MinkUNet34A."""

    PLANES = (32, 64, 128, 256, 256, 128, 64, 64)


class MinkUNet34B(MinkUNet34):
    """MinkUNet34B."""

    PLANES = (32, 64, 128, 256, 256, 128, 64, 32)


class MinkUNet34C(MinkUNet34):
    """MinkUNet34C."""

    PLANES = (32, 64, 128, 256, 256, 128, 96, 96)


class Binary_model(nn.Module):
    """Binary_model."""

    def __init__(self, model_: nn.Module) -> None:
        """
        Initialize the Binary_model with a given model.

        :param model_: The base model to use.
        """
        super().__init__()
        self.model = model_
        self.final_layer = ME.MinkowskiSigmoid()

    def forward(self, inputs: ME.TensorField) -> ME.TensorField:
        """
        Forward pass through the binary model.

        :param inputs: Input tensor.
        :return: Output tensor with sigmoid activation.
        """
        out = self.model(inputs)
        return self.final_layer(out)


MODEL_REGISTRY = {
    "MinkUNet14": MinkUNet14,
    "MinkUNet14C": MinkUNet14C,
    "MinkUNet18": MinkUNet18,
    "MinkUNet18A": MinkUNet18A,
    "MinkUNet18B": MinkUNet18B,
    "MinkUNet18D": MinkUNet18D,
    "MinkUNet34": MinkUNet34,
    "MinkUNet34A": MinkUNet34A,
    "MinkUNet34B": MinkUNet34B,
    "MinkUNet34C": MinkUNet34C,
    "MinkUNet50": MinkUNet50,
    "MinkUNet101": MinkUNet101,
}
