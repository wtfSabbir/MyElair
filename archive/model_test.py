import MinkowskiEngine as ME
import torch.nn as nn
import torch
import math
from typing import Optional, List, Type, Union, Tuple


class SelfAttention(nn.Module):
    """
    Self-Attention layer for 3D sparse tensors using MinkowskiEngine.

    :param in_dim: Dimension of the input features.
    """

    def __init__(self, in_dim: int):
        super(SelfAttention, self).__init__()

        self.linear_query = ME.MinkowskiLinear(in_dim, in_dim // 8)
        self.linear_key = ME.MinkowskiLinear(in_dim, in_dim // 8)
        self.linear_value = ME.MinkowskiLinear(in_dim, in_dim)
        self.pooling = ME.MinkowskiGlobalAvgPooling()
        self.normalized = nn.Softmax(dim=-1)

    def forward(self, x) -> ME.SparseTensor:
        """
        Forward pass of the SelfAttention layer.

        :param x: Input sparse tensor.
        :return: Output sparse tensor after applying self-attention.
        """

        identity = x
        Q = self.linear_query(x)
        K = self.linear_key(x)
        V = self.linear_value(x)
        K_feat_t = K.features.T

        attmap = torch.matmul(Q.F, K_feat_t) / (math.sqrt(Q.F.size(1)))
        out = self.normalized(attmap)
        out = torch.matmul(out, V.F)
        out = ME.SparseTensor(
            features=out,
            tensor_stride=x.tensor_stride,
            device=x.device,
            coordinate_map_key=x.coordinate_map_key,
            coordinate_manager=x.coordinate_manager,
        )
        out = out + identity
        return out


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention layer for 3D sparse tensors using MinkowskiEngine.

    :param in_dim: Dimension of the input features.
    :param num_heads: Number of attention heads.
    """

    def __init__(self, in_dim: int, num_heads: int):
        super(MultiHeadAttention, self).__init__()

        assert (
            in_dim % num_heads == 0
        ), "Dimension d'entrée doit être divisible par le nombre de têtes"
        self.num_heads = num_heads
        self.head_dim = in_dim // num_heads

        self.linear_query = nn.ModuleList(
            [ME.MinkowskiLinear(in_dim, self.head_dim) for _ in range(num_heads)]
        )
        self.linear_key = nn.ModuleList(
            [ME.MinkowskiLinear(in_dim, self.head_dim) for _ in range(num_heads)]
        )
        self.linear_value = nn.ModuleList(
            [ME.MinkowskiLinear(in_dim, self.head_dim) for _ in range(num_heads)]
        )

        # self.pooling = ME.MinkowskiGlobalAvgPooling()
        self.normalized = nn.Softmax(dim=-1)
        self.linear_out = ME.MinkowskiLinear(in_dim, in_dim)

    def forward(self, x) -> ME.SparseTensor:
        identity = x
        all_heads: List[torch.Tensor] = []

        for i in range(self.num_heads):
            Q = self.linear_query[i](x)  # Query for head i
            K = self.linear_key[i](x)  # Key for head i
            V = self.linear_value[i](x)  # Value for head i
            K_feat_t = K.features.T
            attmap = torch.matmul(Q.F, K_feat_t) / (math.sqrt(Q.F.size(1)))
            out = self.normalized(attmap)
            out = torch.matmul(out, V.F)
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
        out = out + identity

        return out


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

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        dilation: int = 1,
        downsample: Optional[nn.Module] = None,
        bn_momentum: float = 0.1,
        dimension: int = -1,
    ):
        super(BasicBlock, self).__init__()
        assert dimension > 0

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

    def forward(self, x) -> ME.SparseTensor:
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
        out = self.relu(out)

        return out


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

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        dilation: int = 1,
        downsample: Optional[nn.Module] = None,
        bn_momentum: float = 0.1,
        dimension: int = -1,
    ) -> None:
        super(Bottleneck, self).__init__()
        assert dimension > 0

        self.conv1 = ME.MinkowskiConvolution(
            inplanes, planes, kernel_size=1, dimension=dimension
        )
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

        self.conv3 = ME.MinkowskiConvolution(
            planes, planes * self.expansion, kernel_size=1, dimension=dimension
        )
        self.norm3 = ME.MinkowskiBatchNorm(
            planes * self.expansion, momentum=bn_momentum
        )

        self.relu = ME.MinkowskiReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x) -> ME.SparseTensor:
        """
        Forward pass of the Bottleneck block.

        :param x: Input sparse tensor.
        :return: Output sparse tensor after passing through the Bottleneck block.
        """
        residual = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.norm3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNetBase(nn.Module):
    """
    Base class for ResNet with MinkowskiEngine convolutions.

    :param in_channels: Number of input channels.
    :param out_channels: Number of output channels.
    :param D: Dimension of the convolutions (e.g., 3 for 3D). Default is 3.
    """

    BLOCK = None
    LAYERS = ()
    INIT_DIM = 64
    PLANES = (64, 128, 256, 512)

    def __init__(self, in_channels, out_channels, D=3):
        nn.Module.__init__(self)
        self.D = D
        assert self.BLOCK is not None

        self.network_initialization(in_channels, out_channels, D)
        self.weight_initialization()

    def network_initialization(self, in_channels, out_channels, D) -> None:

        self.inplanes = self.INIT_DIM
        self.conv1 = nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels, self.inplanes, kernel_size=3, stride=2, dimension=D
            ),
            ME.MinkowskiInstanceNorm(self.inplanes),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiMaxPooling(kernel_size=2, stride=2, dimension=D),
        )

        self.layer1 = self._make_layer(
            self.BLOCK, self.PLANES[0], self.LAYERS[0], stride=2
        )
        self.layer2 = self._make_layer(
            self.BLOCK, self.PLANES[1], self.LAYERS[1], stride=2
        )
        self.layer3 = self._make_layer(
            self.BLOCK, self.PLANES[2], self.LAYERS[2], stride=2
        )
        self.layer4 = self._make_layer(
            self.BLOCK, self.PLANES[3], self.LAYERS[3], stride=2
        )

        self.conv5 = nn.Sequential(
            ME.MinkowskiDropout(),
            ME.MinkowskiConvolution(
                self.inplanes, self.inplanes, kernel_size=3, stride=3, dimension=D
            ),
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

    def _make_layer(
        self,
        block: Type[nn.Module],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilation: int = 1,
        bn_momentum: float = 0.1,
    ) -> nn.Sequential:
        """
        Creates a layer in the ResNet model.

        :param block: Block type to be used in the layer.
        :param planes: Number of output planes.
        :param blocks: Number of blocks.
        :param stride: Stride of the convolution. Default is 1.
        :param dilation: Dilation rate of the convolution. Default is 1.
        :param bn_momentum: Batch normalization momentum. Default is 0.1.
        :return: A sequential container with the blocks for the layer.
        """

        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                ME.MinkowskiConvolution(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    dimension=self.D,
                ),
                ME.MinkowskiBatchNorm(planes * block.expansion, momentum=bn_momentum),
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
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(
                    self.inplanes, planes, stride=1, dilation=dilation, dimension=self.D
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x: ME.SparseTensor) -> torch.Tensor:
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
    BLOCK: Union[nn.Module, None] = None
    DILATIONS: Tuple[int, ...] = (1, 1, 1, 1, 1, 1, 1, 1)
    LAYERS: Tuple[int, ...] = (2, 2, 2, 2, 2, 2, 2, 2)
    PLANES = (32, 64, 128, 256, 256, 128, 96, 96)
    INIT_DIM: int = 32
    OUT_TENSOR_STRIDE: int = 1

    def __init__(self, in_channels: int, out_channels: int, D: int = 3) -> None:
        """
        Initialize the MinkUNetBase model.

        :param in_channels: Number of input channels.
        :param out_channels: Number of output channels.
        :param D: Dimension of the Minkowski space.
        """
        ResNetBase.__init__(self, in_channels, out_channels, D)

    def network_initialization(
        self, in_channels: int, out_channels: int, D: int
    ) -> None:
        """
        Initialize the network layers.

        :param in_channels: Number of input channels.
        :param out_channels: Number of output channels.
        :param D: Dimension of the Minkowski space.
        """

        # Output of the first conv concated to conv6
        self.inplanes = self.INIT_DIM
        self.conv0p1s1 = ME.MinkowskiConvolution(
            in_channels, self.inplanes, kernel_size=2, dimension=D
        )

        bn_momentum = 0.02
        self.bn0 = ME.MinkowskiBatchNorm(self.inplanes, momentum=bn_momentum)

        self.conv1p1s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D
        )
        self.bn1 = ME.MinkowskiBatchNorm(self.inplanes, momentum=bn_momentum)

        self.block1 = self._make_layer(
            self.BLOCK, self.PLANES[0], self.LAYERS[0], bn_momentum=bn_momentum
        )

        self.conv2p2s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D
        )
        self.bn2 = ME.MinkowskiBatchNorm(self.inplanes, momentum=bn_momentum)

        self.block2 = self._make_layer(
            self.BLOCK, self.PLANES[1], self.LAYERS[1], bn_momentum=bn_momentum
        )

        self.conv3p4s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D
        )

        self.bn3 = ME.MinkowskiBatchNorm(self.inplanes, momentum=bn_momentum)
        self.block3 = self._make_layer(
            self.BLOCK, self.PLANES[2], self.LAYERS[2], bn_momentum=bn_momentum
        )

        self.conv4p8s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D
        )
        self.bn4 = ME.MinkowskiBatchNorm(self.inplanes, momentum=bn_momentum)
        self.block4 = self._make_layer(
            self.BLOCK, self.PLANES[3], self.LAYERS[3], bn_momentum=bn_momentum
        )

        self.att_1 = MultiHeadAttention(self.inplanes, num_heads=4)

        self.convtr4p16s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[4], kernel_size=2, stride=2, dimension=D
        )
        self.bntr4 = ME.MinkowskiBatchNorm(self.PLANES[4], momentum=bn_momentum)

        self.inplanes = self.PLANES[4] + self.PLANES[2] * self.BLOCK.expansion
        self.block5 = self._make_layer(
            self.BLOCK, self.PLANES[4], self.LAYERS[4], bn_momentum=bn_momentum
        )

        self.att_2 = MultiHeadAttention(self.inplanes, num_heads=4)

        self.convtr5p8s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[5], kernel_size=2, stride=2, dimension=D
        )
        self.bntr5 = ME.MinkowskiBatchNorm(self.PLANES[5], momentum=bn_momentum)

        self.inplanes = self.PLANES[5] + self.PLANES[1] * self.BLOCK.expansion
        self.block6 = self._make_layer(
            self.BLOCK, self.PLANES[5], self.LAYERS[5], bn_momentum=bn_momentum
        )

        self.convtr6p4s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[6], kernel_size=2, stride=2, dimension=D
        )
        self.bntr6 = ME.MinkowskiBatchNorm(self.PLANES[6], momentum=bn_momentum)

        self.inplanes = self.PLANES[6] + self.PLANES[0] * self.BLOCK.expansion
        self.block7 = self._make_layer(
            self.BLOCK, self.PLANES[6], self.LAYERS[6], bn_momentum=bn_momentum
        )

        self.convtr7p2s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[7], kernel_size=2, stride=2, dimension=D
        )
        self.bntr7 = ME.MinkowskiBatchNorm(self.PLANES[7], momentum=bn_momentum)

        self.inplanes = self.PLANES[7] + self.INIT_DIM
        self.block8 = self._make_layer(
            self.BLOCK, self.PLANES[7], self.LAYERS[7], bn_momentum=bn_momentum
        )

        self.final = ME.MinkowskiConvolution(
            self.PLANES[7] * self.BLOCK.expansion,
            out_channels,
            kernel_size=1,
            bias=True,
            dimension=D,
        )
        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, x) -> ME.TensorField:
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

        out = self.conv2p2s2(out_b1p2)
        out = self.bn2(out)
        out = self.relu(out)
        out_b2p4 = self.block2(out)

        out = self.conv3p4s2(out_b2p4)
        out = self.bn3(out)
        out = self.relu(out)
        out_b3p8 = self.block3(out)

        # tensor_stride=16
        out = self.conv4p8s2(out_b3p8)
        out = self.bn4(out)
        out = self.relu(out)
        out = self.block4(out)

        # out = self.att_1(out)

        # tensor_stride=8
        out = self.convtr4p16s2(out)
        out = self.bntr4(out)
        out = self.relu(out)

        out = ME.cat(out, out_b3p8)
        out = self.block5(out)

        # out = self.att_2(out)

        # tensor_stride=4
        out = self.convtr5p8s2(out)
        out = self.bntr5(out)
        out = self.relu(out)

        out = ME.cat(out, out_b2p4)
        out = self.block6(out)

        # tensor_stride=2
        out = self.convtr6p4s2(out)
        out = self.bntr6(out)
        out = self.relu(out)

        out = ME.cat(out, out_b1p2)
        out = self.block7(out)

        # tensor_stride=1
        out = self.convtr7p2s2(out)
        out = self.bntr7(out)
        out = self.relu(out)

        out = ME.cat(out, out_p1)
        out = self.block8(out)
        out = self.final(out)

        return out


class MinkUNet18(MinkUNetBase):
    BLOCK = BasicBlock
    LAYERS = (2, 2, 2, 2, 2, 2, 2, 2)


class MinkUNet34(MinkUNetBase):
    BLOCK = BasicBlock
    LAYERS = (2, 3, 4, 6, 2, 2, 2, 2)

class MinkUNet14(MinkUNetBase):
    BLOCK = BasicBlock
    LAYERS = (1, 1, 1, 1, 1, 1, 1, 1)

class MinkUNet14C(MinkUNet14):
    PLANES = (32, 64, 128, 256, 192, 192, 128, 128)

class MinkUNet18A(MinkUNet18):
    PLANES = (32, 64, 128, 256, 128, 128, 96, 96)

class MinkUNet18B(MinkUNet18):
    PLANES = (32, 64, 128, 256, 128, 128, 128, 128)

class MinkUNet18D(MinkUNet18):
    PLANES = (32, 64, 128, 256, 384, 384, 384, 384)

class MinkUNet50(MinkUNetBase):
    BLOCK = Bottleneck
    LAYERS = (2, 3, 4, 6, 2, 2, 2, 2)


class MinkUNet101(MinkUNetBase):
    BLOCK = Bottleneck
    LAYERS = (2, 3, 4, 23, 2, 2, 2, 2)

class MinkUNet34(MinkUNetBase):
    BLOCK = BasicBlock
    LAYERS = (2, 3, 4, 6, 2, 2, 2, 2)

class MinkUNet34A(MinkUNet34):
    PLANES = (32, 64, 128, 256, 256, 128, 64, 64)


class MinkUNet34B(MinkUNet34):
    PLANES = (32, 64, 128, 256, 256, 128, 64, 32)


class MinkUNet34C(MinkUNet34):
    PLANES = (32, 64, 128, 256, 256, 128, 96, 96)


class Binary_model(nn.Module):
    def __init__(self, model_) -> None:
        """
        Initialize the Binary_model with a given model.

        :param model_: The base model to use.
        """
        super(Binary_model, self).__init__()
        self.model = model_
        self.final_layer = ME.MinkowskiSigmoid()

    def forward(self, inputs) -> ME.TensorField:
        """
        Forward pass through the binary model.

        :param inputs: Input tensor.
        :return: Output tensor with sigmoid activation.
        """
        out = self.model(inputs)
        out = self.final_layer(out)
        return out

