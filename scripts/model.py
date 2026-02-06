import math

import MinkowskiEngine as ME
import torch
from torch import nn


class SelfAttention(nn.Module):
    """
    Self-Attention layer for 3D sparse tensors using MinkowskiEngine, incorporating voxel coordinates.

    :param in_dim: Dimension of the input features.
    """

    def __init__(self, in_dim: int):
        super(SelfAttention, self).__init__()

        # Ajuster les dimensions d'entrée pour inclure les 3 coordonnées (x, y, z)
        self.linear_query = ME.MinkowskiLinear(in_dim + 3, in_dim // 4)
        self.linear_key = ME.MinkowskiLinear(in_dim + 3, in_dim // 4)
        self.linear_value = ME.MinkowskiLinear(in_dim + 3, in_dim)
        self.normalized = nn.Softmax(dim=-1)  # Corrected 'dimensions' to 'dim'

    def forward(self, x: ME.SparseTensor) -> ME.SparseTensor:
        """
        Forward pass of the SelfAttention layer.

        :param x: Input sparse tensor.
        :return: Output sparse tensor after applying self-attention.
        """
        identity = x

        # Normalize input features to prevent large values
        x_features_normalized = x.F / (torch.norm(x.F, dim=-1, keepdim=True) + 1e-6)

        # Normalize coordinates and clip to prevent large values
        tensor_stride = torch.tensor(x.tensor_stride, device=x.device, dtype=torch.float32)
        tensor_stride = torch.clamp(tensor_stride, min=1e-6)  # Prevent division by zero
        coords_normalized = x.coordinates[:, 1:].float() / tensor_stride.unsqueeze(0)
        coords_normalized = torch.clamp(coords_normalized, min=-100, max=100)  # Clip coordinates

        # Log input statistics for debugging
        if torch.isnan(x_features_normalized).any() or torch.isinf(x_features_normalized).any():
            print(f"Warning: NaNs or Infs in x_features_normalized: {x_features_normalized}")
        if torch.isnan(coords_normalized).any() or torch.isinf(coords_normalized).any():
            print(f"Warning: NaNs or Infs in coords_normalized: {coords_normalized}")

        # Concatenate normalized features and coordinates
        x_features_with_coords = torch.cat([x_features_normalized, coords_normalized], dim=-1)
        if torch.isnan(x_features_with_coords).any() or torch.isinf(x_features_with_coords).any():
            print(f"Warning: NaNs or Infs in x_features_with_coords: {x_features_with_coords}")

        x_with_coords = ME.SparseTensor(
            features=x_features_with_coords,
            coordinate_map_key=x.coordinate_map_key,
            coordinate_manager=x.coordinate_manager,
            tensor_stride=x.tensor_stride,
            device=x.device,
        )

        # Calculate Q, K, V with augmented features
        Q = self.linear_query(x_with_coords)
        K = self.linear_key(x_with_coords)
        V = self.linear_value(x_with_coords)

        # Check for NaNs in Q, K, V
        if torch.isnan(Q.F).any() or torch.isnan(K.F).any() or torch.isnan(V.F).any():
            print(
                f"Warning: NaNs in Q/K/V - Q.F max: {Q.F.abs().max().item()}, K.F max: {K.F.abs().max().item()}, V.F max: {V.F.abs().max().item()}"
            )

        # Compute attention scores with clipping
        K_features_t = K.F.t()  # Use .t() for 2D tensor transpose
        att_map = torch.matmul(Q.F, K_features_t) / math.sqrt(Q.F.size(1))
        att_map = torch.clamp(att_map, min=-100, max=100)  # Clip to prevent overflow

        # Stable softmax
        att_map_max = att_map.max(dim=-1, keepdim=True)[0]
        exp_att_map = torch.exp(att_map - att_map_max)
        out = exp_att_map / (exp_att_map.sum(dim=-1, keepdim=True) + 1e-6)

        if torch.isnan(out).any() or torch.isinf(out).any():
            print(f"Warning: NaNs or Infs in attention output: {out}")

        out = torch.matmul(out, V.F)

        # Create output SparseTensor
        out = ME.SparseTensor(
            features=out,
            tensor_stride=x.tensor_stride,
            device=x.device,
            coordinate_map_key=x.coordinate_map_key,
            coordinate_manager=x.coordinate_manager,
        )

        out = out + identity

        if torch.isnan(out.F).any() or torch.isinf(out.F).any():
            print(f"Warning: NaNs or Infs in final output: {out.F}")

        return out


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention layer for 3D sparse tensors using MinkowskiEngine.

    :param in_dim: Dimension of the input features.
    :param num_heads: Number of attention heads.
    """

    def __init__(self, in_dim: int, num_heads: int):
        super(MultiHeadAttention, self).__init__()

        assert in_dim % num_heads == 0, "Dimension d'entrée doit être divisible par le nombre de têtes"
        self.num_heads = num_heads
        self.head_dim = in_dim // num_heads

        self.linear_query = nn.ModuleList([ME.MinkowskiLinear(in_dim + 3, self.head_dim) for _ in range(num_heads)])
        self.linear_key = nn.ModuleList([ME.MinkowskiLinear(in_dim + 3, self.head_dim) for _ in range(num_heads)])
        self.linear_value = nn.ModuleList([ME.MinkowskiLinear(in_dim + 3, self.head_dim) for _ in range(num_heads)])
        self.linear_out = ME.MinkowskiLinear(in_dim, in_dim)
        self.normalized = nn.Softmax(dim=-1)

    def forward(self, x: ME.SparseTensor) -> ME.SparseTensor:
        identity = x
        all_heads: list[torch.Tensor] = []

        # Normalize input features
        x_features_normalized = x.F / (torch.norm(x.F, dim=-1, keepdim=True) + 1e-6)

        # Normalize coordinates and clip to prevent large values
        tensor_stride = torch.tensor(x.tensor_stride, device=x.device, dtype=torch.float32)
        tensor_stride = torch.clamp(tensor_stride, min=1e-6)
        coords_normalized = x.coordinates[:, 1:].float() / tensor_stride.unsqueeze(0)
        coords_normalized = torch.clamp(coords_normalized, min=-100, max=100)  # Clip coordinates

        # Concatenate features and coordinates
        x_features_with_coords = torch.cat([x_features_normalized, coords_normalized], dim=-1)
        if torch.isnan(x_features_with_coords).any() or torch.isinf(x_features_with_coords).any():
            print(f"Warning: NaNs or Infs in x_features_with_coords: {x_features_with_coords}")

        x_with_coords = ME.SparseTensor(
            features=x_features_with_coords,
            coordinate_map_key=x.coordinate_map_key,
            coordinate_manager=x.coordinate_manager,
        )

        for i in range(self.num_heads):
            Q = self.linear_query[i](x_with_coords)
            K = self.linear_key[i](x_with_coords)
            V = self.linear_value[i](x_with_coords)

            # Check for NaNs in Q, K, V
            if torch.isnan(Q.F).any() or torch.isnan(K.F).any() or torch.isnan(V.F).any():
                print(f"Warning: NaNs in Q/K/V for head {i}")
                print(
                    f"Q.F max: {Q.F.abs().max().item()}, K.F max: {K.F.abs().max().item()}, V.F max: {V.F.abs().max().item()}"
                )

            K_feat_t = K.F.t()
            attmap = torch.matmul(Q.F, K_feat_t) / math.sqrt(Q.F.size(1))
            attmap = torch.clamp(attmap, min=-100, max=100)

            # Stable softmax
            attmap_max = attmap.max(dim=-1, keepdim=True)[0]
            exp_attmap = torch.exp(attmap - attmap_max)
            out = exp_attmap / (exp_attmap.sum(dim=-1, keepdim=True) + 1e-6)

            if torch.isnan(out).any() or torch.isinf(out).any():
                print(f"Warning: NaNs or Infs in attention output for head {i}: {out}")

            out = torch.matmul(out, V.F)
            all_heads.append(out)

        all_heads_concat = torch.cat(all_heads, dim=-1)
        if torch.isnan(all_heads_concat).any() or torch.isinf(all_heads_concat).any():
            print(f"Warning: NaNs or Infs in all_heads_concat: {all_heads_concat}")

        out = ME.SparseTensor(
            features=all_heads_concat,
            tensor_stride=x.tensor_stride,
            device=x.device,
            coordinate_map_key=x.coordinate_map_key,
            coordinate_manager=x.coordinate_manager,
        )
        out = self.linear_out(out)
        out = out + identity

        if torch.isnan(out.F).any() or torch.isinf(out.F).any():
            print(f"Warning: NaNs or Infs in final output: {out.F}")

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
        downsample: nn.Module | None = None,
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
        downsample: nn.Module | None = None,
        bn_momentum: float = 0.1,
        dimension: int = -1,
    ) -> None:
        super(Bottleneck, self).__init__()
        assert dimension > 0

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

        out = out.float()  # remet en torch.float32

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
            ME.MinkowskiConvolution(in_channels, self.inplanes, kernel_size=3, stride=2, dimension=D),
            ME.MinkowskiInstanceNorm(self.inplanes),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiMaxPooling(kernel_size=2, stride=2, dimension=D),
        )

        self.layer1 = self._make_layer(self.BLOCK, self.PLANES[0], self.LAYERS[0], stride=2)
        self.layer2 = self._make_layer(self.BLOCK, self.PLANES[1], self.LAYERS[1], stride=2)
        self.layer3 = self._make_layer(self.BLOCK, self.PLANES[2], self.LAYERS[2], stride=2)
        self.layer4 = self._make_layer(self.BLOCK, self.PLANES[3], self.LAYERS[3], stride=2)

        self.conv5 = nn.Sequential(
            ME.MinkowskiDropout(),
            ME.MinkowskiConvolution(self.inplanes, self.inplanes, kernel_size=3, stride=3, dimension=D),
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
        block: type[nn.Module],
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
            layers.append(block(self.inplanes, planes, stride=1, dilation=dilation, dimension=self.D))

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
    BLOCK: nn.Module | None = None
    DILATIONS: tuple[int, ...] = (1, 1, 1, 1, 1, 1, 1, 1)
    LAYERS: tuple[int, ...] = (2, 2, 2, 2, 2, 2, 2, 2)
    PLANES = (32, 64, 128, 256, 256, 128, 96, 96)
    INIT_DIM: int = 32
    OUT_TENSOR_STRIDE: int = 1

    def __init__(self, in_channels: int, out_channels: int, conf, D: int = 3) -> None:
        """
        Initialize the MinkUNetBase model.

        :param in_channels: Number of input channels.
        :param out_channels: Number of output channels.
        :param conf: Configuration object (OmegaConf).
        :param D: Dimension of the Minkowski space.
        """
        self.conf = conf
        super().__init__(in_channels, out_channels, D)

    def network_initialization(self, in_channels: int, out_channels: int, D: int) -> None:
        """
        Initialize the network layers with dynamic attention placement based on config.

        :param in_channels: Number of input channels.
        :param out_channels: Number of output channels.
        :param D: Dimension of the Minkowski space.
        """
        # Validate config parameters
        if self.conf.self_attention not in [0, 1, 2]:
            raise ValueError("self_attention must be 0, 1, or 2")
        if self.conf.multihead_attention not in [0, 1, 2]:
            raise ValueError("multihead_attention must be 0, 1, or 2")
        if self.conf.multihead_attention > 0 and not hasattr(self.conf, 'num_head'):
            raise ValueError("num_head must be specified when multihead_attention > 0")

        # Output of the first conv concatenated to conv6
        self.inplanes = self.INIT_DIM
        self.conv0p1s1 = ME.MinkowskiConvolution(in_channels, self.inplanes, kernel_size=2, dimension=D)

        bn_momentum = 0.02
        self.bn0 = ME.MinkowskiBatchNorm(self.inplanes, momentum=bn_momentum)

        self.conv1p1s2 = ME.MinkowskiConvolution(self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)
        self.bn1 = ME.MinkowskiBatchNorm(self.inplanes, momentum=bn_momentum)

        self.block1 = self._make_layer(self.BLOCK, self.PLANES[0], self.LAYERS[0], bn_momentum=bn_momentum)

        self.conv2p2s2 = ME.MinkowskiConvolution(self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)
        self.bn2 = ME.MinkowskiBatchNorm(self.inplanes, momentum=bn_momentum)

        self.block2 = self._make_layer(self.BLOCK, self.PLANES[1], self.LAYERS[1], bn_momentum=bn_momentum)

        # Initialize attention modules based on config
        self.att_1 = None
        self.att_2 = None
        self.att_3 = None
        self.att_4 = None

        total_attention = self.conf.self_attention + self.conf.multihead_attention
        if total_attention > 4:
            raise ValueError("Total attention layers (self_attention + multihead_attention) cannot exceed 4")

        # Compute in_dim for block5 attention due to skip connection
        block5_in_dim = self.PLANES[4] + self.PLANES[2] * self.BLOCK.expansion

        if self.conf.self_attention == 1 and self.conf.multihead_attention == 0:
            # Case 1: 1 SelfAttention after block4
            self.att_1 = SelfAttention(self.PLANES[3])  # in_dim=256
        elif self.conf.self_attention == 2 and self.conf.multihead_attention == 0:
            # Case 2: 2 SelfAttention after block3, block5
            self.att_1 = SelfAttention(self.PLANES[2])  # in_dim=128
            self.att_2 = SelfAttention(self.PLANES[4])  # in_dim=384 or 768
        elif self.conf.self_attention == 2 and self.conf.multihead_attention == 1:
            # Case 3: 2 SelfAttention after block3, block5; 1 MultiHeadAttention after block4
            self.att_1 = SelfAttention(self.PLANES[2])  # in_dim=128
            self.att_2 = MultiHeadAttention(self.PLANES[3], num_heads=self.conf.num_head)  # in_dim=256
            self.att_3 = SelfAttention(self.PLANES[4])  # in_dim=384 or 768
        elif self.conf.self_attention == 0 and self.conf.multihead_attention == 1:
            # Case 4: 1 MultiHeadAttention after block4
            self.att_1 = MultiHeadAttention(self.PLANES[3], num_heads=self.conf.num_head)  # in_dim=256
        elif self.conf.self_attention == 0 and self.conf.multihead_attention == 2:
            # Case 5: 2 MultiHeadAttention after block4, block5
            self.att_1 = MultiHeadAttention(self.PLANES[3], num_heads=self.conf.num_head)  # in_dim=256
            self.att_2 = MultiHeadAttention(self.PLANES[4], num_heads=self.conf.num_head)  # in_dim=384 or 768
        elif self.conf.self_attention == 1 and self.conf.multihead_attention == 2:
            # Case 6: 1 SelfAttention after block3; 2 MultiHeadAttention after block4, block5
            self.att_1 = SelfAttention(self.PLANES[2])  # in_dim=128
            self.att_2 = MultiHeadAttention(self.PLANES[3], num_heads=self.conf.num_head)  # in_dim=256
            self.att_3 = MultiHeadAttention(self.PLANES[4], num_heads=self.conf.num_head)  # in_dim=384 or 768
        elif self.conf.self_attention == 2 and self.conf.multihead_attention == 2:
            # Case 7: 2 SelfAttention after block2, block3; 2 MultiHeadAttention after block4, block5
            self.att_1 = SelfAttention(self.PLANES[1])  # in_dim=64
            self.att_2 = SelfAttention(self.PLANES[2])  # in_dim=128
            self.att_3 = MultiHeadAttention(self.PLANES[3], num_heads=self.conf.num_head)  # in_dim=256
            self.att_4 = MultiHeadAttention(self.PLANES[4], num_heads=self.conf.num_head)  # in_dim=384 or 768

        self.conv3p4s2 = ME.MinkowskiConvolution(self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)
        self.bn3 = ME.MinkowskiBatchNorm(self.inplanes, momentum=bn_momentum)
        self.block3 = self._make_layer(self.BLOCK, self.PLANES[2], self.LAYERS[2], bn_momentum=bn_momentum)

        self.conv4p8s2 = ME.MinkowskiConvolution(self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)
        self.bn4 = ME.MinkowskiBatchNorm(self.inplanes, momentum=bn_momentum)
        self.block4 = self._make_layer(self.BLOCK, self.PLANES[3], self.LAYERS[3], bn_momentum=bn_momentum)

        self.convtr4p16s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[4], kernel_size=2, stride=2, dimension=D
        )
        self.bntr4 = ME.MinkowskiBatchNorm(self.PLANES[4], momentum=bn_momentum)

        self.inplanes = self.PLANES[4] + self.PLANES[2] * self.BLOCK.expansion
        self.block5 = self._make_layer(self.BLOCK, self.PLANES[4], self.LAYERS[4], bn_momentum=bn_momentum)

        self.convtr5p8s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[5], kernel_size=2, stride=2, dimension=D
        )
        self.bntr5 = ME.MinkowskiBatchNorm(self.PLANES[5], momentum=bn_momentum)

        self.inplanes = self.PLANES[5] + self.PLANES[1] * self.BLOCK.expansion
        self.block6 = self._make_layer(self.BLOCK, self.PLANES[5], self.LAYERS[5], bn_momentum=bn_momentum)

        self.convtr6p4s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[6], kernel_size=2, stride=2, dimension=D
        )
        self.bntr6 = ME.MinkowskiBatchNorm(self.PLANES[6], momentum=bn_momentum)

        self.inplanes = self.PLANES[6] + self.PLANES[0] * self.BLOCK.expansion
        self.block7 = self._make_layer(self.BLOCK, self.PLANES[6], self.LAYERS[6], bn_momentum=bn_momentum)

        self.convtr7p2s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[7], kernel_size=2, stride=2, dimension=D
        )
        self.bntr7 = ME.MinkowskiBatchNorm(self.PLANES[7], momentum=bn_momentum)

        self.inplanes = self.PLANES[7] + self.INIT_DIM
        self.block8 = self._make_layer(self.BLOCK, self.PLANES[7], self.LAYERS[7], bn_momentum=bn_momentum)

        self.final = ME.MinkowskiConvolution(
            self.PLANES[7] * self.BLOCK.expansion,
            out_channels,
            kernel_size=1,
            bias=True,
            dimension=D,
        )
        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, x: ME.TensorField) -> ME.TensorField:
        """
        Forward pass through the network with dynamic attention.

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
        out_b1p2 = out_b1p2.float()

        out = self.conv2p2s2(out_b1p2)
        out = self.bn2(out)
        out = self.relu(out)
        out_b2p4 = self.block2(out)
        if self.att_1 and self.conf.self_attention == 2 and self.conf.multihead_attention == 2:
            out_b2p4 = self.att_1(out_b2p4).float()  # Case 7: SelfAttention after block2

        out_b2p4 = out_b2p4.float()

        out = self.conv3p4s2(out_b2p4)
        out = self.bn3(out)
        out = self.relu(out)
        out_b3p8 = self.block3(out)
        if self.att_1 and (
            (self.conf.self_attention == 2 and self.conf.multihead_attention in [0, 1]) or
            (self.conf.self_attention == 1 and self.conf.multihead_attention == 2)
        ):
            out_b3p8 = self.att_1(out_b3p8).float()  # Cases 2, 3, 6
        elif self.att_2 and self.conf.self_attention == 2 and self.conf.multihead_attention == 2:
            out_b3p8 = self.att_2(out_b3p8).float()  # Case 7

        out_b3p8 = out_b3p8.float()

        out = self.conv4p8s2(out_b3p8)
        out = self.bn4(out)
        out = self.relu(out)
        out = self.block4(out)
        if self.att_1 and (
            (self.conf.self_attention == 1 and self.conf.multihead_attention == 0) or
            (self.conf.multihead_attention == 1 and self.conf.self_attention == 0)
        ):
            out = self.att_1(out).float()  # Cases 1, 4
        elif self.att_2 and (
            (self.conf.self_attention == 2 and self.conf.multihead_attention == 1) or
            (self.conf.self_attention == 1 and self.conf.multihead_attention == 2)
        ):
            out = self.att_2(out).float()  # Cases 3, 6
        elif self.att_3 and self.conf.self_attention == 2 and self.conf.multihead_attention == 2:
            out = self.att_3(out).float()  # Case 7

        out = self.convtr4p16s2(out)
        out = self.bntr4(out)
        out = self.relu(out)

        out = ME.cat(out, out_b3p8)
        out = self.block5(out)
        if self.att_2 and (
            (self.conf.self_attention == 2 and self.conf.multihead_attention == 0) or
            (self.conf.multihead_attention == 2 and self.conf.self_attention == 0)
        ):
            out = self.att_2(out).float()  # Cases 2, 5
        elif self.att_3 and (
            (self.conf.self_attention == 2 and self.conf.multihead_attention == 1) or
            (self.conf.self_attention == 1 and self.conf.multihead_attention == 2)
        ):
            out = self.att_3(out).float()  # Cases 3, 6
        elif self.att_4 and self.conf.self_attention == 2 and self.conf.multihead_attention == 2:
            out = self.att_4(out).float()  # Case 7

        out = self.convtr5p8s2(out)
        out = self.bntr5(out)
        out = self.relu(out)

        out = ME.cat(out, out_b2p4)
        out = self.block6(out)
        out = out.float()

        out = self.convtr6p4s2(out)
        out = self.bntr6(out)
        out = self.relu(out)

        out = ME.cat(out, out_b1p2)
        out = self.block7(out)
        out = out.float()

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
