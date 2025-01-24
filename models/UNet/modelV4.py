
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from dataclasses import dataclass
from typing import Tuple, List, Optional
from torch.distributions.normal import Normal

@dataclass
class UNet3DConfig:
    """Configuration for UNet3D model"""
    inshape: Tuple[int, ...]
    feature_dim: Optional[int] = None
    encoder_features: Tuple[int, ...] = (8, 32, 32)
    decoder_features: Tuple[int, ...] = (32, 32, 32, 8, 8, 3)


@dataclass
class MultiUNet3DConfig:
    """Configuration for Multi-Feature UNet3D model"""
    inshape: Tuple[int, ...]
    encoder_features: Tuple[int, ...] = (8, 16, 32)
    decoder_features: Tuple[int, ...] = (32, 32, 32, 8, 8, 3)
    additional_dims: Tuple[int, ...] = (4, 8, 16)  # Additional feature dimensions at each level

    def __post_init__(self):
        """Validate configuration parameters"""
        if len(self.additional_dims) != len(self.encoder_features):
            raise ValueError(
                f"Additional dimensions ({len(self.additional_dims)}) must match "
                f"encoder features length ({len(self.encoder_features)})"
            )


class ConvBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, ndims, in_channels, out_channels, stride=1):
        super().__init__()
        Conv = getattr(nn, f"Conv{ndims}d")
        self.main = Conv(in_channels, out_channels, 3, stride, 1)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        out = self.main(x)
        out = self.activation(out)
        return out


class UNet3D(nn.Module):
    """
    A unet architecture. Layer features can be specified directly as a list of encoder and decoder
    features or as a single integer along with a number of unet levels. The default network features
    per layer (when no options are specified) are:
        encoder: [16, 32, 32, 32]
        decoder: [32, 32, 32, 32, 32, 16, 16]
    """

    def __init__(self, inshape, feature_dim=None, encoder_features=[8, 32, 32], decoder_features=[32, 32, 32, 8, 8, 3]):
        """
        inshape: Input shape. e.g. (192, 192, 192)
        feature_dim: Number of input features.
        """
        super().__init__()

        # ensure correct dimensionality
        ndims = len(inshape)
        if ndims not in {2, 3}:
            raise ValueError(f"Expected 2 or 3 dimensions, got {ndims}")

        # Calculate initial channel dimensions
        self.extra_channels = 1 + (feature_dim or 0)
        initial_channels = 1 + (feature_dim or 0)

        # Encoder pathway
        self.encoder = nn.ModuleList()
        prev_channels = initial_channels
        for out_channels in encoder_features:
            self.encoder.append(ConvBlock(ndims, prev_channels, out_channels, stride=2))
            prev_channels = out_channels

        # Decoder pathway with skip connections
        self.decoder = nn.ModuleList()
        reversed_encoder_features = list(reversed(encoder_features))
        
        for i, out_channels in enumerate(decoder_features[:len(encoder_features)]):
            in_channels = prev_channels + reversed_encoder_features[i] if i > 0 else prev_channels
            self.decoder.append(ConvBlock(ndims, in_channels, out_channels))
            prev_channels = out_channels

        # Extra convolutions at full resolution
        self.extra_convs = nn.ModuleList()
        prev_channels += self.extra_channels
        for out_channels in decoder_features[len(encoder_features):]:
            self.extra_convs.append(ConvBlock(ndims, prev_channels, out_channels))
            prev_channels = out_channels

        # Final layer
        Conv = getattr(nn, f"Conv{ndims}d")
        self.final_conv = Conv(decoder_features[-1], 1, kernel_size=3, padding=1)
        self._init_final_layer()

        # Upsampling operation
        self.upsample = nn.Upsample(scale_factor=2, mode="trilinear" if ndims == 3 else "bilinear")


    def _init_final_layer(self) -> None:
        """Initialize the final layer with small weights and zero bias"""
        nn.init.normal_(self.final_conv.weight, mean=0, std=1e-5)
        nn.init.zeros_(self.final_conv.bias)


    def forward(self, x: torch.Tensor, 
                feat1: Optional[torch.Tensor] = None,
                feat2: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the network.
        
        Args:
            x: Input tensor
            feat1: Optional first feature tensor
            feat2: Optional second feature tensor
            
        Returns:
            Output tensor with interpolated features
        """
        # Prepare input with optional features
        if feat1 is not None and feat2 is not None:
            x = torch.cat([x, feat1, feat2], dim=1)

        # Encoder pathway with skip connections
        skip_connections = []
        for encoder_layer in self.encoder:
            skip_connections.append(x)
            x = encoder_layer(x)

        # Decoder pathway
        skip_connections = skip_connections[::-1]
        for i, decoder_layer in enumerate(self.decoder):
            x = decoder_layer(x)
            x = self.upsample(x)
            if i < len(skip_connections):
                x = torch.cat([x, skip_connections[i]], dim=1)

        # Extra convolutions
        for conv in self.extra_convs:
            x = conv(x)

        return self.final_conv(x)


    @torch.jit.ignore
    def get_parameter_count(self) -> int:
        """Return the total number of parameters in the model"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# class UNet3DMulti(nn.Module):
#     """
#     A 3D UNet architecture that supports multiple feature inputs.
    
#     Default network features per layer (when no options are specified):
#         encoder: [8, 16, 32]
#         decoder: [32, 32, 32, 8, 8, 3]
#         additional dimensions: [4, 8, 16]
    
#     Args:
#         inshape: Input shape tuple. e.g. (192, 192, 192)
#         nb_features: Tuple of encoder and decoder features lists.
#                     Format: ((encoder_features), (decoder_features))
#         add_dim: Additional feature dimensions for each encoder level.
#     """

#     def __init__(self, inshape, nb_features=None, add_dim=(4, 8, 16)):
#         super().__init__()
#         """
#         Parameters:
#             inshape: Input shape. e.g. (192, 192, 192)
#             nb_features: Unet convolutional features. Can be specified via a list of lists with
#                 the form [[encoder feats], [decoder feats]], or as a single integer. If None (default),
#                 the unet features are defined by the default config described in the class documentation.
#             nb_levels: Number of levels in unet. Only used when nb_features is an integer. Default is None.
#             feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. Default is 1.
#         """

#         # Validate input dimensions
#         ndims = len(inshape)
#         if ndims not in {2, 3}:
#             raise ValueError(f"Expected 2 or 3 dimensions, got {ndims}")
        

#         # Set default features if none provided
#         if nb_features is None:
#             nb_features = ((8, 16, 32), (32, 32, 32, 8, 8, 3))

#         self.encoder_features, self.decoder_features = nb_features
#         self.additional_dims = add_dim
#         extra_nf, prev_nf = 1, 1

#         self.upsample = nn.Upsample(scale_factor=2, mode="trilinear")

#         # configure encoder (down-sampling path)
#         self.downarm = nn.ModuleList()
#         for i, nf in enumerate(self.encoder_features):
#             if i == 0:
#                 self.downarm.append(ConvBlock(ndims, prev_nf, nf, stride=1))
#             else:
#                 self.downarm.append(ConvBlock(ndims, prev_nf, nf, stride=2))
#             prev_nf = nf + (self.additional_dims[i] * 2)

#         # Decoder pathway
#         self.decoder = nn.ModuleList()
#         reversed_encoder_features = list(reversed(self.encoder_features))
#         reversed_additional_dims = list(reversed(self.additional_dims))
        
#         for i, nf in enumerate(self.decoder_features[: len(self.encoder_features)]):
#             channels = (
#                 prev_nf + reversed_encoder_features[i] + (reversed_additional_dims[i] * 2)
#                 if i > 0
#                 else prev_nf
#             )
#             self.decoder.append(ConvBlock(ndims, channels, nf, stride=1))
#             prev_nf = nf

#         # Extra convolutions at full resolution
#         prev_nf += extra_nf
#         self.extra_convs = nn.ModuleList()
#         for nf in self.decoder_features[len(self.encoder_features) :]:
#             self.extra_convs.append(ConvBlock(ndims, prev_nf, nf, stride=1))
#             prev_nf = nf

#         # Final layer (flow field layer)
#         Conv = getattr(nn, f"Conv{ndims}d")
#         self.final_conv = Conv(self.decoder_features[-1], 1, kernel_size=3, padding=1)

#         # init flow layer with small weights and bias
#         self.final_conv.weight = nn.Parameter(
#             Normal(0, 1e-5).sample(self.final_conv.weight.shape)
#         )
#         self.final_conv.bias = nn.Parameter(torch.zeros(self.final_conv.bias.shape))


#     def forward(self, x, feat_list_1, feat_list_2):
#         # get encoder activations
#         x_enc = [x]

#         for idx, layer in enumerate(self.downarm):
#             x_enc.append(
#                 torch.cat([layer(x_enc[-1]), feat_list_1[idx], feat_list_2[idx]], dim=1)
#             )

#         # conv, upsample, concatenate series
#         x = x_enc.pop()
#         for idx, layer in enumerate(self.decoder):
#             x = layer(x)
#             if idx != len(self.decoder) - 1:
#                 x = self.upsample(x)
#             x = torch.cat([x, x_enc.pop()], dim=1)

#         # extra convs at full resolution
#         for layer in self.extra_convs:
#             x = layer(x)

#         x = self.final_conv(x)

#         return x



from typing import Tuple, List, Optional
import torch
import torch.nn as nn
from torch.distributions.normal import Normal


class ConvBlock(nn.Module):
    """A convolutional block with LeakyReLU activation.
    
    Args:
        ndims: Number of spatial dimensions (2 or 3)
        in_channels: Number of input channels
        out_channels: Number of output channels
        stride: Stride for the convolution (default: 1)
    """
    def __init__(self, ndims: int, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        conv_class = getattr(nn, f"Conv{ndims}d")
        self.conv = conv_class(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=1
        )
        self.activation = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.conv(x))


class UNet3DMulti(nn.Module):
    """A 3D UNet architecture supporting multiple feature inputs.
    
    This network implements a UNet architecture with additional feature inputs
    at each encoder level. It's designed for tasks requiring multi-scale
    feature integration.
    
    Args:
        input_shape: Input shape tuple (D, H, W)
        feature_maps: Tuple of (encoder_features, decoder_features) lists
            Default: ((8, 16, 32), (32, 32, 32, 8, 8, 3))
        additional_dims: Feature dimensions to add at each encoder level
            Default: (4, 8, 16)
            
    Raises:
        ValueError: If input shape is not 2D or 3D
    """
    def __init__(
        self,
        input_shape: Tuple[int, ...],
        feature_maps: Optional[Tuple[Tuple[int, ...], Tuple[int, ...]]] = None,
        additional_dims: Tuple[int, ...] = (4, 8, 16)
    ) -> None:
        super().__init__()
        
        # Validate dimensions
        self.ndims = len(input_shape)
        if self.ndims not in {2, 3}:
            raise ValueError(f"Input must be 2D or 3D, got {self.ndims}D")
            
        # Set default feature maps if none provided
        if feature_maps is None:
            feature_maps = ((8, 16, 32), (32, 32, 32, 8, 8, 3))
            
        self.encoder_features, self.decoder_features = feature_maps
        self.additional_dims = additional_dims
        
        # Initialize layers
        self.upsample = nn.Upsample(scale_factor=2, mode="trilinear")
        self._build_encoder()
        self._build_decoder()
        self._init_final_layer()
        
    def _build_encoder(self) -> None:
        """Builds the encoder (downsampling) pathway."""
        self.encoder = nn.ModuleList()
        prev_channels = 1  # Starting with single-channel input
        
        for i, channels in enumerate(self.encoder_features):
            stride = 2 if i > 0 else 1
            self.encoder.append(
                ConvBlock(self.ndims, prev_channels, channels, stride)
            )
            prev_channels = channels + (self.additional_dims[i] * 2)
            
    def _build_decoder(self) -> None:
        """Builds the decoder (upsampling) pathway."""
        self.decoder = nn.ModuleList()
        self.final_convs = nn.ModuleList()
        
        # Reverse features for decoder path
        rev_encoder_features = list(reversed(self.encoder_features))
        rev_additional_dims = list(reversed(self.additional_dims))
        prev_channels = rev_encoder_features[0] + (rev_additional_dims[0] * 2)
        
        # Build main decoder path
        decoder_features_main = self.decoder_features[:len(self.encoder_features)]
        for i, channels in enumerate(decoder_features_main):
            if i > 0:
                prev_channels += rev_encoder_features[i] + (rev_additional_dims[i] * 2)
            self.decoder.append(ConvBlock(self.ndims, prev_channels, channels))
            prev_channels = channels

        # # Build main decoder path
        # decoder_features_main = self.decoder_features[:len(self.encoder_features)]
        # for i, channels in enumerate(decoder_features_main):
        #     in_channels = (prev_channels + rev_encoder_features[i] + (rev_additional_dims[i] * 2)
        #                 if i > 0 else prev_channels)
        #     self.decoder.append(ConvBlock(self.ndims, in_channels, channels))
        #     prev_channels = channels 

        # Build final convolution layers
        decoder_features_final = self.decoder_features[len(self.encoder_features):]
        #prev_channels += 1  # Add extra channel for concatenation
        for channels in decoder_features_final:
            self.final_convs.append(ConvBlock(self.ndims, prev_channels, channels))
            prev_channels = channels
            
    def _init_final_layer(self) -> None:
        """Initializes the final convolution layer with small weights."""
        conv_class = getattr(nn, f"Conv{self.ndims}d")
        self.final_conv = conv_class(
            in_channels=self.decoder_features[-1],
            out_channels=1,
            kernel_size=3,
            padding=1
        )
        
        # Initialize with small weights and zero bias
        self.final_conv.weight = nn.Parameter(
            Normal(0, 1e-5).sample(self.final_conv.weight.shape)
        )
        self.final_conv.bias = nn.Parameter(
            torch.zeros(self.final_conv.bias.shape)
        )
        
    def forward(
        self,
        x: torch.Tensor,
        feat_list_1: List[torch.Tensor],
        feat_list_2: List[torch.Tensor]
    ) -> torch.Tensor:
        """Forward pass of the network.
        
        Args:
            x: Input tensor
            feat_list_1: First list of additional features
            feat_list_2: Second list of additional features
            
        Returns:
            Output tensor after final convolution
        """
        # Encoder pathway
        encoder_features = [x]
        for i, layer in enumerate(self.encoder):
            features = layer(encoder_features[-1])
            combined = torch.cat([features, feat_list_1[i], feat_list_2[i]], dim=1)
            encoder_features.append(combined)
            
        # Decoder pathway with skip connections
        x = encoder_features.pop()
        for i, layer in enumerate(self.decoder):
            x = layer(x)
            if i < len(self.decoder) - 1:
                x = self.upsample(x)
                x = torch.cat([x, encoder_features.pop()], dim=1)

        # x = encoder_features.pop()
        # for i, layer in enumerate(self.decoder):
        #     x = layer(x)
        #     if i != len(self.decoder) - 1:  # Changed condition
        #         x = self.upsample(x)
        #         x = torch.cat([x, encoder_features.pop()], dim=1)

        # Final convolutions
        for layer in self.final_convs:
            x = layer(x)
            
        return self.final_conv(x)


class down(nn.Module):
    """
    A class for creating neural network blocks containing layers:

    Average Pooling --> Convlution + Leaky ReLU --> Convolution + Leaky ReLU

    This is used in the UNet Class to create a UNet like NN architecture.
    ...
    Methods
    -------
    forward(x)
        Returns output tensor after passing input `x` to the neural network
        block.
    """

    def __init__(self, inChannels, outChannels, filterSize):
        """
        Parameters
        ----------
            inChannels : int
                number of input channels for the first convolutional layer.
            outChannels : int
                number of output channels for the first convolutional layer.
                This is also used as input and output channels for the
                second convolutional layer.
            filterSize : int
                filter size for the convolution filter. input N would create
                a N x N filter.
        """

        super(down, self).__init__()
        # Initialize convolutional layers.
        self.conv1 = nn.Conv3d(
            inChannels,
            outChannels,
            filterSize,
            stride=1,
            padding=int((filterSize - 1) / 2),
        )
        self.conv2 = nn.Conv3d(
            outChannels,
            outChannels,
            filterSize,
            stride=1,
            padding=int((filterSize - 1) / 2),
        )

    def forward(self, x):
        """
        Returns output tensor after passing input `x` to the neural network
        block.
        Parameters
        ----------
            x : tensor
                input to the NN block.
        Returns
        -------
            tensor
                output of the NN block.
        """

        # Average pooling with kernel size 2 (2 x 2).
        x = F.avg_pool3d(x, 2)
        # Convolution + Leaky ReLU
        x = F.leaky_relu(self.conv1(x), negative_slope=0.1)
        # Convolution + Leaky ReLU
        x = F.leaky_relu(self.conv2(x), negative_slope=0.1)
        return x


class up(nn.Module):
    """
    A class for creating neural network blocks containing layers:

    Bilinear interpolation --> Convlution + Leaky ReLU --> Convolution + Leaky ReLU

    This is used in the UNet Class to create a UNet like NN architecture.
    ...
    Methods
    -------
    forward(x, skpCn)
        Returns output tensor after passing input `x` to the neural network
        block.
    """

    def __init__(self, inChannels, outChannels):
        """
        Parameters
        ----------
            inChannels : int
                number of input channels for the first convolutional layer.
            outChannels : int
                number of output channels for the first convolutional layer.
                This is also used for setting input and output channels for
                the second convolutional layer.
        """

        super(up, self).__init__()
        # Initialize convolutional layers.
        self.conv1 = nn.Conv3d(inChannels, outChannels, 3, stride=1, padding=1)
        # (2 * outChannels) is used for accommodating skip connection.
        self.conv2 = nn.Conv3d(2 * outChannels, outChannels, 3, stride=1, padding=1)

    def forward(self, x, skpCn):
        """
        Returns output tensor after passing input `x` to the neural network
        block.
        Parameters
        ----------
            x : tensor
                input to the NN block.
            skpCn : tensor
                skip connection input to the NN block.
        Returns
        -------
            tensor
                output of the NN block.
        """

        # Bilinear interpolation with scaling 2.
        x = F.interpolate(x, scale_factor=2, mode="trilinear")
        # Convolution + Leaky ReLU
        x = F.leaky_relu(self.conv1(x), negative_slope=0.1)
        # Convolution + Leaky ReLU on (`x`, `skpCn`)
        x = F.leaky_relu(self.conv2(torch.cat((x, skpCn), 1)), negative_slope=0.1)
        return x


class Unet3D_2(nn.Module):
    """
    A class for creating UNet like architecture as specified by the
    Super SloMo paper.

    ...
    Methods
    -------
    forward(x)
        Returns output tensor after passing input `x` to the neural network
        block.
    """

    def __init__(self, inChannels, outChannels):
        """
        Parameters
        ----------
            inChannels : int
                number of input channels for the UNet.
            outChannels : int
                number of output channels for the UNet.
        """

        super(Unet3D_2, self).__init__()
        # Initialize neural network blocks.
        self.conv1 = nn.Conv3d(inChannels, 32, 7, stride=1, padding=3)
        self.conv2 = nn.Conv3d(32, 32, 7, stride=1, padding=3)
        self.down1 = down(32, 64, 5)
        self.down2 = down(64, 128, 3)
        self.down3 = down(128, 256, 3)
        self.down4 = down(256, 512, 3)
        self.down5 = down(512, 512, 3)
        self.up1 = up(512, 512)
        self.up2 = up(512, 256)
        self.up3 = up(256, 128)
        self.up4 = up(128, 64)
        self.up5 = up(64, 32)
        self.conv3 = nn.Conv3d(32, outChannels, 3, stride=1, padding=1)

    def forward(self, x):
        """
        Returns output tensor after passing input `x` to the neural network.
        Parameters
        ----------
            x : tensor
                input to the UNet.
        Returns
        -------
            tensor
                output of the UNet.
        """

        x = F.leaky_relu(self.conv1(x), negative_slope=0.1)
        s1 = F.leaky_relu(self.conv2(x), negative_slope=0.1)
        s2 = self.down1(s1)
        s3 = self.down2(s2)
        s4 = self.down3(s3)
        s5 = self.down4(s4)
        x = self.down5(s5)
        x = self.up1(x, s5)
        x = self.up2(x, s4)
        x = self.up3(x, s3)
        x = self.up4(x, s2)
        x = self.up5(x, s1)
        x = F.leaky_relu(self.conv3(x), negative_slope=0.1)
        return x
