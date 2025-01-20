import os
import sys

os.environ["VXM_BACKEND"] = "pytorch"
sys.path.append(r"E:\Kai_2\CODE_Repository\UVI-Net-Extended")

import torch
import voxelmorph as vxm
from torchinfo import summary
from models.UNet.model import Unet3D, Unet3D_multi
from models.UNet.modelV4 import UNet3D, UNet3DMulti

# configure unet features
nb_features = [
    [8, 16, 32],             # encoder features
    [32, 32, 32, 8, 8, 3]    # decoder features
]
inshape = [128, 128, 128]
additional_dims = [4, 8, 16]

#refinement_model = Unet3D(inshape)
#refinement_model = UNet3D(inshape, encoder_features=nb_features[0], decoder_features=nb_features[1])
#refinement_model = Unet3D_multi(inshape)

#refinement_model = Unet3D_multi(inshape, nb_features=nb_features, add_dim=additional_dims)
#refinement_model = UNet3DMulti(inshape, nb_features=nb_features, add_dim=additional_dims)
refinement_model = UNet3DMulti(inshape, feature_maps=nb_features, additional_dims=additional_dims)

# Test with random input
batch_size = 1
x = torch.randn(batch_size, 1, *inshape)

# Calculate feature shapes for each level
feat_shapes = []
current_shape = inshape.copy()
for _ in additional_dims:
    feat_shapes.append(current_shape)
    current_shape = [s // 2 for s in current_shape]  # Divide dimensions by 2 for each level

feat_list_1 = [
    torch.randn(batch_size, dim, *shape)
    for dim, shape in zip(additional_dims, feat_shapes)
]
feat_list_2 = [
    torch.randn(batch_size, dim, *shape)
    for dim, shape in zip(additional_dims, feat_shapes)
]

output = refinement_model(x, feat_list_1, feat_list_2)
summary(refinement_model, input_data=(x, feat_list_1, feat_list_2))
print(refinement_model)
print(output.shape)

"""
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
Unet3D_multi                             [1, 1, 128, 128, 128]     --
├─ModuleList: 1-1                        --                        --
│    └─ConvBlock: 2-1                    [1, 8, 128, 128, 128]     --
│    │    └─Conv3d: 3-1                  [1, 8, 128, 128, 128]     224
│    │    └─LeakyReLU: 3-2               [1, 8, 128, 128, 128]     --
│    └─ConvBlock: 2-2                    [1, 16, 64, 64, 64]       --
│    │    └─Conv3d: 3-3                  [1, 16, 64, 64, 64]       6,928
│    │    └─LeakyReLU: 3-4               [1, 16, 64, 64, 64]       --
│    └─ConvBlock: 2-3                    [1, 32, 32, 32, 32]       --
│    │    └─Conv3d: 3-5                  [1, 32, 32, 32, 32]       27,680
│    │    └─LeakyReLU: 3-6               [1, 32, 32, 32, 32]       --
├─ModuleList: 1-6                        --                        (recursive)
│    └─ConvBlock: 2-4                    [1, 32, 32, 32, 32]       --
│    │    └─Conv3d: 3-7                  [1, 32, 32, 32, 32]       55,328
│    │    └─LeakyReLU: 3-8               [1, 32, 32, 32, 32]       --
├─Upsample: 1-3                          [1, 32, 64, 64, 64]       --
├─ModuleList: 1-6                        --                        (recursive)
│    └─ConvBlock: 2-5                    [1, 32, 64, 64, 64]       --
│    │    └─Conv3d: 3-9                  [1, 32, 64, 64, 64]       55,328
│    │    └─LeakyReLU: 3-10              [1, 32, 64, 64, 64]       --
├─Upsample: 1-5                          [1, 32, 128, 128, 128]    --
├─ModuleList: 1-6                        --                        (recursive)
│    └─ConvBlock: 2-6                    [1, 32, 128, 128, 128]    --
│    │    └─Conv3d: 3-11                 [1, 32, 128, 128, 128]    41,504
│    │    └─LeakyReLU: 3-12              [1, 32, 128, 128, 128]    --
├─ModuleList: 1-7                        --                        --
│    └─ConvBlock: 2-7                    [1, 8, 128, 128, 128]     --
│    │    └─Conv3d: 3-13                 [1, 8, 128, 128, 128]     7,136
│    │    └─LeakyReLU: 3-14              [1, 8, 128, 128, 128]     --
│    └─ConvBlock: 2-8                    [1, 8, 128, 128, 128]     --
│    │    └─Conv3d: 3-15                 [1, 8, 128, 128, 128]     1,736
│    │    └─LeakyReLU: 3-16              [1, 8, 128, 128, 128]     --
│    └─ConvBlock: 2-9                    [1, 3, 128, 128, 128]     --
│    │    └─Conv3d: 3-17                 [1, 3, 128, 128, 128]     651
│    │    └─LeakyReLU: 3-18              [1, 3, 128, 128, 128]     --
├─Conv3d: 1-8                            [1, 1, 128, 128, 128]     82
==========================================================================================
Total params: 196,597
Trainable params: 196,597
Non-trainable params: 0
Total mult-adds (Units.GIGABYTES): 126.69
==========================================================================================
Input size (MB): 96.47
Forward/backward pass size (MB): 1124.07
Params size (MB): 0.79
Estimated Total Size (MB): 1221.33
==========================================================================================

Unet3D_multi(
  (upsample): Upsample(scale_factor=2.0, mode='trilinear')
  (downarm): ModuleList(
    (0): ConvBlock(
      (main): Conv3d(1, 8, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
      (activation): LeakyReLU(negative_slope=0.2)
    )
    (1): ConvBlock(
      (main): Conv3d(16, 16, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
      (activation): LeakyReLU(negative_slope=0.2)
    )
    (2): ConvBlock(
      (main): Conv3d(32, 32, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
      (activation): LeakyReLU(negative_slope=0.2)
    )
  )
  (uparm): ModuleList(
    (0-1): 2 x ConvBlock(
      (main): Conv3d(64, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
      (activation): LeakyReLU(negative_slope=0.2)
    )
    (2): ConvBlock(
      (main): Conv3d(48, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
      (activation): LeakyReLU(negative_slope=0.2)
    )
  )
  (extras): ModuleList(
    (0): ConvBlock(
      (main): Conv3d(33, 8, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
      (activation): LeakyReLU(negative_slope=0.2)
    )
    (1): ConvBlock(
      (main): Conv3d(8, 8, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
      (activation): LeakyReLU(negative_slope=0.2)
    )
    (2): ConvBlock(
      (main): Conv3d(8, 3, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
      (activation): LeakyReLU(negative_slope=0.2)
    )
  )
  (last_layer): Conv3d(3, 1, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
)
"""