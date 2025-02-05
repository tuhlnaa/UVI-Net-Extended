import os
import sys
import torch
from pathlib import Path
from torchinfo import summary

os.environ["VXM_BACKEND"] = "pytorch"
PROJECT_ROOT = Path(__file__).parents[1]
sys.path.append(str(PROJECT_ROOT))

import voxelmorph as vxm
from models.UNet.modelV4 import UNet3D, Unet3D_multi

# configure unet features
nb_features = [
    [8, 32, 32],             # encoder features
    [32, 32, 32, 8, 8, 3]    # decoder features
]
inshape = [128, 128, 128]

refinement_model = UNet3D(inshape, encoder_features=nb_features[0], decoder_features=nb_features[1])

# Test with random input
x = torch.randn(1, 1, 128, 128, 128)
output = refinement_model(x)

print(refinement_model)
print(output.shape)
summary(refinement_model, input_data=(x))


"""
Unet3D(
  (upsample): Upsample(scale_factor=2.0, mode='trilinear')
  (downarm): ModuleList(
    (0): ConvBlock(
      (main): Conv3d(1, 8, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
      (activation): LeakyReLU(negative_slope=0.2)
    )
    (1): ConvBlock(
      (main): Conv3d(8, 32, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
      (activation): LeakyReLU(negative_slope=0.2)
    )
    (2): ConvBlock(
      (main): Conv3d(32, 32, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
      (activation): LeakyReLU(negative_slope=0.2)
    )
  )
  (uparm): ModuleList(
    (0): ConvBlock(
      (main): Conv3d(32, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
      (activation): LeakyReLU(negative_slope=0.2)
    )
    (1): ConvBlock(
      (main): Conv3d(64, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
      (activation): LeakyReLU(negative_slope=0.2)
    )
    (2): ConvBlock(
      (main): Conv3d(40, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
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

==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
Unet3D                                   [1, 1, 128, 128, 128]     --
├─ModuleList: 1-1                        --                        --
│    └─ConvBlock: 2-1                    [1, 8, 64, 64, 64]        --
│    │    └─Conv3d: 3-1                  [1, 8, 64, 64, 64]        224
│    │    └─LeakyReLU: 3-2               [1, 8, 64, 64, 64]        --
│    └─ConvBlock: 2-2                    [1, 32, 32, 32, 32]       --
│    │    └─Conv3d: 3-3                  [1, 32, 32, 32, 32]       6,944
│    │    └─LeakyReLU: 3-4               [1, 32, 32, 32, 32]       --
│    └─ConvBlock: 2-3                    [1, 32, 16, 16, 16]       --
│    │    └─Conv3d: 3-5                  [1, 32, 16, 16, 16]       27,680
│    │    └─LeakyReLU: 3-6               [1, 32, 16, 16, 16]       --
├─ModuleList: 1-6                        --                        (recursive)
│    └─ConvBlock: 2-4                    [1, 32, 16, 16, 16]       --
│    │    └─Conv3d: 3-7                  [1, 32, 16, 16, 16]       27,680
│    │    └─LeakyReLU: 3-8               [1, 32, 16, 16, 16]       --
├─Upsample: 1-3                          [1, 32, 32, 32, 32]       --
├─ModuleList: 1-6                        --                        (recursive)
│    └─ConvBlock: 2-5                    [1, 32, 32, 32, 32]       --
│    │    └─Conv3d: 3-9                  [1, 32, 32, 32, 32]       55,328
│    │    └─LeakyReLU: 3-10              [1, 32, 32, 32, 32]       --
├─Upsample: 1-5                          [1, 32, 64, 64, 64]       --
├─ModuleList: 1-6                        --                        (recursive)
│    └─ConvBlock: 2-6                    [1, 32, 64, 64, 64]       --
│    │    └─Conv3d: 3-11                 [1, 32, 64, 64, 64]       34,592
│    │    └─LeakyReLU: 3-12              [1, 32, 64, 64, 64]       --
├─Upsample: 1-7                          [1, 32, 128, 128, 128]    --
├─ModuleList: 1-8                        --                        --
│    └─ConvBlock: 2-7                    [1, 8, 128, 128, 128]     --
│    │    └─Conv3d: 3-13                 [1, 8, 128, 128, 128]     7,136
│    │    └─LeakyReLU: 3-14              [1, 8, 128, 128, 128]     --
│    └─ConvBlock: 2-8                    [1, 8, 128, 128, 128]     --
│    │    └─Conv3d: 3-15                 [1, 8, 128, 128, 128]     1,736
│    │    └─LeakyReLU: 3-16              [1, 8, 128, 128, 128]     --
│    └─ConvBlock: 2-9                    [1, 3, 128, 128, 128]     --
│    │    └─Conv3d: 3-17                 [1, 3, 128, 128, 128]     651
│    │    └─LeakyReLU: 3-18              [1, 3, 128, 128, 128]     --
├─Conv3d: 1-9                            [1, 1, 128, 128, 128]     82
==========================================================================================
Total params: 162,053
Trainable params: 162,053
Non-trainable params: 0
Total mult-adds (Units.GIGABYTES): 31.54
==========================================================================================
Input size (MB): 8.39
Forward/backward pass size (MB): 438.30
Params size (MB): 0.65
Estimated Total Size (MB): 447.34
==========================================================================================
"""