import os
os.environ["VXM_BACKEND"] = "pytorch"
import torch
import voxelmorph as vxm
from torchinfo import summary

# configure unet features
nb_features = [
    [16, 32, 32, 32],             # encoder features
    [32, 32, 32, 32, 32, 16, 16]  # decoder features
]
inshape = [128, 128, 128]

# build model using VxmDense with modified settings
flow_model = vxm.networks.VxmDense(
    inshape=inshape,
    nb_unet_features=nb_features,
    src_feats=1,    # number of source image features
    trg_feats=1,    # number of target image features
    bidir=True,     # enable bidirectional registration
    int_steps=0,    # disable integration steps to get raw flow fields
    int_downsize=1  # prevent downsampling of flow field
)#.to("cuda")

# Test with random input
x = torch.randn(1, 2, 128, 128, 128)#.to("cuda")
source, target = x[:, 0:1], x[:, 1:2]

# During training mode
flow_model.train()
y_source, y_target, flow_field = flow_model(source, target)

# The flow field will be full size and include both directions
flow_0_1 = flow_field[:, :3]  # first 3 channels
flow_1_0 = -flow_field[:, :3]  # same magnitude, opposite direction

print(y_source.shape)   # torch.Size([1, 1, 128, 128, 128]) Forward transformed image
print(y_target.shape)   # torch.Size([1, 1, 128, 128, 128]) Backward transformed image
print(flow_0_1.shape)   # torch.Size([1, 3, 128, 128, 128]) Forward flow field
print(flow_1_0.shape)   # torch.Size([1, 3, 128, 128, 128]) Backward flow field

print(flow_model)
summary(flow_model, input_data=(source, target))

"""
VxmDense(
  (unet_model): Unet(
    (encoder): ModuleList(
      (0): ModuleList(
        (0): ConvBlock(
          (main): Conv3d(2, 16, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
          (activation): LeakyReLU(negative_slope=0.2)
        )
      )
      (1): ModuleList(
        (0): ConvBlock(
          (main): Conv3d(16, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
          (activation): LeakyReLU(negative_slope=0.2)
        )
      )
      (2-3): 2 x ModuleList(
        (0): ConvBlock(
          (main): Conv3d(32, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
          (activation): LeakyReLU(negative_slope=0.2)
        )
      )
    )
    (decoder): ModuleList(
      (0): ModuleList(
        (0): ConvBlock(
          (main): Conv3d(32, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
          (activation): LeakyReLU(negative_slope=0.2)
        )
      )
      (1-3): 3 x ModuleList(
        (0): ConvBlock(
          (main): Conv3d(64, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
          (activation): LeakyReLU(negative_slope=0.2)
        )
      )
    )
    (remaining): ModuleList(
      (0): ConvBlock(
        (main): Conv3d(48, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        (activation): LeakyReLU(negative_slope=0.2)
      )
      (1): ConvBlock(
        (main): Conv3d(32, 16, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        (activation): LeakyReLU(negative_slope=0.2)
      )
      (2): ConvBlock(
        (main): Conv3d(16, 16, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        (activation): LeakyReLU(negative_slope=0.2)
      )
    )
  )
  (flow): Conv3d(16, 3, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
  (transformer): SpatialTransformer()

===============================================================================================
Layer (type:depth-idx)                        Output Shape              Param #
===============================================================================================
VxmDense                                      [1, 1, 128, 128, 128]     --
├─Unet: 1-1                                   [1, 16, 128, 128, 128]    --
│    └─ModuleList: 2-1                        --                        --
│    │    └─ModuleList: 3-1                   --                        880
│    │    └─ModuleList: 3-2                   --                        13,856
│    │    └─ModuleList: 3-3                   --                        27,680
│    │    └─ModuleList: 3-4                   --                        27,680
│    └─ModuleList: 2-2                        --                        --
│    │    └─ModuleList: 3-5                   --                        27,680
│    │    └─ModuleList: 3-6                   --                        55,328
│    │    └─ModuleList: 3-7                   --                        55,328
│    │    └─ModuleList: 3-8                   --                        55,328
│    └─ModuleList: 2-3                        --                        --
│    │    └─ConvBlock: 3-9                    [1, 32, 128, 128, 128]    41,504
│    │    └─ConvBlock: 3-10                   [1, 16, 128, 128, 128]    13,840
│    │    └─ConvBlock: 3-11                   [1, 16, 128, 128, 128]    6,928
├─Conv3d: 1-2                                 [1, 3, 128, 128, 128]     1,299
├─SpatialTransformer: 1-3                     [1, 1, 128, 128, 128]     --
├─SpatialTransformer: 1-4                     [1, 1, 128, 128, 128]     --
===============================================================================================
Total params: 327,331
Trainable params: 327,331
Non-trainable params: 0
Total mult-adds (Units.GIGABYTES): 156.37
===============================================================================================
Input size (MB): 33.55
Forward/backward pass size (MB): 1545.73
Params size (MB): 1.31
Estimated Total Size (MB): 1580.60
===============================================================================================
"""