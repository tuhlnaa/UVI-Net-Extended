import sys
sys.path.append(r"E:\Kai_2\CODE_Repository\UVI-Net-Extended")

import torch
from torchinfo import summary
from models.feature_extract.model import FeatureExtract

# configure unet features
nb_features = [
    [8, 16, 32],             # encoder features
    [32, 32, 32, 8, 8, 3]    # decoder features
]
inshape = [128, 128, 128]
additional_dims = [4, 8, 16]

feature_model = FeatureExtract()

# Test with random input
batch_size = 1
x = torch.randn(batch_size, 1, *inshape)
output = feature_model(x)

summary(feature_model, input_data=(x))
print(feature_model)
print(len(output), output[0].shape, output[1].shape, output[2].shape)
# 3, torch.Size([1, 4, 128, 128, 128]), torch.Size([1, 8, 64, 64, 64]), torch.Size([1, 16, 32, 32, 32])

"""
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
FeatureExtract                           [1, 4, 128, 128, 128]     --
├─Sequential: 1-1                        [1, 4, 128, 128, 128]     --
│    └─Conv3d: 2-1                       [1, 4, 128, 128, 128]     112
│    └─ReLU: 2-2                         [1, 4, 128, 128, 128]     --
│    └─Conv3d: 2-3                       [1, 4, 128, 128, 128]     436
│    └─ReLU: 2-4                         [1, 4, 128, 128, 128]     --
├─Sequential: 1-2                        [1, 8, 64, 64, 64]        --
│    └─Conv3d: 2-5                       [1, 8, 64, 64, 64]        872
│    └─ReLU: 2-6                         [1, 8, 64, 64, 64]        --
│    └─Conv3d: 2-7                       [1, 8, 64, 64, 64]        1,736
│    └─ReLU: 2-8                         [1, 8, 64, 64, 64]        --
├─Sequential: 1-3                        [1, 16, 32, 32, 32]       --
│    └─Conv3d: 2-9                       [1, 16, 32, 32, 32]       3,472
│    └─ReLU: 2-10                        [1, 16, 32, 32, 32]       --
│    └─Conv3d: 2-11                      [1, 16, 32, 32, 32]       6,928
│    └─ReLU: 2-12                        [1, 16, 32, 32, 32]       --
==========================================================================================
Total params: 13,556
Trainable params: 13,556
Non-trainable params: 0
Total mult-adds (Units.GIGABYTES): 2.17
==========================================================================================
Input size (MB): 8.39
Forward/backward pass size (MB): 176.16
Params size (MB): 0.05
Estimated Total Size (MB): 184.60
==========================================================================================

FeatureExtract(
  (layer1): Sequential(
    (0): Conv3d(1, 4, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
    (1): ReLU()
    (2): Conv3d(4, 4, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
    (3): ReLU()
  )
  (layer2): Sequential(
    (0): Conv3d(4, 8, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
    (1): ReLU()
    (2): Conv3d(8, 8, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
    (3): ReLU()
  )
  (layer3): Sequential(
    (0): Conv3d(8, 16, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
    (1): ReLU()
    (2): Conv3d(16, 16, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
    (3): ReLU()
  )
)
"""