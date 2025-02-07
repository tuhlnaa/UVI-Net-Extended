import sys
from pathlib import Path
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).parents[1]
sys.path.append(str(PROJECT_ROOT))

from data.datasets import ACDCHeartDataset, LungDataset

def acdc_data(path):
    data_path = Path(path)
    batch_size = 1
    num_workers = 0

    # Initialize dataset
    dataset = ACDCHeartDataset(
        data_path=data_path,
        phase="val",  # "train" "val"
        split=90,
        image_size=(128, 128, 32)
    )
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        #shuffle=True,
        num_workers=num_workers,
        pin_memory=True, 
        #drop_last=True,
    )

    print(f"Dataset size: {len(dataset)} samples")
    print(f"Number of batches: {len(data_loader)}\n")

    # Iterate through a few batches
    for batch_idx, (ed_image, es_image, ed_frame, es_frame, video) in enumerate(data_loader):
        print(f"Batch {batch_idx + 1}:")
        print(f"  ED Image shape: {ed_image.shape}")
        print(f"  ES Image shape: {es_image.shape}")
        print(f"  ED frame indices: {ed_frame}")
        print(f"  ES frame indices: {es_frame}")
        print(f"  Video shape: {video.shape}")
        print(f"  Memory format: {ed_image.is_contiguous()}")
        print(f"  Device: {ed_image.device}")
        print(f"  Data type: {ed_image.dtype}\n")

        # Only show first 3 batches
        if batch_idx >= 2:
            break


def lung_data(path):
    data_path = Path(path)
    batch_size = 1
    num_workers = 0

    # Initialize dataset
    dataset = LungDataset(
        data_path=data_path,
        phase="train",  # "train" "val"
        split=68,
        image_size=(128, 128, 128)
    )
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        #shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        #drop_last=True,
    )

    print(f"Dataset size: {len(dataset)} samples")
    print(f"Number of batches: {len(data_loader)}\n")

    # Iterate through a few batches
    for batch_idx, (ed_image, es_image, ed_frame, es_frame, video) in enumerate(data_loader):
        print(f"Batch {batch_idx + 1}:")
        print(f"  ED Image shape: {ed_image.shape}")
        print(f"  ES Image shape: {es_image.shape}")
        print(f"  ED frame indices: {ed_frame}")
        print(f"  ES frame indices: {es_frame}")
        print(f"  Video shape: {video.shape}")
        print(f"  Memory format: {ed_image.is_contiguous()}")
        print(f"  Device: {ed_image.device}")
        print(f"  Data type: {ed_image.dtype}\n")

        # Only show first 3 batches
        if batch_idx >= 2:
            break


if __name__ == "__main__":
    acdc_data(r"D:\Kai\DATA_Set_2\ACDC_database")
    #lung_data(r"D:\Kai\DATA_Set_2\4D-Lung_Preprocessed")

"""
Dataset size: 90 samples
Number of batches: 90

Batch 1:
  ED Image shape: torch.Size([1, 1, 128, 128, 32])
  ES Image shape: torch.Size([1, 1, 128, 128, 32])
  ED frame indices: tensor([0])
  ES frame indices: tensor([11])
  Video shape: torch.Size([1, 1, 128, 128, 32, 30])
  Memory format: True
  Device: cpu
  Data type: torch.float32

Batch 2:
  ED Image shape: torch.Size([1, 1, 128, 128, 32])
  ES Image shape: torch.Size([1, 1, 128, 128, 32])
  ED frame indices: tensor([0])
  ES frame indices: tensor([9])
  Video shape: torch.Size([1, 1, 128, 128, 32, 13])
  Memory format: True
  Device: cpu
  Data type: torch.float32

Batch 3:
  ED Image shape: torch.Size([1, 1, 128, 128, 32])
  ES Image shape: torch.Size([1, 1, 128, 128, 32])
  ED frame indices: tensor([0])
  ES frame indices: tensor([8])
  Video shape: torch.Size([1, 1, 128, 128, 32, 25])
  Memory format: True
  Device: cpu
  Data type: torch.float32

===========================================================
Dataset size: 68 samples
Number of batches: 68

Batch 1:
  ED Image shape: torch.Size([1, 1, 128, 128, 128])
  ES Image shape: torch.Size([1, 1, 128, 128, 128])
  ED frame indices: tensor([0])
  ES frame indices: tensor([5])
  Video shape: torch.Size([1, 1, 128, 128, 128, 6])
  Memory format: True
  Device: cpu
  Data type: torch.float32

Batch 2:
  ED Image shape: torch.Size([1, 1, 128, 128, 128])
  ES Image shape: torch.Size([1, 1, 128, 128, 128])
  ED frame indices: tensor([0])
  ES frame indices: tensor([5])
  Video shape: torch.Size([1, 1, 128, 128, 128, 6])
  Memory format: True
  Device: cpu
  Data type: torch.float32

Batch 3:
  ED Image shape: torch.Size([1, 1, 128, 128, 128])
  ES Image shape: torch.Size([1, 1, 128, 128, 128])
  ED frame indices: tensor([0])
  ES frame indices: tensor([5])
  Video shape: torch.Size([1, 1, 128, 128, 128, 6])
  Memory format: True
  Device: cpu
  Data type: torch.float32
"""