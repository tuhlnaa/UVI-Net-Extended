
import torch
import numpy as np
import nibabel as nib

from pathlib import Path
from typing import Tuple
from torch.utils.data import Dataset


class BaseMedicalDataset(Dataset):
    """Base class for medical image datasets with common functionality."""
    
    def __init__(
        self,
        data_path: str | Path,
        phase: str = "train",
        split: int = 90,
        image_size: Tuple[int, int, int] = (128, 128, 32)
    ) -> None:
        """
        Initialize the base medical dataset.

        Args:
            data_path: Path to the dataset
            phase: Dataset phase ('train' or 'test')
            split: Split index for train/test separation
            image_size: Target size for the images (height, width, depth)
        """
        self.data_path = Path(data_path)
        self.image_size = image_size
        self.paths = self._get_paths(phase, split)


    def _get_paths(self, phase: str, split: int) -> list[str]:
        """Get paths based on phase and split."""
        all_paths = sorted(self.data_path.iterdir())
        return all_paths[:split] if phase == "train" else all_paths[split:]


    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Normalize image to [0, 1] range."""
        min_val = image.min()
        max_val = image.max()
        return (image - min_val) / (max_val - min_val) if max_val > min_val else image


    def _pad_or_crop_3d(self, image: np.ndarray, target_size: Tuple[int, int, int]) -> np.ndarray:
        """
        Pad or crop 3D image to target size.

        Args:
            image: Input 3D image
            target_size: Desired output size

        Returns:
            Padded or cropped image
        """
        current_size = image.shape
        result = np.zeros(target_size)

        # Calculate crop/pad sizes
        starts = [(c - t) // 2 if c > t else 0 
                 for c, t in zip(current_size, target_size)]
        ends = [start + min(current, target) 
               for start, current, target in zip(starts, current_size, target_size)]
        target_starts = [0 if c > t else (t - c) // 2 
                        for c, t in zip(current_size, target_size)]

        # Slice original image and place in result
        slices_orig = tuple(slice(start, end) 
                          for start, end in zip(starts, ends))
        slices_target = tuple(slice(target_start, target_start + min(current, target))
                            for target_start, current, target 
                            in zip(target_starts, current_size, target_size))
        
        result[slices_target] = image[slices_orig]
        return result


    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        return len(self.paths)
    


class ACDCHeartDataset(BaseMedicalDataset):
    """ACDC Heart Dataset loader."""

    def __init__(
        self,
        data_path: str | Path,
        phase: str = "train",
        split: int = 90,
        image_size: Tuple[int, int, int] = (128, 128, 32)
    ) -> None:
        """Initialize ACDC Heart Dataset."""
        if phase == "train":
            data_path = Path(data_path) / "training"
        super().__init__(data_path, phase, split, image_size)


    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, int, int, torch.Tensor]:
        """
        Get a data sample.

        Returns:
            Tuple containing (ED_image, ES_image, ED_index, ES_index, video)
        """
        patient_path = self.paths[index]
        
        # Read frame indices
        with open(patient_path / "Info.cfg") as f:
            ed_frame = int(f.readline().strip().split()[-1])
            es_frame = int(f.readline().strip().split()[-1])

        # Load images
        ed_image = nib.load(patient_path / f"{patient_path.name}_frame{ed_frame:02d}.nii").get_fdata()
        es_image = nib.load(patient_path / f"{patient_path.name}_frame{es_frame:02d}.nii").get_fdata()
        video = nib.load(patient_path / f"{patient_path.name}_4d.nii").get_fdata()

        # Process images
        ed_image = self._pad_or_crop_3d(ed_image, self.image_size)[None, ...]
        es_image = self._pad_or_crop_3d(es_image, self.image_size)[None, ...]
        
        # Process video (handle temporal dimension)
        video_shape = (*self.image_size, video.shape[-1])
        processed_video = np.zeros((1, *video_shape))

        for t in range(video.shape[-1]):
            frame = self._pad_or_crop_3d(video[..., t], self.image_size)
            processed_video[0, ..., t] = self._normalize_image(frame)

        # Normalize images
        ed_image = self._normalize_image(ed_image)
        es_image = self._normalize_image(es_image)

        return (
            torch.from_numpy(ed_image).float(),
            torch.from_numpy(es_image).float(),
            ed_frame - 1,
            es_frame - 1,
            torch.from_numpy(processed_video).float()
        )


class LungDataset(BaseMedicalDataset):
    """Lung Dataset loader."""

    def __init__(
        self,
        data_path: str | Path,
        phase: str = "train",
        split: int = 68,
        image_size: Tuple[int, int, int] = (128, 128, 128)
    ) -> None:
        """Initialize Lung Dataset."""
        super().__init__(data_path, phase, split, image_size)


    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, int, int, torch.Tensor]:
        """
        Get a data sample.

        Returns:
            Tuple containing (ED_image, ES_image, ED_index, ES_index, video)
        """
        patient_path = self.paths[index]
        ed_frame = 0
        es_frame = 5

        # Load images
        ed_image = nib.load(patient_path / f"ct_{patient_path.name}_frame{ed_frame}.nii.gz").get_fdata()
        es_image = nib.load(patient_path / f"ct_{patient_path.name}_frame{es_frame}.nii.gz").get_fdata()

        # Create video tensor
        frames = [ed_image[None, ...]]
        for frame_idx in range(1, es_frame + 1):
            frame = nib.load(patient_path / f"ct_{patient_path.name}_frame{frame_idx}.nii.gz").get_fdata()
            frames.append(frame[None, ...])
        
        video = np.stack(frames, axis=-1)

        return (
            torch.from_numpy(ed_image[None, ...]).float(),
            torch.from_numpy(es_image[None, ...]).float(),
            ed_frame,
            es_frame,
            torch.from_numpy(video).float()
        )