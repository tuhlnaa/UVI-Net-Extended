# Copyright (c) 2025 Your Organization. All rights reserved.
"""
Lung CT Preprocessing Module

This module provides functionality for preprocessing 4D lung CT scans, including
DICOM reading, image resizing, bed removal, and various image processing operations.
"""

from pathlib import Path
from typing import Tuple, Optional, List

import cv2
import nibabel as nib
import numpy as np
import pydicom
import scipy.ndimage as ndimage
import skimage.morphology as morphology
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from tqdm import tqdm


class CTImageProcessor:
    """Class for processing CT images with various preprocessing operations."""
    
    def __init__(self, window_range: Tuple[float, float] = (-1400, 200)):
        """
        Initialize the CT image processor.
        
        Args:
            window_range: Tuple of (lower_bound, upper_bound) for lung window range
        """
        self.lower_bound, self.upper_bound = window_range
        
    @staticmethod
    def normalize_minmax(ct_scan: np.ndarray) -> np.ndarray:
        """Normalize CT scan using min-max scaling."""
        return (ct_scan - ct_scan.min()) / (ct_scan.max() - ct_scan.min())
    
    @staticmethod
    def create_mask(binary_image: np.ndarray) -> np.ndarray:
        """
        Create a mask from binary image using connected component analysis.
        
        Args:
            binary_image: Binary input image
            
        Returns:
            Processed mask array
        """
        labels, _ = ndimage.label(binary_image, structure=np.ones((3, 3, 3)))
        label_count = np.bincount(labels.ravel().astype(np.int32))
        label_count[0] = 0
        
        mask = labels == label_count.argmax()
        masks_list = []
        
        for slice_idx in range(mask.shape[-1]):
            if mask[..., slice_idx].sum() == 0:
                slice_mask = np.zeros_like(mask[..., slice_idx][..., np.newaxis])
            else:
                contours, _ = cv2.findContours(
                    mask[..., slice_idx].astype(np.uint8),
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_NONE
                )
                largest_contour = max(contours, key=cv2.contourArea)
                
                # Create binary mask from contour
                img = Image.new("L", mask[..., 0].shape, 0)
                contour_points = largest_contour[:, 0, :]
                polygon_points = list(zip(contour_points[:, 0], contour_points[:, 1]))
                ImageDraw.Draw(img).polygon(polygon_points, outline=0, fill=1)
                slice_mask = np.array(img)[..., np.newaxis]
            
            masks_list.append(slice_mask)
            
        return np.concatenate(masks_list, axis=-1)
    
    def remove_bed(self, ct_image: np.ndarray, threshold: float = -500) -> np.ndarray:
        """
        Remove bed from CT image using thresholding and morphological operations.
        
        Args:
            ct_image: Input CT image
            threshold: Threshold value for binary segmentation
            
        Returns:
            CT image with bed removed
        """
        # Create binary mask
        binary_mask = (ct_image > threshold).astype(float)
        
        # Apply mask creation multiple times for refinement
        mask = binary_mask
        for _ in range(3):
            mask = self.create_mask(mask)
            
        # Dilate mask
        mask = morphology.dilation(mask, np.ones((3, 3, 3)))
        mask = morphology.dilation(mask, np.ones((3, 3, 3)))
        mask = morphology.dilation(mask, np.ones((3, 3, 3)))
        
        # Apply mask to original image
        output = np.full(ct_image.shape, ct_image.min())
        output[mask == 1] = ct_image[mask == 1]
        
        return output
    
    @staticmethod
    def crop_center(img: np.ndarray, threshold: float = -500) -> np.ndarray:
        """
        Crop image to center based on content above threshold.
        
        Args:
            img: Input image
            threshold: Threshold value for determining content
            
        Returns:
            Center-cropped image
        """
        def find_content_bounds(arr: np.ndarray, axis: int) -> Tuple[int, int]:
            """Find the bounds of content along specified axis."""
            summed = np.any(arr > threshold, axis=tuple(i for i in range(arr.ndim) if i != axis))
            indices = np.where(summed)[0]
            return indices[0], indices[-1]
        
        # Find bounds in each dimension
        h_start, h_end = find_content_bounds(img, 0)
        w_start, w_end = find_content_bounds(img, 1)
        
        # Crop image
        cropped = img[h_start:h_end, w_start:w_end, :]
        
        # Calculate padding
        pad_h = (img.shape[0] - cropped.shape[0]) // 2
        pad_w = (img.shape[1] - cropped.shape[1]) // 2
        
        # Ensure non-negative padding
        pad_h = max(0, pad_h)
        pad_w = max(0, pad_w)
        
        # Calculate remaining padding
        pad_h_end = img.shape[0] - cropped.shape[0] - pad_h
        pad_w_end = img.shape[1] - cropped.shape[1] - pad_w
        
        # Ensure non-negative padding
        pad_h_end = max(0, pad_h_end)
        pad_w_end = max(0, pad_w_end)
        
        # Apply padding
        return np.pad(
            cropped,
            ((pad_h, pad_h_end), (pad_w, pad_w_end), (0, 0)),
            mode="minimum"
        )


class LungCTPreprocessor:
    """Class for preprocessing lung CT scans from DICOM files."""
    
    def __init__(
        self,
        output_dir: Path,
        image_processor: Optional[CTImageProcessor] = None,
        target_shape: Tuple[int, int, int] = (128, 128, 128)
    ):
        """
        Initialize the lung CT preprocessor.
        
        Args:
            output_dir: Directory for saving processed scans
            image_processor: CTImageProcessor instance for image processing
            target_shape: Target shape for resizing CT volumes
        """
        self.output_dir = Path(output_dir)
        self.image_processor = image_processor or CTImageProcessor()
        self.target_shape = target_shape
        
    @staticmethod
    def read_dicom_series(folder_path: Path) -> np.ndarray:
        """
        Read DICOM series from folder.
        
        Args:
            folder_path: Path to folder containing DICOM files
            
        Returns:
            3D numpy array of CT scan
        """
        dicom_files = sorted(
            [pydicom.dcmread(f) for f in folder_path.glob("*.dcm")],
            key=lambda x: int(x.InstanceNumber)
        )
        dicom_files.reverse()
        
        return np.stack(
            [f.pixel_array.astype(np.float32) - 1000 for f in dicom_files],
            axis=-1
        )
    
    @staticmethod
    def resize_volume(
        image_array: np.ndarray,
        target_shape: Tuple[int, int, int],
        rotate: bool = False
    ) -> np.ndarray:
        """
        Resize CT volume to target shape.
        
        Args:
            image_array: Input CT volume
            target_shape: Desired output shape
            rotate: Whether to rotate slices 90 degrees
            
        Returns:
            Resized CT volume
        """
        image_tensor = torch.from_numpy(image_array)[None, None, ...].float()
        image_tensor = F.interpolate(image_tensor, target_shape, mode="trilinear")
        image_array = image_tensor.squeeze().numpy()
        
        if rotate:
            image_array = np.stack(
                [np.rot90(image_array[..., i]) for i in range(image_array.shape[-1])],
                axis=-1
            )
            
        return image_array
    
    def process_scan(
        self,
        patient_id: str,
        scan_idx: int,
        phase_folders: List[Path]
    ) -> None:
        """
        Process and save CT scan phases.
        
        Args:
            patient_id: Patient identifier
            scan_idx: Scan index
            phase_folders: List of folders containing phase DICOM files
        """
        scan_save_dir = self.output_dir / f"{patient_id}_{scan_idx}"
        scan_save_dir.mkdir(parents=True, exist_ok=True)
        
        for phase_idx, folder_path in enumerate(phase_folders):
            # Read and process image
            ct_array = self.read_dicom_series(folder_path)
            
            # Initial resize with rotation
            ct_array = self.resize_volume(ct_array, (256, 256, 256), rotate=True)
            
            # Final resize
            ct_array = self.resize_volume(ct_array, self.target_shape, rotate=False)
            
            # Apply window range
            np.clip(
                ct_array,
                self.image_processor.lower_bound,
                self.image_processor.upper_bound,
                out=ct_array
            )
            
            # Process image
            ct_array = self.image_processor.remove_bed(ct_array)
            ct_array = self.image_processor.crop_center(ct_array)
            ct_array = self.image_processor.normalize_minmax(ct_array)
            
            # Save as NIfTI
            nifti_img = nib.Nifti1Image(ct_array, None)
            output_path = scan_save_dir / f"ct_{patient_id}_{scan_idx}_frame{phase_idx}.nii.gz"
            nib.save(nifti_img, output_path)


def main():
    """Main function to run the preprocessing pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess 4D lung CT scans")
    parser.add_argument(
        "--raw_folder_dir",
        type=Path,
        default=Path("dataset/4D-Lung/"),
        help="Directory containing raw DICOM data"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("dataset/4D-Lung-Preprocessed/"),
        help="Output directory for preprocessed scans"
    )
    args = parser.parse_args()
    
    # Initialize preprocessor
    preprocessor = LungCTPreprocessor(args.output_dir)
    
    # Process all patients
    patient_folders = sorted(args.raw_folder_dir.glob("*_*"))
    data_count = 0
    
    for pt_folder in tqdm(patient_folders, desc="Processing patients"):
        patient_id = pt_folder.name[:3]
        scan_dates = list(pt_folder.glob("*-*/"))
        
        # Filter for high-resolution scans
        high_res_scans = sorted(
            [scan for scan in scan_dates if next(scan.iterdir()).name[:8] == "1.000000"]
        )
        
        if high_res_scans:
            for scan_idx, scan_folder in enumerate(high_res_scans):
                # Find 4D CT series
                phase_folders = sorted(
                    [
                        scan
                        for scan in scan_folder.glob("*-*")
                        if len(list(scan.iterdir())) > 10
                    ]
                )
                
                preprocessor.process_scan(patient_id, scan_idx, phase_folders)
                data_count += 1
    
    print(f"Successfully processed {data_count} samples")


if __name__ == "__main__":
    main()