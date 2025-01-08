"""
Lung CT scan preprocessing module.
Provides tools for processing and preparing CT scan data for deep learning models.
"""

import cv2
import torch
import pydicom

import numpy as np
import nibabel as nib
import torch.nn.functional as F
import scipy.ndimage as ndimage
import skimage.morphology as morphology

from tqdm import tqdm
from pathlib import Path
from functools import partial
from PIL import Image, ImageDraw
from dataclasses import dataclass
from typing import Optional, Tuple, List, Union
from multiprocessing import Pool, cpu_count


@dataclass
class PreprocessingConfig:
    """Configuration for CT scan preprocessing parameters."""
    window_range: Tuple[int, int] = (-1400, 200)  # lung window range
    bed_threshold: float = -500
    initial_shape: Tuple[int, int, int] = (256, 256, 256)
    final_shape: Tuple[int, int, int] = (128, 128, 128)
    dilation_iterations: int = 3
    dilation_kernel: Tuple[int, int, int] = (3, 3, 3)


class CTImageProcessor:
    """Handles CT image processing operations."""
    
    def __init__(self, config: PreprocessingConfig):
        self.config = config
    

    @staticmethod
    def normalize(ct_scan: np.ndarray) -> np.ndarray:
        """Normalize CT scan values to [0, 1] range using min-max scaling."""
        min_val, max_val = ct_scan.min(), ct_scan.max()
        return (ct_scan - min_val) / (max_val - min_val)
    

    def _create_mask(self, image: np.ndarray) -> np.ndarray:
        """Create a binary mask for the CT image."""
        labels, _ = ndimage.label(image, structure=np.ones((3, 3, 3)))
        label_count = np.bincount(labels.ravel().astype(np.int32))
        label_count[0] = 0  # Ignore background

        mask = labels == label_count.argmax()
        mask_new = np.zeros_like(image)

        # Process each slice
        for i in range(mask.shape[-1]):
            if mask[..., i].sum() == 0:
                continue

            # Convert slice to uint8 for OpenCV
            slice_mask = mask[..., i].astype(np.uint8)

            contours, _ = cv2.findContours(slice_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            
            if not contours:
                continue
                
            largest_contour = max(contours, key=cv2.contourArea)
            points = largest_contour[:, 0, :]
            
            img = Image.new("L", mask[..., 0].shape, 0)
            ImageDraw.Draw(img).polygon(
                list(zip(points[:, 0], points[:, 1])),
                outline=0,
                fill=1
            )
            mask_new[..., i] = np.array(img)
            
        return mask_new

    # def _create_mask(self, image: np.ndarray) -> np.ndarray:
    #     """Create a binary mask using only scipy."""
    #     labels, _ = ndimage.label(image, structure=np.ones((3, 3, 3)))
    #     label_count = np.bincount(labels.ravel().astype(np.int32))
    #     label_count[0] = 0
        
    #     mask = labels == label_count.argmax()
        
    #     # Fill holes in each slice
    #     for i in range(mask.shape[-1]):
    #         if mask[..., i].sum() > 0:
    #             mask[..., i] = ndimage.binary_fill_holes(mask[..., i])
        
    #     return mask


    def remove_bed(self, ct_image: np.ndarray) -> np.ndarray:
        """Remove bed artifact from CT scan using thresholding and morphological operations."""
        threshold_mask = ct_image > self.config.bed_threshold
        
        # Iteratively refine mask
        refined_mask = threshold_mask
        for _ in range(3):
            refined_mask = self._create_mask(refined_mask)
            
        # Apply morphological dilation
        kernel = np.ones(self.config.dilation_kernel)
        for _ in range(self.config.dilation_iterations):
            refined_mask = morphology.dilation(refined_mask, kernel)
            
        # Apply mask to original image
        result = np.full_like(ct_image, ct_image.min())
        result[refined_mask == 1] = ct_image[refined_mask == 1]
        return result


    def center_crop(self, img: np.ndarray) -> np.ndarray:
        """
        Center crop the CT image based on content above threshold.
        
        Args:
            img: Input CT image array of shape (H, W, D)
            
        Returns:
            Cropped and padded image of the same shape as input
        """
        def find_content_bounds(array: np.ndarray, axis: int) -> Tuple[int, int]:
            """Find the content boundaries along specified axis."""
            proj = np.any(array > self.config.bed_threshold, axis=axis)
            coords = np.where(proj)[0]
            return coords[0], coords[-1] if len(coords) > 0 else (0, array.shape[axis])

        # Find content boundaries along height and width
        h_start, h_end = find_content_bounds(img, axis=(1, 2))
        w_start, w_end = find_content_bounds(img, axis=(0, 2))
        
        # Crop the image
        cropped = img[h_start:h_end, w_start:w_end, :]
        
        # Calculate padding
        h_pad = [(img.shape[0] - cropped.shape[0]) // 2, 0]
        h_pad[1] = img.shape[0] - cropped.shape[0] - h_pad[0]
        
        w_pad = [(img.shape[1] - cropped.shape[1]) // 2, 0]
        w_pad[1] = img.shape[1] - cropped.shape[1] - w_pad[0]
        
        # Ensure non-negative padding
        pad_dims = [(max(0, p1), max(0, p2)) for p1, p2 in [h_pad, w_pad]]
        pad_dims.append((0, 0))  # No padding for depth dimension
        
        return np.pad(cropped, pad_dims, mode="minimum")
    

class DCMDataset:
    """Handles DICOM dataset operations."""
    
    def __init__(self, config: PreprocessingConfig):
        self.config = config
    

    @staticmethod
    def read_dcm_files(folder_path: Path) -> np.ndarray:
        """Read and stack DICOM files from a directory."""
        files = sorted(
            [pydicom.dcmread(f) for f in folder_path.glob("*.dcm")],
            key=lambda x: int(x.InstanceNumber),
            reverse=True
        )
        return np.stack([f.pixel_array.astype(np.float32) - 1000 for f in files], axis=-1)
    

    @staticmethod
    def resize_volume(image_array: np.ndarray,new_shape: Tuple[int, int, int],rotate: bool = False) -> np.ndarray:
        """Resize volume to target shape using trilinear interpolation."""
        with torch.no_grad():
            image_tensor = torch.from_numpy(image_array)[None, None, ...].float()
            resized = F.interpolate(image_tensor, new_shape, mode="trilinear")
            result = resized.squeeze().numpy()
            
            if rotate:
                result = np.stack([np.rot90(result[..., i]) for i in range(result.shape[-1])], axis=-1)
                
            return result


class LungPreprocessor:
    """Main class for lung CT preprocessing pipeline with multiprocessing support."""
    
    def __init__(
        self,
        config: Optional[PreprocessingConfig] = None,
        output_dir: Optional[Union[str, Path]] = None,
        n_processes: Optional[int] = None
    ):
        self.config = config or PreprocessingConfig()
        self.output_dir = Path(output_dir) if output_dir else Path("preprocessed_data")
        self.image_processor = CTImageProcessor(self.config)
        self.dcm_dataset = DCMDataset(self.config)
        self.n_processes = n_processes or max(1, cpu_count() - 1)  # Leave one CPU free


    def process_scan(self, folder_path: Path) -> np.ndarray:
        """Process a single CT scan through the complete pipeline."""
        image_array = self.dcm_dataset.read_dcm_files(folder_path)
        
        # Two-step resizing with rotation
        image_array = self.dcm_dataset.resize_volume(
            image_array, self.config.initial_shape, rotate=True
        )
        image_array = self.dcm_dataset.resize_volume(
            image_array, self.config.final_shape, rotate=False
        )
        
        # Apply window range
        lower, upper = self.config.window_range
        image_array = np.clip(image_array, lower, upper)
        
        # Process image
        image_array = self.image_processor.remove_bed(image_array)
        image_array = self.image_processor.center_crop(image_array)
        image_array = self.image_processor.normalize(image_array)
        
        return image_array


    def save_nifti(self, image_array: np.ndarray, save_path: Path) -> None:
        """Save processed image as NIfTI file."""
        nib.save(nib.Nifti1Image(image_array, None), save_path)


    def _process_single_scan(self, scan_info: Tuple[Path, Path]) -> bool:
        """Process a single scan with error handling."""
        folder_path, save_path = scan_info
        try:
            processed_array = self.process_scan(folder_path)
            self.save_nifti(processed_array, save_path)
            return True
        except Exception as e:
            print(f"Error processing {folder_path}: {str(e)}")
            return False
        
        
    def process_dataset(self, raw_folder: Path) -> int:
        """Process entire dataset of CT scans using multiprocessing."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Collect all processing tasks
        processing_tasks = []
        pt_folders = sorted(raw_folder.glob("*_*"))
        
        for pt_folder in pt_folders:
            pt_num = pt_folder.name[:3]
            high_res_scans = sorted(
                [scan for scan in pt_folder.glob("*-*/")
                 if next(scan.iterdir()).name[:8] == "1.000000"]
            )
            
            for scan_idx, scan_folder in enumerate(high_res_scans):
                ct_scan_dirs = sorted(
                    [scan for scan in scan_folder.glob("*-*")
                     if len(list(scan.iterdir())) > 10]
                )
                
                scan_save_dir = self.output_dir / f"{pt_num}_{scan_idx}"
                scan_save_dir.mkdir(parents=True, exist_ok=True)
                
                for phase_idx, folder_path in enumerate(ct_scan_dirs):
                    save_path = scan_save_dir / f"ct_{pt_num}_{scan_idx}_frame{phase_idx}.nii.gz"
                    processing_tasks.append((folder_path, save_path))

        # Process tasks in parallel
        with Pool(processes=self.n_processes) as pool:
            results = list(tqdm(
                pool.imap(self._process_single_scan, processing_tasks),
                total=len(processing_tasks),
                desc=f"Processing scans using {self.n_processes} processes"
            ))

        successful_count = sum(1 for result in results if result)
        return successful_count

def main():
    """Main entry point for preprocessing pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess 4D Lung CT scans")
    parser.add_argument("--raw_folder_dir", type=Path, default=Path("dataset/4D-Lung/"), 
                        help="Raw data folder directory")
    parser.add_argument("--output_dir", type=Path, default=Path("dataset/4D-Lung_Preprocessed/"), 
                        help="Output directory for preprocessed data")
    parser.add_argument("--n_processes", type=int, default=None,
                        help="Number of processes to use (default: number of CPU cores - 1)")
    args = parser.parse_args()
    
    preprocessor = LungPreprocessor(
        output_dir=args.output_dir,
        n_processes=args.n_processes
    )
    total_processed = preprocessor.process_dataset(args.raw_folder_dir)
    print(f"Successfully processed {total_processed} samples")


if __name__ == "__main__":
    main()