"""
Data-Efficient Unsupervised Interpolation for 4D Medical Images.
This module implements frame interpolation for medical image sequences.
"""

import os
import torch
import argparse
import random
import numpy as np
import torch.nn.functional as F

from torch.utils.data import DataLoader
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, List, Optional

import losses
from lpips import LPIPS
from utils import datasets, utils
from models.UNet.modelV4 import UNet3D, UNet3DMulti
from models.VoxelMorph.model import VoxelMorph
from models.feature_extract.model import FeatureExtract

@dataclass
class ImageMetrics:
    """Container for image quality metrics."""
    psnr: float
    ncc: float
    ssim: float
    nmse: float
    lpips: float

class ImageInterpolator:
    """Handles medical image interpolation using deep learning models."""
    
    def __init__(self, img_size: Tuple[int, int, int], feature_extract: bool = True):
        self.img_size = img_size
        self.feature_extract = feature_extract
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize models
        self.flow_model = VoxelMorph(img_size).to(self.device)
        self.refinement_model = (UNet3DMulti(img_size) if feature_extract else UNet3D(img_size)).to(self.device)
        self.feature_model = FeatureExtract().to(self.device) if feature_extract else None
        
        # Initialize registration models
        self.reg_model = utils.register_model(img_size, "nearest").to(self.device)
        self.reg_model_bilin = utils.register_model(img_size, "bilinear").to(self.device)
        
        # Initialize LPIPS
        self.lpips_fn = LPIPS(net="alex").to(self.device)


    def load_models(self, checkpoint_path: Path) -> None:
        """Load pretrained model weights."""
        checkpoint = torch.load(checkpoint_path)
        self.flow_model.load_state_dict(checkpoint["flow_model_state_dict"])
        self.refinement_model.load_state_dict(checkpoint["model_state_dict"])
        if self.feature_extract:
            self.feature_model.load_state_dict(checkpoint["feature_model_state_dict"])


    def evaluate_quality(self, original: torch.Tensor, generated: torch.Tensor) -> ImageMetrics:
        """Calculate image quality metrics between original and generated images."""
        max_pixel = 1.0
        mse = torch.mean((original - generated) ** 2)
        
        # Calculate basic metrics
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse.item()))
        ncc = losses.NCC()(original, generated)
        ssim = 1 - losses.SSIM3D()(original, generated)
        nmse = mse / torch.mean(original**2)
        
        # Calculate LPIPS
        lpips_meter = utils.AverageMeter()
        
        # Process slices along different dimensions
        for dim in [2, 3, 4]:  # Process XY, XZ, YZ planes
            slices_orig = self._prepare_slices(original, dim)
            slices_gen = self._prepare_slices(generated, dim)
            
            # Calculate LPIPS for each slice
            lpips_values = self.lpips_fn(slices_orig, slices_gen)
            for val in lpips_values:
                lpips_meter.update(val.item())
        
        return ImageMetrics(
            psnr=psnr,
            ncc=-1 * ncc.item(),
            ssim=ssim.item(),
            nmse=nmse.item(),
            lpips=lpips_meter.avg
        )


    def _prepare_slices(self, volume: torch.Tensor, dim: int) -> torch.Tensor:
        """Prepare 2D slices from 3D volume for LPIPS calculation."""
        slices = []
        for idx in range(volume.shape[dim]):
            # Select and normalize slice
            if dim == 2:
                slice_2d = volume[0, 0, idx, :, :]
            elif dim == 3:
                slice_2d = volume[0, 0, :, idx, :]
            else:  # dim == 4
                slice_2d = volume[0, 0, :, :, idx]
            
            # Convert to RGB and normalize to [-1, 1]
            slice_rgb = (slice_2d * 2 - 1).repeat(3, 1, 1)
            slices.append(slice_rgb)
        
        return torch.stack(slices, dim=0).to(self.device)


    def interpolate_frame(self, image0: torch.Tensor, image1: torch.Tensor, 
                         alpha: float) -> torch.Tensor:
        """Interpolate between two frames with given alpha value."""
        # Calculate flows
        combined = torch.cat((image0, image1), dim=1)
        _, _, flow_0_1, flow_1_0 = self.flow_model(combined)
        
        # Calculate intermediate flows
        flow_0_t = flow_0_1 * alpha
        flow_1_t = flow_1_0 * (1 - alpha)
        
        # Warp images
        image_0_t = self.reg_model_bilin([image0, flow_0_t.float()])
        image_1_t = self.reg_model_bilin([image1, flow_1_t.float()])
        
        # Combine warped images
        image_t_combined = (1 - alpha) * image_0_t + alpha * image_1_t
        
        if not self.feature_extract:
            return image_t_combined + self.refinement_model(image_t_combined)
        
        # Feature extraction and warping
        features0 = self.feature_model(image0)
        features1 = self.feature_model(image1)
        warped_features0, warped_features1 = [], []
        
        for idx, (feat0, feat1) in enumerate(zip(features0, features1)):
            scale_factor = 0.5 ** idx
            size = tuple(x // (2**idx) for x in self.img_size)
            reg_model_feat = utils.register_model(size).to(self.device)
            
            # Scale flows and warp features
            scaled_flow0 = F.interpolate(flow_0_t * scale_factor, scale_factor=scale_factor)
            scaled_flow1 = F.interpolate(flow_1_t * scale_factor, scale_factor=scale_factor)
            
            warped_features0.append(reg_model_feat([feat0, scaled_flow0.float()]))
            warped_features1.append(reg_model_feat([feat1, scaled_flow1.float()]))
        
        # Refinement with features
        diff = self.refinement_model(image_t_combined, warped_features0, warped_features1)
        return image_t_combined + diff


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def create_data_loader(dataset_name: str, split: Optional[int] = None) -> Tuple[DataLoader, Tuple[int, int, int]]:
    """Create data loader and return image size for the specified dataset."""
    if dataset_name == "cardiac":
        img_size = (128, 128, 32)
        split = 90 if split is None else split
        data_dir = Path("dataset/ACDC_database/training")
        dataset = datasets.ACDCHeartDataset(data_dir, phase="test", split=split)
    elif dataset_name == "lung":
        img_size = (128, 128, 128)
        split = 68 if split is None else split
        data_dir = Path("dataset/4D-Lung_Preprocessed")
        dataset = datasets.LungDataset(data_dir, phase="test", split=split)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, 
                       pin_memory=False, drop_last=True)
    return loader, img_size


def main(args):
    """Main evaluation function."""
    set_seed(args.seed)
    
    # Setup
    save_dir = Path(f"outputs/{args.dataset}")
    data_loader, img_size = create_data_loader(args.dataset, args.split)
    
    # Initialize interpolator
    interpolator = ImageInterpolator(img_size, args.feature_extract)
    
    # Load best model
    model_path = sorted(save_dir.glob("*"))[args.model_idx]
    interpolator.load_models(model_path)
    print(f"Model: {model_path.name} loaded!")
    
    # Evaluation metrics
    metrics = {name: utils.AverageMeter() for name in 
              ['psnr', 'nmse', 'ncc', 'lpips', 'ssim']}
    
    # Evaluation loop
    for data in data_loader:
        data = [t.cuda() for t in data]
        image0, image1 = data[0], data[1]
        frame0_idx, frame1_idx = data[2], data[3]
        video = data[4]
        
        # Interpolate intermediate frames
        for frame_idx in range(frame0_idx + 1, frame1_idx):
            alpha = (frame_idx - frame0_idx) / (frame1_idx - frame0_idx)
            
            # Generate intermediate frame
            generated = interpolator.interpolate_frame(image0, image1, alpha)
            ground_truth = video[..., frame_idx]
            
            # Calculate metrics
            frame_metrics = interpolator.evaluate_quality(ground_truth, generated)
            
            # Update metrics
            metrics['psnr'].update(frame_metrics.psnr)
            metrics['ncc'].update(frame_metrics.ncc)
            metrics['ssim'].update(frame_metrics.ssim)
            metrics['nmse'].update(frame_metrics.nmse)
            metrics['lpips'].update(frame_metrics.lpips)
    
    # Print results
    print("\nResults:")
    print("AVG\tPSNR: {:.2f}, NCC: {:.3f}, SSIM: {:.3f}, NMSE: {:.3f}, LPIPS: {:.3f}".format(
        metrics['psnr'].avg, metrics['ncc'].avg, metrics['ssim'].avg,
        metrics['nmse'].avg * 100, metrics['lpips'].avg * 100))
    
    print("STDERR\tPSNR: {:.3f}, NCC: {:.3f}, SSIM: {:.3f}, NMSE: {:.3f}, LPIPS: {:.3f}".format(
        metrics['psnr'].stderr, metrics['ncc'].stderr, metrics['ssim'].stderr,
        metrics['nmse'].stderr * 100, metrics['lpips'].stderr * 100))


if __name__ == "__main__":
    
    
    parser = argparse.ArgumentParser(description='Medical image interpolation evaluation')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--gpu', type=str, help='GPU device ID')
    parser.add_argument('--dataset', type=str, default='cardiac',
                       choices=['cardiac', 'lung'], help='Dataset name')
    parser.add_argument('--model_idx', type=int, default=-1,
                       help='Model checkpoint index')
    parser.add_argument('--split', type=int, help='Dataset split')
    parser.add_argument('--feature_extract', action='store_true', default=True,
                       help='Use feature extraction')
    
    args = parser.parse_args()
    
    # GPU setup
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
    
    main(args)