"""
Data-Efficient Unsupervised Interpolation for 4D Medical Images.
This module implements frame interpolation for medical image sequences.
"""
import torch
import random
import argparse
import numpy as np
import torch.nn.functional as F

from rich.table import Table
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn, TimeRemainingColumn
from pathlib import Path
from dataclasses import dataclass
from torch.utils.data import DataLoader
from typing import Tuple, List, Optional
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics import StructuralSimilarityIndexMeasure

import losses
from utils import utils
from data.datasets import ACDCHeartDataset, LungDataset
from models.VoxelMorph.model import VoxelMorph
from models.UNet.modelV4 import UNet3D, UNet3DMulti
from models.feature_extract.model import FeatureExtract

# Initialize rich console
console = Console()

@dataclass
class ImageMetrics:
    """Container for image quality metrics."""
    psnr: float
    ncc: float
    ssim: float
    nmse: float
    lpips: float


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


class ImageInterpolator:
    """Handles medical image interpolation using deep learning models."""
    
    def __init__(self, img_size: Tuple[int, int, int], device: torch.device, feature_extract: bool = True):
        self.img_size = img_size
        self.feature_extract = feature_extract
        self.device = device
        
        # Initialize models
        self.flow_model = VoxelMorph(img_size).to(self.device)
        self.refinement_model = (UNet3DMulti(img_size) if feature_extract else UNet3D(img_size)).to(self.device)
        self.feature_model = FeatureExtract().to(self.device) if feature_extract else None
        
        # Initialize registration models
        self.reg_model = utils.register_model(img_size, "nearest").to(self.device)
        self.reg_model_bilin = utils.register_model(img_size, "bilinear").to(self.device)
        
        # Initialize LPIPS using torchmetrics
        self.lpips_fn = LearnedPerceptualImagePatchSimilarity(
            net_type="alex",
            reduction="mean",
            normalize=False  # Expects input in [-1, 1] range
        ).to(self.device)


    def load_models(self, checkpoint_path: Path) -> None:
        """Load pretrained model weights."""
        checkpoint = torch.load(checkpoint_path)
        self.flow_model.load_state_dict(checkpoint["flow_model_state_dict"])
        self.refinement_model.load_state_dict(checkpoint["model_state_dict"])
        if self.feature_extract:
            self.feature_model.load_state_dict(checkpoint["feature_model_state_dict"])


    def evaluate_quality(self, original: torch.Tensor, generated: torch.Tensor,  device: torch.device) -> ImageMetrics:
        """Calculate image quality metrics between original and generated images."""
        max_pixel = 1.0
        original = torch.clamp(original, 0, 1)
        generated = torch.clamp(generated, 0, 1)

        mse = torch.mean((original - generated) ** 2)

        # Calculate basic metrics
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse.item()))
        ncc = losses.NCCLoss()(original, generated)
        ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)(original, generated)
        nmse = mse / torch.mean(original**2)
        
        # Calculate LPIPS
        lpips_meter = utils.AverageMeter()
        
        # Process slices along different dimensions
        for dim in [2, 3, 4]:  # Process XY, XZ, YZ planes
            slices_orig = self._prepare_slices(original, dim)
            slices_gen = self._prepare_slices(generated, dim)

            # Calculate LPIPS for each slice pair
            lpips_value = self.lpips_fn(slices_orig, slices_gen)
            lpips_meter.update(lpips_value.item())
        
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


def create_data_loader(base_path: str, dataset_name: str, split: Optional[int] = None) -> Tuple[DataLoader, Tuple[int, int, int]]:
    """Create data loader and return image size for the specified dataset."""
    if args.dataset == "cardiac":
        image_size=(128, 128, 32)
        data_path = r"./dataset/ACDC_database"
        val_dataset = ACDCHeartDataset(
            data_path=base_path,
            phase="val",
            split=90 if split is None else split,
            image_size=image_size
        )
    elif args.dataset == "lung":
        image_size=(128, 128, 128)
        data_path = r"./dataset/4D-Lung_Preprocessed"
        val_dataset = LungDataset(
            data_path=base_path,
            phase="val",
            split=68 if split is None else split,
            image_size=image_size
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    loader = DataLoader(
        val_dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=False, 
        drop_last=True
    )
    return loader, image_size


@utils.timer_decorator
def main(args):
    set_seed(args.seed)
    
    # Device setup
    device = torch.device(args.device)
    if device.type == 'cuda':
        console.print(f"[bold green]GPU:[/] {torch.cuda.get_device_name(device.index)}")
        console.print(f"[bold green]Available GPUs:[/] {torch.cuda.device_count()}")
    
    # Create data loader
    with console.status("[bold yellow]Creating data loader..."):
        data_loader, img_size = create_data_loader(args.data_dir, args.dataset, args.split)
        console.print(f"[bold green]Dataset:[/] {args.dataset}")
        console.print(f"[bold green]Image size:[/] {img_size}")
    
    # Initialize interpolator
    interpolator = ImageInterpolator(img_size, device, args.feature_extract)
    
    # Load checkpoint
    checkpoint_path = Path(args.resume)
    if not checkpoint_path.exists():
        console.print(f"[bold red]Error:[/] Checkpoint not found at {checkpoint_path}")
        return
    interpolator.load_models(checkpoint_path)

    # Initialize metrics
    metrics = {name: utils.AverageMeter() for name in ['psnr', 'nmse', 'ncc', 'lpips', 'ssim']}
    
    # Evaluation progress bar
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        eval_task = progress.add_task("[cyan]Evaluating...", total=len(data_loader))
        
        # Evaluation loop
        for data in data_loader:
            data = [t.to(device) for t in data]
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
                frame_metrics = interpolator.evaluate_quality(ground_truth, generated, device)
                
                # Update metrics
                for name, value in vars(frame_metrics).items():
                    metrics[name].update(value)

            progress.advance(eval_task)

    # Create results table (NMSE and LPIPS should be multiplied by 100)
    table = Table(title="Evaluation Results", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Average", justify="right", style="green")
    table.add_column("Std Error", justify="right", style="yellow")

    for metric in metrics:
        table.add_row(metric.upper(), f"{metrics[metric].avg:.4f}", f"±{metrics[metric].stderr:.4f}")

    console.print(table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Medical image interpolation evaluation')
    parser.add_argument('--resume', type=str, required=True, help='Resume full model and optimizer state from checkpoint')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (e.g., cuda:0, cuda:1, cpu)')

    parser.add_argument('--dataset', type=str, default='cardiac', choices=['cardiac', 'lung'], help='Dataset name')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the dataset directory')
    parser.add_argument('--split', type=int, help='Dataset split')

    parser.add_argument('--feature_extract', action='store_true', default=True, help='Use feature extraction')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    args = parser.parse_args()
    main(args)
