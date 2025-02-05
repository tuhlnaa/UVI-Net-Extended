import os
import random
from argparse import ArgumentParser
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch import optim
from torch.utils.data import DataLoader
import wandb

from utils import utils, datasets
from utils.setup import setup_environment
from tqdm import trange

#from models.feature_extract import FeatureExtract
from models.feature_extract.model import FeatureExtract

#from models.unet import UNet3D, UNet3DMulti
from models.UNet.modelV4 import UNet3D, UNet3DMulti

#from models.voxelmorph import VoxelMorph
from models.VoxelMorph.model import VoxelMorph

import losses
from engine.trainer import Trainer
from engine.validator import Validator


def parse_args():
    """Parse command line arguments."""
    parser = ArgumentParser()
    
    # Basic training parameters
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_epoch", type=int, default=200)
    parser.add_argument("--split", type=int, default=None)
    parser.add_argument("--gpu", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="cardiac", choices=["cardiac", "lung"])
    
    # Model specific parameters
    parser.add_argument("--weight_cycle", type=float, default=1.0)
    parser.add_argument("--weight_diff", type=float, default=1.0)
    parser.add_argument("--weight_ncc", type=float, default=1.0)
    parser.add_argument("--weight_cha", type=float, default=1.0)
    parser.add_argument("--feature_extract", action="store_true", default=True)
    
    return parser.parse_args()


def get_dataset_params(args) -> Tuple[Tuple[int, int, int], int]:
    """Get dataset specific parameters."""
    if args.dataset == "cardiac":
        img_size = (128, 128, 32)
        split = 90 if args.split is None else args.split
    else:  # lung
        img_size = (128, 128, 128)
        split = 68 if args.split is None else args.split
    
    return img_size, split


def setup_models(img_size: Tuple[int, int, int], feature_extract: bool, args) -> Dict[str, torch.nn.Module]:
    """Initialize and setup all required models."""
    refinement_nb_features = [
        [8, 16, 32],             # encoder features
        [32, 32, 32, 8, 8, 3]    # decoder features
    ]
    inshape = [128, 128, 128]
    additional_dims = [4, 8, 16]

    # Initialize flow models
    flow_model = VoxelMorph(img_size).to(args.gpu)
    
    # Initialize reconstruction model
    if feature_extract:
        refinement_model = UNet3DMulti(inshape, feature_maps=refinement_nb_features, additional_dims=additional_dims).to(args.gpu)
        feature_model = FeatureExtract().to(args.gpu)
    else:
        refinement_model = UNet3D(inshape, encoder_features=refinement_nb_features[0], decoder_features=refinement_nb_features[1]).to(args.gpu)
        feature_model = None

    # Initialize spatial transformer
    reg_model = utils.register_model(img_size, "nearest").to(args.gpu)
    reg_model_bilin = utils.register_model(img_size, "bilinear").to(args.gpu)
    
    # Freeze spatial transformer
    for model in [reg_model, reg_model_bilin]:
        for param in model.parameters():
            param.requires_grad = False
            
    return {
        'flow_model': flow_model,
        'refinement_model': refinement_model,
        'feature_model': feature_model,
        'reg_model': reg_model,
        'reg_model_bilin': reg_model_bilin
    }


def setup_dataloaders(args, split: int) -> Tuple[DataLoader, DataLoader]:
    """Setup training and validation dataloaders."""
    if args.dataset == "cardiac":
        data_dir = os.path.join("dataset", "ACDC_database", "training")
        train_dataset = datasets.ACDCHeartDataset(data_dir, phase="train", split=split)
        val_dataset = datasets.ACDCHeartDataset(data_dir, phase="test", split=split)
    else:  # lung
        data_dir = os.path.join("dataset", "4D-Lung_Preprocessed")
        train_dataset = datasets.LungDataset(data_dir, phase="train", split=split)
        val_dataset = datasets.LungDataset(data_dir, phase="test", split=split)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=True,
    )
    
    return train_loader, val_loader


def setup_optimizer(models: Dict[str, torch.nn.Module], lr: float) -> torch.optim.Optimizer:
    """Setup optimizer for training."""
    params = list(models['flow_model'].parameters()) + list(models['refinement_model'].parameters())

    if models['feature_model'] is not None:
        params += list(models['feature_model'].parameters())
        
    return optim.Adam(params, lr=lr, weight_decay=0, amsgrad=True)


def setup_criteria():
    """Setup loss functions."""
    return {
        'ncc': losses.NCCLoss(),
        'cha': losses.CharbonnierLoss,
        'reg': losses.Grad3d(penalty="l2"),
        'l1n': losses.L1_norm()
    }


def save_checkpoint(
    epoch: int,
    models: Dict[str, torch.nn.Module],
    optimizer: torch.optim.Optimizer,
    best_ncc: float,
    save_dir: str,
    current_ncc: float
):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch + 1,
        'flow_model_state_dict': models['flow_model'].state_dict(),
        'model_state_dict': models['refinement_model'].state_dict(),
        'feature_model_state_dict': (models['feature_model'].state_dict() 
                                     if models['feature_model'] is not None 
                                     else None),
        'best_ncc': best_ncc,
        'optimizer': optimizer.state_dict(),
    }
    
    save_path = Path(save_dir) / f'epoch{epoch + 1}_ncc{current_ncc:.4f}.ckpt'
    torch.save(checkpoint, save_path)


def main():
    """Main training loop."""
    args = parse_args()
    setup_environment(args)
    img_size, split = get_dataset_params(args)
    
    # Create experiment directory
    exp_dir = Path("experiments") / args.dataset
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize models, optimizer, and criteria
    models = setup_models(img_size, args.feature_extract, args)
    optimizer = setup_optimizer(models, args.lr)
    criteria = setup_criteria()
    
    # Setup data loaders
    train_loader, val_loader = setup_dataloaders(args, split)
    
    # Initialize trainer and validator
    trainer = Trainer(
        flow_model=models['flow_model'],
        refinement_model=models['refinement_model'],
        feature_model=models['feature_model'],
        optimizer=optimizer,
        reg_model=models['reg_model'],
        reg_model_bilin=models['reg_model_bilin'],
        criterion_ncc=criteria['ncc'],
        criterion_cha=criteria['cha'],
        criterion_reg=criteria['reg'],
        criterion_l1n=criteria['l1n'],
        config=args,
        img_size=img_size,
        device = torch.device(args.gpu)
    )
    
    validator = Validator(
        flow_model=models['flow_model'],
        refinement_model=models['refinement_model'],
        reg_model_bilin=models['reg_model_bilin'],
        criterion_ncc=criteria['ncc']
    )
    
    # Initialize wandb
    wandb.init(project="UVI-Net", name=args.dataset, config=args)
    
    # Training loop
    best_ncc = -1
    for epoch in trange(args.max_epoch):
        train_metrics = trainer.train_epoch(train_loader, epoch)
        val_ncc = validator.validate(val_loader, epoch)

        print(f"Epoch {epoch}, NCC {val_ncc:.5f}\n", flush=True)
        
        # Update best NCC
        if val_ncc > best_ncc:
            best_ncc = val_ncc
            
        # Save checkpoint
        save_checkpoint(epoch, models, optimizer, best_ncc, exp_dir, val_ncc)
        
    print(f"Best NCC: {best_ncc}", flush=True)
    wandb.finish()


if __name__ == "__main__":
    main()