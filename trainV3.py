import os
import wandb
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from tqdm import trange
from torch import optim
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Tuple, Dict

from engine.trainerV3 import Trainer
from engine.validatorV3 import Validator
from models.UNet.model import Unet3D, Unet3D_multi
from models.VoxelMorph.model import VoxelMorph
from models.feature_extract.model import FeatureExtract
import losses
from utils import datasets, utils
from configs.configV3 import get_config
from configs.setupV3 import setup_environment


def create_models(img_size: Tuple[int, int, int], feature_extract: bool):
    """Create and initialize models.
    
    Args:
        img_size: Input image dimensions
        feature_extract: Whether to use feature extraction
        
    Returns:
        Tuple of initialized models
    """
    flow_model = VoxelMorph(img_size).cuda()
    refinement_model = (Unet3D_multi(img_size) if feature_extract 
                       else Unet3D(img_size)).cuda()
    feature_model = FeatureExtract().cuda() if feature_extract else None
    
    return flow_model, refinement_model, feature_model


def create_dataloaders(config) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation dataloaders.
    
    Args:
        config: Configuration object containing dataset parameters
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    if config.dataset == "cardiac":
        data_dir = Path("dataset/ACDC/database/training")
        split = 90 if config.split is None else config.split
        dataset_cls = datasets.ACDCHeartDataset
    elif config.dataset == "lung":
        data_dir = Path("dataset/4D-Lung_Preprocessed")
        split = 68 if config.split is None else config.split
        dataset_cls = datasets.LungDataset
    else:
        raise ValueError(f"Unknown dataset: {config.dataset}")

    train_set = dataset_cls(data_dir, phase="train", split=split)
    val_set = dataset_cls(data_dir, phase="test", split=split)

    train_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=True,
    )

    return train_loader, val_loader


def create_registration_models(img_size: Tuple[int, int, int]):
    """Create and initialize registration models.
    
    Args:
        img_size: Input image dimensions
        
    Returns:
        Tuple of registration models
    """
    reg_model = utils.register_model(img_size, "nearest").cuda()
    reg_model_bilin = utils.register_model(img_size, "bilinear").cuda()
    
    # Freeze registration model parameters
    for model in [reg_model, reg_model_bilin]:
        for param in model.parameters():
            param.requires_grad = False
            
    return reg_model, reg_model_bilin


def create_optimizer(models: list, learning_rate: float):
    """Create optimizer for training.
    
    Args:
        models: List of models to optimize
        learning_rate: Learning rate
        
    Returns:
        Optimizer
    """
    return optim.Adam(
        [p for model in models if model is not None
         for p in model.parameters()],
        lr=learning_rate,
        weight_decay=0,
        amsgrad=True,
    )


def save_checkpoint(
    epoch: int,
    models: Dict[str, torch.nn.Module],
    optimizer: torch.optim.Optimizer,
    best_ncc: float,
    save_path: Path
):
    """Save model checkpoint.
    
    Args:
        epoch: Current epoch number
        models: Dictionary of models to save
        optimizer: Optimizer state to save
        best_ncc: Best NCC score
        save_path: Path to save checkpoint
    """
    checkpoint = {
        "epoch": epoch + 1,
        "best_ncc": best_ncc,
        "optimizer": optimizer.state_dict(),
    }
    
    for name, model in models.items():
        if model is not None:
            checkpoint[f"{name}_state_dict"] = model.state_dict()
    
    torch.save(checkpoint, save_path)


def main():
    # Get configuration
    config = get_config()
    
    # Setup environment
    setup_environment(config)
    
    # Create experiment directory
    exp_dir = Path("experiments") / config.dataset
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Initialize W&B
    wandb.init(project="UVI-Net", name=config.dataset, config=config)

    # Set image size based on dataset
    img_size = (128, 128, 32) if config.dataset == "cardiac" else (128, 128, 128)

    # Create models
    flow_model, refinement_model, feature_model = create_models(
        img_size, config.feature_extract
    )
    reg_model, reg_model_bilin = create_registration_models(img_size)
    
    # Create optimizer
    optimizer = create_optimizer(
        [flow_model, refinement_model, feature_model],
        config.lr
    )

    # Create loss functions
    criterion_ncc = losses.NCC()
    criterion_cha = losses.CharbonnierLoss
    criterion_reg = losses.Grad3d(penalty="l2")
    criterion_l1n = losses.L1_norm()

    # Create data loaders
    train_loader, val_loader = create_dataloaders(config)

    # Create trainer and validator
    trainer = Trainer(
        flow_model=flow_model,
        refinement_model=refinement_model,
        feature_model=feature_model,
        optimizer=optimizer,
        criterion_ncc=criterion_ncc,
        criterion_cha=criterion_cha,
        criterion_reg=criterion_reg,
        criterion_l1n=criterion_l1n,
        reg_model=reg_model,
        reg_model_bilin=reg_model_bilin,
        img_size=img_size,
        config=config,
    )
    
    validator = Validator(
        flow_model=flow_model,
        refinement_model=refinement_model,
        feature_model=feature_model,
        criterion_ncc=criterion_ncc,
        reg_model_bilin=reg_model_bilin,
    )

    # Training loop
    best_ncc = -1
    for epoch in trange(config.max_epoch):
        # Train epoch
        metrics = trainer.train_epoch(train_loader, epoch)
        
        # Validate
        ncc_score = validator.validate(val_loader, epoch)
        
        # Update best score
        if ncc_score > best_ncc:
            best_ncc = ncc_score
        
        # Save checkpoint
        save_checkpoint(
            epoch=epoch,
            models={
                "flow_model": flow_model,
                "refinement_model": refinement_model,
                "feature_model": feature_model,
            },
            optimizer=optimizer,
            best_ncc=best_ncc,
            save_path=exp_dir / f"epoch{epoch + 1}_ncc{ncc_score:.4f}.ckpt"
        )

    print(f"Best NCC: {best_ncc}")
    wandb.finish()


if __name__ == "__main__":
    main()