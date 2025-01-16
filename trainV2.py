import os
os.environ["VXM_BACKEND"] = "pytorch"

import wandb
import torch
import random
import numpy as np
import torch.backends.cudnn as cudnn

from tqdm import trange
from torch.utils.data import DataLoader

from configs.config import get_args
from engine.model_setupV2 import setup_models
from engine.trainerV2 import Trainer
from utils import datasets

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True


def setup_gpu(args):
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    GPU_iden = 0
    GPU_num = torch.cuda.device_count()
    print("Number of GPU: " + str(GPU_num))
    for GPU_idx in range(GPU_num):
        GPU_name = torch.cuda.get_device_name(GPU_idx)
        print("     GPU #" + str(GPU_idx) + ": " + GPU_name)
    torch.cuda.set_device(GPU_iden)
    GPU_avai = torch.cuda.is_available()
    print("Currently using: " + torch.cuda.get_device_name(GPU_iden))
    print("If the GPU is available? " + str(GPU_avai))


def get_data_loaders(args):
    if args.dataset == "cardiac":
        img_size = (128, 128, 32)
        split = 90 if args.split is None else args.split
        data_dir = os.path.join("dataset", "ACDC", "database", "training")
        train_set = datasets.ACDCHeartDataset(data_dir, phase="train", split=split)
        val_set = datasets.ACDCHeartDataset(data_dir, phase="test", split=split)

    elif args.dataset == "lung":
        img_size = (128, 128, 128)
        split = 68 if args.split is None else args.split
        data_dir = os.path.join("dataset", "4D-Lung_Preprocessed")
        train_set = datasets.LungDataset(data_dir, phase="train", split=split)
        val_set = datasets.LungDataset(data_dir, phase="test", split=split)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=True,
    )
    
    return train_loader, val_loader, img_size


def main():
    args = get_args()
    setup_gpu(args)
    set_seed(args.seed)

    # Create experiment directory
    if not os.path.exists(f"experiments/{args.dataset}"):
        os.makedirs(f"experiments/{args.dataset}")

    # Initialize wandb
    wandb.init(project="UVI-Net", name=args.dataset, config=args)

    # Setup data
    train_loader, val_loader, img_size = get_data_loaders(args)

    # Setup models and optimizer
    flow_model, refinement_model, feature_model, reg_model, reg_model_bilin, optimizer = setup_models(args, img_size)

    # Initialize trainer
    trainer = Trainer(
        flow_model=flow_model,
        refinement_model=refinement_model,
        feature_model=feature_model,
        reg_model=reg_model,
        reg_model_bilin=reg_model_bilin,
        optimizer=optimizer,
        args=args,
        img_size=img_size
    )

    # Training loop
    best_ncc = -1
    for epoch in trange(args.max_epoch):
        # Train
        loss_all = trainer.train_epoch(train_loader, epoch)
        
        # Validate
        eval_ncc = trainer.validate(val_loader)
        print(f"Epoch {epoch}, NCC {eval_ncc.avg:.5f}\n", flush=True)
        wandb.log({"Validate/NCC": eval_ncc.avg}, step=epoch)

        # Save checkpoint if improved
        if eval_ncc.avg > best_ncc:
            best_ncc = eval_ncc.avg

        # Save model checkpoint
        torch.save(
            {
                "epoch": epoch + 1,
                "flow_model_state_dict": flow_model.state_dict(),
                "model_state_dict": refinement_model.state_dict(),
                "feature_model_state_dict": feature_model.state_dict() if args.feature_extract else None,
                "best_ncc": best_ncc,
                "optimizer": optimizer.state_dict(),
            },
            f"experiments/{args.dataset}/epoch{epoch + 1}_ncc{eval_ncc.avg:.4f}.ckpt"
        )

    print(f"Best NCC: {best_ncc}", flush=True)
    wandb.finish()


if __name__ == "__main__":
    main()