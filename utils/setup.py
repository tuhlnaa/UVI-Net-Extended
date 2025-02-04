import os
import random
import torch
import numpy as np
import torch.backends.cudnn as cudnn
from typing import Union, Optional
from argparse import Namespace


def setup_environment(args: Namespace) -> None:
    """
    Set up the training environment.
    
    Args:
        args: Command line arguments containing training configuration
    """
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Configure GPU
    #setup_gpu(args.gpu)
    

def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True


def setup_gpu(gpu_id: Optional[str] = None) -> None:
    """
    Configure GPU settings.
    
    Args:
        gpu_id: GPU device ID to use. If None, uses all available GPUs.
    """
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    # Select first GPU as default
    gpu_idx = 0
    torch.cuda.set_device(gpu_idx)

    # Print GPU information
    gpu_count = torch.cuda.device_count()
    print(f"Number of GPUs: {gpu_count}")
    
    for idx in range(gpu_count):
        gpu_name = torch.cuda.get_device_name(idx)
        print(f"     GPU #{idx}: {gpu_name}")
    
    current_gpu = torch.cuda.get_device_name(gpu_idx)
    print(f"Currently using: {current_gpu}")
    print(f"Is GPU available? {torch.cuda.is_available()}")


def get_device(gpu_id: Optional[Union[int, str]] = None) -> torch.device:
    """
    Get the appropriate torch device.
    
    Args:
        gpu_id: GPU device ID to use. If None, uses CUDA if available, else CPU.
    
    Returns:
        torch.device: Device to use for tensor operations
    """
    if gpu_id is not None:
        return torch.device(f'cuda:{gpu_id}')
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')