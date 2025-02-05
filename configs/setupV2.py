import os
import random
from typing import Union

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from .configV2 import Config


def setup_environment(config: Config) -> None:
    """Setup training environment.
    
    Args:
        config: Configuration object containing parameters
    """
    # Set backend
    os.environ["VXM_BACKEND"] = "pytorch"
    
    # Set random seeds for reproducibility
    set_seed(config.seed)
    
    # GPU configuration
    setup_gpu(config.gpu)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True


def setup_gpu(gpu_id: Union[str, None]) -> None:
    """Setup GPU environment.
    
    Args:
        gpu_id: GPU identifier string or None
    """
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    # Set default GPU
    GPU_iden = 0
    torch.cuda.set_device(GPU_iden)
    
    # Print GPU information
    GPU_num = torch.cuda.device_count()
    print(f"Number of GPU: {GPU_num}")
    
    for GPU_idx in range(GPU_num):
        GPU_name = torch.cuda.get_device_name(GPU_idx)
        print(f"     GPU #{GPU_idx}: {GPU_name}")
    
    print(f"Currently using: {torch.cuda.get_device_name(GPU_iden)}")
    print(f"If the GPU is available? {torch.cuda.is_available()}")