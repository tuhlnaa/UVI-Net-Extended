import os
os.environ["VXM_BACKEND"] = "pytorch"

import time
import torch
import voxelmorph as vxm

from losses import Grad3d, Grad

def benchmark_gradients(batch_size=16, channels=3, size=64, iterations=1000):
    # Create test data
    data_2d = torch.randn(batch_size, channels, size, size)
    data_3d = torch.randn(batch_size, channels, size, size, size)
    
    # Initialize both versions
    grad_old_2d = Grad(penalty="l1")
    grad_old_3d = Grad3d(penalty="l1")
    grad_new = vxm.losses.Grad(penalty="l1")
    
    # Benchmark 2D
    start = time.time()
    for _ in range(iterations):
        grad_old_2d(data_2d, None)
    old_2d_time = (time.time() - start) / iterations
    
    start = time.time()
    for _ in range(iterations):
        grad_new.loss(None, data_2d)
    new_2d_time = (time.time() - start) / iterations
    
    # Benchmark 3D
    start = time.time()
    for _ in range(iterations):
        grad_old_3d(data_3d, None)
    old_3d_time = (time.time() - start) / iterations
    
    start = time.time()
    for _ in range(iterations):
        grad_new.loss(None, data_3d)
    new_3d_time = (time.time() - start) / iterations
    
    return {
        '2D Old': old_2d_time,
        '2D New': new_2d_time,
        '3D Old': old_3d_time,
        '3D New': new_3d_time
    }

print(benchmark_gradients())
