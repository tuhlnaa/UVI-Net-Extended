import os
os.environ["VXM_BACKEND"] = "pytorch"

import torch
import numpy as np
import voxelmorph as vxm

from utils.losses import Grad3d, Grad

def test_grad_loss():
    # Test case 1: Simple 2D case with more pronounced differences
    grad_l1 = Grad(penalty="l1")
    grad_l2 = Grad(penalty="l2")
    
    # Create a test pattern with sharp edges and gradual changes
    # Shape: [batch_size, channels, height, width]
    test_input = torch.tensor([[
        [
            [1., 1., 1., 10.],
            [1., 1., 1., 10.],
            [1., 1., 1., 10.],
            [5., 5., 5., 15.]
        ]
    ]], dtype=torch.float32)
    
    # Calculate and print individual gradients before averaging
    dy = torch.abs(test_input[:, :, 1:, :] - test_input[:, :, :-1, :])
    dx = torch.abs(test_input[:, :, :, 1:] - test_input[:, :, :, :-1])
    
    print("2D Gradient Components (L1):")
    print("\nVertical gradients (dy):")
    print(dy[0, 0])
    print("\nHorizontal gradients (dx):")
    print(dx[0, 0])
    
    # Calculate L2 gradients
    dy_l2 = dy * dy
    dx_l2 = dx * dx
    
    print("\n2D Gradient Components (L2):")
    print("\nVertical gradients squared (dy²):")
    print(dy_l2[0, 0])
    print("\nHorizontal gradients squared (dx²):")
    print(dx_l2[0, 0])
    
    # Compute final losses
    loss_l1 = grad_l1(test_input, None)
    loss_l2 = grad_l2(test_input, None)
    
    print("\n2D Gradient Test Results:")
    print(f"L1 loss: {loss_l1.item():.4f}")
    print(f"L2 loss: {loss_l2.item():.4f}")

def test_grad3d_loss():
    # Test case for 3D gradient with more pronounced differences
    grad3d_l1 = Grad3d(penalty="l1")
    grad3d_l2 = Grad3d(penalty="l2")
    
    # grad3d_l1 = vxm.losses.Grad(penalty="l1")
    # grad3d_l2 = vxm.losses.Grad(penalty="l2")

    # Create a 3D test volume with sharp transitions
    # Shape: [batch_size, channels, depth, height, width]
    test_input_3d = torch.ones((1, 1, 4, 4, 4), dtype=torch.float32)
    
    # Add a sharp boundary in the middle
    test_input_3d[0, 0, :2, :, :] = 1.0
    test_input_3d[0, 0, 2:, :, :] = 10.0
    test_input_3d[0, 0, :, :2, :] *= 2.0
    test_input_3d[0, 0, :, :, :2] *= 1.5
    
    # Calculate and print individual gradients before averaging
    dy = torch.abs(test_input_3d[:, :, 1:, :, :] - test_input_3d[:, :, :-1, :, :])
    dx = torch.abs(test_input_3d[:, :, :, 1:, :] - test_input_3d[:, :, :, :-1, :])
    dz = torch.abs(test_input_3d[:, :, :, :, 1:] - test_input_3d[:, :, :, :, :-1])
    
    print("\n3D Gradient Components (L1):")
    print("\nDepth gradients (dy) at middle slice:")
    print(dy[0, 0, 1])
    print("\nHeight gradients (dx) at middle slice:")
    print(dx[0, 0, :, 1])
    print("\nWidth gradients (dz) at middle slice:")
    print(dz[0, 0, :, :, 1])
    
    # Calculate L2 gradients
    dy_l2 = dy * dy
    dx_l2 = dx * dx
    dz_l2 = dz * dz
    
    # Compute final losses
    loss3d_l1 = grad3d_l1(test_input_3d, None)
    loss3d_l2 = grad3d_l2(test_input_3d, None)

    # loss3d_l1 = grad3d_l1.loss(None, test_input_3d)
    # loss3d_l2 = grad3d_l2.loss(None, test_input_3d)

    print("\n3D Gradient Test Results:")
    print(f"3D L1 loss: {loss3d_l1.item():.4f}")
    print(f"3D L2 loss: {loss3d_l2.item():.4f}")
    
    # Print the ratio between L2 and L1 losses to show the difference
    print(f"\nRatio of L2/L1 loss: {loss3d_l2.item() / loss3d_l1.item():.4f}")

if __name__ == "__main__":
    print("========Testing 2D Gradient Loss...========")
    test_grad_loss()
    
    print("\n========Testing 3D Gradient Loss...========")
    test_grad3d_loss()