"""
- First Row:
    - Moving Image: The original source image
    - Fixed Image: The target image
    - Flow Field: Visualization of the displacement field using HSV color coding
- Second Row:
    - Original Grid with Flow: Shows the sampling grid and displacement vectors
    - Deformed Grid: Shows how the grid points move after transformation
    - Warped Image: The final transformed image
- Third Row:
    - Difference Before: Shows initial differences between moving and fixed images
    - Difference After: Shows remaining differences after transformation
    - Checkerboard View: Alternating patches of fixed and warped images for easy comparison
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import nibabel as nib
from matplotlib.colors import LinearSegmentedColormap

class SpatialTransformer(nn.Module):
    # Original SpatialTransformer class remains unchanged
    def __init__(self, size, mode="bilinear", gpu=True):
        super().__init__()
        self.mode = mode
        
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)
        if gpu and torch.cuda.is_available():
            grid = grid.cuda()
            
        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        self.new_locs = self.grid + flow
        shape = flow.shape[2:]

        for i in range(len(shape)):
            self.new_locs[:, i, ...] = 2 * (self.new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        if len(shape) == 2:
            self.new_locs = self.new_locs.permute(0, 2, 3, 1)
            self.new_locs = self.new_locs[..., [1, 0]]
        elif len(shape) == 3:
            self.new_locs = self.new_locs.permute(0, 2, 3, 4, 1)
            self.new_locs = self.new_locs[..., [2, 1, 0]]

        return F.grid_sample(src, self.new_locs, align_corners=True, mode=self.mode)


def load_and_preprocess(file_path, slice_idx=None, device='cuda'):
    """Load and preprocess NIfTI file."""
    img = nib.load(file_path)
    data = img.get_fdata()
    data = (data - data.min()) / (data.max() - data.min())
    
    if slice_idx is not None:
        data = data[:, :, slice_idx]
    
    data_tensor = torch.from_numpy(data).float()
    return data_tensor.unsqueeze(0).unsqueeze(0).to(device)


def create_sample_flow(image_shape, device='cuda'):
    """Create a sample displacement field.
    
    Note: Flow values indicate where to sample FROM, not where to move TO.
    Positive x-flow (sampling from right) results in content moving left
    Positive y-flow (sampling from below) results in content moving up
    """
    flow = torch.zeros((1, 2, *image_shape), device=device)
    # These positive values mean "sample from right and down"
    # Which results in content appearing to move left and up
    flow[:, 0, :, :] = 10  # Sample from right (content moves left)
    flow[:, 1, :, :] = 5   # Sample from below (content moves up)
    return flow


def calculate_flow_visualization(flow):
    """
    Calculate flow field visualization with enhanced color mapping.
    
    - Uses a custom colormap with more distinct colors
    - Scales saturation with flow magnitude to show flow strength
    - Maintains good visibility while varying brightness
    - Adds alpha channel based on flow magnitude
    - Uses a wider range of colors (from dark blue through cyan, green, yellow, to red)
    """
    # Calculate magnitude and angle
    flow_magnitude = torch.sqrt(flow[0, 0]**2 + flow[0, 1]**2)
    flow_angle = torch.atan2(flow[0, 1], flow[0, 0])
    
    # Create custom color mapping
    hsv = np.zeros((*flow_magnitude.shape, 3))
    
    # Hue: Map angles to full color spectrum
    hsv[..., 0] = (flow_angle.cpu().numpy() / (2 * np.pi) + 0.5) % 1.0
    
    # Saturation: Scale with magnitude to show strength of flow
    normalized_magnitude = flow_magnitude.cpu().numpy() / flow_magnitude.max().cpu().numpy()
    hsv[..., 1] = np.clip(normalized_magnitude * 1.5, 0, 1)
    
    # Value: Keep brighter for better visibility but vary with magnitude
    hsv[..., 2] = np.clip(0.8 + normalized_magnitude * 0.2, 0, 1)
    
    # Convert HSV to RGB
    rgb = plt.cm.hsv(hsv[..., 0])
    
    # Create a custom colormap
    colors = ['darkblue', 'blue', 'cyan', 'lightgreen', 'yellow', 'orange', 'red']
    n_bins = 256
    custom_cmap = LinearSegmentedColormap.from_list('custom_flow', colors, N=n_bins)
    
    # Apply colormap
    flow_vis = custom_cmap(hsv[..., 0])
    flow_vis[..., 3] = hsv[..., 1]  # Use saturation as alpha
    
    return flow_vis


def create_checkerboard(shape):
    """Create a checkerboard pattern."""
    return np.indices(shape).sum(axis=0) % 2


def save_visualization(img_data, title, filename, add_colorbar=False):
    """Save individual visualization."""
    plt.figure(figsize=(5, 5))

    if len(img_data.shape) == 3 and img_data.shape[-1] == 3:  # For RGB/HSV
        plt.imshow(img_data)
    else:
        im = plt.imshow(img_data, cmap='gray')
        if add_colorbar:
            plt.colorbar(im)

    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def visualize_transformation_steps(moving_img_path, fixed_img_path, output_dir, slice_idx=None):
    """Generate separate visualizations for each transformation step."""
    # Load and prepare images
    moving = load_and_preprocess(moving_img_path, slice_idx)
    fixed = load_and_preprocess(fixed_img_path, slice_idx)
    
    # Create flow field and transformer
    flow = create_sample_flow(moving.shape[2:])
    transformer = SpatialTransformer(size=moving.shape[2:])
    
    # Apply transformation
    warped = transformer(moving, flow)
    
    # Generate all visualizations
    visualizations = {
        'moving': moving[0, 0].cpu().numpy(),
        'fixed': fixed[0, 0].cpu().numpy(),
        'flow_field': calculate_flow_visualization(flow),
        'original_grid': {
            'x': transformer.grid[0, 0].cpu().numpy(),
            'y': transformer.grid[0, 1].cpu().numpy(),
            'flow_x': flow[0, 0, ::20, ::20].cpu().numpy(),
            'flow_y': flow[0, 1, ::20, ::20].cpu().numpy()
        },
        'deformed_grid': transformer.new_locs[0].cpu().numpy(),
        'warped': warped[0, 0].detach().cpu().numpy(),
        'diff_before': (moving - fixed)[0, 0].cpu().numpy(),
        'diff_after': (warped - fixed)[0, 0].detach().cpu().numpy(),
        'checkerboard': np.where(create_checkerboard(moving.shape[2:]),
                                warped[0, 0].detach().cpu().numpy(),
                                fixed[0, 0].cpu().numpy())
    }
    
    # Save each visualization separately
    save_visualization(visualizations['moving'], 'Moving Image', f'{output_dir}/1_moving.png', add_colorbar=True)
    save_visualization(visualizations['fixed'], 'Fixed Image', f'{output_dir}/2_fixed.png', add_colorbar=True)
    save_visualization(visualizations['flow_field'], 'Flow Field (HSV)', f'{output_dir}/3_flow_field.png')

    # Save grid visualization
    plt.figure(figsize=(5, 5))
    plt.quiver(visualizations['original_grid']['x'][::20, ::20],
              visualizations['original_grid']['y'][::20, ::20],
              visualizations['original_grid']['flow_x'],
              visualizations['original_grid']['flow_y'])
    plt.title('Original Grid with Flow')
    plt.axis('off')
    plt.savefig(f'{output_dir}/4_original_grid.png')
    plt.close()
    
    # Save deformed grid
    plt.figure(figsize=(5, 5))
    plt.scatter(visualizations['deformed_grid'][::20, ::20, 0],
               visualizations['deformed_grid'][::20, ::20, 1],
               c='r', s=1)
    plt.title('Deformed Grid')
    plt.axis('off')
    plt.savefig(f'{output_dir}/5_deformed_grid.png')
    plt.close()
    
    save_visualization(visualizations['warped'], 'Warped Image', f'{output_dir}/6_warped.png', add_colorbar=True)
    save_visualization(visualizations['diff_before'], 'Difference Before', f'{output_dir}/7_diff_before.png', add_colorbar=True)
    save_visualization(visualizations['diff_after'], 'Difference After', f'{output_dir}/8_diff_after.png', add_colorbar=True)
    save_visualization(visualizations['checkerboard'], 'Checkerboard View', f'{output_dir}/9_checkerboard.png', add_colorbar=True)

# Example usage:
if __name__ == "__main__":
    moving = r"E:\Kai_2\CODE_Repository\UVI-Net-Extended\dataset\4D-Lung_Preprocessed\100_0\ct_100_0_frame0.nii.gz"
    fixed = r"E:\Kai_2\CODE_Repository\UVI-Net-Extended\dataset\4D-Lung_Preprocessed\100_0\ct_100_0_frame4.nii.gz"
    output_dir = "outputs"
    import os
    os.makedirs(output_dir, exist_ok=True)
    visualize_transformation_steps(moving, fixed, output_dir, slice_idx=100)