"""
Visualization script for SpatialTransformer that focuses on the transformation process:
- First Row:
    - Source Image: The original image to be transformed
    - Flow Field: Visualization of the displacement field using HSV color coding
    - Warped Image: The final transformed image
- Second Row:
    - Original Grid with Flow: Shows the sampling grid and displacement vectors
    - Deformed Grid: Shows how the grid points move after transformation
    - Flow Magnitude: Heatmap showing the strength of displacement at each point

SpatialTransformer represents where to "sample from" in the source image, not where to "move pixels to".

When the flow field shows an arrow pointing upper-right (positive x, positive y):
- This means "for this position, sample from a point that is up and right".
- Therefore, the content in the warped image will appear to move in the opposite direction (down and left).
- Because if we're sampling from up/right, the content effectively shifts down/left.

When the flow field says (+10, +5):
- For each position in the output, we're saying "look 10 pixels to the right and 5 pixels down in the source image".
- Therefore, content from the right/down position in the source appears in the current position in the output.
- This makes the overall content appear to shift left/up in the warped image.

"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import nibabel as nib
from matplotlib.colors import LinearSegmentedColormap

class SpatialTransformer(nn.Module):
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
    flow_magnitude = torch.sqrt(flow[0, 0]**2 + flow[0, 1]**2)
    flow_angle = torch.atan2(flow[0, 1], flow[0, 0])
    
    hsv = np.zeros((*flow_magnitude.shape, 3))
    hsv[..., 0] = (flow_angle.cpu().numpy() / (2 * np.pi) + 0.5) % 1.0
    hsv[..., 1] = np.clip(flow_magnitude.cpu().numpy() / flow_magnitude.max().cpu().numpy() * 1.5, 0, 1)
    hsv[..., 2] = np.clip(0.8 + flow_magnitude.cpu().numpy() / flow_magnitude.max().cpu().numpy() * 0.2, 0, 1)
    
    rgb = plt.cm.hsv(hsv[..., 0])
    colors = ['darkblue', 'blue', 'cyan', 'lightgreen', 'yellow', 'orange', 'red']
    custom_cmap = LinearSegmentedColormap.from_list('custom_flow', colors, N=256)
    
    flow_vis = custom_cmap(hsv[..., 0])
    flow_vis[..., 3] = hsv[..., 1]
    
    return flow_vis

def save_visualization(img_data, title, filename, add_colorbar=False):
    """Save individual visualization."""
    plt.figure(figsize=(5, 5))
    
    if len(img_data.shape) == 3 and img_data.shape[-1] in [3, 4]:  # For RGB/RGBA
        plt.imshow(img_data)
    else:
        im = plt.imshow(img_data, cmap='gray')
        if add_colorbar:
            plt.colorbar(im)
    
    plt.title(title, fontsize = 20)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

def visualize_transformation_steps(moving_img_path, output_dir, slice_idx=None):
    """Generate visualizations focusing on the transformation process."""
    # Load and prepare image
    moving = load_and_preprocess(moving_img_path, slice_idx)
    
    # Create flow field and transformer
    flow = create_sample_flow(moving.shape[2:])
    transformer = SpatialTransformer(size=moving.shape[2:])
    
    # Apply transformation
    warped = transformer(moving, flow)
    
    # Generate visualizations
    flow_magnitude = torch.sqrt(flow[0, 0]**2 + flow[0, 1]**2)
    
    visualizations = {
        'source': moving[0, 0].cpu().numpy(),
        'flow_field': calculate_flow_visualization(flow),
        'warped': warped[0, 0].detach().cpu().numpy(),
        'original_grid': {
            'x': transformer.grid[0, 0].cpu().numpy(),
            'y': transformer.grid[0, 1].cpu().numpy(),
            'flow_x': flow[0, 0, ::20, ::20].cpu().numpy(),
            'flow_y': flow[0, 1, ::20, ::20].cpu().numpy()
        },
        'deformed_grid': transformer.new_locs[0].cpu().numpy(),
        'flow_magnitude': flow_magnitude.cpu().numpy()
    }
    
    # Save visualizations
    save_visualization(visualizations['source'], 'Source Image', 
                      f'{output_dir}/1_source.png', add_colorbar=True)
    save_visualization(visualizations['flow_field'], 'Flow Field (HSV)', 
                      f'{output_dir}/2_flow_field.png')
    save_visualization(visualizations['warped'], 'Warped Image', 
                      f'{output_dir}/3_warped.png', add_colorbar=True)

    # Save grid visualization
    plt.figure(figsize=(5, 5))
    plt.quiver(visualizations['original_grid']['x'][::20, ::20],
              visualizations['original_grid']['y'][::20, ::20],
              visualizations['original_grid']['flow_x'],
              visualizations['original_grid']['flow_y'])
    plt.title('Original Grid with Flow', fontsize = 20)
    plt.axis('off')
    plt.savefig(f'{output_dir}/4_original_grid.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save deformed grid
    # plt.figure(figsize=(5, 5))
    # plt.scatter(visualizations['deformed_grid'][::20, ::20, 0],
    #             visualizations['deformed_grid'][::20, ::20, 1],
    #             c='r', s=1)
    # plt.title('Deformed Grid')
    # plt.axis('off')
    # plt.savefig(f'{output_dir}/5_deformed_grid.png')
    # plt.close()

    plt.figure(figsize=(5, 5))
    # Get grid points
    grid_points = visualizations['deformed_grid'][::20, ::20]
    
    # Draw horizontal grid lines
    for i in range(grid_points.shape[0]):
        plt.plot(grid_points[i, :, 0], grid_points[i, :, 1], 
                'b-', alpha=0.3, linewidth=0.5)
    
    # Draw vertical grid lines
    for j in range(grid_points.shape[1]):
        plt.plot(grid_points[:, j, 0], grid_points[:, j, 1], 
                'b-', alpha=0.3, linewidth=0.5)
    
    # Plot points with increased size and better visibility
    plt.scatter(grid_points[..., 0], grid_points[..., 1],
               c='red', s=25, alpha=0.6, zorder=2)
    
    # Plot points with increased size and better visibility
    plt.scatter(grid_points[..., 0], grid_points[..., 1],
               c='red', s=25, alpha=0.6, zorder=2)
    
    plt.title('Deformed Grid', fontsize = 20)
    plt.axis('off')
    #plt.axis('equal')  # Maintain aspect ratio
    #plt.grid(False)  # Turn off the default grid
    plt.savefig(f'{output_dir}/5_deformed_grid.png', dpi=150, bbox_inches='tight')
    #plt.savefig(f'{output_dir}/5_deformed_grid.png', dpi=150, bbox_inches='tight')
    plt.close()

    save_visualization(visualizations['flow_magnitude'], 'Flow Magnitude',
                      f'{output_dir}/6_flow_magnitude.png', add_colorbar=True)

# Example usage:
if __name__ == "__main__":
    moving = r"E:\Kai_2\CODE_Repository\UVI-Net-Extended\dataset\4D-Lung_Preprocessed\100_0\ct_100_0_frame0.nii.gz"
    output_dir = "outputs"
    import os
    os.makedirs(output_dir, exist_ok=True)
    visualize_transformation_steps(moving, output_dir, slice_idx=100)