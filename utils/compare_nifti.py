import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

from pathlib import Path


def compare_nifti_files(file1_path: str, file2_path: str, output_dir: str = "comparison_output"):
    """
    Compare two NIfTI files and generate comparison metrics and visualizations.
    
    Args:
        file1_path (str): Path to first NIfTI file
        file2_path (str): Path to second NIfTI file
        output_dir (str): Directory to save comparison results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the NIfTI files
    img1 = nib.load(file1_path)
    img2 = nib.load(file2_path)
    
    # Get the data arrays
    data1 = img1.get_fdata()
    data2 = img2.get_fdata()
    
    # Basic statistics comparison
    stats = {
        'file1': {
            'min': np.min(data1),
            'max': np.max(data1),
            'mean': np.mean(data1),
            'std': np.std(data1),
            'shape': data1.shape
        },
        'file2': {
            'min': np.min(data2),
            'max': np.max(data2),
            'mean': np.mean(data2),
            'std': np.std(data2),
            'shape': data2.shape
        }
    }
    
    # Calculate differences
    if data1.shape == data2.shape:
        diff = data1 - data2
        stats['difference'] = {
            'min': np.min(diff),
            'max': np.max(diff),
            'mean': np.mean(diff),
            'std': np.std(diff),
            'mse': np.mean(diff**2),
            'mae': np.mean(np.abs(diff))
        }
    
    # Save statistics to text file
    with open(os.path.join(output_dir, 'comparison_stats.txt'), 'w') as f:
        for key, value in stats.items():
            f.write(f"\n{key.upper()} Statistics:\n")
            f.write("-" * 20 + "\n")
            for stat_name, stat_value in value.items():
                f.write(f"{stat_name}: {stat_value}\n")
    
    
    # Visualization function
    def plot_middle_slices(data, title, filename, color='gray', vmin=0, vmax=1):
        """Plot middle slices in all three orientations"""
        mid_x = data.shape[0] // 2
        mid_y = data.shape[1] // 2
        mid_z = data.shape[2] // 2
        
        # Increase bottom margin to prevent x-axis label cutoff
        fig = plt.figure(figsize=(15, 4))
        plt.subplots_adjust(wspace=0.4)
        
        # Create subplots with space for colorbars
        ax1 = plt.subplot(131)
        ax2 = plt.subplot(132)
        ax3 = plt.subplot(133)
        
        # Plot images with colorbars
        im1 = ax1.imshow(data[mid_x, :, :], cmap=color, vmin=vmin, vmax=vmax)
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        ax1.set_title('Sagittal')
        
        im2 = ax2.imshow(data[:, mid_y, :], cmap=color, vmin=vmin, vmax=vmax)
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        ax2.set_title('Coronal')
        
        im3 = ax3.imshow(data[:, :, mid_z], cmap=color, vmin=vmin, vmax=vmax)
        plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
        ax3.set_title('Axial')
        
        # Add overall title
        plt.suptitle(title, y=1.05)
        
        # Save figure with tight layout but preserving margins
        plt.savefig(os.path.join(output_dir, filename), 
                   bbox_inches='tight', 
                   dpi=300,
                   pad_inches=0.1)
        plt.close()
    
    # Generate visualizations
    plot_middle_slices(data1, 'File 1 Middle Slices', 'file1_slices.png')
    plot_middle_slices(data2, 'File 2 Middle Slices', 'file2_slices.png')
    
    if data1.shape == data2.shape:
        plot_middle_slices(diff, 'Difference Middle Slices', 'difference_slices.png', color='viridis', vmin=-1, vmax=1)
        
        # Improved histogram visualization
        plt.figure(figsize=(10, 6))
        plt.hist(diff.flatten(), bins=100, edgecolor='black')
        plt.title('Histogram of Differences')
        plt.xlabel('Difference Value')
        plt.ylabel('Frequency')
        plt.xlim([-1, 1])

        # Add padding to prevent label cutoff
        plt.tight_layout(pad=1.5)
        plt.savefig(os.path.join(output_dir, 'difference_histogram.png'),
                   bbox_inches='tight',
                   dpi=300)
        plt.close()

    return stats


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare two NIfTI files')
    parser.add_argument('--file1', help='Path to first NIfTI file')
    parser.add_argument('--file2', help='Path to second NIfTI file')
    parser.add_argument('--output', default='comparison_output', help='Output directory for comparison results')
    args = parser.parse_args()
    
    stats = compare_nifti_files(args.file1, args.file2, args.output)
    print("Comparison completed. Results saved in", args.output)