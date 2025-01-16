import os
import argparse
import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path

def plot_middle_slices(data, output_path, frame_num, color='gray', vmin=0, vmax=1):
    """
    Plot middle slices in all three orientations for a given frame, rotated 90 degrees left
    
    Args:
        data (numpy.ndarray): 3D array containing the image data
        output_path (str or Path): Directory to save the output PNG
        frame_num (int): Frame number for file naming
    """
    mid_x = data.shape[0] // 2
    mid_y = data.shape[1] // 2
    mid_z = data.shape[2] // 2
    
    fig = plt.figure(figsize=(15, 4))
    plt.subplots_adjust(wspace=0.4)
    
    ax1 = plt.subplot(131)
    ax2 = plt.subplot(132)
    ax3 = plt.subplot(133)
    
    # Rotate each slice 90 degrees left by transposing and flipping
    sagittal_slice = data[mid_x, :, :].T[::-1]
    coronal_slice = data[:, mid_y, :].T[::-1]
    axial_slice = data[:, :, mid_z].T[::-1]
    
    im1 = ax1.imshow(sagittal_slice, cmap=color, vmin=vmin, vmax=vmax)
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    ax1.set_title('Sagittal')
    
    im2 = ax2.imshow(coronal_slice, cmap=color, vmin=vmin, vmax=vmax)
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    ax2.set_title('Coronal')
    
    im3 = ax3.imshow(axial_slice, cmap=color, vmin=vmin, vmax=vmax)
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    ax3.set_title('Axial')
    
    plt.suptitle(f'Frame {frame_num:02d}', y=1.05)
    
    output_file = os.path.join(output_path, f'frame_{frame_num:02d}.png')
    plt.savefig(output_file, bbox_inches='tight', dpi=300, pad_inches=0.1)
    plt.close()


def process_lung_dataset(patient_path, output_path):
    """Process frames from LungDataset"""
    os.makedirs(output_path, exist_ok=True)
    
    # Find all frame files
    frame_files = sorted([f for f in os.listdir(patient_path) if f.endswith('.nii.gz')])

    # Process each frame
    for frame_file in frame_files:
        frame_num = int(frame_file.split('frame')[1].split('.')[0])
        img_path = os.path.join(patient_path, frame_file)
        
        try:
            img = nib.load(img_path)
            data = img.get_fdata()
            
            # Normalize data to 0-1 range
            data = (data - data.min()) / (data.max() - data.min())
            
            plot_middle_slices(data, output_path, frame_num)
            print(f"Processed frame {frame_num:02d} from {os.path.basename(patient_path)}")
            
        except Exception as e:
            print(f"Error processing {frame_file}: {str(e)}")


def process_acdc_dataset(patient_path, output_path):
    """Process frames from ACDCHeartDataset"""
    os.makedirs(output_path, exist_ok=True)
    
    # Find the 4D file
    frame_files = [f for f in os.listdir(patient_path) if '_4d.nii' in f]
    
    if not frame_files:
        raise ValueError(f"No 4D NIfTI file found in {patient_path}")
    
    img_path = os.path.join(patient_path, frame_files[0])
    
    try:
        img = nib.load(img_path)
        data_4d = img.get_fdata()
        
        # Expected shape: (216, 256, 10, 30) - last dimension contains frames
        num_frames = data_4d.shape[3]
        
        # Process each frame
        for frame_num in range(num_frames):
            data = data_4d[..., frame_num]
            
            # Normalize data to 0-1 range
            data = (data - data.min()) / (data.max() - data.min())
            
            plot_middle_slices(data, output_path, frame_num + 1)  # +1 to match frame numbering convention
            print(f"Processed frame {frame_num+1:02d} from {os.path.basename(patient_path)}")
            
    except Exception as e:
        print(f"Error processing {img_path}: {str(e)}")


def main():
    parser = argparse.ArgumentParser(description='Generate PNG visualizations for NIfTI frames')
    parser.add_argument('--input_path', required=True, help='Path to patient directory')
    parser.add_argument('--output_path', required=True, help='Path to save PNG outputs')
    parser.add_argument('--dataset_type', required=True, choices=['lung', 'acdc'], 
                      help='Type of dataset (lung or acdc)')
    
    args = parser.parse_args()
    
    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    
    if not input_path.exists():
        raise ValueError(f"Input path does not exist: {input_path}")

    if args.dataset_type == 'lung':
        process_lung_dataset(input_path, output_path)
    else:
        process_acdc_dataset(input_path, output_path)

if __name__ == "__main__":
    main()