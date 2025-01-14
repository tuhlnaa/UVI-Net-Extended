import os
from PIL import Image
from pathlib import Path
import argparse

def create_webp_animation(input_dir, output_path, duration=100, loop=0):
    """
    Create an animated WebP from PNG frames
    
    Args:
        input_dir (str): Directory containing the PNG frames
        output_path (str): Path where the WebP file will be saved
        duration (int): Duration for each frame in milliseconds
        loop (int): Number of times to loop animation (0 = infinite)
    """
    # Get list of frames and sort them
    frames = []
    frame_files = sorted([f for f in os.listdir(input_dir) if f.startswith('frame_') and f.endswith('.png')])
    
    if not frame_files:
        raise ValueError(f"No frame_*.png files found in {input_dir}")
    
    print(f"Found {len(frame_files)} frames")
    
    # Read all frames
    for frame_file in frame_files:
        frame_path = os.path.join(input_dir, frame_file)
        try:
            with Image.open(frame_path) as img:
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                frames.append(img.copy())
            print(f"Processed {frame_file}")
        except Exception as e:
            print(f"Error processing {frame_file}: {str(e)}")
            continue
    
    if not frames:
        raise ValueError("No frames were successfully loaded")
    
    # Save as animated WebP
    try:
        frames[0].save(
            output_path,
            format='WebP',
            append_images=frames[1:],
            save_all=True,
            duration=duration,  # Duration per frame in milliseconds
            loop=loop,  # 0 for infinite loop
            optimize=True,
            quality=90  # Quality factor (0-100)
        )
        print(f"Successfully created animated WebP: {output_path}")
    except Exception as e:
        print(f"Error saving WebP: {str(e)}")


def main():
    parser = argparse.ArgumentParser(description='Convert PNG frames to animated WebP')
    parser.add_argument('--input_dir', required=True, help='Directory containing PNG frames')
    parser.add_argument('--output_path', required=True, help='Path for output WebP file')
    parser.add_argument('--duration', type=int, default=100, help='Duration per frame in milliseconds')
    parser.add_argument('--loop', type=int, default=0, help='Number of animation loops (0 = infinite)')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_path = Path(args.output_path)
    
    if not input_dir.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    create_webp_animation(
        str(input_dir),
        str(output_path),
        duration=args.duration,
        loop=args.loop
    )

if __name__ == "__main__":
    main()