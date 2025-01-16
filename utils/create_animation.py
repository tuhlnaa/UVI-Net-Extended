import os
from PIL import Image
from pathlib import Path
import argparse
import cv2
import numpy as np

def create_animation(input_dir, output_path, duration=100, loop=0):
    """
    Create an animated file (WebP/MP4/MKV) from PNG frames
    
    Args:
        input_dir (str): Directory containing the PNG frames
        output_path (str): Path where the output file will be saved
        duration (int): Duration for each frame in milliseconds
        loop (int): Number of times to loop animation (0 = infinite, only for WebP)
    """
    # Get list of frames and sort them
    frames = []
    frame_files = sorted([f for f in os.listdir(input_dir) if f.startswith('frame_') and f.endswith('.png')])
    
    if not frame_files:
        raise ValueError(f"No frame_*.png files found in {input_dir}")
    
    print(f"Found {len(frame_files)} frames")
    
    # Get output format
    output_format = output_path.suffix.lower()
    
    if output_format == '.webp':
        _create_webp(input_dir, frame_files, output_path, duration, loop)
    elif output_format in ['.mp4', '.mkv']:
        _create_video(input_dir, frame_files, output_path, duration)
    else:
        raise ValueError(f"Unsupported output format: {output_format}. Supported formats: .webp, .mp4, .mkv")


def _create_webp(input_dir, frame_files, output_path, duration, loop):
    """Create WebP animation"""
    frames = []
    for frame_file in frame_files:
        frame_path = os.path.join(input_dir, frame_file)
        try:
            with Image.open(frame_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                frames.append(img.copy())
            print(f"Processed {frame_file}")
        except Exception as e:
            print(f"Error processing {frame_file}: {str(e)}")
            continue
    
    if not frames:
        raise ValueError("No frames were successfully loaded")
    
    try:
        frames[0].save(
            output_path,
            format='WebP',
            append_images=frames[1:],
            save_all=True,
            duration=duration,
            loop=loop,
            optimize=True,
            quality=90
        )
        print(f"Successfully created animated WebP: {output_path}")
    except Exception as e:
        print(f"Error saving WebP: {str(e)}")


def _create_video(input_dir, frame_files, output_path, duration):
    """Create MP4/MKV video"""
    if not frame_files:
        return
    
    # Read first frame to get dimensions
    first_frame = cv2.imread(os.path.join(input_dir, frame_files[0]))
    height, width = first_frame.shape[:2]
    
    # Calculate FPS based on duration (converting from milliseconds to seconds)
    fps = 1000 / duration
    
    # Initialize video writer with VP9 codec
    fourcc = cv2.VideoWriter_fourcc(*'vp09')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    try:
        for frame_file in frame_files:
            frame_path = os.path.join(input_dir, frame_file)
            frame = cv2.imread(frame_path)
            if frame is not None:
                out.write(frame)
                print(f"Processed {frame_file}")
            else:
                print(f"Error reading {frame_file}")
    except Exception as e:
        print(f"Error processing video: {str(e)}")
    finally:
        out.release()
        
    print(f"Successfully created video: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Convert PNG frames to animated WebP/MP4/MKV')
    parser.add_argument('--input_dir', required=True, help='Directory containing PNG frames')
    parser.add_argument('--output_path', required=True, help='Path for output file (.webp/.mp4/.mkv)')
    parser.add_argument('--duration', type=int, default=100, help='Duration per frame in milliseconds')
    parser.add_argument('--loop', type=int, default=0, help='Number of animation loops (0 = infinite, WebP only)')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_path = Path(args.output_path)
    
    if not input_dir.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    create_animation(
        str(input_dir),
        output_path,
        duration=args.duration,
        loop=args.loop
    )

if __name__ == "__main__":
    main()