import torch
import pydicom
import numpy as np
import nibabel as nib
import torch.nn.functional as F

from tqdm import tqdm
from pathlib import Path
from argparse import ArgumentParser

from preprocess_func import bed_removal, center_crop, min_max

def read_dcm_files(folder_path: Path):
    # Read all DICOM files in the folder and sort them
    files = [
        pydicom.dcmread(f)
        for f in folder_path.iterdir()
        if f.suffix == ".dcm"
    ]
    files.sort(key=lambda x: int(x.InstanceNumber))
    files.reverse()

    # Create a 3D numpy array from the DICOM files
    image_array = np.stack(
        [file.pixel_array.astype(np.float32) - 1000 for file in files], axis=-1
    )  # -1000 for mitigating the automatic rescale function

    return image_array

def resize_image(image_array, new_shape=(128, 128, 128), rot90=False):
    image_tensor = torch.from_numpy(image_array)[None, None, ...].float()
    image_tensor = F.interpolate(image_tensor, new_shape, mode="trilinear")
    image_array = image_tensor.squeeze().numpy()

    # Rotate each slice
    if rot90:
        for i in range(image_array.shape[-1]):
            image_array[..., i] = np.rot90(
                image_array[..., i], k=1
            )  # Rotate by 90 degrees
    return image_array

def main(args):
    raw_folder = Path(args.raw_folder_dir)
    output_dir = Path(args.output_dir)
    pt_folder_list = sorted(raw_folder.glob("*_*"))
    lower_bound, upper_bound = -1400, 200  # window range for lung window
    data_count = 0

    for pt_folder in tqdm(pt_folder_list):
        pt_num = pt_folder.name[:3]
        pt_date_list = list(pt_folder.glob("*-*/"))
        high_res_scans = sorted(
            [scan for scan in pt_date_list if next(scan.iterdir()).name[:8] == "1.000000"]
        )

        if high_res_scans:
            for scan_idx, scan_folder in enumerate(high_res_scans):
                ct_scan_4d_dir = sorted(
                    [
                        scan
                        for scan in scan_folder.glob("*-*")
                        if len(list(scan.iterdir())) > 10
                    ]
                )
                scan_save_dir = output_dir / f"{pt_num}_{scan_idx}"
                scan_save_dir.mkdir(parents=True, exist_ok=True)

                for phase_idx, folder_path in enumerate(ct_scan_4d_dir):
                    resized_image_array = read_dcm_files(folder_path)
                    resized_image_array = resize_image(resized_image_array, new_shape=(256, 256, 256), rot90=True)
                    resized_image_array = resize_image(resized_image_array, new_shape=(128, 128, 128), rot90=False)

                    # adjust lung window
                    resized_image_array[resized_image_array < lower_bound] = lower_bound
                    resized_image_array[resized_image_array > upper_bound] = upper_bound

                    # bed removal, center crop, and min-max scaling
                    resized_image_array = bed_removal(resized_image_array)
                    resized_image_array = center_crop(resized_image_array)
                    resized_image_array = min_max(resized_image_array)

                    nifti_img = nib.Nifti1Image(resized_image_array, None)
                    nifti_file_path = scan_save_dir / f"ct_{pt_num}_{scan_idx}_frame{phase_idx}.nii.gz"
                    nib.save(nifti_img, nifti_file_path)
                data_count += 1

    print(f"A total of {data_count} samples are created")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--raw_folder_dir",
        type=str,
        default="dataset/4D-Lung/",
        help="designate the downloaded raw data folder directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="dataset/4D-Lung-Preprocessed/",
        help="output directory for saving the preprocessed outcomes",
    )
    args = parser.parse_args()
    
    main(args)