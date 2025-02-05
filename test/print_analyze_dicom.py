import pydicom
import numpy as np

def load_and_analyze_dicom(file_path):
    # Load the DICOM file
    dicom = pydicom.dcmread(file_path)
    
    # Get basic image information
    print("=== Basic DICOM Information ===")
    print(f"Modality: {dicom.Modality}")
    print(f"Patient Position: {getattr(dicom, 'PatientPosition', 'N/A')}")
    print(f"Image Type: {getattr(dicom, 'ImageType', 'N/A')}")
    
    # Get pixel information
    pixel_array = dicom.pixel_array
    print("\n=== Pixel Data Information ===")
    print(f"Image Shape: {pixel_array.shape}")
    print(f"Data Type: {pixel_array.dtype}")
    print(f"Pixel Value Range: [{np.min(pixel_array)}, {np.max(pixel_array)}]")
    
    # Get pixel spacing and units
    print("\n=== Spatial Information ===")
    print(f"Pixel Spacing: {getattr(dicom, 'PixelSpacing', 'N/A')}")
    print(f"Slice Thickness: {getattr(dicom, 'SliceThickness', 'N/A')}")
    print(f"Image Position (Patient): {getattr(dicom, 'ImagePositionPatient', 'N/A')}")
    
    # Get acquisition information
    print("\n=== Acquisition Information ===")
    print(f"Manufacturer: {getattr(dicom, 'Manufacturer', 'N/A')}")
    print(f"Study Description: {getattr(dicom, 'StudyDescription', 'N/A')}")
    print(f"Series Description: {getattr(dicom, 'SeriesDescription', 'N/A')}")
    
    return dicom

# Usage example
if __name__ == "__main__":
    file_path = "path/to/your/file.dcm"
    file_path = r"D:\Kai\DATA_Set_2\4D-Lung\100_HM10395\10-01-1997-NA-p4-50488\503.000000-P4P100S112I0 Gated 30.0-69092\1-25.dcm"
    dicom_data = load_and_analyze_dicom(file_path)