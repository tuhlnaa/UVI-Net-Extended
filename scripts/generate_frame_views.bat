@echo off
setlocal

rem Configuration
set "DATA_ROOT=D:\Medical_Datasets"

rem Lung Dataset paths
set "LUNG_DATA_DIR=%DATA_ROOT%\LungDataset"
set "LUNG_OUTPUT_DIR=%DATA_ROOT%\LungDataset_Frames"

rem ACDC Dataset paths
set "ACDC_DATA_DIR=%DATA_ROOT%\ACDCHeartDataset"
set "ACDC_OUTPUT_DIR=%DATA_ROOT%\ACDCHeartDataset_Frames"

rem Print header
echo === Medical Image Frame Visualization Generator ===
echo Starting execution at %date% %time%

rem Process Lung Dataset
python utils\generate_frames.py ^
    --input_path "%LUNG_DATA_DIR%" ^
    --output_path "%LUNG_OUTPUT_DIR%" ^
    --dataset_type lung

rem Process ACDC Dataset
python utils\generate_frames.py ^
    --input_path "%ACDC_DATA_DIR%" ^
    --output_path "%ACDC_OUTPUT_DIR%" ^
    --dataset_type acdc

echo === Processing completed at %date% %time% ===

endlocal