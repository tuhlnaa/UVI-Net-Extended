@echo off
setlocal

rem Configuration
set "DATA_PATH=dataset"

set "RAW_FOLDER_DIR=%DATA_PATH%\4D-Lung"
set "OUTPUT_DIR=%DATA_PATH%\4D-Lung-Preprocessed"

rem Print header
echo === Lung CT Preprocessing Pipeline ===
echo Starting execution at %date% %time%

rem Execute the preprocessing script
echo [EXECUTING] Starting preprocessing script...
python preprocess_lung_ct.py ^
    --raw_folder_dir "%RAW_DATA_PATH%" ^
    --output_dir "%OUTPUT_PATH%"

echo === Preprocessing completed at %date% %time% ===
endlocal