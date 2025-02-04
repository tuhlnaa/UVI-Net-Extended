@echo off
setlocal

rem Configuration
set "DATA_PATH=dataset"

set "RAW_FOLDER_DIR=%DATA_PATH%/4D-Lung"
set "OUTPUT_DIR=%DATA_PATH%/4D-Lung_Preprocessed"
set "N_PROCESSES=12"

rem Print header
echo === Lung CT Preprocessing Pipeline ===
echo Starting execution at %date% %time%

rem Execute the preprocessing script
echo [EXECUTING] Starting preprocessing script...
python data/lung_ct_preprocessor.py ^
    --raw_folder_dir "%RAW_FOLDER_DIR%" ^
    --output_dir %OUTPUT_DIR% ^
    --n_processes %N_PROCESSES%

echo === Preprocessing completed at %date% %time% ===
endlocal