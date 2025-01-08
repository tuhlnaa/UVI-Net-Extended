@echo off
setlocal

rem Configuration
set "FILE1=file1.nii.gz"
set "DATA_PATH=file2.nii.gz"
set "OUTPUT_PATH=/comparison_output"


rem Print header
echo === NIfTI File Comparison Tool ===
echo Starting execution at %date% %time%

rem Execute the comparison script
echo [EXECUTING] Starting comparison script...
python compare_nifti.py ^
    --file1 "%FILE1%" ^
    --file2 "%FILE2%" ^
    --output "%OUTPUT_PATH%"

echo === Comparison completed at %date% %time% ===
endlocal