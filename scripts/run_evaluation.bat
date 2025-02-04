@echo off
setlocal

rem Configuration
set "DATA_DIR=.\dataset\your_dataset"
set "RESUME=.\outputs\model_checkpoints\your_model.ckpt"

set "DATASET=cardiac"
set "DEVICE=cuda:0"

rem Print header
echo === CT evaluation pipeline ===
echo Starting execution at %date% %time%

rem Execute the preprocessing script
echo [EXECUTING] Starting evaluation script...
python evaluation.py ^
    --dataset "%DATASET%" ^
    --device %DEVICE% ^
    --data_dir %DATA_DIR% ^
    --resume %RESUME%

echo === Evaluation completed at %date% %time% ===
endlocal