@echo off
setlocal

rem Configuration
set "DATA_ROOT=D:\Medical_Datasets"

rem Dataset paths and outputs
set "LUNG_FRAMES_DIR=%DATA_ROOT%\LungDataset_Frames"
set "ACDC_FRAMES_DIR=%DATA_ROOT%\ACDCHeartDataset_Frames"
set "WEBP_OUTPUT_DIR=%DATA_ROOT%\WebP_Animations"

rem Animation settings
set "CARDIAC_DURATION=53"         rem 800ms (75bpm cardiac cycle) / 30 frames * 2 = 53
set "RESPIRATORY_DURATION=400"    rem 4000ms (15breaths/min) / 10 frames = 800
set "LOOP_COUNT=0"                rem Infinite loop

rem Print header
echo === Medical Image WebP Animation Generator ===
echo Starting execution at %date% %time%

rem Create output directory if it doesn't exist
if not exist "%WEBP_OUTPUT_DIR%" mkdir "%WEBP_OUTPUT_DIR%"

rem Generate WebP animation from Lung frames
echo Processing Lung Dataset (Respiratory Motion)...
python utils\create_animation.py ^
    --input_dir "%LUNG_FRAMES_DIR%" ^
    --output_path "%WEBP_OUTPUT_DIR%\lung_animation.webp" ^
    --duration %RESPIRATORY_DURATION% ^
    --loop %LOOP_COUNT%

rem Generate WebP animation from ACDC frames
echo Processing ACDC Dataset (Cardiac Motion)...
python utils\create_animation.py ^
    --input_dir "%ACDC_FRAMES_DIR%" ^
    --output_path "%WEBP_OUTPUT_DIR%\cardiac_animation.webp" ^
    --duration %CARDIAC_DURATION% ^
    --loop %LOOP_COUNT%

echo === Processing completed at %date% %time% ===

endlocal