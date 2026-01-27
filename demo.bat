@echo off
setlocal EnableExtensions

echo.
echo ==========================================
echo   Behaviour Stat Pipeline - Demo Run
echo ==========================================
echo.

set "PYTHON=python"
set "STATS_CONFIG=configs\stats.yaml"
set "SUBJECTS=data\metadata\subjects.csv"

echo Available assays:
echo   NOR  OFT  EPM  VIT  LM
echo.

set /p "ASSAY=Enter assay to run (default: NOR): "
if "%ASSAY%"=="" set "ASSAY=NOR"

REM Canonicalize
if /I "%ASSAY%"=="NOR" set "ASSAY=NOR"
if /I "%ASSAY%"=="OFT" set "ASSAY=OFT"
if /I "%ASSAY%"=="EPM" set "ASSAY=EPM"
if /I "%ASSAY%"=="VIT" set "ASSAY=VIT"
if /I "%ASSAY%"=="LM"  set "ASSAY=LM"

REM Validate
if /I "%ASSAY%"=="NOR" goto assay_ok
if /I "%ASSAY%"=="OFT" goto assay_ok
if /I "%ASSAY%"=="EPM" goto assay_ok
if /I "%ASSAY%"=="VIT" goto assay_ok
if /I "%ASSAY%"=="LM"  goto assay_ok

echo.
echo Invalid assay: "%ASSAY%"
echo Please enter one of: NOR OFT EPM VIT LM
pause
exit /b 1

:assay_ok

REM Map to raw folder
set "RAW_SUBFOLDER="
if "%ASSAY%"=="NOR" set "RAW_SUBFOLDER=nor"
if "%ASSAY%"=="OFT" set "RAW_SUBFOLDER=oft"
if "%ASSAY%"=="EPM" set "RAW_SUBFOLDER=epm"
if "%ASSAY%"=="VIT" set "RAW_SUBFOLDER=vit"
if "%ASSAY%"=="LM"  set "RAW_SUBFOLDER=lm"

set "RAW_ROOT=data\raw\%RAW_SUBFOLDER%"

echo.
echo Selected assay: %ASSAY%
echo Raw root: "%RAW_ROOT%"
echo Subjects: "%SUBJECTS%"
echo.

echo [0/3] Processing raw tracking data for %ASSAY%...
echo (If this fails, demo continues using existing processed data.)
echo.

%PYTHON% scripts\run_assay.py --assay %ASSAY% --root "%RAW_ROOT%" --subjects "%SUBJECTS%" || goto processing_failed
goto processing_done

:processing_failed
echo.
echo Processing failed (fail-fast safeguard triggered).
echo Continuing demo using existing processed data in data\processed\
echo.
goto processing_done

:processing_done

echo [1/3] Running stats for %ASSAY%...
echo.
%PYTHON% scripts\run_stats.py --config "%STATS_CONFIG%" --assays %ASSAY% || goto stats_failed

echo [2/3] Generating figures for %ASSAY%...
echo.
%PYTHON% scripts\run_figures.py --config "%STATS_CONFIG%" --assays %ASSAY% || goto figs_failed

echo.
echo Demo completed successfully for %ASSAY%!
echo.
pause
exit /b 0

:stats_failed
echo.
echo Stats failed. Check logs in outputs\logs
pause
exit /b 1

:figs_failed
echo.
echo Figure generation failed.
pause
exit /b 1
