@echo off
setlocal EnableExtensions

REM ==========================================
REM  behaviour_stat_pipeline – INTERACTIVE DEMO
REM ==========================================

echo.
echo ==========================================
echo   Behaviour Stat Pipeline - Demo Run
echo ==========================================
echo.

REM ---- Config ----
set "PYTHON=python"
set "STATS_CONFIG=configs\stats.yaml"
set "SUBJECTS=data\metadata\subjects.csv"
REM ----------------

echo Available assays:
echo   NOR  OFT  EPM  VIT  LM
echo.

set /p "ASSAY=Enter assay to run (default: NOR): "
if "%ASSAY%"=="" set "ASSAY=NOR"

REM Canonicalize assay (accept lowercase input)
if /I "%ASSAY%"=="NOR" set "ASSAY=NOR"
if /I "%ASSAY%"=="OFT" set "ASSAY=OFT"
if /I "%ASSAY%"=="EPM" set "ASSAY=EPM"
if /I "%ASSAY%"=="VIT" set "ASSAY=VIT"
if /I "%ASSAY%"=="LM"  set "ASSAY=LM"

REM Validate
if /I not "%ASSAY%"=="NOR" if /I not "%ASSAY%"=="OFT" if /I not "%ASSAY%"=="EPM" if /I not "%ASSAY%"=="VIT" if /I not "%ASSAY%"=="LM" (
    echo.
    echo Invalid assay: "%ASSAY%"
    echo Please enter one of: NOR OFT EPM VIT LM
    pause
    exit /b 1
)

REM Map assay to raw subfolder
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

REM =========================
REM [0/3] PROCESSING (LIVE)
REM =========================
echo [0/3] Processing raw tracking data for %ASSAY%...
echo (If this fails, demo continues using existing processed data.)
echo.

%PYTHON% scripts\run_assay.py --assay %ASSAY% --root "%RAW_ROOT%" --subjects "%SUBJECTS%"
if errorlevel 1 (
    echo.
    echo Processing failed (fail-fast safeguard triggered).
    echo Continuing demo using existing processed data in data\processed\
    echo.
)

REM =========================
REM [1/3] STATS
REM =========================
echo [1/3] Running stats for %ASSAY%...
echo.

%PYTHON% scripts\run_stats.py --config "%STATS_CONFIG%" --assays %ASSAY%
if errorlevel 1 (
    echo.
    echo Stats failed. Check logs in outputs\logs
    pause
    exit /b 1
)

REM =========================
REM [2/3] FIGURES
REM =========================
echo [2/3] Generating figures for %ASSAY%...
echo.

%PYTHON% scripts\run_figures.py --config "%STATS_CONFIG%" --assays %ASSAY%
if errorlevel 1 (
    echo.
    echo Figure generation failed.
    pause
    exit /b 1
)

REM =========================
REM DONE
REM =========================
echo.
echo Demo completed successfully for %ASSAY%!
echo.
echo Processed -> data\processed\
echo Tables    -> outputs\tables\
echo Figures   -> outputs\figures\
echo Logs      -> outputs\logs\
echo.

pause
exit /b 0
