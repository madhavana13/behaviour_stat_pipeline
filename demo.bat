@echo off
REM ==========================================
REM  behaviour_stat_pipeline – INTERACTIVE DEMO
REM ==========================================

echo.
echo ==========================================
echo   Behaviour Stat Pipeline – Demo Run
echo ==========================================
echo.

REM ---- Config ----
set PYTHON=python
set STATS_CONFIG=configs\stats.yaml
set FIG_CONFIG=configs\stats.yaml
REM ----------------

echo Available assays:
echo   NOR  OFT  EPM  VIT  LM 
echo.

set /p ASSAY=Enter assay to run (default: NOR): 

if "%ASSAY%"=="" set ASSAY=NOR

REM Normalize to uppercase
for %%A in (%ASSAY%) do set ASSAY=%%~A

echo.
echo Selected assay: %ASSAY%
echo.

echo [1/2] Running stats for %ASSAY%...
%PYTHON% scripts\run_stats.py ^
    --config %STATS_CONFIG% ^
    --assays %ASSAY%

if errorlevel 1 (
    echo.
    echo Stats failed. Check logs in outputs\logs
    pause
    exit /b 1
)

echo.
echo [2/2] Generating figures for %ASSAY%...
%PYTHON% scripts\run_figures.py ^
    --config %STATS_CONFIG% ^
    --assays %ASSAY%

if errorlevel 1 (
    echo.
    echo Figure generation failed.
    pause
    exit /b 1
)

echo.
echo Demo completed successfully for %ASSAY%!
echo.
echo Tables  → outputs\tables\
echo Figures → outputs\figures\
echo Logs    → outputs\logs\
echo.

pause
