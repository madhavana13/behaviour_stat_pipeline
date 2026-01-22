@echo off
REM ==========================================
REM  behaviour_stat_pipeline – DEMO
REM ==========================================

echo.
echo ==========================================
echo   Behaviour Stat Pipeline – Demo Run
echo ==========================================
echo.

REM ---- Config ----
set PYTHON=python
set ASSAY=NOR
set STATS_CONFIG=configs\stats.yaml
set FIG_CONFIG=configs\figures.yaml
REM ----------------

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
echo Demo completed successfully!
echo.
echo Tables  → outputs\tables\
echo Figures → outputs\figures\
echo Logs    → outputs\logs\
echo.

pause
