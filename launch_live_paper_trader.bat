@echo off
setlocal enabledelayedexpansion

REM ============================================================
REM Cryptobot TFM - Live Paper Trader Launcher
REM ============================================================

cd /d "%~dp0"

call "%UserProfile%\anaconda3\Scripts\activate.bat"

echo.
echo ============================================================
echo   Cryptobot TFM - Live Paper Trader
echo ============================================================
echo.

set count=0
for %%f in (models\*.joblib) do (
    set /a count+=1
    set model_!count!=%%f
    echo [!count!] %%f
)

echo.
echo [0] No model (EMA heuristic baseline)
echo.
set /p model_choice=Select model number: 

if "%model_choice%"=="0" (
    set selected_model=none
) else (
    call set selected_model=%%model_%model_choice%%%
)

if not defined selected_model (
    echo.
    echo Invalid model selection.
    pause
    exit /b
)

echo.
echo Selected model: %selected_model%
echo.
set /p bankroll=Initial bankroll (default 1000): 
if "%bankroll%"=="" set bankroll=1000

echo.
echo Symbol:
echo [1] DOGEUSDT
set symbol=DOGEUSDT

echo.
echo State mode:
echo [1] Fresh run from selected bankroll (recommended)
echo [2] Resume previous live paper state
set /p state_choice=Select state mode (default 1): 
if "%state_choice%"=="2" (
    set state_flag=--resume-state
) else (
    set state_flag=--reset-state
)

echo.
echo Refresh interval fixed to 10 seconds.
echo Policy config: results\execution_logs\live_policy_config.json
echo.
echo ============================================================
echo Launching live paper trader...
echo ============================================================
echo.

python scripts\run_live_paper_trader.py ^
--model %selected_model% ^
--bankroll %bankroll% ^
--symbol %symbol% ^
--refresh 10 ^
--policy-config results\execution_logs\live_policy_config.json ^
%state_flag%

pause
