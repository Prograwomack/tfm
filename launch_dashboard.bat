@echo off

REM ============================================================
REM Cryptobot TFM - Streamlit Dashboard Launcher
REM ============================================================

@echo off

cd /d "%~dp0"

call "%UserProfile%\anaconda3\Scripts\activate.bat"

python -m streamlit run app/streamlit_dashboard.py

pause