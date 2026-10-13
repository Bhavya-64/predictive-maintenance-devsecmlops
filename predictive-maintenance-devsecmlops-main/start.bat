@echo off
REM Double-click to start the app (Windows). First run sets everything up.
cd /d "%~dp0"
if not exist .venv (
    echo Setting up for the first time, this takes a few minutes...
    python -m venv .venv
    call .venv\Scripts\activate.bat
    pip install -r requirements-dev.txt
) else (
    call .venv\Scripts\activate.bat
)
python run.py
pause
