@echo off
cd /d "%~dp0"
echo Creating virtual environment (.venv) ...
py -3.11 -m venv .venv 2>nul || py -3.10 -m venv .venv 2>nul || py -3.12 -m venv .venv 2>nul || python -m venv .venv
if not exist .venv\Scripts\python.exe ( echo Could not create venv. Install Python 3.10-3.12 from python.org. & pause & exit /b 1 )
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
echo.
echo Self-test (no camera needed):
.venv\Scripts\python.exe tools\selftest.py
echo.
echo Done. Start the app with run.bat
pause
