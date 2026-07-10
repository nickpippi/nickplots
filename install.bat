@echo off
title Nickplots - Installation

echo.
echo  Nickplots - Installation
echo  ========================
echo.
echo  Installing dependencies... this may take a few minutes.
echo.
pause

python --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo  ERROR: Python not found on PATH.
    echo  Install Python from https://python.org
    echo  and tick "Add Python to PATH" during the installation.
    echo.
    pause
    exit /b 1
)

python -m pip install --upgrade pip --quiet

python -m pip install customtkinter matplotlib seaborn pandas scipy scikit-learn openpyxl shapely pywebview --quiet

if errorlevel 1 (
    echo.
    echo  Installation failed. Try right-clicking install.bat
    echo  and choosing "Run as administrator".
    echo.
    pause
    exit /b 1
)

echo.
echo  Installation complete! Starting Nickplots...
echo.

cd /d "%~dp0"
start "" python main_web.py
