#!/bin/bash

echo ""
echo "  Nickplots - Installation"
echo "  ========================"
echo ""
echo "  Installing dependencies... this may take a few minutes."
echo ""
read -p "Press [Enter] to continue..."

# Check that Python 3 is installed (macOS uses python3 by default)
if ! command -v python3 &> /dev/null; then
    echo ""
    echo "  ERROR: Python 3 not found on PATH."
    echo "  Install Python from https://python.org or via Homebrew (brew install python)."
    echo ""
    exit 1
fi

# Work from the script's own directory
cd "$(dirname "$0")" || exit

# Use a virtual environment: this avoids the "externally-managed-environment"
# error that recent macOS/Linux Python installs raise on a global pip install.
echo "  Setting up an isolated virtual environment (venv)..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip and install the dependencies inside the virtual environment
python3 -m pip install --upgrade pip --quiet
python3 -m pip install customtkinter matplotlib seaborn pandas scipy scikit-learn openpyxl shapely pywebview --quiet

if [ $? -ne 0 ]; then
    echo ""
    echo "  Installation failed. Check your internet connection or the directory permissions."
    echo ""
    exit 1
fi

echo ""
echo "  Installation complete! Starting Nickplots..."
echo ""

# Run the main program
python3 main_web.py
