#!/bin/bash

# Make sure the script runs from the directory the file lives in
cd "$(dirname "$0")" || exit

# If the virtual environment exists, activate it before running the program
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Run the program
python3 main_web.py
