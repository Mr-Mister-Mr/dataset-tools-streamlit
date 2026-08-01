@echo off
title Launching Dataset Caption Editor...

:: 1. Force the script to run from its own directory
cd /d "%~dp0"

:: 2. Check if a local virtual environment exists, activate it if found
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
) else if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
) else (
    :: If no local venv, activate the dataset_tools conda environment
    if exist "C:\ProgramData\miniconda3\Scripts\activate.bat" (
        call "C:\ProgramData\miniconda3\Scripts\activate.bat" dataset_tools
    )
)

:: 3. Launch the Streamlit application
echo Starting Streamlit app...
streamlit run app.py

pause