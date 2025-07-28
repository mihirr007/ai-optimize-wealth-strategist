@echo off
echo AI Wealth Strategist
echo ===================

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    pause
    exit /b 1
)

REM Check if Poetry is installed
poetry --version >nul 2>&1
if errorlevel 1 (
    echo Error: Poetry is not installed
    echo Please install Poetry from https://python-poetry.org/
    pause
    exit /b 1
)

REM Install dependencies if needed
echo Installing dependencies...
poetry install

REM Run the wealth strategist with sample data
echo.
echo Starting AI Wealth Strategist...
poetry run python src/main.py --sample

pause 