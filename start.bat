@echo off
REM People Analytics System - Windows Launcher
REM ============================================

title People Analytics System

echo ========================================
echo   People Analytics System - Windows
echo ========================================
echo.

REM Configuration
set PA_PORT=5000
set PA_USERNAME=admin
set PA_PASSWORD=admin123
set PYTHON_CMD=python

REM Check if Python is installed
%PYTHON_CMD% --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found!
    echo.
    echo Please install Python 3.10+ from:
    echo https://www.python.org/downloads/
    echo.
    echo Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)

echo Python found:
%PYTHON_CMD% --version
echo.

REM Check if pip is available
%PYTHON_CMD% -m pip --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: pip not found!
    echo Please ensure pip is installed with Python.
    pause
    exit /b 1
)

REM Get the script directory
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

echo Working directory: %CD%
echo.

REM Check if virtual environment exists, create if not
if not exist "venv" (
    echo Creating virtual environment...
    %PYTHON_CMD% -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip >nul 2>&1

REM Install requirements
echo Checking dependencies...
if not exist "venv\Lib\site-packages\flask" (
    echo Installing dependencies (this may take a few minutes)...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ERROR: Failed to install dependencies
        pause
        exit /b 1
    )
)

REM Get IP address for network access
for /f "tokens=2 delims=:" %%a in ('ipconfig ^| findstr /i "IPv4" ^| findstr /v "127.0.0.1"') do (
    set LOCAL_IP=%%a
    goto :found_ip
)
:found_ip
set LOCAL_IP=%LOCAL_IP: =%

echo.
echo ========================================
echo   Network Information
echo ========================================
echo.
echo Local IP: %LOCAL_IP%
echo Port: %PA_PORT%
echo.

echo ========================================
echo   Access URLs
echo ========================================
echo.
echo Local:    http://localhost:%PA_PORT%
echo Network:  http://%LOCAL_IP%:%PA_PORT%
echo.

echo ========================================
echo   Authentication
echo ========================================
echo.
echo Username: %PA_USERNAME%
echo Password: %PA_PASSWORD%
echo.

echo ========================================
echo   Starting Server...
echo ========================================
echo.
echo Press Ctrl+C to stop
echo.

REM Set environment variables
set PA_USERNAME=%PA_USERNAME%
set PA_PASSWORD=%PA_PASSWORD%
set PA_PORT=%PA_PORT%

REM Start the server
python web\app.py

REM Deactivate virtual environment on exit
call venv\Scripts\deactivate.bat

echo.
echo Server stopped.
pause
