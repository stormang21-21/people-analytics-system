@echo off
REM People Analytics System - Windows with ngrok Tunnel
REM ===================================================

title People Analytics System + ngrok
echo ========================================
echo   People Analytics + ngrok Tunnel
echo ========================================
echo.

REM Configuration
set PA_PORT=5000
set PA_USERNAME=admin
set PA_PASSWORD=admin123

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found! Install from python.org
    pause
    exit /b 1
)

REM Check ngrok
where ngrok >nul 2>&1
if errorlevel 1 (
    echo ERROR: ngrok not found!
    echo.
    echo Install ngrok:
    echo 1. Download from https://ngrok.com/download
    echo 2. Extract ngrok.exe to this folder or add to PATH
    echo 3. Run: ngrok config add-authtoken YOUR_TOKEN
    echo.
    echo Get token: https://dashboard.ngrok.com/get-started/your-authtoken
    pause
    exit /b 1
)

cd /d "%~dp0"

REM Activate venv if exists
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
)

REM Install deps if needed
if not exist "venv\Lib\site-packages\flask" (
    echo Installing dependencies...
    pip install -r requirements.txt
)

REM Get IP
for /f "tokens=2 delims=:" %%a in ('ipconfig ^| findstr /i "IPv4"') do (
    set LOCAL_IP=%%a
    goto :found_ip
)
:found_ip
set LOCAL_IP=%LOCAL_IP: =%

echo.
echo Starting People Analytics Server...
echo.

REM Start server in background
start "People Analytics Server" cmd /c "python web\app.py & pause"

timeout /t 3 >nul

echo.
echo Starting ngrok tunnel...
echo.
echo ========================================
echo   Your Public URL will appear below
echo ========================================
echo.

REM Start ngrok - this will show the public URL
grok http %PA_PORT%

echo.
echo ========================================
echo   Access Information
echo ========================================
echo.
echo Local:    http://localhost:%PA_PORT%
echo Username: %PA_USERNAME%
echo Password: %PA_PASSWORD%
echo.

pause
