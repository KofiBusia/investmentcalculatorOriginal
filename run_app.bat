@echo off
echo Setting up Investment Calculator...
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found! Please install Python 3.8+ first.
    echo Download from: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo ✅ Python found

REM Check if requirements are installed
echo Checking dependencies...
pip show flask >nul 2>&1
if errorlevel 1 (
    echo Installing required packages...
    pip install -r requirements.txt
) else (
    echo ✅ All dependencies are installed
)

echo.
echo ⚡ Starting Investment Calculator...
echo.
echo 📍 Access the app at: http://127.0.0.1:5000
echo 📍 Or on your phone/other computer: http://[YOUR-IP-ADDRESS]:5000
echo.
echo⚠️  Note: This will open in your browser automatically.
echo.
python app.py

if errorlevel 1 (
    echo.
    echo ❌ Failed to start the application.
    echo.
    echo 💡 Troubleshooting:
    echo 1. Make sure port 5000 is not in use
    echo 2. Check if all dependencies are installed
    echo 3. Try: python -m pip install --upgrade pip
    echo 4. Then: pip install -r requirements.txt
    pause
)