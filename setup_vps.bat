@echo off
REM ══════════════════════════════════════════════
REM  Apex Predator — VPS Setup Script (Windows)
REM  Run this after cloning from GitHub on VPS
REM ══════════════════════════════════════════════

echo.
echo  ========================================
echo   Apex Predator - VPS Setup
echo  ========================================
echo.

REM ── Step 1: Check Python ──────────────────
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found! Please install Python 3.10+
    echo Download: https://www.python.org/downloads/
    pause
    exit /b 1
)
echo [OK] Python found

REM ── Step 2: Create Virtual Environment ────
echo [..] Creating virtual environment...
python -m venv .venv
call .venv\Scripts\activate.bat
echo [OK] Virtual environment activated

REM ── Step 3: Install Dependencies ──────────
echo [..] Installing dependencies...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
echo [OK] Dependencies installed

REM ── Step 4: Create Directories ────────────
echo [..] Creating directories...
if not exist "models" mkdir models
if not exist "logs" mkdir logs
if not exist "memory_store" mkdir memory_store
if not exist "data" mkdir data
echo [OK] Directories created

REM ── Step 5: Verify Installation ───────────
echo [..] Verifying installation...
python -c "from config.settings import ApexConfig; c = ApexConfig(); print(f'[OK] Config loaded: {c.mt5.symbol}')"
python -c "import torch; print(f'[OK] PyTorch {torch.__version__}')"
python -c "import stable_baselines3; print(f'[OK] SB3 {stable_baselines3.__version__}')"

echo.
echo  ========================================
echo   Setup Complete!
echo  ========================================
echo.
echo  Next steps:
echo  1. Edit config\settings.py with your MT5 credentials
echo  2. Open MetaTrader 5 and login
echo  3. python train.py          (train model)
echo  4. python main.py --live    (start trading)
echo.
pause
