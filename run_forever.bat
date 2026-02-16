@echo off
REM ══════════════════════════════════════════════
REM  Apex Predator — Auto-Start Script
REM  Runs the trading bot and auto-restarts on crash
REM  Add this to Windows Task Scheduler for auto-boot
REM ══════════════════════════════════════════════

:loop
echo [%date% %time%] Starting Apex Predator...

REM Activate venv if exists
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
)

REM Run the trading bot
python main.py --live

echo.
echo [%date% %time%] Bot stopped. Restarting in 30 seconds...
echo Press Ctrl+C to cancel restart.
timeout /t 30 /nobreak

goto loop
