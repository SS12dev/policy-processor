@echo off
echo Installing psutil for monitoring...
pip install psutil
echo.
echo Starting simple agent monitor...
python simple_monitor.py
pause