@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo Starting Conference Copilot...
python copilot.py %*
if errorlevel 1 pause
