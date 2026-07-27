@echo off
setlocal
cd /d "%~dp0"
py scripts\delete_trim_files.py
echo.
pause
