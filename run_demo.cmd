@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Missing .venv. Follow README.md to install Python dependencies.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" run.py demo
set "RUN_RESULT=%ERRORLEVEL%"
echo.
echo Finished with exit code %RUN_RESULT%. Results are in outputs.
pause
exit /b %RUN_RESULT%
