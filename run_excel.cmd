@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Missing .venv. Follow README.md to install Python dependencies.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" run.py train --source excel
if errorlevel 1 goto failed
".venv\Scripts\python.exe" run.py test --source excel
if errorlevel 1 goto failed
echo Finished. Results are in outputs.
pause
exit /b 0
:failed
echo Run failed. See the error above.
pause
exit /b 1
