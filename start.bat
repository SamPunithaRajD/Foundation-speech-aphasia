@echo off
REM FoundationSpeech-Aphasia: build the frontend and serve app + API at http://127.0.0.1:8000
cd /d "%~dp0"

if not exist "backend\.venv\Scripts\python.exe" (
  echo Backend environment missing. Run once:
  echo   cd backend ^&^& python -m venv .venv ^&^& .venv\Scripts\pip install -r requirements.txt
  pause
  exit /b 1
)
if not exist "node_modules" call npm install
call npm run build
if errorlevel 1 (
  echo Frontend build failed.
  pause
  exit /b 1
)

REM open the browser once the server has had a few seconds to start
start "" cmd /c "timeout /t 8 >nul & start http://127.0.0.1:8000"
echo Starting FoundationSpeech-Aphasia at http://127.0.0.1:8000  (Ctrl+C to stop)
"backend\.venv\Scripts\python.exe" -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000
