# Start the app locally: backend and frontend each in their own window.
# Usage (from the repo folder):  .\run-local.ps1
# First run installs what's missing; later runs start straight away.

$root = $PSScriptRoot

# Backend: make the virtual environment and install packages if needed, then start the API
$backend = @"
Set-Location '$root\backend'
if (-not (Test-Path venv)) { python -m venv venv }
.\venv\Scripts\python.exe -m pip install -q -r requirements-dev.txt
if (-not (Test-Path .env)) { Copy-Item .env.example .env; Write-Host 'Created backend\.env from the example: add an AI key to it.' -ForegroundColor Yellow }
.\venv\Scripts\python.exe -m uvicorn app:app --reload --port 8000
"@

# Frontend: install packages if needed, then start the dev server
$frontend = @"
Set-Location '$root\frontend'
if (-not (Test-Path node_modules)) { npm install }
npm run dev
"@

Start-Process powershell -ArgumentList '-NoExit', '-Command', $backend
Start-Process powershell -ArgumentList '-NoExit', '-Command', $frontend

Write-Host 'Starting... open http://localhost:5173 (API: http://localhost:8000/docs)' -ForegroundColor Green
Write-Host 'Stop: press Ctrl+C in each window, or close them.'
