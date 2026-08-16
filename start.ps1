# start.ps1
Write-Host "Starting VisionGuide AI (backend + frontend)..."

$root = $PSScriptRoot

Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$root'; .\visionguide_env\Scripts\Activate.ps1; uvicorn backend.app:app --reload"

Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$root\frontend'; python -m http.server 5500"

Start-Sleep -Seconds 2
Start-Process "http://127.0.0.1:5500"

Write-Host "Backend starting on http://127.0.0.1:8000"
Write-Host "Frontend starting on http://127.0.0.1:5500"