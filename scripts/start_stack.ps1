$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$repo = Resolve-Path (Join-Path $root "..")
$py = Join-Path $repo "venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }

Write-Host "[STACK] Starting API on port 8000..."
Start-Process -FilePath "powershell.exe" -ArgumentList "-NoProfile","-File",(Join-Path $root "start_api.ps1") -WorkingDirectory $repo

if ($env:CELERY_AUTOSTART -eq "true") {
  Write-Host "[STACK] Starting Celery worker..."
  Start-Process -FilePath "powershell.exe" -ArgumentList "-NoProfile","-File",(Join-Path $root "start_celery_worker.ps1") -WorkingDirectory $repo
} else {
  Write-Host "[STACK] CELERY_AUTOSTART is false; not starting worker."
}
