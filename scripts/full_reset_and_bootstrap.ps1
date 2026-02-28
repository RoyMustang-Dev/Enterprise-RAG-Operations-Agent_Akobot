$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$repo = Resolve-Path (Join-Path $root "..")

Write-Host "[RESET] Stopping running API/Celery processes..."
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*uvicorn*app.main:app*' -or $_.CommandLine -like '*celery*app.infra.celery_app*' } | ForEach-Object {
  try { Stop-Process -Id $_.ProcessId -Force } catch {}
}
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like "*$repo*" -and $_.Name -eq "python.exe" } | ForEach-Object {
  try { Stop-Process -Id $_.ProcessId -Force } catch {}
}

Write-Host "[RESET] Removing venv and caches..."
@(
  (Join-Path $repo "venv"),
  (Join-Path $repo "model-cache"),
  (Join-Path $repo "data\\qdrant_storage"),
  (Join-Path $repo ".pytest_cache")
) | ForEach-Object {
  if (Test-Path $_) { Remove-Item $_ -Recurse -Force -ErrorAction SilentlyContinue }
}

Get-ChildItem -Path $repo -Recurse -Force -Directory -Filter "__pycache__" | ForEach-Object {
  try { Remove-Item $_.FullName -Recurse -Force -ErrorAction SilentlyContinue } catch {}
}

Write-Host "[RESET] Clearing pip cache..."
try { python -m pip cache purge | Out-Null } catch {}

Write-Host "[RESET] Recreating venv..."
python -m venv (Join-Path $repo "venv")

$py = Join-Path $repo "venv\\Scripts\\python.exe"

Write-Host "[RESET] Installing requirements..."
& $py -m pip install --upgrade pip
& $py -m pip install -r (Join-Path $repo "requirements.txt")

Write-Host "[RESET] Running bootstrap with auto-fix..."
$env:AUTO_FIX_SYSTEM_DEPS = "true"
& (Join-Path $repo "scripts\\bootstrap_env.ps1")

Write-Host "[RESET] Starting stack..."
$env:CELERY_AUTOSTART = "true"
& (Join-Path $repo "scripts\\start_stack.ps1")
