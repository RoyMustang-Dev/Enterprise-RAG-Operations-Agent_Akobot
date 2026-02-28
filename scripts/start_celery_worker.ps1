$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$repo = Resolve-Path (Join-Path $root "..")
$py = Join-Path $repo "venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }

Write-Host "[CELERY] Starting worker..."
$envPath = Join-Path $repo ".env"
if (Test-Path $envPath) {
  Get-Content $envPath | ForEach-Object {
    if ($_ -match "^\s*#") { return }
    if ($_ -match "^\s*$") { return }
    $parts = $_ -split "=", 2
    if ($parts.Count -eq 2) {
      $key = $parts[0].Trim()
      $val = $parts[1].Trim()
      $env:$key = $val
    }
  }
}
$env:VIRTUAL_ENV = (Join-Path $repo "venv")
$env:PYTHONHOME = ""
$env:PYTHONPATH = ""
$env:PYTHONEXECUTABLE = $py
$env:PATH = (Join-Path $repo "venv\Scripts") + ";" + $env:PATH

# Windows-safe pool to avoid WinError 5 with billiard
& $py -m celery -A app.infra.celery_app.celery_app worker --loglevel=info --pool=solo --concurrency=1
