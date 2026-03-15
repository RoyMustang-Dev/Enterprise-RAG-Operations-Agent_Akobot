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
      Set-Item -Path "env:$key" -Value $val
    }
  }
}
$env:VIRTUAL_ENV = (Join-Path $repo "venv")
$env:PYTHONHOME = ""
$env:PYTHONPATH = ""
$env:PYTHONEXECUTABLE = $py
$env:PATH = (Join-Path $repo "venv\Scripts") + ";" + $env:PATH

# If local ffmpeg is installed, prepend to PATH for STT in worker.
$ffmpegRoot = Join-Path $repo "tools\\ffmpeg"
if (Test-Path $ffmpegRoot) {
  $ffmpegDir = Get-ChildItem -Path $ffmpegRoot -Directory | Select-Object -First 1
  if ($ffmpegDir) {
    $ffmpegBin = Join-Path $ffmpegDir.FullName "bin"
    if (Test-Path $ffmpegBin) {
      $env:PATH = "$ffmpegBin;$env:PATH"
    }
  }
}

# Allow a dedicated override for embeddings provider in Celery worker.
if ($env:CELERY_EMBEDDINGS_PROVIDER) {
  $env:EMBEDDINGS_PROVIDER = $env:CELERY_EMBEDDINGS_PROVIDER
}

# Purge any queued tasks so the worker starts fresh.
try {
  Write-Host "[CELERY] Purging queued tasks before start..."
  & $py -m celery -A app.infra.celery_app.celery_app purge -f
} catch {
  Write-Host "[CELERY] Purge failed; continuing to start worker. Error: $($_.Exception.Message)"
}

# Windows-safe pool to avoid WinError 5 with billiard
& $py -m celery -A app.infra.celery_app.celery_app worker --loglevel=info --pool=solo --concurrency=1
