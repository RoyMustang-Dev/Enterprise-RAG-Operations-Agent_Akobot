$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$repo = Resolve-Path (Join-Path $root "..")
$py = Join-Path $repo "venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }

# Stop any existing API server on port 8000
try {
  $conn = Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue | Select-Object -First 1
  if ($conn -and $conn.OwningProcess) {
    Stop-Process -Id $conn.OwningProcess -Force -ErrorAction SilentlyContinue
  }
} catch {}

# Stop any existing Celery worker processes started from this repo
try {
  Get-CimInstance Win32_Process |
    Where-Object { $_.CommandLine -match "celery" -and $_.CommandLine -match [Regex]::Escape($repo) } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
} catch {}

# Load .env into the current process so child processes inherit it.
$envPath = Join-Path $repo ".env"
if (Test-Path $envPath) {
  Get-Content $envPath | ForEach-Object {
    $line = $_.Trim()
    if (-not $line) { return }
    if ($line.StartsWith("#")) { return }
    $parts = $line -split "=", 2
    if ($parts.Count -eq 2) {
      $name = $parts[0].Trim()
      $value = $parts[1].Trim()
      [System.Environment]::SetEnvironmentVariable($name, $value, "Process")
    }
  }
}

Write-Host "[STACK] Starting API on port 8000..."
$apiArgs = @("-NoProfile", "-File", (Join-Path $root "start_api.ps1"))
if ($env:API_RELOAD -eq "true") {
  $apiArgs += "--reload"
}
Start-Process -FilePath "powershell.exe" -ArgumentList $apiArgs -WorkingDirectory $repo

if ($env:CELERY_AUTOSTART -eq "true") {
  Write-Host "[STACK] Starting Celery worker..."
  Start-Process -FilePath "powershell.exe" -ArgumentList "-NoProfile", "-File", (Join-Path $root "start_celery_worker.ps1") -WorkingDirectory $repo
}
else {
  Write-Host "[STACK] CELERY_AUTOSTART is false; not starting worker."
}
