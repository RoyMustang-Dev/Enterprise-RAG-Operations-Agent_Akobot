$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$repo = Resolve-Path (Join-Path $root "..")
$py = Join-Path $repo "venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }

# Load .env into the current process so settings apply when starting API directly.
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

Write-Host "[API] Starting FastAPI (force venv python)..."
$env:VIRTUAL_ENV = (Join-Path $repo "venv")
$env:PYTHONHOME = ""
$env:PYTHONPATH = ""
$env:PYTHONEXECUTABLE = $py
$env:PATH = (Join-Path $repo "venv\Scripts") + ";" + $env:PATH
$env:TORCH_DISABLE_SHM = "1"

$reload = $env:API_RELOAD -eq "true"
& $py -c "import os,sys, multiprocessing as mp; os.environ['PYTHONEXECUTABLE']=sys.executable; mp.set_executable(sys.executable); import uvicorn; uvicorn.run('app.main:app', host='0.0.0.0', port=8000, reload=$reload)"
