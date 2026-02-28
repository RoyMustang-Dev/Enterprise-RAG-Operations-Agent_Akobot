$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$repo = Resolve-Path (Join-Path $root "..")
$py = Join-Path $repo "venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }

Write-Host "[API] Starting FastAPI (force venv python)..."
$env:VIRTUAL_ENV = (Join-Path $repo "venv")
$env:PYTHONHOME = ""
$env:PYTHONPATH = ""
$env:PYTHONEXECUTABLE = $py
$env:PATH = (Join-Path $repo "venv\Scripts") + ";" + $env:PATH
$env:TORCH_DISABLE_SHM = "1"

& $py -c "import os,sys, multiprocessing as mp; os.environ['PYTHONEXECUTABLE']=sys.executable; mp.set_executable(sys.executable); import uvicorn; uvicorn.run('app.main:app', host='0.0.0.0', port=8000)"
