$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$repo = Resolve-Path (Join-Path $root "..")

Write-Host "[BOOTSTRAP] Repo: $repo"

$py = Join-Path $repo "venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
  $py = "python"
}

Write-Host "[BOOTSTRAP] Using Python: $py"

$envPath = Join-Path $repo ".env"
if (Test-Path $envPath) {
  Write-Host "[BOOTSTRAP] Loading .env variables..."
  Get-Content $envPath | ForEach-Object {
    if ($_ -match "^\s*#") { return }
    if ($_ -match "^\s*$") { return }
    $parts = $_ -split "=", 2
    if ($parts.Count -eq 2) {
      $key = $parts[0].Trim()
      $val = $parts[1].Trim()
      Set-Item -Path ("Env:" + $key) -Value $val
    }
  }
}

Write-Host "[BOOTSTRAP] Installing requirements..."
& $py -m pip install -r (Join-Path $repo "requirements.txt")

function Test-Import($module, $label) {
  try {
    & $py -c "import $module; print('${label}: OK')" | Out-Null
    return $true
  } catch {
    Write-Host "[BOOTSTRAP] ${label}: FAIL"
    return $false
  }
}

function Test-TorchImportClean {
  param($pyPath)
  $env:VIRTUAL_ENV = (Join-Path $repo "venv")
  $env:PYTHONHOME = ""
  $env:PYTHONPATH = ""
  $env:PATH = (Join-Path $repo "venv\Scripts") + ";" + $env:PATH
  try {
    & $pyPath -c "import torch; print(torch.__version__); 
import sys
if torch.cuda.is_available():
    _ = torch.randn(1).cuda()
" | Out-Null
    return $true
  } catch {
    return $false
  }
}

$torchOk = $true
$needsReinstall = $false
$hasGPU = $false

try {
  $null = nvidia-smi 2>&1
  if ($LASTEXITCODE -eq 0) { $hasGPU = $true }
} catch {
  $hasGPU = $false
}

if ($hasGPU) {
  Write-Host "[BOOTSTRAP] NVIDIA GPU Detected via nvidia-smi."
} else {
  Write-Host "[BOOTSTRAP] No NVIDIA GPU detected via nvidia-smi."
}

try {
  $torchCudaOutput = & $py -c "import torch; print(str(torch.cuda.is_available()).lower())" 2>&1
  if ($LASTEXITCODE -ne 0) {
    throw "import failed"
  }
  if ($torchCudaOutput -match "true" -and $hasGPU) {
    Write-Host "[BOOTSTRAP] PyTorch CUDA integration check PASSED."
  } elseif ($torchCudaOutput -match "false" -and $hasGPU) {
    Write-Host "[BOOTSTRAP] GPU is present but installed PyTorch cannot use CUDA. Needs reinstall."
    $needsReinstall = $true
  } else {
    Write-Host "[BOOTSTRAP] PyTorch CPU load ok."
  }
} catch {
  Write-Host "[BOOTSTRAP] PyTorch native DLL dependencies failed to load. Needs reinstall."
  $needsReinstall = $true
}

if ($needsReinstall) {
  $torchOk = $false
  if ($env:AUTO_FIX_SYSTEM_DEPS -eq "true") {
    Write-Host "[BOOTSTRAP] AUTO_FIX_SYSTEM_DEPS enabled. Rebuilding PyTorch environment cleanly..."
    Write-Host "[BOOTSTRAP] Purging old/corrupted Torch packages to prevent DLL collisions..."
    & $py -m pip uninstall -y torch torchvision torchaudio | Out-Null
    
    Write-Host "[BOOTSTRAP] Installing clean PyTorch stack (cu121)..."
    & $py -m pip install --no-cache-dir torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
    
    try {
      $torchCudaOutput2 = & $py -c "import torch; print(str(torch.cuda.is_available()).lower())" 2>&1
      if ($torchCudaOutput2 -match "true") {
        Write-Host "[BOOTSTRAP] PyTorch reinstall successful."
        $torchOk = $true
      } else {
        throw "CUDA still unavailable"
      }
    } catch {
      Write-Host "[BOOTSTRAP] PyTorch still failing after reinstall. Falling back to CPU-only torch..."
      & $py -m pip uninstall -y torch torchvision torchaudio | Out-Null
      & $py -m pip install --no-cache-dir torch==2.5.1+cpu torchvision==0.20.1+cpu torchaudio==2.5.1+cpu --index-url https://download.pytorch.org/whl/cpu
      try {
        & $py -c "import torch" | Out-Null
        $torchOk = $true
      } catch {
        $torchOk = $false
      }
    }
  } else {
    Write-Host "[BOOTSTRAP] Fix: reinstall torch with correct cu121 index to match your architecture."
  }
}

# Final safety: ensure a clean-process import works (matches runtime)
if (-not (Test-TorchImportClean $py)) {
  Write-Host "[BOOTSTRAP] Torch still failing in clean process. Falling back to CPU-only build."
  & $py -m pip install --force-reinstall --no-cache-dir torch==2.5.1+cpu torchvision==0.20.1+cpu torchaudio==2.5.1+cpu --index-url https://download.pytorch.org/whl/cpu
}

$ok = $true
$ok = (Test-Import "torch" "PyTorch") -and $ok -and $torchOk
$ok = (Test-Import "transformers" "Transformers") -and $ok
$ok = (Test-Import "sentence_transformers" "SentenceTransformers") -and $ok
$ok = (Test-Import "paddleocr" "PaddleOCR") -and $ok
$ok = (Test-Import "qdrant_client" "Qdrant Client") -and $ok

Write-Host "[BOOTSTRAP] Ensuring Paddle runtime (auto-install if enabled)..."
& $py -c "from app.infra.model_bootstrap import ensure_paddle_runtime; ensure_paddle_runtime()"

Write-Host "[BOOTSTRAP] Pre-downloading AI models (if enabled) to prevent API latency later..."
& $py -c "from app.infra.model_bootstrap import preload_models; preload_models()"

if ($ok) {
  Write-Host "Success Everything is Up & Running - Start using the Product now!!"
  exit 0
}

Write-Host "[BOOTSTRAP] One or more checks failed. See logs above."
exit 1
