$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$repo = Resolve-Path (Join-Path $root "..")
$dest = Join-Path $repo "tools\\ffmpeg"
$zipPath = Join-Path $repo "tools\\ffmpeg.zip"

if (-not (Test-Path $dest)) {
  New-Item -ItemType Directory -Force -Path $dest | Out-Null
}

$url = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
Write-Host "[FFMPEG] Downloading from $url ..."
Invoke-WebRequest -Uri $url -OutFile $zipPath -UseBasicParsing

Write-Host "[FFMPEG] Extracting..."
Expand-Archive -Path $zipPath -DestinationPath $dest -Force
Remove-Item $zipPath -Force -ErrorAction SilentlyContinue

# Add to PATH for current process
$bin = Get-ChildItem -Path $dest -Directory | Select-Object -First 1
if ($bin) {
  $binPath = Join-Path $bin.FullName "bin"
  if (Test-Path $binPath) {
    $env:PATH = "$binPath;$env:PATH"
    Write-Host "[FFMPEG] Installed at $binPath"
  } else {
    Write-Host "[FFMPEG] Bin path not found in extracted archive."
  }
} else {
  Write-Host "[FFMPEG] Extracted folder not found."
}
