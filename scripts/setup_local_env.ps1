# Bootstraps a project-local virtual environment and cache layout.
param(
    # Install requirements after environment creation by default.
    [switch]$InstallDeps = $true
)

# Stop immediately on first command error.
$ErrorActionPreference = "Stop"
# Resolve project paths from script location to avoid cwd issues.
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot ".."))
$VenvPath = Join-Path $ProjectRoot ".venv"
$CacheRoot = Join-Path $ProjectRoot ".cache"
$PipCache = Join-Path $CacheRoot "pip"
$HFHome = Join-Path $CacheRoot "huggingface"
$HFHub = Join-Path $HFHome "hub"
$TransformersCache = Join-Path $HFHome "transformers"
$TorchHome = Join-Path $CacheRoot "torch"
$PyCachePrefix = Join-Path $ProjectRoot ".pycache"
$PipConfig = Join-Path $ProjectRoot "pip.ini"

# Ensure all local cache directories exist inside project root.
New-Item -ItemType Directory -Force -Path $CacheRoot,$PipCache,$HFHome,$HFHub,$TransformersCache,$TorchHome,$PyCachePrefix | Out-Null

if (-not (Test-Path $VenvPath)) {
    # Create isolated environment under D:\ShortsMetaForge\.venv.
    python -m venv $VenvPath
}

if ($InstallDeps) {
    # Install all runtime dependencies into the local virtual environment directly.
    $VenvPython = Join-Path $VenvPath "Scripts\python.exe"
    $env:PIP_CACHE_DIR = $PipCache
    $env:PIP_CONFIG_FILE = $PipConfig
    & $VenvPython -m pip install --upgrade pip
    & $VenvPython -m pip install -r (Join-Path $ProjectRoot "requirements.txt")
}

# Print resolved locations for quick verification.
Write-Host "Local environment ready."
Write-Host "Venv: $VenvPath"
Write-Host "PIP_CACHE_DIR=$PipCache"
Write-Host "HF_HOME=$HFHome"
Write-Host "TORCH_HOME=$TorchHome"
Write-Host "PYTHONPYCACHEPREFIX=$PyCachePrefix"
Write-Host "Run with: .\\.venv\\Scripts\\python.exe main.py"
