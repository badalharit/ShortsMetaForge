# Activates existing project-local virtual environment and cache routing.
$ErrorActionPreference = "Stop"
# Resolve project root relative to this script to stay location-independent.
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot ".."))
$VenvPath = Join-Path $ProjectRoot ".venv"

if (-not (Test-Path $VenvPath)) {
    # Hard fail when setup script was not run yet.
    throw "Virtual environment not found at $VenvPath. Run scripts/setup_local_env.ps1 first."
}

$CacheRoot = Join-Path $ProjectRoot ".cache"
$PipCache = Join-Path $CacheRoot "pip"
$HFHome = Join-Path $CacheRoot "huggingface"
$HFHub = Join-Path $HFHome "hub"
$TransformersCache = Join-Path $HFHome "transformers"
$TorchHome = Join-Path $CacheRoot "torch"
$PyCachePrefix = Join-Path $ProjectRoot ".pycache"
$PipConfig = Join-Path $ProjectRoot "pip.ini"

# Ensure cache folders exist before exporting env vars.
New-Item -ItemType Directory -Force -Path $CacheRoot,$PipCache,$HFHome,$HFHub,$TransformersCache,$TorchHome,$PyCachePrefix | Out-Null

# Route caches and Python bytecode output to project-local directories.
$env:PIP_CACHE_DIR = $PipCache
$env:PIP_CONFIG_FILE = $PipConfig
$env:HF_HOME = $HFHome
$env:HUGGINGFACE_HUB_CACHE = $HFHub
$env:TRANSFORMERS_CACHE = $TransformersCache
$env:TORCH_HOME = $TorchHome
$env:PYTHONPYCACHEPREFIX = $PyCachePrefix

# Activate the project virtual environment in the current shell.
. (Join-Path $VenvPath "Scripts\Activate.ps1")

Write-Host "Activated project-local environment."
