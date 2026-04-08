$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$venvPath = Join-Path $PSScriptRoot ".venv_face_training"

if (-not (Test-Path -LiteralPath $venvPath)) {
    python -m venv $venvPath
}

$pythonExe = Join-Path $venvPath "Scripts\python.exe"

& $pythonExe -m pip install --upgrade pip setuptools wheel
& $pythonExe -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
& $pythonExe -m pip install datasets huggingface_hub httpx hf-xet typer tqdm scikit-learn pandas pyarrow requests aiohttp multiprocess==0.70.18 dill==0.4.0 python-dateutil pytz tzdata pyyaml
& $pythonExe -m pip install facenet-pytorch==2.6.0 --no-deps

Write-Host "Face training environment ready at $venvPath"
