param(
    [string]$VenvPath = ".venv_person_pipeline"
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$venvFullPath = Join-Path $scriptDir $VenvPath

if (-not (Test-Path $venvFullPath)) {
    py -3.12 -m venv $venvFullPath
}

$pythonExe = Join-Path $venvFullPath "Scripts\\python.exe"

& $pythonExe -m pip install --upgrade pip
& $pythonExe -m pip install `
    matplotlib `
    "numpy<2" `
    opencv-python `
    pillow==10.2.0 `
    polars `
    psutil `
    pyyaml `
    requests `
    scipy `
    tqdm `
    ultralytics-thop
& $pythonExe -m pip install --force-reinstall torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
& $pythonExe -m pip install --force-reinstall "numpy<2" pillow==10.2.0
& $pythonExe -m pip install ultralytics==8.4.33 facenet-pytorch==2.6.0 --no-deps

Write-Host "Person pipeline environment is ready at $venvFullPath"
