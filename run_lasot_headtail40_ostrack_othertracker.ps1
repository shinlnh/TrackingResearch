$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repoRoot

$pythonExe = Join-Path $repoRoot "venv312\Scripts\python.exe"
$runner = Join-Path $repoRoot "OtherTracker\OSTrack\run_ostrack_benchmark.py"
$outDir = Join-Path $repoRoot "OtherTracker\lasot\lasot936\OSTrack"
$logPath = Join-Path $outDir "tracking.log"

New-Item -ItemType Directory -Force -Path $outDir | Out-Null

& $pythonExe -u $runner `
    --dataset lasot_headtail40 `
    --config vitb_384_mae_ce_32x4_ep300 `
    --out-dir $outDir `
    --display-name "OSTrack-384" `
    --tracker-label OSTrack `
    --resume 2>&1 |
    Tee-Object -FilePath $logPath
