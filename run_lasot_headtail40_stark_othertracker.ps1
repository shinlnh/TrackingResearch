$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repoRoot

$pythonExe = Join-Path $repoRoot "venv312\Scripts\python.exe"
$runner = Join-Path $repoRoot "OtherTracker\Stark\run_stark_benchmark.py"
$outDir = Join-Path $repoRoot "OtherTracker\lasot\lasot936\STARK"
$logPath = Join-Path $outDir "tracking.log"

New-Item -ItemType Directory -Force -Path $outDir | Out-Null

& $pythonExe -u $runner `
    --dataset lasot_headtail40 `
    --variant stark_st2 `
    --config baseline_R101 `
    --out-dir $outDir `
    --display-name "STARK-ST101" `
    --tracker-label STARK `
    --resume 2>&1 |
    Tee-Object -FilePath $logPath
