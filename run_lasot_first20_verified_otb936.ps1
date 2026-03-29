$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repoRoot

$pythonExe = Join-Path $repoRoot "venv312\Scripts\python.exe"
$runExperiment = Join-Path $repoRoot "MyECOTracker\pytracking\pytracking\run_experiment.py"
$logDir = Join-Path $repoRoot "MyECOTracker\lasot\result\lasot_first20"
$logPath = Join-Path $logDir "tracking.log"

New-Item -ItemType Directory -Force $logDir | Out-Null
Remove-Item $logPath -ErrorAction SilentlyContinue

# Stop any previous first20 run before starting a clean rerun.
$oldProcs = Get-CimInstance Win32_Process | Where-Object {
    $_.CommandLine -like "*eco_verified_otb936_lasot_first20*"
}
foreach ($proc in $oldProcs) {
    Stop-Process -Id $proc.ProcessId -Force -ErrorAction SilentlyContinue
}

# Delete old result files for the first 20 LaSOT sequences in run 936.
@'
import sys
from pathlib import Path

sys.path.insert(0, str(Path("MyECOTracker/pytracking").resolve()))
from pytracking.experiments.myexperiments import eco_verified_otb936_lasot_first20

_, dataset = eco_verified_otb936_lasot_first20()
base = Path("MyECOTracker/pytracking/pytracking/tracking_results/eco/verified_otb936_936")

removed = 0
for seq in dataset:
    for suffix in (".txt", "_time.txt", "_object_presence_scores.txt"):
        path = base / f"{seq.name}{suffix}"
        if path.exists():
            path.unlink()
            removed += 1

print(f"Removed {removed} old result files from {base}")
'@ | & $pythonExe -

cmd /c "set PYTHONUNBUFFERED=1 && set PYTHONWARNINGS=ignore::FutureWarning && $pythonExe -u $runExperiment myexperiments eco_verified_otb936_lasot_first20 --debug 0 --threads 0 2>&1" |
    Tee-Object -FilePath $logPath

exit $LASTEXITCODE
