$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repoRoot

$pythonExe = Join-Path $repoRoot "venv312\Scripts\python.exe"
$outputDir = Join-Path $repoRoot "OtherTracker\lasot\lasot936\MEEM"
$resultsDir = Join-Path $outputDir "tracking_results\MEEM"
$logPath = Join-Path $outputDir "tracking.log"
$summaryPath = Join-Path $outputDir "summary.csv"

New-Item -ItemType Directory -Force $outputDir | Out-Null
New-Item -ItemType Directory -Force $resultsDir | Out-Null

$cleanupTargets = @(
    (Join-Path $outputDir "tracking.log"),
    (Join-Path $outputDir "summary.csv"),
    (Join-Path $outputDir "meem_matlab_summary.csv")
)
foreach ($target in $cleanupTargets) {
    if (Test-Path $target) {
        Remove-Item -LiteralPath $target -Recurse -Force -ErrorAction SilentlyContinue
    }
}

@'
import pathlib

seq_file = pathlib.Path("OtherTracker/lasot/lasot936/headtail40_sequences.txt")
results_dir = pathlib.Path("OtherTracker/lasot/lasot936/MEEM/tracking_results/MEEM")

removed = 0
for name in [line.strip() for line in seq_file.read_text(encoding="utf-8").splitlines() if line.strip()]:
    for suffix in (".txt", "_time.txt"):
        path = results_dir / f"{name}{suffix}"
        if path.exists():
            path.unlink()
            removed += 1

print(f"Removed {removed} old result files from {results_dir}")
'@ | & $pythonExe -

cmd /c "set PYTHONUNBUFFERED=1 && matlab -batch ""addpath(fullfile(pwd,'OtherTracker','MEEM')); run_meem_lasot_headtail40;"" 2>&1" |
    Tee-Object -FilePath $logPath

$runExitCode = $LASTEXITCODE

@'
import csv
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

repo = Path(".").resolve()
pyroot = repo / "OtherTracker" / "pytracking"
for root in (repo, pyroot):
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

from pytracking.analysis.plot_results import check_and_load_precomputed_results, get_auc_curve, get_prec_curve
from pytracking.evaluation import get_dataset

raw_names = [
    line.strip()
    for line in (repo / "OtherTracker" / "lasot" / "lasot936" / "headtail40_sequences.txt").read_text(encoding="utf-8").splitlines()
    if line.strip()
]
dataset_all = list(get_dataset("lasot"))
by_name = {seq.name: seq for seq in dataset_all}
dataset = [by_name[name] for name in raw_names]

tracker = SimpleNamespace(
    name="MEEM",
    parameter_name="matlab",
    run_id=None,
    display_name="MEEM",
    results_dir=str(repo / "OtherTracker" / "lasot" / "lasot936" / "MEEM" / "tracking_results" / "MEEM"),
)

report_name = "lasot_headtail40_meem_othertracker"
eval_data = check_and_load_precomputed_results(
    [tracker], dataset, report_name, force_evaluation=True, skip_missing_seq=False, verbose=False
)
valid = torch.tensor(eval_data["valid_sequence"], dtype=torch.bool)
auc_curve, auc = get_auc_curve(torch.tensor(eval_data["ave_success_rate_plot_overlap"]), valid)
_, precision = get_prec_curve(torch.tensor(eval_data["ave_success_rate_plot_center"]), valid)

results_dir = Path(tracker.results_dir)
fps_values = []
frames_total = 0
total_time = 0.0
for seq in dataset:
    time_path = results_dir / f"{seq.name}_time.txt"
    times = np.loadtxt(time_path, delimiter="\t")
    times = np.atleast_1d(times).astype(np.float64)
    seq_time = float(np.sum(times))
    frames = int(times.shape[0])
    frames_total += frames
    total_time += seq_time
    fps_values.append(float(frames / seq_time) if seq_time > 0 else float("nan"))

fps_values = np.asarray(fps_values, dtype=np.float64)
summary = {
    "tracker": tracker.display_name,
    "scope": "headtail40",
    "valid_sequences": int(valid.sum().item()),
    "AUC": float(auc[0]),
    "Precision": float(precision[0]),
    "Success50": float(auc_curve[0, 10]),
    "FPS_avg_seq": float(np.nanmean(fps_values)),
    "FPS_median_seq": float(np.nanmedian(fps_values)),
    "FPS_weighted_by_frames": float(frames_total / total_time) if total_time > 0 else float("nan"),
    "total_frames": int(frames_total),
    "total_time_sec": float(total_time),
}

summary_path = repo / "OtherTracker" / "lasot" / "lasot936" / "MEEM" / "summary.csv"
with summary_path.open("w", newline="", encoding="utf-8") as fh:
    writer = csv.DictWriter(fh, fieldnames=list(summary.keys()))
    writer.writeheader()
    writer.writerow(summary)

for key, value in summary.items():
    print(f"{key}={value}")
'@ | & $pythonExe - | ForEach-Object {
    $_
    Add-Content -LiteralPath $logPath -Value $_ -Encoding Unicode
}

if (Test-Path $summaryPath) {
    $summary = Import-Csv -LiteralPath $summaryPath | Select-Object -First 1
    if ($null -ne $summary) {
        $summaryBlock = @(
            ""
            "Summary: LaSOT headtail40"
            "AUC: $($summary.AUC)"
            "Precision: $($summary.Precision)"
            "Success50: $($summary.Success50)"
            "FPS_avg_seq: $($summary.FPS_avg_seq)"
            "FPS_median_seq: $($summary.FPS_median_seq)"
            "FPS_weighted_by_frames: $($summary.FPS_weighted_by_frames)"
        )
        Add-Content -LiteralPath $logPath -Value $summaryBlock -Encoding Unicode
    }
}

exit $runExitCode
