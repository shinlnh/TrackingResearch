$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repoRoot

$pythonExe = Join-Path $repoRoot "venv312\Scripts\python.exe"
$runner = Join-Path $repoRoot "OtherTracker\CSRT\run_csrt_lasot.py"
$sequenceFile = Join-Path $repoRoot "OtherTracker\lasot\lasot936\headtail40_sequences.txt"
$outputDir = Join-Path $repoRoot "OtherTracker\lasot\lasot936\CSRT"
$resultsDir = Join-Path $outputDir "tracking_results\CSRT"
$logPath = Join-Path $outputDir "tracking.log"
$summaryPath = Join-Path $outputDir "summary.csv"

New-Item -ItemType Directory -Force $outputDir | Out-Null
New-Item -ItemType Directory -Force $resultsDir | Out-Null

$cleanupTargets = @(
    (Join-Path $outputDir "tracking.log"),
    (Join-Path $outputDir "manifest.csv"),
    (Join-Path $outputDir "summary.csv")
)
foreach ($target in $cleanupTargets) {
    if (Test-Path $target) {
        Remove-Item -LiteralPath $target -Recurse -Force -ErrorAction SilentlyContinue
    }
}

@'
import pathlib

seq_file = pathlib.Path("OtherTracker/lasot/lasot936/headtail40_sequences.txt")
results_dir = pathlib.Path("OtherTracker/lasot/lasot936/CSRT/tracking_results/CSRT")

removed = 0
for name in [line.strip() for line in seq_file.read_text(encoding="utf-8").splitlines() if line.strip()]:
    for suffix in (".txt", "_time.txt"):
        path = results_dir / f"{name}{suffix}"
        if path.exists():
            path.unlink()
            removed += 1

print(f"Removed {removed} old result files from {results_dir}")
'@ | & $pythonExe -

cmd /c "set PYTHONUNBUFFERED=1 && $pythonExe -u $runner --sequence-file $sequenceFile --output-dir $outputDir --display-name CSRT --tracker-name csrt 2>&1" |
    Tee-Object -FilePath $logPath

$runExitCode = $LASTEXITCODE

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
