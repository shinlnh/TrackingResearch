$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repoRoot

$pythonExe = Join-Path $repoRoot "venv312\Scripts\python.exe"
$runner = Join-Path $repoRoot "OtherTracker\MDNet\run_pymdnet_lasot.py"
$sequenceFile = Join-Path $repoRoot "OtherTracker\lasot\lasot936\headtail40_sequences.txt"
$outputDir = Join-Path $repoRoot "OtherTracker\lasot\lasot936\MDNet"
$logPath = Join-Path $outputDir "tracking.log"
$summaryPath = Join-Path $outputDir "summary.csv"

New-Item -ItemType Directory -Force $outputDir | Out-Null

$cleanupTargets = @(
    (Join-Path $outputDir "tracking.log"),
    (Join-Path $outputDir "manifest.csv"),
    (Join-Path $outputDir "summary.csv"),
    (Join-Path $outputDir "sequence_logs"),
    (Join-Path $outputDir "txt_results")
)
foreach ($target in $cleanupTargets) {
    if (Test-Path $target) {
        Remove-Item -LiteralPath $target -Recurse -Force -ErrorAction SilentlyContinue
    }
}

$oldProcs = Get-CimInstance Win32_Process | Where-Object {
    $_.CommandLine -like "*run_pymdnet_lasot.py*" -and $_.CommandLine -like "*headtail40_sequences.txt*"
}
foreach ($proc in $oldProcs) {
    Stop-Process -Id $proc.ProcessId -Force -ErrorAction SilentlyContinue
}

cmd /c "set PYTHONUNBUFFERED=1 && $pythonExe -u $runner --sequence-file $sequenceFile --output-dir $outputDir 2>&1" |
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
