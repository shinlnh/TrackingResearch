param(
    [Parameter(Mandatory = $true)]
    [string]$Tracker,

    [int]$StartIndex = 1,
    [int]$EndIndex = 40,

    [switch]$Smoke,
    [switch]$KeepExisting
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repoRoot

$pythonExe = Join-Path $repoRoot "venv312\Scripts\python.exe"
$sequenceFile = Join-Path $repoRoot "OtherTracker\lasot\lasot936\headtail40_sequences.txt"
$outputDir = Join-Path $repoRoot ("OtherTracker\lasot\lasot936\" + $Tracker)
$resultsDir = Join-Path $outputDir ("tracking_results\" + $Tracker)
$logPath = Join-Path $outputDir "tracking.log"
$summaryPath = Join-Path $outputDir "summary.csv"
$postprocess = Join-Path $repoRoot "OtherTracker\tools\postprocess_lasot_headtail40_tracker.py"

if ($Smoke) {
    $StartIndex = 1
    $EndIndex = 1
}

New-Item -ItemType Directory -Force $outputDir | Out-Null
New-Item -ItemType Directory -Force $resultsDir | Out-Null

if (-not $KeepExisting) {
    foreach ($target in @($logPath, $summaryPath)) {
        if (Test-Path $target) {
            Remove-Item -LiteralPath $target -Recurse -Force -ErrorAction SilentlyContinue
        }
    }

    @'
import pathlib
import sys

tracker = sys.argv[1]
start_index = int(sys.argv[2])
end_index = int(sys.argv[3])
seq_file = pathlib.Path("OtherTracker/lasot/lasot936/headtail40_sequences.txt")
results_dir = pathlib.Path("OtherTracker/lasot/lasot936") / tracker / "tracking_results" / tracker
names = [line.strip() for line in seq_file.read_text(encoding="utf-8").splitlines() if line.strip()]
start_index = max(1, start_index)
end_index = min(len(names), end_index)
removed = 0
for name in names[start_index - 1:end_index]:
    for suffix in (".txt", "_time.txt"):
        path = results_dir / f"{name}{suffix}"
        if path.exists():
            path.unlink()
            removed += 1
print(f"Removed {removed} old result files from {results_dir}")
'@ | & $pythonExe - $Tracker $StartIndex $EndIndex
}

$matlabCmd = "addpath(fullfile(pwd,'OtherTracker','tools')); run_verified_lasot_headtail40('$Tracker', '', '', '', $StartIndex, $EndIndex);"
cmd /c "set PYTHONUNBUFFERED=1 && matlab -batch ""$matlabCmd"" 2>&1" |
    Tee-Object -FilePath $logPath

$runExitCode = $LASTEXITCODE
if ($runExitCode -ne 0) {
    exit $runExitCode
}

if (-not $Smoke -and $StartIndex -eq 1 -and $EndIndex -eq 40) {
    & $pythonExe $postprocess `
        --repo-root $repoRoot `
        --tracker-name $Tracker `
        --results-dir $resultsDir `
        --summary-path $summaryPath `
        --sequence-file $sequenceFile | ForEach-Object {
            $_
            Add-Content -LiteralPath $logPath -Value $_ -Encoding Unicode
        }
}

exit 0
