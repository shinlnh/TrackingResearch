param(
    [int]$Tail = 20,
    [switch]$Follow
)

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$logDir = Join-Path $repoRoot "MyECOTracker\otb100result"
$prefix = "lasot_eco_verified_otb936_936"

$statusFile = Join-Path $logDir "$prefix.status.txt"
$trackingLog = Join-Path $logDir "$prefix.tracking.log"
$toolkitLog = Join-Path $logDir "$prefix.toolkit.log"
$summaryCsv = Join-Path $logDir "$prefix.toolkit_summary.csv"
$resultsDir = Join-Path $repoRoot "MyECOTracker\pytracking\pytracking\tracking_results\eco\verified_otb936_936"
$lasotTestSetFile = Join-Path $repoRoot "ls\LaSOT_Evaluation_Toolkit\sequence_evaluation_config\testing_set.txt"

$lasotSeqNames = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase)
if (Test-Path $lasotTestSetFile) {
    Get-Content $lasotTestSetFile | ForEach-Object {
        $name = $_.Trim()
        if ($name) {
            [void]$lasotSeqNames.Add($name)
        }
    }
}

$done = 0
if (Test-Path $resultsDir) {
    $done = (Get-ChildItem $resultsDir -Filter *.txt | Where-Object {
        $_.Name -notlike "*_time.txt" -and
        $_.Name -notlike "*_object_presence_scores.txt" -and
        $lasotSeqNames.Contains($_.BaseName)
    } | Measure-Object).Count
}

Write-Host ("Completed bbox files: {0} / 280" -f $done)

if (Test-Path $statusFile) {
    Write-Host ""
    Write-Host "Status:"
    Get-Content $statusFile -Tail $Tail
}

if (Test-Path $trackingLog) {
    Write-Host ""
    Write-Host "Tracking log tail:"
    Get-Content $trackingLog -Tail $Tail
}

if (Test-Path $toolkitLog) {
    Write-Host ""
    Write-Host "Toolkit log tail:"
    Get-Content $toolkitLog -Tail $Tail
}

if (Test-Path $summaryCsv) {
    Write-Host ""
    Write-Host "Toolkit summary:"
    Get-Content $summaryCsv
}

Write-Host ""
Write-Host "Tracking log:"
if ($Follow) {
    Get-Content $trackingLog -Wait -Tail $Tail
}
elseif (Test-Path $trackingLog) {
    Get-Content $trackingLog -Tail $Tail
}
