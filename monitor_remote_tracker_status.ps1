param(
    [Parameter(Mandatory = $true)]
    [string]$RemoteHost,

    [Parameter(Mandatory = $true)]
    [string]$RemoteReportDir,

    [Parameter(Mandatory = $true)]
    [string]$RemoteResultsDir,

    [Parameter(Mandatory = $true)]
    [string]$RemoteSequenceFile,

    [Parameter(Mandatory = $true)]
    [string]$RemoteProcessPattern,

    [Parameter(Mandatory = $true)]
    [string]$LocalLogPath,

    [int]$PollSeconds = 120
)

$ErrorActionPreference = "Continue"

$LocalLogPath = [System.IO.Path]::GetFullPath($LocalLogPath)
$LocalLogDir = Split-Path -Parent $LocalLogPath
New-Item -ItemType Directory -Force -Path $LocalLogDir | Out-Null

function Write-Log {
    param([string]$Message)

    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    $line | Tee-Object -FilePath $LocalLogPath -Append | Out-Null
}

Write-Log "monitor_started"
Write-Log "remote_host=$RemoteHost"
Write-Log "remote_report_dir=$RemoteReportDir"
Write-Log "remote_results_dir=$RemoteResultsDir"
Write-Log "remote_sequence_file=$RemoteSequenceFile"
Write-Log "poll_seconds=$PollSeconds"

while ($true) {
    $summaryStatus = (& ssh $RemoteHost "test -f '$RemoteReportDir/summary.csv' && echo READY || echo WAIT").Trim()
    $processInfo = ((& ssh $RemoteHost "ps -ef | grep -E '$RemoteProcessPattern' | grep -v grep || true") | Out-String).Trim()
    $progressInfo = ((& ssh $RemoteHost "python3 - <<'PY'
from pathlib import Path

sequence_file = Path('$RemoteSequenceFile')
results_dir = Path('$RemoteResultsDir')

sequences = [line.strip() for line in sequence_file.read_text().splitlines() if line.strip()]
finished = [seq for seq in sequences if (results_dir / f'{seq}_time.txt').exists()]

print('finished={}/{}'.format(len(finished), len(sequences)))
if finished:
    print('last={}'.format(finished[-1]))
PY") | Out-String).Trim()

    $processState = if ([string]::IsNullOrWhiteSpace($processInfo)) { "down" } else { "up" }
    $progressLine = $progressInfo -replace "\r?\n", " | "
    Write-Log ("summary={0} process={1} {2}" -f $summaryStatus, $processState, $progressLine)

    if ($summaryStatus -eq "READY") {
        Write-Log "monitor_finished_summary_ready"
        break
    }

    if ($processState -eq "down") {
        Write-Log "monitor_finished_process_missing"
        break
    }

    Start-Sleep -Seconds $PollSeconds
}
