param(
    [Parameter(Mandatory = $true)]
    [string]$RemoteHost,

    [Parameter(Mandatory = $true)]
    [string]$RemoteReportDir,

    [Parameter(Mandatory = $true)]
    [string]$LocalDir,

    [int]$PollSeconds = 300
)

$ErrorActionPreference = "Stop"

$LocalDir = [System.IO.Path]::GetFullPath($LocalDir)
$LogPath = Join-Path $LocalDir "pull.log"

New-Item -ItemType Directory -Force -Path $LocalDir | Out-Null

function Write-Log {
    param([string]$Message)

    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    $line | Tee-Object -FilePath $LogPath -Append
}

Write-Log "remote_host=$RemoteHost"
Write-Log "remote_report_dir=$RemoteReportDir"
Write-Log "local_dir=$LocalDir"
Write-Log "poll_seconds=$PollSeconds"

while ($true) {
    $checkCmd = "test -f '$RemoteReportDir/summary.csv' && echo READY || echo WAIT"
    $status = (& ssh $RemoteHost $checkCmd).Trim()

    if ($LASTEXITCODE -ne 0) {
        Write-Log "ssh_check_failed exit_code=$LASTEXITCODE"
    }
    elseif ($status -eq "READY") {
        Write-Log "summary_detected; pulling report files"
        & scp -r "${RemoteHost}:${RemoteReportDir}/*" "$LocalDir/"
        if ($LASTEXITCODE -ne 0) {
            throw "scp failed with exit code $LASTEXITCODE"
        }

        Write-Log "pull_complete"
        break
    }
    else {
        Write-Log "summary_not_ready"
    }

    Start-Sleep -Seconds $PollSeconds
}
