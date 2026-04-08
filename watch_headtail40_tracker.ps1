param(
    [Parameter(Mandatory = $true)]
    [string]$Tracker,

    [Parameter(Mandatory = $true)]
    [int]$RunnerPid,

    [int]$PollSeconds = 60,
    [int]$StallSeconds = 1800
)

$ErrorActionPreference = 'Continue'

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$outputDir = Join-Path $repoRoot ("OtherTracker\lasot\lasot936\" + $Tracker)
$resultsDir = Join-Path $outputDir ("tracking_results\" + $Tracker)
$logPath = Join-Path $outputDir "tracking.log"
$summaryPath = Join-Path $outputDir "summary.csv"
$watchLog = Join-Path $outputDir "watchdog.log"

New-Item -ItemType Directory -Force $outputDir | Out-Null

function Write-WatchLog {
    param([string]$Message)

    $line = "[{0}] {1}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $Message
    Add-Content -LiteralPath $watchLog -Value $line
}

function Get-DescendantPids {
    param([int]$RootPid)

    $all = Get-CimInstance Win32_Process | Select-Object ProcessId, ParentProcessId
    $queue = New-Object 'System.Collections.Generic.Queue[int]'
    $seen = New-Object 'System.Collections.Generic.HashSet[int]'
    $desc = @()

    $queue.Enqueue($RootPid)
    [void]$seen.Add($RootPid)

    while ($queue.Count -gt 0) {
        $current = $queue.Dequeue()
        $children = $all | Where-Object { $_.ParentProcessId -eq $current }
        foreach ($child in $children) {
            $childPid = [int]$child.ProcessId
            if ($seen.Add($childPid)) {
                $desc += $childPid
                $queue.Enqueue($childPid)
            }
        }
    }

    return $desc
}

$prevResultCount = -1
$prevLogWrite = $null
$lastProgress = Get-Date
$prevCpuByPid = @{}

Write-WatchLog ("watcher started tracker={0} runner_pid={1} poll_seconds={2} stall_seconds={3}" -f $Tracker, $RunnerPid, $PollSeconds, $StallSeconds)

while ($true) {
    $runner = Get-Process -Id $RunnerPid -ErrorAction SilentlyContinue
    $runnerAlive = $null -ne $runner

    $resultFiles = @(Get-ChildItem -LiteralPath $resultsDir -File -ErrorAction SilentlyContinue).Count
    $completedSequences = [math]::Floor($resultFiles / 2)

    $logWrite = $null
    $lastLogLine = 'NO_TRACKING_LOG'
    if (Test-Path $logPath) {
        $logWrite = (Get-Item $logPath).LastWriteTime
        $lastLogLine = Get-Content $logPath -Tail 1 -ErrorAction SilentlyContinue
    }

    if ($resultFiles -ne $prevResultCount -or ($null -ne $logWrite -and ($null -eq $prevLogWrite -or $logWrite -gt $prevLogWrite))) {
        $lastProgress = Get-Date
    }

    $prevResultCount = $resultFiles
    $prevLogWrite = $logWrite

    $matlabSummary = 'none'
    if ($runnerAlive) {
        $descendantPids = Get-DescendantPids -RootPid $RunnerPid
        if ($descendantPids.Count -gt 0) {
            $descendantProcs = Get-Process -Id $descendantPids -ErrorAction SilentlyContinue
            $matlabProcs = $descendantProcs | Where-Object { $_.ProcessName -eq 'MATLAB' } | Sort-Object Id
            if ($matlabProcs) {
                $parts = foreach ($proc in $matlabProcs) {
                    $prevCpu = $prevCpuByPid[$proc.Id]
                    $cpuDelta = if ($null -eq $prevCpu) { 0 } else { [math]::Round(($proc.CPU - $prevCpu), 3) }
                    $prevCpuByPid[$proc.Id] = $proc.CPU
                    "pid={0},threads={1},cpu={2},delta={3}" -f $proc.Id, $proc.Threads.Count, [math]::Round($proc.CPU, 3), $cpuDelta
                }
                $matlabSummary = ($parts -join '; ')
            }
        }
    }

    $stallSecondsNow = [math]::Round(((Get-Date) - $lastProgress).TotalSeconds)
    Write-WatchLog ("runner_alive={0} completed_sequences={1} result_files={2} stall_seconds={3} last_log='{4}' matlab=[{5}]" -f $runnerAlive, $completedSequences, $resultFiles, $stallSecondsNow, $lastLogLine, $matlabSummary)

    if (Test-Path $summaryPath) {
        Write-WatchLog ("summary detected at {0}" -f $summaryPath)
        break
    }

    if (-not $runnerAlive) {
        if (Test-Path $logPath) {
            $tail = (Get-Content $logPath -Tail 12 -ErrorAction SilentlyContinue) -join ' || '
            Write-WatchLog ("runner exited before summary; tail={0}" -f $tail)
        } else {
            Write-WatchLog 'runner exited before summary and tracking.log is missing'
        }
        break
    }

    if ($stallSecondsNow -ge $StallSeconds) {
        Write-WatchLog ("stall warning: no new log/result progress for {0} seconds" -f $stallSecondsNow)
    }

    Start-Sleep -Seconds $PollSeconds
}
