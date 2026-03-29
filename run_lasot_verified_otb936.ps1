param(
    [switch]$SkipTracking,
    [switch]$SkipToolkitEval
)

$ErrorActionPreference = "Stop"
if (Get-Variable PSNativeCommandUseErrorActionPreference -ErrorAction SilentlyContinue) {
    $PSNativeCommandUseErrorActionPreference = $false
}

function Show-NewLogLines {
    param(
        [string]$Path,
        [ref]$LineCount,
        [string]$CombinedLogPath,
        [string]$Color = "Gray"
    )

    if (-not (Test-Path $Path)) {
        return
    }

    $allLines = @(Get-Content $Path)
    if ($allLines.Count -le $LineCount.Value) {
        return
    }

    $newLines = @($allLines | Select-Object -Skip $LineCount.Value)
    foreach ($line in $newLines) {
        Write-Host $line -ForegroundColor $Color
    }
    Add-Content -Path $CombinedLogPath -Value $newLines
    $LineCount.Value = $allLines.Count
}

function Invoke-LoggedProcess {
    param(
        [string]$FilePath,
        [string[]]$ArgumentList,
        [string]$WorkingDirectory,
        [string]$CombinedLogPath,
        [string]$StdoutPath,
        [string]$StderrPath
    )

    if (Test-Path $StdoutPath) {
        Remove-Item $StdoutPath -Force
    }
    if (Test-Path $StderrPath) {
        Remove-Item $StderrPath -Force
    }

    $proc = Start-Process `
        -FilePath $FilePath `
        -ArgumentList $ArgumentList `
        -WorkingDirectory $WorkingDirectory `
        -NoNewWindow `
        -PassThru `
        -RedirectStandardOutput $StdoutPath `
        -RedirectStandardError $StderrPath

    $stdoutLineCount = 0
    $stderrLineCount = 0

    while ($true) {
        Show-NewLogLines -Path $StdoutPath -LineCount ([ref]$stdoutLineCount) -CombinedLogPath $CombinedLogPath -Color "Gray"
        Show-NewLogLines -Path $StderrPath -LineCount ([ref]$stderrLineCount) -CombinedLogPath $CombinedLogPath -Color "Yellow"

        $proc.Refresh()
        if ($proc.HasExited) {
            Show-NewLogLines -Path $StdoutPath -LineCount ([ref]$stdoutLineCount) -CombinedLogPath $CombinedLogPath -Color "Gray"
            Show-NewLogLines -Path $StderrPath -LineCount ([ref]$stderrLineCount) -CombinedLogPath $CombinedLogPath -Color "Yellow"
            $proc.WaitForExit()
            return $proc.ExitCode
        }

        Start-Sleep -Milliseconds 500
    }
}

function Write-Status {
    param(
        [string]$Message,
        [string]$Path
    )

    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $line = "[$timestamp] $Message"
    $line | Tee-Object -FilePath $Path -Append
}

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonExe = Join-Path $repoRoot "venv312\Scripts\python.exe"
$matlabExe = "C:\Program Files\MATLAB\R2024b\bin\matlab.exe"

$logDir = Join-Path $repoRoot "MyECOTracker\otb100result"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$prefix = "lasot_eco_verified_otb936_936"
$statusFile = Join-Path $logDir "$prefix.status.txt"
$trackingLog = Join-Path $logDir "$prefix.tracking.log"
$trackingStdoutLog = Join-Path $logDir "$prefix.tracking.stdout.log"
$trackingStderrLog = Join-Path $logDir "$prefix.tracking.stderr.log"
$toolkitLog = Join-Path $logDir "$prefix.toolkit.log"
$toolkitStdoutLog = Join-Path $logDir "$prefix.toolkit.stdout.log"
$toolkitStderrLog = Join-Path $logDir "$prefix.toolkit.stderr.log"

Write-Status "Pipeline start." $statusFile

if (-not $SkipTracking) {
    Write-Status "Tracking phase start: eco verified_otb936 run 936 on LaSOT test set." $statusFile
    Add-Content -Path $trackingLog -Value ("`n==== " + (Get-Date -Format "yyyy-MM-dd HH:mm:ss") + " tracking start ====")

    $trackingExitCode = Invoke-LoggedProcess `
        -FilePath $pythonExe `
        -ArgumentList @(
            (Join-Path $repoRoot "MyECOTracker\pytracking\pytracking\run_experiment.py"),
            "myexperiments",
            "eco_verified_otb936_lasot",
            "--debug",
            "0",
            "--threads",
            "0"
        ) `
        -WorkingDirectory $repoRoot `
        -CombinedLogPath $trackingLog `
        -StdoutPath $trackingStdoutLog `
        -StderrPath $trackingStderrLog

    if ($trackingExitCode -ne 0) {
        Write-Status "Tracking phase failed with exit code $trackingExitCode." $statusFile
        exit $trackingExitCode
    }

    Write-Status "Tracking phase done." $statusFile
}
else {
    Write-Status "Tracking phase skipped by flag." $statusFile
}

$srcResultsDir = Join-Path $repoRoot "MyECOTracker\pytracking\pytracking\tracking_results\eco\verified_otb936_936"
$dstToolkitDir = Join-Path $repoRoot "ls\LaSOT_Evaluation_Toolkit\tracking_results\MyTrackerECO_tracking_result"
$lasotTestSetFile = Join-Path $repoRoot "ls\LaSOT_Evaluation_Toolkit\sequence_evaluation_config\testing_set.txt"
$lasotSeqNames = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase)
Get-Content $lasotTestSetFile | ForEach-Object {
    $name = $_.Trim()
    if ($name) {
        [void]$lasotSeqNames.Add($name)
    }
}

Write-Status "Syncing bbox txt files into LaSOT toolkit format." $statusFile
New-Item -ItemType Directory -Force -Path $dstToolkitDir | Out-Null
Get-ChildItem $dstToolkitDir -Filter *.txt -ErrorAction SilentlyContinue | Remove-Item -Force

Get-ChildItem $srcResultsDir -Filter *.txt | Where-Object {
    $_.Name -notlike "*_time.txt" -and
    $_.Name -notlike "*_object_presence_scores.txt" -and
    $lasotSeqNames.Contains($_.BaseName)
} | Copy-Item -Destination $dstToolkitDir -Force

$copiedCount = (Get-ChildItem $dstToolkitDir -Filter *.txt | Measure-Object).Count
Write-Status "Toolkit sync done. Bounding box files available: $copiedCount." $statusFile

if (-not $SkipToolkitEval) {
    Write-Status "Toolkit evaluation phase start." $statusFile
    Add-Content -Path $toolkitLog -Value ("`n==== " + (Get-Date -Format "yyyy-MM-dd HH:mm:ss") + " toolkit eval start ====")

    Push-Location (Join-Path $repoRoot "ls\LaSOT_Evaluation_Toolkit")
    try {
        $toolkitExitCode = Invoke-LoggedProcess `
            -FilePath $matlabExe `
            -ArgumentList @("-batch", "run_mytrackereco_testset_evaluation") `
            -WorkingDirectory (Get-Location).Path `
            -CombinedLogPath $toolkitLog `
            -StdoutPath $toolkitStdoutLog `
            -StderrPath $toolkitStderrLog

        if ($toolkitExitCode -ne 0) {
            Write-Status "Toolkit evaluation failed with exit code $toolkitExitCode." $statusFile
            exit $toolkitExitCode
        }
    }
    finally {
        Pop-Location
    }

    $summaryCsv = Join-Path $repoRoot "ls\LaSOT_Evaluation_Toolkit\res_fig\mytrackereco_lasot_testset_summary.csv"
    $summaryTxt = Join-Path $repoRoot "ls\LaSOT_Evaluation_Toolkit\res_fig\mytrackereco_lasot_testset_summary.txt"
    if (Test-Path $summaryCsv) {
        Copy-Item $summaryCsv (Join-Path $logDir "$prefix.toolkit_summary.csv") -Force
    }
    if (Test-Path $summaryTxt) {
        Copy-Item $summaryTxt (Join-Path $logDir "$prefix.toolkit_summary.txt") -Force
    }

    Write-Status "Toolkit evaluation phase done." $statusFile
}
else {
    Write-Status "Toolkit evaluation phase skipped by flag." $statusFile
}

Write-Status "Pipeline complete." $statusFile
