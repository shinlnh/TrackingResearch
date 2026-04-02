param(
    [ValidateSet(
        'otb_only_mytracker_success_plot',
        'otb_only_mytracker_fps_avg',
        'otb_full_tracker_success_plot',
        'otb_full_tracker_fps_avg',
        'lasot_only_mytracker_success_plot',
        'lasot_only_mytracker_fps_avg',
        'lasot_full_tracker_success_plot',
        'lasot_full_tracker_fps_avg'
    )]
    [string]$Target,

    [ValidateSet('otb', 'lasot')]
    [string]$Dataset,

    [ValidateSet('only_mytracker', 'full_tracker')]
    [string]$Scope,

    [ValidateSet('success_plot', 'fps_avg')]
    [string]$Metric
)

$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$configPath = Join-Path $repoRoot 'overall_result\video\video_plot_config.json'
if (-not (Test-Path $configPath)) {
    throw "Missing config file: $configPath"
}

if ($Target) {
    $parts = $Target.Split('_')
    $Dataset = $parts[0]
    $Scope = '{0}_{1}' -f $parts[1], $parts[2]
    if ($parts.Length -eq 5) {
        $Metric = '{0}_{1}' -f $parts[3], $parts[4]
    } else {
        $Metric = $parts[3]
    }
}

if (-not $Dataset -or -not $Scope -or -not $Metric) {
    throw "Provide either -Target or the full set -Dataset/-Scope/-Metric."
}

$config = Get-Content $configPath -Raw | ConvertFrom-Json
$relativeOutDir = $config.$Dataset.$Scope.$Metric
if (-not $relativeOutDir) {
    throw "No output directory configured for dataset=$Dataset scope=$Scope metric=$Metric"
}

$outDir = Join-Path $repoRoot $relativeOutDir
New-Item -ItemType Directory -Path $outDir -Force | Out-Null
$outDirMatlab = ($outDir -replace '\\', '/')

Write-Host "[video] dataset=$Dataset scope=$Scope metric=$Metric"
Write-Host "[video] out_dir=$outDir"

if ($Metric -eq 'success_plot') {
    if ($Dataset -eq 'otb') {
        $matlabCmd = "cd(fullfile(pwd,'otb','otb-toolkit')); run_OPE_video_otb('$Scope', '$outDirMatlab');"
    } else {
        $matlabCmd = "cd(fullfile(pwd,'ls','LaSOT_Evaluation_Toolkit')); run_lasot_video_ope('$Scope', '$outDirMatlab');"
    }
    matlab -batch $matlabCmd
} else {
    venv312\Scripts\python.exe .\OtherTracker\tools\plot_video_fps_avg.py --dataset $Dataset --scope $Scope --out-dir $outDir
}

Get-ChildItem $outDir | Select-Object Name, Length, LastWriteTime | Format-Table -AutoSize
