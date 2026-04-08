# Run MyTracker on Jetson Nano (Windows PowerShell)
# This script runs ECO Tracker (MyTrackerECO) on LaSOT dataset via SSH
# Usage: .\run_mytracker_jetson.ps1 [-JetsonHost helios@192.168.1.162] [-JetsonPort 22] [-Password "041209"]

param(
    [string]$JetsonHost = "helios@192.168.1.162",
    [int]$JetsonPort = 22,
    [string]$Password = "041209",
    [string]$RemoteProjectDir = "~/HELIOS/TransTResearch",
    [ValidateSet('main', 'run_update')]
    [string]$Profile = "main"
)

function Write-Header {
    param([string]$Text)
    Write-Host "`n===========================================" -ForegroundColor Cyan
    Write-Host $Text -ForegroundColor Cyan
    Write-Host "==========================================" -ForegroundColor Cyan
}

Write-Header "Running MyTracker on Jetson Nano"

Write-Host "Configuration:" -ForegroundColor Yellow
Write-Host "  Host: $JetsonHost"
Write-Host "  Port: $JetsonPort"
Write-Host "  Remote Dir: $RemoteProjectDir"
Write-Host "  Tracker: ECO (MyTrackerECO)"
Write-Host "  Profile: $Profile"
Write-Host "  Dataset: LaSOT (head 20 + tail 20 sequences)"
Write-Host ""

switch ($Profile) {
    "main" {
        $Experiment = "eco_verified_otb936_lasot_headtail40"
        $ParamDesc = "verified_otb936_main (run 953)"
    }
    "run_update" {
        $Experiment = "eco_verified_otb936_run_update_lasot_headtail40"
        $ParamDesc = "verified_otb936_run_update (run 954)"
    }
}

# Create remote bash command
$remoteCmd = @"
set -e
cd $RemoteProjectDir
export PYTHONUNBUFFERED=1
export PYTHONWARNINGS='ignore::FutureWarning'

# Activate virtual environment if exists
if [ -f venv/bin/activate ]; then
    source venv/bin/activate
elif [ -f venv312/bin/activate ]; then
    source venv312/bin/activate
fi

# Run MyTracker
echo 'Starting MyTracker (ECO) on LaSOT dataset...'
echo 'Parameters: $ParamDesc, Dataset: LaSOT'
echo ''

python MyECOTracker/pytracking/pytracking/run_experiment.py \
    myexperiments $Experiment \
    --debug 0 \
    --threads 0

echo 'MyTracker run completed!'
"@

Write-Host "Connecting to Jetson Nano via SSH..." -ForegroundColor Green
Write-Host "Note: Make sure you have RSA key authentication configured" -ForegroundColor Yellow
Write-Host ""

try {
    # Use ssh command via System.Diagnostics.Process for better output handling
    $proc = [System.Diagnostics.Process]::new()
    $proc.StartInfo.FileName = "ssh"
    $proc.StartInfo.Arguments = "-p $JetsonPort $JetsonHost bash -s"
    $proc.StartInfo.UseShellExecute = $false
    $proc.StartInfo.RedirectStandardInput = $true
    $proc.StartInfo.RedirectStandardOutput = $true
    $proc.StartInfo.RedirectStandardError = $true
    
    $proc.Start() | Out-Null
    
    # Send command to stdin
    $proc.StandardInput.WriteLine($remoteCmd)
    $proc.StandardInput.Close()
    
    # Read output
    $stdout = $proc.StandardOutput.ReadToEnd()
    $stderr = $proc.StandardError.ReadToEnd()
    
    $proc.WaitForExit()
    
    if ($stdout) { Write-Host $stdout }
    if ($stderr) { Write-Host $stderr -ForegroundColor Red }
    
    if ($proc.ExitCode -ne 0) {
        Write-Host "Error: SSH command failed with exit code $($proc.ExitCode)" -ForegroundColor Red
        exit $proc.ExitCode
    }
}
catch {
    Write-Host "Error: Failed to connect to Jetson Nano" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    exit 1
}

Write-Header "MyTracker Execution Completed!"
Write-Host "Results should be saved in: $RemoteProjectDir/MyECOTracker/lasot/result/" -ForegroundColor Green
