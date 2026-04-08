param(
    [string]$RemoteHost = "jetson-nano",
    [string]$RemoteDir = "~/HELIOS"
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$sourceDir = Join-Path $repoRoot "MyECOTracker"
if (-not (Test-Path $sourceDir)) {
    throw "Missing source directory: $sourceDir"
}

$tempArchive = Join-Path $env:TEMP ("MyECOTracker_code_{0}.tar.gz" -f (Get-Date -Format "yyyyMMdd_HHmmss"))
$remoteArchive = "~/myecotracker_code.tar.gz"

Write-Host "[push] source=$sourceDir"
Write-Host "[push] host=$RemoteHost"
Write-Host "[push] remote_dir=$RemoteDir"
Write-Host "[push] temp_archive=$tempArchive"

Push-Location $repoRoot
try {
    & tar.exe --exclude=MyECOTracker/otb100result --exclude=MyECOTracker/.venv -czf $tempArchive MyECOTracker
    if ($LASTEXITCODE -ne 0) {
        throw "tar failed with exit code $LASTEXITCODE"
    }

    & scp $tempArchive "${RemoteHost}:${remoteArchive}"
    if ($LASTEXITCODE -ne 0) {
        throw "scp failed with exit code $LASTEXITCODE"
    }

    $remoteCmd = "mkdir -p $RemoteDir && tar -xzf $remoteArchive -C $RemoteDir && find $RemoteDir/MyECOTracker/jetson -type f -name '*.sh' -exec sed -i 's/\r$//' {} + && rm -f $remoteArchive"
    & ssh $RemoteHost $remoteCmd
    if ($LASTEXITCODE -ne 0) {
        throw "ssh extract failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
    Remove-Item -LiteralPath $tempArchive -ErrorAction Ignore
}

Write-Host "[push] done"
Write-Host "[push] extracted_to=$RemoteDir/MyECOTracker"
