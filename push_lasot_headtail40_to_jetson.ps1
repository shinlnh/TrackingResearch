param(
    [string]$RemoteHost = "helios",
    [string]$RemoteDatasetDir = "/home/helios/HELIOS/ls/lasot",
    [string]$SequenceFile = "MyECOTracker/jetson/lasot_headtail40_sequences.txt"
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$localDatasetRoot = Join-Path $repoRoot "ls\lasot"
$sequenceFilePath = Join-Path $repoRoot $SequenceFile

if (-not (Test-Path $localDatasetRoot)) {
    throw "Missing local LaSOT dataset root: $localDatasetRoot"
}

if (-not (Test-Path $sequenceFilePath)) {
    throw "Missing sequence file: $sequenceFilePath"
}

$sequences = Get-Content $sequenceFilePath | Where-Object { $_.Trim() } | ForEach-Object { $_.Trim() }

Write-Host "[push-lasot40] host=$RemoteHost"
Write-Host "[push-lasot40] local_dataset_root=$localDatasetRoot"
Write-Host "[push-lasot40] remote_dataset_dir=$RemoteDatasetDir"
Write-Host "[push-lasot40] sequence_count=$($sequences.Count)"

& ssh $RemoteHost "mkdir -p $RemoteDatasetDir"
if ($LASTEXITCODE -ne 0) {
    throw "ssh mkdir failed with exit code $LASTEXITCODE"
}

$index = 0
foreach ($sequence in $sequences) {
    $index += 1
    $parts = $sequence.Split("-", 2)
    if ($parts.Count -ne 2) {
        throw "Invalid sequence name in list: $sequence"
    }

    $className = $parts[0]
    $localSequenceDir = Join-Path $localDatasetRoot (Join-Path $className $sequence)
    if (-not (Test-Path $localSequenceDir)) {
        throw "Missing local sequence directory: $localSequenceDir"
    }

    $remoteClassDir = "$RemoteDatasetDir/$className"
    $remoteSequenceDir = "$remoteClassDir/$sequence"
    $remoteCheck = (& ssh $RemoteHost "if [ -f '$remoteSequenceDir/groundtruth.txt' ] && [ -d '$remoteSequenceDir/img' ]; then echo READY; else echo MISSING; fi").Trim()
    if ($LASTEXITCODE -ne 0) {
        throw "ssh check failed with exit code $LASTEXITCODE"
    }

    if ($remoteCheck -eq "READY") {
        Write-Host ("[push-lasot40] [{0}/{1}] skip {2}" -f $index, $sequences.Count, $sequence)
        continue
    }

    Write-Host ("[push-lasot40] [{0}/{1}] copy {2}" -f $index, $sequences.Count, $sequence)
    & ssh $RemoteHost "mkdir -p '$remoteClassDir'"
    if ($LASTEXITCODE -ne 0) {
        throw "ssh mkdir class dir failed with exit code $LASTEXITCODE"
    }

    & scp -r $localSequenceDir "${RemoteHost}:$remoteClassDir/"
    if ($LASTEXITCODE -ne 0) {
        throw "scp failed for $sequence with exit code $LASTEXITCODE"
    }
}

Write-Host "[push-lasot40] done"
