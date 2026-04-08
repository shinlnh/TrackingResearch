# Setup SSH Keys for Jetson Nano (Windows PowerShell)
# This script generates SSH key pair and adds public key to Jetson Nano
# Usage: .\setup_ssh_keys.ps1 [-JetsonHost 192.168.1.162] [-JetsonPort 22] [-Password "041209"]

param(
    [string]$JetsonHost = "192.168.1.162",
    [int]$JetsonPort = 22,
    [string]$Password = "041209",
    [string]$KeyName = "jetson_rsa",
    [string]$KeyDir = "$env:USERPROFILE\.ssh"
)

function Write-Header {
    param([string]$Text)
    Write-Host "`n========================================" -ForegroundColor Cyan
    Write-Host $Text -ForegroundColor Cyan
    Write-Host "========================================`n" -ForegroundColor Cyan
}

Write-Header "SSH Key Setup for Jetson Nano"

# Check if .ssh directory exists
if (!(Test-Path $KeyDir)) {
    Write-Host "Creating .ssh directory..." -ForegroundColor Yellow
    New-Item -ItemType Directory -Path $KeyDir -Force | Out-Null
}

$KeyPath = Join-Path $KeyDir $KeyName
$PubKeyPath = "$KeyPath.pub"

# Check if key already exists
if ((Test-Path $KeyPath) -or (Test-Path $PubKeyPath)) {
    Write-Host "RSA key already exists at: $KeyPath" -ForegroundColor Yellow
    $answer = Read-Host "Do you want to regenerate it? (y/n)"
    if ($answer -ne 'y') {
        Write-Host "Skipping key generation" -ForegroundColor Green
        $UseExisting = $true
    } else {
        Write-Host "Removing old keys..." -ForegroundColor Yellow
        Remove-Item -Path $KeyPath -Force -ErrorAction SilentlyContinue
        Remove-Item -Path $PubKeyPath -Force -ErrorAction SilentlyContinue
        $UseExisting = $false
    }
} else {
    $UseExisting = $false
}

# Generate new key if needed
if (!$UseExisting) {
    Write-Host "Generating SSH RSA key pair..." -ForegroundColor Green
    Write-Host "Key path: $KeyPath"
    Write-Host "This may take a moment..."
    
    # Use ssh-keygen with no passphrase
    & ssh-keygen -t rsa -b 4096 -f $KeyPath -N '""' -C "helios@$JetsonHost" | Out-Null
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ SSH key pair generated successfully!" -ForegroundColor Green
    } else {
        Write-Host "✗ Failed to generate SSH key" -ForegroundColor Red
        exit 1
    }
}

# Set correct permissions
Write-Host "`nSetting file permissions..." -ForegroundColor Green
icacls $KeyPath /inheritance:r /grant:r "$env:USERNAME`:F" | Out-Null
icacls $PubKeyPath /inheritance:r /grant:r "$env:USERNAME`:F" | Out-Null

Write-Host "✓ Permissions set correctly" -ForegroundColor Green

# Display public key
Write-Host "`nPublic Key Content:" -ForegroundColor Yellow
Write-Host "─────────────────────────────────────────" -ForegroundColor Yellow
Get-Content $PubKeyPath
Write-Host "─────────────────────────────────────────`n" -ForegroundColor Yellow

# Add public key to Jetson Nano
Write-Host "Adding public key to Jetson Nano..." -ForegroundColor Green
Write-Host "Jetson: helios@$JetsonHost`:$JetsonPort" -ForegroundColor Cyan

# Read public key content
$PublicKeyContent = Get-Content $PubKeyPath

# Create remote command to add public key
$RemoteCmd = @"
mkdir -p ~/.ssh
chmod 700 ~/.ssh

# Check if key already exists
if grep -q "$PublicKeyContent" ~/.ssh/authorized_keys 2>/dev/null; then
    echo "Key already exists in authorized_keys"
else
    echo "$PublicKeyContent" >> ~/.ssh/authorized_keys
    chmod 600 ~/.ssh/authorized_keys
    echo "Public key added to authorized_keys"
fi
"@

# Execute via SSH (with password input)
Write-Host "Enter Jetson password (041209) when prompted:" -ForegroundColor Yellow
Write-Host "(This is the last time you'll need to enter password!)`n" -ForegroundColor Yellow

try {
    $pubKey = Get-Content $PubKeyPath
    
    # Use SSH to run the command
    $RemoteCmd | & ssh -p $JetsonPort "helios@$JetsonHost" bash -s
    
    if ($LASTEXITCODE -eq 0) {
        Write-Header "✓ SSH Key Setup Completed Successfully!"
        Write-Host "You can now SSH without password:" -ForegroundColor Green
        Write-Host "  ssh -p $JetsonPort helios@$JetsonHost`n" -ForegroundColor Cyan
        Write-Host "Or use in scripts:" -ForegroundColor Green
        Write-Host "  ssh -i '$KeyPath' -p $JetsonPort helios@$JetsonHost`n" -ForegroundColor Cyan
    } else {
        Write-Host "Warning: SSH key addition may have failed" -ForegroundColor Yellow
    }
}
catch {
    Write-Host "Error during SSH operation" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
}

# Update local SSH config for convenience
Write-Host "`nUpdating SSH config..." -ForegroundColor Green
$SshConfig = Join-Path $KeyDir "config"

$JetsonConfig = @"
Host jetson
    HostName $JetsonHost
    Port $JetsonPort
    User helios
    IdentityFile ~/.ssh/$KeyName
    IdentitiesOnly yes
"@

if (Test-Path $SshConfig) {
    $Content = Get-Content $SshConfig
    if ($Content -notmatch "Host jetson") {
        Add-Content $SshConfig "`n$JetsonConfig"
        Write-Host "✓ Added 'jetson' host to SSH config" -ForegroundColor Green
    } else {
        Write-Host "✓ 'jetson' host already in SSH config" -ForegroundColor Green
    }
} else {
    Set-Content $SshConfig $JetsonConfig
    Write-Host "✓ Created SSH config with 'jetson' host" -ForegroundColor Green
}

Write-Header "Setup Complete!"
Write-Host "You can now connect using:" -ForegroundColor Green
Write-Host "  .\run_mytracker_jetson.ps1" -ForegroundColor Cyan
Write-Host "`nOr directly:" -ForegroundColor Green
Write-Host "  ssh jetson" -ForegroundColor Cyan
Write-Host "  ssh -p $JetsonPort helios@$JetsonHost`n" -ForegroundColor Cyan
