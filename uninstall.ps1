# Aglibol Agent Official Uninstaller for Windows PowerShell
# Usage: irm https://raw.githubusercontent.com/xushiexpresso15/AglibolAgent/main/uninstall.ps1 | iex
# Pass -Purge to also remove all persistent sessions and checkpoints ($HOME\.aglibol)

param (
    [switch]$Purge
)

$ErrorActionPreference = "Continue"

Write-Host "=== Aglibol Agent Uninstaller ===" -ForegroundColor Cyan

$InstallDir = Join-Path $HOME ".aglibol"
$BinDir = Join-Path $InstallDir "bin"
$EnvDir = Join-Path $InstallDir "env"

# 1. Uninstall via uv if present
$hasUv = Get-Command "uv" -ErrorAction SilentlyContinue
if ($hasUv) {
    $uvList = & uv tool list 2>$null
    if ($uvList -match "aglibol-agent") {
        Write-Host "-> Removing uv tool 'aglibol-agent'..." -ForegroundColor Cyan
        & uv tool uninstall aglibol-agent
    }
}

# 2. Uninstall via pipx if present
$hasPipx = Get-Command "pipx" -ErrorAction SilentlyContinue
if ($hasPipx) {
    $pipxList = & pipx list 2>$null
    if ($pipxList -match "aglibol-agent") {
        Write-Host "-> Removing pipx package 'aglibol-agent'..." -ForegroundColor Cyan
        & pipx uninstall aglibol-agent
    }
}

# 3. Clean up User PATH
$UserPath = [Environment]::GetEnvironmentVariable("PATH", "User")
if ($UserPath -like "*$BinDir*") {
    Write-Host "-> Removing $BinDir from User PATH..." -ForegroundColor Cyan
    $NewPath = ($UserPath.Split(';') | Where-Object { $_ -ne $BinDir -and $_ -ne "" }) -join ';'
    [Environment]::SetEnvironmentVariable("PATH", $NewPath, "User")
}

# 4. Remove proxy batch commands
if (Test-Path $BinDir) {
    Write-Host "-> Cleaning executable wrappers in $BinDir..." -ForegroundColor Cyan
    Remove-Item -Path (Join-Path $BinDir "aglibol*.cmd") -Force -ErrorAction SilentlyContinue
}

# 5. Remove dedicated virtualenv
if (Test-Path $EnvDir) {
    Write-Host "-> Removing dedicated virtual environment in $EnvDir..." -ForegroundColor Cyan
    Remove-Item -Recurse -Force $EnvDir -ErrorAction SilentlyContinue
}

# 6. Handle user data ($HOME\.aglibol)
if (Test-Path $InstallDir) {
    if ($Purge) {
        Write-Host "-> Purging all sessions, checkpoints, and data in $InstallDir..." -ForegroundColor Red
        Remove-Item -Recurse -Force $InstallDir -ErrorAction SilentlyContinue
        Write-Host "[OK] All Aglibol Agent files and data completely removed." -ForegroundColor Green
    } else {
        Write-Host "[i] User sessions and checkpoints preserved in $InstallDir." -ForegroundColor Yellow
        Write-Host "    To completely delete all persistent data, run:" -ForegroundColor White
        Write-Host "    Remove-Item -Recurse -Force `"$InstallDir`"" -ForegroundColor Cyan
    }
}

Write-Host "`n[OK] Aglibol Agent has been successfully uninstalled from your system.`n" -ForegroundColor Green
