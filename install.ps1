# Aglibol Agent Official One-Line Installer for Windows PowerShell
# Usage: irm https://raw.githubusercontent.com/Aglibol/AglibolAgent/main/install.ps1 | iex

$ErrorActionPreference = "Stop"

Write-Host @"
    _          _ _ _           _       _                    _   
   / \   __ _ | (_) |__   ___ | |     / \   __ _  ___ _ __ | |_ 
  / _ \ / _` || | | '_ \ / _ \| |    / _ \ / _` |/ _ \ '_ \| __|
 / ___ \ (_| || | | |_) | (_) | |   / ___ \ (_| |  __/ | | | |_ 
/_/   \_\__, ||_|_|_.__/ \___/|_|  /_/   \_\__, |\___|_| |_|\__|
        |___/                             |___/                
  Aglibol Agent — Personal Multi-Agent AI Framework (Ollama-Native)
"@ -ForegroundColor Cyan

$InstallDir = Join-Path $HOME ".aglibol"
$BinDir = Join-Path $InstallDir "bin"
$EnvDir = Join-Path $InstallDir "env"

if (-not (Test-Path $InstallDir)) { New-Item -ItemType Directory -Path $InstallDir | Out-Null }
if (-not (Test-Path $BinDir)) { New-Item -ItemType Directory -Path $BinDir | Out-Null }

# 1. Check for uv
$hasUv = Get-Command "uv" -ErrorAction SilentlyContinue
if ($hasUv) {
    Write-Host "[*] Found 'uv'. Installing Aglibol Agent CLI via uv tool..." -ForegroundColor Green
    & uv tool install --force aglibol-agent
} else {
    # Check for Python >= 3.11
    $PythonCmd = $null
    foreach ($cmd in @("python", "py", "python3")) {
        $c = Get-Command $cmd -ErrorAction SilentlyContinue
        if ($c) {
            $is311 = & $cmd -c "import sys; print(sys.version_info >= (3, 11))" 2>$null
            if ($is311 -eq "True") {
                $PythonCmd = $cmd
                break
            }
        }
    }

    if (-not $PythonCmd) {
        Write-Host "[*] Python 3.11+ not found. Installing standalone 'uv' installer..." -ForegroundColor Yellow
        powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
        $env:PATH = "$HOME\.cargo\bin;$env:PATH"
        & uv tool install aglibol-agent
    } else {
        Write-Host "[*] Found Python ($(& $PythonCmd --version))" -ForegroundColor Green
        Write-Host "[*] Provisioning dedicated virtual environment in $EnvDir..." -ForegroundColor Cyan
        & $PythonCmd -m venv $EnvDir
        & (Join-Path $EnvDir "Scripts\pip.exe") install --upgrade pip --quiet
        & (Join-Path $EnvDir "Scripts\pip.exe") install aglibol-agent --quiet

        # Create lightweight batch proxy in bin
        $BatContent = "@echo off`r`n`"$EnvDir\Scripts\aglibol.exe`" %*"
        Set-Content -Path (Join-Path $BinDir "aglibol.cmd") -Value $BatContent -Encoding ASCII
        Set-Content -Path (Join-Path $BinDir "aglibol-agent.cmd") -Value $BatContent -Encoding ASCII
    }
}

# 2. Ensure User PATH contains $BinDir or $HOME\.local\bin
$UserPath = [Environment]::GetEnvironmentVariable("PATH", "User")
if ($UserPath -notlike "*$BinDir*") {
    Write-Host "[*] Adding $BinDir to User PATH..." -ForegroundColor Cyan
    [Environment]::SetEnvironmentVariable("PATH", "$BinDir;$UserPath", "User")
    $env:PATH = "$BinDir;$env:PATH"
}

# 3. Check for Ollama service
$hasOllama = Get-Command "ollama" -ErrorAction SilentlyContinue
if ($hasOllama) {
    Write-Host "[*] Ollama is installed on this machine." -ForegroundColor Green
} else {
    Write-Host "[!] Note: Ollama was not detected in PATH. Please install Ollama from https://ollama.com if not already done." -ForegroundColor Yellow
}

Write-Host "`n======================================================" -ForegroundColor Green
Write-Host "[OK] Aglibol Agent installed successfully!" -ForegroundColor Green
Write-Host "Open a new terminal and launch the interactive REPL with:" -ForegroundColor White
Write-Host "  aglibol" -ForegroundColor Cyan
Write-Host "======================================================`n" -ForegroundColor Green
