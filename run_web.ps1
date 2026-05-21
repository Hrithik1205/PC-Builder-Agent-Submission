# PC Builder Agent - one-click web UI launcher (Windows)
# Usage:  powershell -ExecutionPolicy Bypass -File .\run_web.ps1

$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot
Set-Location $ProjectRoot

function Find-Python {
    # Prefer real installs over the WindowsApps store stub
    $candidates = @(
        "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe",
        "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe",
        "$env:LOCALAPPDATA\Programs\Python\Python310\python.exe",
        "C:\Program Files\Python312\python.exe",
        "C:\Program Files\Python311\python.exe"
    )
    foreach ($p in $candidates) {
        if (Test-Path $p) { return $p }
    }
    # py launcher (if installed)
    $py = Get-Command py -ErrorAction SilentlyContinue
    if ($py) { return "py" }
    # last resort: python on PATH (may be store stub)
    $python = Get-Command python -ErrorAction SilentlyContinue
    if ($python -and $python.Source -notlike "*WindowsApps*") {
        return $python.Source
    }
    return $null
}

Write-Host "=== PC Builder Agent - Web UI Setup ===" -ForegroundColor Cyan

$python = Find-Python
if (-not $python) {
    Write-Host ""
    Write-Host "Python is NOT installed (only the Microsoft Store stub was found)." -ForegroundColor Red
    Write-Host ""
    Write-Host "Fix (pick ONE):" -ForegroundColor Yellow
    Write-Host "  1. Download Python 3.12 from https://www.python.org/downloads/"
    Write-Host "     - Check 'Add python.exe to PATH' during install"
    Write-Host "     - Then close and reopen PowerShell and run this script again"
    Write-Host ""
    Write-Host "  2. Or in PowerShell (admin optional):"
    Write-Host "     winget install -e --id Python.Python.3.12"
    Write-Host ""
    Write-Host "  3. Disable the fake 'python' shortcut:"
    Write-Host "     Settings -> Apps -> Advanced app settings -> App execution aliases"
    Write-Host "     Turn OFF 'python.exe' and 'python3.exe'"
    Write-Host ""
    exit 1
}

Write-Host "Using Python: $python" -ForegroundColor Green

# Create venv if missing
$venvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    Write-Host "Creating virtual environment..." -ForegroundColor Cyan
    if ($python -eq "py") {
        & py -3.12 -m venv .venv
        if ($LASTEXITCODE -ne 0) { & py -3 -m venv .venv }
    } else {
        & $python -m venv .venv
    }
    if (-not (Test-Path $venvPython)) {
        Write-Host "Failed to create .venv" -ForegroundColor Red
        exit 1
    }
}

$venvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$venvPip = Join-Path $ProjectRoot ".venv\Scripts\pip.exe"

Write-Host "Installing dependencies (first run may take a few minutes)..." -ForegroundColor Cyan
& $venvPip install -q -r requirements.txt

# .env optional
if (-not (Test-Path ".env") -and (Test-Path ".env.example")) {
    Copy-Item ".env.example" ".env"
    Write-Host "Created .env from .env.example" -ForegroundColor DarkGray
}

$env:PYTHONPATH = $ProjectRoot

Write-Host ""
Write-Host "Starting Streamlit at http://localhost:8501" -ForegroundColor Green
Write-Host "Make sure Ollama is running and you have pulled the model:" -ForegroundColor Yellow
Write-Host "  ollama pull qwen2.5:7b-instruct" -ForegroundColor Yellow
Write-Host ""

& (Join-Path $ProjectRoot ".venv\Scripts\streamlit.exe") run src/ui/streamlit_app.py
