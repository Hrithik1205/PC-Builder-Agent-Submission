# Download ALL component CSVs from the dataset GitHub repo into data/csv/
# Use this if the Python app fails with SSL/certificate errors (common on corporate VPN).

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path $PSScriptRoot -Parent
$OutDir = Join-Path $ProjectRoot "data\csv"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$Base = "https://raw.githubusercontent.com/vinayak-ensemble/Computer_Components_Dataset/main/data/csv"

# All 25 CSV files in the source dataset
$Files = @(
    "case-accessory.csv",
    "case-fan.csv",
    "case.csv",
    "cpu-cooler.csv",
    "cpu.csv",
    "external-hard-drive.csv",
    "fan-controller.csv",
    "headphones.csv",
    "internal-hard-drive.csv",
    "keyboard.csv",
    "memory.csv",
    "monitor.csv",
    "motherboard.csv",
    "mouse.csv",
    "optical-drive.csv",
    "os.csv",
    "power-supply.csv",
    "sound-card.csv",
    "speakers.csv",
    "thermal-paste.csv",
    "ups.csv",
    "video-card.csv",
    "webcam.csv",
    "wired-network-card.csv",
    "wireless-network-card.csv"
)

Write-Host "Downloading $($Files.Count) CSV files to $OutDir ..." -ForegroundColor Cyan

$failed = @()
foreach ($f in $Files) {
    $url = "$Base/$f"
    $dest = Join-Path $OutDir $f
    Write-Host "  $f ..." -NoNewline
    try {
        if ($PSVersionTable.PSVersion.Major -ge 7) {
            Invoke-WebRequest -Uri $url -OutFile $dest -UseBasicParsing -SkipCertificateCheck -TimeoutSec 60
        } else {
            Invoke-WebRequest -Uri $url -OutFile $dest -UseBasicParsing -TimeoutSec 60
        }
        Write-Host " ok" -ForegroundColor Green
    } catch {
        Write-Host " FAILED ($_)" -ForegroundColor Red
        $failed += $f
    }
}

if ($failed.Count -gt 0) {
    Write-Host ""
    Write-Host "$($failed.Count) file(s) failed to download:" -ForegroundColor Yellow
    $failed | ForEach-Object { Write-Host "  - $_" -ForegroundColor Yellow }
    exit 1
}

Write-Host ""
Write-Host "All $($Files.Count) files downloaded. Restart the app." -ForegroundColor Green
