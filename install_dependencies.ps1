# PowerShell script to install all dependencies in virtual environment

Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "Installing Policy Processor Dependencies" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host ""

# Check if in virtual environment
if ($env:VIRTUAL_ENV) {
    Write-Host "✓ Virtual environment detected: $env:VIRTUAL_ENV" -ForegroundColor Green
} else {
    Write-Host "⚠ WARNING: Not in a virtual environment!" -ForegroundColor Yellow
    Write-Host "Please run: .venv\Scripts\Activate.ps1" -ForegroundColor Yellow
    Write-Host ""
    $response = Read-Host "Continue anyway? (y/n)"
    if ($response -ne "y") {
        exit 1
    }
}

Write-Host ""
Write-Host "Installing dependencies..." -ForegroundColor Cyan
Write-Host ""

# Install all dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

Write-Host ""
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "Verifying Installation" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host ""

# Verify key packages
$packages = @("loguru", "a2a", "streamlit", "sqlalchemy", "fastapi")

foreach ($package in $packages) {
    try {
        python -c "import $package; print('✓ $package installed')" 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✓ $package installed" -ForegroundColor Green
        } else {
            Write-Host "✗ $package NOT installed" -ForegroundColor Red
        }
    } catch {
        Write-Host "✗ $package NOT installed" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "Installation Complete!" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Run: python setup_new_features.py" -ForegroundColor White
Write-Host "2. Start servers as described in QUICK_START.md" -ForegroundColor White
Write-Host ""
