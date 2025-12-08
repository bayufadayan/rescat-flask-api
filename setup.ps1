#!/usr/bin/env pwsh
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  ResCat ML - Project Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "[1/6] Checking Python installation..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✓ Found: $pythonVersion" -ForegroundColor Green
}
catch {
    Write-Host "✗ Python not found!" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "[2/6] Setting up virtual environment..." -ForegroundColor Yellow
if (Test-Path "venv") {
    Write-Host "✓ Virtual environment already exists" -ForegroundColor Green
}
else {
    Write-Host "  Creating virtual environment..." -ForegroundColor Gray
    python -m venv venv
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ Virtual environment created" -ForegroundColor Green
    }
    else {
        Write-Host "✗ Failed to create virtual environment" -ForegroundColor Red
        exit 1
    }
}

Write-Host ""
Write-Host "[3/6] Activating virtual environment..." -ForegroundColor Yellow
$venvActivate = "venv\Scripts\Activate.ps1"
if (Test-Path $venvActivate) {
    & $venvActivate
    Write-Host "✓ Virtual environment activated" -ForegroundColor Green
}
else {
    Write-Host "✗ Failed to find activation script" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "[4/6] Installing dependencies..." -ForegroundColor Yellow
if (Test-Path "requirements.txt") {
    Write-Host "  This may take a few minutes..." -ForegroundColor Gray
    pip install --upgrade pip | Out-Null
    pip install -r requirements.txt
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ Dependencies installed successfully" -ForegroundColor Green
    }
    else {
        Write-Host "✗ Failed to install dependencies" -ForegroundColor Red
        exit 1
    }
}
else {
    Write-Host "✗ requirements.txt not found" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "[5/6] Setting up environment configuration..." -ForegroundColor Yellow
if (Test-Path ".env") {
    Write-Host "✓ .env file already exists" -ForegroundColor Green
}
else {
    if (Test-Path ".env.example") {
        Copy-Item ".env.example" ".env"
        Write-Host "✓ Created .env from .env.example" -ForegroundColor Green
    }
    else {
        Write-Host "✗ .env.example not found" -ForegroundColor Red
        exit 1
    }
}

Write-Host ""
Write-Host "[6/6] Creating required directories..." -ForegroundColor Yellow
$directories = @("cache/remove-bg", "models", "static/images")
foreach ($dir in $directories) {
    if (!(Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Host "  Created: $dir" -ForegroundColor Gray
    }
}
Write-Host "✓ Directories ready" -ForegroundColor Green

Write-Host ""
Write-Host "Checking model files..." -ForegroundColor Yellow
$modelFiles = @(
    "models/mobilenetv3_small.onnx",
    "models/cat_head_model.onnx",
    "models/imagenet_classes.txt",
    "models/cat_head_classes.json"
)
$missingModels = @()
foreach ($model in $modelFiles) {
    if (Test-Path $model) {
        Write-Host "  ✓ $model" -ForegroundColor Green
    }
    else {
        Write-Host "  ✗ $model (missing)" -ForegroundColor Red
        $missingModels += $model
    }
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Setup Summary" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

if ($missingModels.Count -gt 0) {
    Write-Host "⚠ Setup completed with warnings:" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Missing model files:" -ForegroundColor Yellow
    foreach ($model in $missingModels) {
        Write-Host "  - $model" -ForegroundColor Red
    }
    Write-Host ""
    Write-Host "Please add the model files to continue." -ForegroundColor Yellow
}
else {
    Write-Host "✓ Setup completed successfully!" -ForegroundColor Green
}

Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Edit .env file with your configuration" -ForegroundColor White
Write-Host "  2. Ensure all model files are in models/ directory" -ForegroundColor White
Write-Host "  3. Run the application: python app.py" -ForegroundColor White
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
