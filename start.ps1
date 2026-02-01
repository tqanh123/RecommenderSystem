# Start LightGCN Real-Time Recommender System
# This script starts both the Python inference service and Node.js server

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "🚀 Starting LightGCN Real-Time Recommender System" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Check if MongoDB is running
Write-Host "📡 Checking MongoDB..." -ForegroundColor Yellow
try {
    $mongoCheck = & mongosh --eval "db.adminCommand('ping')" --quiet 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ MongoDB is running" -ForegroundColor Green
    }
} catch {
    Write-Host "❌ MongoDB is not running. Please start MongoDB first!" -ForegroundColor Red
    Write-Host "   Run: mongod" -ForegroundColor Yellow
    exit 1
}

Write-Host ""

# Start Python inference service in background
Write-Host "🐍 Starting Python LightGCN Inference Service..." -ForegroundColor Yellow

# Detect Python environment (conda or venv)
$pythonExe = $null

# Check if conda environment exists
if (Get-Command conda -ErrorAction SilentlyContinue) {
    $condaEnvs = conda env list | Select-String "recommender"
    if ($condaEnvs) {
        Write-Host "   Using conda environment 'recommender'" -ForegroundColor Cyan
        $pythonExe = "conda"
        $pythonArgs = "run", "-n", "recommender", "python", "framework/inference_service.py"
    }
}

# Fallback to venv if conda not found
if (-not $pythonExe) {
    $venvPython = "D:/study/thesis/RecommenderSystem/.venv/Scripts/python.exe"
    if (Test-Path $venvPython) {
        Write-Host "   Using virtual environment (.venv)" -ForegroundColor Cyan
        $pythonExe = $venvPython
        $pythonArgs = @("framework/inference_service.py")
    } else {
        Write-Host "❌ No Python environment found!" -ForegroundColor Red
        Write-Host "   Please install dependencies first:" -ForegroundColor Yellow
        Write-Host "   Option 1 (Conda): conda env create -f framework/environment.yml" -ForegroundColor Yellow
        Write-Host "   Option 2 (Venv):  pip install -r framework/requirements_inference.txt" -ForegroundColor Yellow
        exit 1
    }
}

$pythonJob = Start-Job -ScriptBlock {
    param($exe, $args)
    & $exe $args
} -ArgumentList $pythonExe, $pythonArgs

Start-Sleep -Seconds 3

# Check if Python service started
try {
    $healthCheck = Invoke-WebRequest -Uri "http://localhost:5001/health" -UseBasicParsing -TimeoutSec 2
    Write-Host "✅ Python service is running on http://localhost:5001" -ForegroundColor Green
} catch {
    Write-Host "⚠️  Python service may still be starting..." -ForegroundColor Yellow
}

Write-Host ""

# Start Node.js server
Write-Host "🌐 Starting Node.js Server..." -ForegroundColor Yellow
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "✅ System Ready!" -ForegroundColor Green
Write-Host "   📊 Python Inference: http://localhost:5001" -ForegroundColor White
Write-Host "   🌐 Web Interface:    http://localhost:3000" -ForegroundColor White
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Press Ctrl+C to stop all services" -ForegroundColor Yellow
Write-Host ""

# Start Node.js in foreground
node server.js

# Cleanup on exit
Write-Host "`n🛑 Stopping Python service..." -ForegroundColor Yellow
Stop-Job -Job $pythonJob
Remove-Job -Job $pythonJob
Write-Host "✅ All services stopped" -ForegroundColor Green
