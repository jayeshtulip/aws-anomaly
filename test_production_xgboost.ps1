# Test Production XGBoost Model
$EXTERNAL_URL = "a8739570ffb124c19abef83f51e636c4-342298904.us-east-1.elb.amazonaws.com"
$BASE_URL = "http://${EXTERNAL_URL}:8000"
$MODEL_NAME = "xgboost_anomaly"

Write-Host ("="*80) -ForegroundColor Cyan
Write-Host "PRODUCTION XGBOOST ANOMALY MODEL TEST" -ForegroundColor Green
Write-Host "URL: $BASE_URL" -ForegroundColor White
Write-Host ("="*80) -ForegroundColor Cyan

# Health check
Write-Host "`n[1] Health Check" -ForegroundColor Yellow
try {
    Invoke-RestMethod -Uri "$BASE_URL/v2/health/ready" -Method Get | Out-Null
    Write-Host "Success: Server ready" -ForegroundColor Green
} catch {
    Write-Host "Failed: Server not ready" -ForegroundColor Red
    exit 1
}

# Model status
Write-Host "`n[2] Model Status" -ForegroundColor Yellow
try {
    $model = Invoke-RestMethod -Uri "$BASE_URL/v2/models/$MODEL_NAME" -Method Get
    Write-Host "Success: $($model.name) is loaded" -ForegroundColor Green
} catch {
    Write-Host "Failed: Model not available" -ForegroundColor Red
    exit 1
}

# Get config
Write-Host "`n[3] Model Configuration" -ForegroundColor Yellow
try {
    $config = Invoke-RestMethod -Uri "$BASE_URL/v2/models/$MODEL_NAME/config" -Method Get
    Write-Host "Success: Backend: $($config.backend)" -ForegroundColor Green
    Write-Host "  Platform: $($config.platform)" -ForegroundColor White
    Write-Host "  Max Batch Size: $($config.max_batch_size)" -ForegroundColor White
} catch {
    Write-Host "Warning: Could not get config" -ForegroundColor Yellow
}

Write-Host "`n" -NoNewline
Write-Host ("="*80) -ForegroundColor Cyan
Write-Host "PRODUCTION MODEL READY!" -ForegroundColor Green
Write-Host "`nEndpoint: $BASE_URL/v2/models/$MODEL_NAME/infer" -ForegroundColor Cyan
Write-Host ("="*80) -ForegroundColor Cyan