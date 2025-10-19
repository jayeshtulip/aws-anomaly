# Test Production Ingress
$INGRESS_URL = "k8s-triton-tritonin-48809c3514-1818376050.us-east-1.elb.amazonaws.com"
$MAX_ATTEMPTS = 12
$RETRY_INTERVAL = 10

Write-Host "="*70 -ForegroundColor Cyan
Write-Host "TESTING PRODUCTION INGRESS" -ForegroundColor Green
Write-Host "="*70 -ForegroundColor Cyan
Write-Host ""
Write-Host "URL: http://$INGRESS_URL" -ForegroundColor Cyan
Write-Host "Waiting for ALB health checks to pass..." -ForegroundColor Yellow
Write-Host ""

for ($attempt = 1; $attempt -le $MAX_ATTEMPTS; $attempt++) {
    Write-Host "Attempt $attempt of $MAX_ATTEMPTS..." -ForegroundColor Gray
    
    try {
        $result = Invoke-RestMethod -Uri "http://$INGRESS_URL/v2" -TimeoutSec 10 -ErrorAction Stop
        
        Write-Host ""
        Write-Host "SUCCESS! Triton is accessible!" -ForegroundColor Green
        Write-Host ""
        Write-Host "="*70 -ForegroundColor Cyan
        Write-Host "PRODUCTION ENDPOINT READY" -ForegroundColor Green
        Write-Host "="*70 -ForegroundColor Cyan
        Write-Host ""
        Write-Host "URL:     http://$INGRESS_URL" -ForegroundColor Cyan
        Write-Host "Version: $($result.version)" -ForegroundColor White
        Write-Host ""
        Write-Host "Endpoints:" -ForegroundColor Yellow
        Write-Host "  Health:     http://$INGRESS_URL/v2/health/ready" -ForegroundColor White
        Write-Host "  Models:     http://$INGRESS_URL/v2" -ForegroundColor White
        Write-Host "  XGBoost:    http://$INGRESS_URL/v2/models/xgboost_anomaly" -ForegroundColor White
        Write-Host "  Inference:  http://$INGRESS_URL/v2/models/xgboost_anomaly/infer" -ForegroundColor White
        Write-Host ""
        
        # Test model
        Write-Host "Testing model access..." -ForegroundColor Yellow
        try {
            $model = Invoke-RestMethod -Uri "http://$INGRESS_URL/v2/models/xgboost_anomaly" -TimeoutSec 10 -ErrorAction Stop
            Write-Host "Model is ready: $($model.name)" -ForegroundColor Green
        } catch {
            Write-Host "Model not accessible yet" -ForegroundColor Yellow
        }
        
        Write-Host ""
        Write-Host "="*70 -ForegroundColor Cyan
        Write-Host "YOUR ML MODEL IS LIVE IN PRODUCTION!" -ForegroundColor Green
        Write-Host "="*70 -ForegroundColor Cyan
        
        exit 0
    } catch {
        Write-Host "  Not ready yet..." -ForegroundColor Gray
        if ($attempt -lt $MAX_ATTEMPTS) {
            Start-Sleep -Seconds $RETRY_INTERVAL
        }
    }
}

Write-Host ""
Write-Host "Timeout waiting for ALB" -ForegroundColor Yellow
Write-Host "Check ALB target health or use port-forward" -ForegroundColor Cyan