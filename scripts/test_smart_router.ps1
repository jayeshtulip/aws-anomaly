# Test Smart A/B Router
$AB_URL = "k8s-triton-tritonab-aafed81da0-1339382983.us-east-1.elb.amazonaws.com"

Write-Host ("="*70) -ForegroundColor Cyan
Write-Host "TESTING SMART A/B ROUTER" -ForegroundColor Green
Write-Host ("="*70) -ForegroundColor Cyan
Write-Host ""

$results = @{
    xgboost = @{ success = 0; failed = 0; latencies = @() }
    catboost = @{ success = 0; failed = 0; latencies = @() }
}

Write-Host "Sending 100 requests to generic /v2/infer endpoint..." -ForegroundColor Yellow
Write-Host "(Router will automatically choose XGBoost or CatBoost)" -ForegroundColor Gray
Write-Host ""

for ($i = 1; $i -le 100; $i++) {
    try {
        $start = Get-Date
        
        # Send to generic endpoint - router decides which model
        $response = Invoke-WebRequest `
            -Uri "http://$AB_URL/v2/infer" `
            -Method Post `
            -Body '{"inputs":[{"name":"data","shape":[1,37],"datatype":"FP32","data":[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7]}]}' `
            -ContentType "application/json" `
            -UseBasicParsing `
            -TimeoutSec 10
        
        $latency = ((Get-Date) - $start).TotalMilliseconds
        $variant = $response.Headers['X-Model-Variant'][0]
        $model = $response.Headers['X-Routed-Model'][0]
        
        if ($variant -match "xgboost") {
            $results.xgboost.success++
            $results.xgboost.latencies += $latency
        } else {
            $results.catboost.success++
            $results.catboost.latencies += $latency
        }
        
        if ($i % 10 -eq 0) {
            Write-Host ("Progress: {0,3}/100 | Variant: {1}" -f $i, $variant) -ForegroundColor Gray
        }
        
    } catch {
        Write-Host "Request $i failed: $_" -ForegroundColor Red
        if ($variant -match "xgboost") {
            $results.xgboost.failed++
        } else {
            $results.catboost.failed++
        }
    }
    
    Start-Sleep -Milliseconds 100
}

Write-Host ""
Write-Host ("="*70) -ForegroundColor Cyan
Write-Host "RESULTS" -ForegroundColor Green
Write-Host ("="*70) -ForegroundColor Cyan
Write-Host ""

$total = $results.xgboost.success + $results.catboost.success

Write-Host "XGBoost Control:" -ForegroundColor Cyan
Write-Host ("  Successful: {0} ({1:N1}%)" -f $results.xgboost.success, (($results.xgboost.success/$total)*100)) -ForegroundColor White
Write-Host ("  Failed:     {0}" -f $results.xgboost.failed) -ForegroundColor $(if ($results.xgboost.failed -gt 0) { "Red" } else { "White" })
if ($results.xgboost.latencies.Count -gt 0) {
    $avg = ($results.xgboost.latencies | Measure-Object -Average).Average
    Write-Host ("  Avg Latency: {0:N0}ms" -f $avg) -ForegroundColor White
}

Write-Host ""
Write-Host "CatBoost Variant:" -ForegroundColor Yellow
Write-Host ("  Successful: {0} ({1:N1}%)" -f $results.catboost.success, (($results.catboost.success/$total)*100)) -ForegroundColor White
Write-Host ("  Failed:     {0}" -f $results.catboost.failed) -ForegroundColor $(if ($results.catboost.failed -gt 0) { "Red" } else { "White" })
if ($results.catboost.latencies.Count -gt 0) {
    $avg = ($results.catboost.latencies | Measure-Object -Average).Average
    Write-Host ("  Avg Latency: {0:N0}ms" -f $avg) -ForegroundColor White
}

Write-Host ""
Write-Host "Expected: ~70% XGBoost, ~30% CatBoost" -ForegroundColor Gray
Write-Host ("="*70) -ForegroundColor Cyan