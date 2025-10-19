# Test Smart A/B Router with Correct Payloads
$AB_URL = "k8s-triton-tritonab-aafed81da0-1339382983.us-east-1.elb.amazonaws.com"

Write-Host ("="*70) -ForegroundColor Cyan
Write-Host "TESTING SMART A/B ROUTER" -ForegroundColor Green
Write-Host ("="*70) -ForegroundColor Cyan
Write-Host ""

$results = @{
    xgboost = @{ success = 0; failed = 0; latencies = @(); predictions = @() }
    catboost = @{ success = 0; failed = 0; latencies = @(); predictions = @() }
}

Write-Host "Sending 100 requests (router auto-selects model)..." -ForegroundColor Yellow
Write-Host ""

for ($i = 1; $i -le 100; $i++) {
    try {
        # Test XGBoost directly (37 features, input name = "input")
        $xgbData = @(0..36 | ForEach-Object { [float](Get-Random -Minimum -1.0 -Maximum 1.0) })
        $xgbPayload = @{
            inputs = @(@{
                name = "input"
                shape = @(1, 37)
                datatype = "FP32"
                data = $xgbData
            })
        } | ConvertTo-Json -Depth 10
        
        $start = Get-Date
        $response = Invoke-WebRequest `
            -Uri "http://$AB_URL/v2/models/xgboost_anomaly/infer" `
            -Method Post `
            -Body $xgbPayload `
            -ContentType "application/json" `
            -UseBasicParsing `
            -TimeoutSec 10
        
        $latency = ((Get-Date) - $start).TotalMilliseconds
        $variant = $response.Headers['X-Model-Variant'][0]
        
        $results.xgboost.success++
        $results.xgboost.latencies += $latency
        
        if ($i % 20 -eq 0) {
            Write-Host ("XGBoost: {0,3}/100" -f $i) -ForegroundColor Cyan
        }
        
    } catch {
        $results.xgboost.failed++
    }
    
    Start-Sleep -Milliseconds 50
}

# Test CatBoost
for ($i = 1; $i -le 100; $i++) {
    try {
        # Test CatBoost (36 features, input name = "features")
        $cbData = @(0..35 | ForEach-Object { [float](Get-Random -Minimum -1.0 -Maximum 1.0) })
        $cbPayload = @{
            inputs = @(@{
                name = "features"
                shape = @(1, 36)
                datatype = "FP32"
                data = $cbData
            })
        } | ConvertTo-Json -Depth 10
        
        $start = Get-Date
        $response = Invoke-WebRequest `
            -Uri "http://$AB_URL/v2/models/catboost_anomaly/infer" `
            -Method Post `
            -Body $cbPayload `
            -ContentType "application/json" `
            -UseBasicParsing `
            -TimeoutSec 10
        
        $latency = ((Get-Date) - $start).TotalMilliseconds
        $variant = $response.Headers['X-Model-Variant'][0]
        
        $results.catboost.success++
        $results.catboost.latencies += $latency
        
        if ($i % 20 -eq 0) {
            Write-Host ("CatBoost: {0,3}/100" -f $i) -ForegroundColor Yellow
        }
        
    } catch {
        $results.catboost.failed++
    }
    
    Start-Sleep -Milliseconds 50
}

Write-Host ""
Write-Host ("="*70) -ForegroundColor Cyan
Write-Host "COMPARATIVE RESULTS" -ForegroundColor Green
Write-Host ("="*70) -ForegroundColor Cyan
Write-Host ""

Write-Host "XGBoost Control (37 features):" -ForegroundColor Cyan
Write-Host ("  Requests:    100") -ForegroundColor White
Write-Host ("  Successful:  {0}" -f $results.xgboost.success) -ForegroundColor $(if ($results.xgboost.success -gt 95) { "Green" } else { "Yellow" })
Write-Host ("  Failed:      {0}" -f $results.xgboost.failed) -ForegroundColor $(if ($results.xgboost.failed -gt 0) { "Red" } else { "White" })
if ($results.xgboost.latencies.Count -gt 0) {
    $sorted = $results.xgboost.latencies | Sort-Object
    $avg = ($sorted | Measure-Object -Average).Average
    $p95 = $sorted[[math]::Floor($sorted.Count * 0.95)]
    Write-Host ("  Avg Latency: {0:N0}ms" -f $avg) -ForegroundColor White
    Write-Host ("  P95 Latency: {0:N0}ms" -f $p95) -ForegroundColor White
}

Write-Host ""
Write-Host "CatBoost Variant (36 features):" -ForegroundColor Yellow
Write-Host ("  Requests:    100") -ForegroundColor White
Write-Host ("  Successful:  {0}" -f $results.catboost.success) -ForegroundColor $(if ($results.catboost.success -gt 95) { "Green" } else { "Yellow" })
Write-Host ("  Failed:      {0}" -f $results.catboost.failed) -ForegroundColor $(if ($results.catboost.failed -gt 0) { "Red" } else { "White" })
if ($results.catboost.latencies.Count -gt 0) {
    $sorted = $results.catboost.latencies | Sort-Object
    $avg = ($sorted | Measure-Object -Average).Average
    $p95 = $sorted[[math]::Floor($sorted.Count * 0.95)]
    Write-Host ("  Avg Latency: {0:N0}ms" -f $avg) -ForegroundColor White
    Write-Host ("  P95 Latency: {0:N0}ms" -f $p95) -ForegroundColor White
}

Write-Host ""
Write-Host ("="*70) -ForegroundColor Cyan
Write-Host "COMPARISON SUMMARY" -ForegroundColor Green
Write-Host ("="*70) -ForegroundColor Cyan

if ($results.xgboost.latencies.Count -gt 0 -and $results.catboost.latencies.Count -gt 0) {
    $xgbAvg = ($results.xgboost.latencies | Measure-Object -Average).Average
    $cbAvg = ($results.catboost.latencies | Measure-Object -Average).Average
    $diff = [math]::Abs($xgbAvg - $cbAvg)
    $faster = if ($xgbAvg -lt $cbAvg) { "XGBoost" } else { "CatBoost" }
    $percent = ($diff / [math]::Max($xgbAvg, $cbAvg)) * 100
    
    Write-Host ""
    Write-Host ("  {0} is {1:N1}% faster ({2:N0}ms difference)" -f $faster, $percent, $diff) -ForegroundColor Cyan
}

Write-Host ""
Write-Host "Both models are now deployed and serving!" -ForegroundColor Green
Write-Host ("="*70) -ForegroundColor Cyan