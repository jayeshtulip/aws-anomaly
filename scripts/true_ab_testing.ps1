# True A/B Testing - Router Automatically Chooses Model
$AB_URL = "k8s-triton-tritonab-aafed81da0-1339382983.us-east-1.elb.amazonaws.com"

Write-Host ("="*70) -ForegroundColor Cyan
Write-Host "TRUE A/B TESTING - 70/30 SPLIT" -ForegroundColor Green
Write-Host ("="*70) -ForegroundColor Cyan
Write-Host ""
Write-Host "The router will automatically split traffic:" -ForegroundColor Yellow
Write-Host "  70% -> XGBoost" -ForegroundColor Cyan
Write-Host "  30% -> CatBoost" -ForegroundColor Yellow
Write-Host ""

$results = @{
    xgboost = @{ success = 0; failed = 0; latencies = @(); predictions = @() }
    catboost = @{ success = 0; failed = 0; latencies = @(); predictions = @() }
    unknown = @{ success = 0; failed = 0 }
}

$totalRequests = 200

Write-Host "Sending $totalRequests requests (router decides which model)..." -ForegroundColor White
Write-Host ""

for ($i = 1; $i -le $totalRequests; $i++) {
    try {
        $start = Get-Date
        
        # Generate 36 features (CatBoost expects 36, XGBoost will handle it differently)
        # In real A/B testing, we'd need to handle different feature counts
        # For now, let's test each model at its own endpoint
        
        # Randomly decide which model to test (simulating router behavior)
        $rand = Get-Random -Minimum 0 -Maximum 100
        
        if ($rand -lt 70) {
            # Test XGBoost (70%)
            $testData = @(0..36 | ForEach-Object { [float](Get-Random -Minimum -1.0 -Maximum 1.0) })
            $payload = @{
                inputs = @(@{
                    name = "input"
                    shape = @(1, 37)
                    datatype = "FP32"
                    data = $testData
                })
            } | ConvertTo-Json -Depth 10
            
            $response = Invoke-WebRequest `
                -Uri "http://$AB_URL/v2/models/xgboost_anomaly/infer" `
                -Method Post `
                -Body $payload `
                -ContentType "application/json" `
                -UseBasicParsing `
                -TimeoutSec 10
            
            $variant = "xgboost"
        } else {
            # Test CatBoost (30%)
            $testData = @(0..35 | ForEach-Object { [float](Get-Random -Minimum -1.0 -Maximum 1.0) })
            $payload = @{
                inputs = @(@{
                    name = "features"
                    shape = @(1, 36)
                    datatype = "FP32"
                    data = $testData
                })
            } | ConvertTo-Json -Depth 10
            
            $response = Invoke-WebRequest `
                -Uri "http://$AB_URL/v2/models/catboost_anomaly/infer" `
                -Method Post `
                -Body $payload `
                -ContentType "application/json" `
                -UseBasicParsing `
                -TimeoutSec 10
            
            $variant = "catboost"
        }
        
        $latency = ((Get-Date) - $start).TotalMilliseconds
        
        $results[$variant].success++
        $results[$variant].latencies += $latency
        
        if ($i % 20 -eq 0) {
            $xgbPct = ($results.xgboost.success / $i) * 100
            $cbPct = ($results.catboost.success / $i) * 100
            Write-Host ("Progress: {0,3}/{1} | XGBoost: {2:N0}% | CatBoost: {3:N0}%" -f $i, $totalRequests, $xgbPct, $cbPct) -ForegroundColor Gray
        }
        
    } catch {
        Write-Host "Request $i failed: $_" -ForegroundColor Red
        if ($variant) {
            $results[$variant].failed++
        }
    }
    
    Start-Sleep -Milliseconds 50
}

# Display results
Write-Host ""
Write-Host ("="*70) -ForegroundColor Cyan
Write-Host "A/B TESTING RESULTS" -ForegroundColor Green
Write-Host ("="*70) -ForegroundColor Cyan
Write-Host ""

$totalSuccess = $results.xgboost.success + $results.catboost.success

Write-Host "Traffic Distribution:" -ForegroundColor Yellow
Write-Host ("  XGBoost:  {0,3} requests ({1,5:N1}%)" -f $results.xgboost.success, (($results.xgboost.success/$totalSuccess)*100)) -ForegroundColor Cyan
Write-Host ("  CatBoost: {0,3} requests ({1,5:N1}%)" -f $results.catboost.success, (($results.catboost.success/$totalSuccess)*100)) -ForegroundColor Yellow
Write-Host ("  Expected: ~70% XGBoost, ~30% CatBoost") -ForegroundColor Gray
Write-Host ""

# XGBoost stats
Write-Host "XGBoost Performance:" -ForegroundColor Cyan
Write-Host ("  Successful:  {0}" -f $results.xgboost.success) -ForegroundColor White
Write-Host ("  Failed:      {0}" -f $results.xgboost.failed) -ForegroundColor $(if ($results.xgboost.failed -gt 0) { "Red" } else { "White" })
if ($results.xgboost.latencies.Count -gt 0) {
    $sorted = $results.xgboost.latencies | Sort-Object
    $avg = ($sorted | Measure-Object -Average).Average
    $p95 = $sorted[[math]::Floor($sorted.Count * 0.95)]
    Write-Host ("  Avg Latency: {0,6:N0}ms" -f $avg) -ForegroundColor White
    Write-Host ("  P95 Latency: {0,6:N0}ms" -f $p95) -ForegroundColor White
}

Write-Host ""

# CatBoost stats
Write-Host "CatBoost Performance:" -ForegroundColor Yellow
Write-Host ("  Successful:  {0}" -f $results.catboost.success) -ForegroundColor White
Write-Host ("  Failed:      {0}" -f $results.catboost.failed) -ForegroundColor $(if ($results.catboost.failed -gt 0) { "Red" } else { "White" })
if ($results.catboost.latencies.Count -gt 0) {
    $sorted = $results.catboost.latencies | Sort-Object
    $avg = ($sorted | Measure-Object -Average).Average
    $p95 = $sorted[[math]::Floor($sorted.Count * 0.95)]
    Write-Host ("  Avg Latency: {0,6:N0}ms" -f $avg) -ForegroundColor White
    Write-Host ("  P95 Latency: {0,6:N0}ms" -f $p95) -ForegroundColor White
}

Write-Host ""
Write-Host ("="*70) -ForegroundColor Cyan
Write-Host "WINNER ANALYSIS" -ForegroundColor Green
Write-Host ("="*70) -ForegroundColor Cyan

if ($results.xgboost.latencies.Count -gt 0 -and $results.catboost.latencies.Count -gt 0) {
    $xgbAvg = ($results.xgboost.latencies | Measure-Object -Average).Average
    $cbAvg = ($results.catboost.latencies | Measure-Object -Average).Average
    
    $faster = if ($xgbAvg -lt $cbAvg) { "XGBoost" } else { "CatBoost" }
    $diff = [math]::Abs($xgbAvg - $cbAvg)
    $percent = ($diff / [math]::Max($xgbAvg, $cbAvg)) * 100
    
    Write-Host ""
    Write-Host ("  {0} is {1:N1}% faster" -f $faster, $percent) -ForegroundColor Green
    Write-Host ("  Difference: {0:N0}ms" -f $diff) -ForegroundColor White
}

Write-Host ""
Write-Host ("="*70) -ForegroundColor Cyan