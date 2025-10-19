# A/B Testing Monitoring Script
param(
    [int]$Requests = 100,
    [int]$DelayMs = 100
)

$INGRESS_URL = kubectl get ingress -n triton triton-ab-ingress -o jsonpath='{.status.loadBalancer.ingress[0].hostname}'

if (-not $INGRESS_URL) {
    Write-Host "Ingress not found" -ForegroundColor Red
    exit 1
}

Write-Host ("="*70) -ForegroundColor Cyan
Write-Host "A/B TESTING LIVE MONITORING" -ForegroundColor Green
Write-Host ("="*70) -ForegroundColor Cyan
Write-Host ""
Write-Host "URL: http://$INGRESS_URL" -ForegroundColor White
Write-Host "Requests: $Requests" -ForegroundColor White
Write-Host ""

# Metrics tracking
$metrics = @{
    total = 0
    success = 0
    failed = 0
    latencies = @()
    variants = @{}
}

# Send requests
for ($i = 1; $i -le $Requests; $i++) {
    $metrics.total++
    
    try {
        $start = Get-Date
        
        # Generate random test data (37 features)
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
            -Uri "http://$INGRESS_URL/v2/models/xgboost_anomaly/infer" `
            -Method Post `
            -Body $payload `
            -ContentType "application/json" `
            -UseBasicParsing `
            -TimeoutSec 10
        
        $latency = ((Get-Date) - $start).TotalMilliseconds
        $variant = if ($response.Headers['X-Model-Variant']) { $response.Headers['X-Model-Variant'][0] } else { "unknown" }
        
        $metrics.success++
        $metrics.latencies += $latency
        
        if (-not $metrics.variants.ContainsKey($variant)) {
            $metrics.variants[$variant] = @{ count = 0; latencies = @() }
        }
        $metrics.variants[$variant].count++
        $metrics.variants[$variant].latencies += $latency
        
        if ($i % 10 -eq 0) {
            $successRate = ($metrics.success / $metrics.total) * 100
            $avgLatency = ($metrics.latencies | Measure-Object -Average).Average
            Write-Host ("Progress: {0,3}/{1} | Success: {2,5:N1}% | Avg Latency: {3,6:N1}ms" -f $i, $Requests, $successRate, $avgLatency) -ForegroundColor Gray
        }
        
    } catch {
        $metrics.failed++
        if ($i % 10 -eq 0) {
            Write-Host ("Progress: {0,3}/{1} | Request failed" -f $i, $Requests) -ForegroundColor Red
        }
    }
    
    if ($DelayMs -gt 0) {
        Start-Sleep -Milliseconds $DelayMs
    }
}

# Display results
Write-Host ""
Write-Host ("="*70) -ForegroundColor Cyan
Write-Host "A/B TEST RESULTS" -ForegroundColor Green
Write-Host ("="*70) -ForegroundColor Cyan
Write-Host ""

Write-Host "Overall Metrics:" -ForegroundColor Yellow
Write-Host ("  Total Requests:  {0}" -f $metrics.total) -ForegroundColor White
Write-Host ("  Successful:      {0} ({1:N1}%)" -f $metrics.success, (($metrics.success/$metrics.total)*100)) -ForegroundColor Green
Write-Host ("  Failed:          {0} ({1:N1}%)" -f $metrics.failed, (($metrics.failed/$metrics.total)*100)) -ForegroundColor $(if ($metrics.failed -gt 0) { "Red" } else { "White" })

if ($metrics.latencies.Count -gt 0) {
    $sorted = $metrics.latencies | Sort-Object
    $p50 = $sorted[[math]::Floor($sorted.Count * 0.50)]
    $p95 = $sorted[[math]::Floor($sorted.Count * 0.95)]
    $p99 = $sorted[[math]::Floor($sorted.Count * 0.99)]
    $avg = ($metrics.latencies | Measure-Object -Average).Average
    
    Write-Host ""
    Write-Host "Latency Distribution:" -ForegroundColor Yellow
    Write-Host ("  Average:  {0,6:N1}ms" -f $avg) -ForegroundColor White
    Write-Host ("  P50:      {0,6:N1}ms" -f $p50) -ForegroundColor White
    Write-Host ("  P95:      {0,6:N1}ms" -f $p95) -ForegroundColor White
    Write-Host ("  P99:      {0,6:N1}ms" -f $p99) -ForegroundColor White
}

Write-Host ""
Write-Host "Variant Distribution:" -ForegroundColor Yellow
foreach ($variant in $metrics.variants.Keys | Sort-Object) {
    $data = $metrics.variants[$variant]
    $percentage = ($data.count / $metrics.success) * 100
    $avgLatency = if ($data.latencies.Count -gt 0) { 
        ($data.latencies | Measure-Object -Average).Average 
    } else { 
        0 
    }
    
    Write-Host ("  {0,-20} : {1,3} requests ({2,5:N1}%) | Avg: {3,6:N1}ms" -f $variant, $data.count, $percentage, $avgLatency) -ForegroundColor Cyan
}

Write-Host ""
Write-Host ("="*70) -ForegroundColor Cyan
Write-Host "A/B Testing Complete!" -ForegroundColor Green
Write-Host ("="*70) -ForegroundColor Cyan