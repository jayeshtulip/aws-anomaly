# Test Traffic Split Between XGBoost and CatBoost
$AB_URL = "k8s-triton-tritonab-aafed81da0-1339382983.us-east-1.elb.amazonaws.com"

Write-Host ("="*70) -ForegroundColor Cyan
Write-Host "TESTING TRAFFIC SPLIT" -ForegroundColor Green
Write-Host ("="*70) -ForegroundColor Cyan
Write-Host ""

$variants = @{
    "xgboost-control" = 0
    "catboost-variant" = 0
    "other" = 0
}

Write-Host "Sending 100 requests..." -ForegroundColor Yellow
Write-Host ""

for ($i = 1; $i -le 100; $i++) {
    $testData = @(0..35 | ForEach-Object { [float](Get-Random -Minimum -1.0 -Maximum 1.0) })
    $payload = @{
        inputs = @(@{
            name = "features"
            shape = @(1, 36)
            datatype = "FP32"
            data = $testData
        })
    } | ConvertTo-Json -Depth 10
    
    try {
        $response = Invoke-WebRequest `
            -Uri "http://$AB_URL/v2/models/catboost_anomaly/infer" `
            -Method Post `
            -Body $payload `
            -ContentType "application/json" `
            -UseBasicParsing `
            -TimeoutSec 10
        
        $variant = if ($response.Headers['X-Model-Variant']) { 
            $response.Headers['X-Model-Variant'][0] 
        } else { 
            "other" 
        }
        
        if ($variants.ContainsKey($variant)) {
            $variants[$variant]++
        } else {
            $variants["other"]++
        }
        
        if ($i % 10 -eq 0) {
            Write-Host ("Progress: {0,3}/100" -f $i) -ForegroundColor Gray
        }
    } catch {
        Write-Host "Request $i failed: $_" -ForegroundColor Red
    }
    
    Start-Sleep -Milliseconds 100
}

Write-Host ""
Write-Host ("="*70) -ForegroundColor Cyan
Write-Host "TRAFFIC DISTRIBUTION RESULTS" -ForegroundColor Green
Write-Host ("="*70) -ForegroundColor Cyan
Write-Host ""

$total = $variants["xgboost-control"] + $variants["catboost-variant"] + $variants["other"]

foreach ($variant in $variants.Keys | Sort-Object) {
    $count = $variants[$variant]
    $percent = if ($total -gt 0) { ($count / $total) * 100 } else { 0 }
    $color = switch ($variant) {
        "xgboost-control" { "Cyan" }
        "catboost-variant" { "Yellow" }
        default { "Red" }
    }
    Write-Host ("  {0,-20} : {1,3} requests ({2,5:N1}%)" -f $variant, $count, $percent) -ForegroundColor $color
}

Write-Host ""
Write-Host "Expected Distribution:" -ForegroundColor Gray
Write-Host "  XGBoost:  ~70%" -ForegroundColor Gray
Write-Host "  CatBoost: ~30%" -ForegroundColor Gray
Write-Host ""
Write-Host ("="*70) -ForegroundColor Cyan