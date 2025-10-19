# Complete DVC Pipeline Runner
param(
    [switch]$FullRun,
    [switch]$SkipPull,
    [switch]$SkipPush,
    [string]$Stage
)

$ErrorActionPreference = "Continue"

Write-Host ("="*80) -ForegroundColor Cyan
Write-Host "DVC ML Pipeline Runner" -ForegroundColor Green
Write-Host ("="*80) -ForegroundColor Cyan
Write-Host ""

# Function to print section headers
function Print-Section {
    param([string]$Message)
    Write-Host "`n$Message" -ForegroundColor Yellow
    Write-Host ("-" * $Message.Length) -ForegroundColor Yellow
}

# Check DVC installation
Print-Section "[1/8] Checking DVC installation..."
$dvcInstalled = Get-Command dvc -ErrorAction SilentlyContinue
if ($dvcInstalled) {
    Write-Host "✓ DVC is installed" -ForegroundColor Green
    dvc version
} else {
    Write-Host "✗ DVC not found. Please install: pip install dvc dvc-s3" -ForegroundColor Red
    exit 1
}

# Check DVC status
Print-Section "[2/8] Checking DVC status..."
dvc status

# Pull latest data from S3 (unless skipped)
if (-not $SkipPull) {
    Print-Section "[3/8] Pulling latest data from S3..."
    dvc pull
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ Data pulled successfully" -ForegroundColor Green
    } else {
        Write-Host "⚠ Pull failed or no changes" -ForegroundColor Yellow
    }
} else {
    Write-Host "`n[3/8] Skipping data pull..." -ForegroundColor Yellow
}

# Run pipeline
if ($Stage) {
    Print-Section "[4/8] Running specific stage: $Stage..."
    dvc repro $Stage -v
} elseif ($FullRun) {
    Print-Section "[4/8] Running full pipeline (force)..."
    dvc repro -f -v
} else {
    Print-Section "[4/8] Running pipeline (incremental)..."
    dvc repro -v
}

if ($LASTEXITCODE -ne 0) {
    Write-Host "`n✗ Pipeline failed!" -ForegroundColor Red
    exit 1
}

Write-Host "`n✓ Pipeline completed successfully" -ForegroundColor Green

# Show metrics
Print-Section "[5/8] Pipeline Metrics"
dvc metrics show

# Show differences
Print-Section "[6/8] Metric Differences"
dvc metrics diff

# Show pipeline DAG
Print-Section "[7/8] Pipeline DAG"
dvc dag

# Push results to S3 (unless skipped)
if (-not $SkipPush) {
    Print-Section "[8/8] Pushing results to S3..."
    dvc push
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ Results pushed successfully" -ForegroundColor Green
    } else {
        Write-Host "⚠ Push failed" -ForegroundColor Yellow
    }
} else {
    Write-Host "`n[8/8] Skipping results push..." -ForegroundColor Yellow
}

# Generate report
Write-Host ""
Print-Section "Generating Reports"
dvc dag --md | Out-File -FilePath pipeline_dag.md -Encoding UTF8
Write-Host "✓ Pipeline DAG saved to pipeline_dag.md" -ForegroundColor Green

# Summary
Write-Host ""
Write-Host ("="*80) -ForegroundColor Cyan
Write-Host "Pipeline Complete!" -ForegroundColor Green
Write-Host ("="*80) -ForegroundColor Cyan

Write-Host "`nGenerated Files:" -ForegroundColor Cyan
Write-Host "  • Metrics: metrics/" -ForegroundColor White
Write-Host "  • Models: models/" -ForegroundColor White
Write-Host "  • Plots: plots/" -ForegroundColor White
Write-Host "  • Pipeline DAG: pipeline_dag.md" -ForegroundColor White
Write-Host "  • S3 Storage: s3://triton-models-71544/dvc-storage/" -ForegroundColor White

Write-Host "`nUseful Commands:" -ForegroundColor Cyan
Write-Host "  dvc metrics show     - View all metrics" -ForegroundColor White
Write-Host "  dvc metrics diff     - Compare with previous run" -ForegroundColor White
Write-Host "  dvc plots show       - View plots" -ForegroundColor White
Write-Host "  dvc dag              - View pipeline structure" -ForegroundColor White
Write-Host "  dvc repro [stage]    - Run specific stage" -ForegroundColor White

Write-Host ""
Write-Host ("="*80) -ForegroundColor Cyan