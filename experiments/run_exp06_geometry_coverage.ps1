# GENERATED FROM experiments/configs/exp06_geometry_coverage.yaml - DO NOT EDIT (edit the manifest and re-run swm.exp.gen_sweep)
# exp06_geometry_coverage - TRAINING ONLY, run in your own terminal (GPU + W&B online). Plan: docs/plans/2026-07-27-exp06-geometry-coverage.md
# 6 cells, seed-major order so a -MaxHours cutoff leaves every cell with even seed coverage.
# Estimated per-run minutes: {'ep100': 25, 'ep60': 15} (VERIFY against the first runs; exp05-derived).
# INTERRUPT/RESUME: Ctrl-C anytime; train.resume=true + last.pt resume mid-run; DONE.txt skips finished runs.
# Usage:  cd C:\git_repo\Stellar-World-Model ; .\experiments\run_exp06_geometry_coverage.ps1
#         .\experiments\run_exp06_geometry_coverage.ps1 -DryRun
#         .\experiments\run_exp06_geometry_coverage.ps1 -MaxHours 20.0

param(
  [switch]$DryRun,
  [double]$MaxHours = 20.0
)

$ErrorActionPreference = "Stop"
$env:PYTHONPATH = "src"
$env:PYTHONUNBUFFERED = "1"
$py = "C:\Users\user1\miniconda3\envs\swm\python.exe"
$repo = "C:\git_repo\Stellar-World-Model"

# packed source per geometry (junctioned per cell below); packs must exist BEFORE training starts
$packedSource = @{
  'w1024' = "$repo\experiments\exp00_window1024_seq4\packed"
  'w2048' = "$repo\experiments\exp06_window2048_seq2\packed"
  'w256' = "$repo\experiments\exp01_window256_seq16\packed"
  'w512' = "$repo\experiments\exp06_window512_seq8\packed"
}
foreach ($g in $packedSource.Keys) {
  if (-not (Test-Path (Join-Path $packedSource[$g] "train_index.parquet"))) {
    throw "packed source for $g missing: $($packedSource[$g]) - run swm.data.pack first"
  }
}

# cells: (exp_name, geometry, seed list). ALL config lives in src/swm/configs/experiment/exp06/<cell>.yaml.
$cells = @(
  @('exp06_edge_comb_fb0p02', 'w256', @(0)),
  @('exp06_w256_off', 'w256', @(0, 1, 2, 3, 4, 5)),
  @('exp06_w256_fbwd', 'w256', @(0, 1, 2, 3, 4, 5)),
  @('exp06_w512_off', 'w512', @(0, 1, 2, 3, 4, 5)),
  @('exp06_w1024_off', 'w1024', @(0, 1, 2, 3, 4, 5)),
  @('exp06_w2048_off', 'w2048', @(0, 1, 2, 3, 4, 5))
)

# seed-major expansion
$runs = @()
foreach ($seed in @(0, 1, 2, 3, 4, 5)) {
  foreach ($c in $cells) {
    if ($c[2] -contains $seed) { $runs += , @($c[0], $c[1], $seed) }
  }
}

$sweepStart = Get-Date
$failed = @()
$i = 0
foreach ($r in $runs) {
  $i += 1
  $exp = $r[0]; $geom = $r[1]; $seed = $r[2]
  $comboDir = "$repo\experiments\$exp"
  $doneMarker = "$comboDir\models\B_seed$seed\DONE.txt"
  $elapsedH = ((Get-Date) - $sweepStart).TotalHours

  if (Test-Path $doneMarker) {
    Write-Host "[$i/$($runs.Count)] SKIP $exp seed $seed (DONE marker present)" -ForegroundColor DarkGray
    continue
  }
  if ($elapsedH -gt $MaxHours) {
    Write-Host "[$i/$($runs.Count)] CUTOFF: $([math]::Round($elapsedH,2)) h > MaxHours $MaxHours - stopping" -ForegroundColor Yellow
    break
  }

  $cmd = @("+experiment=exp06/$exp", "variant=B", "seed=$seed")
  if ($DryRun) {
    Write-Host "[$i/$($runs.Count)] [$geom] python -m swm.train $($cmd -join ' ')"
    continue
  }

  if (-not (Test-Path $comboDir)) { New-Item -ItemType Directory -Path $comboDir | Out-Null }
  $packedLink = Join-Path $comboDir "packed"
  if (-not (Test-Path $packedLink)) {
    New-Item -ItemType Junction -Path $packedLink -Target $packedSource[$geom] | Out-Null
  }

  $elapsedMin = [math]::Round(((Get-Date) - $sweepStart).TotalMinutes, 1)
  Write-Host "===== [$i/$($runs.Count)] [$geom] TRAIN $exp seed $seed - ${elapsedMin} min elapsed =====" -ForegroundColor Cyan
  & $py -u -m swm.train @cmd
  if ($LASTEXITCODE -ne 0) {
    Write-Host "RETRY: $exp seed $seed (exit $LASTEXITCODE)" -ForegroundColor Yellow
    & $py -u -m swm.train @cmd
  }
  if ($LASTEXITCODE -ne 0) {
    Write-Host "FAILED: $exp seed $seed (exit $LASTEXITCODE) - continuing" -ForegroundColor Red
    $failed += "$exp seed $seed"
    continue
  }
  Set-Content -Path $doneMarker -Value ("finished " + (Get-Date -Format o)) -Encoding utf8
}

$total = [math]::Round(((Get-Date) - $sweepStart).TotalHours, 2)
if ($failed.Count -gt 0) {
  Write-Host "SWEEP DONE with $($failed.Count) FAILURES (${total} h): $($failed -join '; ')" -ForegroundColor Red
} else {
  Write-Host "SWEEP DONE (${total} h) - tell Claude Code 'exp06_geometry_coverage done' to run the eval fan." -ForegroundColor Green
}
