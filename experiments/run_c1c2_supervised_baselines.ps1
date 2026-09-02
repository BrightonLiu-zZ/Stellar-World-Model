# GENERATED FROM experiments/configs/c1c2_supervised_baselines.yaml - DO NOT EDIT (edit the manifest and re-run swm.exp.gen_supervised)
# c1c2_supervised_baselines - TRAINING, run in your own terminal (GPU + W&B online).
# Plan: docs/roadmap/2026-08-25-ml4ps-pivot-roadmap.md rows C1/C2. Boundary: docs/adr/0012-recipe-unfreeze-and-probe-boundary.md (external baselines, never the probe).
# 66 runs = 2 arms x 11 tasks x 3 seeds. Estimated 1.0 h total; pilot 12.5 min.
# Arm-major, task-priority order so a -MaxHours cutoff leaves the LEADING cells complete.
# INTERRUPT/RESUME: Ctrl-C anytime; DONE.txt markers skip finished runs on the next invocation.
# Usage:  cd C:\git_repo\Stellar-World-Model
#         .\experiments\run_c1c2_supervised_baselines.ps1 -PilotOnly   # 11 runs, then STOP for the gate
#         .\experiments\run_c1c2_supervised_baselines.ps1              # the full queue
#         .\experiments\run_c1c2_supervised_baselines.ps1 -DryRun

param(
  [switch]$DryRun,
  [switch]$PilotOnly,
  [double]$MaxHours = 3.0
)

$ErrorActionPreference = "Stop"
$env:PYTHONPATH = "src"
$env:PYTHONUNBUFFERED = "1"
$py = "C:\Users\user1\miniconda3\envs\swm\python.exe"
$repo = "C:\git_repo\Stellar-World-Model"
$manifest = "$repo\experiments\configs\c1c2_supervised_baselines.yaml"
$root = "$repo\experiments\c1c2_supervised"

# packed source for the v1 population; the pool replays from processed/sequences at first use
$packed = "$repo\experiments\exp01_window256_seq16\packed"
if (-not (Test-Path (Join-Path $packed "train_index.parquet"))) {
  throw "v1 packed source missing: $packed - run swm.data.pack first"
}

# (arm, task, seed). ALL config lives in the manifest; this file only orders the queue.
$pilot = @(
  @('conv_supervised', 'eb', 0),
  @('conv_supervised', 'transit', 0),
  @('conv_supervised', 'pulsating', 0),
  @('conv_supervised', 'rotation', 0),
  @('conv_supervised', 'solar_like_osc', 0),
  @('conv_supervised', 'flare', 0),
  @('conv_supervised', 'numax_hon', 0),
  @('conv_supervised', 'osc_giant', 0),
  @('conv_supervised', 'ijspeert', 0),
  @('conv_supervised', 'rotation_period', 0),
  @('conv_supervised', 'rgb_vs_heb', 0)
)

$full = @(
  @('conv_supervised', 'eb', 0),
  @('conv_supervised', 'eb', 1),
  @('conv_supervised', 'eb', 2),
  @('conv_supervised', 'transit', 0),
  @('conv_supervised', 'transit', 1),
  @('conv_supervised', 'transit', 2),
  @('conv_supervised', 'pulsating', 0),
  @('conv_supervised', 'pulsating', 1),
  @('conv_supervised', 'pulsating', 2),
  @('conv_supervised', 'rotation', 0),
  @('conv_supervised', 'rotation', 1),
  @('conv_supervised', 'rotation', 2),
  @('conv_supervised', 'solar_like_osc', 0),
  @('conv_supervised', 'solar_like_osc', 1),
  @('conv_supervised', 'solar_like_osc', 2),
  @('conv_supervised', 'flare', 0),
  @('conv_supervised', 'flare', 1),
  @('conv_supervised', 'flare', 2),
  @('conv_supervised', 'numax_hon', 0),
  @('conv_supervised', 'numax_hon', 1),
  @('conv_supervised', 'numax_hon', 2),
  @('conv_supervised', 'osc_giant', 0),
  @('conv_supervised', 'osc_giant', 1),
  @('conv_supervised', 'osc_giant', 2),
  @('conv_supervised', 'ijspeert', 0),
  @('conv_supervised', 'ijspeert', 1),
  @('conv_supervised', 'ijspeert', 2),
  @('conv_supervised', 'rotation_period', 0),
  @('conv_supervised', 'rotation_period', 1),
  @('conv_supervised', 'rotation_period', 2),
  @('conv_supervised', 'rgb_vs_heb', 0),
  @('conv_supervised', 'rgb_vs_heb', 1),
  @('conv_supervised', 'rgb_vs_heb', 2),
  @('mlp_raw', 'eb', 0),
  @('mlp_raw', 'eb', 1),
  @('mlp_raw', 'eb', 2),
  @('mlp_raw', 'transit', 0),
  @('mlp_raw', 'transit', 1),
  @('mlp_raw', 'transit', 2),
  @('mlp_raw', 'pulsating', 0),
  @('mlp_raw', 'pulsating', 1),
  @('mlp_raw', 'pulsating', 2),
  @('mlp_raw', 'rotation', 0),
  @('mlp_raw', 'rotation', 1),
  @('mlp_raw', 'rotation', 2),
  @('mlp_raw', 'solar_like_osc', 0),
  @('mlp_raw', 'solar_like_osc', 1),
  @('mlp_raw', 'solar_like_osc', 2),
  @('mlp_raw', 'flare', 0),
  @('mlp_raw', 'flare', 1),
  @('mlp_raw', 'flare', 2),
  @('mlp_raw', 'numax_hon', 0),
  @('mlp_raw', 'numax_hon', 1),
  @('mlp_raw', 'numax_hon', 2),
  @('mlp_raw', 'osc_giant', 0),
  @('mlp_raw', 'osc_giant', 1),
  @('mlp_raw', 'osc_giant', 2),
  @('mlp_raw', 'ijspeert', 0),
  @('mlp_raw', 'ijspeert', 1),
  @('mlp_raw', 'ijspeert', 2),
  @('mlp_raw', 'rotation_period', 0),
  @('mlp_raw', 'rotation_period', 1),
  @('mlp_raw', 'rotation_period', 2),
  @('mlp_raw', 'rgb_vs_heb', 0),
  @('mlp_raw', 'rgb_vs_heb', 1),
  @('mlp_raw', 'rgb_vs_heb', 2)
)

$runs = if ($PilotOnly) { $pilot } else { $full }
$sweepStart = Get-Date
$failed = @()
$i = 0
foreach ($r in $runs) {
  $i += 1
  $arm = $r[0]; $task = $r[1]; $seed = $r[2]
  $doneMarker = "$root\runs\$arm\$task\seed$seed\DONE.txt"
  $elapsedH = ((Get-Date) - $sweepStart).TotalHours

  if (Test-Path $doneMarker) {
    Write-Host "[$i/$($runs.Count)] SKIP $arm/$task seed $seed (DONE marker present)" -ForegroundColor DarkGray
    continue
  }
  if ($elapsedH -gt $MaxHours) {
    Write-Host "[$i/$($runs.Count)] CUTOFF: $([math]::Round($elapsedH,2)) h > MaxHours $MaxHours - stopping" -ForegroundColor Yellow
    break
  }

  $cmd = @("--manifest", $manifest, "--arm", $arm, "--task", $task, "--seed", "$seed")
  if ($DryRun) {
    Write-Host "[$i/$($runs.Count)] python -m swm.train.supervised $($cmd -join ' ')"
    continue
  }

  $elapsedMin = [math]::Round(((Get-Date) - $sweepStart).TotalMinutes, 1)
  Write-Host "===== [$i/$($runs.Count)] TRAIN $arm/$task seed $seed - ${elapsedMin} min elapsed =====" -ForegroundColor Cyan
  & $py -u -m swm.train.supervised @cmd
  if ($LASTEXITCODE -ne 0) {
    Write-Host "RETRY: $arm/$task seed $seed (exit $LASTEXITCODE)" -ForegroundColor Yellow
    & $py -u -m swm.train.supervised @cmd
  }
  if ($LASTEXITCODE -ne 0) {
    Write-Host "FAILED: $arm/$task seed $seed (exit $LASTEXITCODE) - continuing" -ForegroundColor Red
    $failed += "$arm/$task seed $seed"
  }
}

$total = [math]::Round(((Get-Date) - $sweepStart).TotalHours, 2)
if ($failed.Count -gt 0) {
  Write-Host "QUEUE DONE with $($failed.Count) FAILURES (${total} h): $($failed -join '; ')" -ForegroundColor Red
} elseif ($PilotOnly) {
  Write-Host "PILOT DONE (${total} h) - tell Claude Code 'c1c2 pilot done' to score the pre-registered gate." -ForegroundColor Green
} else {
  Write-Host "QUEUE DONE (${total} h) - tell Claude Code 'c1c2 queue done' to score the scorecard rows." -ForegroundColor Green
}
