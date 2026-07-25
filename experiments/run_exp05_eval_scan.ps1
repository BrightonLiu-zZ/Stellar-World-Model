# exp05 morning-after eval scan: readout_sweep over every finished exp05 cell x seed. Writes the
# first-segment window-mu caches (rollout_eval + transit probe consume them) and appends gap rows to
# each cell's results/readout_sweep.csv.
#
# All 12 exp05 cells are the STANDARD z128 / 256-window shape, so every one shares the exp03 z128
# untrained reference (experiments/exp03_eval_cache) - no per-cell capacity-matched cache needed
# (unlike exp04's encoder variants). readout_sweep builds that one untrained ref per invocation from
# the first matched cell's stored config.
#
# Ckpt = best_recon_aux only (mu selected lambda-independently in exp05). Readouts logistic+gbm
# (gbm-on-mu = the info_in_mu / readout-barrier diagnostic), pooling mean. Streams to console (no
# redirects) for the Claude Code monitor. Safe to re-run: mu caches make finished invocations near-instant.
#
# After this, run: python -m swm.eval.rollout_eval  (the rollout-vs-persistence "learned physics" test)
# and the probe-gap aggregation (lambda0 vs lambda>0, >2*SE across seeds).

$ErrorActionPreference = 'Continue'
Set-Location (Split-Path $PSScriptRoot -Parent)
$py = 'C:\Users\user1\miniconda3\envs\swm\python.exe'
$env:PYTHONPATH = 'src'
$env:PYTHONUNBUFFERED = '1'

$std = 'experiments/exp03_eval_cache' # shared z128 untrained reference for all exp05 cells

$failed = @()
$t0 = Get-Date
foreach ($seed in 0..3) {
  $mins = [math]::Round(((Get-Date) - $t0).TotalMinutes, 1)
  Write-Host ("=== exp05_* seed {0} (cache {1}) - elapsed {2} min ===" -f $seed, $std, $mins) -ForegroundColor Cyan
  & $py -m swm.eval.readout_sweep --exp-glob 'exp05_*' --seed $seed --untrained-cache $std `
      --ckpts best_recon_aux --readouts logistic gbm --poolings mean
  if ($LASTEXITCODE -ne 0) {
    Write-Host ("FAILED: exp05_* seed {0} (exit {1})" -f $seed, $LASTEXITCODE) -ForegroundColor Red
    $failed += ("exp05_* seed {0}" -f $seed)
  }
}

$mins = [math]::Round(((Get-Date) - $t0).TotalMinutes, 1)
Write-Host ("=== scan done in {0} min; {1} failed ===" -f $mins, $failed.Count) -ForegroundColor Cyan
$failed | ForEach-Object { Write-Host "  $_" -ForegroundColor Red }
exit $failed.Count
