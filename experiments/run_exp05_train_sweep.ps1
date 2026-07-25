# exp05 dynamics-axis sweep - TRAINING ONLY, run in your own terminal (GPU + W&B online).
# Plan: docs/plans/2026-07-22-exp05-dynamics-axis.md. 12 cells x 4 seeds = 48 runs, SEED-MAJOR order
# (all 12 cells at seed 0, then seed 1, ...) so a -MaxHours cutoff leaves every cell with >=3 seeds.
#   comb (fb0/b0.1/combined w0.3):  off(l0) . fwd x{0.1,0.3,1.0} . fwd_bwd x{...} . multistep x{...}  = 10
#   lpsd hedge (fb0.02/b0.1/log_psd): off(l0) . multistep@1.0                                          =  2
# lambda = target dyn-contribution {0.1,0.3,1.0} x recon(~0.85) / dyn_mode_steady (per-mode, from smoke).
# All fresh (no exp04 reuse); cosine LR decay; mu selected lambda-independently (recon+aux).
# MEASURED per-epoch (smoke 2026-07-23, 60 epochs + ~15 s init): fwd 12.1 s/ep -> ~12.4 min/run;
# fwd_bwd ~15 s/ep -> ~15 min/run; multistep(15-step rollout) 19.7-20.0 s/ep -> ~20 min/run.
# => ~187 min per full seed pass; 4 seeds ~12.5 h. Default -MaxHours 12 caps the night: seeds 0-2 finish
# at ~9.4 h and seed 3 gets through ~11 of 12 cells, so EVERY cell ends with >=3 seeds and most with 4
# (this is exactly why the order is seed-major with the lpsd hedge last). Early stopping (patience 10)
# only makes it faster. Re-run the script another night to fill in whatever the cutoff skipped.
#
# INTERRUPT/RESUME: Ctrl-C anytime. train.resume=true + last.pt (model/opt/scaler/scheduler/RNG/bests)
# resume the interrupted run mid-training; a DONE.txt marker skips finished runs. Just re-run the same
# command - it picks up where it stopped. A resumed run appears as a 2nd W&B run.
#
# Usage:  cd C:\git_repo\Stellar-World-Model ; .\experiments\run_exp05_train_sweep.ps1
#         .\experiments\run_exp05_train_sweep.ps1 -DryRun        # print the 48-run manifest, train nothing
#         .\experiments\run_exp05_train_sweep.ps1 -MaxHours 12   # stop launching new runs past cutoff
# After it finishes (or you stop it), tell Claude Code "exp05 done" -> it runs the eval fan + rollout eval.

param(
  [switch]$DryRun,
  [double]$MaxHours = 12.0
)

$ErrorActionPreference = "Stop"
$env:PYTHONPATH = "src"
$env:PYTHONUNBUFFERED = "1"
$py = "C:\Users\user1\miniconda3\envs\swm\python.exe"
$repo = "C:\git_repo\Stellar-World-Model"
$packedSource = "$repo\experiments\exp01_window256_seq16\packed" # exp05 shares the exp01 256/16 packed windows

# ==== LAMBDA TABLE - CALIBRATED 2026-07-23 from the pre-launch smoke + exp04 ====
# lambda = target_contribution * recon / dyn_mode_steady   (dyn is flat after ~ep5, so the 14-epoch
# smoke and the 60-epoch exp04 values are directly comparable).
#   comb fwd       dyn 0.0076 (exp04, 3 seeds), recon ~1.00
#   comb fwd_bwd   dyn 0.0152 (= 2x fwd; the two one-step terms are summed under one lambda)
#   comb multistep dyn 0.0210 (smoke 2026-07-23), recon 1.028
#   lpsd multistep dyn 0.0283 (smoke 2026-07-23), recon 0.931  -> its OWN lambda, not comb's
# NOTE the achieved contribution UNDERSHOOTS the target at high lambda (raising lambda drives dyn down);
# that is expected - the axis still spans ~2 orders of magnitude and the collapse monitor shows the bite.
$lam = @{
  'comb_fwd_0.1'   = 13;  'comb_fwd_0.3'   = 40;  'comb_fwd_1.0'   = 130
  'comb_fbwd_0.1'  = 7;   'comb_fbwd_0.3'  = 20;  'comb_fbwd_1.0'  = 66
  'comb_multi_0.1' = 5;   'comb_multi_0.3' = 14;  'comb_multi_1.0' = 48
  'lpsd_multi_1.0' = 32
}

# Cells (exp_name, override list, tag) - within-seed order = scientific priority (nulls + flagship first,
# lpsd hedge last), so a mid-last-seed cutoff drops the least important runs.
$comb = "+experiment=exp05/base_comb"
$lpsd = "+experiment=exp05/base_lpsd"
$cells = @(
  @("exp05_comb_off",      @($comb, "model.dyn_mode=fwd",       "train.lambda_dyn=0"),                       "comb-off"),
  @("exp05_comb_multi_c1p0",@($comb,"model.dyn_mode=multistep", "train.lambda_dyn=$($lam['comb_multi_1.0'])"),"comb-multi"),
  @("exp05_comb_fwd_c1p0", @($comb, "model.dyn_mode=fwd",       "train.lambda_dyn=$($lam['comb_fwd_1.0'])"), "comb-fwd"),
  @("exp05_comb_fbwd_c1p0",@($comb, "model.dyn_mode=fwd_bwd",   "train.lambda_dyn=$($lam['comb_fbwd_1.0'])"),"comb-fbwd"),
  @("exp05_comb_multi_c0p3",@($comb,"model.dyn_mode=multistep", "train.lambda_dyn=$($lam['comb_multi_0.3'])"),"comb-multi"),
  @("exp05_comb_fwd_c0p3", @($comb, "model.dyn_mode=fwd",       "train.lambda_dyn=$($lam['comb_fwd_0.3'])"), "comb-fwd"),
  @("exp05_comb_fbwd_c0p3",@($comb, "model.dyn_mode=fwd_bwd",   "train.lambda_dyn=$($lam['comb_fbwd_0.3'])"),"comb-fbwd"),
  @("exp05_comb_multi_c0p1",@($comb,"model.dyn_mode=multistep", "train.lambda_dyn=$($lam['comb_multi_0.1'])"),"comb-multi"),
  @("exp05_comb_fwd_c0p1", @($comb, "model.dyn_mode=fwd",       "train.lambda_dyn=$($lam['comb_fwd_0.1'])"), "comb-fwd"),
  @("exp05_comb_fbwd_c0p1",@($comb, "model.dyn_mode=fwd_bwd",   "train.lambda_dyn=$($lam['comb_fbwd_0.1'])"),"comb-fbwd"),
  @("exp05_lpsd_off",      @($lpsd, "model.dyn_mode=fwd",       "train.lambda_dyn=0"),                       "lpsd-off"),
  @("exp05_lpsd_multi_c1p0",@($lpsd,"model.dyn_mode=multistep", "train.lambda_dyn=$($lam['lpsd_multi_1.0'])"),"lpsd-multi")
)

# Seed-major expansion: seed 0 over all 12 cells, then seed 1, ... so a cutoff leaves uniform coverage.
$runs = @()
foreach ($seed in 0..3) {
  foreach ($c in $cells) {
    $runs += , @($c[0], $seed, $c[1], $c[2])
  }
}

$sweepStart = Get-Date
$failed = @()
$i = 0
foreach ($r in $runs) {
  $i += 1
  $exp = $r[0]; $seed = $r[1]; $overrides = $r[2]; $block = $r[3]

  $comboDir = "$repo\experiments\$exp"
  $doneMarker = "$comboDir\models\B_seed$seed\DONE.txt"
  $elapsedH = ((Get-Date) - $sweepStart).TotalHours

  if (Test-Path $doneMarker) {
    Write-Host "[$i/$($runs.Count)] SKIP $exp seed $seed (DONE marker present)" -ForegroundColor DarkGray
    continue
  }
  if ($elapsedH -gt $MaxHours) {
    Write-Host "[$i/$($runs.Count)] CUTOFF: $([math]::Round($elapsedH,2)) h elapsed > MaxHours $MaxHours - not launching $exp seed $seed or anything after it" -ForegroundColor Yellow
    break
  }

  $cmd = @($overrides) + @("exp_name=$exp", "variant=B", "seed=$seed")
  if ($DryRun) {
    Write-Host "[$i/$($runs.Count)] [$block] python -m swm.train $($cmd -join ' ')"
    continue
  }

  if (-not (Test-Path $comboDir)) { New-Item -ItemType Directory -Path $comboDir | Out-Null }
  $packedLink = Join-Path $comboDir "packed"
  if (-not (Test-Path $packedLink)) {
    New-Item -ItemType Junction -Path $packedLink -Target $packedSource | Out-Null
  }

  $elapsedMin = [math]::Round(((Get-Date) - $sweepStart).TotalMinutes, 1)
  Write-Host "===== [$i/$($runs.Count)] [$block] TRAIN $exp seed $seed - ${elapsedMin} min elapsed =====" -ForegroundColor Cyan
  & $py -u -m swm.train @cmd
  if ($LASTEXITCODE -ne 0) {
    # one immediate retry: transient WinError 8 (memmap commit exhaustion) is recoverable; resume=true
    # means the retry continues mid-run, losing at most one epoch.
    Write-Host "RETRY: $exp seed $seed (exit $LASTEXITCODE)" -ForegroundColor Yellow
    & $py -u -m swm.train @cmd
  }
  if ($LASTEXITCODE -ne 0) {
    Write-Host "FAILED: $exp seed $seed (exit $LASTEXITCODE) - continuing with the next run" -ForegroundColor Red
    $failed += "$exp seed $seed"
    continue
  }
  Set-Content -Path $doneMarker -Value ("finished " + (Get-Date -Format o)) -Encoding utf8
}

$total = [math]::Round(((Get-Date) - $sweepStart).TotalHours, 2)
if ($failed.Count -gt 0) {
  Write-Host "SWEEP DONE with $($failed.Count) FAILURES (${total} h): $($failed -join '; ')" -ForegroundColor Red
} else {
  Write-Host "SWEEP DONE (${total} h) - tell Claude Code 'exp05 done' to run the eval fan + rollout eval + aggregation." -ForegroundColor Green
}
