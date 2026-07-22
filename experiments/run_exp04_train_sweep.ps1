# exp04 overnight sweep - TRAINING ONLY, run in your own terminal (GPU + W&B online).
# Plan: docs/plans/2026-07-19-exp04-confirm-encoder-kl.md. 33 ranked runs + 6-slot bonus tail:
#   [1] 3-seed confirm of the exp03 winner + 2 runners-up (seeds 1-2)         6 runs
#   [2] KL-corner cross beta {0.05,0.1,0.2} x fb {0.01,0.02,0.04} minus center 8 runs
#   [3] encoder tier 1 (z64, z32) x seeds 0,1                                  4 runs
#   [4] 100-epoch winner under-run probe                                       1 run
#   [5] encoder tier 2 (k9, k15, d5) x seeds 0,1                               6 runs
#   [6] encoder tier 3+combo (w2x, whalf, d3, z64k9) x seeds 0,1               8 runs
#   [7] BONUS if time remains: corner seed-1 repeats, z64/z32 seed 2           6 runs
# ~16.5 min/run (measured on exp03) -> guaranteed blocks ~9.5 h, bonus ~1.7 h.
#
# INTERRUPT/RESUME: Ctrl-C anytime. Every run trains with train.resume=true, and last.pt stores
# model/optimizer/scaler/RNG/bests, so re-running this script resumes the interrupted run mid-training
# and skips runs already finished (a DONE.txt marker is written on clean exit; finished vs interrupted
# cannot be told apart from last.pt alone because early stop also leaves epoch < max_epochs).
# A resumed run shows up as a second W&B run; the killed first part is ignored by the curve dump.
#
# Usage:  cd C:\git_repo\Stellar-World-Model ; .\experiments\run_exp04_train_sweep.ps1
#         .\experiments\run_exp04_train_sweep.ps1 -DryRun        # print the manifest, train nothing
#         .\experiments\run_exp04_train_sweep.ps1 -MaxHours 10   # stop launching new runs past cutoff
# After it finishes (or you stop it for good), tell Claude Code "exp04 done" -> it runs the eval fan.

param(
  [switch]$DryRun,
  [double]$MaxHours = 12.0
)

$ErrorActionPreference = "Stop"
$env:PYTHONPATH = "src"
$env:PYTHONUNBUFFERED = "1"
$py = "C:\Users\user1\miniconda3\envs\swm\python.exe"
$repo = "C:\git_repo\Stellar-World-Model"
$packedSource = "$repo\experiments\exp01_window256_seq16\packed"

# Manifest entry: exp_name, seed, override list (experiment ref + knobs). Ranked order = plan order.
$runs = @(
  # [1] 3-seed confirm (existing exp03 configs, identity with seed 0 preserved)
  @("exp03_fb0p02_b0p1_lpsd", 1, @("+experiment=exp03_klsweep", "train.beta_target=0.1", "train.free_bits=0.02", "train.recon_aux.type=log_psd", "train.recon_aux.weight=0.1"), "confirm"),
  @("exp03_fb0p02_b0p1_lpsd", 2, @("+experiment=exp03_klsweep", "train.beta_target=0.1", "train.free_bits=0.02", "train.recon_aux.type=log_psd", "train.recon_aux.weight=0.1"), "confirm"),
  @("exp03_fb0_b0p1_comb",    1, @("+experiment=exp03_klsweep", "train.beta_target=0.1", "train.free_bits=0.0",  "train.recon_aux.type=combined", "train.recon_aux.weight=0.3"), "confirm"),
  @("exp03_fb0_b0p1_comb",    2, @("+experiment=exp03_klsweep", "train.beta_target=0.1", "train.free_bits=0.0",  "train.recon_aux.type=combined", "train.recon_aux.weight=0.3"), "confirm"),
  @("exp03_fb0p05_b0p3_lpsd", 1, @("+experiment=exp03_klsweep", "train.beta_target=0.3", "train.free_bits=0.05", "train.recon_aux.type=log_psd", "train.recon_aux.weight=0.1"), "confirm"),
  @("exp03_fb0p05_b0p3_lpsd", 2, @("+experiment=exp03_klsweep", "train.beta_target=0.3", "train.free_bits=0.05", "train.recon_aux.type=log_psd", "train.recon_aux.weight=0.1"), "confirm"),
  # [2] KL-corner cross (exp04/base = winner recipe; only fb/beta override)
  @("exp04_fb0p01_b0p05_lpsd", 0, @("+experiment=exp04/base", "train.free_bits=0.01", "train.beta_target=0.05"), "corner"),
  @("exp04_fb0p02_b0p05_lpsd", 0, @("+experiment=exp04/base", "train.free_bits=0.02", "train.beta_target=0.05"), "corner"),
  @("exp04_fb0p04_b0p05_lpsd", 0, @("+experiment=exp04/base", "train.free_bits=0.04", "train.beta_target=0.05"), "corner"),
  @("exp04_fb0p01_b0p1_lpsd",  0, @("+experiment=exp04/base", "train.free_bits=0.01", "train.beta_target=0.1"),  "corner"),
  @("exp04_fb0p04_b0p1_lpsd",  0, @("+experiment=exp04/base", "train.free_bits=0.04", "train.beta_target=0.1"),  "corner"),
  @("exp04_fb0p01_b0p2_lpsd",  0, @("+experiment=exp04/base", "train.free_bits=0.01", "train.beta_target=0.2"),  "corner"),
  @("exp04_fb0p02_b0p2_lpsd",  0, @("+experiment=exp04/base", "train.free_bits=0.02", "train.beta_target=0.2"),  "corner"),
  @("exp04_fb0p04_b0p2_lpsd",  0, @("+experiment=exp04/base", "train.free_bits=0.04", "train.beta_target=0.2"),  "corner"),
  # [3] encoder tier 1 - latent bottleneck
  @("exp04_enc_z64", 0, @("+experiment=exp04/enc_z64"), "enc-tier1"),
  @("exp04_enc_z64", 1, @("+experiment=exp04/enc_z64"), "enc-tier1"),
  @("exp04_enc_z32", 0, @("+experiment=exp04/enc_z32"), "enc-tier1"),
  @("exp04_enc_z32", 1, @("+experiment=exp04/enc_z32"), "enc-tier1"),
  # [4] under-run probe
  @("exp04_winner_ep100", 0, @("+experiment=exp04/winner_ep100"), "probe"),
  # [5] encoder tier 2 - receptive field
  @("exp04_enc_k9",  0, @("+experiment=exp04/enc_k9"),  "enc-tier2"),
  @("exp04_enc_k9",  1, @("+experiment=exp04/enc_k9"),  "enc-tier2"),
  @("exp04_enc_k15", 0, @("+experiment=exp04/enc_k15"), "enc-tier2"),
  @("exp04_enc_k15", 1, @("+experiment=exp04/enc_k15"), "enc-tier2"),
  @("exp04_enc_d5",  0, @("+experiment=exp04/enc_d5"),  "enc-tier2"),
  @("exp04_enc_d5",  1, @("+experiment=exp04/enc_d5"),  "enc-tier2"),
  # [6] encoder tier 3 + interaction combo
  @("exp04_enc_w2x",    0, @("+experiment=exp04/enc_w2x"),    "enc-tier3"),
  @("exp04_enc_w2x",    1, @("+experiment=exp04/enc_w2x"),    "enc-tier3"),
  @("exp04_enc_whalf",  0, @("+experiment=exp04/enc_whalf"),  "enc-tier3"),
  @("exp04_enc_whalf",  1, @("+experiment=exp04/enc_whalf"),  "enc-tier3"),
  @("exp04_enc_d3",     0, @("+experiment=exp04/enc_d3"),     "enc-tier3"),
  @("exp04_enc_d3",     1, @("+experiment=exp04/enc_d3"),     "enc-tier3"),
  @("exp04_enc_z64k9",  0, @("+experiment=exp04/enc_z64k9"),  "enc-tier3"),
  @("exp04_enc_z64k9",  1, @("+experiment=exp04/enc_z64k9"),  "enc-tier3"),
  # [7] bonus tail (pre-ranked; only reached if the night runs clean and under MaxHours)
  @("exp04_fb0p02_b0p05_lpsd", 1, @("+experiment=exp04/base", "train.free_bits=0.02", "train.beta_target=0.05"), "bonus"),
  @("exp04_fb0p02_b0p2_lpsd",  1, @("+experiment=exp04/base", "train.free_bits=0.02", "train.beta_target=0.2"),  "bonus"),
  @("exp04_fb0p01_b0p05_lpsd", 1, @("+experiment=exp04/base", "train.free_bits=0.01", "train.beta_target=0.05"), "bonus"),
  @("exp04_fb0p01_b0p1_lpsd",  1, @("+experiment=exp04/base", "train.free_bits=0.01", "train.beta_target=0.1"),  "bonus"),
  @("exp04_enc_z64", 2, @("+experiment=exp04/enc_z64"), "bonus"),
  @("exp04_enc_z32", 2, @("+experiment=exp04/enc_z32"), "bonus")
)

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
    # one immediate retry: transient WinError 8 (memmap commit exhaustion) hit 2/12 smoke runs and is
    # recoverable; train.resume=true means the retry continues mid-run, losing at most one epoch
    Write-Host "RETRY: $exp seed $seed (exit $LASTEXITCODE)" -ForegroundColor Yellow
    & $py -u -m swm.train @cmd
  }
  if ($LASTEXITCODE -ne 0) {
    # continue-on-failure (changed from exp03's break): one bad cell must not kill the night
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
  Write-Host "SWEEP DONE (${total} h) - tell Claude Code 'exp04 done' to run the eval fan + aggregation." -ForegroundColor Green
}
