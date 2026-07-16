# exp03 KL-schedule x objective sweep - TRAINING ONLY, run in your own terminal (GPU + W&B online).
# 36 combos = free_bits {0, 0.02, 0.05, 0.1} x beta_target {0.1, 0.3, 1.0} x objective {none, logpsd_amp w0.1,
# combined w0.3}, variant B / seed 0, window 256 / seq_len 16 (junction-reuse exp01 packed), dual checkpoints
# (best.pt legacy monitor + best_recon_aux.pt KL-free) per the exp03 forensic (experiments/exp03_forensics/).
# Runs execute in RANKED order (controls + likely-informative cells first) so a morning Ctrl-C still leaves
# the decisive cells trained. Budget estimate ~9-12 h on the RTX 4060 (~10-20 min/run, max_epochs 60).
# After it finishes (or you stop it), tell Claude Code "sweep done" -- it runs the eval fan + aggregation.
#
# Usage:  cd C:\git_repo\Stellar-World-Model ; .\experiments\run_exp03_train_sweep.ps1

$ErrorActionPreference = "Stop"
$env:PYTHONPATH = "src"
$env:PYTHONUNBUFFERED = "1"
$py = "C:\Users\user1\miniconda3\envs\swm\python.exe"
$packedSource = "C:\git_repo\Stellar-World-Model\experiments\exp01_window256_seq16\packed"

# objective slug -> extra overrides (none = pure time-MSE control, defaults already type=none/weight=0)
$objectiveOverrides = @{
  "none" = @()
  "lpsd" = @("train.recon_aux.type=log_psd", "train.recon_aux.weight=0.1")
  "comb" = @("train.recon_aux.type=combined", "train.recon_aux.weight=0.3")
}

# Ranked combo list: fb = free_bits, b = beta_target, suffix = objective.
# Block A: controls + free-bits axis on the plain objective (incl. the exact exp01-replica cell fb0p1_b1p0_none).
# Block B: objective x {floor, beta} interaction cells (the H4 payoff region).
# Block C: grid completion.
$combos = @(
#  @("exp03_fb0p1_b1p0_none",  1.0,  0.1,  "none"),
#  @("exp03_fb0_b0p3_none",    0.3,  0.0,  "none"),
#  @("exp03_fb0p02_b0p3_none", 0.3,  0.02, "none"),
#  @("exp03_fb0p05_b0p3_none", 0.3,  0.05, "none"),
#  @("exp03_fb0p02_b0p1_none", 0.1,  0.02, "none"),
#  @("exp03_fb0p02_b1p0_none", 1.0,  0.02, "none"),
#  @("exp03_fb0p02_b0p3_lpsd", 0.3,  0.02, "lpsd"),
#  @("exp03_fb0p02_b0p3_comb", 0.3,  0.02, "comb"),
#  @("exp03_fb0_b0p3_lpsd",    0.3,  0.0,  "lpsd"),
#  @("exp03_fb0_b0p3_comb",    0.3,  0.0,  "comb"),
#  @("exp03_fb0p05_b0p3_lpsd", 0.3,  0.05, "lpsd"),
#  @("exp03_fb0p05_b0p3_comb", 0.3,  0.05, "comb"),
#  @("exp03_fb0p02_b0p1_lpsd", 0.1,  0.02, "lpsd"),
#  @("exp03_fb0p02_b0p1_comb", 0.1,  0.02, "comb"),
  @("exp03_fb0_b0p1_lpsd",    0.1,  0.0,  "lpsd"),
  @("exp03_fb0_b0p1_comb",    0.1,  0.0,  "comb"),
  @("exp03_fb0p05_b0p1_lpsd", 0.1,  0.05, "lpsd"),
  @("exp03_fb0p05_b0p1_comb", 0.1,  0.05, "comb"),
  @("exp03_fb0_b0p1_none",    0.1,  0.0,  "none"),
  @("exp03_fb0_b1p0_none",    1.0,  0.0,  "none"),
  @("exp03_fb0p05_b0p1_none", 0.1,  0.05, "none"),
  @("exp03_fb0p05_b1p0_none", 1.0,  0.05, "none"),
  @("exp03_fb0p1_b0p1_none",  0.1,  0.1,  "none"),
  @("exp03_fb0p1_b0p3_none",  0.3,  0.1,  "none"),
  @("exp03_fb0_b1p0_lpsd",    1.0,  0.0,  "lpsd"),
  @("exp03_fb0p02_b1p0_lpsd", 1.0,  0.02, "lpsd"),
  @("exp03_fb0p05_b1p0_lpsd", 1.0,  0.05, "lpsd"),
  @("exp03_fb0p1_b0p1_lpsd",  0.1,  0.1,  "lpsd"),
  @("exp03_fb0p1_b0p3_lpsd",  0.3,  0.1,  "lpsd"),
  @("exp03_fb0p1_b1p0_lpsd",  1.0,  0.1,  "lpsd"),
  @("exp03_fb0_b1p0_comb",    1.0,  0.0,  "comb"),
  @("exp03_fb0p02_b1p0_comb", 1.0,  0.02, "comb"),
  @("exp03_fb0p05_b1p0_comb", 1.0,  0.05, "comb"),
  @("exp03_fb0p1_b0p1_comb",  0.1,  0.1,  "comb"),
  @("exp03_fb0p1_b0p3_comb",  0.3,  0.1,  "comb"),
  @("exp03_fb0p1_b1p0_comb",  1.0,  0.1,  "comb")
)

$sweepStart = Get-Date
$i = 0
foreach ($c in $combos) {
  $i += 1
  $combo = $c[0]; $beta = $c[1]; $freeBits = $c[2]; $objective = $c[3]

  # isolated experiment folder; packed is a junction back to the shared exp01 windows
  $comboDir = "C:\git_repo\Stellar-World-Model\experiments\$combo"
  if (-not (Test-Path $comboDir)) { New-Item -ItemType Directory -Path $comboDir | Out-Null }
  $packedLink = Join-Path $comboDir "packed"
  if (-not (Test-Path $packedLink)) {
    New-Item -ItemType Junction -Path $packedLink -Target $packedSource | Out-Null
  }

  $elapsed = [math]::Round(((Get-Date) - $sweepStart).TotalMinutes, 1)
  Write-Host "===== [$i/36] TRAIN $combo (beta=$beta free_bits=$freeBits obj=$objective) - ${elapsed} min elapsed =====" -ForegroundColor Cyan
  $overrides = @("+experiment=exp03_klsweep", "exp_name=$combo",
                 "train.beta_target=$beta", "train.free_bits=$freeBits") + $objectiveOverrides[$objective]
  & $py -u -m swm.train @overrides variant=B seed=0 train.resume=false
  if ($LASTEXITCODE -ne 0) { Write-Host "FAILED: $combo (exit $LASTEXITCODE)" -ForegroundColor Red; break }
}
$total = [math]::Round(((Get-Date) - $sweepStart).TotalHours, 2)
Write-Host "SWEEP DONE ($i/36 combos, ${total} h) - tell Claude Code 'sweep done' to run the eval fan + aggregation." -ForegroundColor Green
