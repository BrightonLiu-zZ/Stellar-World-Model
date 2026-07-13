# exp02 objective sweep - TRAINING ONLY, run in your own terminal (GPU + W&B online).
# 10 combos = 5 loss forms x 2 settings, variant B / seed 0, window 256 / seq_len 16 (reuse exp01 packed).
# Each writes checkpoints to experiments/<exp_name>/models/B_seed0/ and logs to W&B online
# (project stellar-world-model, one group per exp_name, run named <exp_name>_B_seed0).
# After all 10 finish, tell Claude Code "sweep done" -- it runs extract/probe/skyline + aggregation.
#
# Usage:  cd C:\git_repo\Stellar-World-Model ; .\experiments\run_exp02_train_sweep.ps1
# (WSL2 note: the equivalent bash is the same overrides; set PYTHONPATH=src and use the WSL python.)

$ErrorActionPreference = "Stop"
$env:PYTHONPATH = "src"
$env:PYTHONUNBUFFERED = "1"
$py = "C:\Users\user1\miniconda3\envs\swm\python.exe"

# base experiment YAML | exp_name (isolated folder) | the one swept override
$runs = @(
  @("exp02a_logpsd_amp",   "exp02a_logpsd_amp_w0p1",   "train.recon_aux.weight=0.1"),
  @("exp02a_logpsd_amp",   "exp02a_logpsd_amp_w0p3",   "train.recon_aux.weight=0.3"),
  @("exp02b_logpsd_shape", "exp02b_logpsd_shape_w0p1", "train.recon_aux.weight=0.1"),
  @("exp02b_logpsd_shape", "exp02b_logpsd_shape_w0p3", "train.recon_aux.weight=0.3"),
  @("exp02c_hf_time",      "exp02c_hf_time_w0p3",      "train.recon_aux.weight=0.3"),
  @("exp02c_hf_time",      "exp02c_hf_time_w0p6",      "train.recon_aux.weight=0.6"),
  @("exp02d_combined",     "exp02d_combined_w0p1",     "train.recon_aux.weight=0.1"),
  @("exp02d_combined",     "exp02d_combined_w0p3",     "train.recon_aux.weight=0.3"),
  @("exp02e_masked",       "exp02e_masked_f015",       "train.recon_aux.mask_frac=0.15"),
  @("exp02e_masked",       "exp02e_masked_f030",       "train.recon_aux.mask_frac=0.30")
)

foreach ($r in $runs) {
  $base = $r[0]; $combo = $r[1]; $override = $r[2]
  Write-Host "===== TRAIN $combo =====" -ForegroundColor Cyan
  & $py -u -m swm.train "+experiment=$base" "exp_name=$combo" "$override" variant=B seed=0 train.resume=false
  if ($LASTEXITCODE -ne 0) { Write-Host "FAILED: $combo (exit $LASTEXITCODE)" -ForegroundColor Red; break }
}
Write-Host "ALL TRAININGS DONE - tell Claude Code 'sweep done' to run eval + aggregation." -ForegroundColor Green
