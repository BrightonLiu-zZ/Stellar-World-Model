# Downstream probe re-score on the exp05 arms (plan 2026-07-25).
#
# Asks whether exp05's criterion-1 gain (dynamics-weighted SSL > dynamics-off SSL, established on the
# four v1 tasks) TRANSFERS to the downstream menu: the asteroseismic block, flare, P_rot, and Ijspeert.
#
# Arms: exp05_comb_off x4 seeds (the paired lambda=0 baseline) + exp05_comb_fbwd_c1p0 x4 seeds (the
# winner, lambda=66) + the reused capacity-matched untrained arm. Encoder geometry is identical across
# all three (window 256, z128, ch 32/64/128/256, k5, GRU 256x1), which is what makes the untrained cache
# a valid capacity match rather than a silent mismatch.
#
# CPU-only, no training, no GPU required. Safe as a backgrounded Claude Code job.
# Run from the repo root:  .\experiments\run_new_task_exp05.ps1

$ErrorActionPreference = "Stop"
$env:PYTHONPATH = "src"
$env:PYTHONUNBUFFERED = "1"
$py = "C:\Users\user1\miniconda3\envs\swm\python.exe"
$out = "experiments/new_task_exp05"

$offArms  = @("comb_off_s0", "comb_off_s1", "comb_off_s2", "comb_off_s3")
$dynArms  = @("comb_fbwd_c1p0_s0", "comb_fbwd_c1p0_s1", "comb_fbwd_c1p0_s2", "comb_fbwd_c1p0_s3")
$allArms  = $offArms + $dynArms + @("untrained")

# 1. Pool mu extraction, one invocation per cell (each shares a single npz replay across its 4 seeds).
& $py -m swm.eval.new_task_extract --ckpt-dir experiments/exp05_comb_off/models `
    --arms $offArms --out-dir "$out/mu_cache" --device cpu
& $py -m swm.eval.new_task_extract --ckpt-dir experiments/exp05_comb_fbwd_c1p0/models `
    --arms $dynArms --out-dir "$out/mu_cache" --device cpu

# 2. Scorecard over the full probe menu. --dump-scores is required by the star-bootstrap CI in step 5.
& $py -m swm.eval.new_task_scorecard --arms $allArms `
    --cache-dir "$out/mu_cache" --subset-cache-dir "$out/subset_mu_cache" `
    --with-rotation --with-ijspeert `
    --out "$out/new_task_scorecard.csv" --dump-scores "$out/per_star_scores.parquet"

# 3. Flare Level B window localization (arm-agnostic, reads the same mu cache).
& $py -m swm.eval.flare_window_eval --arms $allArms `
    --cache-dir "$out/mu_cache" --out "$out/flare_window_eval.csv"

# 4. A1/A2 engineered-feature ceilings -- the "beats classic features" bar. No checkpoints needed.
& $py -m swm.eval.new_task_ceiling --with-subset --out "$out/ceiling_A1A2.csv"

# 5. Paired transfer analysis: 4-seed delta + 2*SE gate, star-bootstrap CI on the small probes.
& $py experiments/analyze_new_task_exp05.py

Write-Host "run_new_task_exp05 complete -> $out"
