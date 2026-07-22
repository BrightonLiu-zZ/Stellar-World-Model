# exp04 morning-after eval scan (grill 2026-07-20): readout_sweep quick scan over every finished
# exp04 run + the exp03 confirm-block seeds. Writes the first-segment window-mu caches the transit
# probe consumes, and appends gap rows to each experiment's results/readout_sweep.csv.
#
# Grouping is load-bearing: readout_sweep builds ONE untrained reference per invocation from the
# FIRST matched experiment's stored config, cached per --untrained-cache dir. Standard-shape runs
# (corner cells, winner_ep100, exp03 confirm) share the exp03 z128 cache; every encoder variant gets
# its own cache dir so the trained-vs-untrained gap stays capacity-matched.
#
# Ckpt = best_recon_aux only (grill Q6). Readouts logistic+gbm, pooling mean (quick-scan tier).
# Every step streams to the console (no redirects) so the Claude Code monitor shows live progress.
# Safe to re-run: mu caches make finished invocations near-instant.

$ErrorActionPreference = 'Continue'
Set-Location (Split-Path $PSScriptRoot -Parent)
$py = 'C:\Users\user1\miniconda3\envs\swm\python.exe'
$env:PYTHONPATH = 'src'
$env:PYTHONUNBUFFERED = '1'

$std = 'experiments/exp03_eval_cache'
$jobs = @(
    # standard z128 shape: 8 corner cells seed 0, bonus corner seed 1s, the 100-epoch probe
    @{ glob = 'exp04_fb0p0*';           seed = 0; cache = $std },
    @{ glob = 'exp04_fb0p0*';           seed = 1; cache = $std },   # only the 4 bonus dirs have seed 1; rest warn+skip
    @{ glob = 'exp04_winner_ep100';     seed = 0; cache = $std },
    # exp03 confirm block: 2 extra seeds per kept cell (seed 0 already scored in exp03)
    @{ glob = 'exp03_fb0p02_b0p1_lpsd'; seed = 1; cache = $std },
    @{ glob = 'exp03_fb0p02_b0p1_lpsd'; seed = 2; cache = $std },
    @{ glob = 'exp03_fb0_b0p1_comb';    seed = 1; cache = $std },
    @{ glob = 'exp03_fb0_b0p1_comb';    seed = 2; cache = $std },
    @{ glob = 'exp03_fb0p05_b0p3_lpsd'; seed = 1; cache = $std },
    @{ glob = 'exp03_fb0p05_b0p3_lpsd'; seed = 2; cache = $std }
)
# encoder variants: per-variant untrained cache (capacity-matched gap), seeds 0+1, z64/z32 also seed 2
$variants = 'enc_z64', 'enc_z32', 'enc_k9', 'enc_k15', 'enc_d5', 'enc_d3', 'enc_w2x', 'enc_whalf', 'enc_z64k9'
foreach ($v in $variants) {
    $seeds = if ($v -in 'enc_z64', 'enc_z32') { 0, 1, 2 } else { 0, 1 }
    foreach ($s in $seeds) {
        $jobs += @{ glob = "exp04_$v"; seed = $s; cache = "experiments/exp04_eval_cache/$v" }
    }
}

$failed = @()
$t0 = Get-Date
for ($i = 0; $i -lt $jobs.Count; $i++) {
    $j = $jobs[$i]
    $mins = [math]::Round(((Get-Date) - $t0).TotalMinutes, 1)
    Write-Host ("=== [{0}/{1}] {2} seed {3} (cache {4}) - elapsed {5} min ===" -f `
        ($i + 1), $jobs.Count, $j.glob, $j.seed, $j.cache, $mins) -ForegroundColor Cyan
    & $py -m swm.eval.readout_sweep --exp-glob $j.glob --seed $j.seed --untrained-cache $j.cache `
        --ckpts best_recon_aux --readouts logistic gbm --poolings mean
    if ($LASTEXITCODE -ne 0) {
        Write-Host ("FAILED: {0} seed {1} (exit {2})" -f $j.glob, $j.seed, $LASTEXITCODE) -ForegroundColor Red
        $failed += ("{0} seed {1}" -f $j.glob, $j.seed)
    }
}

$mins = [math]::Round(((Get-Date) - $t0).TotalMinutes, 1)
Write-Host ("=== scan done in {0} min; {1} failed ===" -f $mins, $failed.Count) -ForegroundColor Cyan
$failed | ForEach-Object { Write-Host "  $_" -ForegroundColor Red }
exit $failed.Count
