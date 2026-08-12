# Pre-exp08 validation suite (handoff tmp/checks_before_exp08.md, grilled 2026-08-06).
#
# Runs all five checks end to end, then frees the multi-GB mu caches it built. No training: every step
# is an encode pass or an analysis over artifacts already on disk, so this is safe to run unattended.
#
#   CHK-1  the 7-probe ADR-0010 downstream menu on the frozen recipe, against two PRE-REGISTERED gates
#          Gate 0  does the frozen recipe beat untrained on the asteroseismic block at all? (never
#                  measured: every asteroseismic number in the project is an exp05-arm, ep60 number)
#          Gate 1  paired hann0p3 - comb0p3, the taper cost the handoff asked about
#          Six untrained INITS, not one, so the reference contributes its own variance to every SE.
#   CHK-2  the centre-of-window artifact the taper created, per star and per cell, plus the stitch
#          comb with an untrained control
#   CHK-3  what channel the probes read: mu against two nested engineered bases
#   CHK-4  what the dynamics arm's extra active units carry: CCA, asymmetric residual, scaling control
#   CHK-5  the selection-guard regression test (the F18 "bug" was an analysis artifact; the guard has
#          always been in the loop -- this pins it)
#
# Run from the repo root in the swm env:
#     .\experiments\run_exp08_prechecks.ps1
#     .\experiments\run_exp08_prechecks.ps1 -KeepCaches      # keep the ~12 GB of mu caches
#     .\experiments\run_exp08_prechecks.ps1 -Smoke           # ~5 min end-to-end shakedown, tiny outputs
#
# Time: roughly 4 h on the RTX 4060 (pool extraction ~35 min, the per-cell v1-subset caches ~85 min,
# scorecard ~10 min, CHK-2 ~10 min, CHK-3/4 ~90 min). Disk: ~12 GB transient, ~40 MB kept unless
# -KeepCaches. -Smoke is ~30 min: it narrows the fan to one recipe pair and two seeds, not the pool,
# because the scorecard's catalog-only populations are empty in a small TIC sample.

[CmdletBinding()]
param(
    [switch]$KeepCaches,   # skip the cleanup step and leave the pool/subset mu caches on disk
    [switch]$Smoke         # tiny limits everywhere; proves the wiring, produces no reportable numbers
)

$ErrorActionPreference = "Stop"
$env:PYTHONPATH = "src"
$env:PYTHONUNBUFFERED = "1"
$py = "C:\Users\user1\miniconda3\envs\swm\python.exe"

# -Smoke writes EVERYTHING into a throwaway namespace, so a shakedown can never leave limited-data
# numbers sitting at the paths the notebook and the writeups read.
$out = if ($Smoke) { "experiments/exp08_prechecks_smoke" } else { "experiments/exp08_prechecks" }
$muCache = "$out/mu_cache"
$subsetCache = "$out/subset_mu_cache"
$analysisPrefix = if ($Smoke) { "$out/exp07" } else { "experiments/exp07" }
New-Item -ItemType Directory -Force $out | Out-Null

# Frozen recipe first, its paired baseline second, the wide-latent arm third. lpsd0p3 is here because
# it holds 18.25 active units at selection against the winner's 5.8 (F25b), so scoring it on the
# DOWNSTREAM menu closes open question Q3 for the cost of one extra pool replay.
# -Smoke narrows the FAN (one recipe, two seeds) rather than the pool, because the scorecard's
# catalog-only populations (rgb_vs_heb, solar_like_osc) are empty in a small TIC sample and the run
# dies on an empty StandardScaler -- which proves nothing about the wiring.
$recipes = if ($Smoke) { @("hann0p3", "comb0p3") } else { @("hann0p3", "comb0p3", "lpsd0p3") }
$arms = @("fbwd", "off")
# Requested seeds. NOT every cell has all of them: exp07 ran ten cells at seeds 0-3 and extended only
# the winner and its baseline (hann0p3, comb0p3) to 6, so lpsd0p3 exists at seeds 0-3 only. Each cell's
# actual seeds are discovered from its models/ dir below rather than assumed -- assuming cost a run.
$seeds = if ($Smoke) { @(0, 1) } else { @(0, 1, 2, 3, 4, 5) }

function Get-CellSeeds($cell, $wanted) {
    $dir = "experiments/$cell/models"
    if (-not (Test-Path $dir)) { throw "no models dir for $cell" }
    $have = Get-ChildItem $dir -Directory -Filter "B_seed*" | ForEach-Object {
        [int]($_.Name -replace '^B_seed', '')
    } | Where-Object { Test-Path "$dir/B_seed$_/best_recon_aux.pt" } | Sort-Object
    $use = $wanted | Where-Object { $have -contains $_ }
    $skip = $wanted | Where-Object { $have -notcontains $_ }
    if ($skip) { Write-Host "  $cell : seeds $($skip -join ',') not trained, using $($use -join ',')" -ForegroundColor Yellow }
    if (-not $use) { throw "$cell has none of the requested seeds" }
    return $use
}
$limitFlag = if ($Smoke) { @("--limit", "4000") } else { @() }
$starFlag = if ($Smoke) { @("--limit", "150") } else { @() }

function Step($n, $text) { Write-Host "`n=== [$n] $text" -ForegroundColor Cyan }

$started = Get-Date

# --------------------------------------------------------------------------- CHK-5 (fast, runs first)
Step "CHK-5" "selection-guard regression test"
& $py -m pytest src/swm/tests/test_dual_checkpoint.py -q
if ($LASTEXITCODE -ne 0) { throw "CHK-5 selection-guard test failed" }

# --------------------------------------------------------------------------------------------- CHK-1
Step "CHK-1" "downstream menu: pool mu extraction ($($recipes.Count) cells x $($seeds.Count) seeds)"
$allArms = @()
foreach ($recipe in $recipes) {
    foreach ($arm in $arms) {
        $cell = "exp07_${recipe}_${arm}"
        $cellSeeds = Get-CellSeeds $cell $seeds
        $cellArms = $cellSeeds | ForEach-Object { "${recipe}_${arm}_s$_" }
        $allArms += $cellArms
        # one invocation per cell: the 22,860-star npz replay is shared across that cell's seeds
        & $py -m swm.eval.new_task_extract --ckpt-dir "experiments/$cell/models" `
            --arms $cellArms --out-dir $muCache @limitFlag
        if ($LASTEXITCODE -ne 0) { throw "new_task_extract failed for $cell" }

        # The v1-subset cache (rotation_period + ijspeert ride it) MUST be built per cell with that
        # cell's own checkpoints. new_task_scorecard takes a single --ckpt-dir and would otherwise
        # build every arm's subset mu from one cell's weights -- an arm label pointing at the wrong
        # network, which nothing downstream could detect. Prebuilding here means the scorecard finds
        # every cache on disk and never consults its --ckpt-dir at all.
        & $py experiments/build_subset_mu_cache.py --ckpt-dir "experiments/$cell/models" `
            --arms $cellArms --cache-dir $subsetCache
        if ($LASTEXITCODE -ne 0) { throw "build_subset_mu_cache failed for $cell" }
    }
}

# Six random inits rather than the historical single seed-0 one. A one-init reference has no measured
# spread, which makes every trained-vs-untrained delta/SE anticonservative in exactly the F17 way.
$untrainedArms = @("untrained") + ($seeds | Where-Object { $_ -ne 0 } | ForEach-Object { "untrained_s$_" })
& $py -m swm.eval.new_task_extract --ckpt-dir "experiments/exp07_hann0p3_fbwd/models" `
    --arms $untrainedArms --out-dir $muCache @limitFlag
if ($LASTEXITCODE -ne 0) { throw "new_task_extract failed for the untrained arms" }
# the untrained arms read only the GEOMETRY off that ckpt-dir, so one reference cell is correct here
& $py experiments/build_subset_mu_cache.py --ckpt-dir "experiments/exp07_hann0p3_fbwd/models" `
    --arms $untrainedArms --cache-dir $subsetCache
if ($LASTEXITCODE -ne 0) { throw "build_subset_mu_cache failed for the untrained arms" }
$allArms += $untrainedArms

Step "CHK-1" "scorecard over the full probe menu"
# Guard: every subset cache must already exist. If one is missing the scorecard would build it from
# --ckpt-dir, which is a single cell's weights -- exactly the silent wrong-network failure the
# per-cell prebuild above exists to prevent. Fail loudly instead.
$missing = $allArms | Where-Object { -not (Test-Path (Join-Path $subsetCache "$_.npz")) }
if ($missing) { throw "subset mu cache missing for: $($missing -join ', ') -- refusing to let the scorecard build them from one cell's checkpoints" }

# All 11 tasks are computed because the scorecard emits them in one pass; analyze_exp08_gates.py flags
# the four non-ADR-0010 ones reportable=False rather than dropping them (numax_hatt stays available as
# the robustness footnote). --dump-scores feeds the per-star join in CHK-2's second pass.
& $py -m swm.eval.new_task_scorecard --arms $allArms `
    --cache-dir $muCache --subset-cache-dir $subsetCache `
    --ckpt-dir "experiments/exp07_hann0p3_fbwd/models" `
    --with-rotation --with-ijspeert `
    --out "$out/new_task_scorecard.csv" --dump-scores "$out/per_star_scores.parquet"
if ($LASTEXITCODE -ne 0) { throw "new_task_scorecard failed" }

Step "CHK-1" "flare Level-B window localization (the stated null, reported never claimed)"
& $py -m swm.eval.flare_window_eval --arms $allArms --cache-dir $muCache `
    --out "$out/flare_window_eval.csv"

Step "CHK-1" "engineered-feature ceilings (the 'beats classic features' bar)"
& $py -m swm.eval.new_task_ceiling --with-subset --out "$out/ceiling_A1A2.csv"

Step "CHK-1" "pre-registered Gate 0 / Gate 1"
& $py experiments/analyze_exp08_gates.py --out-dir $out
if ($LASTEXITCODE -ne 0) { throw "analyze_exp08_gates failed" }

# --------------------------------------------------------------------------------------------- CHK-2
Step "CHK-2" "centre-of-window artifact: per-cell severity and the per-star probe-cost correlation"
& $py experiments/analyze_exp07_centre_artifact.py --seeds @seeds @starFlag `
    --out-prefix "${analysisPrefix}_centre"
if ($LASTEXITCODE -ne 0) { throw "analyze_exp07_centre_artifact failed" }

Step "CHK-2" "window-stitch harmonic comb, with the untrained control that says what kind of thing it is"
$strips = if ($Smoke) { 25 } else { 200 }
& $py experiments/analyze_exp07_stitch_spectrum.py --n-strips $strips --seeds @seeds `
    --out-prefix "${analysisPrefix}_stitch_spectrum"
if ($LASTEXITCODE -ne 0) { throw "analyze_exp07_stitch_spectrum failed" }

# ---------------------------------------------------------------------------------------- CHK-3 + 4
Step "CHK-3/4" "what mu encodes, and what the dynamics arm's extra units carry"
$starLimit = if ($Smoke) { @("--limit-stars", "400") } else { @() }
& $py experiments/analyze_exp07_mu_channel.py --seeds @seeds @starLimit `
    --out-prefix $analysisPrefix
if ($LASTEXITCODE -ne 0) { throw "analyze_exp07_mu_channel failed" }

# ------------------------------------------------------------------------------------------- cleanup
if (-not $KeepCaches) {
    Step "cleanup" "removing the pool and subset mu caches (every consumer above has already run)"
    foreach ($dir in @($muCache, $subsetCache)) {
        if (Test-Path $dir) {
            $mb = [math]::Round(((Get-ChildItem $dir -File | Measure-Object Length -Sum).Sum / 1MB), 1)
            Remove-Item $dir -Recurse -Force
            Write-Host "  removed $dir ($mb MB)"
        }
    }
    if (-not $Smoke) {
        # superseded by the 6-seed fan above; it was a single seed of a single cell
        $stale = "experiments/exp07_forensics/new_task_mu_cache"
        if (Test-Path $stale) { Remove-Item $stale -Recurse -Force; Write-Host "  removed $stale (superseded)" }
    }
    Write-Host "  kept: every CSV/parquet under $out and experiments/exp07_*.csv"
} else {
    Write-Host "`n-KeepCaches set: mu caches retained under $out"
}

$elapsed = (Get-Date) - $started
Write-Host ("`nrun_exp08_prechecks complete in {0:hh\:mm\:ss} -> {1}" -f $elapsed, $out) -ForegroundColor Green
Write-Host "Next: execute src/notebooks/exp08_design_forensics.ipynb for the written-up verdicts."
