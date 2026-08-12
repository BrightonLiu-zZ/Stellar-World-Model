# exp07 MIL pooling sweep - one invocation per cell so a kill loses at most one cell's work.
# swm.eval.mil_sweep writes only at the end and does read-concat-write on --out, so each cell gets its
# own part file; the notebook (section K3) merges them and drops the untrained arm's duplicate rows.
# Prereq: bag caches from  python -m swm.eval.mil_cache --cells <same list> --seeds 0 1 2 3 --scope first
# Usage:  cd C:\git_repo\Stellar-World-Model ; .\experiments\run_exp07_mil_sweep.ps1

$ErrorActionPreference = "Stop"
$env:PYTHONPATH = "src"
$env:PYTHONUNBUFFERED = "1"
$py = "C:\Users\user1\miniconda3\envs\swm\python.exe"
$cells = @("exp07_hann0p3_fbwd", "exp07_hann0p3_off", "exp07_comb0p3_fbwd", "exp07_comb0p3_off")

foreach ($cell in $cells) {
  $out = "experiments/mil_pooling/mil_sweep_exp07_$cell.csv"
  if (Test-Path $out) {
    Write-Host "SKIP $cell (part file exists)"
    continue
  }
  Write-Host "=== $cell -> $out"
  & $py -m swm.eval.mil_sweep --cells $cell --seeds 0 1 2 3 --scope first --out $out
  if ($LASTEXITCODE -ne 0) { throw "mil_sweep failed for $cell" }
}
Write-Host "all cells done"
