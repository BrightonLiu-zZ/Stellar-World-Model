"""Generate the C1/C2 supervised-baseline queue from its single manifest.

`swm.exp.gen_sweep` could not be reused: it asserts window*seq_len == 4096 and emits per-cell Hydra
YAMLs for `swm.train`'s unlabelled sequence dataset, neither of which describes a supervised run over
two disjoint labelled populations. The one-manifest rule is kept regardless (/stellar-ablation-
experiment rule 1) -- and kept more strictly than usual, because `swm.train.supervised` reads the
manifest at run time, so there are no generated config copies that could drift from it. The only
generated artifact is the runner.

    python -m swm.exp.gen_supervised experiments/configs/c1c2_supervised_baselines.yaml
    python -m swm.exp.gen_supervised experiments/configs/c1c2_supervised_baselines.yaml --dry-run

Output (carries a "GENERATED ... DO NOT EDIT" banner):
    experiments/run_c1c2_supervised_baselines.ps1   arm-major, task-priority, DONE-marker resumable
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

log = logging.getLogger(__name__)

BANNER = "# GENERATED FROM {manifest} - DO NOT EDIT (edit the manifest and re-run swm.exp.gen_supervised)"


def expand_runs(manifest: dict) -> list[tuple[str, str, int]]:
    """The full queue as (arm, task, seed), arm-major then task-priority then seed.

    Arm-major puts all of C1 ahead of C2, and the manifest's task order is the priority order, so a
    -MaxHours cutoff leaves the LEADING cells complete rather than leaving every cell half-seeded.
    """
    order = manifest["queue"]["order"]
    task_names = []
    for task in manifest["tasks"]:
        task_names.append(task["name"])
    missing = set(task_names) ^ set(order)
    assert not missing, f"queue.order and tasks disagree on: {sorted(missing)}"
    runs = []
    for arm_name in manifest["queue"]["arm_order"]:
        arm = manifest["arms"][arm_name]
        for task_name in order:
            for seed in arm["seeds"]:
                runs.append((arm_name, task_name, int(seed)))
    return runs


def pilot_runs(manifest: dict) -> list[tuple[str, str, int]]:
    """The seed-0 pilot wave: one arm, one seed, every task -- the pre-registered gate's input."""
    pilot = manifest["queue"]["pilot"]
    runs = []
    for task_name in manifest["queue"]["order"]:
        runs.append((pilot["arm"], task_name, int(pilot["seed"])))
    return runs


def render(manifest: dict, manifest_path: Path) -> str:
    """Emit the resumable PowerShell queue. ASCII only, so PS 5.1 cannot mis-parse it."""
    runs = expand_runs(manifest)
    pilot = pilot_runs(manifest)
    root = manifest["paths"]["root"]
    budget = manifest["budget"]
    lines = [
        BANNER.format(manifest=manifest_path.as_posix()),
        f"# {manifest['name']} - TRAINING, run in your own terminal (GPU + W&B online).",
        f"# Plan: {manifest['plan']} rows C1/C2. Boundary: {manifest['adr']} (external baselines, never the probe).",
        f"# {len(runs)} runs = {len(manifest['queue']['arm_order'])} arms x {len(manifest['tasks'])} tasks"
        f" x {len(manifest['arms'][manifest['queue']['arm_order'][0]]['seeds'])} seeds."
        f" Estimated {budget['full_hours']} h total; pilot {budget['pilot_minutes']} min.",
        "# Arm-major, task-priority order so a -MaxHours cutoff leaves the LEADING cells complete.",
        "# INTERRUPT/RESUME: Ctrl-C anytime; DONE.txt markers skip finished runs on the next invocation.",
        "# Usage:  cd C:\\git_repo\\Stellar-World-Model",
        "#         .\\experiments\\run_c1c2_supervised_baselines.ps1 -PilotOnly   # 11 runs, then STOP for the gate",
        "#         .\\experiments\\run_c1c2_supervised_baselines.ps1              # the full queue",
        "#         .\\experiments\\run_c1c2_supervised_baselines.ps1 -DryRun",
        "",
        "param(",
        "  [switch]$DryRun,",
        "  [switch]$PilotOnly,",
        f"  [double]$MaxHours = {float(budget['full_hours']) * 3}",
        ")",
        "",
        '$ErrorActionPreference = "Stop"',
        '$env:PYTHONPATH = "src"',
        '$env:PYTHONUNBUFFERED = "1"',
        '$py = "C:\\Users\\user1\\miniconda3\\envs\\swm\\python.exe"',
        '$repo = "C:\\git_repo\\Stellar-World-Model"',
        f'$manifest = "$repo\\{manifest_path.as_posix().replace("/", chr(92))}"',
        f'$root = "$repo\\{root.replace("/", chr(92))}"',
        "",
        "# packed source for the v1 population; the pool replays from processed/sequences at first use",
        '$packed = "$repo\\experiments\\exp01_window256_seq16\\packed"',
        'if (-not (Test-Path (Join-Path $packed "train_index.parquet"))) {',
        '  throw "v1 packed source missing: $packed - run swm.data.pack first"',
        "}",
        "",
        "# (arm, task, seed). ALL config lives in the manifest; this file only orders the queue.",
        "$pilot = @(",
    ]
    for arm, task, seed in pilot:
        lines.append(f"  @('{arm}', '{task}', {seed}),")
    lines[-1] = lines[-1].rstrip(",")
    lines += [")", "", "$full = @("]
    for arm, task, seed in runs:
        lines.append(f"  @('{arm}', '{task}', {seed}),")
    lines[-1] = lines[-1].rstrip(",")
    lines += [
        ")",
        "",
        "$runs = if ($PilotOnly) { $pilot } else { $full }",
        "$sweepStart = Get-Date",
        "$failed = @()",
        "$i = 0",
        "foreach ($r in $runs) {",
        "  $i += 1",
        "  $arm = $r[0]; $task = $r[1]; $seed = $r[2]",
        '  $doneMarker = "$root\\runs\\$arm\\$task\\seed$seed\\DONE.txt"',
        "  $elapsedH = ((Get-Date) - $sweepStart).TotalHours",
        "",
        "  if (Test-Path $doneMarker) {",
        '    Write-Host "[$i/$($runs.Count)] SKIP $arm/$task seed $seed (DONE marker present)" -ForegroundColor DarkGray',
        "    continue",
        "  }",
        "  if ($elapsedH -gt $MaxHours) {",
        '    Write-Host "[$i/$($runs.Count)] CUTOFF: $([math]::Round($elapsedH,2)) h > MaxHours $MaxHours - stopping" -ForegroundColor Yellow',
        "    break",
        "  }",
        "",
        '  $cmd = @("--manifest", $manifest, "--arm", $arm, "--task", $task, "--seed", "$seed")',
        "  if ($DryRun) {",
        '    Write-Host "[$i/$($runs.Count)] python -m swm.train.supervised $($cmd -join \' \')"',
        "    continue",
        "  }",
        "",
        "  $elapsedMin = [math]::Round(((Get-Date) - $sweepStart).TotalMinutes, 1)",
        '  Write-Host "===== [$i/$($runs.Count)] TRAIN $arm/$task seed $seed - ${elapsedMin} min elapsed =====" -ForegroundColor Cyan',
        "  & $py -u -m swm.train.supervised @cmd",
        "  if ($LASTEXITCODE -ne 0) {",
        '    Write-Host "RETRY: $arm/$task seed $seed (exit $LASTEXITCODE)" -ForegroundColor Yellow',
        "    & $py -u -m swm.train.supervised @cmd",
        "  }",
        "  if ($LASTEXITCODE -ne 0) {",
        '    Write-Host "FAILED: $arm/$task seed $seed (exit $LASTEXITCODE) - continuing" -ForegroundColor Red',
        '    $failed += "$arm/$task seed $seed"',
        "  }",
        "}",
        "",
        "$total = [math]::Round(((Get-Date) - $sweepStart).TotalHours, 2)",
        "if ($failed.Count -gt 0) {",
        '  Write-Host "QUEUE DONE with $($failed.Count) FAILURES (${total} h): $($failed -join \'; \')" -ForegroundColor Red',
        "} elseif ($PilotOnly) {",
        '  Write-Host "PILOT DONE (${total} h) - tell Claude Code \'c1c2 pilot done\' to score the pre-registered gate." -ForegroundColor Green',
        "} else {",
        '  Write-Host "QUEUE DONE (${total} h) - tell Claude Code \'c1c2 queue done\' to score the scorecard rows." -ForegroundColor Green',
        "}",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Generate the C1/C2 runner from its manifest.")
    ap.add_argument("manifest")
    ap.add_argument("--dry-run", action="store_true", help="print the runner instead of writing it")
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    try:
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = yaml.safe_load(handle)
    except OSError as err:
        log.error(f"cannot read manifest {manifest_path}: {err}")
        raise
    text = render(manifest, manifest_path)
    if args.dry_run:
        sys.stdout.write(text)
        return 0
    out_path = Path(manifest["paths"]["runner"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="ascii")
    log.info(f"wrote {out_path} ({len(expand_runs(manifest))} runs, "
             f"{len(pilot_runs(manifest))} in the pilot wave)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
