"""Generate an experiment's per-cell Hydra YAMLs + sweep runner from its single manifest.

One experiment = ONE hand-written manifest (experiments/configs/<expNN>_<slug>.yaml); everything the
sweep needs is generated from it so nothing drifts (the exp05 layout spread the real config over two
base YAMLs and a lambda table inside the .ps1 - see /stellar-ablation-experiment rule 1).

    python -m swm.exp.gen_sweep experiments/configs/exp06_geometry_coverage.yaml
    python -m swm.exp.gen_sweep experiments/configs/exp06_geometry_coverage.yaml --dry-run
    python -m swm.exp.gen_sweep experiments/configs/exp06_geometry_coverage.yaml --allow-tbd

Outputs (each carries a "GENERATED ... DO NOT EDIT" banner):
    src/swm/configs/experiment/<expNN>/<cell>.yaml   one complete, self-contained config per cell
    experiments/run_<name>.ps1                       seed-major resumable queue (ASCII-only for PS 5.1)

Cells whose overrides still hold the sentinel TBD_PILOT (constants awaiting a calibration pilot) abort
generation unless --allow-tbd, which generates everything else and lists what was excluded - refill the
manifest and regenerate to pick them up (DONE markers make the rerun a no-op for finished runs).
"""
from __future__ import annotations

import argparse
import copy
import logging
import sys
from pathlib import Path
from typing import Any

import yaml

log = logging.getLogger(__name__)

TBD = "TBD_PILOT"
BANNER = "# GENERATED FROM {manifest} - DO NOT EDIT (edit the manifest and re-run swm.exp.gen_sweep)"


def deep_merge(base: dict, delta: dict) -> dict:
    """Nested-dict merge, delta wins; lists/scalars replace wholesale."""
    out = copy.deepcopy(base)
    for key, val in delta.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


def apply_dotted(cfg: dict, overrides: dict[str, Any]) -> dict:
    """Apply {'train.lambda_dyn': 66} style overrides onto a nested dict, creating no new branches."""
    out = copy.deepcopy(cfg)
    for dotted, val in overrides.items():
        node = out
        *parents, leaf = dotted.split(".")
        for part in parents:
            assert part in node, f"override {dotted!r} targets unknown branch {part!r}"
            node = node[part]
        assert leaf in node, f"override {dotted!r} targets unknown key {leaf!r}"
        node[leaf] = val
    return out


def expand_cells(manifest: dict) -> list[dict]:
    """Resolve every sweep cell to its full config + bookkeeping fields."""
    base = manifest["base"]
    recipes = manifest["recipes"]
    cells = []
    for cell in manifest["sweep"]["cells"]:
        recipe = cell.get("recipe", next(iter(recipes)))
        assert recipe in recipes, f"cell {cell['exp_name']}: unknown recipe {recipe!r}"
        cfg = deep_merge(base, recipes[recipe] or {})
        cfg = apply_dotted(cfg, cell.get("overrides", {}))
        window, seq_len = int(cfg["data"]["window"]), int(cfg["data"]["seq_len"])
        assert window * seq_len == 4096, (
            f"cell {cell['exp_name']}: window*seq_len = {window * seq_len}, breaks the fixed-horizon invariant"
        )
        tbd = [k for k, v in cell.get("overrides", {}).items() if v == TBD]
        cells.append(
            {
                "exp_name": cell["exp_name"],
                "recipe": recipe,
                "geometry": cell["geometry"],
                "seeds": cell.get("seeds", manifest["sweep"]["seeds"]),
                "cfg": cfg,
                "tbd": tbd,
                "note": cell.get("note") or cell.get("prediction") or "",
            }
        )
    return cells


def render_cell_yaml(cell: dict, manifest: dict, manifest_path: Path) -> str:
    """One complete Hydra experiment-group file: every knob in one place, checkable without the runner."""
    packed_source = manifest["paths"]["packed_sources"][cell["geometry"]]
    header = [
        "# @package _global_", # MUST be line 1 - Hydra only honors the directive at the top of the file
        BANNER.format(manifest=manifest_path.as_posix()),
        f"# {manifest['name']} cell {cell['exp_name']} (recipe {cell['recipe']}, geometry {cell['geometry']})",
        f"# packed: junction experiments/{cell['exp_name']}/packed -> {packed_source} (the runner creates it).",
        "# Running STAND-ALONE (e.g. a pilot under a throwaway exp_name) must add:",
        f"#   paths.packed_dir={manifest['paths'].get('repo_root', 'C:/git_repo/Stellar-World-Model')}/{packed_source}",
    ]
    if cell["note"]:
        header.append(f"# {cell['note']}")
    body = {"exp_name": cell["exp_name"], **cell["cfg"]}
    return "\n".join(header) + "\n\n" + yaml.safe_dump(body, sort_keys=False, default_flow_style=False, width=110)


def render_runner(manifest: dict, cells: list[dict], manifest_path: Path) -> str:
    """Seed-major resumable PS 5.1 queue, mirroring run_exp05_train_sweep.ps1 (DONE markers, retry,
    -DryRun, -MaxHours) - but with zero config content: each run is just +experiment=<expNN>/<cell>."""
    name = manifest["name"]
    sweep = manifest["sweep"]
    exp_group = name.split("_")[0] # exp06
    seed_universe = sorted({s for c in cells for s in c["seeds"]})
    geometries = sorted({c["geometry"] for c in cells})
    src_lines = [
        f"  '{g}' = \"$repo\\{manifest['paths']['packed_sources'][g]}\"".replace("/", "\\") for g in geometries
    ]
    cell_lines = [
        f"  @('{c['exp_name']}', '{c['geometry']}', @({', '.join(str(s) for s in c['seeds'])}))" for c in cells
    ]
    per_run = sweep.get("per_run_minutes", {})
    lines = f"""{BANNER.format(manifest=manifest_path.as_posix())}
# {name} - TRAINING ONLY, run in your own terminal (GPU + W&B online). Plan: {manifest['plan']}
# {len(cells)} cells, seed-major order so a -MaxHours cutoff leaves every cell with even seed coverage.
# Estimated per-run minutes: {per_run} (VERIFY against the first runs; exp05-derived).
# INTERRUPT/RESUME: Ctrl-C anytime; train.resume=true + last.pt resume mid-run; DONE.txt skips finished runs.
# Usage:  cd C:\\git_repo\\Stellar-World-Model ; .\\experiments\\run_{name}.ps1
#         .\\experiments\\run_{name}.ps1 -DryRun
#         .\\experiments\\run_{name}.ps1 -MaxHours {float(sweep['budget_hours'])}

param(
  [switch]$DryRun,
  [double]$MaxHours = {float(sweep['budget_hours'])}
)

$ErrorActionPreference = "Stop"
$env:PYTHONPATH = "src"
$env:PYTHONUNBUFFERED = "1"
$py = "{manifest['env']['python'].replace('/', chr(92))}"
$repo = "C:\\git_repo\\Stellar-World-Model"

# packed source per geometry (junctioned per cell below); packs must exist BEFORE training starts
$packedSource = @{{
{chr(10).join(src_lines)}
}}
foreach ($g in $packedSource.Keys) {{
  if (-not (Test-Path (Join-Path $packedSource[$g] "train_index.parquet"))) {{
    throw "packed source for $g missing: $($packedSource[$g]) - run swm.data.pack first"
  }}
}}

# cells: (exp_name, geometry, seed list). ALL config lives in src/swm/configs/experiment/{exp_group}/<cell>.yaml.
$cells = @(
{(',' + chr(10)).join(cell_lines)}
)

# seed-major expansion
$runs = @()
foreach ($seed in @({', '.join(str(s) for s in seed_universe)})) {{
  foreach ($c in $cells) {{
    if ($c[2] -contains $seed) {{ $runs += , @($c[0], $c[1], $seed) }}
  }}
}}

$sweepStart = Get-Date
$failed = @()
$i = 0
foreach ($r in $runs) {{
  $i += 1
  $exp = $r[0]; $geom = $r[1]; $seed = $r[2]
  $comboDir = "$repo\\experiments\\$exp"
  $doneMarker = "$comboDir\\models\\B_seed$seed\\DONE.txt"
  $elapsedH = ((Get-Date) - $sweepStart).TotalHours

  if (Test-Path $doneMarker) {{
    Write-Host "[$i/$($runs.Count)] SKIP $exp seed $seed (DONE marker present)" -ForegroundColor DarkGray
    continue
  }}
  if ($elapsedH -gt $MaxHours) {{
    Write-Host "[$i/$($runs.Count)] CUTOFF: $([math]::Round($elapsedH,2)) h > MaxHours $MaxHours - stopping" -ForegroundColor Yellow
    break
  }}

  $cmd = @("+experiment={exp_group}/$exp", "variant={manifest['eval']['variant']}", "seed=$seed")
  if ($DryRun) {{
    Write-Host "[$i/$($runs.Count)] [$geom] python -m swm.train $($cmd -join ' ')"
    continue
  }}

  if (-not (Test-Path $comboDir)) {{ New-Item -ItemType Directory -Path $comboDir | Out-Null }}
  $packedLink = Join-Path $comboDir "packed"
  if (-not (Test-Path $packedLink)) {{
    New-Item -ItemType Junction -Path $packedLink -Target $packedSource[$geom] | Out-Null
  }}

  $elapsedMin = [math]::Round(((Get-Date) - $sweepStart).TotalMinutes, 1)
  Write-Host "===== [$i/$($runs.Count)] [$geom] TRAIN $exp seed $seed - ${{elapsedMin}} min elapsed =====" -ForegroundColor Cyan
  & $py -u -m swm.train @cmd
  if ($LASTEXITCODE -ne 0) {{
    Write-Host "RETRY: $exp seed $seed (exit $LASTEXITCODE)" -ForegroundColor Yellow
    & $py -u -m swm.train @cmd
  }}
  if ($LASTEXITCODE -ne 0) {{
    Write-Host "FAILED: $exp seed $seed (exit $LASTEXITCODE) - continuing" -ForegroundColor Red
    $failed += "$exp seed $seed"
    continue
  }}
  Set-Content -Path $doneMarker -Value ("finished " + (Get-Date -Format o)) -Encoding utf8
}}

$total = [math]::Round(((Get-Date) - $sweepStart).TotalHours, 2)
if ($failed.Count -gt 0) {{
  Write-Host "SWEEP DONE with $($failed.Count) FAILURES (${{total}} h): $($failed -join '; ')" -ForegroundColor Red
}} else {{
  Write-Host "SWEEP DONE (${{total}} h) - tell Claude Code '{name} done' to run the eval fan." -ForegroundColor Green
}}
"""
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="print the run manifest, write nothing")
    parser.add_argument("--allow-tbd", action="store_true",
                        help="generate around cells with TBD_PILOT constants instead of aborting")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    try:
        manifest = yaml.safe_load(args.manifest.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as e:
        log.error(f"cannot read manifest {args.manifest}: {e}")
        return 1

    repo_root = args.manifest.resolve().parents[2] # experiments/configs/<file> -> repo root
    cells = expand_cells(manifest)

    blocked = [c for c in cells if c["tbd"]]
    if blocked and not args.allow_tbd:
        for c in blocked:
            log.error(f"BLOCKED: {c['exp_name']} awaits pilot constants: {c['tbd']}")
        log.error("fill constants in the manifest and re-run, or pass --allow-tbd to generate around them")
        return 2
    ready = [c for c in cells if not c["tbd"]]

    per_run = manifest["sweep"].get("per_run_minutes", {})
    total_min = 0.0
    log.info(f"{manifest['name']}: {len(ready)} cells ready, {len(blocked)} blocked on pilots")
    for c in ready:
        est = per_run.get(f"ep{c['cfg']['train']['max_epochs']}", 0)
        total_min += est * len(c["seeds"])
        log.info(f"  {c['exp_name']:28s} {c['geometry']:6s} recipe={c['recipe']:12s} "
                 f"dyn={c['cfg']['model']['dyn_mode']:8s} lambda={c['cfg']['train']['lambda_dyn']!s:>4s} "
                 f"ep{c['cfg']['train']['max_epochs']} seeds={c['seeds']}")
    n_runs = sum(len(c["seeds"]) for c in ready)
    log.info(f"total: {n_runs} runs, est {total_min / 60:.1f} h (budget {manifest['sweep']['budget_hours']} h)")
    if args.dry_run:
        return 0

    exp_group = manifest["name"].split("_")[0]
    cfg_dir = repo_root / "src" / "swm" / "configs" / "experiment" / exp_group
    cfg_dir.mkdir(parents=True, exist_ok=True)
    for c in ready:
        out = cfg_dir / f"{c['exp_name']}.yaml"
        out.write_text(render_cell_yaml(c, manifest, args.manifest), encoding="utf-8", newline="\n") # BOM-free
        log.info(f"wrote {out.relative_to(repo_root)}")

    runner_text = render_runner(manifest, ready, args.manifest)
    non_ascii = {ch for ch in runner_text if ord(ch) > 127}
    assert not non_ascii, f"runner must be pure ASCII for PS 5.1, found: {non_ascii!r}" # ps51-encoding rule
    runner = repo_root / "experiments" / f"run_{manifest['name']}.ps1"
    runner.write_text(runner_text, encoding="ascii", newline="\r\n")
    log.info(f"wrote {runner.relative_to(repo_root)}")
    if blocked:
        log.warning(f"excluded (regenerate after pilots): {[c['exp_name'] for c in blocked]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
