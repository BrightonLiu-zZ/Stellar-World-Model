"""Dump per-epoch W&B run histories to one CSV per run (exp03 forensic step 1a).

Pulls every logged metric row for the finished runs of a W&B project from the cloud API and
writes them under an output directory, one CSV per run named by the run's W&B name, plus a
runs_manifest.csv mapping run id --> name / group / state / epoch count.
The forensic plots (exp03) read these CSVs instead of touching the API again.

Run (from repo root, in the swm env):
    python -m swm.eval.dump_wandb_history --out experiments/exp03_forensics/curves
    python -m swm.eval.dump_wandb_history --groups exp02 --out experiments/exp03_forensics/curves
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import pandas as pd
import wandb
from tqdm.auto import tqdm

log = logging.getLogger(__name__)

default_entity = "brighton_zz-uc-san-diego"
default_project = "stellar-world-model"


def fetch_history(run, attempts: int = 3, wait_s: float = 5.0) -> pd.DataFrame:
    """
    Pull the full logged history of one finished run as a DataFrame, retrying transient API failures.
    scan_history returns every logged row (history() subsamples past 500 rows); runs here log once
    per epoch so the frame is epochs x metrics.
    """
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            rows = []
            for row in run.scan_history():  # generator over every logged step, no subsampling
                rows.append(row)
            return pd.DataFrame(rows)
        except Exception as error:  # network boundary: W&B API / HTTP failures
            last_error = error
            log.error(f"history pull failed for {run.name} (attempt {attempt + 1}/{attempts}): {error}")
            time.sleep(wait_s)
    raise RuntimeError(f"could not pull history for {run.name}") from last_error


def main() -> None:
    parser = argparse.ArgumentParser(description="dump W&B per-epoch histories to CSVs")
    parser.add_argument("--entity", default=default_entity)
    parser.add_argument("--project", default=default_project)
    parser.add_argument("--groups", nargs="*", default=None, help="keep runs whose group starts with any of these prefixes")
    parser.add_argument("--out", required=True, help="output directory for the per-run CSVs")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    api = wandb.Api(timeout=60)
    runs = list(api.runs(f"{args.entity}/{args.project}"))
    selected = []
    for run in runs:
        if run.state != "finished":
            log.warning(f"skipping {run.name} (state {run.state})")
            continue
        if args.groups is not None:
            group = run.group or ""
            keep = False
            for prefix in args.groups:
                if group.startswith(prefix):
                    keep = True
            if not keep:
                continue
        selected.append(run)

    manifest_rows = []
    for run in tqdm(selected, desc="dump runs", total=len(selected)):
        history = fetch_history(run)
        csv_path = out_dir / f"{run.name}.csv"
        history.to_csv(csv_path, index=False)
        manifest_rows.append({
            "run_id": run.id,
            "name": run.name,
            "group": run.group,
            "state": run.state,
            "n_rows": len(history),
            "csv": csv_path.name,
        })
        log.info(f"{run.name}: {len(history)} rows --> {csv_path}")

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(out_dir / "runs_manifest.csv", index=False)
    log.info(f"wrote manifest with {len(manifest)} runs to {out_dir / 'runs_manifest.csv'}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
