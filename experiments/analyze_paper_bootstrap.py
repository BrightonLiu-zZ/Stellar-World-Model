"""Paired bootstrap over TEST STARS for every contrast the paper prints.

Why (Yue Ma review 2026-09-10, item 2): the paper's 2*SE bands come from re-training (seed spread). They
say nothing about whether the SAME test stars happened to favour one arm. This script resamples the
test stars with replacement, re-scores every arm on the identical resampled list from its saved
per-star predictions, and looks at the distribution of the difference. No model is retrained.

Seeds are handled by averaging: on each resample, an arm's metric is the mean over its seeds (six for
the mu arms and for C1/C2, one for `features` and `untrained`). The resample therefore varies the
stars only, which is the question this check asks; seed variation is already the published band.

A resample whose positives (or negatives) vanish carries no ranking information and is dropped; the
count of dropped resamples is reported per task. That can only happen on the 161-star rgb_vs_heb
and the 150-star rotation_period cells, which is exactly where Yue Ma asked to look.

Contrasts, in the paper's own names (Figure 1a/b and the abstract counts):
    fusion - features          "improves a frozen linear readout ... on 7 of 10"
    untrained_fusion - features  the control: "while an untrained encoder does not"
    fusion - dynoff_fusion     "the latent-dynamics term is what buys the complementarity"
    fusion - C1                "ahead of the supervised Conv1D on 6 of 10 ... indistinguishable on 4"
    fusion - C2                "ahead of the raw-flux MLP on 9 of 10"
    mu - features              Table 1, the mu-only column

Run (swm env, repo root, PYTHONPATH=src, CPU, ~5 min):
    python experiments/analyze_paper_bootstrap.py
    python experiments/analyze_paper_bootstrap.py --n-boot 50 --tasks eb     # smoke
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, r2_score, roc_auc_score
from tqdm.auto import tqdm

repo_root = Path(__file__).resolve().parents[1]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S", handlers=[logging.StreamHandler(sys.stdout)], force=True)
log = logging.getLogger("paper_bootstrap")

PAPER_TASKS = ["pulsating", "eb", "rotation", "transit", "osc_giant", "solar_like_osc", "rgb_vs_heb",
               "numax_hon", "rotation_period", "flare"]
# paper arm name -> (family, arm_set) in the two dumps
ARMS = {"features": ("features", "features_only"),
        "mu": ("hann0p3_fbwd", "mu"),
        "fusion": ("hann0p3_fbwd", "features_plus_mu"),
        "untrained_fusion": ("untrained", "features_plus_mu"),
        "dynoff_fusion": ("hann0p3_off", "features_plus_mu"),
        "c1": ("conv_supervised", "conv_supervised"),
        "c2": ("mlp_raw", "mlp_raw")}
CONTRASTS = [("fusion", "features"), ("untrained_fusion", "features"), ("fusion", "dynoff_fusion"),
             ("fusion", "c1"), ("fusion", "c2"), ("mu", "features")]
# the paper's own delta and 2*SE columns in table1_data.csv, for the side-by-side verdict
PAPER_COLUMNS = {("fusion", "features"): ("d_features", "d_features_2se"),
                 ("untrained_fusion", "features"): ("d_features_untrained", None),
                 ("fusion", "c1"): ("d_c1", "d_c1_2se"),
                 ("fusion", "c2"): ("d_c2", "d_c2_2se")}
# fusion - dynoff is the paper's dynamics ablation, paired by seed, in dynamics_ablation.csv (drop, drop_2se)
CONTRAST_LABEL = {"fusion-features": "fusion $-$ features", "untrained_fusion-features": "untrained fusion $-$ features",
                  "fusion-dynoff_fusion": "fusion $-$ dynamics-off fusion", "fusion-c1": "fusion $-$ C1",
                  "fusion-c2": "fusion $-$ C2", "mu-features": r"$\mu$ $-$ features"}
TASK_LABEL = {"pulsating": "pulsating", "eb": "eclipsing binary", "rotation": "rotation", "transit": "transit",
              "osc_giant": "oscillating giants", "solar_like_osc": "solar-like osc.", "rgb_vs_heb": r"RGB vs.\ HeB",
              "numax_hon": r"$\nu_{\max}$", "rotation_period": "rotation period", "flare": "flare"}


def metric_value(shape: str, y: np.ndarray, score: np.ndarray) -> float:
    """One arm's headline metric on one star list; NaN when the list carries a single class."""
    if shape == "regression":
        return float(r2_score(y, score))
    if y.min() == y.max():
        return np.nan
    if shape == "contrastive":
        return float(roc_auc_score(y, score))
    return float(average_precision_score(y, score))


def arm_matrix(dumps: pd.DataFrame, family: str, arm_set: str, task: str,
               tics: np.ndarray) -> np.ndarray:
    """(n_seeds, n_star) score matrix for one arm on one task, aligned to `tics`; fails if a star is missing."""
    rows = dumps[(dumps["family"] == family) & (dumps["arm_set"] == arm_set) & (dumps["task"] == task)]
    assert not rows.empty, f"no dumped predictions for {family}/{arm_set}/{task}"
    seeds = sorted(rows["seed"].unique())
    matrix = np.empty((len(seeds), len(tics)), dtype=float)
    for i, seed in enumerate(seeds):
        one = rows[rows["seed"] == seed].set_index("tic_id")["score"]
        assert set(one.index) == set(tics), (
            f"{family}/{arm_set}/{task} seed {seed}: test stars differ from the reference list "
            f"({len(one)} vs {len(tics)})")
        matrix[i] = one.reindex(tics).to_numpy()
    return matrix


def write_tex(out: pd.DataFrame, path: Path) -> None:
    """Appendix table: one row per task, one column group per printed contrast (point, 95% CI, share > 0)."""
    contrasts = ["fusion-features", "fusion-c1", "fusion-c2", "fusion-dynoff_fusion"]
    eol = r" \\"
    lines = ["% GENERATED by experiments/analyze_paper_bootstrap.py -- do not hand-edit",
             r"\begin{tabular}{l" + "rr" * len(contrasts) + "}", r"\toprule",
             "task & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{CONTRAST_LABEL[c]}}}" for c in contrasts) + eol,
             " & " + " & ".join(r"95\% CI & $f_{>0}$" for _ in contrasts) + eol, r"\midrule"]
    for task in PAPER_TASKS:
        cells = []
        for c in contrasts:
            r = out[(out["task"] == task) & (out["contrast"] == c)].iloc[0]
            body = f"[{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]"
            # bold = the interval excludes zero (\textbf does not reach inside math mode; \mathbf does)
            ci = rf"$\mathbf{{{body}}}$" if (r["ci_lo"] > 0 or r["ci_hi"] < 0) else f"${body}$"
            cells += [ci, f"{r['frac_positive']:.2f}"]
        lines.append(f"{TASK_LABEL[task]} & " + " & ".join(cells) + eol)
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info(f"wrote {path}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Paired star-bootstrap of every contrast the paper prints.")
    ap.add_argument("--in-dir", default="experiments/paper_bootstrap")
    ap.add_argument("--table", default="paper/build/table1_data.csv")
    ap.add_argument("--ablation", default="paper/build/dynamics_ablation.csv")
    ap.add_argument("--tex-out", default="paper/tables/tableC_bootstrap.tex")
    ap.add_argument("--tasks", nargs="+", default=PAPER_TASKS)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    in_dir = repo_root / args.in_dir
    dumps = pd.concat([pd.read_parquet(in_dir / "linear_star_scores.parquet"),
                       pd.read_parquet(in_dir / "c1c2_star_scores.parquet")], ignore_index=True)
    table = pd.read_csv(repo_root / args.table).set_index("task")
    ablation = pd.read_csv(repo_root / args.ablation).set_index("task")
    rng = np.random.default_rng(args.seed)

    rows = []
    for task in tqdm(args.tasks, desc="tasks", total=len(args.tasks)):
        # The reference star list and labels come from the fusion arm; every other arm must match it.
        ref = dumps[(dumps["family"] == "hann0p3_fbwd") & (dumps["arm_set"] == "features_plus_mu")
                    & (dumps["task"] == task)]
        shape = ref["shape"].iloc[0]
        ref0 = ref[ref["seed"] == ref["seed"].min()].sort_values("tic_id")
        tics = ref0["tic_id"].to_numpy()
        y = ref0["y"].to_numpy()
        for name, (family, arm_set) in ARMS.items():
            other = dumps[(dumps["family"] == family) & (dumps["arm_set"] == arm_set) & (dumps["task"] == task)]
            other0 = other[other["seed"] == other["seed"].min()].set_index("tic_id")["y"].reindex(tics)
            assert np.allclose(other0.to_numpy(), y), f"{name}/{task}: labels differ from the fusion arm's"
        matrices = {name: arm_matrix(dumps, fam, aset, task, tics) for name, (fam, aset) in ARMS.items()}
        n = len(tics)

        full = {name: float(np.nanmean([metric_value(shape, y, m[i]) for i in range(len(m))]))
                for name, m in matrices.items()}
        boot = {name: np.full(args.n_boot, np.nan) for name in ARMS}
        for b in range(args.n_boot):
            idx = rng.integers(0, n, size=n)
            yb = y[idx]
            if shape != "regression" and yb.min() == yb.max():
                continue  # single-class resample: no ranking metric exists; left NaN and counted below
            for name, m in matrices.items():
                boot[name][b] = np.mean([metric_value(shape, yb, m[i, idx]) for i in range(len(m))])
        n_valid = int(np.isfinite(boot["fusion"]).sum())

        for left, right in CONTRASTS:
            deltas = boot[left] - boot[right]
            deltas = deltas[np.isfinite(deltas)]
            lo, hi = np.percentile(deltas, [2.5, 97.5])
            point = full[left] - full[right]
            paper_delta, paper_2se = np.nan, np.nan
            if (left, right) in PAPER_COLUMNS:
                d_col, se_col = PAPER_COLUMNS[(left, right)]
                paper_delta = float(table.loc[task, d_col])
                paper_2se = float(table.loc[task, se_col]) if se_col else np.nan
            elif (left, right) == ("fusion", "dynoff_fusion"):
                paper_delta = float(ablation.loc[task, "drop"])
                paper_2se = float(ablation.loc[task, "drop_2se"])
            paper_verdict = ("n/a" if not np.isfinite(paper_2se) else
                             "ahead" if paper_delta > paper_2se else
                             "behind" if paper_delta < -paper_2se else "tied")
            boot_verdict = "ahead" if lo > 0 else "behind" if hi < 0 else "tied"
            rows.append({"task": task, "shape": shape, "n_test": n, "contrast": f"{left}-{right}",
                         "delta_point": point, "boot_mean": float(deltas.mean()),
                         "ci_lo": float(lo), "ci_hi": float(hi),
                         "frac_positive": float((deltas > 0).mean()),
                         "n_boot_valid": n_valid, "n_boot": args.n_boot,
                         "paper_delta": paper_delta, "paper_2se": paper_2se,
                         "paper_verdict": paper_verdict, "boot_verdict": boot_verdict,
                         "agree": paper_verdict == "n/a" or paper_verdict == boot_verdict})

    out = pd.DataFrame(rows)
    out_path = in_dir / "paired_bootstrap.csv"
    out.to_csv(out_path, index=False)
    log.info(f"wrote {out_path} ({len(out)} rows)")

    show = out[["task", "contrast", "delta_point", "ci_lo", "ci_hi", "frac_positive", "n_boot_valid",
                "paper_verdict", "boot_verdict", "agree"]]
    with pd.option_context("display.width", 200, "display.max_rows", 200, "display.float_format",
                           "{:.4f}".format):
        print(show.to_string(index=False))
    write_tex(out, repo_root / args.tex_out)
    disagreements = out[(out["paper_verdict"] != "n/a") & (~out["agree"])]
    print(f"\nverdicts compared: {int((out['paper_verdict'] != 'n/a').sum())}, "
          f"disagreements: {len(disagreements)}")
    if not disagreements.empty:
        print(disagreements[["task", "contrast", "paper_verdict", "boot_verdict"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
