"""exp10 forensics -- build the DERIVED mu caches F-A, F-C and F-D score through the F1 machinery.

Every forensic here asks "what happens to the fusion delta if the mu columns are different?", and the
cheapest honest way to ask that is to write a new `{arm}.npz` cache holding the transformed columns and
then run `analyze_f1_fusion_scorecard.py` against it unchanged. Nothing about the scoring path moves, so
a difference in the output is a difference in the columns and nothing else.

WHAT A DERIVED CACHE IS. One row per star (the star-level pooled vector) stored in the per-window block
layout the real caches use, so `load_mu_cache` reads it and F1's `pool(..., "mean")` returns that row
untouched. This is exact for readout `mean` and MEANINGLESS for `mean_std` (the std over a single row is
zero), so every F1 invocation against these caches must pass `--readouts mean`.

TRANSFORM FAMILIES (`--family`):
    identity     mean-pooled mu, 128 cols. The footing gate: F1 over this must reproduce the published
                 `eb` @ `mean` mu score 0.7710 to 5e-4, and it keeps the arm names so F1's own
                 reproduction_check fires without being told anything.
    pca          F-A. PCA-k for k in {4, 8, 16, 32, 64} plus a whitened un-truncated PCA-128.
    windowstats  F-C. mean / mean+std / mean+std+q10+q90 / mean+std+max over each star's windows.
    ensemble     F-D. the 6 fbwd seeds concatenated (768 cols), PCA-64 and PCA-128 of that concat, and
                 the per-dim mean over seeds.

LEAKAGE RULE. Every PCA is fit on the TRAIN split of the population it will be applied to, per seed, per
population. Fitting on all stars would leak the test set into the transform and void the forensic; the
splits used are the probes' own, because they are the cache's own.

MU-CACHE TRAP (exp09 method debt, carried into exp10). Caches key on `{arm}.npz` with no checkpoint and
no transform in the key, and readers short-circuit on exists(). Every family therefore gets its OWN
output directory and this script refuses to write into `exp08_menu_channel/`.

Run (repo root, swm env, PYTHONPATH=src; CPU-only, seconds to a couple of minutes per family):
    PYTHONUNBUFFERED=1 python experiments/exp10_build_derived_mu.py --family identity
    PYTHONUNBUFFERED=1 python experiments/exp10_build_derived_mu.py --family pca
"""
from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from tqdm.auto import tqdm

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_exp08_menu_channel import load_mu_cache  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S", handlers=[logging.StreamHandler(sys.stdout)], force=True)
log = logging.getLogger("exp10_derived_mu")

source_home = repo_root / "experiments" / "exp08_menu_channel"
out_home = repo_root / "experiments" / "exp10_forensics"
fbwd_seeds = [0, 1, 2, 3, 4, 5]
pca_ks = [4, 8, 16, 32, 64]
splits = ["train", "test"]
# The two populations F1 scores. `subset` carries the four v1 tasks plus rotation_period and ijspeert;
# `pool` carries the five pool-borne menu probes. Both need every derived arm or F1 cannot score it.
populations = {"subset": "subset_mu_cache", "pool": "mu_cache"}


def star_stats(blocks: list[np.ndarray], stat: str) -> np.ndarray:
    """
    Reduce every star's (n_window, z) block to one row under one window statistic.
    `mean` is the published pooling; the others are F-C's candidate localized channels, each of which
    keeps information that averaging over a star's windows throws away.
    Returns an (n_star, z * n_stat) matrix in the cache's own star order.
    """
    rows = []
    for block in blocks:
        parts = [block.mean(axis=0)]
        if stat == "mean":
            pass
        elif stat == "mean_std":
            parts.append(block.std(axis=0))
        elif stat == "mean_std_q10_q90":
            parts.append(block.std(axis=0))
            parts.append(np.quantile(block, 0.1, axis=0))
            parts.append(np.quantile(block, 0.9, axis=0))
        elif stat == "mean_std_max":
            parts.append(block.std(axis=0))
            parts.append(block.max(axis=0))
        else:
            raise ValueError(f"unknown window statistic {stat}")
        rows.append(np.concatenate(parts))
    return np.stack(rows, axis=0).astype(np.float32)


def write_cache(path: Path, tics: dict[str, list[int]], values: dict[str, np.ndarray]) -> None:
    """
    Persist one derived arm in the `{split}_mu / {split}_counts / {split}_tics` layout `load_mu_cache`
    reads, with exactly one window row per star.
    Fails loud rather than overwriting: a silently reused cache under a new label is the exp09 trap this
    whole directory layout exists to avoid.
    """
    assert not path.exists(), f"refusing to overwrite an existing derived cache at {path}"
    assert "exp08_menu_channel" not in str(path), f"derived caches must never be written into {path}"
    arrays = {}
    for split in splits:
        arrays[f"{split}_mu"] = values[split].astype(np.float32)
        arrays[f"{split}_counts"] = np.ones(len(tics[split]), dtype=np.int64)
        arrays[f"{split}_tics"] = np.array(tics[split], dtype=np.int64)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)


def link_or_copy(source: Path, target: Path) -> None:
    """
    Give one derived cache a second arm name without a second copy on disk.
    F-D needs six identically-valued arms so that F1's GBM fits pick up random_state 0-5 and the delta
    stays paired against the engineered arm; the mu is the same for all six, so a hardlink says that.
    """
    if target.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError as exc:
        log.warning(f"hardlink {target.name} failed ({exc}); copying instead")
        shutil.copy2(source, target)


def load_pooled(population: str, arm: str, stat: str = "mean") -> tuple[dict, dict]:
    """
    Read one source arm's per-window cache and reduce it to (tics, star-level rows) per split.
    This is the only place the published caches are touched, and it is read-only.
    """
    cache = load_mu_cache(source_home / populations[population] / f"{arm}.npz")
    tics, values = {}, {}
    for split in splits:
        tics[split] = list(cache[split][0])
        values[split] = star_stats(cache[split][1], stat)
    return tics, values


def build_identity(out_dir: Path) -> list[str]:
    """The passthrough arms: mean-pooled mu under the ORIGINAL arm names, so F1's repro gate fires."""
    arms = []
    for arm in [f"hann0p3_fbwd_s{s}" for s in fbwd_seeds] + ["untrained"]:
        for population, sub in populations.items():
            tics, values = load_pooled(population, arm)
            write_cache(out_dir / sub / f"{arm}.npz", tics, values)
        arms.append(arm)
    return arms


def build_pca(out_dir: Path) -> list[str]:
    """F-A: PCA-k and whitened PCA-128 of the mean-pooled mu, fit per (arm, population) on train only."""
    arms = []
    sources = [f"hann0p3_fbwd_s{s}" for s in fbwd_seeds] + ["untrained"]
    transforms = []
    for k in pca_ks:
        transforms.append((f"pca{k}", k, False))
    transforms.append(("whiten128", 128, True))
    for arm in tqdm(sources, desc="pca source arms", total=len(sources)):
        for population, sub in populations.items():
            tics, values = load_pooled(population, arm)
            for name, k, whiten in transforms:
                model = PCA(n_components=k, whiten=whiten, random_state=0)
                model.fit(values["train"])  # fit on TRAIN stars only; test never enters the basis
                projected = {}
                for split in splits:
                    projected[split] = model.transform(values[split])
                write_cache(out_dir / sub / f"{derived_arm(arm, name)}.npz", tics, projected)
    for name, _, _ in transforms:
        for arm in sources:
            arms.append(derived_arm(arm, name))
    return arms


def build_windowstats(out_dir: Path) -> list[str]:
    """F-C: the three richer window statistics beside the published `mean` (which `identity` already holds)."""
    arms = []
    sources = [f"hann0p3_fbwd_s{s}" for s in fbwd_seeds] + ["untrained"]
    stats = ["mean_std", "mean_std_q10_q90", "mean_std_max"]
    for arm in tqdm(sources, desc="windowstat source arms", total=len(sources)):
        for population, sub in populations.items():
            for stat in stats:
                tics, values = load_pooled(population, arm, stat)
                write_cache(out_dir / sub / f"{derived_arm(arm, stat)}.npz", tics, values)
    for stat in stats:
        for arm in sources:
            arms.append(derived_arm(arm, stat))
    return arms


def build_ensemble(out_dir: Path) -> list[str]:
    """
    F-D: the six fbwd seeds pooled into one arm, and two ways of shrinking that 768-column arm back down.
    There is no encoder-seed spread left after this by construction, so each variant is written under six
    identical seed names purely to give F1's GBM its own random_state spread; the CSV and the README say
    so rather than letting a reader read encoder variance into a hardlink.
    """
    variants = ["ens_concat", "ens_pca64", "ens_pca128", "ens_seedmean"]
    for population, sub in populations.items():
        per_seed = []
        star_order = None
        for seed in fbwd_seeds:
            tics, values = load_pooled(population, f"hann0p3_fbwd_s{seed}")
            if star_order is None:
                star_order = tics
            for split in splits:
                assert star_order[split] == tics[split], f"{population}/{split}: seed star lists differ"
            per_seed.append(values)

        concat, seedmean = {}, {}
        for split in splits:
            stack = []
            for values in per_seed:
                stack.append(values[split])
            concat[split] = np.concatenate(stack, axis=1)
            seedmean[split] = np.mean(np.stack(stack, axis=0), axis=0)

        tables = {"ens_concat": concat, "ens_seedmean": seedmean}
        for k in [64, 128]:
            model = PCA(n_components=k, random_state=0)
            model.fit(concat["train"])  # fit on TRAIN stars only
            projected = {}
            for split in splits:
                projected[split] = model.transform(concat[split])
            tables[f"ens_pca{k}"] = projected

        for variant in variants:
            first = out_dir / sub / f"{variant}_s0.npz"
            write_cache(first, star_order, tables[variant])
            for seed in fbwd_seeds[1:]:
                link_or_copy(first, out_dir / sub / f"{variant}_s{seed}.npz")

    arms = []
    for variant in variants:
        for seed in fbwd_seeds:
            arms.append(f"{variant}_s{seed}")
    return arms


def derived_arm(source_arm: str, transform: str) -> str:
    """
    Name a derived arm so `arm_parts` recovers (family, seed) and the family names the transform.
    `hann0p3_fbwd_s3` + `pca16` --> `hann0p3_fbwd_pca16_s3`; `untrained` --> `untr_pca16_s0`.
    The family deliberately does NOT end in `_fbwd`, which keeps F1's fbwd-minus-off arm contrast from
    firing on a pair of arms that never had an off twin.
    The untrained twin is `untr_`, NOT `untrained_`: `arm_parts` HARDCODES the family to "untrained" for
    any name starting with that word, so every untrained variant would collapse into one family at seed
    0 and silently overwrite the others in the summary. Measured, not guessed -- it happened here first.
    """
    if source_arm == "untrained":
        return f"untr_{transform}_s0"
    stem, _, seed = source_arm.rpartition("_s")
    return f"{stem}_{transform}_s{seed}"


builders = {"identity": build_identity, "pca": build_pca, "windowstats": build_windowstats,
            "ensemble": build_ensemble}


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the exp10 forensics derived mu caches.")
    ap.add_argument("--family", required=True, choices=list(builders))
    ap.add_argument("--out-root", default=None, help="default experiments/exp10_forensics/<family-dir>")
    args = ap.parse_args()

    dir_of = {"identity": "identity", "pca": "fa_pca", "windowstats": "fc_windowstats",
              "ensemble": "fd_ensemble"}
    out_dir = Path(args.out_root) if args.out_root else out_home / dir_of[args.family]
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"source {source_home} (read-only) --> {out_dir}")

    arms = builders[args.family](out_dir)
    # newline="\n": the arm list is consumed by a shell `for` loop, and a CRLF ends up inside the arm
    # name itself, where it fails as an unreadable family/seed rather than as a bad file.
    (out_dir / "arms.txt").write_text("\n".join(arms) + "\n", encoding="utf-8", newline="\n")
    log.info(f"wrote {len(arms)} derived arms x {len(populations)} populations under {out_dir}")
    log.info(f"arm list at {out_dir / 'arms.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
