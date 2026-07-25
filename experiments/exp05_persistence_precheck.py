"""exp05 pre-flight: persistence-gap pre-check (grill 2026-07-22) -- does one-step GRU dynamics beat
copy-last (persistence)? If the GRU only learned the identity map, the free-running rollout axis has no
signal. Compares within-star mse(mu_{t+1}, mu_t) [persistence] vs the trained GRU one-step dyn (logged),
in the SAME raw-mu space as train.losses.dynamics_loss (F.mse_loss(pred_next, mu_seq[:,1:])).

Reads the cached first-segment window-mu of the exp03 leader arms (consecutive windows within the first
segment per star). Result (2026-07-22): GRU beats persistence 2.6-6.7x and persist/mu_var ~= 1.0 (adjacent
latents ~uncorrelated) -> dyn is small only because the latent SCALE is tiny, NOT because it's trivial ->
the difficulty axis has signal. Feeds docs/plans/2026-07-22-exp05-dynamics-axis.md.

Run (repo root):  python experiments/exp05_persistence_precheck.py
"""
import numpy as np
from pathlib import Path

CACHES = {
    "lpsd_s0": r"experiments/exp03_fb0p02_b0p1_lpsd/models/B_seed0/extracted/first_segment_window_mu_best_recon_aux.npz",
    "lpsd_s1": r"experiments/exp03_fb0p02_b0p1_lpsd/models/B_seed1/extracted/first_segment_window_mu_best_recon_aux.npz",
    "comb_s0": r"experiments/exp03_fb0p02_b0p1_comb/models/B_seed0/extracted/first_segment_window_mu_best_recon_aux.npz",
}
ROOT = Path(__file__).resolve().parents[1]
GRU_DYN = {"lpsd_s0": 0.0146, "lpsd_s1": 0.0146, "comb_s0": 0.0075}  # logged final train dyn (approx)


def persistence_stats(npz, split="train"):
    """Within-star persistence mse = mean over adjacent-window pairs of (mu_{t+1}-mu_t)^2.
    Matches F.mse_loss reduction; grouped by *_counts so pairs never cross star boundaries."""
    mu = npz[f"{split}_mu"].astype(np.float64)
    counts = npz[f"{split}_counts"]
    diffs = []; idx = 0; multiwin_stars = 0
    for c in counts:
        c = int(c)
        block = mu[idx:idx + c]; idx += c
        if c >= 2:
            diffs.append(block[1:] - block[:-1]); multiwin_stars += 1
    D = np.concatenate(diffs, axis=0)
    return dict(persistence_mse=float((D ** 2).mean()), mu_var=float(mu.var()),
                n_pairs=len(D), z=mu.shape[1], n_stars=len(counts), multiwin_stars=multiwin_stars)


def main():
    print("## WITHIN-STAR persistence baseline vs trained GRU one-step dyn (same raw-mu MSE space)")
    for name, rel in CACHES.items():
        p = ROOT / rel
        if not p.exists():
            print(f"  {name}: MISSING ({rel})"); continue
        st = persistence_stats(np.load(p, allow_pickle=False), split="train")
        gru = GRU_DYN.get(name, float("nan"))
        print(f"  {name}: persistence_mse={st['persistence_mse']:.5f}  GRU_dyn~{gru:.4f}  "
              f"GRU beats persist by {st['persistence_mse']/gru:.1f}x  |  mu_var={st['mu_var']:.4f}  "
              f"persist/mu_var={st['persistence_mse']/st['mu_var']:.3f}  "
              f"multiwin_stars={st['multiwin_stars']}/{st['n_stars']}")


if __name__ == "__main__":
    main()
