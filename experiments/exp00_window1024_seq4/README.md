# exp00 — window 1024 × seq_len 4 (baseline)

Canonical v1 geometry. This is the migrated Stage-1/2 baseline (previously in the repo-root `processed/packed`, `models/`, `results/`), relocated here 2026-07-10 for uniform per-experiment layout. Re-verified after the move: `extract`+`probe` reproduce the pre-move `results_table.csv` byte-for-byte (max abs diff 0.0 across A/B/C × transit/eb/pulsating).

- **Config:** `data/default.yaml` (window 1024, seq_len 4), experiment `exp00_window1024_seq4`.
- **Variants:** A (λ=0), B (λ=1), C (λ=5), seed 0.
- **Diagnostic notebook:** `src/notebooks/model_diagnostics.ipynb` (points here via `exp_dir`).

## Headline test-split PR-AUC (B, seed 0)

| task | trained_B | untrained_enc | gap |
|---|---|---|---|
| pulsating | 0.744 | 0.736 | **+0.008** |
| eb | 0.747 | 0.640 | +0.107 |
| transit | 0.106 | — | ~base rate (excluded from gate) |

The near-zero pulsating gap is the disease exp01 targets: the trained encoder barely beats a random-init one (low-pass shortcut).
