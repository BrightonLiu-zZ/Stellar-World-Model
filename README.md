# Stellar-World-Model

Self-supervised latent dynamics world model on TESS stellar light curves.

## Overview

The central hypothesis: **variability types (stellar rotation, planetary transits) are linearly separable from a latent space trained purely on raw PDCSAP flux, with no labels during pretraining.** A GRU dynamics head predicts the next latent state z_{t+1} from preceding states z_{1:t}, forcing the encoder to factor out temporal structure rather than just reconstruct it.

Primary evaluation (v1): binary variability classification {rotation, transit} via logistic regression on frozen encoder embeddings. A v1-supplementary rotation-period regression task fits linear regression of `rotation_period` (days) on the rotation=1 subset. Secondary evaluation (v1b): spectroscopic regression {Teff, log g, [Fe/H]} via linear regression.

The core ablation compares Variant A (VAE reconstruction + KL only) against Variant B (full world model with latent dynamics objective) to isolate the causal contribution of the dynamics term.

## Architecture

```
Input [B, 1024, 1]  — one PDCSAP_FLUX window, 1024 cadences (~13.7 days)
→ Encoder: 4× (Conv1D + BN + ReLU + MaxPool) → FC → (μ, log σ²) → z ∈ ℝ^128
→ Dynamics: GRU — predicts ẑ_{t+1} from z_{1:t}
→ Decoder: FC → 4× (ConvTranspose1D + BN + ReLU) → [B, 1024, 1]
```

Loss: `MSE(recon) + β·KL + λ·MSE(ẑ_{t+1}, encoder(x_{t+1}))`

Training sequences: `SEQ_LEN=4` consecutive NaN-free windows from a single continuous segment (~5.7 days at T=1024, stride=1024); sequences never cross sector or gap boundaries.

Downstream evaluation: freeze encoder, mean-pool z over SEQ_LEN windows → one embedding per segment, fit logistic / linear regression. No fine-tuning, no MLP heads.

See [docs/architecture.md](docs/architecture.md) for full design rationale, ablation variants, baseline comparisons (FALCO, Astromer 2, ASTRAFier), and data layout.

## Pipeline

| Stage | Script / Notebook | Description |
|---|---|---|
| 0a | `src/notebooks/characterize_data_v2.ipynb` (legacy) → `processed/spoc_sector_map.csv` | TIC list for the bulk pipeline (Tmag<10, no plx cut per ADR-0002): 195,883 TICs × SPOC sector pairs. `processed/df_final.csv` is the legacy 34k-row output kept for reference. |
| 0b | `src/build_sequences_bulk.py` (canonical) / `src/build_sequences.py` (deprecated) | Bulk MAST curl-script download of SPOC PDCSAP_FLUX, segment at NaN gaps and at time gaps > 5× median cadence (ADR-0003), MAD-normalize, slide windows, save NaN-free windows to `processed/sequences/*.npz` |
| 0c | `src/build_labels.py` | Cross-match TIC IDs to APOGEE DR17 → GSP-Spec → LAMOST DR11 → `labels/stellar_params.csv` |
| 0d | `src/build_variability_labels.py` | Cross-match TIC IDs to TARS + flatwrm2 + TOI → `labels/variability_labels_star.csv` |
| 1 | *(not started)* | Train Conv1D-VAE + GRU on SDSC Expanse |
| 2 | *(not started)* | Linear probe evaluation — per-class F1 / R² |

See [docs/STATUS.md](docs/STATUS.md) for current counts and progress on each stage.

## Data

### Input
- **Source:** TESS SPOC 2-min cadence, PDCSAP_FLUX only (never SAP_FLUX)
- **Sample:** ~195k TICs (Tmag < 10, no parallax cut; see `docs/adr/0002-drop-plx-cut.md`) with ≥ 1 SPOC sector — listed in `processed/spoc_sector_map.csv` (669k (TIC, sector) pairs). The earlier 34k-star sample (Tmag<7, plx>10 mas) is retained only as `processed/df_final.csv` and is no longer used.
- **Access:** MAST bulk curl scripts per sector (canonical) → FITS via `astropy.io.fits`; `lightkurve` retained for single-star debugging. No FITS files are committed to this repo.

### Labels
- **v1 — variability (primary):** binary `[rotation, transit]` per star
  - Rotation: TARS (Boyle, Bouma & Mann 2026)
  - Transits: NASA Exoplanet Archive TOI (non-retired only)
  - Flares: flatwrm2 (Vida et al. 2025) — produced and retained in `variability_labels_star.csv`, excluded from v1 eval (see `docs/adr/0001-drop-flare-from-v1-eval.md`)
- **v1-supplementary — rotation period:** `rotation_period` (days) regression on the rotation=1 subset; same frozen encoder, linear regression head
- **v1b — spectroscopic (supplementary):** {Teff, log g, [Fe/H]} — APOGEE DR17 → Gaia DR3 GSP-Spec → LAMOST DR11 (priority fallback); ~13.8% match rate on this bright sample

### Key constraints
- NaN windows are **discarded** — no interpolation, zero-fill, or padding at any stage
- Sequences never stitch across sectors or the mid-sector downlink gap
- Sequences shorter than SEQ_LEN are discarded (not padded)

## Setup

```bash
conda env create -f environment.yml
conda activate astro
```

`environment.yml` pins the direct dependencies used by this project (Python 3.10, numpy, pandas, scipy, astropy, astroquery, lightkurve, tenacity, scikit-learn, PyTorch). Full transitive dependency versions are captured in the conda env export used to generate it.

## Usage

```bash
# Stage 0b (canonical) — bulk download + window TESS light curves via MAST curl scripts
python src/build_sequences_bulk.py
python src/build_sequences_bulk.py --resume                     # skip (TIC, sector) pairs already done
python src/build_sequences_bulk.py --sectors 1-101 --workers 16

# Stage 0b (deprecated) — lightkurve-per-star, single-star debugging only
python src/build_sequences.py --resume

# Stage 0c — spectroscopic label cross-match (supplementary)
python src/build_labels.py
python src/build_labels.py --resume
python src/build_labels.py --limit 5            # smoke test on 5 stars

# Stage 0d — variability label cross-match (primary)
python src/build_variability_labels.py
python src/build_variability_labels.py --resume
python src/build_variability_labels.py --limit 5
```

All scripts support `--resume` (checkpoint-based skip of completed TICs) and `--limit N` for smoke testing.

## Project Structure

```
src/
  build_sequences_bulk.py       Stage 0b (canonical)
  build_sequences.py            Stage 0b (deprecated, lightkurve-per-star)
  build_labels.py               Stage 0c
  build_variability_labels.py   Stage 0d
  notebooks/                    Stage 0a + EDA + sanity checks
processed/spoc_sector_map.csv   Stage 0a output: (TIC, tmag, sector) for the bulk pipeline (195k TICs)
processed/
  df_final.csv                  legacy Stage 0a output (34k TICs, plx>10), no longer used
  sequences/                    per-segment .npz files — Stage 0b canonical output (gap-guarded, ADR-0003)
  sequences_legacy/             pre-ADR-0003 .npz files, read-only legacy
  build_sequences_bulk_progress.csv   (TIC, sector) checkpoint for the bulk script
labels/
  variability_labels_star.csv   multi-label variability annotations (Stage 0d output; stale 34k sample)
  stellar_params.csv            spectroscopic labels (Stage 0c output; stale 34k sample)
models/                         model checkpoints (Stage 1, not yet populated)
docs/
  architecture.md               full design doc
  labels-sources.md             catalog details and acceptance criteria
  STATUS.md                     live pipeline progress
  adr/                          architecture decision records (ADR-0001..0003)
```

## References

- **TARS** — Boyle, A. W., Bouma, L. G., & Mann, A. W. (2026). *TESS All-Sky Rotation Survey*. arXiv:2603.05586. Data: Zenodo record 19917941 (v2, current).
- **flatwrm2** — Vida, K., et al. (2025). *flatwrm2 TESS flare catalog, sectors 1–69*. A&A. arXiv:2412.12989. Data: Zenodo (public).
- **TOI** — NASA Exoplanet Archive TESS Object of Interest list. Accessed via `astroquery.ipac.nexsci`.
- **APOGEE DR17** — Abdurrouf et al. (2022). VizieR `III/286/catalog`.
- **Gaia DR3 GSP-Spec** — Recio-Blanco et al. (2023). VizieR `I/355/paramp`.
- **LAMOST DR11** — Accessed via VizieR `V/162/dr11sl`.
