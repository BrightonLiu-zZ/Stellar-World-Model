# exp01 — window 256 × seq_len 16 (window-shrink ablation)

**Plan:** [`docs/plans/2026-07-09-window-shrink-ablation-exp01.md`](../../docs/plans/2026-07-09-window-shrink-ablation-exp01.md)

## Hypothesis
Shrinking the reconstruction window 1024 → 256 (z_dim fixed at 128) quadruples decoder bits-per-timestep, so the low-pass shortcut should weaken and the encoder should retain short-term structure. seq_len raised 4 → 16 holds the physical horizon at 4096 cadences, isolating the granularity knob. Windows are subdivided from the stored 1024-cadence `.npz` at pack time (Path A); per-segment MAD normalization makes this identical to a full rebuild.

## Directional gate
- **Leg 1 (mechanism):** recon overlay + residual spectrum at 256 visibly retain fast structure the 1024 decoder smoothed away.
- **Leg 2 (bellwether):** pulsating PR-AUC gap `trained_B(256) − untrained(256)` **> +0.008** (the 1024 gap).

Greenlight scale-up = Leg 1 ∧ Leg 2. eb reported as secondary; transit excluded (data-side).

## Outcome (2026-07-10, B/seed0)

**Verdict: mechanism confirmed, bellwether a wash.** Shrinking the window demonstrably broke the low-pass shortcut, but it did **not** make the self-supervised encoder beat an untrained one — the absolute gains came from added capacity, not from learning.

### Leg 1 — mechanism: PASS
- **Latent usage: ~6 → 128 active dims.** At 1024 the bottleneck collapsed to ~6 of 128 dims; at 256 all 128 stayed active for the whole run.
- **Reconstruction retains structure** (`figs/recon_overlay_*.png`): the 1024 decoder reconstructs near-flat noise; the 256 decoder tracks the dominant variability (pulsation envelopes, EB baseline wander, smooth trends). The low-pass shortcut is visibly weaker.

### Leg 2 — bellwether (trained − untrained gap): does NOT clear the bar
First-segment features, consistent pipeline at both windows (`figs/gap_table_*.csv`):

| task | trained 256 | untrained 256 | gap 256 | gap 1024 | gap widened? |
|---|---|---|---|---|---|
| pulsating | 0.771 | 0.768 | **+0.003** | +0.001 | +0.003 (still ≈ 0) |
| eb | 0.765 | 0.712 | +0.053 | +0.093 | **−0.040 (narrowed)** |
| transit | 0.111 | 0.081 | +0.031 | +0.019 | +0.012 (≈ base rate; excluded) |

Headline all-window trained PR-AUC (`results/results_table.csv`): pulsating **0.771** (↑ from 0.744), eb **0.781** (↑ from 0.747).

**Why the wash:** the smaller window lifted the *untrained* encoder too (pulsating 0.736→0.768, eb 0.640→0.712). Random conv features at 4× bits/step already separate the classes, so SSL still adds ~nothing on top. `trained ≈ untrained` for pulsating; the eb gap actually shrank. The gate ("pulsating gap widens materially past the ~+0.008 at 1024") is **not** cleanly met.

### Implication
Window-shrink fixes the *reconstruction* disease (low-pass) but is **not** the lever that makes SSL beat untrained. That bottleneck lives in the objective/probe (the parallel KL-tuning track, plan 2026-06-19), not in granularity. Recommend: **do not** promote 256×16 to canonical on these results; keep it as a documented sweep. A follow-up (exp02) could pair the small window with an objective change (e.g. spectral/high-pass recon loss) rather than shrinking further to 128.

## Skyline ceiling suite (plan 2026-07-11) — verdict Branch α (objective change)

Run at this config (B/seed0) to measure the headroom above 0.77 before spending exp02 compute. Modules `src/swm/eval/{features,skyline}.py`; append-only results in `results/skyline_{results,gate}.csv`; figure `figs/skyline_exp01_window256_seq16.png`. Same first-segment protocol as the gap table (reproduces trained-256 pulsating 0.7711 / untrained 0.7678).

Test PR-AUC (±95% paired star-bootstrap CI), tasks pulsating + eb:

| task | untrained | trained | A1 logistic (feats) | A2 GBM (feats) | B1 supervised conv | GBM on **trained μ** | GBM on **untrained μ** | base |
|---|---|---|---|---|---|---|---|---|
| pulsating | 0.768 | 0.771 | 0.789 | **0.858** | 0.831 | 0.767 | **0.824** | 0.107 |
| eb | 0.712 | 0.765 | 0.742 | **0.801** | 0.768 | 0.745 | 0.731 | 0.097 |

- **Gate (`A1 − untrained > 2·SE_diff`): FAILS both** (pulsating +0.021 vs 0.049; eb +0.029 vs 0.041) → A1 ≈ untrained. Label-shuffle → base rate (0.110/0.112), pipeline leak-free.
- **Disambiguating diagnostic — nonlinear (GBM) probe on the encoder μ (`info_in_mu`, added 2026-07-11):** FALSE both tasks. GBM extracts **nothing extra** over the linear probe on the trained μ (pulsating −0.005, eb −0.020; both < 2·SE_diff). The discriminative signal is **not in the trained μ** — linearly *or* nonlinearly.
- **The kicker:** GBM on the **untrained** (random-init) μ scores **0.824** for pulsating — *higher* than the trained μ (0.767) and higher than the trained linear probe (0.771). SSL training **degrades** μ below a random projection for the pulsating task: the MSE-reconstruction objective smooths away exactly the fast-oscillation variance a random conv preserves.
- Supervised-gap fraction `(trained−untrained)/(B1−untrained)`: pulsating **0.05**, eb **0.95**.

**Verdict — Branch α (objective change), reversing the initial Branch-β read.** Branch β (pooling/probe on cached μ) is **ruled out**: you cannot re-pool information that is absent from μ, and `info_in_mu=False` shows it is absent. The signal demonstrably exists in the data (features→GBM 0.858, *random*-μ→GBM 0.824) but the current MSE objective squeezes it out of the learned μ — SSL training makes pulsating separability *worse* than random init. The correct lever is the **training objective (exp02: spectral / high-pass reconstruction)**, and the evidence is now positive, not merely "not ruled out." **eb is effectively solved** (SSL captures 95% of the supervised gap; nonlinear/untrained μ add nothing) — focus exp02 on pulsating. Caveat: untrained μ is a single fixed-seed random projection (magnitude may wobble), but the qualitative "trained ≤ random under GBM" is robust and low-pass-consistent. Next: own dated plan for exp02, upgrade the Slack proposal with this headroom evidence. Not started this session.
