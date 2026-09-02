# L1 — the flare label audit: both classes rebuilt, one at a time (2026-08-26)

`experiments/analyze_l1_flare_negatives.py` → `experiments/l1_flare_negatives/l1_{probe,summary,validation}.csv`
+ `l1_star_scores.parquet`. Roadmap row **L1**, decision **D21**, extended by the 2026-08-26 grilling.
Upstream: `src/swm/eval/flare_search_universe.py` → `labels/qc/flare_search_universe.parquet`.

*(`docs/` and `*.csv` are gitignored in this repo, so this file is the committed record of the verdict.)*

## What D21 asked for, and why it changed

D21 specified route **(d)+(e)**: build high-confidence negatives from a *second* TESS flare catalog,
then visually inspect if flare is still null. Two things measured on day one redirected it.

**(1) The route-(d) gate fired NO.** Pietras et al. 2022 (ApJ 935, 143; WARPFINDER, ~330k stars,
S1–39) is the named candidate. It publishes **no analyzed-star table** — only per-sector counts and
flare examples — and it carries its own undisclosed-to-us selection (*"all stars with spectral types
earlier than F1 were rejected"*, plus quality-flag cuts), which bites hard on a Tmag<10 sample and
would need a TIC Teff cross-match merely to *guess* at their universe. It has the same trap flatwrm2
has.

**(2) A better instrument existed in the paper we already depend on.** Seli et al. 2025 Sect. 2.1
Eq. (1) states flatwrm2's own search selection exactly: smooth with a 31-point (1 hr) running median,
keep only if `σ_ratio = STD(smoothed)/STD(original) > 0.4`. Table 1 gives the funnel: 1,258,154 2-min
light curves through S69 → **444,963 searched** (35.4%) → 121,895 vetted flares. `σ_ratio` is
**scale-invariant**, so our per-segment MAD normalization does not perturb it and the selection is
reconstructible from our packed windows alone — no network, no second catalog, no license risk.
Adopted as route **(g)**.

Two corrections to the standing record fell out of reading the source:

- **The cut direction was recorded backwards** in the session handoff (`> 0.4` reads there as the
  discard rule; it is the **keep** rule). The consequence the handoff drew — the searched population
  is biased toward astrophysically variable stars — is unaffected and stands.
- **The citation was wrong.** ADR-0001, `README.md` (×2) and `docs/labels-sources.md` all said
  *"Vida et al. 2025"*. `flatwrm2` is the **detector** (Vida et al. 2021, an LSTM built for *Kepler*);
  **Seli et al. 2025** retrained it for TESS 2-min and produced the catalog. All four sites corrected.

## The measurement that reframed the task

`flare` is not a known null being defended. At the F1 headline it is a **printed win** —
`features ⊕ µ` − `features` = **+0.0506, >2·SE**, third-largest of eleven tasks. L1 is therefore an
**audit of a printed number**, and it has a specific confound to test:

> flatwrm2 only searched curves with `σ_ratio > 0.4`, so **every flare positive is by construction a
> star whose astrophysical variation dominates its noise**. Meanwhile 10,000 of the 20,838 negatives
> are the `quiet` pool, defined as matched in *no* variability catalog. The probe could score on
> `flare` by detecting variability that the catalog's own pre-cut wrote into the label.

Restricting negatives to searched-and-clean forces them through the **same gate the positives
passed**. That is the matched control for exactly this confound.

Both label classes are broken, in opposite directions, so the design is a 2×2 rather than one
corrected number — four population masks over the **existing** µ caches. Zero GPU, zero re-extraction,
so the exp09 cache-key trap does not apply.

| | negatives = all `flare_ever=0` | negatives = searched-and-clean |
|---|---|---|
| **positives = all `flare_ever=1`** | `as_published` | `neg_searched` |
| **positives = flare on screen** | `pos_covered` | `matched` (headline) |

### Event coverage — the positive-class hole, measured

Folding Seli Table-3 peak times onto the cadences we actually packed:

| positive-class definition | n | frac of 2,022 |
|---|---|---|
| `flare_ever=1` with packed data (**today's label**) | 2,022 | 100% |
| flare catalogued in a sector we hold on disk | 1,279 | 63.3% |
| flare inside a retained window, over **all** the star's npz | 658 | 32.5% |
| flare inside a retained window, in **the segment the eval encodes** | 392 | **19.4%** |

The eval bag scope is `first-segment`, so **81% of the positive class shows the model no flare**. This
confirms the previously-unverified "19%" that had been quoted to the advisor; it was computable from
`labels/qc/flare_window_labels_pool.parquet`, which already existed.

## Controls

| control | result |
|---|---|
| `as_published` fusion delta vs F1's published flare row, 6 seeds | **4.2e-17** |
| `as_published` `features_only` vs F1's, deterministic arm | **5.6e-17** |
| reconstruction recall on demonstrably-searched light curves | **0.984** (1,631/1,657), floor 0.90 |

The first is not a tautology: `as_published` applies no mask, so it *is* the F1 measurement reached
through a different script. If it disagreed, the masks would not be the only thing that changed.

**The validation gate is per-(star, sector), and the star-level version of it cannot pass.** 332 pool
positives have no S≤69 data at all, capping a star-level recall at 0.836 against a 0.90 floor — the
route would have been voided for a reason unrelated to fidelity. The gate as run asks: for every
(star, sector) where the catalog records a flare **and** we hold that sector, does our recomputed
`σ_ratio` clear 0.4? It does for 98.4%, against an unconditional base rate of 73.9% — real lift, which
is what licenses applying the reconstruction to stars with no ground truth.

### Why our pass rate (73.9%) sits above Seli's published 35.4%

`σ_ratio` is an SNR statistic — the denominator carries photon noise the hour-wide median removes — so
it rises toward bright stars, and our corpus is cut at Tmag<10 while Seli's population runs far
fainter. Both axes reproduce that, and both bracket the published rate:

| Tmag | pass rate | | pool stratum | pass rate |
|---|---|---|---|---|
| < 6 | 0.969 | | **quiet** | **0.423** |
| 8.5–9.0 | 0.745 | | solar_like_osc | 0.748 |
| 9.0–9.5 | 0.549 | | rgb_vs_heb | 0.935 |
| **9.5–10.0** | **0.428** | | osc_giant | 0.993 |

At the faint edge of our own cut the rate is 0.428, approaching 0.354; and the pool is 56% catalog
positives by construction, with `quiet` alone at 0.423. Nothing indicates a broken reconstruction —
and route (g) has real bite, deleting ~58% of the quiet pool while retaining 75–99% of the variable
strata.

## Verdict — the hypothesised confound is REFUTED

Fusion delta (`features ⊕ µ` − `features`), readout `mean`, 6 seeds, paired star bootstrap (2,000):

| cell | n_pos | prevalence | features alone | **fbwd** | off | untrained | 2·SE | boot 95% CI | claimable |
|---|---|---|---|---|---|---|---|---|---|
| `as_published` | 304 | 0.089 | 0.474 | **+0.0506** | +0.0330 | +0.0060 | 0.0078 | [+0.020, +0.079] | **yes** |
| `neg_searched` | 304 | 0.148 | 0.550 | **+0.0507** | +0.0305 | +0.0124 | 0.0071 | [+0.018, +0.083] | **yes** |
| `pos_covered` | 66 | 0.021 | 0.388 | +0.0718 | +0.0412 | +0.0031 | 0.0276 | [−0.020, +0.156] | no |
| `matched` | 66 | 0.036 | 0.462 | +0.0539 | +0.0236 | −0.0260 | 0.0221 | [−0.035, +0.131] | no |

**The negative-class fix moves the number by +0.0001.** Deleting 1,375 of 3,125 test negatives — 58%
of the quiet pool — moving prevalence 0.089 → 0.148 and the engineered baseline 0.474 → 0.550, leaves
the fusion delta at +0.0507 against +0.0506. D21's premise is that the negative class is what is
broken about `flare`; at full power that premise is testable, and **for this statistic it is false**.

Supporting controls hold. The dynamics contrast Δ(fbwd) − Δ(off) survives and slightly strengthens
(+0.0176 → +0.0202, both >2·SE), and the untrained arm sits at +0.006/+0.012 with CIs spanning zero,
so the trained arm's gain is not generic column addition. `µ` alone remains below `features` alone in
every cell, unchanged from the menu pattern.

### The outcome matches no pre-registered branch, and is recorded as a mismatch

Pre-registration (fixed before any cell was scored) had a win row (Δ > 2·SE) and a null row (Δ within
2·SE). `matched` is Δ = +0.0539 with 2·SE = 0.0221 — the **seed gate fires** — while the bootstrap CI
[−0.035, +0.131] **fails claimability**. Neither row applies. Following the exp09-P8 precedent this is
recorded as a mismatch rather than assigned to the nearest branch.

**It is underpowered, not null.** The point estimate barely moved (+0.051 → +0.054); the CI widened
because n_pos fell 304 → 66. That is precisely the failure mode the bootstrap amendment was added to
expose. **More data does not fix it:** the unused `val` split would roughly double n_pos to ~130 and
shrink the CI half-width from ≈0.083 to ≈0.059, still wider than the 0.054 effect, while breaking
comparability with every other table. Not spent.

### Reporting rule — the user's call, taken 2026-08-26

Offered (A) record the mismatch and claim the full-power result, or (B) treat not-claimable as null,
which triggers D21's two-defense condition. **The user chose (B), the conservative reading of their
own D21 condition.** Consequently:

> `flare` is carried as a **null** for ML4PS, and D21's condition binds: the null is printable only
> with **both** defenses — this negative-set audit **and** a manual visual pass. The visual pass is a
> gate on printing flare at all, not an optional extra.

Recorded honestly: the measurement itself does not read as a null at full power. (B) is a reporting
policy applied on top of it, and the numbers above are what a reader is owed beside it.

## Estimator notes that bind

- Prevalence changes by design across cells, so paired deltas keep their **sign, not their magnitude**
  (R8-F1). Absolute PR-AUC is not comparable between cells and is quoted per cell only.
- Every restricted cell is a **small probe** by the standing rule (n_test < 400 or n_pos < 100), so
  the fusion rows carry R1's paired star bootstrap and `claimable` requires both gates. This was added
  **before** any cell was scored, and it makes the win branch harder to reach, never easier — the only
  direction an estimator change may run after pre-registration.
- An uncovered positive is **dropped, never relabelled negative**: a flare star whose flare we did not
  pack is not evidence of absence. An unsearched negative is likewise dropped. A star with no S≤69
  data is **unknown** — neither searched nor unsearched — and is dropped rather than guessed. 659 of
  3,125 test negatives fall out this way before `σ_ratio` is applied at all.
- Never pooled across the dynamics arm (F21); the fbwd−off contrast is the one statistic carrying both
  arms' seed spreads (F17).

## What this does NOT license

- No claim that `flare_ever=0` stars are non-flarers. flatwrm2's low recall still applies **within**
  the searched set; "searched-and-clean" is a strictly better negative than "absent from a catalog",
  not a certified one.
- No change to `labels/variability_labels_star.csv`. The reconstruction lives in a separate
  `labels/qc/` artifact and is joined at probe time (ADR-0011 precedent: schema-touching label changes
  wait for the next label regeneration, not mid-freeze).
- No merging of second-catalog *positives*. One variable at a time.
