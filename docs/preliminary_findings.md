# Preliminary findings

These are exploratory, retrospective observations made while building the
framework. They are not benchmark results and not publication claims. Each
entry states what was run, what it shows, and what it does not show.

## 1. Whole FOVs versus random cells in the JCS and JCB settings (2026-09-23)

**Question.** Both cell-number rules we reproduce treat the cells of a sample
as if they were drawn at random:

- Zweifach (JCB 2024) assumes spatially uniform samples, with averaging noise
  `SD / sqrt(n)`.
- Marcotti et al. (JCS 2026) subsample random cells from the whole well.

A microscope instead acquires whole fields. Does a well mean estimated from
whole FOVs have the error that random-cell reasoning predicts?

**Method** (`data/crick_idr/fov_vs_cells.py`, complete source plates, fixed seed):

- For every well, draw `k` whole FOVs at random, 400 times, and take the
  relative RMSE of their mean against the mean of all imaged fields of that
  well.
- Compare this with the exact expected error of the same number of cells drawn
  at random from the whole well: simple random sampling with finite-population
  correction.
- Nothing is fitted or modelled on the FOV side.
- Metrics are the papers' own: nuclear / (nuclear + cytoplasmic) mean
  intensity of fascin (idr0139, the JCS case study) or YAP/TAZ (idr0028
  screen B).

| dataset | FOVs `k` | cells | error from `k` FOVs | error from random cells | ratio, median [IQR] | wells with ratio > 1.5 |
|---|---|---|---|---|---|---|
| idr0139, 308 wells, 4 FOVs/well | 1 | 329 | 4.2% | 1.3% | 3.2 [2.0–4.2] | 87% |
| | 2 | 657 | 2.4% | 0.74% | 3.2 [2.0–4.3] | 87% |
| | 3 | 987 | 1.3% | 0.42% | 3.1 [2.0–4.3] | 87% |
| idr0028 B, 1,144 wells, 28 FOVs/well | 1 | 33 | 1.8% | 1.2% | 1.5 [1.3–1.7] | 48% |
| | 4 | 132 | 0.82% | 0.54% | 1.5 [1.4–1.7] | 53% |
| | 8 | 264 | 0.52% | 0.34% | 1.5 [1.4–1.7] | 53% |
| | 14 | 462 | 0.33% | 0.22% | 1.5 [1.3–1.7] | 53% |

Errors are relative RMSE of the well mean, as medians across wells. Every number in this entry is taken from the tracked outputs `data/crick_idr/results/fov_vs_cells_idr0139_fascin.csv` and `fov_vs_cells_idr0028_screenB_yaptaz.csv`, generated from Crick commit `a81d20c` with checksum-verified inputs.

**What this shows**

- **The random-cell assumption understates sampling error in both datasets.**
  The factor is about 3 (roughly 10× in variance) in the JCS paper's own data,
  and about 1.5 (roughly 2.3× in variance) in idr0028.
- **The factor does not shrink as more FOVs are acquired.** That is
  consistent with substantial FOV-level structure rather than a small-sample
  effect.
- **It is metric- and dataset-dependent.** An earlier exploratory run on
  idr0028 plate 1A (negative-control wells, mean nuclear YAP/TAZ intensity)
  gave a variance inflation of about 4.7, against about 2.3 for the ratio. So
  "keep FOVs" is not a constant correction factor; it has to be measured.
- **Consequences for the JCB criterion on idr0139** (single-sample form of
  the criterion, from the Table 1 statistics).
  - The formula asks for 32 cells at a 5% target and 127 at 2.5%, using
    between-cell CV = 0.131 / 0.465 from Table 1. A single FOV (~330 cells)
    satisfies both.
  - If those cells were random draws, only 1% of wells would miss 2.5%, and
    none would miss 5%.
  - With the actual single FOV, **75% of wells miss 2.5% and 35% miss 5%**.
    With 2 FOVs, 46% miss 2.5% and 3% miss 5%.
  - In the JCB paper's own terms, the 5% target (independent samples) is
    roughly reached at 1–2 FOVs. The 2.5% target recommended for paired
    designs is not reached until about 3 FOVs.
- **Consequences for the ~250–300-cell threshold observed in the JCS case
  study.** The paper presents this as an observation for its example, not a
  universal recommendation, and itself calls the stopping point "somewhat
  arbitrary". On idr0139 it is less than one FOV. Their random-cell
  subsampling reproduces the random-cell column above by construction, so it
  cannot reveal the FOV-level error.
- **Their biological conclusions still hold.** The leptomycin B effect (about
  0.10 in `Fascin_Ratio`) is several times larger than FOV-level sampling
  error. This is consistent with the JCS paper's own point that large effects
  are robust to sample size.
- **Regime-map reading, preliminary:**
  - cell-number guidance is adequate for large effects and for a 5% averaging
    target when a few FOVs are imaged anyway;
  - it is misleading when many cells come from one or two FOVs, and for
    tighter targets, because cell count then overstates the independent
    information acquired.

**What this does not show**

- **Not a benchmark.** It says nothing about whether any FOV-aware
  multidimensional strategy stops better than the baselines; that is the
  benchmark question.
- **FOVs are not shown to be independent.** Adjacent fields may be correlated,
  so whole-FOV resampling can itself be optimistic (AGENTS.md §3).
- **The source is unknown.** Between-FOV variation may be technical
  (illumination, focus, local density, edge effects) or biological. It is not
  decomposed here. Either way it enters every acquisition.
- **Finite references.** The reference is the mean of the imaged fields,
  not the whole well. idr0139 has only 4 fields per well, so its `k`-FOV errors
  are measured against a 4-field reference, and the ratio for `k = 3` is
  especially constrained.
- **Limited scope.** One metric per dataset, and one cell line each.
- **One unexplained number.** The JCS Table 1 SEM column (0.013, 0.017, 0.024,
  0.013) matches neither `SD / sqrt(n)` (0.0035–0.0048) nor a 4-FOV standard
  error (0.0034–0.0176). It was not investigated further.

The data used are described in `data/crick_idr/README.md`. The complete plates
are downloaded by the script; only the paper's wells are committed.
