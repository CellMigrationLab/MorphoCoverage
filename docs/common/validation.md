# Validation protocol: the benchmark that chooses the rules

This page fixes, before any confirmatory run, how each rule of MorphoCoverage is chosen
and validated: the purpose, the principles, the selection rule, the designs, and the
modules P0-P11, each in the ADEMP form of Morris, White and Crowther (2019): **A**im,
**D**ata-generating mechanisms, **E**stimands, **M**ethods, **P**erformance measures. It
holds no results. It is fixed before the confirmatory runs, to decide empirically what
MorphoCoverage implements; it is not a registered study (owner, 2026-10-13). Any change
after it was fixed is recorded under [Deviations](#deviations) with its reason and date
(plan v2, adopted by the owner, 2026-10-06).

The code is in [`data/simulations/decisions/`](../../data/simulations/decisions/README.md):
one design skeleton, separate outcome generators, one runner, one script per module, the
candidate methods the package does not hold yet, and the notebook the owner runs the
benchmark with. P0 is a test file, [`tests/test_validation_p0.py`](../../tests/test_validation_p0.py).

## Purpose

Choose the best strategy for each rule of the toolbox by a fair benchmark fixed in advance,
change the tool to the winner, and report it in the supplementary information (SI). Three
kinds of evidence, kept apart:

1. **Proofs and deterministic tests** (P0): estimability, rotation invariance, leakage. No
   simulation.
2. **Confirmatory simulations** (P1-P5 first, then P6-P11): they choose what ships.
3. **Stress tests**: they define limitations, with few runs, and never choose.

## Principles

1. The biological repeat is the unit of replication; claims are contrasts within repeats.
2. Normalisation is for seeing; contrasts are for claims.
3. Contrasts in three regimes: **directly paired** (the repeats hold every condition of
   the contrast), else **model-estimated** (the fixed-block model on per-repeat condition
   summaries, `y_rc = mu + alpha_c + beta_r + e_rc`), else **not estimable**, said with its
   reason. Estimability is decided by algebra (the contrast in the row space of the design
   matrix, by SVD with a fixed tolerance); the connectivity graph of conditions and
   repeats gives the explanation.
4. Every embedding analysis that depends on geometry only is invariant to a **global**
   orthogonal change of basis (a hard gate). This rules out coordinatewise centring,
   scaling and tests; it permits centroid or geometric-median centring, global scalar
   scaling, L2 normalisation, Euclidean or cosine distances, PCA subspaces, equivariant
   whitening and singular-value block scaling. No verdict depends on UMAP coordinates;
   HDBSCAN reads the embedding's Euclidean geometry for inference, never UMAP coordinates.
   A batch-specific rotation (each repeat rotated its own way) is a real distortion, a
   scenario, not the invariance test.
5. Sufficiency is target-specific: a mean; a rare state (prevalence and occupancy kept
   apart); a covariance fingerprint.
6. The false discovery rate is controlled within each declared family, and the report says
   so.
7. The default estimand is the **cell-weighted repeat mean** (owner, 2026-10-06);
   field-balanced and well-balanced means are declared options. The scale is declared per
   feature, today's automatic rule (log ratio when every value is positive, else
   difference) the default.

## Selection rule

Fixed here, before the confirmatory runs, for every module; implemented in
[`summary.py`](../../data/simulations/decisions/summary.py), which writes each module's
`summary.csv` and `summary.md`.

**Hard gates.** A method failing any one is ineligible:

- the correct estimand (the declared one; a comparator estimating another, such as a
  mixed model on log values, is reported and never selected);
- finite and converged: at most 1% of the core replicates failed (an exception, or a
  missing or non-finite primary output);
- rotation invariance for embedding methods (P3: the same p-value, or the same centred
  values rotated, after a random global rotation, on the first 5 replicates of every
  point);
- no prohibited information and no leakage (P0);
- correct detection of what is not estimable (P2).

**Calibration gate**, on the core null designs (grid A): 10,000 replicates each; for each
method, the one-sided 95% upper confidence bound of its rejection rate at nominal 0.05,
simultaneous over the `K` gated designs (Clopper-Pearson at level `1 - 0.05 / K`), at most
**0.060** on every one of them. No lower bound: a conservative method is eligible and pays
in power. FDR procedures (P5): the same on partial nulls (0, 20, 50 and 80% non-null),
the bound on the mean false discovery proportion, normal, simultaneous over the designs.
A sequential rule (P4): the same on its false-stop rate. The Monte Carlo SE of a rate `p`
over `R` replicates is `sqrt(p (1 - p) / R)`: 0.22 points at 5% with 10,000.

**Coverage gate**, for methods giving 95% intervals: the simultaneous lower bound of the
coverage on every core design at least **0.93**; the mean length is reported.

**Power.** Each module declares its alternative classes; each class weighs the same (its
designs averaged first), whatever the number of designs in it. Among the eligible methods:

1. a method is dropped when another is at least as powerful in every class and more
   powerful by more than the tie margin in one (Pareto);
2. the objective is the mean power over the classes;
3. methods whose objective is within **2 points**, or within **2 Monte Carlo SE of the
   paired difference** (the same simulated datasets for every method, the difference
   computed per replicate), of the best are tied;
4. ties are broken by simplicity (a rank declared with each method in its script), then
   runtime, then interpretability.

In P4 the objective is the mean number of fields at the stop (lower is better), the tie
margin 1 field; in P3's centring question, the error of the biology after centring (lower
is better).

**Paired comparison.** Every method runs on the same simulated datasets: replicate `i` of
design `d` of module `m` is simulated from `SeedSequence([seed, crc32(m), crc32(d), i])`,
and the failure rate and runtime of every method are recorded.

**Stress designs** (grid C) are reported, never gated. "Not assessed" is said only for
designs the tool can detect from the data (two repeats, a disconnected design), never for
an assumption it cannot observe (heavy tails, a well batch it was not told of).

**Replicates.** Core null designs 10,000; core alternatives 1,000; stress 1,000; screening
250 per point. A comparator that is slow and never selected (a mixed model) runs on the
first 1,000 replicates of each design only.

**Workflow.** Development simulations (seed 0: smoke and pilot runs) → methods and tuning
frozen → this protocol committed → confirmatory runs on a new seed, **20261006**, fixed
here. The pilot (1% of the replicates) checks timing, memory, convergence and the
generators only; it is never read as a result.

**What ships.** The winner of each question ships; the owner verifies the implementation,
not the choice. When no method is eligible, none ships as validated for that design: the
report states the limitation, and the least miscalibrated method may be kept only as a
recorded deviation. Any departure from this rule is a documented deviation.

## Grid

**A. Reference (core) designs.** The calibration gate reads these. Around a centre, one
factor at a time, at 3, 4 and 6 biological repeats: wells per condition and repeat 1 or 2;
about 5, 15 or 30 fields per well; 5, 20 or 80 cells per field (low, moderate, high);
field variance share 0.02, 0.1 or 0.3 (low, moderate, high field ICC); balanced or
moderately unbalanced (field counts with a CV of 0.5, overdispersed cell counts); nearly
Gaussian (lognormal, CV 0.3) or skewed (CV 0.8). The centre: 4 repeats, one well, 15
fields, 20 cells, field share 0.1, well share 0.05, repeat share 0.2, balanced, CV 0.3.

**B. Screening.** 32 points per module on a seeded Latin hypercube (McKay, Beckman and
Conover 1979) over the nuisance factors, 250 runs each: to find interactions and
boundaries, never to select.

**C. Stress.** An extreme repeat; a faulty well; heavy tails; informative cluster size;
near-zero denominators; unrecorded wells; MNAR loss; severe imbalance; a confounded
acquisition batch; condition heteroscedasticity; phenotype-dependent segmentation failure;
QC exclusion correlated with condition; plate or session effects; a shared control with few
wells; a treatment changing the cell number; near-duplicate features; values exactly at
thresholds; empty fields omitted from the tables. Each module lists the ones it runs.

**Real-data anchoring.** The ranges above are checked against summary statistics of real
tables: observations per field, the share of empty fields, fields per well, the well,
field and residual variance shares, CV, skewness, density differences between conditions,
correlation spectra, and for embeddings the effective rank, anisotropy and repeat centroid
shifts. They come from the lab's datasets and idr0028 (public), computed by
[`real_parameters.py`](../../data/simulations/decisions/real_parameters.py) into
`data/simulations/decisions/results/real_parameters.csv`, and from the owner's ECM tables,
kept outside the repository (the script reads a path and writes summaries only). idr0139 is
held out and is not read. Ranges come from medians and quantiles; the provenance table
below records them. A **semi-synthetic** layer uses real control observations with an
injected effect: P3 adds it when `DECISIONS_REAL_EMBEDDING` names a table of real control
embeddings (half the fields of each repeat relabelled and shifted).

**Simulator.** One design skeleton
([`design.py`](../../data/simulations/decisions/design.py): repeats, conditions, wells
recorded or not, fields, cells, informative cluster size, empty fields recorded or not,
MCAR, MAR, MNAR and condition-related missingness, connected and disconnected designs,
imaging sessions and acquisition order) and separate outcome generators: scalar features
(`scalar.py`, with the truth of every estimand computed by quadrature), embeddings
(`embedding.py`, global rotation versus batch-specific rotation), populations
(`populations.py`) and sequential acquisition (`sequential.py`).

## Modules

| | Module | Cost | Status |
|---|---|---|---|
| P0 | Analytical and design validation | low | tests, never dropped |
| P1 | Core repeat inference | medium | confirmatory, minimum SI |
| P2 | Incomplete and complex contrasts | low-medium | confirmatory, minimum SI |
| P3 | Whole-embedding inference | high | confirmatory, minimum SI |
| P4 | Sequential mean sufficiency | high | confirmatory, minimum SI |
| P5 | Multiple testing | low | confirmatory, minimum SI |
| P6 | Rare sufficiency | medium | later; exploratory until run |
| P7 | Populations | high | later; exploratory until run |
| P8 | Covariance fingerprint | high | later; exploratory until run |
| P9 | Beyond the mean | medium-high | later; exploratory until run |
| P10 | Novelty and combined spaces | medium | later; exploratory until run |
| P11 | Separation score | low | later |

"Ours" below is the package's rule today; a candidate the package does not hold is in
`data/simulations/decisions/candidates/`, never wired into the reports.

### P0. Analytical and design validation

- **Aim**: prove what algebra can prove, so no simulation has to.
- **Data**: constructed inputs (random designs; an anisotropic embedding; small tables).
- **Estimands**: none; properties.
- **Methods**: (1) estimability of the candidate contrast layer on complete, connected
  incomplete, disconnected and parent-chain designs, a contrast without a control, and 40
  random designs, the algebraic answer checked against the graph's; the model's estimate
  equal to the mean paired difference in a complete design. (2) Rotation equivariance of
  every embedding operation the package applies: the robust scaling, the centring of the
  tables, the whole-table test, the populations' distance space, the separation score and
  the unusual-observation score (expected to fail: they read the coordinates one at a
  time), and the projection test, the shuffle test, the populations' purity, the
  neighbours of the separation score, the MFA weights and the PCA (expected to hold). A
  failure is recorded as `xfail(strict=True)` with its reason, so a fix shows as an
  unexpected pass and must remove the mark. (3) Leakage: the unusual-observation
  threshold fitted on the reference alone; the population rule's held-out check never
  fitted on the rows it holds out; a sequential rule never reading a field after its stop;
  the package never importing the benchmark's extra. The covariance fingerprint's
  training-only fit has its own regression test with that change (plan v2, Phase 1).
- **Performance**: pass or fail; the expected failures are P0's findings for P3.

### P1. Core repeat inference

- **Aim**: the test and scale of the per-feature verdict ("significant across repeats")
  for the default estimand, with the field- and well-balanced estimands reported beside it.
- **Data** (`p1_repeat_inference.py`): scalar features, two conditions. Core: grid A at 3,
  4 and 6 repeats (30 null designs); alternatives at each number of repeats. Stress: one
  extreme repeat, a faulty well, t3 tails, a gamma with CV 1, informative cluster size,
  near-zero controls, background-subtracted values with negatives, zero-inflated counts,
  bounded values, saturation, unrecorded wells, MNAR loss, severe imbalance, a session
  confounded with the treatment (a 2-fold gap), heteroscedasticity, phenotype-dependent
  segmentation loss, QC exclusion of treated fields, plate effects, a treatment halving
  the cell number, empty fields omitted. Screening over repeats, wells, fields, cells,
  field and well shares, imbalance and CV. A null family of 20 features at 3, 4 and 6
  repeats, for the moderated t (amendment of 2026-10-06).
- **Estimands**: the log ratio (or difference) of the condition's to the control's
  cell-weighted mean; the field-balanced and well-balanced versions as their own
  questions. The truth is computed, not simulated (quadrature, with informative cluster
  size tilting the cell-weighted mean).
- **Methods**: ours, the paired t-test on the repeat means on the automatic scale (Lord et
  al. 2020); the same on a declared scale; always on differences; the fixed-block model
  (equal to the paired test in a complete two-condition design: a check of the engine);
  the exact sign-flip test (reference); the moderated paired t across the dataset's
  features (limma-style empirical Bayes, Smyth 2004; amendment of 2026-10-06, see
  [Deviations](#deviations)). Comparators, never selected: a random-intercept
  mixed model on field means (Laird and Ware 1982; statsmodels) and Welch's t on the
  cells pooled (the reference that ignores the hierarchy).
- **Performance**: type I error (the gate), 95% interval coverage (the gate) and length,
  bias, RMSE, power, failure rate, runtime.
- **Alternative classes**: a multiplicative effect (0.5 and 1 SD), an additive effect
  (0.5 and 1 SD, normal values), an effect varying between repeats (1 SD, spread 0.5 SD),
  an unbalanced design (0.75 SD); a family of 20 features, a fifth with a 1 SD
  multiplicative effect (amendment of 2026-10-06).

### P2. Incomplete and complex contrasts

- **Aim**: how a requested contrast is computed when its conditions do not share every
  repeat: directly paired, model-estimated, or not estimable.
- **Data** (`p2_contrasts.py`): scalar features (lognormal, multiplicative effects) on
  per-repeat condition summaries (log cell-weighted means), at 4 and 6 repeats. Core: a
  complete three-condition design; a connected incomplete design (each repeat the control
  and two of three conditions); a control missing from one repeat; a parent chain (P with C
  in half the repeats, C with T in the others: T against P only through C); a matched
  four-arm interaction; the same interaction from two separate experiment sets; and a
  disconnected design whose target is not estimable. Stress: heteroscedasticity,
  heterogeneous effects, an extreme repeat, QC exclusion of treated fields, a weak bridge
  (one repeat linking two sets).
- **Estimands**: the contrast `sum_c w_c alpha_c` on the log scale (a pairwise log ratio,
  a chain, an interaction).
- **Methods**: ours, the direct paired t from 3 repeats holding every condition of the
  contrast (a contrast it cannot answer is not rejected); the fixed-block model wherever
  the contrast is estimable; the three-regime layer (direct, else model, else not
  estimable). Comparator: a random repeat intercept (statsmodels).
- **Performance**: type I error, coverage, bias, power, the share of contrasts answered;
  hard gates: the algebra agrees with the graph on every checked contrast, and a target
  that is not estimable is said so.
- **Alternative classes**: complete; incomplete (connected, missing control); chain;
  interaction (matched and separate sets).

### P3. Whole-embedding inference

- **Aim**: after the invariance gate, the whole-table test of a mean shift, separately the
  test of a change in distribution, and the centring of embedding tables per repeat.
- **Data** (`p3_embedding.py`): embeddings of 32 or 128 columns (spectrum `j^-0.7` in a
  random basis), an additive repeat batch, 20 cells per field or one image per field (6
  fields), 3, 4 and 6 repeats (12 core null designs). Semi-synthetic real control
  embeddings when given. Stress: rotation, scale and nonlinear batches of twice the size,
  unrecorded wells, a low-rank table (rank 5 of 128), 384 columns, unbalanced fields.
- **Estimands**: whether the condition's mean moves from the control's the same way in
  every repeat (mean shift); whether its distribution differs (distributional); for
  centring, the condition's pooled shift after centring against the true shift (in
  within-group SDs) and the spread left between repeats.
- **Methods**, mean shift: ours, the package's whole-table repeat verdict (leave-one-repeat-
  out projections, the paired t corrected by `sqrt((n - 2) / (2 (n - 1)))`, columns scaled
  per comparison); the same projection test on unscaled field means; the paired
  high-dimensional inner-product U-statistic of the Chen and Qin (2010) family with a
  normal p and with an exact sign-flip p; Hotelling's T² on the first `n - 2` principal
  components (Hotelling 1931); a shrinkage Hotelling with the Ledoit and Wolf (2004)
  covariance and sign-flip p; the per-coordinate minimum p (Bonferroni), a non-invariant
  negative comparator. Distributional: PERMANOVA with permutations restricted within
  repeats (Anderson 2001), the energy distance (Székely and Rizzo 2013) and the MMD with a
  Gaussian kernel (Gretton et al. 2012), summed over repeats. Centring: ours, the median of
  the per-condition coordinatewise medians; the equal-weighted mean of the shared
  conditions' centroids; their geometric median (Weiszfeld 1937); the additive vector model
  `y_rc = alpha_c + b_r` on condition centroids; location-scale per coordinate (ComBat
  without its empirical Bayes step, Johnson, Li and Rabinovic 2007) and Harmony (Korsunsky
  et al. 2019) as limited references.
- **Performance**: rotation invariance (hard gate), type I error, power; for centring,
  the error of the biology and the remaining batch.
- **Alternative classes**: mean: a dense shift (0.3 and 0.6 within-group SD), a sparse
  shift (3 coordinates), a shift along the batch direction. Distributional: those and a
  covariance-only change (a null for the mean question). Centring: additive, scale,
  rotation and nonlinear batches, and an incomplete design.

### P4. Sequential mean sufficiency

- **Aim**: the interval behind "enough fields for the mean", judged as the full sequential
  process the microscope runs, not as a fixed-sample interval.
- **Data** (`p4_sequential.py`): one sample (one condition in one repeat), up to 60 fields
  a well, looks after 5 fields and every 5 after; the centre and one factor at a time (two
  wells, 5 or 80 cells, field share 0.02 or 0.3, overdispersed counts, CV 0.8). Stress:
  unrecorded wells, autocorrelated fields along the order (0.6), informative cluster size,
  MNAR loss, empty fields recorded or omitted, t3 tails, wells imaged one after the other.
- **Estimand**: the sample's cell-weighted mean over the wells that could have been
  sampled, not only the wells imaged (a deviation of 2026-10-13); the stop asks for a
  relative precision of 5%.
- **Methods**: ours, the package's bootstrap of wells then fields, widened for the units
  resampled (Ren et al. 2010); the ratio-estimator SE with a t quantile; a two-stage
  bootstrap; a moving-block bootstrap along the acquisition order (Künsch 1989); the
  ratio-estimator interval at `1 - 0.05 / K` for `K` looks (Bonferroni over looks);
  Lan-DeMets O'Brien-Fleming spending (Lan and DeMets 1983; O'Brien and Fleming 1979);
  a normal-mixture confidence sequence (Howard et al. 2021).
- **Performance**: the false-stop rate (a stop whose estimate misses the truth by more
  than the tolerance; the gate), the coverage at the stop, the expected fields at the stop
  (the objective), the share never stopping.
- **Classes**: each core design is its own class.

### P5. Multiple testing

- **Aim**: the correction family of the per-feature verdicts.
- **Data** (`p5_multiple_testing.py`): the per-feature paired t simulated at the level of
  repeat log ratios (P1 fixes the test): 3 conditions against one shared control, 10 or 100
  features over 3 tables, 4 repeats, 0, 20, 50 or 80% non-null, features independent or
  equicorrelated 0.5; two more families of null tests stand for the report's other
  families. Stress: 30 conditions, 1,000 features, 3 repeats, near-duplicate features
  (correlation 0.9), a noisy shared control, heterogeneous effects, 10 tables.
- **Estimand**: the false discovery rate within the family, and the false discoveries
  across the report's families.
- **Methods**: ours, one Benjamini-Hochberg family over every explainable feature and
  condition of the sub-experiment (Benjamini and Hochberg 1995); BH within each table;
  Benjamini-Yekutieli (2001) as a reference; weighted BH with weights fixed in advance (the
  readout 4) (Genovese, Roeder and Wasserman 2006); hierarchical selection over tables
  (Benjamini and Bogomolov 2014).
- **Performance**: FDR (the gate), power (the share of non-null tests found; the
  objective), family-wise error, discoveries, false discoveries across the report.
- **Alternative classes**: 20, 50 and 80% non-null.

### P6-P11 (later)

Each will be written in the same form, and this page amended as a recorded deviation,
before its confirmatory run. Planned content:

- **P6, rare sufficiency**: prevalence and occupancy as separate targets; binomial versus
  beta-binomial (Williams 1975) versus hierarchical logistic; prevalence 0.1-5%, clustered
  or uniform over fields.
- **P7, populations**: HDBSCAN noise (Campello, Moulavi and Sander 2013), balanced
  sampling, stability across biological repeats; enrichment by the paired t versus
  beta-binomial versus Dirichlet-multinomial (Mosimann 1962) or a global composition
  test; generator `populations.py` (none, clear, weak, rare, split, batch-only groups).
- **P8, covariance fingerprint**: training-only PCA, a field-aware k, shrinkage
  covariance, validation grouped by fields and wells.
- **P9, beyond the mean**: the claim defined first (variance, named quantiles), else it
  stays exploratory.
- **P10, novelty and combined spaces**: an empirical threshold from held-out control
  fields; MFA (Escofier and Pagès 1994) as a sensitivity; the MAD floor, including whether
  novelty's `RobustStandardize` takes the geometry rule of `statistics.robust_sd` (it keeps
  its zero rule until then, owner, 2026-10-13); the unusualness threshold.
- **P11, separation score**: chance from a permutation that keeps the hierarchy.

## Minimum SI and exploratory labels

The minimum SI is P0-P5. Until their module validates them, these are labelled
"exploratory" in the report, the page, the docs and the SI: populations (P7), beyond the
mean (P9), the covariance fingerprint (P8), the rare target (P6) and novelty (P10).

## Provenance of the design ranges

To be filled from `data/simulations/decisions/results/real_parameters.csv` (lab, idr0028)
and the ECM summaries (outside the repository) before the confirmatory runs; a range
changed after this commit is a deviation.

| Parameter | Grid value(s) | Lab datasets | idr0028 | ECM | Notes |
|---|---|---|---|---|---|
| Observations per field | 5, 20, 80 | | | | |
| Share of empty fields | 0 (core), 0.2 (stress) | | | | |
| Fields per well | 5, 15, 30 (P4: up to 60) | | | | |
| Wells per condition and repeat | 1, 2 | | | | |
| Repeat variance share | 0.2 | | | | |
| Well variance share | 0.05 (0.15 stress) | | | | |
| Field variance share | 0.02, 0.1, 0.3 | | | | |
| CV | 0.3, 0.8 | | | | |
| Skewness, excess kurtosis | lognormal, gamma, t3 | | | | |
| Density against the control | 1 (0.5 stress) | | | | |
| Correlation effective rank (explainable) | independent, 0.5, 0.9 | | | | |
| Embedding dimensions, effective rank | 32, 128 (384, rank 5 stress) | | | | |
| Embedding anisotropy | spectrum `j^-0.7` | | | | |
| Repeat centroid shift (within-group SDs) | additive batch, size 1 | | | | |

## How the benchmark is run

The owner runs it on Google Colab with
[`benchmark_colab.ipynb`](../../data/simulations/decisions/benchmark_colab.ipynb): get the
code (the repository is private: a token in Colab's secrets, or an uploaded zip), mount
Google Drive and choose the output folder, choose the modules (P0 and P1-P5) and the size
(smoke, pilot or full), run (a progress line after each chunk, with the time left), then
summarise and zip the results. Each module script also runs from a terminal:

    PYTHONPATH=src:data/simulations python -m decisions.p1_repeat_inference \
        --out results --size full --workers 8 --chunk 100
    PYTHONPATH=src:data/simulations python -m decisions.p1_repeat_inference summarise --out results

A run resumes from the chunks already in `--out`, so a run cut by a timeout loses at most
the chunks in progress. What is sent back is each module's `summary.md` and `summary.csv`
(with `selection_<question>.csv` and `points.csv`), and P0's `p0.txt`.

## Deviations

- **2026-10-06, amendment to P1, before any confirmatory run** (owner, audit 10). A verdict
  on 3 biological repeats rested on three nearly identical log ratios (SD 0.0018 against a
  median of 0.18 over its family): the paired t-test divides by the repeats' own SD, so
  repeats that agree by chance give a tiny p. P1 adds a candidate, the moderated paired t
  across a family of features (limma-style empirical Bayes, Smyth 2004:
  `candidates/repeat_means.moderated_t`), and the design points it needs: a null family of
  20 features at 3, 4 and 6 repeats (`core_null_n*_family`) and the alternative class
  `family` (20 features, a fifth with a 1 SD multiplicative effect). Until P1 selects, the
  report keeps the paired t and says when a verdict "rests on 3 near-identical repeats".
- **2026-10-13, not a registered study** (owner). The protocol is fixed before its
  confirmatory runs to decide empirically what MorphoCoverage implements. It is not
  deposited, and the documentation and the paper do not call it a pre-registration. The
  confirmatory seed, the selection rule and the order of work stay as written.
- **2026-10-13, P4's estimand, before any P4 run** (owner). The estimand was the sample's
  mean "given its wells", the wells imaged. It becomes the mean over the wells that could
  have been sampled, the quantity the package's bootstrap of wells then fields targets.
  `sequential.Sequence.truth` changes to it before P4's pilot (roadmap BENCH-1).

## References

- Anderson MJ (2001). A new method for non-parametric multivariate analysis of variance.
  *Austral Ecology* 26:32-46.
- Benjamini Y, Bogomolov M (2014). Selective inference on multiple families of hypotheses.
  *J R Stat Soc B* 76:297-318.
- Benjamini Y, Hochberg Y (1995). Controlling the false discovery rate. *J R Stat Soc B*
  57:289-300.
- Benjamini Y, Yekutieli D (2001). The control of the false discovery rate in multiple
  testing under dependency. *Ann Stat* 29:1165-1188.
- Campello RJGB, Moulavi D, Sander J (2013). Density-based clustering based on
  hierarchical density estimates. *PAKDD*.
- Chen SX, Qin YL (2010). A two-sample test for high-dimensional data with applications to
  gene-set testing. *Ann Stat* 38:808-835.
- Clopper CJ, Pearson ES (1934). The use of confidence or fiducial limits illustrated in
  the case of the binomial. *Biometrika* 26:404-413.
- Escofier B, Pagès J (1994). Multiple factor analysis. *Comput Stat Data Anal*
  18:121-140.
- Genovese CR, Roeder K, Wasserman L (2006). False discovery control with p-value
  weighting. *Biometrika* 93:509-524.
- Gretton A, Borgwardt KM, Rasch MJ, Schölkopf B, Smola A (2012). A kernel two-sample
  test. *J Mach Learn Res* 13:723-773.
- Hotelling H (1931). The generalization of Student's ratio. *Ann Math Stat* 2:360-378.
- Howard SR, Ramdas A, McAuliffe J, Sekhon J (2021). Time-uniform, nonparametric,
  nonasymptotic confidence sequences. *Ann Stat* 49:1055-1080.
- Johnson WE, Li C, Rabinovic A (2007). Adjusting batch effects in microarray expression
  data using empirical Bayes methods. *Biostatistics* 8:118-127.
- Korsunsky I et al. (2019). Fast, sensitive and accurate integration of single-cell data
  with Harmony. *Nat Methods* 16:1289-1296.
- Künsch HR (1989). The jackknife and the bootstrap for general stationary observations.
  *Ann Stat* 17:1217-1241.
- Laird NM, Ware JH (1982). Random-effects models for longitudinal data. *Biometrics*
  38:963-974.
- Lan KKG, DeMets DL (1983). Discrete sequential boundaries for clinical trials.
  *Biometrika* 70:659-663.
- Ledoit O, Wolf M (2004). A well-conditioned estimator for large-dimensional covariance
  matrices. *J Multivar Anal* 88:365-411.
- Lord SJ, Velle KB, Mullins RD, Fritz-Laylin LK (2020). SuperPlots: communicating
  reproducibility and variability in cell biology. *J Cell Biol* 219:e202001064.
- McKay MD, Beckman RJ, Conover WJ (1979). A comparison of three methods for selecting
  values of input variables in the analysis of output from a computer code.
  *Technometrics* 21:239-245.
- Morris TP, White IR, Crowther MJ (2019). Using simulation studies to evaluate statistical
  methods. *Stat Med* 38:2074-2102.
- Mosimann JE (1962). On the compound multinomial distribution. *Biometrika* 49:65-82.
- O'Brien PC, Fleming TR (1979). A multiple testing procedure for clinical trials.
  *Biometrics* 35:549-556.
- Ren S, Lai H, Tong W, Aminzadeh M, Hou X, Lai S (2010). Nonparametric bootstrapping for
  hierarchical data. *J Appl Stat* 37:1487-1498.
- Smyth GK (2004). Linear models and empirical Bayes methods for assessing differential
  expression in microarray experiments. *Stat Appl Genet Mol Biol* 3:Article 3.
- Székely GJ, Rizzo ML (2013). Energy statistics: a class of statistics based on
  distances. *J Stat Plan Inference* 143:1249-1272.
- Weiszfeld E (1937). Sur le point pour lequel la somme des distances de n points donnés
  est minimum. *Tohoku Math J* 43:355-386.
- Williams DA (1975). The analysis of binary responses from toxicological experiments
  involving reproduction and teratogenicity. *Biometrics* 31:949-952.
