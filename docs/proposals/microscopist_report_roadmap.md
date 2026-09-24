# Roadmap: implementing the microscopist's report

**Status:** for review (2026-09-24). Implements
`docs/proposals/microscopist_report.md`, which gives the reasons. This roadmap
gives the order, the deliverables, the definitions and the acceptance
criteria. Nothing here is implemented yet.

**How to review.** Each milestone lists what it builds, how it is validated,
and what must be decided first. The decisions that need the project owner
(AGENTS.md section 15) are collected in the last section. A milestone is done
only when its acceptance criteria pass, its limits are written down, and its
real-data result is recorded in `docs/preliminary_findings.md`, including a
null or negative result.

## Principles every milestone follows

1. **Retrospective representability is not the prospective sampling
   requirement** (AGENTS.md section 3). No best or minimal subset is reported
   as a recommendation.
2. **What an estimator says is not what the sample achieved.** Planning
   numbers come from strategy-independent outcome metrics on FOVs the
   estimator never saw, reported beside the estimator's decisions.
3. **Every answer is conditional on the representation**, and representation
   checks apply only to declared positive controls.
4. **Protocol recommendations are validated across held-out samples or
   repeats**, never only across FOVs within the samples used to build them.
5. **Existing contracts hold:** FOV is the resampling unit; groups are never
   pooled silently; features are explicit; zero-observation FOVs count as
   effort; random procedures take seeds; no future-FOV leakage in anything
   prospective.

## Data this roadmap needs

| Need | Why | Have now |
|---|---|---|
| Samples with many FOVs (≥ 30, ideally ≥ 60) | reservoirs plus a stream long enough for the curve | idr0028: 12 wells × 28 FOVs (too short for a reservoir plus a long stream) |
| Many samples, several biological repeats | protocol curves and held-out validation across samples | idr0028: 12 wells on 3 plates; idr0139: 4 FOVs per well (too few FOVs) |
| Stage coordinates | spatial redundancy, dispersed orders, stage travel | idr0028: yes |
| A declared positive control | question 1 | idr0028 knockdowns (siYAP, siLATS1, siARAP2) as conditions |
| Candidate positions from an overview | sampling-frame bias (M8) | none |

**Milestone M0 therefore includes finding at least one deeper dataset,** for
example the large tiled images processed FOV by FOV that the project owner
can provide (tiles give many FOVs and stage coordinates). Without it, M1–M3
can be built and tested on idr0028 but their curves stop at about 18 FOVs.

## M0 — Foundations

**Goal:** the shared pieces every later milestone uses.

Deliverables:

- `OutcomeCriterion(metric: OutcomeMetric, tolerance: float)`: "preserved"
  means `metric.evaluate(reference, subset) <= tolerance`, or `>=` for
  higher-is-better metrics such as `RareStateDetected`. It records the
  metric's `config_id` and the tolerance. `None` from the metric means
  undefined, counted separately and never as met.
- `SamplingHierarchy(fov: str, sample: str, biological_repeat: str | None,
  technical_levels: tuple[str, ...] = ())`: a declaration of which columns are
  the FOV, the sample and the independent biological unit. Validation checks
  nesting (each FOV in one sample, each sample in one repeat). Nothing is ever
  inferred from column names.
- A dataset-registration note in `docs/benchmarking.md` for the deeper
  dataset(s), with roles `development` / `held_out` fixed before any curve is
  computed.

Validation: unit tests for tolerance direction, undefined outcomes, and
hierarchy nesting errors (a FOV in two samples, a sample in two repeats).

Acceptance: both types are JSON-serializable provenance; hierarchy validation
raises clear errors on every nesting violation.

## M1 — Realized reliability curves, per sample

**Question:** *if I acquire `k` representative FOVs, how often does the result
preserve what I care about in unseen FOVs, and how often would my stopping rule
have said enough?*

Design (the proposal's fixed-outer-reservoir design):

1. For each sample (analysis group) and each outer split `s = 1..S`: split
   once into a fixed reservoir of whole FOVs or blocks (`ReservoirSplit`) and a
   candidate stream.
2. For each `k` in the declared FOV counts (at most the stream length) and
   each draw `d = 1..D`: draw `k` stream FOVs at random.
3. Record the strategy's decision on those `k` FOVs only (`run_strategies` on
   the subset), and whether each `OutcomeCriterion` is met against the
   reservoir.
4. Aggregate per sample, strategy and `k`.

Deliverable: `morphocoverage.reliability.realized_reliability(objects, fovs, *,
feature_cols, strategies, criteria, fov_counts, reservoir, n_splits, draws,
random_state, fov_col=None, group_cols=None, observation_unit=None) ->
pd.DataFrame`. One row per sample × strategy × `k` × criterion:

| column | meaning |
|---|---|
| `fovs` | `k` |
| `draws` | usable draws over all splits |
| `rule_said_enough` | share of draws where the strategy stopped |
| `outcome_met` | share where the criterion was met on the reservoir |
| `premature_stop` | share where the rule said enough and the outcome was not met |
| `missed_opportunity` | share where the rule said continue and the outcome was already met |
| `outcome_met_split_sd` | spread of `outcome_met` across outer splits |
| `undefined` | draws whose outcome or decision was undefined |
| `reservoir_fovs`, `stream_fovs` | sizes, for the stated limit on `k` |

A secondary mode, `reference="rest"`, scores against the sample's remaining
FOVs and is labelled as a changing-reference cross-validation.

Validation:

- *Contract tests on constructed scenarios:* whole FOVs only (never cells);
  zero-observation FOVs can be drawn; group independence; determinism under a
  seed; the strategy never sees reservoir FOVs (the existing
  `_NoReservoirProbe` pattern); every `k` in one split uses the same reservoir.
- *Known-truth tests:*
  - homogeneous FOVs: `outcome_met` rises with `k` toward 1;
  - a strategy that always stops: `rule_said_enough = 1` and
    `premature_stop = 1 - outcome_met`;
  - a clustered rare state: `premature_stop` for a population rule stays high
    while the rare criterion fails.
- *Real data:* idr0028 with `RelativeMeanError(YAPTAZ ratio)` within the
  JCB-style 5% target and a distribution criterion. Report the curve and its
  spread across splits in preliminary findings.

Acceptance: the known-truth tests pass; the premature-stop rate is exact
arithmetic of the two shares; runtime for idr0028 (12 samples, `S = 10`,
`D = 50`, 6 values of `k`, 3 strategies) is recorded and under 10 minutes.

## M2 — First, sustained and realized adequacy

**Question:** *when would a live acquisition have stopped, when did the
criterion settle, and when were the unseen FOVs actually preserved?*

Definitions (from the proposal, fixed here as the contract):

- **first apparent adequacy** = the first evaluated prefix whose decision is
  stop;
- **sustained apparent adequacy** = one past the last evaluated prefix whose
  decision is continue (undefined if the last prefix continues; "not reached");
- **realized adequacy** = the first prefix whose outcomes against the fixed
  reservoir meet every declared criterion (and "sustained realized" = one past
  the last prefix that does not).

Deliverable: `adequacy_points(...)` over acquisition orders (recorded and
permuted, both on the stream of a fixed reservoir), returning per sample,
strategy and order the three points, and summaries over orders (median,
10–90%). The two apparent points use only decisions; the realized points use
only outcomes.

Validation:

- a constructed sequence stop at 16, stop to 23, continue at 24, stop from 25:
  first = 16, sustained = 25;
- a rule that never continues after its first stop: first = sustained;
- realized adequacy is unchanged when the strategy changes (it must not depend
  on the estimator).

Acceptance: the tests above; `sufficiency_report` shows all three points per
criterion and states which uses decisions and which uses outcomes.

## M3 — Protocol-level reliability across held-out samples

**Question:** *what FOV number should I put in the protocol for experiments
like this one?*

Deliverables:

- `protocol_reliability(per_sample, hierarchy, *, validation="held_out" |
  "leave_one_repeat_out")`: from M1 per-sample outcomes, the share of samples
  whose criterion is met at each `k` (with a per-sample threshold, e.g.
  `outcome_met >= 0.9`), reported with explicit denominators ("11 of 12
  eligible samples; 18 could not be evaluated at this depth") and also
  aggregated per biological repeat.
- Validation across samples: thresholds and settings chosen on development
  samples only; the protocol curve computed on held-out samples, or by leaving
  one repeat out at a time. Samples that need far more FOVs than the rest are
  named.

Validation:

- a constructed set with one very heterogeneous sample: it is named, and the
  curve's denominators at deep `k` exclude short samples explicitly;
- tuning on held-out samples raises (the existing `BenchmarkRegistry`
  guarantee).

Acceptance: denominators always shown; per-repeat aggregation present; the
held-out rule enforced. On idr0028 with 3 plates, the report states that
across-repeat uncertainty is large.

## M4 — Inspect-these-fields report

**Question:** *which FOVs should I go back and look at?*

Deliverable: `inspect_fovs(...)` returning, per flagged FOV, plain reasons with
numbers and a way back to the image: image file, stage position, cell count.
Reasons, each computed independently and only reported when above a declared
threshold:

- only FOV containing a new state (support novelty clusters), with its cells;
- leave-one-FOV-out influence on declared summaries (relative change of the
  mean or chosen quantiles of named features);
- cell count far from the sample median;
- differs strongly from its spatial neighbours (needs stage coordinates).

No combined importance score. The report never labels a FOV as artifact or
biology.

Validation: constructed scenarios where one FOV holds the only rare state, one
drives a tail quantile, and one is dense; each is flagged with the right
reason and no other FOV is. Duplicated cells within a FOV must not make it
more influential than its FOV-level effect.

Acceptance: those tests; the real-data example on idr0028 lists FOVs with
reasons and image files, and at least one is checked by eye by the owner.

## M5 — Positive-control report

**Question:** *could this representation recognize the phenotype I care about
if it appeared?*

Deliverable: `positive_control_report(host, controls, novelty, *, fov_counts,
draws, random_state)`, generalizing `data/crick_idr/positive_control.py`.
A declared control is either whole control FOVs (for example a knockdown well)
or cells planted into host FOVs. It reports per setting the detection rate and
the host false-alarm rate. The text is always of the form "detected the
declared positive-control phenotype in X% of candidate FOVs at a Y%
false-alarm rate"; without a control, "not assessed".

Validation: reproduces the tracked positive-control numbers exactly from the
same seed; a control identical to the host gives detection equal to the
false-alarm rate.

Acceptance: those tests; the report enters the sufficiency report as question
1.

## M6 — Effort projection: more FOVs or more samples?

**Question:** *would another FOV or another independent sample reduce the
uncertainty of my sample-level measurement more?*

Deliverables:

- Variance components of a declared scalar readout (for example a FOV-weighted
  or cell-weighted sample mean) under the `SamplingHierarchy`: cells within
  FOV, FOVs within sample, samples within repeat, repeats. Method: nested
  method-of-moments (ANOVA) estimators for unbalanced designs, with
  confidence intervals from a hierarchical bootstrap that resamples at every
  declared level, using numpy/scipy only.
- Projected standard error of the readout for counterfactual allocations at
  equal total FOV acquisition (for example 6 × 40, 8 × 30, 10 × 24), each with
  an interval from the bootstrap.
- Text only when the intervals separate: "additional samples would reduce the
  sampling uncertainty more than additional FOVs"; otherwise "cannot tell from
  these data".

Validation:

- simulated hierarchies with known components (including zero between-sample
  variance, and FOV variance dominating): components recovered within their
  intervals in about 95% of runs, projections match simulated designs;
- unbalanced designs and zero-observation FOVs handled;
- with 3 repeats, the between-repeat interval is wide and no recommendation
  is made.

Acceptance: those simulations; the idr0028 result recorded, expected to say
"cannot tell" at the repeat level.

Later (not in M6): a user cost model `C = n_samples * C_sample + n_FOV * C_FOV`
to speak of equal budget.

## M7 — Spatial-order efficiency with stage travel

**Question:** *could the same FOVs have been acquired in a better order?*

Deliverables:

- `AcquisitionOrder.dispersed(seed)`: greedy farthest-point order on stage
  coordinates from a random start.
- Stage travel of an order: summed Euclidean path length between successive
  FOVs; acquisition time when per-move and per-FOV times are given.
- A comparison per sample: recorded, random and dispersed orders, each with
  FOVs to first, sustained and realized adequacy (M2) and stage travel.
- Spatial redundancy stated in µm from the semivariogram (the distance below
  which neighbouring FOVs are more alike than the sill suggests), as a
  limitation, not an adequacy claim.

Validation: a constructed gradient sample where dispersed orders reach realized
adequacy with fewer FOVs but more stage travel; a spatially uniform sample
where orders do not differ beyond noise.

Acceptance: those tests; the idr0028 comparison recorded (its 28 FOVs per
well are a small grid, so differences may be small: a null result is
acceptable).

## M8 — Sampling-frame bias

**Question:** *were the imaged locations representative of the specimen?*

Needs a new input: a table of all candidate positions from an overview image,
with covariates such as local density or region. This changes the input
contract (AGENTS.md section 15) and needs upstream support in MorphoBricks
step 1 (CellPatcher). Not started until approved.

## M9 — The report

**Goal:** assemble M1–M7 into the question-first report
(`docs/proposals/microscopist_report.md`), with the plain-language front door
mapping user choices to criteria and outcome tolerances, provisional presets
labelled as such, and every answer stating its criterion, representation and
whether orders were random or recorded. Includes a worked notebook from a
published CellProfiler export to the answers, and a helper that builds the FOV
table from an observation table with a warning that FOVs without cells are
lost.

Acceptance: the notebook runs on idr0028 and on the deeper dataset; every
sentence in the report is traceable to a table column.

## Cross-cutting

- **Contracts:** each new function gets contract tests on the section 9
  scenarios where applicable (whole-FOV reveal, group independence,
  determinism, no leakage); new outcome criteria wrap registered metrics, so
  `METRICS_UNDER_CONTRACT` still covers them.
- **Runtime:** each milestone records its runtime on real data; the reliability
  and adequacy replays are the expensive parts and are parallelized over
  samples only if needed, without new dependencies.
- **Provenance:** every table records strategies' and criteria's
  `config_id`s, the reservoir settings and seeds.
- **Docs:** each milestone updates `docs/benchmarking.md`,
  `docs/architecture.md`, the README question table and the preliminary
  findings.

## Decisions needed from the project owner

| # | Decision | Needed by |
|---|---|---|
| 1 | Which deeper dataset(s) to register, and their development / held-out roles | M0 |
| 2 | The default outcome tolerances offered in presets (for example 5% relative mean error), all labelled provisional | M1 |
| 3 | Whether `realized_reliability`, `adequacy_points`, `protocol_reliability`, `inspect_fovs`, `positive_control_report` and `SamplingHierarchy` join the top-level `__all__` (AGENTS.md section 15) or stay in submodules | M1 onwards |
| 4 | The per-sample threshold that makes a sample "adequately represented" at `k` in the protocol curve (for example `outcome_met >= 0.9`) | M3 |
| 5 | Whether the effort projection may use a new dependency (for example statsmodels mixed models) or stays with numpy/scipy moment estimators and bootstrap (proposed) | M6 |
| 6 | The input contract for candidate positions, and upstream work in CellPatcher | M8 |

## Proposed schedule of reviews

One PR per milestone, each reviewed before the next starts: M0 + M1 together
(the foundation and the headline curve), then M2, M3, M4, M5, M6, M7, and M9 as
the milestones it needs land. After M1 and after M3, an external audit of the
kind already done for modules 1–5, asking specifically for a dataset on which
the curve or the protocol recommendation is misleading while all tests pass
(AGENTS.md section 14).
