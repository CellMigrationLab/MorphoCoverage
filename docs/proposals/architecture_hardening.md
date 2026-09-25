# Proposal: architecture hardening before the strategy zoo

**Status:** all items approved by the project owner (2026-09-23), with
`runtime_seconds` kept as an alias of `compute_seconds`. All seven items are
implemented. Every item changes a public contract
(AGENTS.md section 15).

**As implemented (items 1–3).** `Evidence`/`Quantity` in `evidence.py`;
`Estimator` in `strategies/base.py`; `DecisionPolicy`, `ThresholdPolicy`,
`EffortPolicy`, the per-objective combination and `redecide` in `policies.py`;
`RareTarget` in `objectives.py`; `config_id` in `provenance.py`. Differences
from the text below: the quantity field is `component` (`population` or
`rare`) rather than a free-text `estimates` label, so that policies can be
checked against the objective; availability (`insufficient_data`,
`not_applicable`) replaces `None` scores; and the per-component verdicts are
`StrategyResult.component_decisions` rather than a separate
`ObjectiveDecision` type. `score` and `predicted_remaining` were removed; the
JCB requirement is available as `required_objects` / `required_fovs`.

**As implemented (item 4).** `morphocoverage.benchmark` provides
`BenchmarkRegistry`, `AcquisitionOrder`, `run_benchmark` and `BenchmarkRecord`;
`morphocoverage.outcomes` provides the metrics. They are importable
submodules and are not added to the top-level `__all__`. Differences from the
text below:

- The first two metrics are `RelativeMeanError` (the realized whole-FOV error)
  and `RandomCellRelativeError` (the random-cell comparator from
  `fov_vs_objects.py`). The design effect follows from their ratio across
  permuted orders.
- `RareStateDetected` is added. "Excess FOVs beyond an empirical adequacy
  point" is not implemented yet, because no adequacy definition has been
  agreed.
- `fov_vs_objects.py` (then `fov_vs_cells.py`) was left unchanged. It now
  reads the complete plates stored in `data/crick_idr/`, so its tracked
  results are reproducible without the 2 GB download.

**As implemented (item 5).** `morphocoverage.transforms` provides
`FeatureTransform`, `FittedTransform`, `Standardize`, `PCA` and `Chain`, as a
submodule. Transforms are fitted on a `ValidatedInput`, which is how the
framework delivers the visible prefix. The contract "output at step k is
unchanged when future FOVs are altered" covers every transform on every §9
scenario, and a transform with cross-call state fails it. A transform that
cannot be fitted returns a reason rather than guessing a scale. No estimator
uses a transform yet.

**As implemented (item 6).**
- **Interface.** `Estimator.start(template, *, objective)` returns an `EstimatorState` or `None`. `EstimatorState.update(new)` receives only the newly revealed FOV, and `evaluate_acquisition_prefixes(..., incremental=True)` drives it.
- **Contract.** For every strategy under contract, on every §9 scenario, incremental replay must equal full recomputation to relative 1e-9. `ScalarPrecisionEstimator` is the first incremental estimator; it keeps per-FOV sufficient statistics.
- **Timing.** The `Timing` fields are `recompute_seconds`, `update_seconds`, `initial_fit_seconds`, `framework_overhead_seconds` and `total_decision_seconds`. `compute_seconds` is recompute plus update, and `runtime_seconds` remains its alias.
- **Difference from the text below.** Warm-up updates before `min_fovs` are charged to `initial_fit_seconds`.
- **Measured once on idr0028** (JCB FOV-unit estimator with pilot, one run): median estimator time per decision fell from 2.8 ms to 0.5 ms. Framework overhead, about 2.2 ms, then dominated.

**Origin:** the ChatGPT audit of `main` at `da3953c` (2026-09-23), plus Claude's
earlier review. The small, clear defects from that audit are fixed separately.
This document covers the design items, so that new strategy families (MMD,
kNN, ecological estimators, …) are built on contracts that will not need to
change again.

## 1. Separate estimation from the stopping decision

**Problem.** A strategy currently computes a statistic and applies its own
threshold. That causes several problems:

- comparisons depend on arbitrary tolerances;
- a threshold sweep means re-running the strategy;
- `balanced` sampling cannot be expressed;
- a result can claim an objective its method ignores. For example,
  `ScalarPrecisionStrategy` returns `STOP` under `RARE_DISCOVERY`.

**Proposal.**

```text
Estimator.estimate(data) -> Evidence            # what the visible data say
DecisionPolicy.decide(evidence, objective) -> ObjectiveDecision
```

- **`Evidence`** is a set of named quantities. Each quantity carries
  `name`, `value`, `direction` (`"lower_is_better"` or
  `"higher_is_better"`), `units` and `estimates` (what it is evidence about:
  `"population_precision"`, `"distribution_change"`,
  `"rare_detection_probability"`, …), plus diagnostics.
- **`ObjectiveDecision`** holds one verdict per objective component
  (`population`, `rare`) and an overall verdict:
  - `population` needs a population verdict;
  - `rare_discovery` needs a rare verdict;
  - `balanced` stops only if both components stop.
  
  If a required component has no evidence of a suitable kind, the result is
  `UNSUPPORTED`, never `STOP`.
- **Existing baselines split cleanly:**
  - Fixed FOV and fixed object counts become policies on the framework-owned
    effort quantities (`n_fovs`, `n_objects`). They need no estimator.
  - JCB-style precision becomes an estimator (averaging CV) plus a threshold
    policy.
  - Marcotti-style rules become estimators (descriptor range or spread) plus
    tolerance policies.
- **Replay stores evidence, not decisions.** Any number of policies and
  thresholds can then be applied afterwards without re-running estimators,
  which gives Pareto curves cheaply.
- **Compatibility.** `SamplingStrategy` stays as an estimator-plus-policy pair,
  so the contract suite and existing tests keep working during migration.

## 2. Structured rare-state target

**Problem.** `ObjectiveConfig` has only `rare_prevalence`, so it cannot express
FOV incidence. A 1% state spread over every FOV and a 1% state confined to one
FOV are different acquisition problems.

**Proposal.** A `RareTarget` with:

- at least one of `object_prevalence` or `fov_incidence`;
- `detection_probability`, required;
- `min_fov_occurrences`, a required integer ≥ 1;
- a declared phenotype definition: either an externally labelled column or the
  estimator's resolution parameter. Without it, "a 1% state" is not a testable
  quantity.

`ObjectiveConfig(rare=RareTarget(...))` replaces the three loose fields.

## 3. Configuration identity and score metadata

**Problem.** `scalar_precision_object` at 2.5% and at 5% share a name, and
`score` has no stated direction or units.

**Proposal.**

- Add `config_id`: a stable hash of the class plus its canonical JSON
  parameters. It appears on every estimator, policy and result, and names stay
  human-readable.
- The score metadata (direction, units) comes from item 1's `Evidence`.

## 4. Benchmark protocol, and pluggable evaluation criteria

**Problem.** Nothing yet prevents tuning tolerances, bandwidths or resolutions
on the same real data used to claim success. Evaluation criteria also cannot
be swapped, because no outcome layer exists.

**Proposal.**

- **Dataset roles.** Every dataset is declared `development` or `held_out` in a
  registry. Tuning is allowed only on development data. When data are scarce,
  use leave-one-dataset-out.
- **Canonical benchmark record**, one row per run:
  - dataset and role;
  - group;
  - acquisition-order regime (recorded order, or permutation with its seed);
  - estimator and policy `config_id`s;
  - stop step;
  - effort at stop: FOVs, objects and area;
  - outcome metrics;
  - compute times.
- **Outcome metrics are retrospective functions** of (full acquisition, stop
  prefix) → value. They are strategy-independent, and each one is swappable,
  with the same contract-test treatment as strategies. Examples:
  - relative error of the mean against the full group;
  - distribution distance to the full group;
  - first detection of an independently labelled state;
  - excess FOVs beyond an empirical adequacy point.
- **The first metrics reuse existing, reviewed code.** The design-effect and
  error calculations in `data/crick_idr/fov_vs_objects.py` become the first two.

This is also what AGENTS.md section 10, items 4–5, needs.

## 5. Leakage-safe preprocessing

**Problem.** Multivariate methods need scaling, PCA and similar steps. If each
strategy implements them, both leakage and unfair comparisons become likely.

**Proposal.**

- A small `FeatureTransform` contract, `fit(visible) -> fitted` and
  `fitted.apply(visible)`, fitted on the visible prefix only.
- The fitted parameters are recorded in provenance.
- Transforms are declared as estimator parameters.
- A contract test shows that a transform's output at step k is unchanged when
  future FOVs are altered.

## 6. Incremental interface and honest timing

**Problem.** Replay recomputes every prefix, and `runtime_seconds` covers only
the `evaluate` call. That is documented now, but it is still not smart-microscopy
latency.

**Proposal.**

- An optional `estimator.start(context) -> state` and
  `state.update(new_fov_rows, new_objects) -> Evidence`.
- A shared contract test requires incremental evidence at step k to equal
  full recomputation on prefix k.
- Report `initial_fit_seconds`, `update_seconds`, `framework_overhead_seconds`
  and `total_decision_seconds`, and rename `runtime_seconds` to
  `compute_seconds`. The rename breaks the public API.
- Full recomputation remains the reference implementation.

## 7. JCB multi-sample planning

The current strategy is the paper's single-sample/pilot variant. The full
procedure normalizes to the control mean and pools within-sample SD after
mean-centring each sample. That needs data from several samples, so it belongs
in a separate planning function:

- it is run on declared pilot groups that are disjoint from the evaluated
  groups;
- it produces a required number of cells that is passed to the evaluation as a
  fixed parameter.

This keeps prospective replay leakage-free.

**As implemented (item 7).** `plan_cells_to_average`, `CellCountPlan` and
`PlannedCellCountStrategy` in `strategies/jcb_planning.py`, exported from
`morphocoverage.strategies` and, as approved by the project owner, the
top-level `__all__`.
Disjointness is enforced at evaluation rather than declared up front: the
plan records its pilot samples, and a group containing one reports
`not_applicable`. See `docs/architecture.md`.

## Items from the audit not adopted as proposed

- **Empty FOVs in the FOV-level precision variant.** Counting empty FOVs as
  sampled units is the standard ratio-estimator treatment for cluster
  sampling. Their only effect on the variance is the `k / (k - 1)` factor.
  They also correctly lower the projected `required_fovs` yield per FOV. The
  behaviour is kept, and the estimand and degrees-of-freedom choice are now
  documented. Empirical calibration of the FOV-level bound belongs in the
  benchmark (item 4), as the audit also suggests.
- **A new sample-hierarchy abstraction.** Explicit `fov_col` and `group_cols`
  already express the plate/well case: the real-data tests group by
  `condition`, `repeat` and `well`. `image_file` is only the default. The
  project owner decided that one image file is one FOV, and that multi-position
  containers must be split upstream. Revisit only if a dataset cannot be
  expressed this way.

## Proposed order

1. Items 1–3 together, in one PR: they change the strategy and result
   contracts once.
2. Item 4: benchmark registry, record and outcome metrics.
3. Item 5, before the first multivariate estimator.
4. Item 6, before any smart-microscopy latency claim.
5. Item 7, as an addition to the JCB baseline.
6. Then the strategy families.
