# Modular architecture

MorphoCoverage is designed as a strategy-comparison framework rather than a single estimator.

## Core flow

```text
MorphoBricks-compatible feature tables
        |
        v
metadata/FOV resolution
        |
        v
condition x repeat grouping
        |
        v
ValidatedInput
        |
        +--> Estimator A --> Evidence --+
        +--> Estimator B --> Evidence --+--> decision policies --> decision per component
        |                               |
        v                               v
canonical StrategyResult (evidence, component decisions, effort, provenance, config_id)
        |
        v
real-dataset sequential benchmark
```

The shared runner owns validation, MorphoBricks metadata handling, grouping, timing, provenance and the final decision. Estimators own only their mathematical logic, and policies own only the decision rule.

## MorphoBricks defaults

The framework follows the shared MorphoBricks metadata conventions:

- `condition`
- `repeat`
- `position_id` (when file layout is not one-file-per-position)
- `image_file`
- `mask_file`
- `patch_file`

When the caller does not provide an FOV identifier, `position_id` is preferred, followed by `image_file` when one file equals one position, then `fov_id`. One FOV is one independently selected spatial position; z/c/t observations from that position remain nested within it. A candidate column present in only one of the linked tables is an error, never a reason to fall back to the next candidate.

When the caller does not provide grouping columns, `condition` and `repeat` are used automatically. This keeps biological/sample groups independent without requiring every strategy to understand those columns. A canonical column present in only one linked table raises instead of silently pooling, and non-canonical spellings such as `Repeat` raise unless named explicitly.

### FOV identity versus analysis grouping

Two concepts are kept separate:

- **FOV identity** (`ValidatedInput.fov_key`): the columns that jointly identify one acquired FOV. It always contains the analysis group columns plus any canonical `condition`/`repeat` columns, then the FOV column.
- **Analysis groups** (`ValidatedInput.group_cols`): which FOVs are evaluated together.

Passing `group_cols=[]` is an explicit request to pool FOVs into one analysis. It does not remove `condition`/`repeat` from FOV identity, so `R1/img_001.tif` and `R2/img_001.tif` remain two FOVs. Strategies that need FOV identity must use `fov_key`, not `fov_col` alone.

Metadata columns are never inferred as phenotype features. Strategies read phenotype values through `ValidatedInput.feature_matrix()`.

## Strategy contract

A strategy is estimators plus decision policies:

```python
class Estimator:
    name: str
    parameters: dict            # defines config_id

    def estimate(self, data, *, objective) -> Evidence:
        ...

class DecisionPolicy:
    def decide(self, evidence, effort) -> (Component, Decision | None):
        ...

SamplingStrategy(estimators, policies, name=...)
```

- **Evidence** is a set of named `Quantity` values. Each has a value, a `direction` (`lower_is_better` or `higher_is_better`), `units`, the objective `component` it is evidence about (`population` or `rare`), and an `availability`. `insufficient_data` means more acquisition may make it computable; `not_applicable` means it is undefined for these data. Either needs a reason, and NaN is never a value. Free-form diagnostics accompany the quantities.
- **Policies** are `ThresholdPolicy(quantity, threshold)`, whose component is the quantity's own, and `EffortPolicy("n_fovs" | "n_observations", target)` (`n_objects` remains a compatibility alias) on framework-owned effort. Custom policies subclass `DecisionPolicy`.
- **Combination** per objective is described in `docs/sampling_objectives.md`. A required component without a policy is `unsupported`, never `stop`.
- **`redecide(result, policies, name=...)`** applies other policies to stored results. It equals re-running the same estimators with those policies, which makes threshold sweeps and Pareto curves cheap.
- **`config_id`** is a 16-hex-digit hash of the class and canonical-JSON parameters of each estimator and policy. A strategy's id combines its parts' ids and ignores names, so `scalar_precision_object` at 2.5% and at 5% share a name but not an id.

An estimator receives one already validated analysis group at a time.

An estimator instance is configuration only. The framework calls `estimate` on a fresh deep copy, with its own copy of the input, for every analysis group and every acquisition prefix, so no fitted state (scaler, PCA, reference sample) can flow between groups, prefixes or estimators. `estimate` is therefore a full recomputation on the visible data. Efficient incremental updates will need an explicit, separately tested state interface; they must not rely on hidden instance state.

Feature preprocessing goes through `morphocoverage.transforms`:
- a `FeatureTransform` (`Standardize`, `PCA`, `Chain`) is declared as an estimator parameter;
- it is fitted inside `estimate` on the visible prefix the estimator receives, so it cannot see future FOVs or other groups;
- its fitted state is recorded with `fitted.provenance()`;
- a transform that cannot be fitted (too few observations, zero spread) says why instead of guessing a scale.

A contract test alters every FOV acquired after step k and requires every transform's state and output at steps up to k to be unchanged. Upstream normalization over the whole dataset cannot be detected and must be recorded as provenance.

An estimator must not:

- split conditions or repeats;
- infer feature columns;
- read microscopy images;
- silently inspect withheld FOVs;
- redefine the user's sampling objective;
- decide: thresholds belong to policies.

## Sampling objective

The objective is independent of the strategy.

Current objective types are:

- `population`
- `rare_discovery`
- `balanced`

Rare-discovery and balanced objectives require a `RareTarget`: object prevalence and/or FOV incidence, detection probability, minimum number of FOV occurrences, and a declared phenotype definition.

A strategy may perform well for one objective and poorly for another. That is a benchmarking result, not an API failure.

## Canonical result

The framework builds every result; strategies never construct one. A `StrategyResult` contains:

- strategy name and `config_id`;
- objective;
- evidence (quantities and diagnostics; `diagnostics` is a shortcut);
- overall decision: `continue`, `stop`, `unsupported` or `None` (undefined);
- `component_decisions`, one per component;
- number of FOVs and number of observations;
- resolved condition/repeat group identity;
- strategy parameters, and `configuration` (name, type, `config_id` and parameters of each estimator and policy);
- feature columns and FOV key;
- `compute_seconds`, with `runtime_seconds` as an alias, and `timing`, which breaks the whole decision down (see below).

Effort is framework-owned: estimators cannot report `n_fovs` or `n_objects` as quantities, so a count that drops zero-observation FOVs cannot enter a decision. `compute_seconds` times the estimators' own calls only.

This common contract is what allows many methods to be compared without changing the data pipeline.

## Sequential real-data evaluation

Deeply sampled real datasets are the primary scientific benchmark.

The sequential runner reveals a real acquisition prefix at a time while preserving:

- acquisition order;
- FOV identity;
- condition/repeat grouping;
- zero-observation FOVs.

Each strategy is evaluated using only the prefix available at that step.

Leakage guards enforced by the runner:

- the complete linked tables are validated once, so observations from FOVs missing from the FOV table raise rather than never being revealed;
- acquisition order must have no ties within an analysis group; a pooled analysis therefore needs an order defined across the pooled FOVs;
- the acquisition order column cannot be a phenotype feature;
- visible tables are re-indexed and categorical columns drop unused categories, so neither row labels nor category levels reveal withheld FOVs or other groups.

The runner cannot detect leakage introduced upstream, such as features normalized or projected over the whole dataset before they reach MorphoCoverage. That must be recorded as provenance.

The framework deliberately keeps strategy execution separate from the later benchmark outcome calculation so a method is not rewarded by being judged only with its own internal metric. `morphocoverage.benchmark` records one run per dataset, group, acquisition order and strategy, and `morphocoverage.outcomes` scores each stop against the full acquisition, or, with `run_benchmark(..., reservoir=ReservoirSplit(...))`, against a held-out reservoir; see `docs/benchmarking.md`.

## Sampling-unit checks

`morphocoverage.sampling_unit` answers section 4 (CHECK) of `docs/questions.md`: whether the cells of a completed acquisition behave like the random cells that cell-number rules assume. `field_variability` gives the variance components of one feature (reusing `retrospective.effort.nested_variance_components`), the design effect and the fields that differ more than random cells would. `cell_rule_check` and `descriptor_spread_check` run the JCB rule and the JCS Fig. 2A procedure twice, with random cells and with whole fields. `unusual_cells` labels observations far from declared reference groups (standardized features, mean distance to the nearest reference observations with `scipy.spatial.cKDTree`, thresholded against reference observations of other fields; a group is never its own reference), following the reference-sampling, provenance and seed-check design of fiberfeatures' pixel categories without its scikit-learn mixture model. `field_clustering` takes any per-observation state labels and compares their field incidence with the random-cell expectation. All are retrospective and group-local; the names stay in the subpackage and are not part of the top-level `__all__`. Comparisons between groups (the JCS treatment comparisons with whole fields) stay in `data/crick_idr/jcs_effect_fields.py` until multi-group outcomes have their own contract (AGENTS.md section 9).

## Computational benchmarking

Every result carries a `Timing` breakdown:

| Field | What it times |
|---|---|
| `recompute_seconds` | full `estimate` calls on the visible prefix |
| `update_seconds` | incremental `update` calls on the newly revealed FOV |
| `initial_fit_seconds` | `start` calls, plus warm-up updates before `min_fovs` (first result of a replay only) |
| `framework_overhead_seconds` | everything else: building the prefix or the new-FOV input, copying, policies, the result |
| `total_decision_seconds` | the sum: MorphoCoverage's share of the time between an FOV being revealed and a decision |

`evaluate_acquisition_prefixes(..., incremental=True)` replays the acquisition as an online microscope would.

- Estimators whose `start` returns an `EstimatorState` receive only each new FOV.
- The others are recomputed on the prefix. The prefix is built only if some estimator needs it.
- Decisions are the same either way; only the timing differs.
- `ScalarPrecisionEstimator` (per-FOV sufficient statistics, so an update costs O(new observations + acquired FOVs)) and `FOVIncidencePlanEstimator` are incremental.

`run_benchmark` records, per run, the decision latency up to the stop, and its median and maximum per step. These are estimator and framework times only; feature extraction and microscope control are not included.

Later benchmarking will extend this with:

- total runtime across a complete acquisition;
- high-percentile latency across runs;
- peak memory;
- empirical scaling with FOV count, observation count, and feature dimension;
- compute-to-acquisition ratio for smart microscopy.

## Current baseline strategies

The initial `FixedFOVStrategy` and `FixedObjectStrategy` are architectural baselines only: an `EffortPolicy` with no estimator. They exist to prove that strategies can be swapped through one shared runner and serve as current-practice comparators (`baseline_comparison`, the idr0028 replays).

Each baseline strategy keeps its published parameterization and is built from an exported estimator plus a `ThresholdPolicy`:

| Strategy | Estimator | Quantity (lower is better) |
|---|---|---|
| `ScalarPrecisionStrategy` | `ScalarPrecisionEstimator` | `averaging_cv`, or `pilot_projected_averaging_cv` with a pilot |
| `DescriptorPlateauStrategy` | `DescriptorPlateauEstimator` | `max_descriptor_range` |
| `DescriptorSpreadStrategy` | `DescriptorSpreadEstimator` | `max_descriptor_spread` |

All three are population evidence, so under `rare_discovery` they return `unsupported`.

`ScalarPrecisionStrategy` reproduces the **single-sample/pilot variant** of the cell-averaging criterion of Zweifach (J Cell Biol 2024, doi:10.1083/jcb.202401074) for one explicitly named scalar feature, following AGENTS.md §10. The paper's multi-sample planning procedure is a separate step, `plan_cells_to_average` with `PlannedCellCountStrategy` (below).

- **Scalar parameter:** the cell/object mean of the named feature, which must be positive-valued; its averaging CV is `SD / (mean * sqrt(n))`.
- **Criterion:** stop once `n >= ceil((SD / target) ** 2)` (`required_objects`). The target is required, with no default: the paper suggests 5% for independent samples and 2.5% for matched designs.
- **SD estimate:** `sd_estimate="point"` or `"upper_95"`, the chi-square upper bound of the 95% confidence interval for SD that the paper recommends for small pilots. This reproduces the paper's worked example (51/108 cells at 2.5%, 13/27 at 5%), which is covered by a test.
- **Planning versus sequential:** `pilot_fovs` plans the required number of cells once from the first acquired FOVs, as in the paper. The estimator reports the CV this plan projects for the visible effort, which reaches the target exactly when the planned number is reached. Without it the criterion is re-checked after every FOV; this is an adaptation, labelled by the strategy name.
- **Spatial uniformity:** the paper assumes it. `resampling_unit="object"` is the published criterion. `resampling_unit="fov"` is a MorphoCoverage variant that estimates the same averaging noise with FOVs as the sampling unit, and reports a design effect. It is not part of the paper; the gap between the two variants is what the regime map should quantify.
- **Not reproduced by this strategy:** pooling SD across samples after subtracting each sample's mean, and normalizing to the mean of all control samples. Both need other analysis groups, which strategies never see; they are the planning step below.

`plan_cells_to_average` is the paper's **multi-sample procedure**, run once on pilot data before evaluation. It divides every value by the average of the control samples (selected with `control`, e.g. `{"condition": "DMSO"}`), subtracts each sample's mean (samples identified by `sample_cols`), and computes the SD of the whole centred data set. It returns a frozen `CellCountPlan` with the required number of cells.

- **Choices the paper leaves open:** the control average is the mean of the control samples' means, each sample counting once; the SD uses `ddof=1` over all centred cells and the upper bound uses `n_cells - 1` degrees of freedom, which reproduces the worked example from two dishes of 10 cells. A pooled-variance estimate would use `n_cells - n_samples`.
- **Leakage safety:** `PlannedCellCountStrategy(plan)` holds the required count fixed and stops each evaluated group once it has acquired that many cells. The plan records every pilot sample; an evaluated group that contains one, or whose samples cannot be identified from `sample_cols`, gets `not_applicable` evidence and an undefined decision rather than being judged on its own planning data.
- **Observation units:** planning raises on rows that are not object-like; the strategy reports `not_applicable` for them.

`FixedAreaStrategy` stops at a fixed physical imaged area, summed from a declared FOV-table area column (zero-observation FOVs included; missing, zero or negative areas `not_applicable`). It is the nominal area acquired, not the unique specimen area: overlapping tiles count their overlap more than once. It is more comparable than a fixed FOV count when field size differs; on the `unequal_fov_area` scenario the two rules stop at different FOV counts.

`FOVBootstrapStrategy` (in `morphocoverage.strategies`) is the first multivariate MorphoCoverage estimator from the roadmap (`docs/proposals/candidate_estimators_roadmap.md`, Phase 1; the section numbers cited in the estimator modules refer to it). It resamples whole visible FOVs, including zero-observation FOVs, and reports the bootstrap standard error of a population summary (the per-feature centroid, every covariance entry, or chosen per-feature quantiles) two ways: the root-mean-square over components and the largest component. The strategy stops on the largest by default (`aggregate="max"`), since the RMS is an average that adding stable features dilutes: one unresolved feature among 100 constant ones has a tenth of its own SE. For one feature without a transform, the centroid variant approaches the FOV-unit JCB rule's analytic standard error times `sqrt((k - 1) / k)`, which a test checks; it adds joint uncertainty over several features, the covariance and the quantiles. Two PCA summaries follow roadmap 1.6–1.7: `eigenspectrum` (explained-variance fractions) and `subspace`, which reports the root-mean-square largest principal angle, in radians, between each resample's leading `n_components` subspace and the visible data's; near-equal eigenvalues at the cut correctly make it large. With `block_col` (an FOV-table column of spatial blocks, e.g. from `reservoir.spatial_blocks`) whole blocks are resampled instead of single FOVs, a spatial block bootstrap (roadmap 8.4): on the neighbour-correlation scenario it widens the centroid standard error by more than half, as independent-FOV resampling is optimistic there. Blocks are drawn with equal probability as clusters, so a resample's random total size is part of the uncertainty: with blocks of very different occupancy this can dominate the standard error even when the design sampled FOVs rather than blocks (the diagnostics report `block_sizes`). Blocks should be at least as large as the correlation length and of similar occupancy, and `min_fovs` also requires that many blocks, since a bootstrap over two blocks is meaningless. PCA resamples with too little spread to define the summary make it `insufficient_data` when they are more than 5% of the resamples, instead of being dropped, which would report false certainty; below that share they are dropped and counted, so availability is a property of the data rather than of the number of resamples. The transform, number of resamples, seed and a minimum number of FOVs with observations are explicit. It is blind to phenotypes never seen, understates variance with few FOVs, is weak for tail quantiles, and is optimistic when neighbouring FOVs are correlated; low uncertainty is not evidence that no unseen phenotype remains.

`SupportNoveltyStrategy` is a leave-one-FOV-out support estimator (roadmap 4.1–4.2). At a declared resolution `epsilon` (in the transformed feature space), an observation is covered when observations from at least `min_other_fovs` other FOVs lie within `epsilon`; holding out each visible FOV in turn, `lofo_novel_fraction` estimates the fraction of a new FOV's observations that would fall outside the current support. At the FOV level, `lofo_novel_fov_fraction` estimates the probability that another FOV shows at least `min_novel_observations` observations outside the current support, and `lofo_novel_fov_upper` is its Clopper–Pearson upper bound, which stays high until enough FOVs have been seen. It counts FOVs, not cells, on both sides: a state seen in one FOV stays novel however many cells show it there, and a dense neighbouring FOV counts once. `epsilon` and `min_other_fovs` define the estimand and have no defaults; the value depends on resolution and degrades in high dimensions, and a phenotype absent from every visible FOV cannot contribute.

`IncidenceCoverageStrategy` applies ecological incidence estimators to externally fixed labels (roadmap Phase 5), with the FOV as the sampling unit: Chao and Jost (2012) sample coverage (`incidence_coverage`) and Chao2 richness, reported as completeness `S_obs / S_chao2` (`chao2_completeness`). It also reports the probability that the next FOV shows at least one new labelled state (`prob_new_state_next_fov`, under the Chao et al. 2014 model), the conservative Good–Turing bound `min(1, Q1/T)`, and a small-sample-safe leave-one-FOV-out rate with its Clopper–Pearson upper bound (`new_state_fov_upper`: 0.78 after two identical FOVs at 95%, where the model probabilities are 0). It reports `expected_new_states_next_fov`, the Chao et al. (2014) incidence extrapolation of how many new labelled states one more FOV would reveal; in simulations with independent FOVs it tracks the realized number and slightly undershoots it. Many cells of one state in one FOV are one incidence. `label_col` is an observation label column fixed independently of the acquisition; it must not be a phenotype feature, and partitions fitted to the evaluated data are not valid inputs because they break the meaning of singletons. Spatially correlated FOVs violate the independence the estimators assume, and coverage refers only to the declared label system.

`FOVIncidencePlanStrategy` is a planning comparator for the rare component (roadmap 7.1). From the objective's `RareTarget` it computes the smallest number of independent FOVs with `P(Binomial(n, fov_incidence) >= min_fov_occurrences) >= detection_probability` (`scipy.stats.binom`) and stops at that count. It uses no phenotype data, so it is a fixed-FOV rule, not an adaptive estimator. A target stated only as object prevalence gives `not_applicable` evidence: prevalence is never converted to FOV incidence, which would need observations per FOV and within-FOV clustering. Spatial clustering of the rare state makes the binomial plan anti-conservative; a test keeps a counterexample.

`RareIncidenceStrategy` is the adaptive counterpart (roadmap 7.3–7.4). With an externally labelled rare state (`label_col`, booleans or 0/1, not a feature), `x` positive FOVs among `k` visible FOVs and an explicit `Beta(prior_a, prior_b)` prior on per-FOV incidence, it reports `rare_incidence_risk`: the posterior probability that the incidence is at least the target's `fov_incidence` while the state has not yet recurred in `min_fov_occurrences` FOVs (zero once it has). It stops the rare component when the risk is at most `max_risk`. This is a Bayesian statement given the prior ("the posterior probability that the incidence is at least the target is 4%"), not the frequentist detection probability of the binomial plan; `max_risk = 1 - detection_probability` makes the two rules similar, not equal. With a uniform prior and no detections it stops exactly one FOV before the binomial plan; it stops as soon as the state recurs. Recurrence counts positions, not cells, and clustering across neighbouring FOVs makes the posterior overconfident.

`morphocoverage.spatial` holds spatial diagnostics (roadmap 8.2), not stopping rules: `empirical_semivariogram` (Matheron's estimator in distance bins, via `scipy.spatial.distance.pdist`) and `fov_semivariogram`, which applies it to the FOV means of the selected features against stage coordinates. Where the semivariogram levels off is a scale for spatial blocks in the reservoir split and the block bootstrap. Run it retrospectively or on the visible prefix, never to tune a prospective estimator on future FOVs.

`DescriptorPlateauStrategy` and `DescriptorSpreadStrategy` operationalize the sample-size advice of Marcotti et al. (J Cell Sci 2026, doi:10.1242/jcs.264367): increase the number of cells until the mean, median, SD and IQR of one metric "stabilise". The paper gives no numeric rule, so each tolerance is an explicit user choice in the feature's units. Both strategies are labelled operationalizations, not published rules.

- **Plateau (Fig. 2B):** the paper compares against the full well, which a prospective strategy cannot see. Instead, stability is judged over a trailing window of acquired FOVs that added cells; empty FOVs add no state.
- **Spread (Fig. 2A):** the max − min of each descriptor across `n_resamples` resamples, drawn with replacement at the current size, because drawing without replacement collapses at the full visible size. `resampling_unit="object"` resamples cells, as in the paper, which assumes random cell sampling. `"fov"` resamples whole FOVs, including empty ones, as a MorphoCoverage variant. With few FOVs, the FOV bootstrap has few distinct resamples and is unreliable.

A counterexample test shows that the fixed, JCB-style and Marcotti-style cell-number rules all stop before a late, spatially clustered rare state. That is a constructed failure mode, not evidence about real data.

None of these are scientific defaults.

## Extension rule

The step-by-step checklist for adding a strategy is in AGENTS.md section 20.

New strategies should be added behind the existing interface rather than adding strategy-specific branches to the runner.

Examples of families not yet implemented include:

- MMD / energy / Wasserstein distribution-distance strategies;
- kNN and density novelty;
- FOV-distribution methods;
- predictive or conformal approaches.

A new method should not require changes to MorphoBricks metadata handling or group separation unless the data contract itself genuinely changes.

See [`morphobricks_fov_contract.md`](morphobricks_fov_contract.md) for what upstream MorphoBricks tools must emit for FOV identity to be correct.

## Multidimensional FOV semantics

The framework's FOV is a spatial acquisition position, not a plane or file. `position_id` is the preferred explicit identifier when file layout does not map one-to-one to positions. Rows produced slice-by-slice or frame-by-frame may vary in `z`, `c`, and `t`, but those columns are within-FOV metadata and all rows from one position are revealed together during sequential replay.


## Offline and online execution share one core

MorphoCoverage has two orchestration surfaces over the same estimator/policy machinery:

```text
completed acquisition                  live microscope
        |                                   |
evaluate_acquisition_prefixes()        OnlineSession.add_fov()
        |                                   |
        +-------------+---------------------+
                      |
                SamplingStrategy
              Estimator(s) + Policy
                      |
                StrategyResult
```

`OnlineSession` owns only prospective orchestration. It does not know future FOVs and does not implement strategy-specific logic. The first FOV fixes the session's identity/grouping contract; later updates cannot silently change it. Incremental estimators use `EstimatorState`; non-incremental estimators are recomputed on accumulated visible observations.

## Observation semantics

The historical API name `objects` is retained for compatibility, but rows are generic phenotype observations. `ValidatedInput.observations`, `n_observations`, and `observation_unit` are the canonical semantics. This permits cell-, nucleus-, patch-, slice-, embedding- and FOV-summary tables without changing strategy plumbing.

NaN phenotype values may enter the framework. They are never automatically imputed or discarded. Methods must either support missingness explicitly or report that their evidence is not applicable/available.


### Lean feature tables

A feature table does not have to duplicate all FOV-level metadata. If observation
rows contain a globally unique FOV/position identifier and the FOV table carries
`condition`, `repeat`, or explicitly requested grouping columns, validation
attaches those columns to the observations before grouping.

This join is performed only when one FOV identifier maps to exactly one metadata
value. Reused basenames or position IDs across groups are not guessed: the
observation table must then carry the relevant identity columns or use a
globally unique FOV identifier.


## Observation-unit compatibility

The input layer is generic, but individual methods need not be. Each estimator
must make its observation-unit assumptions explicit.

- generic distributional/novelty methods may accept cells, nuclei, patches,
  slices, embeddings or FOV summaries where scientifically meaningful;
- cell/object-derived current-practice baselines report their evidence as
  not applicable for other units (decision `None`), rather than raising, so a
  benchmark or live session with several strategies keeps running;
- the scope follows the method's sampling assumption, not its name: the
  object-unit JCB criterion, the descriptor plateau and the object-unit
  descriptor spread need object-like rows (`object`, `cell`, `nucleus`); the
  FOV-unit JCB variant and the FOV-unit descriptor spread resample whole FOVs
  and accept any unit;
- `FixedObservationStrategy` is the generic row-count baseline;
- `FixedObjectStrategy` is retained for backward compatibility; for
  non-object units it gives no decision;
- `observation_unit` labels are normalized (lowercase, singular, `_` for
  spaces), so `Cells`, `cell` and `cells` are one unit.

A generic input contract must never silently change the meaning of a published
cell-count method.

## Retrospective analyses

`morphocoverage.retrospective` holds the retrospective analyses of the microscopist's report (`docs/proposals/microscopist_report_roadmap.md`). They judge sets of FOVs against FOVs no estimator saw, through `OutcomeMetric` values and declared `OutcomeCriterion` tolerances, and never feed a prospective estimator. `replay_acquisitions` is the shared engine for Steps 2, 3 and 8: it reuses `sequential`'s prefix machinery and the estimators and policies a live session runs, so decisions are identical to online acquisition, while outcomes against the held-back reservoir are recorded separately. The names stay in the subpackage and are not part of the top-level `__all__`.

## Computation

How the slow analyses are made fast without changing what they compute. `docs/optimization_log.md` has every change tried, with timings and the identity checks.

- **Subsets and prefixes.** `evaluator.FovSubsets` looks up each observation's FOV once, so a subset of FOVs is one vectorized mask. Replay prefixes are slices of rank-sorted observations, returned in table order, so they are the same tables a mask would give.
- **Support novelty.** `_supporting_fovs` picks the fastest exact counting method (pairwise distances for small problems or eight or more dimensions, one tree over all references, or one tree per reference FOV in dense neighbourhoods). Every path decides distances near `epsilon` with one definition, `<= epsilon`, which a test checks at exact ties.
- **Repeated reads.** `ValidatedInput` caches its feature matrix and the FOV positions of its own FOV table (each read returns a copy; copies made with `dataclasses.replace` start empty). Code that edits a table in place and reads it again calls `clear_cache()`.
- **Worker processes.** `run_strategies`, `evaluate_acquisition_prefixes`, `run_benchmark`, `replay_acquisitions`, `sampling_reliability` and `representation_control` take `n_jobs` (`-1`: every CPU).
  - Random choices are seeded from each unit's identity or drawn in the calling process, so results do not depend on `n_jobs`. They are bit for bit identical when BLAS runs single-threaded; otherwise a multithreaded BLAS in the calling process can change the last digit (see `morphocoverage._parallel`).
  - Units are as coarse as keeps every worker busy: whole groups, else single runs or reliability splits, else every strategy apart (for one sample compared across several strategies).
  - Workers start from a clean forkserver (or spawn) with one BLAS thread each; the server starts once per process, in about a second. Classes sent to workers must come from an importable module, not a notebook cell.
