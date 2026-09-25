# Benchmarking strategies

The questions being benchmarked are the rows marked **Benchmark** in `docs/questions.md` (R2, R3, P1, P5, P6, P10, O6); the other questions are measurements or diagnostics, reported but not ranked.

MorphoCoverage is a method-comparison framework. Candidate strategies must be evaluated using common criteria that are independent of any one strategy's internal score.

The proposed next-generation estimator and benchmark roadmap is documented in [`docs/proposals/candidate_estimators_roadmap.md`](proposals/candidate_estimators_roadmap.md).

## Primary scientific benchmark: deeply sampled real datasets

The primary benchmark will use real microscopy datasets acquired substantially deeper than routine experiments.

For a sample with FOVs (F_1,\ldots,F_n), strategies are evaluated on acquisition prefixes (F_1,\ldots,F_k) while later FOVs remain hidden. The hidden real FOVs are then used to measure what additional acquisition actually contributed.

The complete acquisition is an empirical reference reservoir, not assumed biological ground truth.

Synthetic data remain useful for unit tests, explicit counterexamples, and regression tests after a failure mode is discovered. They are not the primary scientific benchmark.

## Protocol as implemented

`morphocoverage.benchmark` and `morphocoverage.outcomes` fix how runs are recorded. They do not select a winner.

- **Dataset roles.** Every dataset is registered in a `BenchmarkRegistry` as `development` or `held_out`. A run declares the datasets its settings were tuned on (`tuned_on`). Those must be development datasets, and a run on one of them is recorded as in-sample. Performance claims belong to held-out or out-of-sample runs. `leave_one_dataset_out` yields folds when data are scarce. The committed idr0028 subset in `data/crick_idr/` is a development dataset (see below); idr0139 is test data only.
- **Acquisition order.** `AcquisitionOrder.recorded()` replays the recorded order. `AcquisitionOrder.permuted(seed)` replays a uniformly random order of whole FOVs. A permutation assumes the FOVs of a group are exchangeable, which the recorded order need not be (stage drift, neighbouring fields), so both kinds are recorded.
- **Stop.** A strategy stops at the first prefix whose decision is `stop`. A run that never stops, including one whose decision is `unsupported`, is recorded with `stopped=False`, its final decision, and effort and outcomes at the full acquisition.
- **Record.** `run_benchmark` returns one `BenchmarkRecord` per dataset, analysis group, order and strategy. Each record holds:
  - the dataset, its role, `in_sample` and `tuned_on`;
  - the group, the order (and its seed) and the objective;
  - the strategy name, its `config_id`, and the estimator and policy ids;
  - the stop step, the final decision, and effort at the stop (FOVs, observations with their declared unit, and area) against the totals;
  - the outcomes, each with its metric's `config_id`;
  - the compute time up to the stop, and the maximum compute time of any one step.
  
  `records_to_frame` flattens the records to one row each.
- **Reference.** By default outcomes compare the stop with the full group, which contains the stop prefix: every error goes to zero as the prefix grows, whatever the strategy. `run_benchmark(..., reservoir=ReservoirSplit(fraction, seed, block_col))` instead splits each analysis group before replay (`morphocoverage.reservoir`). Strategies replay only the stream; outcomes are scored against the held-out reservoir, which does not grow with the prefix. Whole blocks go to one side: `block_col` names an FOV-table column of blocks (`spatial_blocks` builds a square grid from stage coordinates); without it FOVs are assumed exchangeable. The reservoir is a simple random sample of `ceil(fraction * n_blocks)` blocks, so every block and every FOV has the same chance of being held out whatever the block's size, and the reservoir represents the stream's FOV population on average; the share of FOVs held out then varies with unequal blocks. Blocks keep the FOVs inside one block together, but FOVs on either side of a block edge can still be close neighbours: blocks limit spatial leakage only as far as they are larger than the correlation length (or follow wells or tiles). The stream is a thinned acquisition: holding FOVs out also removes them from the recorded order, so successive stream FOVs are further apart than successive acquired FOVs (about twice as far for single-FOV blocks at `fraction=0.5`), which weakens neighbour correlation and can make stopping look easier. Report reservoir runs as thinned-stream validation, prefer contiguous blocks, and read recorded-order results alongside a replay without a reservoir. Records state `reference`, `reservoir_fovs` and the split settings. Repeat over several seeds so conclusions do not depend on one partition. `reference_error_floor` scores random halves of the reservoir's blocks against each other; for mean-type errors with equal, independent halves this is about twice the reservoir's own standard error, a conservative noise scale that adequacy tolerances should sit well above. With unequal or correlated blocks it is only approximate, and it is never a decision threshold: an error below it does not mean imaging was sufficient. Report the stop FOVs, reservoir FOVs, held-out error, this noise scale, the split seed and blocks, and the variation over several splits; one split should not drive a recommendation.
- **Outcome metrics** are functions of (full group, stop prefix) → value. They never see a strategy's evidence or decision, so two strategies that stop at the same prefix get identical outcomes; a contract test checks this. They are retrospective and must never be used inside an estimator. The first metrics are:
  - `RelativeMeanError(feature)`: the realized relative error of the observation mean at the stop.
  - `RandomCellRelativeError(feature)`: the expected relative RMSE had the same number of cells been drawn at random from the full group (simple random sampling with finite-population correction). It is the comparator in `data/crick_idr/fov_vs_cells.py`. Across permuted orders, the RMSE of `RelativeMeanError` divided by this value is the error inflation from acquiring whole FOVs.
  - `RareStateDetected(label_col, min_fov_occurrences)`: whether the stop prefix shows an independently labelled state in enough FOVs. It is `None` when the full acquisition never does.
  - Multivariate distribution outcomes in `morphocoverage.multivariate_outcomes`, over explicit `features`, optionally standardized by the reference's mean and SD: `CentroidError` (RMS difference of feature means), `CovarianceError` (relative Frobenius error), `QuantileError` (RMS difference of chosen quantiles), `FrechetError`, `EnergyError` and `MMDError` (U-statistics with a seeded subsample limit because they are quadratic; unbiased only against a reference disjoint from the stop, such as a reservoir, and for independent observations: against the full acquisition, which contains the stop, they are biased downward and descriptive only; with cells correlated within FOVs their null expectation is positive and shrinks as FOVs are added, and single values can be negative, so compare them across runs at matched effort, never against zero), and `SlicedWassersteinError` (seeded projections). The pure distance functions are in `morphocoverage.distances`. The plug-in metrics (centroid, covariance, quantile, Fréchet) are biased upward for small stops, so compare them at matched observation counts; the U-statistics are not when observations are independent, but cells within one FOV usually are not, so match FOV and observation counts for them too. All of them pool observations, so FOVs are weighted by their number of observations: they score the cell-weighted phenotype distribution, and a missed sparse FOV with different biology barely moves them. They are outcomes, not stopping rules.

- **Timing.** Every record carries estimator compute time, whole-decision latency up to the stop, and the median and maximum per-step decision latency. Only decisions up to the actual stop contribute to these summaries; later replay steps are retrospective and must not inflate the latency of a microscope that would already have stopped. `run_benchmark(..., incremental=True)` replays incremental estimators FOV by FOV, as online acquisition would. The `Timing` breakdown on each result separates estimator time from framework overhead.

Adequacy points are now defined (`morphocoverage.retrospective.adequacy_points`, below). Not yet implemented: excess FOVs beyond them in benchmark records, and aggregation into a regime map.

## Retrospective sufficiency of one dataset

`sufficiency_report(objects, fovs, feature_cols=..., strategies=..., objective=..., orders=..., novelty=..., novelty_target=..., calibration=...)` answers two questions per analysis group of an analysed dataset:

1. **Did each criterion say enough?** (`enough`) For each criterion, judged by its own threshold and not by held-out FOVs: `sampled_enough` (yes, no or undefined on the acquired FOVs; a fixed count reports that its target was reached), after how many FOVs it was first met across the declared acquisition orders (median and 10-90% range), and the FOVs imaged beyond that point. Verdicts hold only for each criterion and threshold, so compare several.
2. **Would one more FOV show something new?** (`next_fov`, given a `SupportNoveltyEstimator` that defines "new") The chance, its nominal upper bound (Clopper–Pearson on leave-one-FOV-out outcomes, which are not independent, so its coverage is approximate and is checked by the calibration below), and, for a `novelty_target`, the extra FOVs needed before the bound can fall below it: at least this many if none of them shows anything new, and this many if new FOVs keep appearing at the current rate. The target is also added as a criterion to question 1. `new_observations` lists the observations counted as new so they can be inspected in the images: debris or segmentation errors count as new too.

With `few_fovs`, it also counts the FOVs holding a state that fewer than `few_fovs` other FOVs show: a state recurring in two neighbouring FOVs is not new to the next FOV, yet not established either. It also states what the FOVs rule out: a state that no FOV has shown yet is present in at most `1 - (1 - confidence) ** (1 / k)` of FOVs (about `3 / k` at 95%; `unseen_state_max_frequency`), with the extra FOVs needed to rule out states above `novelty_target`. A small chance for the next FOV is not evidence that no rare state remains (preliminary findings, entry 3). `summary()` states both answers in sentences. `final` and `stops` keep the full evidence and stop distributions.

The chance can be checked on the dataset itself (`calibration=CalibrationSettings(...)`, or `morphocoverage.calibration.calibrate_next_fov_novelty`). Random subsets of `k` FOVs predict the chance, and the remaining FOVs, judged against the same subset with the same definition, give the actual rate. A FOV shows something new when at least `min_novel_observations` (`m`) of its unsupported cells form one cluster (linked within `epsilon`), so scattered outliers do not add up to a phenotype. Support is symmetric: another FOV covers a cell only with at least `m` cells of its own within `epsilon`, so a single stray cell cannot cancel a new state. On idr0028 (12 wells of 28 FOVs; 30 standardized intensity features; `epsilon = sqrt(30)`), pooled over wells with 20 draws each:

| FOVs imaged | `m = 2`: predicted | actual | mean nominal 95% bound | bound held | `m = 3`: predicted | actual | bound held |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 5 | 0.168 | 0.136 | 0.61 | 100% | 0.063 | 0.050 | 100% |
| 10 | 0.088 | 0.083 | 0.37 | 100% | 0.025 | 0.023 | 100% |
| 20 | 0.057 | 0.057 | 0.22 | 95% | 0.016 | 0.009 | 99% |

The definition moves the answer a lot. Counting any 3 unsupported cells in a FOV gave 0.20, 0.12 and 0.08, mostly scattered cells; requiring a cluster but letting one cell of another FOV support it gave 0.008, 0.001 and 0.000, because stray cells cancelled real groups. At half the distance (`0.5 sqrt(30)`, `m = 3`) about 18% of FOVs still show a new 3-cell state after 20 FOVs, and the bound held in only 91% of draws there. In every setting the prediction tracks the actual rate. With no new FOVs among `k` the bound is `1 - 0.05 ** (1 / k)` (0.14 at 20 FOVs) whatever the data. Random subsets treat FOVs as exchangeable, so this does not test drift along the recorded order, and the actual rate is measured on few remaining FOVs when `k` is close to the group size.

It is self-assessment of one dataset; whether a criterion is right in general is a benchmark question.

## How many FOVs next time? Sampling reliability

`morphocoverage.retrospective.sampling_reliability` answers the planning question without any stopping rule (roadmap Step 1). For each analysis group and outer split it holds back a fixed reservoir of whole FOVs (`ReservoirSplit`), draws `k` random FOVs from the rest, and assesses each declared `OutcomeCriterion` (an outcome metric with a tolerance and direction, for example the well mean within 5%) against the same reservoir for every `k`. The curve gives, per `k`, the share of draws in which the criterion was preserved (averaged over splits, with its spread), the metric's median and 10–90% range, and the gain from the previous `k`. The ceiling table gives how often the whole stream preserved the criterion: the reservoir is itself a finite sample, so the curve cannot be read above it. `summary()` states the curve in sentences. Draws and splits describe one sample and are not independent replicates; statements across samples count samples or repeats (roadmap Step 4). `SamplingHierarchy` declares which columns are the FOV, the sample and the independent unit; nothing is inferred from column names.

## Does this stopping rule work? Calibration and adequacy points

A stopping rule can depend on the order in which FOVs arrive, so it is checked sequentially, separately from sampling reliability (roadmap Steps 2 and 3). `morphocoverage.retrospective.replay_acquisitions` holds back the same fixed reservoir per split as `sampling_reliability` and replays the stream in the recorded order (when an acquisition-order column is given) and in seeded random orders (10 by default). After every FOV it records each strategy's decision, computed from the visible prefix only, and, separately, whether the prefix preserves each `OutcomeCriterion` against the reservoir. Outcomes never depend on a strategy.

- `stopping_calibration` finds each rule's first stop and reports, over replayed acquisitions, how often it stopped, the median FOVs at the stop, and the shares of correct stops (preserved at the stop), premature stops (not preserved), stops that could not be judged, and runs that never stopped, with whether those had it by the last FOV. For a fixed-FOV rule in random orders the share of correct stops estimates the same quantity as `sampling_reliability` at that `k`; a test checks this.
- `adequacy_points` gives, per replayed acquisition, the first and the sustained apparent adequacy (from decisions only; a later continue, unsupported or undefined decision breaks a sustained stop) and the first and the sustained realized adequacy (from outcomes only, for all criteria together and each on its own; a later failure breaks it, an undefined outcome is skipped). Sustained realized adequacy is the preferred answer to "how many FOVs did this experiment really need?".

Both have a `summary()` in plain sentences. Replayed acquisitions are hold-outs and orders of the same FOVs, not independent experiments.

## Across samples, fields to inspect, controls, effort and order

The remaining retrospective analyses (roadmap Steps 4–8 and 10) each answer one question and have a plain-language `summary()`:

- `protocol_reliability(per_sample, hierarchy, threshold=..., strata=...)` counts, from per-sample reliability curves, the samples adequately represented at each number of FOVs (reliability at least `threshold`, provisional preset 0.9), with explicit denominators, per independent unit, each sample's difficulty, and a leave-one-unit-out check of a number chosen on the other units. Intervals resample the declared independent units, so more draws or splits never narrow them; with fewer than five units the summary says the uncertainty is large. Pooling across conditions is an explicit `strata=[]`.
- `inspect_fovs` lists fields to reopen, one reason per line with its numbers: a new state (support novelty), influence on a declared FOV-weighted summary (mean or an interpolated quantile), an observation count far from the median, or a difference from the nearest stage neighbours. There is no combined score and no FOV is called an artifact.
- `representation_control` judges declared positive-control FOVs (and positive cells planted into reference FOVs) against random reference FOVs, paired within `pair_cols`: detection and false-alarm rates, "positive only" without a reference, "not assessed" without a control.
- `effort_projection` estimates nested variance components of a scalar mean (Henderson's method I for unbalanced designs) with a hierarchical bootstrap, and projects the standard error at equal total FOVs; a difference between allocations is stated only when its interval excludes 1 and there are at least five independent units. Levels that cannot be separated (one sample per repeat) raise. The bootstrap intervals are approximate: in a simulation with 8 samples their coverage was 80–88% for the sample and field components, not 95%.
- `order_efficiency` compares recorded, random and dispersed (`AcquisitionOrder.dispersed(seed)`, greedy farthest point on stage coordinates) orders of the same positions by sustained realized adequacy and stage travel; `redundancy_distance` gives the distance below which neighbouring fields are more alike than random pairs.
- `microscopist_report` assembles the answers in the order of the proposal's questions, with a sampling profile of measured quantities; a question without its analysis says "not assessed" and what it needs.

**Registered datasets** (`data/crick_idr/datasets.py`, a `BenchmarkRegistry`). idr0028 (committed subset: 12 wells × 28 FOVs on 3 plates) is the **development** dataset: every preset and definition of new was chosen on it, and it sets no protocol. idr0139 (10 wells × 4 FOVs on one plate, the JCS case study) is registered **held out**: never used to choose a setting, so `check_tuning` refuses a run that claims to be tuned on it. It is shallow (a 2-field reservoir judges 1 or 2 stream fields), so it can refute a development conclusion but not establish a field number; `heldout_idr0139.py` runs that check (`docs/preliminary_findings.md`, entry 7). Deeper held-out data are still needed.

## The score: fields saved at matched risk

`morphocoverage.retrospective.frontier` turns replayed acquisitions into one number per rule family, question and risk level, and keeps the curve behind it.

- **Risk** of a rule at one setting: the share of replayed acquisitions whose acquisition, ended at the rule's stop (or at the end of the stream when it never stopped), did not preserve the declared outcome in the held-back fields. Outcomes that could not be judged are left out of the share and counted.
- **Effort**: the mean number of fields acquired, the stop or the whole stream.
- **Baseline**: a fixed number of fields `k`, for every `k` up to the longest stream, computed from the same replays. It needs no data, so it is the null every rule must beat.
- **Chosen setting** at risk level `alpha`: the cheapest setting from which risk stays within `alpha` at every setting with more effort, so a lucky dip in a noisy curve is not taken as safe (the rule of `protocol_reliability`).
- **Score**: fields saved = the baseline's effort at `alpha` minus the family's. Zero is the null result; negative is worse than a fixed count. The interval resamples the declared independent units, never hold-outs or orders.
- **Adaptivity**: the Spearman correlation between where the rule stopped and where held-back fields confirmed the outcome lastingly. A rule that saves fields by stopping later on harder samples has a positive value.
- **Transfer**: the setting chosen on a development dataset to meet `alpha`, scored on a held-out dataset. A threshold is only as good as its risk on data it was not chosen on.

A family is either strategies replayed side by side (`family`) or one replayed configuration re-decided at many thresholds (`sweep`, which uses `redecide` on results kept with `replay_acquisitions(keep_results=True)`, so no estimator runs twice).

The benchmark on the committed data is `data/crick_idr/benchmark.py`. Its protocol (`benchmark_protocol.py`: questions, criteria, families and their threshold grids, risk levels, replay design, units) is locked by digest before the first run, and the runner refuses to run a changed protocol; a change needs a new version. Stage 1 compares each rule per cell and per field; stage 2 compares the richer rules with those and with the fixed count, question by question (`docs/questions.md`, the **Benchmark** rows). Results are in `docs/preliminary_findings.md`, entry 8.

## Scientific performance dimensions

Strategies should be compared on several independent outcomes, not only a metric related to their own score.

### Future-information prediction

How well does the strategy anticipate the information subsequently found in unseen FOVs?

Possible independent outcomes include changes in:

- feature means, variances, and quantiles;
- multivariate distributions;
- FOV-level distributions;
- minority-state representation;
- downstream biological results.

### Stopping performance

For each strategy and stopping rule record:

- stopping FOV;
- observations acquired at stopping, with their declared `observation_unit`;
- imaged area at stopping when known;
- premature-stopping rate;
- information remaining after stopping;
- excess FOVs acquired after an empirical adequacy target;
- fraction of acquisition saved.

### Rare-phenotype performance

When rare/minority phenotypes can be defined independently of the strategy, report:

- probability of stopping before first detection;
- probability of stopping before the required number of independent FOV occurrences;
- performance as a function of phenotype prevalence and FOV incidence.

Rare-phenotype labels should not be created by the same method being evaluated whenever an independent definition is available.

### Robustness

Evaluate sensitivity to:

- FOV acquisition order;
- biological replicate;
- dataset type;
- phenotype representation;
- feature preprocessing;
- hyperparameters.

### Comparison with current practice

At minimum benchmark:

- fixed observation/cell count;
- fixed FOV count;
- fixed imaged area when available;
- JCB/Zweifach-style scalar averaging precision;
- simple population-summary stabilization.

The objective is a regime map: identify when simple current-practice guidance is sufficient and when richer FOV-aware strategies add value.

## Computational performance

Runtime is a first-class benchmark because an intended application is smart microscopy. The retrospective analyses are offline, but they should still run in minutes: strategy comparisons, replays, reliability curves and representation controls take `n_jobs` to use several CPUs, and `docs/architecture.md` ("Computation") describes where the time goes, and `docs/optimization_log.md` records every optimization tried, its speedup and how results were shown to be identical.

For every strategy record at least:

- initial fitting time when applicable;
- per-FOV update latency;
- median and high-percentile update latency;
- total runtime over an acquisition;
- peak memory where practical;
- whether the method supports incremental updates or requires full recomputation.

Benchmark scaling against:

- number of FOVs;
- number of observations;
- feature dimensionality.

## Compute-to-acquisition ratio

For prospective smart microscopy define:

[
R_{compute}=\frac{T_{update}}{T_{available}}
]

where (T_{available}) is the time available before the next acquisition decision is required.

A strategy with (R_{compute}<1) can in principle update without delaying acquisition. This is more informative than imposing one universal latency threshold because acquisition times differ across microscopy workflows.

The eventual end-to-end loop also includes feature-extraction and microscope-control latency, but MorphoCoverage should separately report its own computational contribution.

## No single winner required

Do not initially collapse scientific performance, under-sampling risk, acquisition efficiency, runtime, and memory into one arbitrary score.

Report strategies as multidimensional profiles and identify Pareto-optimal trade-offs.

Different winners may be appropriate for:

- offline analysis;
- online continue/stop smart microscopy;
- later targeted acquisition.

A method that is slightly less accurate but orders of magnitude faster may be the better smart-microscopy strategy, while a slower method may remain preferable for offline experimental design.


## Online equivalence

`OnlineSession` is the deployment counterpart of incremental replay. It receives one FOV at a time and knows nothing about future acquisition. Contract tests compare live session outputs with `evaluate_acquisition_prefixes(..., incremental=True)` on the same revealed sequence. Scientific evidence and decisions must match; only timing context may differ.

Methods without an incremental `EstimatorState` remain deployable because the session can recompute them from accumulated visible observations. Benchmarking determines whether that latency is acceptable.
