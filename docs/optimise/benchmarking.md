# Benchmarking strategies

The questions being benchmarked are the rows marked **Benchmark** in the question tables of [REPORT](../report/questions.md), [OPTIMISE](optimise.md) and [LIVE](../live/live_session.md) (R2, R3, P1, P5, P6, P10, O6); the other questions are measurements or diagnostics, reported but not ranked.

MorphoCoverage is a method-comparison framework. Candidate strategies must be evaluated using common criteria that are independent of any one strategy's internal score.

The candidate estimators not yet built, with the benchmark requirements they must meet, are listed in [`docs/proposals/candidate_estimators_roadmap.md`](../proposals/candidate_estimators_roadmap.md).

## Primary scientific benchmark: deeply sampled real datasets

The primary benchmark will use real microscopy datasets acquired substantially deeper than routine experiments.

For a sample with FOVs (F_1,\ldots,F_n), strategies are evaluated on acquisition prefixes (F_1,\ldots,F_k) while later FOVs remain hidden. The hidden real FOVs are then used to measure what additional acquisition actually contributed.

The complete acquisition is an empirical reference reservoir, not assumed biological ground truth.

Synthetic data remain useful for unit tests, explicit counterexamples, and regression tests after a failure mode is discovered. They are not the primary scientific benchmark.

## Protocol as implemented

`morphocoverage.benchmark` and `morphocoverage.outcomes` fix how runs are recorded. They do not select a winner.

- **Dataset roles.** Every dataset is registered in a `BenchmarkRegistry` as `development` or `held_out`. A run declares the datasets its settings were tuned on (`tuned_on`). Those must be development datasets, and a run on one of them is recorded as in-sample. Performance claims belong to held-out or out-of-sample runs. `leave_one_dataset_out` yields folds when data are scarce. The committed idr0028 subset in `data/crick_idr/` is a development dataset (see below); idr0139 is test data only.
- **Acquisition order.** `AcquisitionOrder.recorded()` replays the recorded order. `AcquisitionOrder.permuted(seed)` replays a uniformly random order of whole FOVs, drawn per analysis group from the seed and the group's identity (`benchmark.permuted_fovs`, which `replay_acquisitions` also uses, seeded there from its `random_state`), so a group's order never depends on the other groups or on the table's row order. A permutation assumes the FOVs of a group are exchangeable, which the recorded order need not be (stage drift, neighbouring fields), so both kinds are recorded.
- **Stop.** A strategy stops at the first prefix whose decision is `stop` (the objective's decision; with `run_benchmark(..., stop_on="rule")`, the rule's own verdict, `rule_decision`, whatever the objective's other components say, which the criteria tables of `sufficiency_report` use). A run that never stops, including one whose decision is `unsupported`, is recorded with `stopped=False`, its final decision, and effort and outcomes at the full acquisition.
- **Record.** `run_benchmark` returns one `BenchmarkRecord` per dataset, analysis group, order and strategy. Each record holds:
  - the dataset, its role, `in_sample` and `tuned_on`;
  - the group, the order (and its seed) and the objective;
  - the strategy name, its `config_id`, and the estimator and policy ids;
  - the stop step, `stop_on`, the final decision (the one `stop_on` reads), and effort at the stop (FOVs, observations with their declared unit, and area, which must be numeric, positive and finite for every field) against the totals;
  - the outcomes, each with its metric's `config_id`;
  - the compute time up to the stop, and the maximum compute time of any one step.
  
  `records_to_frame` flattens the records to one row each.
- **Reference.** By default outcomes compare the stop with the full group, which contains the stop prefix: every error goes to zero as the prefix grows, whatever the strategy. `run_benchmark(..., reservoir=ReservoirSplit(fraction, seed, block_col))` instead splits each analysis group before replay (`morphocoverage.reservoir`). Strategies replay only the stream; outcomes are scored against the held-out reservoir, which does not grow with the prefix. Whole blocks go to one side: `block_col` names an FOV-table column of blocks (`spatial_blocks` builds a square grid from stage coordinates); without it FOVs are assumed exchangeable. The reservoir is a simple random sample of `ceil(fraction * n_blocks)` blocks, so every block and every FOV has the same chance of being held out whatever the block's size, and the reservoir represents the stream's FOV population on average; the share of FOVs held out then varies with unequal blocks. Blocks keep the FOVs inside one block together, but FOVs on either side of a block edge can still be close neighbours: blocks limit spatial leakage only as far as they are larger than the correlation length (or follow wells or tiles). The stream is a thinned acquisition: holding FOVs out also removes them from the recorded order, so successive stream FOVs are further apart than successive acquired FOVs (about twice as far for single-FOV blocks at `fraction=0.5`), which weakens neighbour correlation and can make stopping look easier. Report reservoir runs as thinned-stream validation, prefer contiguous blocks, and read recorded-order results alongside a replay without a reservoir. Records state `reference`, `reservoir_fovs` and the split settings. `ReservoirSplit.split` and `reference_error_floor` take one analysis group and refuse several: a split never pools groups. Repeat over several seeds so conclusions do not depend on one partition. `reference_error_floor` scores random halves of the reservoir's blocks against each other (each half drawn with `split_seed(seed, i)`, as the reliability and replay splits are); for mean-type errors with equal, independent halves this is about twice the reservoir's own standard error, a conservative noise scale that adequacy tolerances should sit well above. With unequal or correlated blocks it is only approximate, and it is never a decision threshold: an error below it does not mean imaging was sufficient. Report the stop FOVs, reservoir FOVs, held-out error, this noise scale, the split seed and blocks, and the variation over several splits; one split should not drive a recommendation.
- **Outcome metrics** are functions of (full group, stop prefix) → value. They never see a strategy's evidence or decision, so two strategies that stop at the same prefix get identical outcomes; a contract test checks this. They are retrospective and must never be used inside an estimator. Every metric takes an optional keyword `label`, its name in the records, so two settings of one metric can be told apart (`run_benchmark` refuses two metrics of one name and points to `label=`). A stop with fewer observations than a metric needs (`OutcomeMetric.min_observations`: one for a mean, two for a covariance) counts as not preserved when the reference can be judged, as an empty stop does. The first metrics are:
  - `RelativeMeanError(feature)`: the realized relative error of the observation mean at the stop.
  - `RandomObjectRelativeError(feature)`: the expected relative RMSE had the same number of cells been drawn at random from the reference: simple random sampling with finite-population correction when the reference contains the stop (the full group), and without it against a held-out reservoir, whose FOVs the stop never sees (then any `n` is allowed, even more objects than the reservoir holds). It is the comparator in `data/crick_idr/fov_vs_objects.py`. Across permuted orders, the RMSE of `RelativeMeanError` divided by this value is the error inflation from acquiring whole FOVs.
  - `RareStateDetected(label_col, min_fov_occurrences)`: whether the stop prefix shows an independently labelled state in enough FOVs. It is `None` when the full acquisition never does, and when a label is missing.
  - Multivariate distribution outcomes in `morphocoverage.multivariate_outcomes`, over explicit `features`, optionally standardized by the reference's mean and SD: `CentroidError` (RMS difference of feature means), `CovarianceError` (relative Frobenius error), `QuantileError` (RMS difference of chosen quantiles), `FrechetError`, `EnergyError` and `MMDError` (U-statistics with a seeded subsample limit because they are quadratic; unbiased only against a reference disjoint from the stop, such as a reservoir, and for independent observations: against the full acquisition, which contains the stop, they are biased downward and descriptive only; with cells correlated within FOVs their null expectation is positive and shrinks as FOVs are added, and single values can be negative, so compare them across runs at matched effort, never against zero), and `SlicedWassersteinError` (seeded projections). The pure distance functions are in `morphocoverage.distances`. The plug-in metrics (centroid, covariance, quantile, Fréchet) are biased upward for small stops, so compare them at matched observation counts; the U-statistics are not when observations are independent, but cells within one FOV usually are not, so match FOV and observation counts for them too. All of them pool observations, so FOVs are weighted by their number of observations: they score the cell-weighted phenotype distribution, and a missed sparse FOV with different biology barely moves them. They are outcomes, not stopping rules.

- **Timing.** Every record carries estimator compute time, whole-decision latency up to the stop, and the median and maximum per-step decision latency. Only decisions up to the actual stop contribute to these summaries; later replay steps are retrospective and must not inflate the latency of a microscope that would already have stopped. `run_benchmark(..., incremental=True)` replays incremental estimators FOV by FOV, as online acquisition would. The `Timing` breakdown on each result separates estimator time from framework overhead.

Adequacy points are now defined (`morphocoverage.retrospective.adequacy_points`, in [Stopping rules](../live/stopping_rules.md#does-this-stopping-rule-work-calibration-and-adequacy-points)). Not yet implemented: excess FOVs beyond them in benchmark records, and aggregation into a regime map.

**Registered datasets** (`data/crick_idr/datasets.py`, a `BenchmarkRegistry`). idr0028 (committed subset: 12 wells × 28 FOVs on 3 plates) is the **development** dataset: every preset and definition of new was chosen on it, and it sets no protocol. idr0139 (10 wells × 4 FOVs on one plate, the JCS case study) is registered **held out**, its status per choice in [AGENTS.md](../../AGENTS.md) section 9, and `check_tuning` refuses a run that claims to be tuned on it. It is shallow (a 2-field reservoir judges 1 or 2 stream fields), so it can refute a development conclusion but not establish a field number; `heldout_idr0139.py` runs that check (`docs/preliminary_findings.md`, entry 7). Deeper held-out data are still needed.

## The score: fields saved at matched risk

`morphocoverage.retrospective.frontier` turns replayed acquisitions into one number per rule family, question and risk level, and keeps the curve behind it.

- **Risk** of a rule at one setting: the share of replayed acquisitions whose acquisition, ended at the rule's stop (or at the end of the stream when it never stopped), did not preserve the declared outcome in the held-back fields. Outcomes that could not be judged are left out of the share and counted.
- **Effort**: the mean number of fields acquired, the stop or the whole stream.
- **Baseline**: a fixed number of fields `k`, for every `k` up to the longest stream, computed from the same replays. It needs no data, so it is the null every rule must beat.
- **Chosen setting** at risk level `alpha`: the cheapest setting from which risk stays within `alpha` at every setting with more effort, so a lucky dip in a noisy curve is not taken as safe (the rule of `protocol_reliability`).
- **Score**: fields saved = the baseline's effort at `alpha` minus the family's. Zero is the null result; negative is worse than a fixed count. The approximate 95% interval resamples the declared independent units, never hold-outs or orders, and is widened for their number as every whole-field interval is; it is withheld below 5 units (a percentile interval over 2 units covers the truth only about half the time), or when fewer than 90% of the resamples reached `alpha` for both the family and the fixed rule, since the others would be a selected subset.
- **Adaptivity**: the Spearman correlation between where the rule stopped and where held-back fields confirmed the outcome lastingly. A rule that saves fields by stopping later on harder samples has a positive value.
- **Transfer**: the setting chosen on a development dataset to meet `alpha`, scored on a held-out dataset. A threshold is only as good as its risk on data it was not chosen on.

A family is either strategies replayed side by side (`family`) or one replayed configuration re-decided at many thresholds (`sweep`, which uses `redecide` on results kept with `replay_acquisitions(keep_results=True)`, so no estimator runs twice). Each family records how its reservoirs and orders were drawn, and families from replays that drew others are refused rather than scored against these reservoirs.

The benchmark on the committed data is `data/crick_idr/benchmark.py`. Its protocol (`benchmark_protocol.py`: questions, criteria, families and their threshold grids, risk levels, replay design, units) is locked by digest before the first run of each version (`benchmark_protocol.lock`, one line per version), and the runner refuses to run a changed protocol; a change needs a new version. Stage 1 compares each rule per object and per field; stage 2 compares the richer rules with those and with the fixed count, question by question (the **Benchmark** rows of the question tables). Results are in `docs/preliminary_findings.md`, entry 8.

## The `benchmark` workflow: which rule needs the fewest fields for the same risk?

```python
from morphocoverage.workflows import benchmark, default_families

bench = benchmark(objects, fovs, setup, reservoir_fraction=0.35)
print(bench.summary())                        # a table of the rules per criterion and risk level
print(bench.summary(detail="full"))           # every rule in sentences
```

Each criterion opens with its marker: ✅ when the rule that saves most
fields does so with an interval above 0, or when no rule beats the fixed
count (the null result) with an interval; ⚠️ when the largest saving is not
resolved or has no interval; ➖ when a fixed count never met the risk level
lastingly, so no rule is scored. With several rules, the one that saves most
is picked after looking at every rule's own interval: the headline says it
"saves the most here", a description of these data, not a test of the best
rule (owner, 2026-10-05).

`default_families(setup.readout)` holds the rule families and threshold grids
of the locked protocol (version 5, whose grids are those of versions 1 to 4): the mean-precision rule and the JCS descriptor
spread, each per object and per field, and the FOV bootstrap of the mean
(on the readout only, whatever other features are declared).
`default_families` takes `random_state` for its resampling rules; `benchmark`
passes `Setup.random_state`, and the locked protocol uses 0. Pass `families=`
to compare others. The grids are absolute (the JCS spread and the
bootstrap's standard error are in the readout's units), so a readout in other
units needs other grids: the summary warns for every family whose grid does
not bracket the data (it never stopped, or stopped at the first field it
could decide, at every setting) and says to widen it. The fields saved carry
an approximate 95% interval from resampling the independent units, withheld
below 5 units, where such an interval covers the truth far less often than
it says.

Tuning belongs on development data and scoring on held-out data
([`benchmarking.md`](benchmarking.md)). The workflow does not know a
dataset's role: keeping the two apart is up to you.

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

Runtime is a first-class benchmark because an intended application is smart microscopy. The retrospective analyses are offline, but they should still run in minutes: strategy comparisons, replays, reliability curves and representation controls take `n_jobs` to use several CPUs, and `docs/common/architecture.md` ("Performance and parallelism") describes what runs in parallel and how the results are kept identical to a serial run.

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
