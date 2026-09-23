# Benchmarking strategies

MorphoCoverage is a method-comparison framework. Candidate strategies must be evaluated using common criteria that are independent of any one strategy's internal score.

The proposed next-generation estimator and benchmark roadmap is documented in [`docs/proposals/candidate_estimators_roadmap.md`](proposals/candidate_estimators_roadmap.md).

## Primary scientific benchmark: deeply sampled real datasets

The primary benchmark will use real microscopy datasets acquired substantially deeper than routine experiments.

For a sample with FOVs (F_1,\ldots,F_n), strategies are evaluated on acquisition prefixes (F_1,\ldots,F_k) while later FOVs remain hidden. The hidden real FOVs are then used to measure what additional acquisition actually contributed.

The complete acquisition is an empirical reference reservoir, not assumed biological ground truth.

Synthetic data remain useful for unit tests, explicit counterexamples, and regression tests after a failure mode is discovered. They are not the primary scientific benchmark.

## Protocol as implemented

`morphocoverage.benchmark` and `morphocoverage.outcomes` fix how runs are recorded. They do not select a winner.

- **Dataset roles.** Every dataset is registered in a `BenchmarkRegistry` as `development` or `held_out`. A run declares the datasets its settings were tuned on (`tuned_on`). Those must be development datasets, and a run on one of them is recorded as in-sample. Performance claims belong to held-out or out-of-sample runs. `leave_one_dataset_out` yields folds when data are scarce. The Crick/IDR subset in `data/crick_idr/` is test data, not a benchmark dataset.
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
- **Outcome metrics** are functions of (full group, stop prefix) → value. They never see a strategy's evidence or decision, so two strategies that stop at the same prefix get identical outcomes; a contract test checks this. They are retrospective and must never be used inside an estimator. The first metrics are:
  - `RelativeMeanError(feature)`: the realized relative error of the observation mean at the stop.
  - `RandomCellRelativeError(feature)`: the expected relative RMSE had the same number of cells been drawn at random from the full group (simple random sampling with finite-population correction). It is the comparator in `data/crick_idr/fov_vs_cells.py`. Across permuted orders, the RMSE of `RelativeMeanError` divided by this value is the error inflation from acquiring whole FOVs.
  - `RareStateDetected(label_col, min_fov_occurrences)`: whether the stop prefix shows an independently labelled state in enough FOVs. It is `None` when the full acquisition never does.

- **Timing.** Every record carries estimator compute time, whole-decision latency up to the stop, and the median and maximum per-step decision latency. Only decisions up to the actual stop contribute to these summaries; later replay steps are retrospective and must not inflate the latency of a microscope that would already have stopped. `run_benchmark(..., incremental=True)` replays incremental estimators FOV by FOV, as online acquisition would. The `Timing` breakdown on each result separates estimator time from framework overhead.

Not yet implemented: excess FOVs beyond an empirical adequacy point (it needs an agreed adequacy definition), distribution-level outcome metrics, and aggregation into a regime map.

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

Runtime is a first-class benchmark because an intended application is smart microscopy.

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
