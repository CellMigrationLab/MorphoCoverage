# Benchmarking strategies

MorphoCoverage is a method-comparison framework. Candidate strategies must be evaluated using common criteria that are independent of any one strategy's internal score.

## Primary scientific benchmark: deeply sampled real datasets

The primary benchmark will use real microscopy datasets acquired substantially deeper than routine experiments.

For a sample with FOVs (F_1,\ldots,F_n), strategies are evaluated on acquisition prefixes (F_1,\ldots,F_k) while later FOVs remain hidden. The hidden real FOVs are then used to measure what additional acquisition actually contributed.

The complete acquisition is an empirical reference reservoir, not assumed biological ground truth.

Synthetic data remain useful for unit tests, explicit counterexamples, and regression tests after a failure mode is discovered. They are not the primary scientific benchmark.

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
- cells/objects acquired at stopping;
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

- fixed object/cell count;
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
- number of objects;
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
