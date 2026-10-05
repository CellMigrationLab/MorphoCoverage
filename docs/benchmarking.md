# Benchmarking strategies

The questions being benchmarked are the rows marked **Benchmark** in `docs/questions.md` (R2, R3, P1, P5, P6, P10, O6); the other questions are measurements or diagnostics, reported but not ranked.

MorphoCoverage is a method-comparison framework. Candidate strategies must be evaluated using common criteria that are independent of any one strategy's internal score.

The candidate estimators not yet built, with the benchmark requirements they must meet, are listed in [`docs/proposals/candidate_estimators_roadmap.md`](proposals/candidate_estimators_roadmap.md).

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

Adequacy points are now defined (`morphocoverage.retrospective.adequacy_points`, below). Not yet implemented: excess FOVs beyond them in benchmark records, and aggregation into a regime map.

## Retrospective sufficiency of one dataset

`sufficiency_report(objects, fovs, feature_cols=..., strategies=..., objective=..., orders=..., novelty=..., novelty_target=..., calibration=...)` answers two questions per analysis group of an analysed dataset:

1. **Did each criterion say enough?** (`enough`) For each criterion, judged by its own threshold and not by held-out FOVs: `sampled_enough` (`yes`, `no` or `undefined` on the acquired FOVs; a fixed-effort rule, an `EffortPolicy` or a strategy flagged `effort_rule`, reports `target reached`, since reaching a fixed effort says nothing about the data), after how many FOVs it was first met across the declared acquisition orders (median and 10-90% range), and the FOVs imaged beyond that point. The recorded and the permuted orders are kept apart (`order_kind` in the stops table; the answer gives each kind in its own clause, for example "in 2 of 9 random orders, first met after ..."), never pooled into one median. A rule that cannot be judged yet gives the reason (`insufficient_reason`), and a rule's caveats (the FOV-size warning of the per-field mean-precision rule, the object-unit JCS assumption of random objects) are carried into its sentence (`warnings`, printed as `Caution:`). Verdicts hold only for each criterion and threshold, so compare several.
2. **Would one more FOV show something new?** (`next_fov`, given a `SupportNoveltyEstimator` that defines "new") The chance, its nominal upper bound (Clopper–Pearson on leave-one-FOV-out outcomes, which are not independent, so its coverage is approximate and is checked by the calibration below), and, for a `novelty_target`, the extra FOVs needed before the bound can fall below it: at least this many if none of them shows anything new, and this many if new FOVs keep appearing at the current rate. The target is also added as a criterion to question 1. `new_observations` lists the observations counted as new so they can be inspected in the images: debris or segmentation errors count as new too.

With `few_fovs`, it also counts the FOVs holding a state that fewer than `few_fovs` other FOVs show: a state recurring in two neighbouring FOVs is not new to the next FOV, yet not established either. It also states what the FOVs rule out: a state that no FOV has shown yet is present in at most `1 - (1 - confidence) ** (1 / k)` of FOVs (about `3 / k` at 95%; `unseen_state_max_frequency`), with the extra FOVs needed to rule out states above `novelty_target`. A small chance for the next FOV is not evidence that no rare state remains (preliminary findings, entry 3). `summary()` states both answers in short (the rules in one table across the groups, each caveat once); `summary(detail="full")` gives every number group by group, and the `answer` columns keep one sentence per group and rule. `final` and `stops` keep the full evidence and stop distributions.

The chance can be checked on the dataset itself (`calibration=CalibrationSettings(...)`, or `morphocoverage.calibration.calibrate_next_fov_novelty`). Random subsets of `k` FOVs predict the chance, and the remaining FOVs, judged against the same subset with the same definition, give the actual rate. A FOV shows something new when at least `min_novel_observations` (`m`) of its unsupported cells form one cluster (linked within `epsilon`), so scattered outliers do not add up to a phenotype. Support mirrors that rule: another FOV covers a cell only with at least `m` cells of its own within `epsilon`, so a single stray cell cannot cancel a new state. The two are not exactly symmetric, since support counts cells in a ball of radius `epsilon` while a new state is a single-linkage chain that can reach beyond it (the module docstring of `support_novelty` gives a case that is counted new in two FOVs although it recurs).

No committed script or results table holds this check on the committed data. The table below was computed on 2026-09-28, at the code of the third audit, with the call beneath it (it replaces a table from an unscripted run of 2026-09-24, before later changes to the calibration and to support novelty, whose values were close but not identical). On idr0028 (12 wells of 28 FOVs; 30 standardized intensity features; `epsilon = sqrt(30)`), pooled over wells with 20 draws each:

| FOVs imaged | `m = 2`: predicted | actual | mean nominal 95% bound | bound held | `m = 3`: predicted | actual | bound held |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 5 | 0.159 | 0.134 | 0.61 | 100% | 0.054 | 0.052 | 100% |
| 10 | 0.091 | 0.079 | 0.38 | 99.6% | 0.025 | 0.025 | 100% |
| 20 | 0.060 | 0.057 | 0.23 | 96% | 0.013 | 0.016 | 99.6% |

```python
# from the repository root, with data/crick_idr on sys.path
import numpy as np
import datasets
from morphocoverage.calibration import calibrate_next_fov_novelty
from morphocoverage.strategies import SupportNoveltyEstimator
from morphocoverage.transforms import Standardize

d = datasets.load("idr0028_screenB")
for m in (2, 3):
    novelty = SupportNoveltyEstimator(np.sqrt(30), 1, Standardize(), 2, m)
    table = calibrate_next_fov_novelty(
        d.objects, d.fovs, feature_cols=list(d.features), novelty=novelty,
        fov_counts=[5, 10, 20], draws=20, random_state=0,
        group_cols=list(d.groups), observation_unit="nucleus",
    )
    print(m, table.groupby("fovs_imaged")[["predicted", "actual", "bound", "bound_held"]].mean())
```

The prediction tracks the actual rate at these settings (at 20 FOVs with `m = 3` it is slightly below it). The bound is conservative here, which is not the same as calibrated: the mean bound sits far above the actual rate, and a bound that ignores the data, `1 - 0.05 ** (1 / k)`, would hold about as often, so "bound held" cannot tell the two apart; that needs settings where the actual rate approaches the bound. Each row pools 12 wells of 20 draws, so a share near 96% has a Monte Carlo standard error of at least about 1.3 points (more, since draws within a well are not independent), and one well alone far more. The fourth audit reproduced the table exactly at `random_state=0`, with similar values at other seeds; it is still not tracked as a script and results table. The unscripted run of 2026-09-24 also found that the definition moves the answer a lot: counting any unsupported cells rather than a cluster gave much higher rates, mostly from scattered cells; letting one cell of another FOV support a cluster gave rates near zero, because stray cells cancelled real groups; and at half the distance the bound held less often. Those variants were not rerun, so their numbers are not given here. With no new FOVs among `k` the bound is `1 - 0.05 ** (1 / k)` (0.14 at 20 FOVs) whatever the data. Random subsets treat FOVs as exchangeable, so this does not test drift along the recorded order, and the actual rate is measured on few remaining FOVs when `k` is close to the group size.

In the calibration, a group too small for a count gets a row with `draws=0` and its `reason` rather than being dropped; a criterion passed twice to `sufficiency_report` is refused. It is self-assessment of one dataset; whether a criterion is right in general is a benchmark question.

## How many FOVs next time? Sampling reliability

`morphocoverage.retrospective.sampling_reliability` answers the planning question without any stopping rule (P1 of `docs/questions.md`). For each analysis group and outer split it holds back a fixed reservoir of whole FOVs (`ReservoirSplit`), draws `k` random FOVs from the rest, and assesses each declared `OutcomeCriterion` (an outcome metric with a tolerance and direction, for example the well mean within 5%) against the same reservoir for every `k`. The curve gives, per `k`, the share of draws in which the criterion was preserved (averaged over splits, with its spread and the Monte Carlo standard error of that mean, `preserved_mc_se`), the metric's median and 10–90% range, and the gain from the previous `k`. The ceiling table gives how often the whole stream preserved the criterion: the reservoir is itself a finite sample, so the curve cannot be read above it, and its own error adds to that of the `k` fields, so the curve understates how often `k` fields preserve the sample's true value and a field count read from it errs high. `summary()` states the curve in sentences. Draws and splits describe one sample and are not independent replicates; statements across samples count samples or repeats (P2). With few blocks the splits repeat hold-outs, which differ only by the random draws, so the summary says how many of the splits were distinct hold-outs. A group too small to split (fewer than two FOVs or blocks) is skipped and listed with its reason in `ReliabilityResult.skipped` (and `ReplayResult.skipped` for replays); the other groups are analysed as usual. `SamplingHierarchy` declares which columns are the FOV, the sample and the independent unit; nothing is inferred from column names.

## Does this stopping rule work? Calibration and adequacy points

A stopping rule can depend on the order in which FOVs arrive, so it is checked sequentially, separately from sampling reliability (P6 for the calibration; R2 and P4 for the adequacy points). `morphocoverage.retrospective.replay_acquisitions` holds back the same fixed reservoir per split as `sampling_reliability` and replays the stream in the recorded order (when an acquisition-order column is given) and in seeded random orders (10 by default). After every FOV it records each strategy's decision, computed from the visible prefix only, and, separately, whether the prefix preserves each `OutcomeCriterion` against the reservoir. Outcomes never depend on a strategy.

- `stopping_calibration` finds each rule's first stop and reports, over replayed acquisitions, how often it stopped, the median FOVs at the stop (with its Monte Carlo standard error over the reservoir splits, each split's median stop counting once, since its orders share one reservoir; owner, 2026-10-05), and the shares of correct stops (preserved at the stop), stops that could not be judged, and runs that never stopped, with whether those had it by the last FOV. The headline risk is the share **not preserved at the stop**. It splits into premature stops (the whole stream was not shown to fail the criterion) and stops the whole stream did not fix either (not preserved by every stream FOV together). The held-back reference is itself a finite sample, so its noise alone can put the whole stream outside the tolerance: the premature share is a lower bound on the stops that came too early, and the total not preserved at the stop an upper bound; the summary prints both. For a fixed-FOV rule in random orders the share of correct stops estimates the same quantity as `sampling_reliability` at that `k`; a test checks this.
- `adequacy_points` gives, per replayed acquisition, the first and the sustained apparent adequacy (from decisions only; a later continue, unsupported or undefined decision breaks a sustained stop) and the first and the sustained realized adequacy (from outcomes only, for all criteria together and each on its own; a later failure breaks it, an undefined outcome is skipped). Sustained realized adequacy is the preferred answer to "how many FOVs did this experiment really need?". For example, decisions that say stop at FOV 16, stay stop to 23, continue at 24 and stop from 25 on give a first apparent adequacy of 16 and a sustained one of 25; outcomes preserved at 10 and 11, not at 12 and from 13 on give a first realized adequacy of 10 and a sustained one of 13. The apparent points use only decisions and the realized points only outcomes; they are never mixed.

Both have a `summary()` in plain sentences. Replayed acquisitions are hold-outs and orders of the same FOVs, not independent experiments.

## Across samples, fields to inspect, controls, effort and order

The remaining retrospective analyses (P2, R5, R1, P10, P7, P8 and the assembled report) each answer one question and have a plain-language `summary()`:

- `protocol_reliability(per_sample, hierarchy, threshold=..., strata=...)` counts, from per-sample reliability curves, the samples adequately represented at each number of FOVs (reliability at least `threshold`, provisional preset 0.9), with explicit denominators, per independent unit, each sample's difficulty, and a leave-one-unit-out check of a number chosen on the other units. Intervals resample the declared independent units, so more draws or splits never narrow them, and are never narrower than the exact Clopper-Pearson interval over the units in whole counts, each unit a success when at least `protocol_share` of its samples are adequate and a failure otherwise (`scipy.stats.binomtest`; owner, 2026-10-05). When every unit or none is adequate, resampling would give zero width, and the interval is that exact one (5 of 5 units is compatible with a true share of about 48%); more samples inside one unit never narrow it. Samples whose stream is too short for a `k`, and criteria that could not be judged, are counted apart (`beyond_stream`, `undefined`); with fewer than five units the summary says the uncertainty is large. Pooling across conditions is an explicit `strata=[]`.
- `inspect_fovs` lists fields to reopen, one reason per line with its numbers: a new state (support novelty), influence on a declared FOV-weighted summary (mean or an interpolated quantile), an observation count far from the median, or a difference from the nearest stage neighbours. `novelty_cols` names the features the new-state reason is judged on, so the other reasons can read the readout. There is no combined score and no FOV is called an artifact.
- `representation_control` judges declared positive-control FOVs (and positive cells planted into reference FOVs) against random reference FOVs holding observations, paired within `pair_cols`: detection and false-alarm rates, "positive only" without a reference, "not assessed" without a control. It also measures both rates leaving one field out within the analysis groups of the inspection (`lofo_group_cols`), which the report's "expected by chance" clause and its detection-adjusted bound read.
- `effort_projection` estimates nested variance components of a scalar mean (Henderson's method I for unbalanced designs) with a hierarchical bootstrap, gives the standard error of the acquired design from its actual sample and repeat sizes, and projects the standard error of balanced allocations at equal total FOVs, compared with the acquired allocation made balanced (the cost of the acquired imbalance is stated apart); a difference between allocations is stated only when its interval excludes 1 and there are at least five independent units. Where levels cannot be separated (a single repeat, or one sample per repeat) the repeat level is left out of that stratum and the summary says so; a stratum that cannot be estimated at all is listed in `EffortResult.skipped` with its reason, and the other strata are still estimated. With `Setup.normalize`, the projection reads raw values, so normalising to the control does not remove the between-repeat variance. The bootstrap intervals are approximate: in a simulation with 8 samples their coverage was 80–88% for the sample and field components, not 95%, measured over 40 runs (a Monte Carlo standard error of about 5 to 6 points).
- `order_efficiency` compares recorded, random and dispersed (`AcquisitionOrder.dispersed(seed)`, greedy farthest point on stage coordinates) orders of the same positions by sustained realized adequacy and stage travel, and refuses adequacy points from another replay; `redundancy_distance` gives the distance below which neighbouring fields are more alike than random pairs (a group too small to measure gets a row with its `reason`).
- `microscopist_report` assembles the answers in the order of the questions of `docs/questions.md`, with a sampling profile of measured quantities; a question without its analysis says "not assessed" and what it needs.

**Registered datasets** (`data/crick_idr/datasets.py`, a `BenchmarkRegistry`). idr0028 (committed subset: 12 wells × 28 FOVs on 3 plates) is the **development** dataset: every preset and definition of new was chosen on it, and it sets no protocol. idr0139 (10 wells × 4 FOVs on one plate, the JCS case study) is registered **held out**: never used to choose a setting, so `check_tuning` refuses a run that claims to be tuned on it. It is shallow (a 2-field reservoir judges 1 or 2 stream fields), so it can refute a development conclusion but not establish a field number; `heldout_idr0139.py` runs that check (`docs/preliminary_findings.md`, entry 7). Deeper held-out data are still needed.

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

The benchmark on the committed data is `data/crick_idr/benchmark.py`. Its protocol (`benchmark_protocol.py`: questions, criteria, families and their threshold grids, risk levels, replay design, units) is locked by digest before the first run of each version (`benchmark_protocol.lock`, one line per version), and the runner refuses to run a changed protocol; a change needs a new version. Stage 1 compares each rule per object and per field; stage 2 compares the richer rules with those and with the fixed count, question by question (`docs/questions.md`, the **Benchmark** rows). Results are in `docs/preliminary_findings.md`, entry 8.

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

Runtime is a first-class benchmark because an intended application is smart microscopy. The retrospective analyses are offline, but they should still run in minutes: strategy comparisons, replays, reliability curves and representation controls take `n_jobs` to use several CPUs, and `docs/architecture.md` ("Performance and parallelism") describes what runs in parallel and how the results are kept identical to a serial run.

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
