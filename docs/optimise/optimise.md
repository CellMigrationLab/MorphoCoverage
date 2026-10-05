# Optimise: what should the next experiment do?

OPTIMISE answers "What should I do next time?" from completed datasets, judging
subsets of fields on fields held back. Use it after a report, to plan how many fields,
wells and repeats the next experiment needs. This page is the `optimise` workflow, its
questions P1 to P10, and the analyses on held-back fields behind them; how rules are
scored against each other is [Benchmarking](benchmarking.md).

## The questions

This is the experimental-design part of MorphoCoverage. A number here is a
recommendation only when it rests on what representative samples of FOVs
preserved in FOVs they did not include, and only once it holds on samples it
was not chosen on.

| # | Question | Answered by | Status |
|---|---|---|---|
| P1 | **How many FOVs should I acquire per sample next time?** At 10, 15, 20, 25 FOVs, how often does representative sampling preserve the required information in unseen FOVs? | `sampling_reliability` against a fixed held-back reservoir, with its ceiling | done within a sample. **Benchmark** (a method's recommended number against the realized need) |
| P2 | **Does that recommendation generalize across samples and biological repeats?** Not "20 worked in this well", but "20 worked in 75% of held-out samples; 25 in 93%". | `protocol_reliability`, counted by independent unit, with a leave-one-unit-out check; held-out datasets (`data/crick_idr/datasets.py`) | partial: implemented; needs at least 5 independent units and deeper held-out data |
| P3 | **What did the additional imaging buy?** From 5 to 10 FOVs, 10 to 20, 20 to 30. | the `gain` column of the reliability curve; adequacy points | partial: the gains are computed but not printed as increments |
| P4 | **When did additional imaging become mostly redundant?** At the same point for common and rare phenotypes? | sustained realized adequacy per criterion, with its 90th percentile | partial: rare phenotypes need independent labels |
| P5 | **Would a simple sampling rule have been sufficient?** A fixed number of FOVs, a fixed area, a fixed object count, the JCB/Zweifach rule, the JCS descriptor rule? | the packaged baselines, replayed and calibrated; fields saved at matched risk (`retrospective.frontier`, `data/crick_idr/benchmark.py`); whether the JCB and JCS object-count rules hold when their objects come as whole fields (`sampling_unit.object_rule_check`, `descriptor_spread_check`, as in the Foundation) | done for the mean on idr0028, with transfer to idr0139; partial for planted rare states (semi-synthetic, and only 2 independent plates, too few for an interval on the fields saved); the distribution questions need deeper data. The object-rule checks are exploratory (`docs/preliminary_findings.md`, entry 9), not benchmark results. **Benchmark** |
| P6 | **How reliable is a proposed automated stopping rule?** When it says STOP: how often is it correct, how often premature, how much acquisition does it save? This is ONLINE 6 seen from the planning side. | `stopping_calibration` on replayed acquisitions; savings and risk at matched effort (`retrospective.frontier`) | done. **Benchmark** |
| P7 | **Am I repeatedly imaging essentially the same thing?** How strong is spatial redundancy, and over what distance? | `redundancy_distance`, FOV semivariogram; a group is flagged only beyond a null that shuffles the stage positions among its fields | done, as a diagnostic, not an adequacy claim |
| P8 | **Could I acquire the same information more efficiently?** Would random or spatially dispersed orders beat the recorded raster, and at what cost in stage travel and time? | `order_efficiency` with recorded, random and dispersed orders | done |
| P9 | **Were my selected locations representative of the specimen?** | needs every candidate position from an overview (a new input contract) | gap |
| P10 | **Should I spend effort on more FOVs or more independent samples?** Is the remaining uncertainty within FOVs, between FOVs, between samples or between repeats, and would the next unit of effort be better spent on a FOV, a sample or a repeat? | `effort_projection` on a declared hierarchy, or on the one `Setup` derives from a `technical_repeat` column (field within technical repeat within repeat), at equal total FOVs; JCB multi-sample planning | partial: implemented; needs at least 5 independent units; no cost model yet. **Benchmark** (projection against JCB planning, by simulation) |

## The `optimise` workflow

```python
from morphocoverage.workflows import document, optimise

plan = optimise(objects, fovs, setup, fov_counts=[2, 4, 8, 12], reservoir_fraction=0.35)
print(plan.summary())                         # questions 4, 5, 7, 8 and the rule calibration
print(plan.summary(detail="full"))            # every sentence and number
print(document(result, plan).summary())       # one report with every question
```

The plan opens with its own title and an **At a glance** block, which adds
the rule check ("How reliable was each stopping rule?", ⚠️ when some rule
stopped where a criterion was not preserved in held-back fields in more than
10% of the acquisitions of a group and order kind); the rule check's section
comes before "What this dataset cannot tell me", and the settings close it.

`document` refuses a report and a plan computed under different setups
(`n_jobs` aside), since the declaration it prints first must be the one every
answer was computed under; each part's settings sentence is kept among the
closing notes.

When the FOV table has a `technical_repeat` column and no `hierarchy` is
declared, `optimise` derives one (the field within the technical repeat
within the repeat, one stratum per condition) and answers question 7, "more
fields or more technical repeats?", from the variance between technical
repeats and between fields at equal total fields; where every repeat holds a
single technical repeat the repeat cannot be a variance level and the answer
says so. On idr0028, with 28 fields per well, most of the uncertainty of a
condition's mean sits between wells wherever there are two to see it.
A declared `hierarchy` also keeps one stratum per condition unless
`effort_strata` names others; `effort_strata=()` pools the conditions, as an
explicit request. Protocol reliability across samples needs every sample and
stratum to be an analysis group: when `group_cols` pool one (the repeat, the
technical repeat, the condition), that part of question 4 says "not assessed"
and why, as question 7 does when `group_cols` leave the technical repeat out.

`reservoir_fraction` has no default. It sets the share of each group's fields
held back as the reference that subsets are judged against, and that share is
a design choice. Planning advice rests on what random subsets preserved in
those held-back fields. It never rests on the best subset. Every declared
criterion gets its own curve: the mean within each tolerance, each fingerprint
criterion, and the labelled rare state, so the answer to "how many fields?"
is read per target, and a target the held-back fields cannot judge says so.

## Held-back outcomes

`morphocoverage.retrospective.sampling_reliability` answers the planning question without any stopping rule (P1 above). For each analysis group and outer split it holds back a fixed reservoir of whole FOVs (`ReservoirSplit`), draws `k` random FOVs from the rest, and assesses each declared `OutcomeCriterion` (an outcome metric with a tolerance and direction, for example the well mean within 5%) against the same reservoir for every `k`. The curve gives, per `k`, the share of draws in which the criterion was preserved (averaged over splits, with its spread and the Monte Carlo standard error of that mean, `preserved_mc_se`, clustered on the hold-out: splits that held back the same fields count as one, so it stops shrinking once the distinct hold-outs are used up; owner, 2026-10-05), the metric's median and 10–90% range, and the gain from the previous `k`. The ceiling table gives how often the whole stream preserved the criterion: the reservoir is itself a finite sample, so the curve cannot be read above it, and its own error adds to that of the `k` fields, so the curve understates how often `k` fields preserve the sample's true value and a field count read from it errs high. `summary()` states the curve in sentences. Draws and splits describe one sample and are not independent replicates; statements across samples count samples or repeats (P2). With few blocks the splits repeat hold-outs, which differ only by the random draws, so the summary says how many of the splits were distinct hold-outs. A group too small to split (fewer than two FOVs or blocks) is skipped and listed with its reason in `ReliabilityResult.skipped` (and `ReplayResult.skipped` for replays); the other groups are analysed as usual. `SamplingHierarchy` declares which columns are the FOV, the sample and the independent unit; nothing is inferred from column names.

The other analyses on held-back fields each answer one question and have a plain-language
`summary()`:

- `effect_preservation(objects, fovs, feature=..., control=..., fov_counts=..., tolerance=0.2)` asks, for one feature, whether `k` fields a side preserve each biological repeat's difference from the control (#81, phase 1; owner, 2026-10-05). The effect is the per-repeat Cohen's d on the observations that the condition analysis reports. In each of `n_splits` splits, half of each side's fields (`reservoir_fraction`) are held back at random, and their d is the reference, the same for every `k`; `k` fields a side are drawn from the other fields `n_draws` times, and a draw preserves the difference when its d has the reference's sign and lies within `tolerance` of it (in units of d; the preset 0.2 is Cohen's "small" effect, printed with the result). A split whose held-back d has a 95% interval including 0 (the field bootstrap of question 2), or fewer than two held-back fields with observations on a side, gives no reference; a `k` without any such split is **not evaluable**, never "not preserved". The table gives the share per repeat, condition and `k`, the draws judged and a status; a `workflows.Setup` passed as `setup` fills the feature (its readout), the control and the columns. It is a standalone analysis: `optimise` and the report do not use it until the owner settles the multi-group outcome contract.
- `protocol_reliability(per_sample, hierarchy, threshold=..., strata=...)` counts, from per-sample reliability curves, the samples adequately represented at each number of FOVs (reliability at least `threshold`, provisional preset 0.9), with explicit denominators, per independent unit, each sample's difficulty, and a leave-one-unit-out check of a number chosen on the other units. Intervals resample the declared independent units, so more draws or splits never narrow them, and are never narrower than the exact Clopper-Pearson interval over the units in whole counts, each unit a success when at least `protocol_share` of its samples are adequate and a failure otherwise (`scipy.stats.binomtest`; owner, 2026-10-05). When every unit or none is adequate, resampling would give zero width, and the interval is that exact one (5 of 5 units is compatible with a true share of about 48%); more samples inside one unit never narrow it. Samples whose stream is too short for a `k`, and criteria that could not be judged, are counted apart (`beyond_stream`, `undefined`); with fewer than five units the summary says the uncertainty is large. Pooling across conditions is an explicit `strata=[]`.
- `effort_projection` estimates nested variance components of a scalar mean (Henderson's method I for unbalanced designs) with a hierarchical bootstrap, gives the standard error of the acquired design from its actual sample and repeat sizes, and projects the standard error of balanced allocations at equal total FOVs, compared with the acquired allocation made balanced (the cost of the acquired imbalance is stated apart); a difference between allocations is stated only when its interval excludes 1 and there are at least five independent units, and the summary names the smallest standard error that resolved here: it is picked after looking at every allocation's own interval, so it describes these data and is not tested as the best allocation (owner, 2026-10-05). Where levels cannot be separated (a single repeat, or one sample per repeat) the repeat level is left out of that stratum and the summary says so; a stratum that cannot be estimated at all is listed in `EffortResult.skipped` with its reason, and the other strata are still estimated. With `Setup.normalize`, the projection reads raw values, so normalising to the control does not remove the between-repeat variance. The bootstrap intervals are approximate: in a simulation with 8 samples their coverage was 80–88% for the sample and field components, not 95%, measured over 40 runs (a Monte Carlo standard error of about 5 to 6 points).
- `order_efficiency` compares recorded, random and dispersed (`AcquisitionOrder.dispersed(seed)`, greedy farthest point on stage coordinates) orders of the same positions by sustained realized adequacy and stage travel, and refuses adequacy points from another replay; `redundancy_distance` gives the distance below which neighbouring fields are more alike than random pairs (a group too small to measure gets a row with its `reason`). The report counts such a group apart, never as a group without redundancy: "in 1 of the 2 groups assessed ...; 1 more not assessed", with the reason (owner, 2026-10-05).
