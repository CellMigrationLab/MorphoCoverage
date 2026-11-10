# MorphoCoverage

MorphoCoverage is a feature-agnostic Python project for studying whether microscopy experiments have sampled a declared phenotypic representation adequately. It treats **fields of view (FOVs), defined as independently selected spatial acquisition positions, rather than individual observations/cells or z-planes, as the primary operational acquisition units** and asks whether additional representative imaging is likely to provide useful new phenotypic information.

## Questions MorphoCoverage is built to answer

The canonical list, grouped by the three moments of use (REPORT: what can I trust in a finished acquisition; OPTIMISE: what should I do next time; LIVE: should I acquire another field now) with what information may answer each question and its status, starts at [`docs/README.md`](docs/README.md). The table below maps the main questions to the functions that answer them.

For a completed experiment, the aim is to tell a microscopist whether the imaging effort was sufficient, what the additional FOVs actually bought, and how to spend the imaging budget in the next experiment. Every answer depends on the declared phenotype representation (the features and the definition of "new") and is stated with the criterion behind it.

| Question | Today |
| --- | --- |
| Foundation: why count fields, and not only objects? How variable are my fields, and would the published object-count rules have held with whole fields? | `morphocoverage.sampling_unit`: `fov_variability` (design effect, fields that differ more than random objects would), `object_rule_check` and `descriptor_spread_check` (the JCB and JCS rules with random objects and with whole fields) |
| Are rare or unusual objects clustered in a few fields? | `morphocoverage.sampling_unit`: `unusual_objects` against declared control groups, `fov_clustering` on any state labels (field incidence against random objects, fields needed) |
| Can my features detect the phenotype I care about? | `morphocoverage.retrospective.representation_control`: detection and false-alarm rates for declared positive and reference controls, or "not assessed" |
| Did I image enough, for common and for rare phenotypes? | `sufficiency_report`: criteria met, first stop points across acquisition orders, rare-state statements |
| Would one more FOV show something new, and is that chance right? | `sufficiency_report` next-FOV chance and nominal bound, flagged cells, `calibrate_next_fov_novelty` |
| What is limiting my sampling? | the sampling profile of `microscopist_report`: control detection, design effect, where the variance of the mean sits, spatial correlation distance, fields holding new states |
| How many FOVs should I image next time? | `morphocoverage.retrospective.sampling_reliability`: how often k random fields preserve what you declared (for example the well mean within 5%) in fixed held-back fields, per sample; `protocol_reliability` across samples, counted by independent unit |
| What did the extra FOVs buy? | `morphocoverage.retrospective.adequacy_points`: when a rule first and lastingly said stop, and when held-back fields confirmed it was enough, first and lastingly |
| Can I trust an automated stopping rule? | `morphocoverage.retrospective.stopping_calibration`: how often a rule's stop was too early for what you declared, in held-back fields, over replayed orders |
| Which FOVs and cells should I inspect? | `inspect_fovs`: fields to reopen, each with its reason and numbers (new state, influence on a summary, observation count, difference from stage neighbours), image file and stage position |
| More FOVs, or more biological samples? | `effort_projection`: nested variance components of a scalar mean and projected standard errors at equal total FOVs; "cannot tell" unless the intervals separate |
| Could I have sampled more efficiently? | `order_efficiency`: recorded, random and dispersed orders of the same positions with stage travel; `redundancy_distance`; a comparison with all candidate positions is not implemented |
| Do the conditions differ, and how confidently? | `morphocoverage.sampling_unit.condition_contrast`: across the biological repeats, the direction in every repeat, Cohen's d and the paired t-test on the repeat averages; the lab's Plot&Stats cell-level randomization test on Cohen's d, labelled with its chance of a positive without an effect and balanced when the repeats are not; both on raw and on normalised values (assumptions and limitations: [`docs/report/conditions.md`](docs/report/conditions.md)); within each repeat, Cohen's d and the JCS effect size on objects and with the field as the unit, and a randomization test over whole technical repeats (each side at least two) or, without a `technical_repeat` column, over whole fields |

A completed dataset may be represented by a few well-chosen FOVs, while a new experiment that samples FOVs at random needs more: **retrospective representability is not the prospective sampling requirement.** `morphocoverage.retrospective.microscopist_report` assembles these answers question by question ([`docs/report/questions.md`](docs/report/questions.md)). Underneath, several candidate estimators answer these questions; none is assumed to be universally correct.

## Modes

MorphoCoverage has three workflows, one per moment of use ([`docs/README.md`](docs/README.md)); each fixes what information may answer its questions. BENCHMARK belongs to OPTIMISE: it compares methods on those questions and answers none of its own.

| Mode | The question | May use | Entry points |
| --- | --- | --- | --- |
| **REPORT** | I have finished imaging. What can I trust, and what can't I? (opened by the Foundation: why count fields, and not only objects?) | every acquired FOV of this dataset; declared control groups | `morphocoverage.retrospective.microscopist_report`; `morphocoverage.sampling_unit` (`fov_variability`, `object_rule_check`, `descriptor_spread_check`, `unusual_objects`, `fov_clustering`); `sufficiency.sufficiency_report`; `representation_control`, `inspect_fovs`, `calibrate_next_fov_novelty` |
| **OPTIMISE** | What should I do next time? | completed datasets, judged on held-back FOVs | `morphocoverage.retrospective`: `sampling_reliability`, `protocol_reliability`, `effect_preservation` (whether k fields a side keep each repeat's difference from the control), `adequacy_points`, `stopping_calibration`, `effort_projection`, `order_efficiency`, `redundancy_distance`; `plan_objects_to_average` |
| **LIVE** | I am imaging now. Should I acquire another field? | only the FOVs acquired so far | `workflows.live_session` with `LiveFolder` or `LiveTables`; `OnlineSession.add_fov`, with any `SamplingStrategy`; the same estimators and policies as replay |
| **BENCHMARK**, in OPTIMISE | Which stopping rule needs the fewest fields for the same risk of stopping too early? | development datasets for tuning, held-out datasets for scoring | `morphocoverage.retrospective.frontier` (fields saved at matched risk), `replay_acquisitions`, `morphocoverage.benchmark`; the locked protocol in `data/crick_idr/benchmark.py`; [`docs/optimise/benchmarking.md`](docs/optimise/benchmarking.md) |

MorphoCoverage reads two linked tables, one row per acquired field and one row per observation; their mandatory and optional columns are in [`docs/common/data_format.md`](docs/common/data_format.md), and `morphocoverage.check_tables(objects, fovs, feature_cols=[...])` lists every problem in a dataset, with where it is and how to fix it. Features measured plane by plane are turned into one feature per plane first (`morphocoverage.planes_as_features`: `area` becomes `area_z0`, `area_z1`, ...), and those names are declared. `morphocoverage.workflows.scan_design(objects, fovs)` scans the design first: what the dataset is, whether it was loaded correctly (copies, balance over the repeats), which columns would say more, and the questions to answer before the analysis; `workflows.guide()` (or `python -m morphocoverage.workflows cells.csv --folder results`) asks them one at a time, saves the answers in the results folder so a later session can continue or start over, and runs the report, written as one HTML page (`workflows.html_report`: every answer, the figures, an interactive PCA and UMAP, a glossary of every test with its papers ([`docs/common/glossary.md`](docs/common/glossary.md)), and every file inside it, so the page alone is what to share), with the whole folder also zipped. To run a new dataset through the workflows, declare it once in a `morphocoverage.workflows.Setup`, including what the acquisition must preserve (a population summary, a phenotypic fingerprint, a rare phenotype) and whether to normalise every feature to the control condition's mean per repeat; every document opens with what is in the dataset and that declaration, and closes with what the dataset cannot tell. With several features, question 9 ranks the features that separate the conditions by the cell-level test (Benjamini-Hochberg gated, by |d|), with the paired test on the repeat means beside it as the second gate; `workflows.embedding_plots` draws the PCA (the best-ranked features as arrows) and UMAP of all of them. `workflows.plots` draws PlotsOfData-style figures (biological repeats side by side, every object a point in its condition's colour, shaped by technical repeat, each repeat's median with its interval from resampling technical repeats and then whole fields). Then call `report`, `optimise`, `live_session` with `LiveFolder` (files on disk) or `LiveTables` (tables in memory), or `benchmark`. Each returns a `summary()` in sentences ([`docs/README.md`](docs/README.md)): [running a report](docs/report/running_a_report.md), [optimise](docs/optimise/optimise.md), [the live session](docs/live/live_session.md), [benchmarking](docs/optimise/benchmarking.md). The notebook [`notebooks/MorphoCoverage_Workflows_Example`](notebooks/MorphoCoverage_Workflows_Example/MorphoCoverage_Workflows_Example.ipynb) runs the three workflows in turn, REPORT, OPTIMISE with its benchmark, and LIVE, on the example dataset or your own tables, locally or on Google Colab.

The report opens with the foundation every other answer rests on: a microscope acquires phenotype observations through spatial fields, not as an arbitrary random sample of individual objects, and whether that matters for a dataset is measured, not assumed (the Foundation section of [`docs/report/questions.md`](docs/report/questions.md); plain-language results: [`docs/report/objects_or_fields.md`](docs/report/objects_or_fields.md)). Everything except LIVE is retrospective.

## Project boundary

MorphoCoverage consumes numeric phenotype feature tables together with FOV identity and optional grouping metadata. A feature row is a generic **observation**: it may be a cell, nucleus, patch, slice-level measurement, FOV summary, embedding, or another declared unit.

It does **not**:

- read microscopy images or segmentation masks;
- extract morphometric, texture, intensity, or deep-learning features;
- depend on CellPatcher, Nyxus, DINO, Torch, or another feature extractor;
- replace the user's biological inference across independent repeats;
- control a microscope.

It does report, when a control condition is declared, how the conditions differ in the acquired fields (effect sizes and a randomization test with the field as the unit, beside the object-level numbers of the lab's Plot&Stats notebook), and whether a stop at `k` fields would have preserved that difference is a retrospective benchmark outcome. Those other concerns belong upstream or downstream. MorphoCoverage focuses on the statistical problem of **FOV-aware phenotypic sampling adequacy**. The mathematical solution is deliberately open: candidate estimators must be compared against simple baselines and unseen FOVs before any one approach becomes the default.

## Why FOVs matter

Microscopes acquire phenotype observations through spatial fields, not as an arbitrary random sample of individual cells. Five hundred cells from two FOVs may contain very different sampling information from five hundred cells distributed across twenty FOVs. How different depends on two things: the share of the variance that lies between fields and the number of objects a field holds (the design effect of a whole-field mean is about `1 + (m - 1) * ICC`), so a small between-field share with hundreds of cells per field can still make a cell count badly understate the sampling error. Whether that difference is important for a dataset is a central hypothesis to test, not an assumption to protect. Cells remain the phenotype observations; fields describe how they were sampled.

MorphoCoverage therefore preserves the hierarchy

```text
observation (cell / nucleus / patch / slice / FOV summary / embedding)
    -> FOV = spatial acquisition position
    -> optional sample/repeat/condition
```

and never treats cell count alone as a complete description of acquisition effort. FOVs themselves are not assumed independent; stage coordinates, physical area, and sampling design can matter.

## What the report looks like

`workflows.report(objects, fovs, setup).summary()` opens with a verdict per question, then one short
section per question: its answer, a table, the cautions found in these data (⚠️) and the standing
assumptions it rests on. `summary(detail="full")` gives every sentence and number. Markers: ✅ fine,
⚠️ look at this, ❌ a problem that changes how to read the answers, ➖ not assessed. An excerpt of the
idr0028 report, as `workflows.report` gives it
(written by `data/crick_idr/report_steps_idr0028.py` as `microscopist_report_idr0028.txt`; the full form is
`microscopist_report_idr0028_full.txt`):

```text
Microscopist's report — 9550 nuclei in 336 fields

At a glance
  Key: ✅ yes, or a clear answer · ⚠️ partly, a trend only, or something to look at · ❌ check the
    data first · ➖ no conclusion possible here. The word beside each marker says which.
  ⚠️ The data — data checks: no warning; integrity: nothing found; acquisition: 3 fields far from
     their condition and repeat
  ✅ Foundation. Why count fields? — a mean from whole fields varies 2.1x as much as from random
     nuclei (median); the mean-precision count rule still held with whole fields in all 12 wells
  ⚠️ 1. Can the features see the phenotype? — YAP fields recognized in 100% of checks; false alarms
     7%, both pooled over all biological repeats; not judged in biological repeat 2B (the positive
     control was not imaged here) (exploratory)
  …

⚠️ 2. Did I image enough?
  Each feature alike, 31 features, ordered as in “Do the conditions differ?” (significant across
    repeats first); the rule of the last column: the mean-precision rule, per nucleus, 5%:
    feature                                             difference from the control shown  to show the rest                             wells whose mean meets the rule
    Nuclei_Intensity_MedianIntensity_YAPTAZ             partly: shown in 4 of 6            about 44 fields a side; 1 too small to show  12 of 12 wells
    Nuclei_Intensity_MeanIntensity_YAPTAZ               partly: shown in 4 of 6            about 63 fields a side; 1 too small to show  12 of 12 wells
    …
```

## Current status

MorphoCoverage is a strategy-comparison framework. No default coverage estimator has been chosen.

- **Input validation.** Linked FOV and phenotype-observation tables with MorphoBricks defaults. `position_id` is the preferred explicit spatial-position identifier; `image_file` is a fallback only when one file corresponds to one position. `condition`, `biological_repeat` (the former name `repeat` is refused) and, when present, `technical_repeat` are separate analysis groups that stay part of FOV identity even when pooled. Feature columns are always explicit, acquired FOVs with zero observations are kept, and FOV-level grouping metadata can be inherited from the FOV table when the position identifier is unambiguous; and `z`, `c` and `t` are within-position metadata: they are refused as features, and a feature measured per plane becomes one feature per plane with `planes_as_features`. `observation_unit` records what one feature row means. NaN feature values may be loaded but are never silently imputed or dropped; each method must state how it handles them.
- **Swappable strategies.** A strategy is one or more estimators, which report named evidence with its direction, units and objective component, plus decision policies that turn the evidence into continue/stop. Estimators receive one analysis group at a time; the framework fills counts, provenance (`config_id`) and compute time. `redecide` applies other thresholds to stored results without re-running. A strategy with no evidence about a required objective component returns `unsupported`, never `stop`. Current-practice baselines:
  - fixed FOV, fixed imaged area, generic observation-count, and legacy object/cell-count baselines;
  - a JCB/Zweifach-style scalar precision rule, re-estimated within each group or planned once from disjoint pilot samples (`plan_objects_to_average`, `PlannedObjectCountStrategy`);
  - Marcotti et al.-style descriptor stabilization.

  Candidate MorphoCoverage estimators (`docs/common/architecture.md`; those not yet built are in `docs/proposals/candidate_estimators_roadmap.md`), all in `morphocoverage.strategies` and none selected as a default:
  - `FOVBootstrapStrategy`: whole-FOV (or spatial-block) bootstrap uncertainty of the centroid, covariance, quantiles, PCA eigenspectrum or PCA subspace;
  - `SupportNoveltyStrategy`: leave-one-FOV-out fraction of observations outside the other FOVs' support at a declared resolution;
  - `IncidenceCoverageStrategy`: Chao–Jost incidence sample coverage and Chao2 completeness for externally fixed labels, with the FOV as the sampling unit;
  - rare-state rules for a `RareTarget` with FOV incidence: a fixed binomial plan (`FOVIncidencePlanStrategy`) and a beta-binomial risk that stops on recurrence (`RareIncidenceStrategy`).
- **Was it sampled enough, and would one more FOV show something new?** `morphocoverage.sufficiency.sufficiency_report` answers both for a finished dataset, per analysis group. For each criterion (strategy and thresholds) it says whether the acquired FOVs meet it and after how many FOVs it was first met across acquisition orders. Given a definition of "new" (`SupportNoveltyEstimator`), it gives the chance that one more FOV shows a phenotype the other FOVs do not (a cluster of similar unsupported cells, so scattered outliers do not count), a nominal upper bound whose coverage is checked empirically, how many more FOVs are needed to bring that bound below a target, the frequency above which an unseen state is ruled out (about 3/k after k FOVs, for a state this definition of new detects, in fields that are independent, representative draws: a random or dispersed order, not a raster; for a state detected in only part of the fields that hold it, the detection-adjusted bound of `representation_control` applies), and the cells counted as new so they can be checked in the images. `report.summary()` states the answers in sentences.
- **Is that chance right?** `morphocoverage.calibration.calibrate_next_fov_novelty` checks it on the dataset itself: it predicts from random subsets of FOVs and compares with how often the remaining FOVs actually showed something new.
- **Objects or fields?** `morphocoverage.sampling_unit`, whose variability and object-rule checks open the report (the unusual objects and their clustering are reported under "What is limiting my sampling?"), measures how much fields differ and how much an object count understates the error of a field-sampled mean. It runs the mean-precision rule and the JCS stabilisation procedure with random objects and with whole fields, labels objects unlike declared control groups, and tests whether any labelled state is clustered in a few fields (`docs/report/objects_or_fields.md`, `docs/preliminary_findings.md` entries 9 and 10).
- **Leakage-safe replay.** Recorded acquisitions are evaluated prefix by prefix: `evaluate_acquisition_prefixes`.
- **Benchmark protocol.** Datasets are declared as development or held out, and tuning on held-out data raises. `run_benchmark` replays recorded or seeded-permuted acquisition orders and records one row per run: effort at the stop, strategy-independent outcome metrics, and compute time (`morphocoverage.benchmark`, `morphocoverage.outcomes`). Outcomes can be scored against a held-out evaluation reservoir split by spatial block instead of the self-containing full acquisition (`morphocoverage.reservoir`), and include multivariate distribution errors (`morphocoverage.multivariate_outcomes`: centroid, covariance, quantile, Fréchet, energy and MMD U-statistics, sliced Wasserstein). `morphocoverage.spatial` gives FOV semivariograms for choosing block sizes. `morphocoverage.retrospective.frontier` scores rule families by fields saved against a fixed field count at matched risk of stopping too early, with approximate 95% intervals over independent units (widened for few units, and withheld below 5 units or when fewer than 90% of the resamples reach the risk level) and transfer from development to held-out data; the first locked run is in `docs/preliminary_findings.md`, entry 8.
- **Leakage-safe preprocessing.** Feature transforms (`Standardize`, `PCA`, `Chain`) are fitted on the visible prefix only, with the fitted state recorded as provenance (`morphocoverage.transforms`).
- **Incremental updates and latency.** `evaluate_acquisition_prefixes(..., incremental=True)` feeds supporting estimators one new FOV at a time. A contract test requires the result to equal full recomputation. Every result reports a timing breakdown: recomputation, update, initial fit, framework overhead and total decision time.
- **Live microscope API.** `OnlineSession` receives one newly acquired spatial position at a time and immediately returns the same `StrategyResult` objects used by retrospective replay. It never needs the completed acquisition or any metadata about future FOVs. Estimators without incremental support still work by recomputing on the visible session history.
- **Objects, not cells.** Public names say objects, because a row can be a cell, nucleus, patch, slice or embedding: `plan_objects_to_average`, `ObjectCountPlan`, `PlannedObjectCountEstimator`, `PlannedObjectCountStrategy`, `outcomes.RandomObjectRelativeError`, `RepresentationControl(planted_objects=...)`. The former cell names still work with a `DeprecationWarning` and will be removed later. Reported diagnostics were renamed with them (`cv_objects`, `cv_objects_upper_95`, `assumes_random_object_sampling`, the estimator `planned_objects`, the metric `random_object_relative_error:<feature>`) without aliases.
- **Contract tests.** Every strategy must pass them on constructed acquisitions (AGENTS.md section 9).
- **Real data, ready to use.** Four Crick/IDR datasets in `data/crick_idr/`: the papers' wells of idr0028 screen B and idr0139 with every intensity feature, and the complete plates (1,144 and 308 wells) with the readout. `datasets.load(name)` returns the linked tables with their features, readout, control condition, grouping and benchmark role; `data/crick_idr/run_all.py` reruns every analysis in `docs/preliminary_findings.md`.

```python
import pandas as pd
from morphocoverage import (
    ObjectiveConfig, SamplingObjective, ScalarPrecisionStrategy, FixedFOVStrategy,
    evaluate_acquisition_prefixes,
)

# The stored tables name the well and the plate "well" and "repeat"; the MorphoBricks
# columns for them are technical_repeat and biological_repeat.
names = {"well": "technical_repeat", "repeat": "biological_repeat"}
fovs = pd.read_csv("data/crick_idr/idr0028_screenB_fovs.csv.gz", dtype={"repeat": str})
objects = pd.read_csv("data/crick_idr/idr0028_screenB_nuclei.csv.gz", dtype={"repeat": str})
fovs, objects = fovs.rename(columns=names), objects.rename(columns=names)
feature = "Nuclei_Intensity_MeanIntensity_YAPTAZ"

results = evaluate_acquisition_prefixes(
    objects,
    fovs,
    strategies=[
        FixedFOVStrategy(10),
        ScalarPrecisionStrategy(feature, 0.05, "object", "upper_95"),
    ],
    feature_cols=[feature],
    acquisition_order_col="acquisition_order",
    objective=ObjectiveConfig(SamplingObjective.POPULATION),
    group_cols=["condition", "biological_repeat", "technical_repeat"],
)
```

For live acquisition, the same strategies can be used without a completed dataset:

```python
from morphocoverage import OnlineSession

session = OnlineSession(
    strategies=[FixedFOVStrategy(10)],
    feature_cols=["feature_a", "feature_b"],
    objective=ObjectiveConfig(SamplingObjective.POPULATION),
    fov_col="position_id",
    group_cols=["condition", "biological_repeat"],
    observation_unit="cell",
)

results = session.add_fov(
    observations=current_feature_table,
    fov_metadata={
        "condition": "control",
        "biological_repeat": "R1",
        "position_id": "P017",
    },
)
```

Only the current position is provided. Future positions do not need to exist yet. See [`docs/live/online_acquisition.md`](docs/live/online_acquisition.md) for the live-acquisition contract and timing model.

To add a strategy, follow AGENTS.md section 20. Preliminary real-data observations are in [`docs/preliminary_findings.md`](docs/preliminary_findings.md).

## Planned scientific development

Development should proceed by falsification rather than by prematurely committing to one estimator:

1. maintain strict linked FOV/observation contracts and the common offline/online execution path;
2. implement substantially different candidate solution families behind the common estimator interface;
3. benchmark them primarily on deeply sampled real microscopy datasets, with constructed data reserved for contracts/counterexamples;
4. compare every method with simple/current-practice baselines and strategy-independent outcomes on unseen FOVs;
5. characterize scientific accuracy, under-sampling risk, acquisition efficiency and online latency as separate dimensions;
6. select a default method only if evidence supports one;
7. validate representative continue/stop prospectively on the microscope;
8. investigate targeted next-FOV selection only after representative stopping is validated.

Read [`docs/common/project_aim.md`](docs/common/project_aim.md) for the detailed scientific aims, relationship to current cell-number guidance, smart-microscopy goal, assumptions, adversarial tests, and failure criteria. See [`AGENTS.md`](AGENTS.md) before making changes; it is the authoritative engineering context for AI coding agents and contributors.

## Development

Python 3.11+ is required. Runtime dependencies: `numpy`, `pandas`, `scipy`, `matplotlib` (only for the figures of `morphocoverage.plots`, `plots_embedding` and `workflows.plots`; importing `morphocoverage.workflows` loads it, the core, `retrospective` and `sampling_unit` do not) and `umap-learn` (the UMAP of `workflows.embedding_plots`, loaded only when one is drawn; it brings scikit-learn and numba).

```bash
python -m pip install -e ".[dev]"
pytest            # on every CPU (pytest-xdist); pytest -n 0 runs serially
ruff check src tests data
ruff format --check src tests data
```

Run the whole suite before every push: a pull request on GitHub runs only the minimal checks (AGENTS.md section 13).

## License

MIT. See [`LICENSE`](LICENSE).
