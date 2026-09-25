# MorphoCoverage

MorphoCoverage is a feature-agnostic Python project for studying whether microscopy experiments have sampled a declared phenotypic representation adequately. It treats **fields of view (FOVs), defined as independently selected spatial acquisition positions, rather than individual observations/cells or z-planes, as the primary operational acquisition units** and asks whether additional representative imaging is likely to provide useful new phenotypic information.

## Questions MorphoCoverage is built to answer

The canonical list, grouped by the three moments of use (REPORT: what can I trust in a finished acquisition; OPTIMISE: what should I do next time; ONLINE: should I acquire another field now) and a fourth section (CHECK: can I count cells, or must I count fields), with what information may answer each question and its status, is [`docs/questions.md`](docs/questions.md). The table below maps the main questions to the functions that answer them.

For a completed experiment, the aim is to tell a microscopist whether the imaging effort was sufficient, what the additional FOVs actually bought, and how to spend the imaging budget in the next experiment. Every answer depends on the declared phenotype representation (the features and the definition of "new") and is stated with the criterion behind it.

| Question | Today |
| --- | --- |
| Can I count cells, or must I count fields? How variable are my fields? | `morphocoverage.sampling_unit`: `field_variability` (design effect, fields that differ more than random cells would), `cell_rule_check` and `descriptor_spread_check` (the JCB and JCS rules with random cells and with whole fields) |
| Are rare or unusual cells clustered in a few fields? | `morphocoverage.sampling_unit`: `unusual_cells` against declared control groups, `field_clustering` on any state labels (field incidence against random cells, fields needed) |
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

A completed dataset may be represented by a few well-chosen FOVs, while a new experiment that samples FOVs at random needs more: **retrospective representability is not the prospective sampling requirement.** `morphocoverage.retrospective.microscopist_report` assembles these answers question by question (`docs/proposals/microscopist_report.md`). Underneath, several candidate estimators answer these questions; none is assumed to be universally correct.

## Project boundary

MorphoCoverage consumes numeric phenotype feature tables together with FOV identity and optional grouping metadata. A feature row is a generic **observation**: it may be a cell, nucleus, patch, slice-level measurement, FOV summary, embedding, or another declared unit.

It does **not**:

- read microscopy images or segmentation masks;
- extract morphometric, texture, intensity, or deep-learning features;
- depend on CellPatcher, Nyxus, DINO, Torch, or another feature extractor;
- perform biological-condition hypothesis testing;
- control a microscope.

Those concerns belong upstream or downstream. MorphoCoverage focuses on the statistical problem of **FOV-aware phenotypic sampling adequacy**. The mathematical solution is deliberately open: candidate estimators must be compared against simple baselines and unseen FOVs before any one approach becomes the default.

## Why FOVs matter

Microscopes acquire images, not independent cells. Five hundred cells from two FOVs may contain very different sampling information from five hundred cells distributed across twenty FOVs when phenotypes are spatially structured. Whether that difference is important is a central hypothesis to test, not an assumption to protect.

MorphoCoverage therefore preserves the hierarchy

```text
observation (cell / nucleus / patch / slice / FOV summary / embedding)
    -> FOV = spatial acquisition position
    -> optional sample/repeat/condition
```

and never treats cell count alone as a complete description of acquisition effort. FOVs themselves are not assumed independent; stage coordinates, physical area, and sampling design can matter.

## Current status

MorphoCoverage is a strategy-comparison framework. No default coverage estimator has been chosen.

- **Input validation.** Linked FOV and phenotype-observation tables with MorphoBricks defaults. `position_id` is the preferred explicit spatial-position identifier; `image_file` is a fallback only when one file corresponds to one position. `condition` × `repeat` are separate analysis groups that stay part of FOV identity even when pooled. Feature columns are always explicit, acquired FOVs with zero observations are kept, and FOV-level grouping metadata can be inherited from the FOV table when the position identifier is unambiguous; and when `z`, `c`, and/or `t` describe acquisition dimensions, they are preserved as within-position metadata and omitted from the explicit phenotype feature list. `observation_unit` records what one feature row means. NaN feature values may be loaded but are never silently imputed or dropped; each method must state how it handles them.
- **Swappable strategies.** A strategy is one or more estimators, which report named evidence with its direction, units and objective component, plus decision policies that turn the evidence into continue/stop. Estimators receive one analysis group at a time; the framework fills counts, provenance (`config_id`) and compute time. `redecide` applies other thresholds to stored results without re-running. A strategy with no evidence about a required objective component returns `unsupported`, never `stop`. Current-practice baselines:
  - fixed FOV, fixed imaged area, generic observation-count, and legacy object/cell-count baselines;
  - a JCB/Zweifach-style scalar precision rule, re-estimated within each group or planned once from disjoint pilot samples (`plan_cells_to_average`, `PlannedCellCountStrategy`);
  - Marcotti et al.-style descriptor stabilization.

  Candidate MorphoCoverage estimators from the estimator roadmap proposal (`docs/proposals/candidate_estimators_roadmap.md`), all in `morphocoverage.strategies` and none selected as a default:
  - `FOVBootstrapStrategy`: whole-FOV (or spatial-block) bootstrap uncertainty of the centroid, covariance, quantiles, PCA eigenspectrum or PCA subspace;
  - `SupportNoveltyStrategy`: leave-one-FOV-out fraction of observations outside the other FOVs' support at a declared resolution;
  - `IncidenceCoverageStrategy`: Chao–Jost incidence sample coverage and Chao2 completeness for externally fixed labels, with the FOV as the sampling unit;
  - rare-state rules for a `RareTarget` with FOV incidence: a fixed binomial plan (`FOVIncidencePlanStrategy`) and a beta-binomial risk that stops on recurrence (`RareIncidenceStrategy`).
- **Was it sampled enough, and would one more FOV show something new?** `morphocoverage.sufficiency.sufficiency_report` answers both for a finished dataset, per analysis group. For each criterion (strategy and thresholds) it says whether the acquired FOVs meet it and after how many FOVs it was first met across acquisition orders. Given a definition of "new" (`SupportNoveltyEstimator`), it gives the chance that one more FOV shows a phenotype the other FOVs do not (a cluster of similar unsupported cells, so scattered outliers do not count), a nominal upper bound whose coverage is checked empirically, how many more FOVs are needed to bring that bound below a target, the frequency above which an unseen state is ruled out (about 3/k after k FOVs), and the cells counted as new so they can be checked in the images. `report.summary()` states the answers in sentences.
- **Is that chance right?** `morphocoverage.calibration.calibrate_next_fov_novelty` checks it on the dataset itself: it predicts from random subsets of FOVs and compares with how often the remaining FOVs actually showed something new.
- **Leakage-safe replay.** Recorded acquisitions are evaluated prefix by prefix: `evaluate_acquisition_prefixes`.
- **Benchmark protocol.** Datasets are declared as development or held out, and tuning on held-out data raises. `run_benchmark` replays recorded or seeded-permuted acquisition orders and records one row per run: effort at the stop, strategy-independent outcome metrics, and compute time (`morphocoverage.benchmark`, `morphocoverage.outcomes`). Outcomes can be scored against a held-out evaluation reservoir split by spatial block instead of the self-containing full acquisition (`morphocoverage.reservoir`), and include multivariate distribution errors (`morphocoverage.multivariate_outcomes`: centroid, covariance, quantile, Fréchet, energy and MMD U-statistics, sliced Wasserstein). `morphocoverage.spatial` gives FOV semivariograms for choosing block sizes.
- **Leakage-safe preprocessing.** Feature transforms (`Standardize`, `PCA`, `Chain`) are fitted on the visible prefix only, with the fitted state recorded as provenance (`morphocoverage.transforms`).
- **Incremental updates and latency.** `evaluate_acquisition_prefixes(..., incremental=True)` feeds supporting estimators one new FOV at a time. A contract test requires the result to equal full recomputation. Every result reports a timing breakdown: recomputation, update, initial fit, framework overhead and total decision time.
- **Live microscope API.** `OnlineSession` receives one newly acquired spatial position at a time and immediately returns the same `StrategyResult` objects used by retrospective replay. It never needs the completed acquisition or any metadata about future FOVs. Estimators without incremental support still work by recomputing on the visible session history.
- **Contract tests.** Every strategy must pass them on constructed acquisitions (AGENTS.md section 9).
- **Real test data.** Crick/IDR feature tables in `data/crick_idr/`.

```python
import pandas as pd
from morphocoverage import (
    ObjectiveConfig, SamplingObjective, ScalarPrecisionStrategy, FixedFOVStrategy,
    evaluate_acquisition_prefixes,
)

fovs = pd.read_csv("data/crick_idr/idr0028_screenB_fovs.csv.gz", dtype={"repeat": str})
objects = pd.read_csv("data/crick_idr/idr0028_screenB_nuclei.csv.gz", dtype={"repeat": str})
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
    group_cols=["condition", "repeat", "well"],
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
    group_cols=["condition", "repeat"],
    observation_unit="cell",
)

results = session.add_fov(
    observations=current_feature_table,
    fov_metadata={
        "condition": "control",
        "repeat": "R1",
        "position_id": "P017",
    },
)
```

Only the current position is provided. Future positions do not need to exist yet. See [`docs/online.md`](docs/online.md) for the live-acquisition contract and timing model.

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

Read [`docs/project_aim.md`](docs/project_aim.md) for the detailed scientific aims, relationship to current cell-number guidance, smart-microscopy goal, assumptions, adversarial tests, and failure criteria. See [`AGENTS.md`](AGENTS.md) before making changes; it is the authoritative engineering context for AI coding agents and contributors.

## Development

Python 3.11+ is required.

```bash
python -m pip install -e ".[dev]"
pytest            # on every CPU (pytest-xdist); pytest -n 0 runs serially
ruff check src tests data
ruff format --check src tests data
```

## License

MIT. See [`LICENSE`](LICENSE).
