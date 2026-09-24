# MorphoCoverage

MorphoCoverage is a feature-agnostic Python project for studying whether microscopy experiments have sampled a declared phenotypic representation adequately. It treats **fields of view (FOVs), defined as independently selected spatial acquisition positions, rather than individual observations/cells or z-planes, as the primary operational acquisition units** and asks whether additional representative imaging is likely to provide useful new phenotypic information.

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

  Candidate MorphoCoverage estimators from the roadmap proposal (`docs/proposals/`), all in `morphocoverage.strategies` and none selected as a default:
  - `FOVBootstrapStrategy`: whole-FOV (or spatial-block) bootstrap uncertainty of the centroid, covariance, quantiles, PCA eigenspectrum or PCA subspace;
  - `SupportNoveltyStrategy`: leave-one-FOV-out fraction of observations outside the other FOVs' support at a declared resolution;
  - `IncidenceCoverageStrategy`: Chao–Jost incidence sample coverage and Chao2 completeness for externally fixed labels, with the FOV as the sampling unit;
  - rare-state rules for a `RareTarget` with FOV incidence: a fixed binomial plan (`FOVIncidencePlanStrategy`) and a beta-binomial risk that stops on recurrence (`RareIncidenceStrategy`).
- **Was it sampled enough?** `morphocoverage.sufficiency.sufficiency_report` runs several strategies on a finished dataset and reports, per analysis group, each strategy's verdict on the full acquisition, all its evidence (including the probability that another FOV would show a new phenotype, with an upper confidence bound), and where it would have stopped across declared acquisition orders.
- **Leakage-safe replay.** Recorded acquisitions are evaluated prefix by prefix: `evaluate_acquisition_prefixes`.
- **Benchmark protocol.** Datasets are declared as development or held out, and tuning on held-out data raises. `run_benchmark` replays recorded or seeded-permuted acquisition orders and records one row per run: effort at the stop, strategy-independent outcome metrics, and compute time (`morphocoverage.benchmark`, `morphocoverage.outcomes`). Outcomes can be scored against a held-out evaluation reservoir split by spatial block instead of the self-containing full acquisition (`morphocoverage.reservoir`), and include multivariate distribution errors (`morphocoverage.multivariate_outcomes`: centroid, covariance, quantile, Fréchet, unbiased energy and MMD, sliced Wasserstein). `morphocoverage.spatial` gives FOV semivariograms for choosing block sizes.
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
pytest
ruff check src tests data
```

## License

MIT. See [`LICENSE`](LICENSE).
