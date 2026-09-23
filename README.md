# MorphoCoverage

MorphoCoverage is a feature-agnostic Python project for studying whether microscopy experiments have sampled a declared phenotypic representation adequately. It treats **fields of view (FOVs), rather than individual objects/cells, as the primary operational acquisition units** and asks whether additional representative imaging is likely to provide useful new phenotypic information.

## Project boundary

MorphoCoverage consumes object-level feature representations together with FOV identity and optional grouping metadata.

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
object/cell -> FOV -> optional sample/repeat/condition
```

and never treats cell count alone as a complete description of acquisition effort. FOVs themselves are not assumed independent; stage coordinates, physical area, and sampling design can matter.

## Current status

MorphoCoverage is a strategy-comparison framework. No default coverage estimator has been chosen.

- **Input validation.** Linked FOV and object tables with MorphoBricks defaults. `image_file` identifies the FOV, and `condition` × `repeat` are separate analysis groups that stay part of FOV identity even when pooled. Feature columns are always explicit, and acquired FOVs with zero objects are kept.
- **Swappable strategies.** Every method implements one `SamplingStrategy` interface and receives one analysis group at a time. The framework fills counts, provenance and runtime. Current strategies are all current-practice baselines:
  - fixed FOV and fixed cell counts;
  - a JCB/Zweifach-style scalar precision rule;
  - Marcotti et al.-style descriptor stabilization.
- **Leakage-safe replay.** Recorded acquisitions are evaluated prefix by prefix: `evaluate_acquisition_prefixes`.
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

To add a strategy, follow AGENTS.md section 20. Preliminary real-data observations are in [`docs/preliminary_findings.md`](docs/preliminary_findings.md).

## Planned scientific development

Development should proceed by falsification rather than by prematurely committing to one estimator:

1. strict linked FOV/object data contracts;
2. synthetic benchmark datasets and a leakage-safe sequential holdout framework;
3. faithful implementations of simple/current-practice baselines, including fixed cells/FOVs and scalar cell-averaging precision;
4. substantially different candidate solution families kept separate;
5. comparison against measurable outcomes in unseen FOVs;
6. selection of a default method only if evidence supports one;
7. prospective continue/stop validation;
8. smart-microscopy targeting only after representative stopping is validated.

Read [`docs/project_aim.md`](docs/project_aim.md) for the detailed scientific aims, relationship to current cell-number guidance, smart-microscopy goal, assumptions, adversarial tests, and failure criteria. See [`AGENTS.md`](AGENTS.md) before making changes; it is the authoritative engineering context for AI coding agents and contributors.

## Development

Python 3.11+ is required.

```bash
python -m pip install -e ".[dev]"
pytest
ruff check src tests
```

## License

MIT. See [`LICENSE`](LICENSE).
