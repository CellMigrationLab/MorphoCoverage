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

This repository is intentionally starting small. The first implemented layer is the strict input contract needed by later coverage estimators.

The current API validates:

- an object table containing explicitly selected numeric feature columns and FOV identity;
- an optional complete FOV table, so acquired fields with zero detected objects are not silently lost;
- optional grouping columns such as `condition` and `repeat`.

Feature columns are never inferred automatically from all numeric columns.

```python
import pandas as pd
from morphocoverage import validate_input_tables

objects = pd.DataFrame(
    {
        "condition": ["control", "control", "control"],
        "repeat": ["R1", "R1", "R1"],
        "fov_id": ["FOV_1", "FOV_1", "FOV_2"],
        "area": [120.0, 135.0, 98.0],
        "eccentricity": [0.45, 0.62, 0.31],
    }
)

fovs = pd.DataFrame(
    {
        "condition": ["control", "control", "control"],
        "repeat": ["R1", "R1", "R1"],
        "fov_id": ["FOV_1", "FOV_2", "FOV_3"],
    }
)

validated = validate_input_tables(
    objects,
    feature_cols=["area", "eccentricity"],
    fov_col="fov_id",
    fovs=fovs,
    group_cols=["condition", "repeat"],
)
```

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
