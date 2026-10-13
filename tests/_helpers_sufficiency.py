"""An acquisition with a new state, shared by the sufficiency tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from morphocoverage.benchmark import AcquisitionOrder
from morphocoverage.objectives import ObjectiveConfig, SamplingObjective


def _novel_acquisition(n_fovs=12, state_fovs=(3, 7), outlier_fovs=(5,)):
    """Common cells everywhere; its own 3-cell state in ``state_fovs``, one odd cell elsewhere."""
    rng = np.random.default_rng(0)
    rows = []
    for j in range(n_fovs):
        cells = rng.normal(10, 0.1, size=(20, 2))
        if j in state_fovs:
            cells = np.vstack([cells, rng.normal(20 + 5 * j, 0.05, size=(3, 2))])
        if j in outlier_fovs:
            cells = np.vstack([cells, [[50.0, 50.0]]])
        rows += [
            {"image_file": f"f{j}", "label_id": i, "a": a, "b": b} for i, (a, b) in enumerate(cells)
        ]
    fovs = pd.DataFrame({"image_file": [f"f{j}" for j in range(n_fovs + 1)]})  # one empty
    return pd.DataFrame(rows), fovs


POPULATION = ObjectiveConfig(SamplingObjective.POPULATION)


PERMUTED = [AcquisitionOrder.permuted(seed) for seed in range(5)]
