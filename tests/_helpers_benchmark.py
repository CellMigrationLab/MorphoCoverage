"""Builders shared by the benchmark, replay and live tests: objectives and two-group tables."""

import numpy as np
import pandas as pd

from morphocoverage import ObjectiveConfig, SamplingObjective
from morphocoverage.outcomes import RelativeMeanError
from morphocoverage.retrospective import OutcomeCriterion
from morphocoverage.strategies.support_novelty import SupportNoveltyEstimator

POPULATION = ObjectiveConfig(SamplingObjective.POPULATION)


MEAN_F = OutcomeCriterion(RelativeMeanError("f"), 0.05, "lower_is_better")


def _two_groups(n_a=12, n_b=12, seed=0):
    """Two conditions of ``n_a`` and ``n_b`` positions, 10 objects each, with an order."""
    rng = np.random.default_rng(seed)
    fovs = pd.DataFrame(
        {
            "condition": ["a"] * n_a + ["b"] * n_b,
            "biological_repeat": "r1",
            "repeat_group": "r1",
            "position_id": [f"p{i:02d}" for i in range(n_a + n_b)],
            "order": list(range(n_a)) + list(range(n_b)),
        }
    )
    objects = fovs.loc[fovs.index.repeat(10), ["condition", "biological_repeat", "position_id"]]
    objects = objects.reset_index(drop=True)
    objects["f"] = rng.normal(10, 1, len(objects)) + np.repeat(rng.normal(0, 1, len(fovs)), 10)
    return objects, fovs


def _novelty_tables():
    rng = np.random.default_rng(0)
    fovs = pd.DataFrame({"position_id": [f"p{i:02d}" for i in range(30)]})
    objects = pd.DataFrame({"position_id": np.repeat(fovs.position_id, 10)})
    objects["f"] = rng.normal(0, 1, len(objects)) + np.repeat(rng.choice([0, 0, 0, 5], 30), 10)
    settings = {
        "feature_cols": ["f"],
        "novelty": SupportNoveltyEstimator(0.5, 1, None, 3),
        "fov_counts": [5, 10],
        "draws": 20,
        "random_state": 0,
    }
    return objects, fovs, settings
