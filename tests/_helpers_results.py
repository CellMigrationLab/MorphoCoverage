"""Builders shared by the tests of results, the live text and the order check."""

from __future__ import annotations

import numpy as np
import pandas as pd
from _helpers_strategies import READOUT

from morphocoverage import FixedFOVStrategy
from morphocoverage.evidence import Evidence, Quantity
from morphocoverage.objectives import ObjectiveConfig, SamplingObjective
from morphocoverage.results import StrategyResult
from morphocoverage.strategies import ScalarPrecisionStrategy
from morphocoverage.workflows import LiveTables, Setup, live_session


def _available_quantity() -> Quantity:
    return Quantity.available(0.1, direction="lower_is_better", units="x", component="population")


def _strategy_result() -> StrategyResult:
    configuration = {"estimators": [{"config_id": "e", "parameters": {"k": 1}}], "policies": []}
    return StrategyResult(
        strategy="s",
        config_id="c",
        objective=ObjectiveConfig(SamplingObjective.POPULATION),
        evidence=Evidence({"score": _available_quantity()}),
        decision=None,
        parameters=configuration,
        configuration=configuration,
        group={"condition": "a"},
        n_fovs=3,
        n_observations=9,
    )


def _flat(text: str) -> str:
    return " ".join(text.split())


def _well(*, drift: float = 0.0, n_fields: int = 16, rare_rows: int = 0, seed: int = 5):
    """A 4-column well in raster order; the readout drifts by ``drift`` per field."""
    rng = np.random.default_rng(seed)
    objects, fovs = [], []
    for rank in range(n_fields):
        row, column = divmod(rank, 4)
        name = f"f{rank:02d}.tif"
        values = 100.0 * (1 + drift * rank) * np.exp(0.1 * rng.standard_normal(30))
        rare = np.zeros(30, bool)
        if rare_rows and row >= n_fields // 4 - rare_rows:
            rare[::4] = True
        objects.append(
            pd.DataFrame(
                {
                    "condition": "MOCK",
                    "biological_repeat": "1A",
                    "repeat_group": "1A",
                    "image_file": name,
                    READOUT: values,
                }
            ).assign(rare=rare)
        )
        fovs.append(
            {
                "condition": "MOCK",
                "biological_repeat": "1A",
                "repeat_group": "1A",
                "image_file": name,
                "order": rank,
                "stage_x": 500.0 * column,
                "stage_y": 500.0 * row,
            }
        )
    return objects, pd.DataFrame(fovs)


def _updates(setup, objects, fovs):
    """Feed the fields one at a time, each with its objects, as an acquisition does."""
    live = LiveTables(live_session(setup))
    updates = []
    for step in range(len(fovs)):
        updates += live.update(pd.concat(objects[: step + 1]), fovs.iloc[: step + 1])
    return updates


def _nucleus_setup(**extra):
    rules = (
        FixedFOVStrategy(8),
        ScalarPrecisionStrategy(READOUT, 0.05, "object", "upper_95"),
        ScalarPrecisionStrategy(READOUT, 0.05, "fov", "upper_95"),
    )
    return Setup(readout=READOUT, observation_unit="nucleus", rules=rules, **extra)
