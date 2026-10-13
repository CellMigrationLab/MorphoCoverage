"""One observation per field: one definition of new everywhere (owner, 2026-10-06, plan v2).

A definition of new needing ``m`` similar observations in one field cannot fire when each
field holds one, so the estimator reads its field version itself (``per_field``: new when no
other field holds one within the radius, a state seen once): the report, ``optimise``,
``benchmark`` and a live session then give one verdict.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from _helpers_embedding import FEATURES, _one_per_field

from morphocoverage import ObjectiveConfig
from morphocoverage.data_check import validate_input_tables
from morphocoverage.strategies.support_novelty import (
    FOV_QUANTITY,
    UPPER_QUANTITY,
    SupportNoveltyStrategy,
    per_field,
)
from morphocoverage.transforms import RobustStandardize
from morphocoverage.workflows import Setup, benchmark, live_session, optimise
from morphocoverage.workflows.benchmarks import Family

sys.path.insert(0, str(Path(__file__).parent))

RULE = SupportNoveltyStrategy(np.sqrt(len(FEATURES)), 0.5, 1, RobustStandardize(), 5, 3)
FIELD_RULE = per_field(RULE)
OBJECTIVE = ObjectiveConfig("distribution")


@pytest.fixture(scope="module")
def images():
    """18 images of one condition and repeat each, one image per field."""
    objects, fovs, _ = _one_per_field()
    return objects.assign(biological_repeat="R0"), fovs.assign(biological_repeat="R0")


def _setup(rule: SupportNoveltyStrategy) -> Setup:
    return Setup(readout="f1", features=FEATURES, rules=(rule,), observation_unit="image")


def test_the_field_version_is_new_when_no_other_field_holds_one():
    assert (FIELD_RULE.min_other_fovs, FIELD_RULE.min_novel_observations) == (1, 1)
    assert per_field(FIELD_RULE) is FIELD_RULE


def test_the_estimator_reads_the_field_version_on_fields_of_one(images):
    objects, fovs = images
    objects = objects.copy()
    objects.loc[0, "f1"] += 10.0  # one image unlike every other
    data = validate_input_tables(
        objects, fovs=fovs, feature_cols=list(FEATURES), observation_unit="image"
    )
    declared = RULE.estimators[0].estimate(data, objective=OBJECTIVE).quantities
    field = FIELD_RULE.estimators[0].estimate(data, objective=OBJECTIVE).quantities
    assert declared[UPPER_QUANTITY].value == field[UPPER_QUANTITY].value
    # Not vacuous: as declared, 3 similar images in one field, no field could be new.
    assert field[FOV_QUANTITY].value > 0


def _calibration_runs(plan) -> pd.DataFrame:
    runs = plan.results["calibration"].runs
    return runs[["condition", "split", "order_id", "stopped", "stop_fovs"]].reset_index(drop=True)


def test_optimise_gives_the_field_version_s_verdict(images):
    objects, fovs = images
    arguments = {
        "fov_counts": [6, 9],
        "reservoir_fraction": 0.3,
        "n_splits": 2,
        "draws": 5,
        "permuted_orders": 2,
        "dispersed_orders": 0,
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        declared = optimise(objects, fovs, _setup(RULE), **arguments)
        field = optimise(objects, fovs, _setup(FIELD_RULE), **arguments)
    pd.testing.assert_frame_equal(_calibration_runs(declared), _calibration_runs(field))
    assert declared.setup.rules == (RULE,)  # the Plan keeps the declared Setup


def test_benchmark_gives_the_field_version_s_verdict(images):
    objects, fovs = images
    found = []
    for rule in (RULE, FIELD_RULE):
        family = Family("novelty", rule, UPPER_QUANTITY, (0.3, 0.5, 0.8))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = benchmark(
                objects,
                fovs,
                _setup(rule),
                reservoir_fraction=0.3,
                families=[family],
                n_splits=2,
                permuted_orders=2,
                n_boot=50,
            )
        found.append(result.summary(detail="full"))
    assert found[0] == found[1]


def test_a_live_session_gives_the_field_version_s_verdict(images):
    objects, fovs = images
    mine = fovs[fovs["condition"].eq("ctrl")].reset_index(drop=True)
    verdicts = []
    for rule in (RULE, FIELD_RULE):
        session = live_session(_setup(rule))
        last = None
        for i in range(len(mine)):
            row = mine.iloc[[i]]
            observed = objects.merge(row, on=list(row.columns))
            last = session.add_fov(observed, row)
        quantity = last[0].evidence.quantities[UPPER_QUANTITY]
        verdicts.append((quantity.value, last[0].rule_decision))
    assert verdicts[0] == verdicts[1]
