"""Column lists are read once, in order, and never reinterpreted (issue 222)."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from morphocoverage import integrity_checks
from morphocoverage.normalize import normalize_to_control
from morphocoverage.retrospective import (
    SamplingHierarchy,
    dataset_overview,
    effort_projection,
    inspect_fovs,
)


def _tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(0)
    rows, fovs = [], []
    for condition in ("ctrl", "drug"):
        for repeat in ("R1", "R2", "R3"):
            for f in range(6):
                pid = f"{condition}{repeat}{f}"
                fovs.append(
                    {"position_id": pid, "condition": condition, "biological_repeat": repeat}
                )
                for _ in range(6):
                    rows.append({**fovs[-1], "x": rng.normal(), "y": rng.normal()})
    return pd.DataFrame(rows), pd.DataFrame(fovs)


OBJECTS, FOVS = _tables()
HIERARCHY = SamplingHierarchy("position_id", "condition")


def _quiet(call):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return call()


CALLS = {
    "integrity feature_cols": lambda v: integrity_checks(OBJECTS, FOVS, feature_cols=v),
    "overview feature_cols": lambda v: dataset_overview(OBJECTS, FOVS, feature_cols=v),
    "inspect novelty_cols": lambda v: inspect_fovs(
        OBJECTS, FOVS, feature_cols=["x", "y"], novelty_cols=v, fov_col="position_id"
    ),
    "hierarchy technical_levels": lambda v: SamplingHierarchy("f", "s", None, v),
    "effort strata": lambda v: effort_projection(
        OBJECTS, FOVS, feature="x", hierarchy=HIERARCHY, strata=v
    ),
    "normalize feature_cols": lambda v: normalize_to_control(
        OBJECTS, FOVS, feature_cols=v, control={"condition": "ctrl"}
    ),
}


@pytest.mark.parametrize("name", CALLS)
@pytest.mark.parametrize("bad", ["xy", {"x", "y"}, {"x": 1}], ids=["string", "set", "mapping"])
def test_strings_sets_and_mappings_are_refused_not_reinterpreted(name, bad):
    with pytest.raises(TypeError, match="column names"):
        _quiet(lambda: CALLS[name](bad))


def test_a_generator_is_consumed_exactly_once():
    expected = _quiet(lambda: dataset_overview(OBJECTS, FOVS, feature_cols=["x", "y"]))
    once = _quiet(lambda: dataset_overview(OBJECTS, FOVS, feature_cols=(c for c in ["x", "y"])))
    assert once.features == expected.features == ("x", "y")
    report = _quiet(lambda: integrity_checks(OBJECTS, FOVS, feature_cols=(c for c in ["x", "y"])))
    assert report is not None
    inspected = _quiet(
        lambda: inspect_fovs(
            OBJECTS,
            FOVS,
            feature_cols=(c for c in ["x", "y"]),
            novelty_cols=(c for c in ["x"]),
            fov_col="position_id",
        )
    )
    assert inspected is not None


def test_technical_levels_keep_their_order():
    assert SamplingHierarchy("f", "s", None, ["plate", "well"]).technical_levels == (
        "plate",
        "well",
    )
    assert SamplingHierarchy("f", "s").technical_levels == ()


@pytest.mark.parametrize("bad", ["xy", {"x", "y"}, {"x": 1}], ids=["string", "set", "mapping"])
def test_the_sampling_unit_feature_selections_refuse_what_they_cannot_read(bad):
    from morphocoverage.sampling_unit import feature_ranking, unusual_objects

    control = {"condition": "ctrl"}
    with pytest.raises(TypeError, match="column names"):
        _quiet(lambda: feature_ranking(OBJECTS, FOVS, features=bad, control=control))
    with pytest.raises(TypeError, match="column names"):
        _quiet(
            lambda: unusual_objects(
                OBJECTS, FOVS, feature_cols=bad, reference=control, fov_col="position_id"
            )
        )
