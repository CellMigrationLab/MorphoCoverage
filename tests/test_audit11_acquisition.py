"""Issue 277: one report cannot combine results computed on different acquisitions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from morphocoverage._acquisition import acquisition_identity, check_one_acquisition
from morphocoverage.retrospective.report import microscopist_report
from morphocoverage.sampling_unit import fov_variability, object_rule_check
from morphocoverage.schema import validate_input_tables


def _tables(prefix: str = "f", n: int = 6, seed: int = 0):
    rng = np.random.default_rng(seed)
    fovs = pd.DataFrame(
        {
            "condition": "a",
            "biological_repeat": "R1",
            "position_id": [f"{prefix}{i}" for i in range(n)],
        }
    )
    rows = [
        {"condition": "a", "biological_repeat": "R1", "position_id": p, "x": float(v)}
        for p in fovs["position_id"]
        for v in rng.normal(size=8)
    ]
    return pd.DataFrame(rows), fovs


def _variability(objects, fovs):
    return fov_variability(objects, fovs, feature="x")


def test_the_identity_ignores_row_order_and_values_but_not_the_fields():
    objects, fovs = _tables()
    one = validate_input_tables(objects, fovs=fovs, feature_cols=["x"])
    shuffled = validate_input_tables(
        objects.assign(x=objects["x"] * 3), fovs=fovs.iloc[::-1], feature_cols=["x"]
    )
    other_objects, other_fovs = _tables("g")
    other = validate_input_tables(other_objects, fovs=other_fovs, feature_cols=["x"])
    assert acquisition_identity(one) == acquisition_identity(shuffled)
    assert acquisition_identity(one) != acquisition_identity(other)


def test_results_of_two_acquisitions_are_refused_before_any_text():
    first = _variability(*_tables("f"))
    other_objects, other_fovs = _tables("g", seed=1)
    second = object_rule_check(other_objects, other_fovs, feature="x", n_draws=50)
    with pytest.raises(ValueError, match="different acquisitions"):
        check_one_acquisition({"variability": first, "object_rules": second})
    with pytest.raises(ValueError, match="different acquisitions"):
        microscopist_report(variability=first, object_rules=second)


def test_results_of_one_acquisition_with_other_settings_compose():
    objects, fovs = _tables()
    first = fov_variability(objects, fovs, feature="x")
    again = fov_variability(objects.assign(x=objects["x"] + 5), fovs, feature="x")
    assert check_one_acquisition({"a": first, "b": again}) == []
    assert microscopist_report(variability=first) is not None


def test_a_result_without_an_identity_is_named_not_assumed_compatible():
    objects, fovs = _tables()
    result = _variability(objects, fovs)

    class Bare:
        def __init__(self) -> None:
            self.settings: dict = {}

    assert check_one_acquisition({"variability": result, "effort": Bare()}) == ["effort"]
    assert check_one_acquisition({"effort": Bare()}) == []
