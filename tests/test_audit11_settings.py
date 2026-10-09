"""Scientific scalar settings are real numbers, never string coercions (issue 236)."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from morphocoverage._checks import check_column, check_flag, is_positive_finite, is_share
from morphocoverage.retrospective.frontier import frontier
from morphocoverage.retrospective.replay import ReplayResult
from morphocoverage.workflows import Setup, optimise


def test_the_scalar_helpers_accept_real_numbers_only():
    assert is_share(0.5) and is_share(np.float32(0.5)) and is_share(1, closed_high=True)
    assert not is_share(1) and not is_share(0)
    for bad in ("0.5", True, float("nan"), float("inf"), None, [0.5]):
        assert not is_share(bad) and not is_share(bad, closed_high=True)
    assert is_positive_finite(2) and is_positive_finite(np.float64(0.1))
    for bad in ("2", True, 0, -1, float("inf"), float("nan")):
        assert not is_positive_finite(bad)


def test_flags_and_single_column_names_are_checked_by_type():
    assert check_flag(True, "x") is True and check_flag(np.bool_(False), "x") is False
    for bad in (1, 0, "no", None, [True]):
        with pytest.raises(TypeError, match="True or False"):
            check_flag(bad, "x")
    assert check_column(None, "c") is None and check_column("order", "c") == "order"
    for bad in (["order"], ("a", "b"), {"a": 1}, "", "  ", 3):
        with pytest.raises(TypeError, match="one column name"):
            check_column(bad, "c")


def _tables():
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
                    rows.append({**fovs[-1], "x": rng.normal()})
    return pd.DataFrame(rows), pd.DataFrame(fovs)


@pytest.mark.parametrize(
    ("argument", "bad"),
    [
        ("reservoir_fraction", "0.35"),
        ("reservoir_fraction", True),
        ("reservoir_fraction", float("nan")),
        ("protocol_threshold", "0.9"),
        ("protocol_threshold", True),
        ("protocol_threshold", float("inf")),
    ],
)
def test_optimise_refuses_string_bool_and_non_finite_shares(argument, bad):
    objects, fovs = _tables()
    setup = Setup(readout="x", control={"condition": "ctrl"}, observation_unit="cell")
    arguments = {"fov_counts": [2, 3], "reservoir_fraction": 0.35, argument: bad}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError, match="between 0 and 1|share in"):
            optimise(objects, fovs, setup, **arguments)


def test_frontier_alphas_are_real_risk_levels():
    for bad in (["0.05"], [True], [float("nan")], [0.0], [1.0]):
        with pytest.raises(ValueError, match="risk levels"):
            frontier(object.__new__(ReplayResult), [], criterion="c", alphas=bad, unit_cols=["g"])
