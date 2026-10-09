"""A missing paired p or power carries its reason, never inferred from NaN (#90, #308, #309)."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from morphocoverage.sampling_unit.contrast_repeats import (
    FEW_REPEATS,
    NO_SPREAD,
    untested_reason,
)
from morphocoverage.sampling_unit.repeat_plan import paired_power, repeats_needed
from morphocoverage.workflows import Setup, report


def _identical_effects(repeats: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Every repeat holds the same change (x2) exactly: the repeat effects have no spread."""
    rows, fovs = [], []
    for condition, level in (("ctrl", 10.0), ("drug", 20.0)):
        for r in range(repeats):
            for f in range(5):
                pid = f"{condition}{r}{f}"
                fovs.append(
                    {"position_id": pid, "condition": condition, "biological_repeat": f"R{r}"}
                )
                for k in range(4):
                    rows.append({**fovs[-1], "x": level + (k - 1.5), "y": 1.0 + (f + k) % 3})
    return pd.DataFrame(rows), pd.DataFrame(fovs)


def test_the_reason_is_read_from_the_repeats_held_not_from_a_missing_p():
    assert untested_reason(2, float("nan")) == FEW_REPEATS
    assert untested_reason(3, float("nan")) == NO_SPREAD
    assert untested_reason(5, 0.03) is None


@pytest.mark.parametrize("repeats", [3, 4])
def test_identical_effects_are_never_described_as_needing_three_repeats(repeats):
    objects, fovs = _identical_effects(repeats)
    setup = Setup(
        readout="x", features=("x", "y"), control={"condition": "ctrl"}, observation_unit="cell"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = report(objects, fovs, setup)
    across = result.results["contrast"].repeats
    assert list(across["status"]) == [NO_SPREAD]
    text = " ".join(result.summary(detail="full").split())
    assert "the paired test across repeats needs 3" not in text
    assert "has the same change in every biological repeat" in text


def test_power_planning_marks_a_zero_variance_as_degenerate_not_zero_power():
    assert np.isnan(paired_power(0.5, 0.0, 5, 0.05))
    assert np.isnan(repeats_needed(0.5, 0.0, 0.05))
    assert paired_power(1.0, 1.0, 1, 0.05) == 0.0  # no test can run on one repeat
    assert repeats_needed(0.5, 1.0, 0.05) >= 3  # an ordinary case is unchanged


@pytest.mark.parametrize(
    "call",
    [
        lambda: repeats_needed(0.5, 1.0, 5),
        lambda: repeats_needed(0.5, 1.0, 0.05, power=1.5),
        lambda: repeats_needed(np.nan, 1.0, 0.05),
        lambda: repeats_needed(0.5, -1.0, 0.05),
        lambda: repeats_needed("0.5", 1.0, 0.05),
        lambda: repeats_needed(True, 1.0, 0.05),
        lambda: paired_power(0.5, np.inf, 5, 0.05),
        lambda: paired_power(0.5, 1.0, 2.5, 0.05),
        lambda: paired_power(0.5, 1.0, True, 0.05),
    ],
)
def test_malformed_planning_parameters_are_refused(call):
    with pytest.raises(ValueError):
        call()
