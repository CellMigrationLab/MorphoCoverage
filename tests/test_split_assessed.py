"""#301: repeats whose split into two groups cannot be assessed leave the denominator."""

import numpy as np
import pandas as pd

from morphocoverage.sampling_unit.distribution import (
    SHAPE_COLUMNS,
    DistributionContrast,
    unassessed_words,
)
from morphocoverage.workflows._distribution import split_words

_SPLIT = {"bimodality": 0.7, "kurtosis": -1.5, "control_bimodality": 0.4}
_ONE_PEAK = {"bimodality": 0.35, "kurtosis": 0.0, "control_bimodality": 0.4}


def _few(cells: int) -> dict:
    return {
        "bimodality": np.nan,
        "kurtosis": np.nan,
        "control_bimodality": 0.4,
        "observations": cells,
    }


def _result(repeats: dict[str, list[dict]]) -> DistributionContrast:
    rows = []
    for condition, shapes in repeats.items():
        for i, shape in enumerate(shapes):
            rows.append(
                {
                    "feature": "x",
                    "condition": condition,
                    "biological_repeat": f"r{i + 1}",
                    "observations": 50,
                    "control_observations": 50,
                    **shape,
                }
            )
    settings = {"observation_unit": "cell", "observation_unit_declared": True}
    return DistributionContrast(pd.DataFrame(), settings, pd.DataFrame(rows, columns=SHAPE_COLUMNS))


def test_unassessable_repeats_leave_the_denominator_and_are_named():
    """2 repeats split, 3 too small to assess: 2 of 2, not 2 of 5 (owner, 2026-10-05)."""
    result = _result({"drug": [_SPLIT, _SPLIT, _few(2), _few(3), _few(3)]})
    [row] = result.splits().to_dict("records")
    assert (row["split_repeats"], row["assessed"], row["repeats"]) == (2, 2, 5)
    assert row["not_assessed"] == "3 repeats not assessed: 2, 3 and 3 cells"
    words = split_words(result, "drug")
    assert "in 2 of 2 repeats assessed, 3 repeats not assessed: 2, 3 and 3 cells" in words
    assert result.not_assessed().empty


def test_an_unassessable_repeat_is_no_evidence_either_way():
    # 1 of 3 assessed repeats split: not half, whatever the fourth would have shown.
    result = _result({"drug": [_SPLIT, _ONE_PEAK, _ONE_PEAK, _few(3)]})
    assert result.splits().empty and split_words(result, "drug") == ""
    # 1 of 2 assessed: half, so it is said, the third named.
    result = _result({"drug": [_SPLIT, _ONE_PEAK, _few(3)]})
    [row] = result.splits().to_dict("records")
    assert (row["split_repeats"], row["assessed"]) == (1, 2)
    assert row["not_assessed"] == "1 repeat not assessed: 3 cells"


def test_every_repeat_assessable_keeps_the_rule():
    result = _result(
        {"drug": [_SPLIT, _SPLIT, _ONE_PEAK, _ONE_PEAK], "other": [_SPLIT] + [_ONE_PEAK] * 2}
    )
    [row] = result.splits().to_dict("records")
    assert row["condition"] == "drug" and (row["split_repeats"], row["assessed"]) == (2, 4)
    assert row["repeats"] == 4 and row["not_assessed"] == ""
    assert split_words(result, "drug").endswith(
        "in 2 of 4 repeats), where the control's do not (see its figure)"
    )


def test_a_condition_with_no_assessable_repeat_is_not_assessed():
    flat = {"bimodality": np.nan, "kurtosis": np.nan, "control_bimodality": 0.4}
    control_few = {**_ONE_PEAK, "control_bimodality": np.nan, "control_observations": 1}
    result = _result({"drug": [_few(3), flat, control_few]})
    assert result.splits().empty
    [row] = result.not_assessed().to_dict("records")
    assert row["repeats"] == 3 and row["reasons"] == "3 cells, 1 control cell and no spread"
    assert split_words(result, "drug") == (
        "a split into two groups is not assessed on x (3 repeats: 3 cells, 1 control cell and "
        "no spread)"
    )
    assert unassessed_words(result.shapes.iloc[:0]) == ""
