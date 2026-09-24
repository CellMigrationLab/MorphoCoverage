"""Tests for the inspect-these-fields report (roadmap Step 5)."""

import numpy as np
import pandas as pd
import pytest

from morphocoverage.retrospective import Summary, inspect_fovs
from morphocoverage.strategies import SupportNoveltyEstimator

FEATURES = ["feature_a", "feature_b"]
NEW, TAIL, DENSE = "f03", "f07", "f12"


def _acquisition(seed=5, duplicate_tail=1):
    """20 FOVs of 20 cells; one holds a new state, one a feature_b tail, one is dense.

    The tail cells are spread along feature_a, so they are not one new state.
    """
    rng = np.random.default_rng(seed)
    names = [f"f{i:02d}" for i in range(20)]
    fovs = pd.DataFrame(
        {
            "condition": "control",
            "repeat": "R1",
            "image_file": names,
            "stage_x": np.arange(20) % 5 * 100.0,
            "stage_y": np.arange(20) // 5 * 100.0,
        }
    )
    rows = []
    for name in names:
        n = 100 if name == DENSE else 20
        values = rng.normal(0.0, 1.0, size=(n, 2))
        if name == TAIL:
            tail = np.column_stack([np.linspace(-15, 15, 19), np.linspace(3, 5, 19)])
            values = np.repeat(np.vstack([values[:1], tail]), duplicate_tail, axis=0)
        if name == NEW:
            state = np.column_stack([rng.normal(8, 0.1, 8), rng.normal(0, 0.1, 8)])
            values = np.vstack([values, state])
        rows.append(pd.DataFrame(values, columns=FEATURES).assign(image_file=name))
    objects = pd.concat(rows, ignore_index=True).assign(condition="control", repeat="R1")
    return objects, fovs


NOVELTY = SupportNoveltyEstimator(1.5, 1, None, 5, 5)


def _inspect(objects, fovs, **kwargs):
    settings = {"summaries": [Summary("feature_b", 0.97)], "novelty": NOVELTY}
    settings.update(kwargs)
    return inspect_fovs(objects, fovs, feature_cols=FEATURES, **settings)


def test_each_constructed_field_is_listed_for_its_reason_and_no_other_field_is():
    result = _inspect(*_acquisition())
    listed = result.fovs.groupby("image_file")["reason"].apply(set).to_dict()
    assert listed == {NEW: {"new state"}, TAIL: {"influence"}, DENSE: {"count"}}
    new = result.fovs[result.fovs["reason"] == "new state"].iloc[0]
    assert new["value"] == 8 and new["observations"] == 28
    tail = result.fovs[result.fovs["reason"] == "influence"].iloc[0]
    assert tail["value"] < -0.1 and "lowers the 97th percentile of feature_b" in tail["detail"]
    dense = result.fovs[result.fovs["reason"] == "count"].iloc[0]
    assert dense["observations"] == 100 and dense["median_observations"] == 20


def test_duplicated_cells_do_not_raise_a_fields_influence():
    once = _inspect(*_acquisition(duplicate_tail=1)).fovs
    thrice = _inspect(*_acquisition(duplicate_tail=3)).fovs
    value = once.loc[once["reason"] == "influence", "value"].iloc[0]
    again = thrice.loc[thrice["reason"] == "influence", "value"].iloc[0]
    assert value == pytest.approx(again)


def test_empty_fields_and_neighbour_differences_are_traceable():
    objects, fovs = _acquisition()
    objects = objects[objects["image_file"] != "f15"]
    shifted = objects["image_file"] == "f18"
    objects.loc[shifted, "feature_a"] += 6
    result = _inspect(objects, fovs, stage_cols=("stage_x", "stage_y"), summaries=[])
    rows = result.fovs.set_index(["image_file", "reason"])
    assert rows.loc[("f15", "count"), "observations"] == 0
    assert "no observations" in rows.loc[("f15", "count"), "detail"]
    assert "above its 4 nearest fields" in rows.loc[("f18", "neighbours"), "detail"]
    assert rows.loc[("f18", "neighbours"), "stage_x"] == 300.0
    text = result.summary()
    assert "f18 (stage_x 300.0, stage_y 300.0)" in text
    for word in ("outlier", "artifact", "bad", "score"):
        assert word not in text.lower().replace("none of these is called an error", "")


def test_thresholds_are_declared_and_nothing_is_listed_below_them():
    quiet = _inspect(*_acquisition(), novelty=None, summaries=[], count_ratio=10)
    assert quiet.fovs.empty
    assert "No field stands out" in quiet.summary()
    with pytest.raises(ValueError, match="among feature_cols"):
        _inspect(*_acquisition(), summaries=[Summary("area", "mean")])
    with pytest.raises(ValueError):
        Summary("feature_a", 1.5)
    with pytest.raises(ValueError):
        _inspect(*_acquisition(), count_ratio=1)
