"""Question 6: a field's influence scales with its group (owner, 2026-10-05).

On the owner's ECM data, 3-5 images a group, leaving almost any image out moved its group's
mean by the fixed 10%, so 91 of 236 images were listed. A field is now listed only when its
leave-one-out change is also more than 3 robust SDs from its group's median change
(``inspection.stands_out``), and a group of fewer than 4 fields is not judged, named once.
The report's influence reads the values the contrast compared, raw with several controls.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from ecm_design import CONTROLS, MASTER, PARENTS, tables

from morphocoverage.retrospective import Summary, inspect_fovs
from morphocoverage.retrospective.inspection import stands_out
from morphocoverage.sampling_unit.contrast import condition_contrast
from morphocoverage.workflows import Setup, report
from morphocoverage.workflows._result_influence import with_result_influence

PLANTED = "drug_r1_0"


def _noisy(fields: dict[str, int], cv: float = 0.4, seed: int = 2) -> tuple:
    """``ctrl`` and ``drug`` in 3 repeats, ``fields[repeat]`` fields a group whose means
    vary by ``cv`` between fields (log-normal), 10 cells a field; field ``drug_r1_0`` holds
    4 times the others' values."""
    rng = np.random.default_rng(seed)
    rows, fovs = [], []
    for repeat, n in fields.items():
        for condition in ("ctrl", "drug"):
            for field in range(n):
                key = {
                    "condition": condition,
                    "biological_repeat": repeat,
                    "position_id": f"{condition}_{repeat}_{field}",
                }
                fovs.append(key)
                level = 10.0 * rng.lognormal(0.0, cv) * (4 if key["position_id"] == PLANTED else 1)
                rows += [{**key, "x": v} for v in rng.normal(level, 0.5, 10)]
    return pd.DataFrame(rows), pd.DataFrame(fovs)


def _on_result(objects: pd.DataFrame, fovs: pd.DataFrame, **kwargs):
    setup = Setup(readout="x", control={"condition": "ctrl"}, observation_unit="cell")
    contrast = condition_contrast(
        objects, fovs, feature="x", control=setup.control, observation_unit="cell"
    )
    inspection = inspect_fovs(objects, fovs, feature_cols=["x"], **setup.common)
    return with_result_influence(inspection, objects, fovs, contrast, setup, **kwargs)


def _flagged(result) -> set[str]:
    frame = result.fovs
    return set(frame.loc[frame["reason"] == "influence", "position_id"]) if len(frame) else set()


def test_a_change_stands_out_by_its_distance_from_the_groups_median_change():
    # A lone field 4 times the others moves the mean n - 1 times as much as a typical
    # field: never 3 times the median change in a group of 4, but far in robust SDs.
    values = np.array([10.0, 11.0, 9.5, 40.0])
    mean = values.mean()
    changes = ((4 * mean - values) / 3 - mean) / mean
    assert not abs(changes[3]) > 3 * np.median(np.abs(changes))
    assert list(stands_out(changes)) == [False, False, False, True]
    assert stands_out(changes, None).all()  # the threshold alone
    spread = np.array([5.0, 14.0, 9.0, 12.0, 7.0])  # noisy fields, none apart
    changes = ((5 * spread.mean() - spread) / 4 - spread.mean()) / spread.mean()
    assert (np.abs(changes) >= 0.10).sum() == 2 and not stands_out(changes).any()


def test_in_small_noisy_groups_only_the_planted_field_sways_the_result():
    objects, fovs = _noisy({"r1": 5, "r2": 5, "r3": 4})
    before = _flagged(_on_result(objects, fovs, spread=None))
    assert PLANTED in before and len(before) > 5  # the 10% alone: noise
    result = _on_result(objects, fovs)
    assert _flagged(result) == {PLANTED}
    assert result.settings["influence_spread"] == 3.0
    assert result.settings["skipped_small"] == []


def test_groups_of_fewer_than_four_fields_are_named_once_as_not_judged():
    objects, fovs = _noisy({"r1": 5, "r2": 3, "r3": 3})
    result = _on_result(objects, fovs)
    assert _flagged(result) == {PLANTED}
    small = result.settings["skipped_small"]
    assert len(small) == 4 and {s["reason"] for s in small} == {"fewer than 4 fields"}
    text = " ".join(result.summary().split())
    said = "Influence not judged: fewer than 4 fields in 4 groups (ctrl · repeat r2,"
    assert text.count(said) == 1
    full = " ".join(result.summary(detail="full").split())
    assert "Influence not judged: fewer than 4 fields in ctrl · repeat r2" in full


def test_the_field_weighted_summary_follows_the_same_rule():
    objects, fovs = _noisy({"r1": 5, "r2": 5, "r3": 3})
    common = {"feature_cols": ["x"], "summaries": [Summary("x", "mean")], "count_ratio": None}
    old = inspect_fovs(objects, fovs, influence_spread=None, **common)
    result = inspect_fovs(objects, fovs, **common)
    assert len(_flagged(old)) > 3 and _flagged(result) == {PLANTED}
    assert len(result.settings["skipped_small"]) == 2
    with pytest.raises(ValueError, match="influence_spread"):
        inspect_fovs(objects, fovs, influence_spread=0, **common)


def test_with_several_controls_the_influence_reads_the_raw_values_the_contrast_compares():
    # The report used to pass the values normalised to each control while the contrast
    # holds raw means, so every field "lowered" its mean by a made-up share.
    objects, fovs = tables(cells=4)
    objects = objects.drop(columns="technical_repeat")
    fovs = fovs.drop(columns="technical_repeat")
    bright = objects["fov_id"] == "VEC_SiFBN2_R2_w0_f0"
    objects.loc[bright, "area"] *= 3
    setup = Setup(
        readout="area",
        features=("area",),
        control=CONTROLS,
        control_parents=PARENTS,
        master_control=MASTER,
        observation_unit="cell",
        normalize="control_mean",
        fov_col="fov_id",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = report(objects, fovs, setup)
    found = result.results["inspection"].fovs
    found = found[found["reason"] == "influence"]
    assert list(found["fov_id"]) == ["VEC_SiFBN2_R2_w0_f0"]
    group = objects[(objects["condition"] == "VEC_SiFBN2") & (objects["biological_repeat"] == "R2")]
    mean, total = group["area"].mean(), len(group)
    expected = (mean * total - objects.loc[bright, "area"].sum()) / (total - bright.sum()) / mean
    assert found["value"].iloc[0] == pytest.approx(expected - 1)
