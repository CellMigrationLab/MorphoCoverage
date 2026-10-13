"""Tests for the inspect-these-fields report (R5 of docs/report/questions.md)."""

from __future__ import annotations

import inspect as source
import re

import numpy as np
import pandas as pd
import pytest
from _helpers_retrospective import (
    DENSE,
    EMPTY,
    FEATURES,
    INSPECTION_NOVELTY,
    NEW,
    TAIL,
    _acquisition,
    _grid_well,
    _homogeneous_tables,
    _inspect,
    _plate,
    _pooled,
)
from _helpers_retrospective import FEATURES as INSPECTED
from _helpers_tables import _labelled_tables
from scenarios import FEATURE_COLS, SCENARIOS

from morphocoverage._wording import percent
from morphocoverage.bounds import upper_bound
from morphocoverage.retrospective import Summary, inspect_fovs, report
from morphocoverage.retrospective.inspection import InspectionResult
from morphocoverage.retrospective.report_profile import _new_state_rows


def test_each_constructed_field_is_listed_for_its_reason_and_no_other_field_is():
    result = _inspect(*_acquisition())
    listed = result.fovs.groupby("image_file")["reason"].apply(set).to_dict()
    assert listed == {NEW: {"new state"}, TAIL: {"influence"}, DENSE: {"count"}}
    new = result.fovs[result.fovs["reason"] == "new state"].iloc[0]
    assert new["value"] == 8 and new["observations"] == 28
    tail = result.fovs[result.fovs["reason"] == "influence"].iloc[0]
    assert tail["value"] < -0.1
    assert "lowers the field-weighted 97th percentile of feature_b" in tail["detail"]
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
    assert "no cells" in rows.loc[("f15", "count"), "detail"]
    assert "above its 4 nearest fields" in rows.loc[("f18", "neighbours"), "detail"]
    assert rows.loc[("f18", "neighbours"), "stage_x"] == 300.0
    text = result.summary(detail="full")
    assert "f18 (stage_x 300.0, stage_y 300.0)" in text
    assert "f18" in result.summary()
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


def test_zero_centred_features_are_not_checked_for_relative_influence():
    rng = np.random.default_rng(2)
    names = [f"f{i:02d}" for i in range(20)]
    fovs = pd.DataFrame({"condition": "c", "biological_repeat": "r", "image_file": names})
    objects = pd.DataFrame(
        {
            "condition": "c",
            "biological_repeat": "r",
            "image_file": np.repeat(names, 20),
            "a": rng.normal(0, 1, 400),
        }
    )
    result = inspect_fovs(
        objects, fovs, feature_cols=["a"], summaries=[Summary("a", "mean")], count_ratio=None
    )
    assert result.fovs.empty  # used to flag 19 of 20 fields at about 99%
    assert "influence on the mean of a was not checked" in result.summary()


def test_a_lone_deviant_field_is_found_in_a_small_group():
    names = [f"f{i}" for i in range(8)]
    fovs = pd.DataFrame(
        {
            "condition": "c",
            "biological_repeat": "r",
            "image_file": names,
            "stage_x": np.arange(8) * 100.0,
            "stage_y": 0.0,
        }
    )
    rng = np.random.default_rng(3)
    values = rng.normal(10, 1, 80)
    values[:10] += 100
    objects = pd.DataFrame(
        {
            "condition": "c",
            "biological_repeat": "r",
            "image_file": np.repeat(names, 10),
            "a": values,
        }
    )
    result = inspect_fovs(
        objects,
        fovs,
        feature_cols=["a"],
        stage_cols=("stage_x", "stage_y"),
        n_neighbours=3,
        count_ratio=None,
    )
    assert set(result.fovs["image_file"]) == {"f0"}


def test_fields_at_the_same_stage_position_are_not_their_own_neighbours():
    names = [f"f{i}" for i in range(10)]
    fovs = pd.DataFrame(
        {
            "condition": "c",
            "biological_repeat": "r",
            "image_file": names,
            "stage_x": 0.0,
            "stage_y": 0.0,
        }
    )
    rng = np.random.default_rng(4)
    objects = pd.DataFrame(
        {
            "condition": "c",
            "biological_repeat": "r",
            "image_file": np.repeat(names, 10),
            "a": rng.normal(10, 1, 100),
        }
    )
    result = inspect_fovs(
        objects, fovs, feature_cols=["a"], stage_cols=("stage_x", "stage_y"), count_ratio=None
    )
    assert result.fovs.empty


def test_a_group_whose_stage_neighbours_cannot_be_compared_says_so():
    objects, fovs = _acquisition()
    fovs.loc[3, "stage_x"] = np.nan
    result = _inspect(objects, fovs, stage_cols=("stage_x", "stage_y"))
    assert result.settings["skipped_neighbours"] == [
        {
            "group": "condition=control, biological_repeat=R1",
            "reason": "stage coordinates are missing",
            "label": "control · repeat R1",
        }
    ]
    assert "Stage neighbours were not compared" in result.summary()
    assert "neighbour" not in set(result.fovs["reason"])


def test_a_generator_of_features_is_read_once():
    # The summaries' membership check used to consume it before validation (#222).
    objects, fovs = _acquisition()
    listed = _inspect(objects, fovs)
    generated = inspect_fovs(
        objects,
        fovs,
        feature_cols=(c for c in FEATURES),
        summaries=[Summary("feature_b", 0.97)],
        novelty=INSPECTION_NOVELTY,
    )
    pd.testing.assert_frame_equal(listed.fovs, generated.fovs)


def test_one_stage_axis_given_twice_is_refused():
    # The neighbours of a diagonal line are not those of the stage (#232).
    with pytest.raises(ValueError, match="not one axis twice"):
        _inspect(*_acquisition(), stage_cols=("stage_x", "stage_x"))


def test_the_inspection_refuses_a_label_column_that_is_not_there():
    objects, fovs = _labelled_tables(lambda n: [0] * n)
    with pytest.raises(ValueError, match="is not a column of the observations"):
        inspect_fovs(objects, fovs, feature_cols=["x"], fov_col="position_id", label_col="nope")


def _spots():
    """Two conditions of 12 fields: 3 fields of A without spots; B's w is negative."""
    rng = np.random.default_rng(0)
    fov_rows, object_rows = [], []
    for condition in ("A", "B"):
        for i in range(12):
            name = f"{condition}{i:02d}"
            fov_rows.append({"condition": condition, "image_file": name})
            spots = np.zeros(20) if i < 3 else rng.poisson(5, 20).astype(float) + 1
            z = rng.normal(5, 1, 20) if condition == "A" else rng.normal(-3, 1, 20)
            object_rows.append(
                pd.DataFrame({"image_file": name, "condition": condition, "spots": spots, "w": z})
            )
    return pd.concat(object_rows, ignore_index=True), pd.DataFrame(fov_rows)


def test_g8_a_summary_that_is_zero_is_reported_not_dropped():
    objects, fovs = _spots()
    result = inspect_fovs(
        objects,
        fovs,
        feature_cols=["spots"],
        count_ratio=None,
        summaries=[Summary("spots", 0.1)],
        group_cols=["condition"],
    )
    skipped = pd.DataFrame(result.settings["skipped_summaries"]).set_index("group")
    assert skipped.loc["condition=A", "reason"].startswith("is 0 here")
    # The group by its value (the section names its column), once per group in full.
    assert "was not checked in A: it is 0 here" in result.summary(detail="full")
    assert "was not checked in 2 groups (A and B): it is 0 here" in " ".join(
        result.summary().split()
    )


def test_g9_a_summary_skipped_in_one_group_names_that_group():
    objects, fovs = _spots()
    result = inspect_fovs(
        objects,
        fovs,
        feature_cols=["w"],
        count_ratio=None,
        summaries=[Summary("w", "mean")],
        group_cols=["condition"],
    )
    skipped = result.settings["skipped_summaries"]
    assert [(s["group"], s["summary"]) for s in skipped] == [("condition=B", "mean of w")]
    assert result.settings["checked"] == ["influence"]  # checked in A
    text = result.summary()
    assert "The influence on the mean of w was not checked in B" in text
    assert "(influence checked)" in text
    assert "Analysis groups: condition." in result.summary(detail="full")


def test_g10_nothing_checked_is_not_called_nothing_stands_out():
    objects, fovs = _spots()
    result = inspect_fovs(
        objects, fovs, feature_cols=["spots"], count_ratio=None, group_cols=["condition"]
    )
    assert result.settings["checked"] == []
    assert result.summary().startswith("No field was checked")
    assert "No field stands out" not in result.summary()
    # Only a reason that was skipped everywhere is not a check either.
    negative = inspect_fovs(
        objects[objects["condition"] == "B"],
        fovs[fovs["condition"] == "B"],
        feature_cols=["w"],
        summaries=[Summary("w", "mean")],
        count_ratio=None,
    )
    assert negative.summary().startswith("No field was checked")


def test_g11_the_influence_threshold_is_a_positive_number():
    objects, fovs = _acquisition()
    for threshold in (None, 0, -0.1, "0.1", np.inf, True):
        with pytest.raises(ValueError, match="influence_threshold"):
            inspect_fovs(objects, fovs, feature_cols=INSPECTED, influence_threshold=threshold)
    with pytest.raises(ValueError, match="neighbour_threshold"):
        inspect_fovs(objects, fovs, feature_cols=INSPECTED, neighbour_threshold=None)
    inspect_fovs(objects, fovs, feature_cols=INSPECTED, influence_threshold=np.float32(0.2))


def test_g3_the_expected_count_gives_the_rates_upper_bound():
    inspection = InspectionResult(
        pd.DataFrame({"reason": ["new state"] * 4, "condition": list("abcd"), "f": list("wxyz")}),
        ("condition",),
        {"fov_key": ["condition", "f"], "n_fovs": 336, "n_fovs_judged": 336},
    )
    statement = _new_state_rows(inspection, _pooled(reference_lofo=(0, 56)))[0]["statement"]
    high = upper_bound(0, 56, 0.95)
    # "about 0, up to 18": never "0 to 17.5 ... expected by chance" (audit 10, A20).
    assert "false alarms alone would give about 0, up to 18, of the 336 fields judged" in statement
    assert "so these are compatible with chance" in statement
    assert f"0 of 56 fields, 0%, at most {percent(high)} at 95% confidence" in statement
    assert high == pytest.approx(0.052, abs=1e-3)
    assert "about 0 of 336" not in statement


def _counted(counts, stage=False):
    names = [f"f{i}" for i in range(len(counts))]
    fovs = pd.DataFrame({"condition": "a", "biological_repeat": "r", "image_file": names})
    if stage:
        fovs["x"] = [float(i % 5) for i in range(len(names))]
        fovs["y"] = [float(i // 5) for i in range(len(names))]
    rng = np.random.default_rng(len(counts))
    rows = [
        {"condition": "a", "biological_repeat": "r", "image_file": name, "v": 10.0 + value}
        for name, n in zip(names, counts, strict=True)
        for value in rng.normal(0.0, 1.0, n)
    ]
    return pd.DataFrame(rows), fovs


def test_g11_one_observation_is_one_cell_not_one_cells():
    objects, fovs = _counted([1, 12, 15, 14, 13, 16])
    result = inspect_fovs(objects, fovs, feature_cols=["v"], observation_unit="cell")
    assert "f0: 1 cell; the median is 13.5." in result.summary(detail="full")
    assert re.search(r"f0 +1 cell; the median is 13\.5", result.summary())


def test_g12_an_empty_neighbour_is_not_counted_among_the_fields_compared():
    counts = [6] * 25
    counts[7] = 0  # the field above the centre holds no cell
    objects, fovs = _counted(counts, stage=True)
    objects.loc[objects["image_file"] == "f12", "v"] += 30.0
    result = inspect_fovs(
        objects, fovs, feature_cols=["v"], stage_cols=("x", "y"), count_ratio=None
    )
    rows = result.fovs[(result.fovs["reason"] == "neighbours")].set_index("image_file")
    detail = rows.loc["f12", "detail"]
    assert "above 3 of its 4 nearest fields (1 field without a value)" in detail


def test_g19_the_influence_reason_names_the_field_weighted_summary():
    objects, fovs = _counted([20] * 6)
    objects.loc[objects["image_file"] == "f0", "v"] += 20.0
    result = inspect_fovs(
        objects, fovs, feature_cols=["v"], summaries=[Summary("v", "mean")], count_ratio=None
    )
    detail = result.fovs.set_index("image_file").loc["f0", "detail"]
    assert detail.startswith("removing it lowers the field-weighted mean of v by")


def test_g21_percentile_labels_use_the_shared_ordinal_and_rates_one_lookup():
    labels = [Summary("area", q).label for q in (0.01, 0.22, 0.11, 0.025)]
    assert labels == [
        "1st percentile of area",
        "22nd percentile of area",
        "11th percentile of area",
        "2.5th percentile of area",
    ]
    assert "_ordinal_suffix" not in source.getsource(source.getmodule(Summary))
    assert 'set_index("candidate")' not in source.getsource(report.microscopist_report)


def test_p9_5_neighbour_ties_do_not_depend_on_the_row_order_of_the_fov_table():
    rng = np.random.default_rng(4)
    fovs, objects = _grid_well(rng, "w", 0.0, n=5)
    fovs, objects = pd.DataFrame(fovs), pd.DataFrame(objects)
    # One deviant field in the middle of a regular grid: its 4 nearest are tied with others.
    objects.loc[objects["position_id"] == "w_2_2", "x"] += 3.0
    objects.loc[objects["position_id"] == "w_1_3", "x"] -= 2.0

    def flagged(fov_table):
        result = inspect_fovs(
            objects,
            fov_table,
            feature_cols=["x"],
            summaries=[Summary("x", "mean")],
            stage_cols=("x_um", "y_um"),
            n_neighbours=3,
            neighbour_threshold=2.0,
        )
        rows = result.fovs[result.fovs["reason"] == "neighbours"]
        return sorted(zip(rows["position_id"], rows["detail"], strict=True))

    first = flagged(fovs)
    assert first and any("fields tied in distance taken by field name" in d for _, d in first)
    for seed in range(3):
        assert flagged(fovs.sample(frac=1, random_state=seed)) == first


def test_f5_f11_f13_inspection_says_what_it_could_not_look_for():
    from _helpers_retrospective import FEATURES, NOVELTY

    objects, fovs = _plate(shift=0.0)
    objects.loc[objects.index[0], "feature_b"] = np.nan
    result = inspect_fovs(
        objects,
        fovs,
        feature_cols=FEATURES,
        novelty=NOVELTY,
        count_ratio=None,
        group_cols=["condition", "biological_repeat"],
    )
    assert "New states were not looked for in" in result.summary()
    assert "missing values in feature_b (1 observation)" in result.summary()
    labels = [Summary("area", q).label for q in (0.01, 0.02, 0.03, 0.11, 0.22, 0.025)]
    assert labels[:5] == [f"{n} percentile of area" for n in ("1st", "2nd", "3rd", "11th", "22nd")]
    assert labels[5] == "2.5th percentile of area"


def test_f11_missing_values_never_make_a_field_weigh_less():
    objects = pd.DataFrame(
        {
            "image_file": ["a", "a", "a", "a", "b", "c"],
            "x": [1.0, np.nan, np.nan, np.nan, 3.0, 2.0],
        }
    )
    fovs = pd.DataFrame({"image_file": ["a", "b", "c"]})
    result = inspect_fovs(
        objects,
        fovs,
        feature_cols=["x"],
        summaries=[Summary("x", "mean")],
        count_ratio=None,
        influence_threshold=0.2,
        # The threshold alone: 3 fields are too few to judge whether one stands out.
        influence_spread=None,
    )
    # Every field weighs one: the mean is 2 and leaving out b gives 1.5 (a 25% change).
    detail = result.fovs.set_index("image_file")["detail"]
    assert detail["b"] == "removing it lowers the field-weighted mean of x by 25%"


def test_fields_with_observations_are_listed_when_most_fields_are_empty():
    # 3 fields with cells among 20: the median count is 0, and nothing was listed,
    # neither the empty fields nor the three that hold every observation.
    from morphocoverage.retrospective import inspect_fovs

    built = SCENARIOS["homogeneous"]()
    kept = built.fovs["image_file"].iloc[:3]
    objects = built.objects[built.objects["image_file"].isin(kept)]
    result = inspect_fovs(objects, built.fovs, feature_cols=list(FEATURE_COLS))
    listed = result.fovs[result.fovs["reason"] == "count"]
    assert sorted(listed["image_file"]) == sorted(kept)
    assert "most fields have none (the median is 0)" in result.summary()


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"novelty": "x"}, TypeError, "novelty must be a SupportNoveltyEstimator"),
        ({"influence_threshold": 0}, ValueError, "influence_threshold must be a positive number"),
        (
            {"count_ratio": 1},
            ValueError,
            "count_ratio must be greater than 1, or None to skip the count reason",
        ),
        ({"neighbour_threshold": 0}, ValueError, "neighbour_threshold must be a positive number"),
        (
            {"influence_spread": 0},
            ValueError,
            "influence_spread must be a positive number, or None",
        ),
        ({"n_neighbours": 0}, ValueError, "n_neighbours must be a positive integer"),
    ],
)
def test_inspect_fovs_checks_its_settings_first(kwargs, error, message):
    with pytest.raises(error) as raised:
        inspect_fovs(EMPTY, EMPTY, feature_cols=["a"], **kwargs)
    assert str(raised.value) == message


def test_inspect_fovs_skips_a_summary_without_values_and_checks_the_stage_columns():
    objects, fovs = _homogeneous_tables()
    result = inspect_fovs(
        objects.assign(empty=np.nan),
        fovs,
        feature_cols=[*FEATURE_COLS, "empty"],
        summaries=[Summary("feature_a", "mean"), Summary("empty", "mean")],
        count_ratio=None,
    )
    assert result.settings["skipped_summaries"] == [
        {
            "group": "condition=control, biological_repeat=R1",
            "summary": "mean of empty",
            "reason": "has no value here",
            "label": "control · repeat R1",
        }
    ]
    with pytest.raises(ValueError) as raised:
        inspect_fovs(objects, fovs, feature_cols=list(FEATURE_COLS), stage_cols=("x", "y"))
    assert str(raised.value) == "stage columns missing from the FOV table: ['x', 'y']"
