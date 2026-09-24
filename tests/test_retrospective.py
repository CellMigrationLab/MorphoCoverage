"""Tests for the retrospective layer: outcome criteria, hierarchy and sampling reliability."""

import json
import warnings

import numpy as np
import pandas as pd
import pytest
from scenarios import FEATURE_COLS, SCENARIOS

from morphocoverage.evaluator import iter_analysis_groups
from morphocoverage.outcomes import OutcomeMetric, RareStateDetected, RelativeMeanError
from morphocoverage.retrospective import (
    OutcomeCriterion,
    Preservation,
    SamplingHierarchy,
    sampling_reliability,
)
from morphocoverage.schema import fov_table_from_observations, validate_input_tables

MEAN_A = OutcomeCriterion(RelativeMeanError("feature_a"), 0.02, "lower_is_better")


def _reliability(scenario, criteria=(MEAN_A,), counts=(1, 3, 5, 10), **kwargs):
    built = SCENARIOS[scenario]()
    settings = {"reservoir_fraction": 0.3, "n_splits": 4, "draws": 10, "random_state": 0}
    settings.update(kwargs)
    return sampling_reliability(
        built.objects,
        built.fovs,
        feature_cols=list(FEATURE_COLS),
        criteria=list(criteria),
        fov_counts=list(counts),
        **settings,
    )


# --- OutcomeCriterion --------------------------------------------------------------


def test_criteria_judge_in_their_direction_and_never_preserve_undefined():
    lower = OutcomeCriterion(RelativeMeanError("x"), 0.05, "lower_is_better")
    higher = OutcomeCriterion(RareStateDetected("rare", 1), 1, "higher_is_better")
    assert lower.judge(0.05) is Preservation.PRESERVED
    assert lower.judge(0.051) is Preservation.NOT_PRESERVED
    assert higher.judge(1.0) is Preservation.PRESERVED
    assert higher.judge(0.0) is Preservation.NOT_PRESERVED
    for criterion in (lower, higher):
        assert criterion.judge(None) is Preservation.UNDEFINED
        assert criterion.judge(float("nan")) is Preservation.UNDEFINED
    assert lower.name == "relative_mean_error:x <= 0.05"
    assert higher.name == "rare_state_detected:rare:1 >= 1"


def test_criteria_are_serializable_provenance_with_a_plain_label():
    criterion = OutcomeCriterion(
        RelativeMeanError("x"), 0.05, "lower_is_better", "the well mean of x within 5%"
    )
    record = json.loads(json.dumps(criterion.parameters))
    assert record["threshold"] == 0.05 and record["direction"] == "lower_is_better"
    assert record["metric_config_id"] == RelativeMeanError("x").config_id
    assert criterion.label == "the well mean of x within 5%"
    assert MEAN_A.label == MEAN_A.name


@pytest.mark.parametrize(
    "build",
    [
        lambda: OutcomeCriterion("relative_mean_error", 0.05, "lower_is_better"),
        lambda: OutcomeCriterion(RelativeMeanError("x"), float("inf"), "lower_is_better"),
        lambda: OutcomeCriterion(RelativeMeanError("x"), True, "lower_is_better"),
        lambda: OutcomeCriterion(RelativeMeanError("x"), 0.05, "smaller"),
    ],
)
def test_invalid_criteria_are_refused(build):
    with pytest.raises((TypeError, ValueError)):
        build()


# --- SamplingHierarchy -------------------------------------------------------------


def _plate_fovs():
    # Well names are reused across plates, as in idr0028.
    return pd.DataFrame(
        {
            "image_file": ["a", "b", "c", "d", "e", "f"],
            "well": ["J23", "J23", "B6", "J23", "J23", "K16"],
            "plate": ["1A", "1A", "1A", "2A", "2A", "2A"],
        }
    )


def test_units_are_identified_with_every_outer_level():
    hierarchy = SamplingHierarchy("image_file", "well", "plate")
    hierarchy.validate(_plate_fovs())
    assert hierarchy.levels == ("image_file", "well", "plate")
    assert hierarchy.independent_unit == "plate"
    # J23 of 1A and J23 of 2A are two samples.
    assert hierarchy.count(_plate_fovs(), "well") == 4
    assert hierarchy.count(_plate_fovs(), "plate") == 2
    assert SamplingHierarchy("image_file", "well").independent_unit == "well"
    assert json.loads(json.dumps(hierarchy.parameters))["biological_repeat"] == "plate"


def test_hierarchy_validation_names_the_problem():
    hierarchy = SamplingHierarchy("image_file", "well", "plate")
    with pytest.raises(ValueError, match="missing from the FOV table"):
        hierarchy.validate(_plate_fovs().drop(columns="plate"))
    with pytest.raises(ValueError, match="missing values"):
        hierarchy.validate(_plate_fovs().assign(well=[None, "J23", "B6", "J23", "J23", "K16"]))
    with pytest.raises(ValueError, match="more than once"):
        hierarchy.validate(pd.concat([_plate_fovs(), _plate_fovs().iloc[:1]]))
    with pytest.raises(ValueError, match="distinct"):
        SamplingHierarchy("well", "well")
    with pytest.raises(ValueError, match="not a level"):
        hierarchy.unit_key("condition")


# --- fov_table_from_observations ---------------------------------------------------


def test_fov_table_from_observations_warns_and_carries_fov_level_columns():
    built = SCENARIOS["zero_object_fovs"]()
    objects = built.objects.merge(built.fovs[["image_file", "stage_x"]], on="image_file")
    with pytest.warns(UserWarning, match="without observations are missing"):
        table = fov_table_from_observations(
            objects, ["condition", "repeat", "image_file"], carry=["stage_x"]
        )
    # The four acquired empty FOVs are lost, which is exactly what the warning says.
    assert len(table) == len(built.fovs) - len(built.truth["empty_fovs"])
    assert not set(built.truth["empty_fovs"]) & set(table["image_file"])
    assert list(table.columns) == ["condition", "repeat", "image_file", "stage_x"]
    with pytest.raises(ValueError, match="varies within FOV"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fov_table_from_observations(objects, ["image_file"], carry=["feature_a"])


# --- sampling_reliability ----------------------------------------------------------


class _Probe(OutcomeMetric):
    """Records what each assessment saw; returns 0 so everything is preserved."""

    name = "probe"

    def __init__(self, data):
        self.data = data
        self.calls = []

    def evaluate(self, full, stopped):
        key = list(self.data.fov_key)
        held = set(full.fovs[key].itertuples(index=False, name=None))
        seen = set(stopped.fovs[key].itertuples(index=False, name=None))
        cells = stopped.objects.groupby(key).size()
        self.calls.append((frozenset(held), frozenset(seen), len(stopped.fovs), cells))
        return 0.0


def _probe_run(scenario):
    built = SCENARIOS[scenario]()
    data = validate_input_tables(built.objects, fovs=built.fovs, feature_cols=list(FEATURE_COLS))
    probe = _Probe(data)
    criterion = OutcomeCriterion(probe, 0.0, "lower_is_better")
    result = sampling_reliability(
        built.objects,
        built.fovs,
        feature_cols=list(FEATURE_COLS),
        criteria=[criterion],
        fov_counts=[2, 5, 8],
        reservoir_fraction=0.25,
        n_splits=3,
        draws=5,
    )
    return data, probe, result


def test_whole_fovs_are_drawn_and_never_from_the_reservoir():
    data, probe, _ = _probe_run("unequal_objects_per_fov")
    full_counts = data.objects.groupby(list(data.fov_key)).size()
    for held, seen, _, cells in probe.calls:
        assert not held & seen
        for fov, count in cells.items():
            assert full_counts[fov] == count  # every cell of a drawn FOV, never a subset


def test_every_fov_count_in_a_split_is_judged_against_the_same_reservoir():
    _, probe, _ = _probe_run("homogeneous")
    reservoirs = {}
    for held, seen, n, _ in probe.calls:
        if n in (2, 5, 8):
            reservoirs.setdefault(held, set()).add(n)
    assert len(reservoirs) == 3  # one reservoir per split
    assert all(counts == {2, 5, 8} for counts in reservoirs.values())


def test_fovs_without_observations_can_be_drawn_and_count_as_acquired():
    data, probe, _ = _probe_run("zero_object_fovs")
    key = list(data.fov_key)
    empty = {
        row
        for row in data.fovs[key].itertuples(index=False, name=None)
        if row[-1] in SCENARIOS["zero_object_fovs"]().truth["empty_fovs"]
    }
    assert any(seen & empty for _, seen, _, _ in probe.calls)
    assert all(len(seen) == n for _, seen, n, _ in probe.calls)


def test_groups_are_independent():
    built = SCENARIOS["condition_repeat_groups"]()
    both = _reliability("condition_repeat_groups", counts=(2, 4))
    data = validate_input_tables(built.objects, fovs=built.fovs, feature_cols=list(FEATURE_COLS))
    group, group_data = next(iter(iter_analysis_groups(data)))
    alone = sampling_reliability(
        group_data.objects,
        group_data.fovs,
        feature_cols=list(FEATURE_COLS),
        criteria=[MEAN_A],
        fov_counts=[2, 4],
        reservoir_fraction=0.3,
        n_splits=4,
        draws=10,
        random_state=0,
    )
    mask = np.ones(len(both.curve), dtype=bool)
    for column, value in group.items():
        mask &= both.curve[column].to_numpy() == value
    left = both.curve[mask].reset_index(drop=True)
    pd.testing.assert_frame_equal(left, alone.curve.reset_index(drop=True))
    assert len(both.curve) == 4 * 2


def test_deterministic_under_a_seed():
    first, again = _reliability("homogeneous"), _reliability("homogeneous")
    pd.testing.assert_frame_equal(first.curve, again.curve)
    assert not _reliability("homogeneous", random_state=5).curve.equals(first.curve)


def test_more_fields_preserve_a_homogeneous_mean_more_often():
    result = _reliability("homogeneous", counts=(1, 3, 10), n_splits=6, draws=30)
    preserved = result.curve.sort_values("fovs")["preserved"].to_numpy(dtype=float)
    assert preserved[0] < preserved[-1]
    assert result.curve["gain"].iloc[0] is None
    assert (result.curve["undefined_draws"] == 0).all()
    ceiling = result.ceiling["whole_stream_preserved"].iloc[0]
    assert preserved[-1] <= ceiling + 0.2


def test_a_clustered_rare_state_needs_many_fields():
    rare = OutcomeCriterion(RareStateDetected("true_state", 1), 1, "higher_is_better")
    result = _reliability(
        "rare_clustered",
        criteria=[rare],
        counts=(2, 10, 25),
        reservoir_fraction=0.3,
        n_splits=6,
        draws=30,
    )
    curve = result.curve.sort_values("fovs")
    preserved = curve["preserved"].to_numpy(dtype=float)
    assert preserved[0] < 0.2 and preserved[0] < preserved[-1]
    # Splits whose reservoir lacks the state cannot judge it: undefined, never preserved.
    assert (curve["undefined_draws"] + curve["defined_draws"] > 0).all()


def test_fov_counts_beyond_the_stream_are_not_reported():
    result = _reliability("condition_repeat_groups", counts=(2, 50))
    assert set(result.curve["fovs"]) == {2}


def test_summary_speaks_in_fields_and_states_its_limits():
    criterion = OutcomeCriterion(
        RelativeMeanError("feature_a"), 0.02, "lower_is_better", "the mean of feature_a within 2%"
    )
    text = _reliability("homogeneous", criteria=[criterion], counts=(1, 5)).summary()
    assert "how many fields do you need for the mean of feature_a within 2%?" in text
    assert "1 field: enough in" in text and "5 fields: enough in" in text
    assert "held-back fields" in text and "nothing is said beyond" in text
    assert "reservoir" not in text and "criterion" not in text


@pytest.mark.parametrize(
    "kwargs",
    [
        {"criteria": []},
        {"criteria": [MEAN_A, MEAN_A]},
        {"counts": (0, 3)},
        {"reservoir_fraction": 1.0},
        {"n_splits": 0},
        {"draws": 0},
        {"random_state": 0.5},
    ],
)
def test_invalid_settings_are_refused(kwargs):
    with pytest.raises((TypeError, ValueError)):
        _reliability("homogeneous", **kwargs)
