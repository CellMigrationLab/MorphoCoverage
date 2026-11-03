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
from morphocoverage.retrospective.reliability import _clustered_se
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
            objects, ["condition", "biological_repeat", "image_file"], carry=["stage_x"]
        )
    # The four acquired empty FOVs are lost, which is exactly what the warning says.
    assert len(table) == len(built.fovs) - len(built.truth["empty_fovs"])
    assert not set(built.truth["empty_fovs"]) & set(table["image_file"])
    assert list(table.columns) == ["condition", "biological_repeat", "image_file", "stage_x"]
    with pytest.raises(ValueError, match="varies within FOV"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fov_table_from_observations(objects, ["image_file"], carry=["feature_a"])


def test_fov_table_from_observations_reads_carry_once():
    """A generator of carried columns is still checked for constancy within each FOV (#421).

    It was consumed by the column selection, so the check looped over nothing and
    ``drop_duplicates`` kept an arbitrary first stage position of a field with two.
    """
    objects = pd.DataFrame({"image_file": ["A", "A"], "stage_x": [10.0, 20.0], "f": [1.0, 2.0]})
    for carry in (["stage_x"], ("stage_x",), (c for c in ["stage_x"])):
        with pytest.raises(ValueError, match="'stage_x' varies within FOV"):
            fov_table_from_observations(objects, ["image_file"], carry=carry)
    with pytest.raises(ValueError, match="named twice"):
        fov_table_from_observations(objects, ["image_file"], carry=["stage_x", "stage_x"])
    with pytest.raises(TypeError, match="not a string"):
        fov_table_from_observations(objects, ["image_file"], carry="stage_x")
    constant = objects.assign(stage_x=10.0)
    with pytest.warns(UserWarning, match="without observations are missing"):
        table = fov_table_from_observations(
            constant, (c for c in ["image_file"]), carry=(c for c in ["stage_x"])
        )
    assert table.to_dict("list") == {"image_file": ["A"], "stage_x": [10.0]}


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
    for held, _, n, _ in probe.calls:
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
        # 10 splits: with 6, no hold-out of the label-independent seeds (audit 4, P9-2)
        # split the two state fields between reservoir and stream.
        n_splits=10,
        draws=30,
    )
    curve = result.curve.sort_values("fovs")
    preserved = curve["preserved"].to_numpy(dtype=float)
    assert preserved[0] < 0.2 and preserved[0] < preserved[-1]
    # Splits whose reservoir lacks the state cannot judge it: undefined, never preserved.
    assert (curve["undefined_draws"] + curve["defined_draws"] > 0).all()


def test_fov_counts_beyond_the_stream_say_so_instead_of_vanishing():
    result = _reliability("condition_repeat_groups", counts=(2, 50))
    beyond = result.curve[result.curve["fovs"] == 50]
    assert len(beyond) == 4  # one row per group, so no sample disappears downstream
    assert (beyond["splits"] == 0).all() and beyond["preserved"].isna().all()
    assert "50 fields: more than the" in result.summary()


def test_draws_of_only_empty_fovs_are_not_enough_rather_than_undefined():
    # Audit regression: 60% of FOVs are empty; a draw of empty FOVs used to be
    # dropped as undefined, which inflated the share at 1 field to about 97%.
    rng = np.random.default_rng(0)
    names = [f"f{i:02d}" for i in range(40)]
    fovs = pd.DataFrame({"condition": "c", "biological_repeat": "r", "image_file": names})
    full = [n for i, n in enumerate(names) if i % 5 < 2]
    objects = pd.DataFrame(
        {
            "condition": "c",
            "biological_repeat": "r",
            "image_file": np.repeat(full, 20),
            "feature_a": rng.normal(10, 0.1, 20 * len(full)),
        }
    )
    result = sampling_reliability(
        objects,
        fovs,
        feature_cols=["feature_a"],
        criteria=[OutcomeCriterion(RelativeMeanError("feature_a"), 0.05, "lower_is_better")],
        fov_counts=[1],
        reservoir_fraction=0.3,
        n_splits=5,
        draws=100,
    )
    row = result.curve.iloc[0]
    assert row["undefined_draws"] == 0
    assert 0.25 < row["preserved"] < 0.55  # about the 40% of FOVs with observations


def test_the_split_does_not_depend_on_the_row_order_of_the_fov_table():
    built = SCENARIOS["homogeneous"]()
    settings = {
        "feature_cols": list(FEATURE_COLS),
        "criteria": [MEAN_A],
        "fov_counts": [3],
        "reservoir_fraction": 0.3,
        "n_splits": 3,
        "draws": 10,
    }
    forward = sampling_reliability(built.objects, built.fovs, **settings)
    backward = sampling_reliability(built.objects, built.fovs.iloc[::-1], **settings)
    pd.testing.assert_frame_equal(forward.curve, backward.curve)
    pd.testing.assert_frame_equal(forward.ceiling, backward.ceiling)


def test_summary_speaks_in_fields_and_states_its_limits():
    criterion = OutcomeCriterion(
        RelativeMeanError("feature_a"), 0.02, "lower_is_better", "the mean of feature_a within 2%"
    )
    result = _reliability("homogeneous", criteria=[criterion], counts=(1, 5))
    text = result.summary(detail="full")
    assert "how many fields do you need for the mean of feature_a within 2%?" in text
    # Every listed field count has its row, with its share, in the full form's table.
    rows = [line.split() for line in text.splitlines()]
    shares = {int(r[0]): r[1] for r in rows if len(r) > 1 and r[0].isdigit() and "%" in r[1]}
    assert {1, 5} <= set(shares) and all(value.endswith("%") for value in shares.values())
    assert "held-back fields" in text and "nothing is said beyond" in text
    short = result.summary()
    assert "the mean of feature_a within 2%" in short and "nothing is said beyond" in short
    for words in (text, short):
        assert "reservoir" not in words and "criterion" not in words


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


# 4 analysis groups: 2 workers take whole groups, 6 workers take every split apart.
@pytest.mark.parametrize("n_jobs", [2, 6], ids=["groups", "splits_apart"])
def test_worker_processes_give_identical_curves(n_jobs):
    one = _reliability("condition_repeat_groups", counts=(2, 4))
    two = _reliability("condition_repeat_groups", counts=(2, 4), n_jobs=n_jobs)
    pd.testing.assert_frame_equal(one.curve, two.curve)
    pd.testing.assert_frame_equal(one.draws, two.draws)
    pd.testing.assert_frame_equal(one.ceiling, two.ceiling)


def _three_blocks():
    """Twelve fields in three blocks whose means differ, so each hold-out is another reference."""
    rng = np.random.default_rng(235)
    objects, fovs = [], []
    for i in range(12):
        where = {"condition": "A", "biological_repeat": "r1", "position_id": f"p{i:02d}"}
        fovs.append({**where, "block": f"b{i % 3}"})
        offset = (9.6, 10.0, 10.4)[i % 3]
        objects += [{**where, "f": offset + rng.normal(0, 1)} for _ in range(6)]
    return pd.DataFrame(objects), pd.DataFrame(fovs)


def test_issue_235_the_mc_se_stops_shrinking_once_the_hold_outs_are_used_up():
    objects, fovs = _three_blocks()
    criterion = OutcomeCriterion(RelativeMeanError("f"), 0.03, "lower_is_better")
    settings = {"feature_cols": ["f"], "criteria": [criterion], "fov_counts": [2]}
    settings.update(reservoir_fraction=0.3, draws=20)
    rows = {}
    for block_col in ("block", None):
        for n_splits in (12, 120):
            result = sampling_reliability(
                objects, fovs, n_splits=n_splits, block_col=block_col, **settings
            )
            row = result.curve.iloc[0]
            rows[block_col, n_splits] = (row, result.ceiling["distinct_holdouts"].iloc[0])
    # Three blocks give three references: ten times the splits do not divide the error
    # by sqrt(10), as an SD over every split would.
    (few, three), (many, also_three) = rows["block", 12], rows["block", 120]
    assert three == also_three == 3
    assert many["preserved_mc_se"] > 0.6 * few["preserved_mc_se"]
    assert many["preserved_mc_se"] > 3 * many["preserved_split_sd"] / np.sqrt(120)
    # Genuinely distinct hold-outs: the SD over splits divided by sqrt(splits), as before.
    row, distinct = rows[None, 12]
    assert distinct == 12
    assert row["preserved_mc_se"] == pytest.approx(row["preserved_split_sd"] / np.sqrt(12))


def test_issue_235_repeating_a_hold_out_adds_no_reference():
    shares = [0.2, 0.5, 0.8]
    expected = float(np.std(shares, ddof=1) / np.sqrt(3))
    for repeats in (1, 5, 50):
        per_split = pd.Series([s for s in shares for _ in range(repeats)])
        holdouts = pd.Series([h for h in "abc" for _ in range(repeats)])
        assert _clustered_se(per_split, holdouts) == pytest.approx(expected)
    # One reference, however many splits: no Monte Carlo error over references.
    assert _clustered_se(pd.Series([0.2, 0.4]), pd.Series(["a", "a"])) is None
