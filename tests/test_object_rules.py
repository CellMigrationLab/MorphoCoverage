"""Tests of ``sampling_unit.object_rules``: would an object-count rule hold with whole fields?"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from _helpers_sampling_unit import (
    CONTROL,
    _error,
    _field_rows,
    _flat_text,
    _join,
    _plain,
    _screen,
    _well_tables,
    _well_tables_with_empty_fields,
    _wells,
)
from _helpers_tables import _one_condition_tables, _variability

from morphocoverage._acquisition import check_one_acquisition
from morphocoverage._wording import percent_fine
from morphocoverage.retrospective.report import microscopist_report
from morphocoverage.sampling_unit import (
    condition_contrast,
    descriptor_spread_check,
    object_rule_check,
    unusual_objects,
)
from morphocoverage.sampling_unit.contrast_sampling import MIN_FIELDS


def test_the_count_rule_with_whole_fields_says_its_draws_repeat_fields() -> None:
    # A16: "fields taken" 4-18 of 3 acquired fields, the draws with replacement unsaid.
    rng = np.random.default_rng(12)
    rows = []
    for well, fields in (("w1", 3), ("w2", 8)):
        for field in range(fields):
            for label in range(20):
                rows.append(
                    {
                        "condition": "ctrl",
                        "biological_repeat": "r1",
                        "technical_repeat": well,
                        "image_file": f"{well}_{field}.tif",
                        "label_id": label,
                        "x": 10 + rng.normal(0, 1 + field % 3),
                    }
                )
    objects = pd.DataFrame(rows)
    fovs = objects[["condition", "biological_repeat", "technical_repeat", "image_file"]]
    result = object_rule_check(
        objects, fovs.drop_duplicates(), feature="x", observation_unit="cell", n_draws=50
    )
    status = dict(zip(result.table["technical_repeat"], result.table["status"], strict=False))
    assert status["w1"] == f"fewer than {MIN_FIELDS} fields with cells"
    assert status["w2"] == "ok"
    short = " ".join(result.summary().split())
    full = " ".join(result.summary("full").split())
    assert "drawn at random with replacement" in short
    assert "with replacement, until that many, so it can exceed the fields acquired" in full


def test_results_of_two_acquisitions_are_refused_before_any_text():
    first = _variability(*_one_condition_tables("f"))
    other_objects, other_fovs = _one_condition_tables("g", seed=1)
    second = object_rule_check(other_objects, other_fovs, feature="x", n_draws=50)
    with pytest.raises(ValueError, match="different acquisitions"):
        check_one_acquisition({"variability": first, "object_rules": second})
    with pytest.raises(ValueError, match="different acquisitions"):
        microscopist_report(variability=first, object_rules=second)


def test_h11_jcb_rules_on_negative_values_are_not_applicable():
    objects, fovs = _well_tables([("ctrl", "r1", "w1", 20, -2.0, 0.5)], per_field=30)
    result = object_rule_check(objects, fovs, feature="x", n_draws=50)
    assert result.table["status"].str.startswith("not applicable").all()
    assert "non-negative measurements" in " ".join(result.summary().split())
    # A positive group normalized by a control with negative values: also not applicable.
    treated, t_fovs = _well_tables([("drug", "r1", "w2", 10, 20.0, 0.5)], per_field=30)
    control, c_fovs = _well_tables([("ctrl", "r1", "w1", 10, 0.5, 0.5)], per_field=30)
    both = object_rule_check(
        pd.concat([treated, control]),
        pd.concat([t_fovs, c_fovs]),
        feature="x",
        control=CONTROL,
        n_draws=50,
    )
    assert both.table["status"].str.startswith("not applicable").all()
    fine = object_rule_check(treated, t_fovs, feature="x", n_draws=50)
    assert (fine.table["status"] == "ok").all()


def test_h8_a_constant_feature_is_reported_not_a_crash():
    objects, fovs = _well_tables_with_empty_fields([("ctrl", "r1", "w", 6, 0.0, 0.3)], per_field=20)
    for value in (0.1, 1.0):
        objects["x"] = value
        result = object_rule_check(objects, fovs, feature="x", n_draws=50)
        assert result.table["status"].str.startswith("the feature does not vary").all()
        assert "the feature does not vary" in " ".join(result.summary().split())


def test_h11_the_field_rule_is_also_given_in_acquired_fields():
    objects, fovs = _well_tables_with_empty_fields(
        [("ctrl", "r1", "w", 40, 10.0, 1.0)], per_field=10, empty_every=2
    )
    result = object_rule_check(objects, fovs, feature="x", targets=[0.05], n_draws=200)
    row = result.table.iloc[0]
    assert row["fovs_with_observations"] == 20 and row["fovs"] == 40
    assert row["fov_rule_acquired"] == 2 * row["fov_rule_fovs"]
    text = result.summary(detail="full")
    rule = f"{int(row['fov_rule_fovs'])} ({int(row['fov_rule_fovs_upper95'])})"
    acquired = f"{int(row['fov_rule_acquired'])} ({int(row['fov_rule_acquired_upper95'])})"
    assert "fields with cells" in text and "20 of 40" in text
    assert rule in text and acquired in text
    # The short form gives the fields to acquire.
    assert acquired in result.summary()


def test_h12_a_zero_control_mean_is_not_too_few_fields():
    spec = [("ctrl", "r1", "w", 6, 0.0, 0.0), ("drug", "r1", "w", 6, 3.0, 0.5)]
    objects, fovs = _well_tables_with_empty_fields(spec, per_field=30)
    rng = np.random.default_rng(0)
    counts = rng.poisson(2.0, len(objects)).astype(float)
    objects["mitoses"] = np.where(objects["condition"] == "ctrl", 0.0, counts)
    result = object_rule_check(objects, fovs, feature="mitoses", control=CONTROL, n_draws=50)
    assert result.table["status"].str.contains("the reference mean is 0").all()
    few = object_rule_check(
        *_well_tables_with_empty_fields([("ctrl", "r1", "w", 1, 10.0, 0.1)]), feature="x"
    )
    # Audit 10 (A16): below 5 fields the count rule is not assessable (owner, 2026-10-06).
    assert few.table["status"].eq("fewer than 5 fields with cells").all()


def _shifted(scale):
    objects, fovs = _wells(2, repeats=("r1", "r2"), fields=6)
    objects.loc[objects["biological_repeat"] == "r2", "x"] *= scale
    return objects, fovs


def test_p1_4_the_object_rule_divides_by_the_repeats_own_control_mean():
    objects, fovs = _shifted(2.0)
    result = object_rule_check(objects, fovs, feature="x", control=CONTROL, n_draws=50)
    table = result.table[result.table["target"] == 0.05]
    for repeat in ("r1", "r2"):
        in_repeat = objects[objects["biological_repeat"] == repeat]
        reference = in_repeat.loc[in_repeat["condition"] == "ctrl", "x"].mean()
        for row in table[table["biological_repeat"] == repeat].to_dict("records"):
            group = in_repeat[in_repeat["technical_repeat"] == row["technical_repeat"]]
            sd = (group["x"] / reference).std(ddof=1)
            assert row["rule_objects"] == int(np.ceil((sd / 0.05) ** 2))
    # The same CV in both repeats: the same rule, although the levels differ 2x.
    by_repeat = table.groupby("biological_repeat")["rule_objects"].median()
    assert by_repeat["r1"] == pytest.approx(by_repeat["r2"], rel=0.35)
    # Read unwrapped: the count rule's sentence grew (audit 10, A16), so its line breaks moved.
    text = " ".join(result.summary().split())
    assert "of the same biological repeat" in text
    assert "differ by 2.0x" in text


def test_p5_6_a_unit_change_in_one_repeat_is_flagged_and_a_repeat_without_control_refused():
    objects, fovs = _shifted(0.1)
    text = object_rule_check(objects, fovs, feature="x", control=CONTROL, n_draws=50).summary()
    text = " ".join(text.split())
    assert "differ by 10.0x" in text and "change of units" in text
    keep = ~((objects["biological_repeat"] == "r2") & (objects["condition"] == "ctrl"))
    result = object_rule_check(objects[keep], fovs, feature="x", control=CONTROL, n_draws=50)
    r2 = result.table[result.table["biological_repeat"] == "r2"]
    assert (r2["status"].str.contains("no control object in this group's repeat")).all()
    assert (result.table[result.table["biological_repeat"] == "r1"]["status"] == "ok").all()


def _sorted(table):
    """A result table in a fixed row order (groups come in order of first appearance)."""
    keys = [
        c for c in ("condition", "biological_repeat", "technical_repeat", "target", "n", "sampling")
    ]
    keys = [c for c in keys if c in table.columns]
    return table.sort_values(keys, kind="stable").reset_index(drop=True)


def test_p9_4_row_order_never_changes_the_contrast_draws_or_unusual_objects():
    objects, fovs = _wells(2, repeats=("r1", "r2"))
    rng = np.random.default_rng(3)
    shuffled_objects = objects.iloc[rng.permutation(len(objects))]
    shuffled_fovs = fovs.iloc[rng.permutation(len(fovs))]
    runs = [
        condition_contrast(o, f, feature="x", control=CONTROL, object_counts=(30,), n_draws=10)
        for o, f in ((objects, fovs), (shuffled_objects, shuffled_fovs))
    ]
    pd.testing.assert_frame_equal(runs[0].effects, runs[1].effects)
    pd.testing.assert_frame_equal(runs[0].table, runs[1].table)
    labels = [
        unusual_objects(o, f, feature_cols=["x", "y"], reference=CONTROL, per_fov=5).labels
        for o, f in ((objects, fovs), (shuffled_objects, shuffled_fovs))
    ]
    pd.testing.assert_series_equal(labels[0].sort_index(), labels[1].sort_index())
    rules = [
        object_rule_check(o, f, feature="x", control=CONTROL, n_draws=50).table
        for o, f in ((objects, fovs), (shuffled_objects, shuffled_fovs))
    ]
    pd.testing.assert_frame_equal(*(_sorted(r) for r in rules))
    spreads = [
        descriptor_spread_check(o, f, feature="x", threshold=40, n_draws=10).curves
        for o, f in ((objects, fovs), (shuffled_objects, shuffled_fovs))
    ]
    pd.testing.assert_frame_equal(*(_sorted(r) for r in spreads))


def test_g11_g12_spread_ratios_and_small_groups_are_stated():
    rng = np.random.default_rng(0)
    objects = pd.DataFrame(
        {
            "image_file": np.repeat([f"f{j}" for j in range(20)], 30),
            "x": rng.integers(0, 3, 600).astype(float),
            "condition": np.repeat(["A"] * 19 + ["B"], 30),
        }
    )
    fovs = objects[["image_file", "condition"]].drop_duplicates()
    text = descriptor_spread_check(
        objects, fovs, feature="x", threshold=200, group_cols=["condition"]
    ).summary()
    text = " ".join(text.split())
    assert "nanx" not in text and "no ratio" in text
    assert "the random-object samples never varied" in text
    assert "JCS check in B: not checked: 30 cells, fewer than the threshold of 200." in text


def test_control_means_are_compared_within_a_sub_experiment():
    objects, fovs = _screen()
    check = object_rule_check(
        objects, fovs, feature="x", control={"condition": "ctrl"}, observation_unit="cell"
    )
    # Set B is twice as bright as set A, but each set's repeats agree: no caution.
    assert "control means of the repeats differ" not in check.summary(detail="full")


def _object_rule_edges():
    from morphocoverage.sampling_unit import object_rule_check

    out = []
    ctrl = _field_rows("ctrl", "r1", 6, 10, 5.0, 0)
    neg = _field_rows("neg", "r1", 6, 10, -1.0, 1)
    flat = _field_rows("flat", "r1", 6, 10, 5.0, 2, sd=0.0)
    objects, fovs = _join(ctrl, neg, flat)
    own = object_rule_check(objects, fovs, feature="x", n_draws=50)
    out.append(own.settings["reference"])
    out.append(own.table[["condition", "target", "status"]].to_dict("records"))
    out.append(_error(lambda: object_rule_check(objects, fovs, feature="x", targets=[1.5])))
    out.append(_error(lambda: object_rule_check(objects, fovs, feature="x", repeat_col="plate")))
    no_repeat = fovs.drop(columns="biological_repeat")
    obj_no_repeat = objects.drop(columns="biological_repeat")
    implicit = object_rule_check(
        obj_no_repeat, no_repeat, feature="x", control={"condition": "ctrl"}, n_draws=50
    )
    out.append((implicit.settings["repeat_col"], implicit.settings["reference"]))
    zero_ctrl = _field_rows("ctrl", "r1", 6, 10, 0.0, 3, sd=0.0)
    objects2, fovs2 = _join(zero_ctrl, _field_rows("A", "r1", 6, 10, 2.0, 4))
    zero = object_rule_check(
        objects2, fovs2, feature="x", control={"condition": "ctrl"}, n_draws=50
    )
    out.append(zero.table[["condition", "status"]].to_dict("records"))
    empty_ctrl = objects.copy()
    empty_ctrl.loc[empty_ctrl["condition"] == "ctrl", "x"] = np.nan
    out.append(
        _error(
            lambda: object_rule_check(empty_ctrl, fovs, feature="x", control={"condition": "ctrl"})
        )
    )
    return out


def _descriptor_spread():
    from morphocoverage.sampling_unit import descriptor_spread_check

    objects, fovs = _join(
        _field_rows("A", "r1", 8, 10, 5.0, 0), _field_rows("B", "r1", 2, 4, 5.0, 1)
    )
    result = descriptor_spread_check(
        objects, fovs, feature="x", threshold=20, sizes=[10, 20, 40], n_draws=5
    )
    return (
        result.curves.round(12).to_dict("records"),
        _plain(result.at_threshold.round(12)),
        result.settings["skipped_groups"],
        result.settings["sizes"],
    )


EXPECTED_OBJECT_RULE_EDGES = [
    "each group's own mean",
    [
        {"condition": "ctrl", "target": 0.05, "status": "ok"},
        {"condition": "ctrl", "target": 0.025, "status": "ok"},
        {
            "condition": "neg",
            "target": 0.05,
            "status": (
                "not applicable: the feature takes negative values here or in the "
                "control, and an error relative to the mean depends on where zero "
                "lies; the mean-precision rule applies to non-negative measurements"
            ),
        },
        {
            "condition": "neg",
            "target": 0.025,
            "status": (
                "not applicable: the feature takes negative values here or in the "
                "control, and an error relative to the mean depends on where zero "
                "lies; the mean-precision rule applies to non-negative measurements"
            ),
        },
        {
            "condition": "flat",
            "target": 0.05,
            "status": (
                "the feature does not vary, so the mean of any objects is exact and "
                "the rule has nothing to average"
            ),
        },
        {
            "condition": "flat",
            "target": 0.025,
            "status": (
                "the feature does not vary, so the mean of any objects is exact and "
                "the rule has nothing to average"
            ),
        },
    ],
    ("ValueError", "targets must be fractions between 0 and 1, such as 0.05"),
    (
        "ValueError",
        (
            "repeat_col 'plate' is not a column of the FOV table; name the "
            "biological-repeat column, or pass repeat_col=None to pool every "
            "repeat's control"
        ),
    ),
    (None, "mean of all objects of the control groups, ctrl"),
    [
        {
            "condition": "ctrl",
            "status": (
                "not applicable: the reference mean is 0, so an error relative to it is undefined"
            ),
        },
        {
            "condition": "ctrl",
            "status": (
                "not applicable: the reference mean is 0, so an error relative to it is undefined"
            ),
        },
        {
            "condition": "A",
            "status": (
                "not applicable: the feature takes negative values here or in the "
                "control, and an error relative to the mean depends on where zero "
                "lies; the mean-precision rule applies to non-negative measurements"
            ),
        },
        {
            "condition": "A",
            "status": (
                "not applicable: the feature takes negative values here or in the "
                "control, and an error relative to the mean depends on where zero "
                "lies; the mean-precision rule applies to non-negative measurements"
            ),
        },
    ],
    ("ValueError", "the control groups have no observations of the feature"),
]


EXPECTED_DESCRIPTOR_SPREAD = (
    [
        {
            "condition": "A",
            "biological_repeat": "r1",
            "n": 10,
            "sampling": "random objects",
            "mean": 1.011882642154,
            "median": 1.336262605698,
            "SD": 0.321238548615,
            "IQR": 0.723321769231,
        },
        {
            "condition": "A",
            "biological_repeat": "r1",
            "n": 10,
            "sampling": "whole FOVs",
            "mean": 0.845868371038,
            "median": 0.798137152131,
            "SD": 0.367719164414,
            "IQR": 0.793376704051,
        },
        {
            "condition": "A",
            "biological_repeat": "r1",
            "n": 20,
            "sampling": "random objects",
            "mean": 0.323133860101,
            "median": 0.671891660735,
            "SD": 0.235482088344,
            "IQR": 0.419081793469,
        },
        {
            "condition": "A",
            "biological_repeat": "r1",
            "n": 20,
            "sampling": "whole FOVs",
            "mean": 0.442431208084,
            "median": 0.465370329454,
            "SD": 0.343440648224,
            "IQR": 0.867125557839,
        },
        {
            "condition": "A",
            "biological_repeat": "r1",
            "n": 40,
            "sampling": "random objects",
            "mean": 0.381956840632,
            "median": 0.554668861871,
            "SD": 0.196664982822,
            "IQR": 0.747053879725,
        },
        {
            "condition": "A",
            "biological_repeat": "r1",
            "n": 40,
            "sampling": "whole FOVs",
            "mean": 0.48332114371,
            "median": 0.471245317568,
            "SD": 0.256845232001,
            "IQR": 0.666273594412,
        },
    ],
    [
        {
            "condition": "A",
            "biological_repeat": "r1",
            "descriptor": "mean",
            "threshold": 20,
            "random_objects_spread": 0.323133860101,
            "whole_fovs_spread": 0.442431208084,
            "ratio": 1.369188632678,
            "objects_to_match": None,
            "largest_size_tried": 40,
        },
        {
            "condition": "A",
            "biological_repeat": "r1",
            "descriptor": "median",
            "threshold": 20,
            "random_objects_spread": 0.671891660735,
            "whole_fovs_spread": 0.465370329454,
            "ratio": 0.69262703595,
            "objects_to_match": 20.0,
            "largest_size_tried": 40,
        },
        {
            "condition": "A",
            "biological_repeat": "r1",
            "descriptor": "SD",
            "threshold": 20,
            "random_objects_spread": 0.235482088344,
            "whole_fovs_spread": 0.343440648224,
            "ratio": 1.458457628941,
            "objects_to_match": None,
            "largest_size_tried": 40,
        },
        {
            "condition": "A",
            "biological_repeat": "r1",
            "descriptor": "IQR",
            "threshold": 20,
            "random_objects_spread": 0.419081793469,
            "whole_fovs_spread": 0.867125557839,
            "ratio": 2.069108158247,
            "objects_to_match": None,
            "largest_size_tried": 40,
        },
    ],
    [
        {
            "group": "condition=B, biological_repeat=r1",
            "label": "B · repeat r1",
            "identity": [["B", "str"], ["r1", "str"]],
            "observations": 8,
        }
    ],
    [10, 20, 40],
)


def test_object_rule_edges():
    assert _object_rule_edges() == EXPECTED_OBJECT_RULE_EDGES


def test_descriptor_spread():
    assert _descriptor_spread() == EXPECTED_DESCRIPTOR_SPREAD


def test_the_jcb_check_full_keeps_every_number_and_the_short_names_misses(foundation_checks):
    result = foundation_checks["object_rules"]
    full = _flat_text(result.summary(detail="full"))
    ok = result.table[result.table["status"] == "ok"]
    assert len(ok)
    for row in ok.to_dict("records"):
        for value in (
            str(int(row["rule_objects"])),
            f"{row['fovs_used']:.1f}",
            percent_fine(row["predicted_error"]),
            percent_fine(row["real_error"]),
            f"{row['error_ratio']:.1f}x",
            f"{int(row['fov_rule_fovs'])} ({int(row['fov_rule_fovs_upper95'])})",
        ):
            assert value in full
    short = _flat_text(result.summary())
    assert "the mean of all MOCK cells of the same biological repeat" in short
    missed = ok[~ok["meets_target"].astype(bool)]
    # A strong field effect: the rule's promise breaks with whole fields.
    assert len(missed) and "misses its target with whole fields" in short
