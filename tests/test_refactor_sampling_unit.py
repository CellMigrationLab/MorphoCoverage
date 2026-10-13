"""Characterisation tests of the sampling-unit functions refactored without a change of
behaviour (refactor R3, 2026-10-12): the branches no golden run reached, pinned on the code
before the refactor, the text to the character and the numbers as they were.
"""

from __future__ import annotations

import warnings
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


def _fields(condition, repeat, n_fields, per_field, mean, seed, sd=1.0, prefix="f"):
    rng = np.random.default_rng(seed)
    rows = [
        {
            "condition": condition,
            "biological_repeat": repeat,
            "image_file": f"{prefix}{j}",
            "x": float(v),
        }
        for j in range(n_fields)
        for v in mean + rng.normal(0, sd, per_field)
    ]
    fovs = pd.DataFrame(
        {
            "condition": condition,
            "biological_repeat": repeat,
            "image_file": [f"{prefix}{j}" for j in range(n_fields)],
        }
    )
    return pd.DataFrame(rows), fovs


def _join(*parts):
    return (
        pd.concat([p[0] for p in parts], ignore_index=True),
        pd.concat([p[1] for p in parts], ignore_index=True),
    )


def _plain(frame):
    """The rows of ``frame`` with missing values as ``None``, so they compare equal."""
    return frame.astype(object).where(frame.notna(), None).to_dict("records")


def _error(call):
    """The type and message of the exception ``call`` raises."""
    with pytest.raises(Exception) as caught:
        call()
    return type(caught.value).__name__, str(caught.value)


def _across_fallbacks():
    from morphocoverage.sampling_unit.across_text import across_parts

    base = {
        "feature": "area",
        "status": "ok",
        "values": "raw",
        "fold_change": np.nan,
        "effect": 0.5,
        "p_paired_t": 0.02,
        "q_paired_t": 0.04,
        "held": ["R1", "R2", "R3"],
        "repeats": 3,
        "lower": 0,
        "higher": 3,
    }
    across = pd.DataFrame(
        [
            {
                **base,
                "kind": "across_normalised",
                "comparison": "B/cB vs D/cD",
                "condition": "B",
                "condition_control": "cB",
                "reference": "D",
                "reference_control": "cD",
            },
            {
                **base,
                "kind": "vs_master_raw",
                "comparison": "cB vs M",
                "condition": "cB",
                "condition_control": None,
                "reference": "M",
                "reference_control": None,
                "lower": 1,
                "higher": 2,
            },
            {
                **base,
                "kind": "vs_master_raw",
                "comparison": "cD vs M",
                "condition": "cD",
                "condition_control": None,
                "reference": "M",
                "reference_control": None,
                "status": "fewer",
            },
        ]
    )
    contrast = SimpleNamespace(
        across=across,
        settings={
            "repeat_col": "biological_repeat",
            "feature": "area",
            "across_features": None,
            "acquisition_gaps": [{"controls": ["cB", "M"]}],
            "master_repeats": ("M", ["R1", "R2", "R3"], ["R1", "R2", "R3", "R4"]),
            "family_features": 1,
        },
    )
    return "\n".join(across_parts(contrast))


def _balance_held_out():

    return _balance([("A", 1), ("B", 2), ("C", 0)]) + _balance(
        [("A", 1), ("B", 2), ("C", 4), ("D", 3)]
    )


def _balance(held_out):
    from morphocoverage.sampling_unit.cells_text import balance_lines

    rows = []
    for name, held in held_out:
        rows.append(
            {
                "condition": name,
                "control_condition": "ctrl",
                "unbalanced": held > 0,
                "spread": 1.1,
                "other_control_repeats": held,
                "control_shift": 2.5,
                "p_balanced": np.nan,
                "sparse_repeats": [],
                "repeats": 3,
                "balanced_repeats": 0,
            }
        )
    cells = pd.DataFrame(rows)
    contrast = SimpleNamespace(
        cells=cells, settings={"repeat_col": "biological_repeat", "control_only_repeats": ["R9"]}
    )
    return balance_lines(contrast)


def _clustering_errors():
    from morphocoverage.sampling_unit import fov_clustering

    objects, fovs = _fields("A", "r1", 4, 3, 1.0, 0)
    objects["state"] = ["a", "b"] * 6
    calls = [
        {"label_col": "missing"},
        {"label_col": "state", "detection_probability": 1.5},
        {"label_col": "state", "states": "a"},
        {"label_col": "state", "states": []},
        {"label_col": "state", "states": ["a", "a"]},
    ]
    return [_error(lambda c=c: fov_clustering(objects, fovs, **c)) for c in calls]


def _contrast_edges():
    from morphocoverage.sampling_unit import condition_contrast

    objects, fovs = _join(_fields("ctrl", "r1", 3, 5, 1.0, 0), _fields("ctrl", "r2", 3, 5, 1.0, 1))
    error = _error(
        lambda: condition_contrast(objects, fovs, feature=["x"], control={"condition": "ctrl"})
    )
    result = condition_contrast(objects, fovs, feature="x", control={"condition": "ctrl"})
    return error, result.settings["reason"], result.table.shape, result.effects.shape


def _across_one_repeat():
    from morphocoverage.sampling_unit.contrast_repeats import across_repeats

    table = pd.DataFrame(
        {
            "condition": ["A", "B"],
            "control_condition": ["c", "c"],
            "biological_repeat": ["r1", "r1"],
            "mean": [2.0, 3.0],
            "control_mean": [1.0, 1.0],
            "d_objects": [0.5, 0.7],
        }
    )
    out = across_repeats(table, "biological_repeat", 0)
    return list(out.columns), len(out)


def _draw_rows_missing_sampling():
    from morphocoverage.sampling_unit.contrast_text import _draw_rows

    table = pd.DataFrame(
        {"biological_repeat": ["r1"], "condition": ["A"], "control_condition": ["c"]}
    )
    effects = pd.DataFrame(
        [
            {
                "biological_repeat": "r1",
                "condition": "A",
                "n_objects": 50,
                "sampling": "random objects",
                "median": 0.5,
                "q25": 0.25,
                "q75": 0.75,
                "same_sign": 0.9,
                "fields_treated": np.nan,
                "resampled": None,
            },
            {
                "biological_repeat": "r1",
                "condition": "A",
                "n_objects": 100,
                "sampling": "whole FOVs",
                "median": 0.4,
                "q25": 0.2,
                "q75": 0.6,
                "same_sign": np.nan,
                "fields_treated": 4.5,
                "resampled": "technical_repeat",
            },
        ]
    )
    contrast = SimpleNamespace(
        table=table, effects=effects, settings={"repeat_col": "biological_repeat"}
    )
    return _draw_rows(contrast)


def _discovery_and_ranking_errors():
    from morphocoverage.sampling_unit import feature_discovery, feature_ranking

    objects, fovs = _join(_fields("ctrl", "r1", 3, 5, 1.0, 0), _fields("A", "r1", 3, 5, 2.0, 1))
    control = {"condition": "ctrl"}
    out = []
    for function in (feature_discovery, feature_ranking):
        out.append(_error(lambda f=function: f(objects, fovs, features=[], control=control)))
        out.append(
            _error(lambda f=function: f(objects, fovs, features=["x"], control=control, alpha=1.5))
        )
    only, only_fovs = _fields("ctrl", "r1", 3, 5, 1.0, 0)
    ranked = feature_ranking(only, only_fovs, features=["x"], control=control)
    out.append((ranked.settings["reason"], ranked.tests.shape, ranked.ranking.shape))
    return out


def _legend_none():
    from morphocoverage.sampling_unit.legend_text import figure_legend

    return figure_legend(None, pd.DataFrame(), "cells")


def _object_rule_edges():
    from morphocoverage.sampling_unit import object_rule_check

    out = []
    ctrl = _fields("ctrl", "r1", 6, 10, 5.0, 0)
    neg = _fields("neg", "r1", 6, 10, -1.0, 1)
    flat = _fields("flat", "r1", 6, 10, 5.0, 2, sd=0.0)
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
    zero_ctrl = _fields("ctrl", "r1", 6, 10, 0.0, 3, sd=0.0)
    objects2, fovs2 = _join(zero_ctrl, _fields("A", "r1", 6, 10, 2.0, 4))
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

    objects, fovs = _join(_fields("A", "r1", 8, 10, 5.0, 0), _fields("B", "r1", 2, 4, 5.0, 1))
    result = descriptor_spread_check(
        objects, fovs, feature="x", threshold=20, sizes=[10, 20, 40], n_draws=5
    )
    return (
        result.curves.round(12).to_dict("records"),
        _plain(result.at_threshold.round(12)),
        result.settings["skipped_groups"],
        result.settings["sizes"],
    )


def _populations_edges():
    from morphocoverage.sampling_unit import find_populations

    rng = np.random.default_rng(0)
    rows = []
    for condition, shift in (("ctrl", 0.0), ("A", 0.0)):
        for j in range(15):
            for _ in range(10):
                rows.append(
                    {
                        "condition": condition,
                        "biological_repeat": "r1",
                        "image_file": f"{condition}{j}",
                        "a": rng.normal(shift, 1),
                        "b": rng.normal(0, 1),
                    }
                )
    objects = pd.DataFrame(rows)
    planted = objects.index[:90]
    objects.loc[planted, ["a", "b"]] += 9.0
    fovs = objects[["condition", "biological_repeat", "image_file"]].drop_duplicates()
    sampled = find_populations(objects, fovs, features=["a", "b"], max_points=150, min_cells=20)
    missing = objects.assign(a=np.nan)
    none = find_populations(missing, fovs, features=["a", "b"])
    keep = ["population", "kind", "observations", "fields", "repeats", "defined_by"]
    return (
        sampled.table[keep].to_dict("records"),
        {k: sampled.settings[k] for k in ("sampled", "clustered", "assigned", "min_cells")},
        sampled.labels.value_counts(dropna=False).to_dict(),
        none.table.shape,
        {
            k: none.settings[k]
            for k in (
                "features",
                "observations",
                "incomplete_observations",
                "clustered",
                "assigned",
            )
        },
    )


def _repeat_plan_edges():
    from morphocoverage.sampling_unit.repeat_plan import repeat_plan

    def row(condition, repeat, d, low, high, mean=2.0, se=0.1):
        return {
            "condition": condition,
            "control_condition": "c",
            "biological_repeat": repeat,
            "d_objects": d,
            "d_low": low,
            "d_high": high,
            "fields": 6,
            "control_fields": 6,
            "mean": mean,
            "control_mean": 1.0,
            "log_ratio_se": se,
            "difference_se": se,
            "between_share": 0.2,
        }

    table = pd.DataFrame(
        [
            row("A", "r1", 0.5, np.nan, np.nan),  # no interval anywhere
            row("A", "r2", 0.6, np.nan, np.nan),
            row("B", "r1", 0.5, 0.1, 0.9),  # one interval of two repeats
            row("B", "r2", 0.7, np.nan, np.nan),
            row("C", "r1", 0.5, 0.1, 0.9, mean=2.0, se=0.0),  # identical ratios, no error
            row("C", "r2", 0.5, 0.1, 0.9, mean=2.0, se=0.0),
            row("C", "r3", 0.5, 0.1, 0.9, mean=2.0, se=0.0),
            row("D", "r1", 0.5, 0.1, 0.9, se=np.nan),  # a missing error
            row("D", "r2", np.inf, 0.2, 1.0, se=np.nan),
        ]
    )
    empty = repeat_plan(table.drop(columns="d_low"))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        plan = repeat_plan(table)
    return (
        list(empty.columns),
        len(empty),
        _plain(plan),
        [(w.category.__name__, str(w.message)) for w in caught],
    )


def _plan_parts_unestimated():
    from morphocoverage.sampling_unit.repeat_plan_text import plan_parts

    plan = pd.DataFrame(
        [
            {
                "condition": "A",
                "control_condition": "c",
                "repeats": 2,
                "d_mean": 0.5,
                "planned_on": "d",
                "change": np.nan,
                "repeats_needed": np.nan,
                "repeats_needed_twice_fields": np.nan,
                "fields_per_repeat": 4.0,
                "status": "1 of 2 repeats with an interval of d",
            },
            {
                "condition": "B",
                "control_condition": "c",
                "repeats": 1,
                "d_mean": 0.4,
                "planned_on": "d",
                "change": np.nan,
                "repeats_needed": np.nan,
                "repeats_needed_twice_fields": np.nan,
                "fields_per_repeat": np.inf,
                "status": "one biological repeat: the variation between repeats is unknown",
            },
        ]
    )
    contrast = SimpleNamespace(
        plan=plan, table=pd.DataFrame(), settings={"feature": "area", "unit": None}
    )
    return plan_parts(contrast, full=True)


def _repeat_cautions_flat():
    from morphocoverage.sampling_unit.repeats_text import repeat_cautions

    repeats = pd.DataFrame(
        [
            {
                "condition": "A",
                "status": "the repeat effects are identical, so the paired test has "
                "no variance to test with",
                "repeats": 3,
                "p_paired_t": np.nan,
                "q_paired_t_normalised": np.nan,
                "p_sign_flip": np.nan,
            },
            {
                "condition": "B",
                "status": "the repeat effects are identical, so the paired test has "
                "no variance to test with",
                "repeats": 4,
                "p_paired_t": np.nan,
                "q_paired_t_normalised": np.nan,
                "p_sign_flip": np.nan,
            },
        ]
    )
    contrast = SimpleNamespace(repeats=repeats, settings={"repeat_col": "biological_repeat"})
    return repeat_cautions(contrast, full=False), repeat_cautions(contrast, full=True)


def _unusual_edges():
    from morphocoverage.sampling_unit import unusual_objects

    parts = [_fields("ctrl", f"r{r}", 4, 6, 1.0, r) for r in range(2)]
    objects, fovs = _join(*parts, _fields("A", "r0", 4, 6, 1.0, 7))
    objects["y"] = np.random.default_rng(3).normal(0, 1, len(objects))
    error = _error(
        lambda: unusual_objects(
            objects, fovs, feature_cols=["x", "y"], reference={"condition": "ctrl"}, quantile=1.0
        )
    )
    missing = objects.copy()
    missing.loc[missing["condition"] == "A", "y"] = np.nan
    result = unusual_objects(
        missing, fovs, feature_cols=["x", "y"], reference={"condition": "ctrl"}, seed_checks=1
    )
    return error, result.groups["status"].tolist()


def _whole_table_edges():
    from morphocoverage.sampling_unit.whole_table import _table_rows

    meta = pd.DataFrame(
        {
            "condition": ["A", "c", "A", "c"],
            "biological_repeat": ["r1", "r1", "r2", "r2"],
            "field": ["f1", "f2", "f3", "f4"],
        }
    )
    values = np.arange(8, dtype=float).reshape(4, 2)
    return _table_rows("T", values, meta, [("A", "c"), ("A", "x")], 9, 0)


EXPECTED_ACROSS_FALLBACKS = (
    "  Compared across experiments, secondary to the verdicts above: the "
    "paired t-test on the biological\n    repeat means, one value per "
    "biological repeat, each family corrected on its own:\n    The same "
    "treatment in two experiments, each condition normalised to its own "
    "control: in every\n      biological repeat holding all four, the log "
    "ratio of each to its control, the one minus the\n      other; the "
    "larger ratio is named.\n      comparison    larger ratio, by "
    "biological repeat  change  paired t: p (q)  biological repeats\n      "
    "B/cB vs D/cD  B/cB in 3 of 3                      +0.5    0.020 "
    "(0.040)    R1, R2 and R3\n    The master control M was imaged in 3 of "
    "the 4 biological repeats, not in R4. The controls\n      against it "
    "on the raw values, not normalised, in the biological repeats holding "
    "both, on each\n      feature's scale: the log ratio of the control's "
    "mean to the master's, else their difference;\n      raw comparisons "
    "across experiments assume they were imaged and processed alike.\n     "
    " comparison  by biological repeat  change  paired t: p (q)  "
    "biological repeats\n      cB vs M     1 lower, 2 higher     +0.5    "
    "0.020 (0.040)    R1, R2 and R3\n      ⚠️ cB vs M: joins controls "
    "apart in intensity (cB vs M), so it may show acquisition rather\n     "
    "    than biology.\n    B/cB vs D/cD: a value is not positive, so the "
    "differences of the raw differences are compared\n      instead of log "
    "ratios.\n    cB vs M: a value is not positive, so the raw differences "
    "are compared instead of log ratios.\n    Not compared: cD vs M: fewer "
    "than 3 biological repeats hold every condition of the comparison."
)


EXPECTED_BALANCE_HELD_OUT = [
    (
        "Balance: for A and B, the control as provided is pooled from "
        "biological repeats the condition was not imaged in (1 and 2 of them); "
        "no balanced test could be drawn, so the test is given as provided "
        "only where the control comes from the condition's own biological "
        "repeats. The control means differ up to 2.5x between biological "
        "repeats, so a raw test pooling them would compare each condition with "
        "other repeats' level: read the normalised columns. The ctrl of "
        "biological repeat R9, which holds no other condition, is in no "
        "comparison."
    ),
    (
        "Balance: for every condition, the control as provided is pooled from "
        "biological repeats the condition was not imaged in (1 to 4 of them); no "
        "balanced test could be drawn, so the test is given as provided only "
        "where the control comes from the condition's own biological repeats. "
        "The control means differ up to 2.5x between biological repeats, so a "
        "raw test pooling them would compare each condition with other "
        "repeats' level: read the normalised columns. The ctrl of biological "
        "repeat R9, which holds no other condition, is in no comparison."
    ),
]


EXPECTED_CLUSTERING_ERRORS = [
    ("ValueError", "label_col 'missing' is not a column of objects"),
    ("ValueError", "detection_probability and confidence must be between 0 and 1"),
    ("ValueError", "states must be a list of 'state' values, such as ['rare']"),
    ("ValueError", "states must be a list of 'state' values, such as ['rare']"),
    ("ValueError", "states names a state twice: ['a', 'a']"),
]


EXPECTED_CONTRAST_EDGES = (
    ("TypeError", "feature must be one column name"),
    ("Every analysis group is the control, ctrl: no other condition to compare."),
    (0, 0),
    (0, 0),
)


EXPECTED_ACROSS_ONE_REPEAT = (
    [
        "condition",
        "control_condition",
        "repeats",
        "lower",
        "higher",
        "mean_difference",
        "fold_change",
        "d_mean",
        "d_min",
        "d_max",
        "p_sign_flip",
        "q_sign_flip",
        "p_sign_flip_floor",
        "p_sign_flip_exact",
        "p_paired_t",
        "q_paired_t",
        "p_paired_t_normalised",
        "q_paired_t_normalised",
        "experiment",
        "status",
        "repeat_contrasts",
        "ci_low",
        "ci_high",
        "q_screen",
    ],
    0,
)


EXPECTED_DRAW_ROWS_MISSING_SAMPLING = (
    [
        ["A vs c, repeat r1", "50", "0.50 (0.25 to 0.75), 90%", "–", "–", "fields only"],
        ["A vs c, repeat r1", "100", "–", "0.40 (0.20 to 0.60)", "about 4.5", "technical_repeat"],
    ],
    ["contrast", "cells", "random cells", "whole fields", "fields", "resampled"],
)


EXPECTED_DISCOVERY_AND_RANKING_ERRORS = [
    ("ValueError", "features must name at least one column"),
    ("ValueError", "alpha must be between 0 and 1"),
    ("ValueError", "features must name at least one column"),
    ("ValueError", "alpha must be between 0 and 1"),
    (("Every analysis group is the control, ctrl: no other condition to compare."), (0, 0), (0, 0)),
]


EXPECTED_LEGEND_NONE = None


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


EXPECTED_POPULATIONS_EDGES = (
    [
        {
            "population": "P1",
            "kind": "main",
            "observations": 210,
            "fields": 21,
            "repeats": 1,
            "defined_by": "low b (d -9.2), low a (d -9.0)",
        },
        {
            "population": "G1",
            "kind": "group",
            "observations": 90,
            "fields": 9,
            "repeats": 1,
            "defined_by": "high b (d +9.2), high a (d +9.0)",
        },
    ],
    {"sampled": 150, "clustered": True, "assigned": 1.0, "min_cells": 20},
    {"P1": 210, "G1": 90},
    (0, 17),
    {
        "features": [],
        "observations": 0,
        "incomplete_observations": 300,
        "clustered": False,
        "assigned": 0.0,
    },
)


EXPECTED_REPEAT_PLAN_EDGES = (
    [
        "condition",
        "control_condition",
        "repeats",
        "d_mean",
        "between_sd",
        "within_se",
        "planned_on",
        "change",
        "fields_per_repeat",
        "limited_by",
        "repeats_needed",
        "repeats_needed_twice_fields",
        "alpha_planned",
        "experiment",
        "status",
    ],
    0,
    [
        {
            "condition": "A",
            "control_condition": "c",
            "repeats": 2,
            "d_mean": 0.55,
            "between_sd": None,
            "within_se": None,
            "planned_on": "d",
            "change": None,
            "fields_per_repeat": None,
            "limited_by": "",
            "repeats_needed": None,
            "repeats_needed_twice_fields": None,
            "alpha_planned": 0.0125,
            "experiment": 0,
            "status": (
                "0 of 2 repeats with an interval of d (two fields a side): the "
                "variation between repeats is not estimated"
            ),
        },
        {
            "condition": "B",
            "control_condition": "c",
            "repeats": 2,
            "d_mean": 0.6,
            "between_sd": None,
            "within_se": 0.2040853827698616,
            "planned_on": "log ratio",
            "change": 2.0,
            "fields_per_repeat": 4.0,
            "limited_by": "",
            "repeats_needed": None,
            "repeats_needed_twice_fields": None,
            "alpha_planned": 0.0125,
            "experiment": 0,
            "status": (
                "1 of 2 repeats with an interval of d (two fields a side): the "
                "variation between repeats is not estimated"
            ),
        },
        {
            "condition": "C",
            "control_condition": "c",
            "repeats": 3,
            "d_mean": 0.5,
            "between_sd": 0.0,
            "within_se": 0.2040853827698616,
            "planned_on": "log ratio",
            "change": 2.0,
            "fields_per_repeat": 4.0,
            "limited_by": "within repeats",
            "repeats_needed": None,
            "repeats_needed_twice_fields": None,
            "alpha_planned": 0.0125,
            "experiment": 0,
            "status": (
                "the estimated variance of the repeat effects is zero: the plan is not "
                "defined there, which is not evidence of no power"
            ),
        },
        {
            "condition": "D",
            "control_condition": "c",
            "repeats": 2,
            "d_mean": np.inf,
            "between_sd": 0.0,
            "within_se": 0.2040853827698616,
            "planned_on": "d",
            "change": None,
            "fields_per_repeat": 4.0,
            "limited_by": "within repeats",
            "repeats_needed": None,
            "repeats_needed_twice_fields": None,
            "alpha_planned": 0.0125,
            "experiment": 0,
            "status": ("a value the plan needs is missing, so the repeats needed are not planned"),
        },
    ],
    [
        ("RuntimeWarning", "invalid value encountered in subtract"),
        ("RuntimeWarning", "invalid value encountered in subtract"),
    ],
)


EXPECTED_PLAN_PARTS_UNESTIMATED = (
    [
        (
            "  Planning the next experiment on area (for a new experiment, if the "
            "difference and the variation\n    seen here recur; not what the "
            "current result needs): the biological repeats a paired t-test on\n    "
            "the repeat means needs for 80% power, corrected over the conditions "
            "of its sub-experiment (a\n    follow-up on area alone), with the "
            "fields per repeat as now and with twice as many; then the\n    fields "
            "a side each repeat needs to show the difference on its own:"
        ),
        (
            "    condition  change across repeats  mean d per repeat (cells)  "
            "repeats needed  with twice the fields  fields a side per repeat\n    "
            "A vs c     –                      +0.50                      –        "
            "       –                      about 4\n    B vs c     –               "
            "       +0.40                      –               –                   "
            "   more than 4 times the fields"
        ),
        (
            "  A: fewer than two of the repeats have two fields a side or more, so "
            "the variation between repeats\n    is not estimated."
        ),
        (
            "  B was imaged in one biological repeat: a second repeat shows how "
            "much the difference varies\n    between repeats."
        ),
    ],
    [
        (
            "Repeats needed: for a paired t-test at p ≤ 0.05/m, m the conditions "
            "of the sub-experiment (what Benjamini-Hochberg asks at most), on the "
            "values that test compares: each repeat's change (its mean over its "
            "control's, a fold change; the difference of the means for a feature "
            "with a value that is not positive; the table gives their geometric "
            "mean, as the verdict does), its spread between the repeats and its "
            "sampling error within one. All projected from the repeats here, so "
            "the plan is a guide."
        ),
        (
            "The fields a side per repeat are the most any repeat needs, projected "
            "from its interval of d (narrowing as 1/sqrt of the number)."
        ),
    ],
)


EXPECTED_REPEAT_CAUTIONS_FLAT = (
    [
        (
            "A and B have the same change in every biological repeat, so the "
            "paired test has no variance to test with: no p is given, and the "
            "change is described by its direction and size."
        )
    ],
    [
        (
            "A and B have the same change in every biological repeat, so the "
            "paired test has no variance to test with: no p is given, and the "
            "change is described by its direction and size."
        )
    ],
)


EXPECTED_UNUSUAL_EDGES = (
    ("ValueError", "quantile must be between 0 and 1"),
    ["ok", "ok", "not assessed: none of its 24 cells has a value of every feature"],
)


EXPECTED_WHOLE_TABLE_EDGES = [
    {
        "table": "T",
        "condition": "A",
        "control": "c",
        "repeats": 2,
        "status": "not tested: one observation a side and repeat",
    },
    {
        "table": "T",
        "condition": "A",
        "control": "x",
        "repeats": 0,
        "status": "not tested: no repeat holds both",
    },
]


def test_across_fallbacks():
    assert _across_fallbacks() == EXPECTED_ACROSS_FALLBACKS


def test_balance_held_out():
    assert _balance_held_out() == EXPECTED_BALANCE_HELD_OUT


def test_clustering_errors():
    assert _clustering_errors() == EXPECTED_CLUSTERING_ERRORS


def test_contrast_edges():
    assert _contrast_edges() == EXPECTED_CONTRAST_EDGES


def test_across_one_repeat():
    assert _across_one_repeat() == EXPECTED_ACROSS_ONE_REPEAT


def test_draw_rows_missing_sampling():
    assert _draw_rows_missing_sampling() == EXPECTED_DRAW_ROWS_MISSING_SAMPLING


def test_discovery_and_ranking_errors():
    assert _discovery_and_ranking_errors() == EXPECTED_DISCOVERY_AND_RANKING_ERRORS


def test_legend_none():
    assert _legend_none() == EXPECTED_LEGEND_NONE


def test_object_rule_edges():
    assert _object_rule_edges() == EXPECTED_OBJECT_RULE_EDGES


def test_descriptor_spread():
    assert _descriptor_spread() == EXPECTED_DESCRIPTOR_SPREAD


def test_populations_edges():
    assert _populations_edges() == EXPECTED_POPULATIONS_EDGES


def test_repeat_plan_edges():
    assert _repeat_plan_edges() == EXPECTED_REPEAT_PLAN_EDGES


def test_plan_parts_unestimated():
    assert _plan_parts_unestimated() == EXPECTED_PLAN_PARTS_UNESTIMATED


def test_repeat_cautions_flat():
    assert _repeat_cautions_flat() == EXPECTED_REPEAT_CAUTIONS_FLAT


def test_unusual_edges():
    assert _unusual_edges() == EXPECTED_UNUSUAL_EDGES


def test_whole_table_edges():
    assert _whole_table_edges() == EXPECTED_WHOLE_TABLE_EDGES
