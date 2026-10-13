"""Break the synthetic acquisition the way people break tables, and check each mistake is caught.

Every case starts from the clean tables of ``data/synthetic`` (one repeat, to
keep the file fast) and makes one mistake seen in real data: a misspelt
column, a label typed differently in the two tables, numbers with decimal
commas, a table read twice, a wrong argument. Each is one of three kinds:

- **refused**: ``check_tables`` reports an error that says what is wrong
  (and, where one can tell, what was probably meant), and the validator
  and the ``report`` workflow refuse the tables before any analysis runs;
- **warned**: the tables can be analysed, and ``check_tables`` warns;
- **harmless**: nothing changes, the answers are those of the clean tables.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest
from _helpers_synthetic import FEATURES, FIELD_KEY, load

from morphocoverage import check_tables, validate_input_tables
from morphocoverage.sampling_unit import fov_variability
from morphocoverage.workflows import Setup, report

CONTROL = {"condition": "control"}


def _first_repeat() -> tuple[pd.DataFrame, pd.DataFrame]:
    objects, fovs, _ = load()
    first = fovs["biological_repeat"].iloc[0]
    return (
        objects[objects["biological_repeat"] == first].reset_index(drop=True),
        fovs[fovs["biological_repeat"] == first].reset_index(drop=True),
    )


CLEAN = _first_repeat()


@dataclass(frozen=True)
class Mistake:
    """One human error: how it is made, what it should give, and the words that explain it."""

    name: str
    make: Callable[[pd.DataFrame, pd.DataFrame], tuple]
    kind: str  # "refused", "warned" or "harmless"
    says: tuple[str, ...] = ()
    arguments: dict | None = None


def _rename(column: str, to: str) -> Callable:
    return lambda o, f: (o.rename(columns={column: to}), f.rename(columns={column: to}))


def _set(column: str, rows: slice | list, value: object) -> Callable:
    def make(o, f):
        o = o.copy()
        if isinstance(value, str) and o[column].dtype != object:
            o[column] = o[column].astype(object)
        o.loc[rows, column] = value
        return o, f

    return make


def _relabel(value: str, typed: str) -> Callable:
    def make(o, f):
        o = o.copy()
        o.loc[o["condition"] == value, "condition"] = typed
        return o, f

    return make


def _numbers_as_text(o, f):
    o, f = o.copy(), f.copy()
    f["biological_repeat"] = 5
    o["biological_repeat"] = "05"
    return o, f


def _planes(o, f):
    o = o.copy()
    o["z"] = o["label_id"] % 3
    o["image_file"] = o["image_file"].str.replace(".tif", "") + "_z" + o["z"].astype(str) + ".tif"
    return o, o[FIELD_KEY].drop_duplicates()


def _partly_empty_position(o, f):
    f = f.assign(position_id=[f"p{i}" for i in range(len(f))])
    f.loc[:1, "position_id"] = np.nan
    return o.merge(f[[*FIELD_KEY, "position_id"]]), f


MISTAKES = [
    Mistake(
        "a capitalised column",
        _rename("biological_repeat", "Repeat"),
        "refused",
        ("Column 'Repeat' looks like the MorphoBricks column 'biological_repeat'",),
    ),
    Mistake(
        "a space instead of an underscore",
        _rename("technical_repeat", "technical repeat"),
        "refused",
        ("Column 'technical repeat' looks like the MorphoBricks column 'technical_repeat'",),
    ),
    Mistake(
        "a byte-order mark before the first header",
        _rename("condition", "﻿condition"),
        "refused",
        ("looks like the MorphoBricks column 'condition'",),
    ),
    Mistake(
        "a field missing from the FOV table",
        lambda o, f: (o, f.drop(index=0)),
        "refused",
        ("Some observations belong to fields that are not in the FOV table.",),
    ),
    Mistake(
        "a field listed twice",
        lambda o, f: (o, pd.concat([f, f.iloc[[5]]])),
        "refused",
        ("The FOV table lists a field more than once.",),
    ),
    Mistake(
        "a blank condition",
        _set("condition", slice(0, 3), ""),
        "refused",
        ("The group column 'condition' is empty in some rows",),
    ),
    Mistake(
        "a missing field identifier",
        _set("image_file", [7], np.nan),
        "refused",
        ("The field identifier 'image_file' is empty in some rows",),
    ),
    Mistake(
        "decimal commas",
        lambda o, f: (o.assign(area=o["area"].map(lambda v: str(v).replace(".", ","))), f),
        "refused",
        ("Feature 'area' is not numeric.", "a decimal point, not a comma"),
    ),
    Mistake(
        "n/a typed in a feature",
        _set("area", [2, 9], "n/a"),
        "refused",
        ("Feature 'area' is not numeric.", "area='n/a'"),
    ),
    Mistake(
        "an infinite value",
        _set("area", [4], np.inf),
        "refused",
        ("Feature 'area' contains infinite values.",),
    ),
    Mistake(
        "a feature column missing",
        lambda o, f: (o.drop(columns="intensity"), f),
        "refused",
        ("1 feature column named in feature_cols is not in the observation table.",),
    ),
    Mistake(
        "a trailing space in a feature name",
        lambda o, f: (o.rename(columns={"area": "area "}), f),
        "refused",
        ("the table has 'area ' for 'area'",),
    ),
    Mistake(
        "a leading space in a label",
        _relabel("drug", " drug"),
        "refused",
        ("'condition' differs only in spaces or case (' drug' against 'drug')",),
    ),
    Mistake(
        "a label typed with a capital",
        _relabel("drug", "Drug"),
        "refused",
        ("'condition' differs only in spaces or case ('Drug' against 'drug')",),
    ),
    Mistake(
        "a repeat typed as a number in one table",
        _numbers_as_text,
        "refused",
        ("'biological_repeat' is text in one table and a number in the other ('05' against 5)",),
    ),
    Mistake(
        "a well column partly empty",
        lambda o, f: (o, f.assign(technical_repeat=f["technical_repeat"].where(f.index > 2))),
        "refused",
        ("The group column 'technical_repeat' is empty in some rows",),
    ),
    Mistake(
        "position_id partly empty",
        _partly_empty_position,
        "refused",
        ("The field identifier 'position_id' is empty in some rows",),
    ),
    Mistake(
        "z-planes saved as separate fields",
        _planes,
        "refused",
        ("may be one plane/channel/frame of a position rather than a spatial FOV",),
    ),
    Mistake(
        "a text column declared as a feature",
        lambda o, f: (o.assign(intensity="high"), f),
        "refused",
        ("Feature 'intensity' is not numeric.",),
    ),
    Mistake(
        "a group column declared as a feature",
        lambda o, f: (o, f),
        "refused",
        ("A column is used both as a feature and as a group or field identifier.",),
        {"feature_cols": ["area", "biological_repeat"]},
    ),
    Mistake(
        "feature_cols as one string",
        lambda o, f: (o, f),
        "refused",
        ("feature_cols is a single string",),
        {"feature_cols": "area"},
    ),
    Mistake(
        "group_cols as one string",
        lambda o, f: (o, f),
        "refused",
        ("group_cols is a single string",),
        {"group_cols": "condition"},
    ),
    Mistake(
        "a number as the observation unit",
        lambda o, f: (o, f),
        "refused",
        ("The observation unit is not a word.", "observation_unit=5."),
        {"observation_unit": 5},
    ),
    Mistake(
        "the control misspelt",
        lambda o, f: (o, f),
        "refused",
        ("No field matches the control {'condition': 'Control'}.", "'control' and 'drug'"),
        {"control": {"condition": "Control"}},
    ),
    Mistake(
        "a cell file appended twice",
        lambda o, f: (pd.concat([o, o], ignore_index=True), f),
        "warned",
        ("observations are an exact copy of another row.",),
    ),
    Mistake(
        "labels repeated in a field",
        lambda o, f: (o.assign(label_id=1), f),
        "warned",
        ("appear twice in the same field.",),
    ),
    Mistake(
        "an empty observation table",
        lambda o, f: (o.iloc[:0], f),
        "warned",
        ("The observation table has no rows.",),
    ),
    Mistake(
        "no FOV table",
        lambda o, f: (o, None),
        "warned",
        ("No FOV table was given.",),
    ),
    Mistake(
        "a feature missing everywhere",
        lambda o, f: (o.assign(eccentricity=np.nan), f),
        "warned",
        ("Feature 'eccentricity' is missing in 100% of observations.",),
    ),
    Mistake(
        "fields of another condition in the FOV table",
        lambda o, f: (o, pd.concat([f, f.head(3).assign(condition="other")])),
        "warned",
        ("13 fields of the FOV table hold no observation.",),
    ),
    Mistake(
        "rows in another order", lambda o, f: (o.sample(frac=1, random_state=1), f), "harmless"
    ),
    Mistake(
        "an index saved as a column",
        lambda o, f: (o.reset_index().rename(columns={"index": "Unnamed: 0"}), f),
        "harmless",
    ),
    Mistake(
        "repeated index labels", lambda o, f: (o.set_axis(np.zeros(len(o), int)), f), "harmless"
    ),
    Mistake(
        "condition only in the FOV table, with unique file names",
        lambda o, f: _unique_names(o, f, drop="condition"),
        "harmless",
    ),
    Mistake(
        "position_id empty everywhere, as CellPatcher writes it",
        lambda o, f: (o.assign(position_id=""), f.assign(position_id="")),
        "harmless",
    ),
]


def _unique_names(o, f, drop):
    """Globally unique file names, so a group column can live in the FOV table only."""
    rename = lambda t: t.assign(image_file=t["technical_repeat"] + "_" + t["image_file"])
    return rename(o).drop(columns=drop), rename(f)


def _check(objects, fovs, arguments):
    arguments = arguments or {}
    return check_tables(
        objects,
        fovs,
        feature_cols=arguments.get("feature_cols", list(FEATURES)),
        observation_unit=arguments.get("observation_unit", "cell"),
        group_cols=arguments.get("group_cols"),
        control=arguments.get("control", CONTROL),
    )


def _validate(objects, fovs, arguments):
    arguments = arguments or {}
    return validate_input_tables(
        objects,
        fovs=fovs,
        feature_cols=arguments.get("feature_cols", list(FEATURES)),
        observation_unit=arguments.get("observation_unit", "cell"),
        group_cols=arguments.get("group_cols"),
    )


def _setup(arguments):
    arguments = arguments or {}
    features = arguments.get("feature_cols", list(FEATURES))
    return Setup(
        readout="area",
        # A string reaches Setup as given, which refuses it as the user would see.
        features=tuple(features) if isinstance(features, list) else features,
        control=arguments.get("control", CONTROL),
        observation_unit=arguments.get("observation_unit", "cell"),
        group_cols=arguments.get("group_cols"),
    )


@pytest.mark.parametrize("mistake", MISTAKES, ids=[m.name for m in MISTAKES])
def test_each_mistake_is_caught_and_explained(mistake):
    objects, fovs = mistake.make(CLEAN[0].copy(), CLEAN[1].copy())
    check = _check(objects, fovs, mistake.arguments)
    # The full form says where each problem is; wrapped lines are joined to compare.
    text = " ".join(check.summary(detail="full").split())
    for words in mistake.says:
        assert " ".join(words.split()) in text, f"{mistake.name}: {words!r} not in\n{text}"
    short = " ".join(check.summary().split())
    for problem in check.problems:  # the short form names every problem and its fix
        assert " ".join(problem.title.split()) in short and " ".join(problem.fix.split()) in short
    if mistake.kind == "refused":
        assert not check.ok
        control_only = mistake.arguments is not None and set(mistake.arguments) == {"control"}
        if not control_only:  # the control is the workflows' argument, not the validator's
            with pytest.raises((ValueError, TypeError)):
                _validate(objects, fovs, mistake.arguments)
        with pytest.raises((ValueError, TypeError)) as refused:
            report(objects, fovs, _setup(mistake.arguments))
        # A refusal names the problem; it is never a bare KeyError or IndexError.
        assert type(refused.value) in (ValueError, TypeError)
    elif mistake.kind == "warned":
        assert check.ok and any(w in text for w in mistake.says)
    else:
        assert check.ok
        # One empty field per well, as built: the only warning of the clean tables.
        assert [w.title for w in check.warnings] == [
            "10 fields of the FOV table hold no observation."
        ]
        clean = fov_variability(*CLEAN, feature="area", observation_unit="cell").groups
        after = fov_variability(objects, fovs, feature="area", observation_unit="cell").groups
        pd.testing.assert_series_equal(
            clean["icc"].reset_index(drop=True), after["icc"].reset_index(drop=True)
        )


def test_the_workflows_refuse_what_they_cannot_run():
    objects, fovs = CLEAN
    with pytest.raises(TypeError, match="the workflows need the FOV table"):
        report(objects, None, _setup(None))
    with pytest.raises(ValueError, match="'condition' holds \\['control', 'drug'\\]"):
        report(objects, fovs, _setup({"control": {"condition": "Control"}}))
