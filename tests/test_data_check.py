"""check_tables: every problem at once, in plain words, and in agreement with the validator."""

import numpy as np
import pandas as pd
import pytest
from scenarios import FEATURE_COLS, SCENARIOS

from morphocoverage import DataCheck, check_tables, validate_input_tables


def _broken() -> tuple[pd.DataFrame, pd.DataFrame]:
    """One dataset with one of each common mistake."""
    fovs = pd.DataFrame(
        {
            "condition": ["MOCK", "MOCK", "drug", "drug", "drug"],
            "Repeat": ["r1"] * 5,  # misspelt
            "technical_repeat": ["B02", "B02", None, "C02", "C03"],  # one empty
            "image_file": ["f1.tif", "f1.tif", "f2.tif", "f3.tif", "f4.tif"],  # f1 twice
        }
    )
    objects = pd.DataFrame(
        {
            "condition": ["MOCK", "MOCK", "drug", "drug", "drug", "drug"],
            "image_file": ["f1.tif", "f1.tif", "f2.tif", "f3.tif", "f3.tif", "f3.tif"],
            "area": [1.0, 2.0, np.inf, 3.0, 4.0, np.nan],  # one infinite, one missing
            "intensity": ["12,5", "3.1", "4", "n/a", "5", "6"],  # not numeric
        }
    )
    return objects, fovs


def _titles(check: DataCheck, level: str) -> str:
    return "\n".join(p.title for p in check.problems if p.level == level)


def test_clean_tables_pass_and_say_what_was_understood():
    built = SCENARIOS["condition_repeat_groups"]()
    check = check_tables(
        built.objects, built.fovs, feature_cols=list(FEATURE_COLS), observation_unit="cell"
    )
    assert check.ok and not check.errors and not check.warnings
    text = check.summary()
    assert text.startswith("MorphoCoverage data check: 0 errors, 0 warnings; the tables can be")
    assert "A field is identified by 'image_file' within condition, repeat." in text
    assert "Analysis groups: condition, repeat." in text
    check.raise_if_errors()


def test_every_problem_is_reported_at_once_with_where_and_how_to_fix():
    objects, fovs = _broken()
    check = check_tables(
        objects, fovs, feature_cols=["area", "intensity", "perimeter"], observation_unit="cell"
    )
    errors = _titles(check, "error")
    assert "Column 'Repeat' looks like the MorphoBricks column 'repeat'" in errors
    assert "The group column 'technical_repeat' is empty in some rows" in errors
    assert "The FOV table lists a field more than once." in errors
    assert "1 feature column named in feature_cols is not in the observation table." in errors
    assert "Feature 'area' contains infinite values." in errors
    assert "Feature 'intensity' is not numeric." in errors
    # A value copied from the FOV table is reported once, where it came from.
    assert errors.count("'technical_repeat' is empty") == 1
    warnings = _titles(check, "warning")
    assert "1 field of the FOV table holds no observation." in warnings
    assert "Feature 'area' is missing in 16.7% of observations." in warnings
    text = check.summary()
    assert (
        "Where: observation table, 2 rows: row 0 (intensity='12,5'); row 3 (intensity='n/a')"
        in (text)
    )
    assert "Where: observation table, 1 row: row 2 (area=inf)." in text
    assert "np.float64" not in text and "nan" not in text.lower().replace("nan)", "")
    assert text.rstrip().endswith("is in docs/data_format.md.")
    with pytest.raises(ValueError, match="6 errors"):
        check.raise_if_errors()


def test_fields_missing_from_the_fov_table_are_named():
    fovs = pd.DataFrame(
        {
            "condition": ["MOCK", "drug"],
            "technical_repeat": ["B02", "C02"],
            "image_file": ["a", "b"],
        }
    )
    objects = pd.DataFrame(
        {"condition": ["MOCK", "drug"], "image_file": ["a", "z"], "x": [1.0, 2.0]}
    )
    check = check_tables(objects, fovs, feature_cols=["x"], observation_unit="cell")
    problem = next(p for p in check.errors if "not in the FOV table" in p.title)
    assert problem.where == "observation table, 1 row: row 1 (image_file='z')."


def test_warnings_for_a_missing_fov_table_unit_and_control():
    built = SCENARIOS["condition_repeat_groups"]()
    check = check_tables(built.objects, feature_cols=list(FEATURE_COLS))
    assert check.ok
    warnings = _titles(check, "warning")
    assert "No FOV table was given." in warnings and "not declared" in warnings
    control = built.fovs["condition"].iloc[0]
    lacking = built.fovs[~((built.fovs["condition"] == control) & (built.fovs["repeat"] == "R2"))]
    kept = built.objects.merge(lacking[["condition", "repeat", "image_file"]])
    check = check_tables(
        kept,
        lacking,
        feature_cols=list(FEATURE_COLS),
        observation_unit="cell",
        control={"condition": control},
    )
    assert "1 repeat has no control field." in _titles(check, "warning")
    wrong = check_tables(
        kept,
        lacking,
        feature_cols=list(FEATURE_COLS),
        observation_unit="cell",
        control={"condition": "nothing"},
    )
    assert "No field matches the control {'condition': 'nothing'}." in _titles(wrong, "error")


def test_empty_optional_columns_are_explained_not_refused():
    built = SCENARIOS["condition_repeat_groups"]()
    objects = built.objects.assign(position_id="", technical_repeat=None)
    fovs = built.fovs.assign(position_id="", technical_repeat=None)
    check = check_tables(objects, fovs, feature_cols=list(FEATURE_COLS), observation_unit="cell")
    assert check.ok
    assert "'position_id' is empty in every row, so it is read as absent." in check.understood
    assert "'technical_repeat' is empty in every row, so it is read as absent." in check.understood


def test_input_that_is_not_a_table_is_explained():
    check = check_tables([1, 2], None, feature_cols="x")
    assert not check.ok and "The observation table is not a table." in check.summary()
    built = SCENARIOS["homogeneous"]()
    check = check_tables(built.objects, built.fovs, feature_cols=FEATURE_COLS[0])
    assert "feature_cols is a single string" in _titles(check, "error")


def test_the_validator_raises_with_the_full_report_after_its_own_first_line():
    objects, fovs = _broken()
    with pytest.raises(ValueError) as caught:
        validate_input_tables(objects, fovs=fovs, feature_cols=["area"])
    message = str(caught.value)
    first, _, rest = message.partition("\n\n")
    assert "non-canonical spelling of MorphoBricks column 'repeat'" in first
    assert rest.startswith("MorphoCoverage data check:")
    assert "The FOV table lists a field more than once." in rest


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_check_tables_and_the_validator_agree_on_every_scenario(name):
    built = SCENARIOS[name]()
    arguments = {"feature_cols": list(FEATURE_COLS), "observation_unit": "cell"}
    assert check_tables(built.objects, built.fovs, **arguments).ok
    validate_input_tables(built.objects, fovs=built.fovs, **arguments)
    # Each broken variant is refused by both.
    variants = {
        "duplicate field": (built.objects, pd.concat([built.fovs, built.fovs.iloc[[0]]])),
        "unlisted field": (
            built.objects,
            built.fovs[built.fovs["image_file"] != built.objects["image_file"].iloc[0]],
        ),
        "missing group value": (built.objects.assign(condition=None), built.fovs),
        "infinite feature": (built.objects.assign(**{FEATURE_COLS[0]: np.inf}), built.fovs),
    }
    for label, (objects, fovs) in variants.items():
        assert not check_tables(objects, fovs, **arguments).ok, label
        with pytest.raises((ValueError, TypeError)):
            validate_input_tables(objects, fovs=fovs, **arguments)
