"""The identity of the acquisition a result was computed on (``_acquisition``, ``_source``)."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from _helpers_tables import _one_condition_tables, _variability

from morphocoverage._acquisition import acquisition_identity, check_one_acquisition
from morphocoverage.schema import validate_input_tables
from morphocoverage.workflows import Setup, document, html_report, optimise, report
from morphocoverage.workflows._source import source_identity


def test_the_identity_ignores_row_order_and_values_but_not_the_fields():
    objects, fovs = _one_condition_tables()
    one = validate_input_tables(objects, fovs=fovs, feature_cols=["x"])
    shuffled = validate_input_tables(
        objects.assign(x=objects["x"] * 3), fovs=fovs.iloc[::-1], feature_cols=["x"]
    )
    other_objects, other_fovs = _one_condition_tables("g")
    other = validate_input_tables(other_objects, fovs=other_fovs, feature_cols=["x"])
    assert acquisition_identity(one) == acquisition_identity(shuffled)
    assert acquisition_identity(one) != acquisition_identity(other)


def _experiment(name: str, group_cols: list | None = None):
    objects = pd.DataFrame({"experiment": name, "fov_id": "f1", "x": [1.0, 2.0]})
    fovs = pd.DataFrame({"experiment": [name], "fov_id": ["f1"]})
    groups = ["experiment"] if group_cols is None else group_cols
    return validate_input_tables(
        objects, fovs=fovs, feature_cols=["x"], fov_col="fov_id", group_cols=groups
    )


def test_a_custom_identity_column_is_part_of_the_acquisition_identity():
    """Field f1 of experiment run_A and of run_B are two acquisitions (#416).

    Only the MorphoBricks group columns and the FOV column were hashed, so a custom
    identity column named in ``group_cols`` was invisible and the two collided.
    """
    run_a, run_b = _experiment("run_A"), _experiment("run_B")
    assert acquisition_identity(run_a) != acquisition_identity(run_b)
    with pytest.raises(ValueError, match="different acquisitions"):
        check_one_acquisition(
            {"a": _variability_settings(run_a), "b": _variability_settings(run_b)}
        )


def test_the_identity_ignores_column_order_and_morphobricks_group_selection():
    """Reordered FOV-table columns, and grouping by fewer MorphoBricks columns, keep it."""
    objects, fovs = _one_condition_tables()
    base = validate_input_tables(objects, fovs=fovs, feature_cols=["x"])
    columns = list(reversed(fovs.columns))
    reordered = validate_input_tables(objects, fovs=fovs[columns], feature_cols=["x"])
    pooled = validate_input_tables(objects, fovs=fovs, feature_cols=["x"], group_cols=[])
    assert acquisition_identity(base) == acquisition_identity(reordered)
    assert acquisition_identity(base) == acquisition_identity(pooled)


def _variability_settings(data):
    from types import SimpleNamespace

    return SimpleNamespace(settings={"acquisition": acquisition_identity(data)})


def test_a_result_without_an_identity_is_named_not_assumed_compatible():
    objects, fovs = _one_condition_tables()
    result = _variability(objects, fovs)

    class Bare:
        def __init__(self) -> None:
            self.settings: dict = {}

    assert check_one_acquisition({"variability": result, "effort": Bare()}) == ["effort"]
    assert check_one_acquisition({"effort": Bare()}) == []


def _tables(seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    rows, fovs = [], []
    for condition in ("ctrl", "drug"):
        for repeat in ("R1", "R2", "R3"):
            for f in range(6):
                pid = f"{condition}{repeat}{f}"
                fovs.append(
                    {
                        "position_id": pid,
                        "condition": condition,
                        "biological_repeat": repeat,
                        "repeat_group": repeat,
                    }
                )
                for _ in range(8):
                    rows.append({**fovs[-1], "x": rng.normal(), "y": rng.normal()})
    return pd.DataFrame(rows), pd.DataFrame(fovs)


SETUP = Setup(
    readout="x", features=("x", "y"), control={"condition": "ctrl"}, observation_unit="cell"
)


@pytest.fixture(scope="module")
def run():
    objects, fovs = _tables()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return objects, fovs, report(objects, fovs, SETUP)


def test_the_identity_ignores_row_and_column_order():
    objects, fovs = _tables()
    base = source_identity(objects, fovs)
    shuffled = objects.sample(frac=1, random_state=1)[list(reversed(objects.columns))]
    assert source_identity(shuffled, fovs.iloc[::-1]) == base


def test_the_identity_sees_a_value_a_row_a_column_and_a_type():
    objects, fovs = _tables()
    base = source_identity(objects, fovs)
    changed = objects.copy()
    changed.loc[3, "x"] += 1e-9
    assert source_identity(changed, fovs) != base
    assert source_identity(objects.iloc[:-1], fovs) != base  # a row
    assert source_identity(objects, fovs.iloc[:-1]) != base  # a zero-object FOV removed
    assert source_identity(objects.assign(z=0), fovs) != base  # a column
    typed = pd.DataFrame({"position_id": [1, "1"], "v": [1.0, 2.0]})
    texts = pd.DataFrame({"position_id": ["1", "1"], "v": [1.0, 2.0]})
    assert source_identity(typed, typed) != source_identity(texts, texts)


def test_the_report_records_its_source(run):
    objects, fovs, result = run
    assert result.source == source_identity(objects, fovs)


def test_the_page_refuses_other_tables_and_names_why(run, tmp_path):
    objects, fovs, result = run
    other = objects.assign(x=objects["x"] + 100)
    with pytest.raises(ValueError, match="not the tables the report was run on"):
        html_report(result, other, fovs, tmp_path)
    with pytest.raises(ValueError, match="not the tables the report was run on"):
        html_report(result, objects, fovs.iloc[:-1], tmp_path)  # a FOV missing
    assert not (tmp_path / "report.html").exists()


def test_a_report_without_a_source_needs_the_explicit_opt_out(run, tmp_path):
    from dataclasses import replace

    objects, fovs, result = run
    legacy = replace(result, source=None)
    with pytest.raises(ValueError, match="check_source=False"):
        html_report(legacy, objects, fovs, tmp_path)


def test_document_takes_one_report_and_one_plan_of_one_acquisition(run):
    objects, fovs, result = run
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plan = optimise(objects, fovs, SETUP, fov_counts=[2, 3], reservoir_fraction=0.35)
        other = optimise(
            _tables(1)[0], _tables(1)[1], SETUP, fov_counts=[2, 3], reservoir_fraction=0.35
        )
    assert document(result, plan) is not None
    with pytest.raises(ValueError, match="one acquisition"):
        document(result, other)
    with pytest.raises(ValueError, match="one report and one plan"):
        document(result, result)
