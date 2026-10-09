"""One contract for a rare-state label column everywhere it is read (issue 344, #233, #351)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from morphocoverage import validate_input_tables
from morphocoverage._rare_label import check_rare_labels, known_flags
from morphocoverage.objectives import RareTarget
from morphocoverage.retrospective import inspect_fovs
from morphocoverage.strategies.rare_incidence import _positive_fovs
from morphocoverage.workflows import Setup
from morphocoverage.workflows._rare_share import with_share


def test_booleans_and_zero_one_with_missing_are_valid_and_missing_is_unknown():
    for good in (
        pd.Series([True, False, True]),
        pd.Series([1, 0, 1]),
        pd.Series([1.0, np.nan, 0.0]),
        pd.Series([True, None, False], dtype="boolean"),
    ):
        check_rare_labels(good, "label")
    known, flags = known_flags(pd.Series([1.0, np.nan, 0.0, 1.0]), "label")
    assert list(known) == [True, False, True, True]
    assert list(flags) == [True, False, False, True]  # unknown is never the state


@pytest.mark.parametrize("bad", [[2, 0, 1], ["no", "yes"], ["false", "0"], [0.5, 1.0]])
def test_other_values_are_refused_naming_the_column(bad):
    with pytest.raises(ValueError, match="label 'x' must hold booleans or 0/1"):
        check_rare_labels(pd.Series(bad), "label 'x'")


def _tables(label):
    fovs = pd.DataFrame({"condition": "a", "position_id": [f"f{i}" for i in range(8)]})
    objects = fovs.loc[fovs.index.repeat(3)].reset_index(drop=True)
    rng = np.random.default_rng(0)
    objects = objects.assign(x=rng.normal(size=len(objects)), state=label(len(objects)))
    return objects, fovs


def test_the_estimator_and_the_inspection_refuse_the_same_labels():
    objects, fovs = _tables(lambda n: [2] + [0] * (n - 1))
    data = validate_input_tables(
        objects=objects, fovs=fovs, feature_cols=["x"], fov_col="position_id"
    )
    with pytest.raises(ValueError, match="must hold booleans or 0/1"):
        _positive_fovs(data, "state")
    with pytest.raises(ValueError, match="must hold booleans or 0/1"):
        inspect_fovs(objects, fovs, feature_cols=["x"], fov_col="position_id", label_col="state")


def test_the_inspection_refuses_a_label_column_that_is_not_there():
    objects, fovs = _tables(lambda n: [0] * n)
    with pytest.raises(ValueError, match="is not a column of the observations"):
        inspect_fovs(objects, fovs, feature_cols=["x"], fov_col="position_id", label_col="nope")


def test_a_user_column_named_like_the_derived_share_is_refused_not_trusted():
    objects, _ = _tables(lambda n: [0] * n)
    rare = RareTarget(
        detection_probability=0.9, min_fov_occurrences=2, phenotype="state", object_prevalence=0.05
    )
    setup = Setup(readout="x", rare=rare, rare_label="state", observation_unit="cell")
    assert "state share" in with_share(objects, setup)
    with pytest.raises(ValueError, match="rename your column"):
        with_share(objects.assign(**{"state share": 99.0}), setup)
