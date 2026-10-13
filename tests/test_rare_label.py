"""Tests of ``_rare_label``: the rare label's values, and a missing value as unknown."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from morphocoverage._rare_label import check_rare_labels, known_flags


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
