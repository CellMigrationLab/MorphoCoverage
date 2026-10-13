"""Tests of ``_checks``: the argument checks shared by the package."""

from __future__ import annotations

import numpy as np
import pytest

from morphocoverage._checks import (
    check_column,
    check_flag,
    check_seed,
    is_positive_finite,
    is_share,
)
from morphocoverage.bounds import check_confidence


def test_the_scalar_helpers_accept_real_numbers_only():
    assert is_share(0.5) and is_share(np.float32(0.5)) and is_share(1, closed_high=True)
    assert not is_share(1) and not is_share(0)
    for bad in ("0.5", True, float("nan"), float("inf"), None, [0.5]):
        assert not is_share(bad) and not is_share(bad, closed_high=True)
    assert is_positive_finite(2) and is_positive_finite(np.float64(0.1))
    for bad in ("2", True, 0, -1, float("inf"), float("nan")):
        assert not is_positive_finite(bad)


def test_flags_and_single_column_names_are_checked_by_type():
    assert check_flag(True, "x") is True and check_flag(np.bool_(False), "x") is False
    for bad in (1, 0, "no", None, [True]):
        with pytest.raises(TypeError, match="True or False"):
            check_flag(bad, "x")
    assert check_column(None, "c") is None and check_column("order", "c") == "order"
    for bad in (["order"], ("a", "b"), {"a": 1}, "", "  ", 3):
        with pytest.raises(TypeError, match="one column name"):
            check_column(bad, "c")


def test_b8_seeds_and_b9_confidence_accept_numpy_and_refuse_negatives():
    assert check_seed(np.int64(7)) == 7
    with pytest.raises(ValueError, match="non-negative"):
        check_seed(-1)
    assert check_confidence(np.float32(0.95)) == pytest.approx(0.95)
