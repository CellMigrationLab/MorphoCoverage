"""The robust SD under its two rules (audit 12, H4; owner, 2026-10-12): the detector rule
falls back to the SD only where the MAD is 0, so a far value cannot hide itself; the
geometry rule falls back where the MAD is under a quarter of the SD, so a tie-dominated
column cannot outweigh the others, and leaves out a column constant up to rounding."""

from __future__ import annotations

import numpy as np
import pytest

from morphocoverage.embedding import _standardised, robust_scale
from morphocoverage.retrospective.inspection import stands_out
from morphocoverage.sampling_unit.acquisition import _scale
from morphocoverage.statistics import MAD_TO_SD, MAD_TO_SD_ROUNDED, robust_sd


def _mad(values: np.ndarray) -> float:
    return float(np.median(np.abs(values - np.median(values))))


def test_the_detector_rule_still_flags_a_far_field():
    # One field 40 SDs away among 12: the floor would let the SD it inflates hide it.
    residuals = np.random.default_rng(1).normal(size=12)
    residuals[0] = 40.0
    detector = robust_sd(residuals, "detector")
    assert detector == MAD_TO_SD * _mad(residuals)
    assert residuals[0] / detector >= 5
    geometry = robust_sd(residuals[:, None], "geometry")[0]
    assert geometry == pytest.approx(residuals.std(ddof=1))
    assert residuals[0] / geometry < 5


def test_the_detector_rule_takes_the_sd_only_where_the_mad_is_zero():
    tied = np.array([0.0] * 7 + [0.5, -0.5, 2.0])
    assert robust_sd(tied, "detector") == pytest.approx(tied.std(ddof=1))
    assert robust_sd(np.array([3.0]), "detector") == 0.0


def test_the_geometry_rule_gives_a_column_of_half_zeros_no_huge_weight():
    # Audit 7B: just under half the values tied at 0, a tiny MAD.
    rng = np.random.default_rng(0)
    relu = np.abs(rng.normal(size=2000))
    relu[:980] = 0.0
    matrix = np.column_stack([relu, rng.normal(size=2000)])
    weight = (relu / robust_sd(matrix, "geometry")[0]).var()
    assert weight <= 16
    assert (relu / robust_sd(relu, "detector")).var() > 100  # the MAD alone


def test_the_geometry_rule_treats_a_constant_with_rounding_noise_as_constant():
    rng = np.random.default_rng(2)
    noisy = 3.0 + rng.choice([0.0, 1e-7, -1e-7], size=50)  # one value, single-precision noise
    normal = rng.normal(size=50)
    matrix = np.column_stack([noisy, np.full(50, 0.1), normal])
    scale = robust_sd(matrix, "geometry")
    assert scale[0] == 0.0 and scale[1] == 0.0 and scale[2] > 0
    _, varies = _standardised(matrix)
    assert list(varies) == [False, False, True]


def test_each_caller_keeps_its_mad_constant():
    values = np.random.default_rng(3).normal(size=(40, 2))
    _, scale = robust_scale(values)
    assert list(scale) == [MAD_TO_SD_ROUNDED * _mad(column) for column in values.T]
    assert _scale(values[:, 0]) == MAD_TO_SD_ROUNDED * _mad(values[:, 0])


def test_stands_out_falls_back_to_the_sd_where_the_mad_is_zero():
    # A discrete summary: most changes tied, so the MAD is 0 and every change off the
    # median used to stand out.
    changes = np.array([0.1] * 10 + [0.12, 1.0])
    assert _mad(changes) == 0
    assert list(stands_out(changes)) == [False] * 11 + [True]


def test_an_unknown_rule_is_refused():
    with pytest.raises(ValueError, match="rule must be 'detector' or 'geometry'"):
        robust_sd(np.ones(3), "floor")


def test_one_value_has_no_spread_under_either_rule_418():
    # A 1-D sample of one value: the geometry rule indexed a second axis that is not there.
    for rule in ("detector", "geometry"):
        assert robust_sd(np.array([3.0]), rule) == 0.0
    assert robust_sd(np.array([[3.0]]), "geometry").tolist() == [0.0]
    assert robust_sd(np.array([[3.0, 4.0]]), "geometry").tolist() == [0.0, 0.0]
    assert robust_sd(np.array([2.0, 2.0, 2.0]), "geometry") == 0.0
