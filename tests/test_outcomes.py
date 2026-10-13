"""Tests of ``outcomes``: the retrospective outcome metrics."""

import math

import numpy as np
import pandas as pd
import pytest
from _helpers_strategies import _fovs, _validated_values

from morphocoverage.multivariate_outcomes import MMDError
from morphocoverage.outcomes import RandomObjectRelativeError, RareStateDetected, RelativeMeanError
from morphocoverage.strategies.rare_incidence import _positive_fovs


def test_d_1_random_object_error_drops_the_correction_against_a_disjoint_reference():
    rng = np.random.default_rng(0)
    data = _validated_values([rng.normal(10, 1, size=(20, 2)) for _ in range(10)])
    stop, reservoir = _fovs(data, range(7)), _fovs(data, range(7, 10))
    metric = RandomObjectRelativeError("a")
    reference = reservoir.feature_values("a")
    # 140 stop objects against a 60-object reservoir: no longer refused.
    expected = math.sqrt(reference.var(ddof=1) / 140) / abs(reference.mean())
    assert metric.evaluate(reservoir, stop) == pytest.approx(expected)
    # A stop inside the reference keeps the finite-population correction.
    full = data.feature_values("a")
    corrected = math.sqrt(full.var(ddof=1) / 140 * (1 - 140 / 200)) / abs(full.mean())
    assert metric.evaluate(data, stop) == pytest.approx(corrected)


def test_d_6_rare_state_detected_counts_like_the_estimator_and_accepts_missing_labels():
    data = _validated_values([[[0, 0], [1, 1]], [[0, 1]], [[2, 2]]])
    data.objects["rare"] = [True, False, False, True]
    metric = RareStateDetected("rare", 2)
    assert metric.evaluate(data, data) == 1.0
    assert metric.evaluate(data, _fovs(data, [0, 1])) == 0.0
    assert _positive_fovs(data, "rare") == 2  # the one count, shared with the estimator
    for missing in (pd.array([True, pd.NA, False, True], dtype="boolean"), [1.0, np.nan, 0, 1]):
        data.objects["rare"] = missing
        assert metric.evaluate(data, data) is None
    data.objects["rare"] = [2, 0, 0, 1]
    with pytest.raises(ValueError, match="booleans or 0/1"):
        metric.evaluate(data, data)


def test_d_7_a_label_names_the_metric_and_the_default_name_is_unchanged():
    narrow = MMDError(("a", "b"), True, 500, 0, 0.5, label="mmd_bw0.5")
    wide = MMDError(("a", "b"), True, 500, 0, 2.0, label="mmd_bw2")
    assert (narrow.name, wide.name) == ("mmd_bw0.5", "mmd_bw2")
    plain = MMDError(("a", "b"), True, 500, 0, 0.5)
    assert plain.name == "mmd_error:a,b"
    assert "label" not in plain.parameters
    assert narrow.parameters["label"] == "mmd_bw0.5"
    # Without a label the config id is the one computed from the settings alone.
    assert plain.config_id == MMDError(("a", "b"), True, 500, 0, 0.5, label=None).config_id
    assert narrow.config_id != plain.config_id
    assert RelativeMeanError("a", label="mean a").name == "mean a"
    assert RelativeMeanError("a").name == "relative_mean_error:a"
    assert RandomObjectRelativeError("a", label="srs a").name == "srs a"
    assert RareStateDetected("rare", 1, label="rare seen").name == "rare seen"
    with pytest.raises(TypeError, match="label"):
        RelativeMeanError("a", label="")
