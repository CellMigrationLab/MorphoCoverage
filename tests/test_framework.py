"""Tests of ``framework``: the high-level entry points for strategy comparison."""

from _helpers_tables import _cells, _fields

import morphocoverage as mc


def test_a15_the_effort_is_worded_in_the_singular_for_one():
    objects = _cells().iloc[[0]]
    (result,) = mc.run_strategies(
        objects,
        fovs=_fields(objects),
        strategies=[mc.FixedFOVStrategy(5)],
        feature_cols=["area"],
        objective=mc.ObjectiveConfig(),
        observation_unit="cell",
    )
    assert result.explain().endswith("after 1 FOV, 1 cell")
