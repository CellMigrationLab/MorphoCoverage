"""Owner decisions of 2026-10-05 on the workflows: live stops, guided answers, HTML figures.

- #85: ``LiveFolder.watch(until_stop=...)`` follows the objective's overall decision;
  ``stop_on="rule"`` keeps the stop on every rule's own verdict.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from morphocoverage import FixedFOVStrategy, ObjectiveConfig
from morphocoverage.evidence import Component
from morphocoverage.policies import EffortPolicy
from morphocoverage.strategies.base import SamplingStrategy
from morphocoverage.workflows import LiveFolder, Setup, live_session, write_fov

SETUP = Setup(readout="x", observation_unit="cell")


def _fields(folder, n: int = 6, condition: str = "MOCK") -> None:
    """``n`` fields of one well, deterministic."""
    for index in range(n):
        values = pd.DataFrame({"x": 100.0 + np.arange(5, dtype=float) + index})
        metadata = {
            "condition": condition,
            "biological_repeat": "R1",
            "technical_repeat": "A01",
            "image_file": f"A01_f{index:03d}.tif",
        }
        write_fov(folder, values, metadata)


def _fingerprint_rule(fovs: int) -> SamplingStrategy:
    """A rule that decides the fingerprint only, after ``fovs`` fields."""
    policy = EffortPolicy("n_fovs", fovs, Component.DISTRIBUTION)
    return SamplingStrategy([], [policy], name=f"fingerprint_{fovs}")


def _watch(folder, rules, objective: str = "population", **kwargs) -> int:
    session = live_session(SETUP, rules=rules, objective=ObjectiveConfig(objective))
    seen = LiveFolder(folder, session).watch(
        interval=0.01, timeout=2, until_stop=True, on_update=None, **kwargs
    )
    return len(seen)


def test_85_one_target_one_rule_stops_as_before(tmp_path):
    _fields(tmp_path)
    assert _watch(tmp_path, (FixedFOVStrategy(2),)) == 2
    assert _watch(tmp_path, (FixedFOVStrategy(2),), stop_on="rule") == 2


def test_85_two_targets_stop_when_both_stop(tmp_path):
    # Each rule alone is unsupported for the phenotype objective; together they decide it.
    _fields(tmp_path)
    rules = (FixedFOVStrategy(2), _fingerprint_rule(3))
    assert _watch(tmp_path, rules, "phenotype") == 3


def test_85_a_target_no_rule_decides_never_stops_the_watch(tmp_path):
    _fields(tmp_path)
    rules = (FixedFOVStrategy(2),)
    assert _watch(tmp_path, rules, "phenotype") == 6  # every field, no stop
    assert _watch(tmp_path, rules, "phenotype", stop_on="rule") == 2


def test_85_one_target_still_continuing_keeps_the_watch_going(tmp_path):
    _fields(tmp_path)
    rules = (FixedFOVStrategy(2), _fingerprint_rule(5))
    assert _watch(tmp_path, rules, "phenotype") == 5
    three = (FixedFOVStrategy(2), FixedFOVStrategy(3), FixedFOVStrategy(4))
    assert _watch(tmp_path, three) == 4


def test_85_a_rule_on_a_target_not_required_does_not_hold_the_stop(tmp_path):
    # The live view's overall ignores a fingerprint rule under the population objective.
    _fields(tmp_path)
    rules = (FixedFOVStrategy(2), _fingerprint_rule(5))
    assert _watch(tmp_path, rules) == 2
    assert _watch(tmp_path, rules, stop_on="rule") == 5


def test_85_the_condition_label_does_not_change_the_stop(tmp_path):
    for condition in ("MOCK", "drug"):
        folder = tmp_path / condition
        _fields(folder, condition=condition)
        assert _watch(folder, (FixedFOVStrategy(2), _fingerprint_rule(3)), "phenotype") == 3


def test_85_stop_on_is_checked(tmp_path):
    _fields(tmp_path, 1)
    with pytest.raises(ValueError, match="stop_on"):
        _watch(tmp_path, (FixedFOVStrategy(2),), stop_on="every_rule")
