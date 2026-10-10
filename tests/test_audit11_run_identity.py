"""Issue 230: a frontier point is scored only against the run its decisions came from."""

from __future__ import annotations

import pandas as pd
import pytest

from morphocoverage.retrospective._run_identity import run_fingerprint
from morphocoverage.retrospective.frontier import family, frontier
from tests.test_retrospective_frontier import MEAN, _mixed, _oracle, _replay


@pytest.fixture(scope="module")
def mixed():
    return _mixed(seed=0)


def _fovs(ids: list) -> pd.DataFrame:
    return pd.DataFrame({"condition": "a", "position_id": ids})


def test_fingerprint_tells_one_from_text_and_order_and_reservoir_changes():
    where = {"condition": "a", "split": 0}
    base = run_fingerprint(where, _fovs([1, 2]), _fovs([3]), ["condition", "position_id"])
    same = run_fingerprint(where, _fovs([1, 2]), _fovs([3]), ["condition", "position_id"])
    text = run_fingerprint(where, _fovs(["1", "2"]), _fovs([3]), ["condition", "position_id"])
    order = run_fingerprint(where, _fovs([2, 1]), _fovs([3]), ["condition", "position_id"])
    held = run_fingerprint(where, _fovs([1, 2]), _fovs([4]), ["condition", "position_id"])
    assert base == same
    assert len({base, text, order, held}) == 4


def test_reservoir_row_order_does_not_move_the_fingerprint():
    where = {"split": 0}
    key = ["position_id"]
    one = run_fingerprint(where, _fovs([1]), _fovs([3, 4]), key)
    other = run_fingerprint(where, _fovs([1]), _fovs([4, 3]), key)
    assert one == other


def test_a_family_without_a_fingerprint_is_refused(mixed):
    replay = _replay(*mixed, [_oracle(2, 12)], seed=0)
    own = family(replay, "oracle", {_oracle(2, 12).config_id: 12}).drop(columns="run_fingerprint")
    with pytest.raises(ValueError, match="no run fingerprint"):
        frontier(replay, [own], criterion=MEAN.name, alphas=[0.1], unit_cols=["biological_repeat"])


def test_a_tampered_fingerprint_is_refused(mixed):
    replay = _replay(*mixed, [_oracle(2, 12)], seed=0)
    own = family(replay, "oracle", {_oracle(2, 12).config_id: 12})
    own = own.assign(run_fingerprint="0" * 64)
    with pytest.raises(ValueError, match="another acquisition"):
        frontier(replay, [own], criterion=MEAN.name, alphas=[0.1], unit_cols=["biological_repeat"])


def test_the_replays_own_family_is_accepted(mixed):
    replay = _replay(*mixed, [_oracle(2, 12)], seed=0)
    own = family(replay, "oracle", {_oracle(2, 12).config_id: 12})
    result = frontier(
        replay, [own], criterion=MEAN.name, alphas=[0.1], unit_cols=["biological_repeat"]
    )
    assert "oracle" in set(result.scores["family"])


def test_same_seeds_on_another_table_are_refused(mixed):
    replay = _replay(*mixed, [_oracle(2, 12)], seed=0)
    other = _replay(*_mixed(seed=1), [_oracle(2, 12)], seed=0)
    alien = family(other, "oracle", {_oracle(2, 12).config_id: 12})
    with pytest.raises(ValueError, match="another acquisition"):
        frontier(
            replay, [alien], criterion=MEAN.name, alphas=[0.1], unit_cols=["biological_repeat"]
        )
