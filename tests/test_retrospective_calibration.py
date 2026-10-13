"""Tests of ``retrospective.calibration``: the stops of the replays against what they preserved."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest
from _helpers_sufficiency import POPULATION

from morphocoverage import FixedFOVStrategy
from morphocoverage.outcomes import RelativeMeanError
from morphocoverage.retrospective.adequacy import adequacy_points
from morphocoverage.retrospective.calibration import (
    StoppingCalibration,
    _table,
    calibration_summary,
    stopping_calibration,
)
from morphocoverage.retrospective.criteria import OutcomeCriterion
from morphocoverage.retrospective.replay import replay_acquisitions


def _calibration_runs(premature: int, beyond: int, correct: int) -> pd.DataFrame:
    """Runs of one rule: stops fixed later (premature), not fixable, and preserved."""
    rows = []
    kinds = [("not_preserved", "preserved")] * premature
    kinds += [("not_preserved", "not_preserved")] * beyond
    kinds += [("preserved", "preserved")] * correct
    for i, (at_stop, at_end) in enumerate(kinds):
        rows.append(
            {
                "split": i,
                "order_id": "permuted-0",
                "order": "permuted",
                "strategy": "rule",
                "config_id": "c1",
                "criterion": "mean",
                "stopped": True,
                "stop_fovs": 3 + i % 3,
                "stream_fovs": 20,
                "status_at_stop": at_stop,
                "status_at_end": at_end,
            }
        )
    return pd.DataFrame(rows)


def test_p2_9_calibration_headlines_the_stops_not_preserved_with_bounds():
    runs = _calibration_runs(premature=2, beyond=4, correct=4)
    table = _table(runs, [])
    row = table.iloc[0]
    assert row["not_preserved_at_stop"] == pytest.approx(0.6)
    assert row["premature"] == pytest.approx(0.2)
    total = row["premature"] + row["not_preserved_by_stream"]
    assert total == pytest.approx(row["not_preserved_at_stop"])
    text = calibration_summary(StoppingCalibration(runs, table, (), {"labels": {}}))
    assert "60%, 20% to 60% too early" in text
    assert "held-back fields' own noise can also cause" in " ".join(text.split())
    # A median stop carries its Monte Carlo error.
    assert "median (MC SE)" in text and re.search(r"  4 \(\d", text)


def _short_group_replay():
    rng = np.random.default_rng(0)
    rows = [
        {"condition": group, "biological_repeat": "r", "image_file": f"{group}{j}", "a": value}
        for group, n in (("A", 8), ("B", 20))
        for j in range(n)
        for value in rng.normal(10, 1, 20)
    ]
    objects = pd.DataFrame(rows)
    fovs = objects[["condition", "biological_repeat", "image_file"]].drop_duplicates()
    return replay_acquisitions(
        objects,
        fovs.reset_index(drop=True),
        feature_cols=["a"],
        strategies=[FixedFOVStrategy(3)],
        criteria=[OutcomeCriterion(RelativeMeanError("a"), 0.05, "lower_is_better")],
        objective=POPULATION,
        reservoir_fraction=0.5,
        min_fovs=6,
        n_splits=2,
    )


def test_e2_a_group_too_short_to_decide_never_stopped_rather_than_vanishing():
    replay = _short_group_replay()
    calibration = stopping_calibration(replay)
    short = calibration.table[calibration.table["condition"] == "A"]
    assert len(short) and (short["never_stopped"] == 1.0).all()
    assert "fewer than the 6 fields it waits for before deciding" in calibration.summary(
        detail="full"
    )
    assert "never stopped" in calibration.summary()
    apparent = adequacy_points(replay).apparent
    assert set(apparent["condition"]) == {"A", "B"}
    assert apparent.loc[apparent["condition"] == "A", "first_apparent"].isna().all()


def test_dispersed_orders_are_named_in_every_summary(gradient_replay):
    calibration = stopping_calibration(gradient_replay).summary(detail="full")
    adequacy = adequacy_points(gradient_replay).summary()
    # The full form names the dispersed orders "spread out" in its tables' order column.
    assert re.search(r"\n +spread out +\d+% of \d+", calibration)
    short = stopping_calibration(gradient_replay).summary()
    assert "spread out" in short and "dispersed" not in short
    assert "the same positions spread out" in adequacy and "dispersed" not in adequacy
    full = adequacy_points(gradient_replay).summary(detail="full")
    assert "  Spreading the same positions out: judged on held-back fields" in full
    assert "\n  dispersed:" not in calibration and "\n  dispersed:" not in adequacy
