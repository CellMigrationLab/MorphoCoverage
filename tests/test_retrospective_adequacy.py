"""Tests of ``retrospective.adequacy``: when the replayed runs reached adequacy, and its words."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from _helpers_retrospective import (
    _adequacy,
    _f_replay,
    _plate_protocol,
    _scenario_reliability,
    _tables,
    _well_values,
)

from morphocoverage.benchmark import AcquisitionOrder
from morphocoverage.outcomes import RelativeMeanError
from morphocoverage.retrospective import AdequacyResult, OutcomeCriterion, adequacy_points
from morphocoverage.retrospective.adequacy import (
    _median_point,
    _tail_point,
    last_judged_step,
    sustained_realized_adequacy,
)
from morphocoverage.retrospective.frontier import frontier
from morphocoverage.retrospective.spatial import EfficiencyResult


def test_f12_an_undefined_tail_leaves_sustained_realized_unreached():
    statuses = ["preserved", "not_preserved", "preserved", "preserved", "undefined", "undefined"]
    steps = list(range(1, 7))
    assert sustained_realized_adequacy(steps, statuses) is None
    assert last_judged_step(steps, statuses) == 4
    # Undefined steps in between are still skipped.
    assert sustained_realized_adequacy([1, 2, 3], ["preserved", "undefined", "preserved"]) == 1
    objects, fovs = _tables(n_fovs=16, scale=0.2)
    objects.loc[objects.index[objects["position_id"].str.endswith("_05")][0], "f"] = np.nan
    loose = OutcomeCriterion(RelativeMeanError("f"), 0.01, "lower_is_better", "the mean")
    replay = _f_replay(
        objects,
        fovs,
        criteria=[loose],
        orders=[AcquisitionOrder.recorded()],
        acquisition_order_col="order",
        reservoir_fraction=0.25,
        n_splits=3,
        random_state=3,
    )
    result = adequacy_points(replay)
    realized = result.realized[result.realized["criterion"] == "all criteria"]
    assert realized["sustained_realized"].isna().all()
    assert (realized["last_judged"].dropna() < realized["stream_fovs"].min()).all()
    assert "the last fields could not be judged" in result.summary()


def test_e4_e5_medians_and_tails_say_what_the_runs_did():
    streams = pd.Series([20] * 11)
    half = pd.Series([5, pd.NA], dtype="Int64")
    assert "in half or more of runs" in _median_point(half, streams)
    # 10 of 11 runs reached field 10: at least 90% of runs, so the tail is field 10.
    one_missing = pd.Series([10] * 10 + [pd.NA], dtype="Int64")
    assert _tail_point(one_missing, streams) == "by field 10 in 90% of runs"
    # 6 of 7 runs by field 10 is under 90%: the tail is the next observed count.
    seven = pd.Series([10] * 6 + [11], dtype="Int64")
    assert _tail_point(seven, streams) == "by field 11 in 90% of runs"


def test_adequacy_and_order_summaries_state_the_tail(gradient_replay):
    adequacy = adequacy_points(gradient_replay)
    text = " ".join(adequacy.summary(detail="full").split())
    assert "(medians; by field" in text or "in more than 10% of runs" in text
    assert "90% of runs by" in adequacy.summary()
    realized = adequacy.realized[adequacy.realized["criterion"] == "all criteria"]
    one_order = realized[realized["order"] == "recorded"]["sustained_realized"]
    tail = float(
        np.quantile(one_order.astype("Float64").to_numpy(dtype=float, na_value=np.inf), 0.9)
    )
    if np.isfinite(tail):
        assert f"by field {tail:.0f} in 90% of runs" in text


def test_adequacy_and_frontier_refuse_another_object():
    with pytest.raises(TypeError, match="^replay must be a ReplayResult$"):
        adequacy_points(None)
    with pytest.raises(TypeError, match="^replay must be a ReplayResult$"):
        frontier(None, [], criterion="x", alphas=[0.1], unit_cols=[])


def test_every_summary_refuses_an_unknown_detail():
    reliability = _scenario_reliability()
    protocol = _plate_protocol(_well_values(("P1", "P2"), 0.95, 0.99))
    for result in (reliability, protocol):
        with pytest.raises(ValueError, match="short"):
            result.summary(detail="brief")
        assert result.summary() == result.summary(detail="short")
    with pytest.raises(ValueError, match="full"):
        AdequacyResult(pd.DataFrame(), pd.DataFrame(), ()).summary(detail="all")


def test_a_group_with_no_judged_run_is_said_out_of_the_range():
    """#245: the realized ranges and the order ranges say how many groups were not judged."""
    result = _adequacy()
    keys = list(result.group_cols)
    realized = result.realized.copy()
    first = tuple(realized[keys].drop_duplicates().iloc[0])
    realized.loc[realized[keys].apply(tuple, axis=1) == first, "judged_fovs"] = 0
    short = AdequacyResult(result.apparent, realized, result.group_cols, result.settings)
    assert "; not judged in 1 " in short.summary()
    assert "not judged in" not in result.summary()
    rows = [
        {
            "g": g,
            "order": "permuted",
            "runs": 2,
            "unjudged": u,
            "reached": 1.0,
            "median_fovs": m,
            "median_travel": 100.0,
            "stream_fovs": 10,
            "longest_stream_fovs": 10,
        }
        for g, u, m in (("a", 0, 3.0), ("b", 0, 5.0), ("c", 2, None))
    ]
    text = EfficiencyResult(pd.DataFrame(), pd.DataFrame(rows), ("g",), {}).summary()
    assert "3 to 5 (not judged in 1 group)" in " ".join(text.split())
