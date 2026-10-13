"""Tests of ``results``: StrategyResult, its views and the Decision vocabulary."""

from __future__ import annotations

import pytest
from _helpers_results import _flat, _nucleus_setup, _strategy_result, _updates, _well
from _helpers_strategies import READOUT, _live, _overall_stop, _raster_well

import morphocoverage as mc
from morphocoverage.order_check import RANDOM_SAMPLE_ASSUMPTION, RARE_ASSUMPTION
from morphocoverage.results import (
    Decision,
    _distinct,
    _target_decisions,
    combine_decisions,
    decision_view,
    stop_caution,
    stop_caution_parts,
)
from morphocoverage.workflows import Setup


def test_a_strategy_result_owns_its_mappings_and_they_do_not_alias_each_other():
    result = _strategy_result()
    assert result.parameters == result.configuration
    assert result.parameters is not result.configuration
    with pytest.raises(TypeError, match="read-only"):
        result.configuration["estimators"][0]["parameters"]["k"] = 2
    with pytest.raises(TypeError, match="read-only"):
        result.group["condition"] = "b"
    assert result.parameters["estimators"][0]["parameters"]["k"] == 1


def _owned_results():
    import pandas as pd

    from morphocoverage.embedding import Embedding
    from morphocoverage.retrospective.overview import DatasetOverview
    from morphocoverage.sampling_unit.clustering import FOVClustering
    from morphocoverage.transforms import FittedTransform, Standardize

    table = pd.DataFrame({"a": [1, 2]})
    points = pd.DataFrame({"x": [0.0, 1.0]})
    settings = {"k": [1, 2]}
    return [
        (FOVClustering(table, (), settings), settings, "table", table),
        (Embedding(points, points.copy(), (0.5, 0.2), settings), settings, "points", points),
        (DatasetOverview({"n": 1}, table, ("a",), settings), settings, "layout", table),
        (
            FittedTransform(Standardize(), ("a",), 2, 1, {"s": [1.0]}),
            {"s": [1.0]},
            None,
            None,
        ),
    ]


def test_result_families_own_their_settings_and_tables():
    for result, source, frame_name, frame in _owned_results():
        container = result.state if hasattr(result, "state") else result.settings
        key = "s" if hasattr(result, "state") else "k"
        before = list(container[key])
        source[key].append(99)  # the caller edits what it passed in
        assert list(container[key]) == before, type(result).__name__
        with pytest.raises(TypeError, match="read-only"):
            container[key].append(1)
        with pytest.raises(TypeError, match="read-only"):
            container["new"] = 1
        if frame is not None:
            frame.iloc[0, 0] = -1  # the caller edits its table
            assert getattr(result, frame_name).iloc[0, 0] != -1, type(result).__name__


def test_a18_no_decision_to_combine_is_refused_not_a_stop():
    with pytest.raises(ValueError, match="at least one decision"):
        combine_decisions([])
    assert combine_decisions([mc.Decision.STOP]) is mc.Decision.STOP


def test_p2_3_view_names_the_criterion_and_the_standard_error():
    objects, fovs = _raster_well(drift=0.0)
    updates = _live(Setup(readout=READOUT), objects, fovs)
    stop = next(u for u in updates if _overall_stop(u))
    view = decision_view(stop.results)
    assert "met: scalar_precision_object_upper95 averaging_cv <= 0.05" in view
    assert "scalar_precision_fov_upper95 averaging_cv <= 0.05" in view
    assert "an averaging CV of 5%: the standard error is 5% of the mean, not a ±5% bound" in view
    before = next(u for u in updates if not _overall_stop(u))
    assert "not yet met: scalar_precision_fov_upper95 averaging_cv <= 0.05" in decision_view(
        before.results
    )
    rules = [s for s in Setup(readout=READOUT).declaration() if s.startswith("Stopping rules")]
    stop = "the standard error of the mean is 5% of the mean (its averaging CV), not on a bound"
    assert f"{stop} of ±5%" in rules[0]


def test_the_short_block_marks_continue_and_stop_and_names_what_is_missing():
    updates = _updates(_nucleus_setup(), *_well())
    first, eighth = updates[0].summary(), updates[7].summary()
    assert first.startswith("⏳ field 1 · MOCK · repeat 1A — continue: 1 of 3 rules say enough")
    assert "fixed 8 fields (7 more fields)" in _flat(first)
    assert "mean-precision rule, per field, 5%" in first
    assert eighth.startswith("✅ field 8 · MOCK · repeat 1A — stop: all 3 rules say enough")
    # The counts: this field, the group so far, in the declared unit.
    assert "f07.tif: 30 nuclei (read in" in eighth and "240 nuclei in 8 fields so far" in eighth
    # One table of the rules: each rule's verdict and its evidence against its threshold.
    assert "rule  " in eighth and "says" in eighth and "stop at" in eighth
    for result in updates[7].results:
        for name, quantity in result.evidence.quantities.items():
            if quantity.value is not None and name == "averaging_cv":
                assert _distinct(quantity.value, 0.05)[0] in eighth
    assert "✅ enough  8 fields" in eighth
    # A quantity not yet computable says why, once.
    assert "not computable yet (fewer than 2 acquired FOVs" in _flat(first)


def test_the_full_block_keeps_every_explain_line_and_every_view_entry():
    setup = _nucleus_setup(acquisition_order_col="order", stage_cols=("stage_x", "stage_y"))
    for update in _updates(setup, *_well(drift=0.01)):
        full = _flat(update.summary(detail="full"))
        for result in update.results:
            assert _flat(result.explain(update.order_check)) in full
            assert f"config {result.config_id}" in full
        for entry in _target_decisions(update.results, update.order_check):
            assert _flat(entry.text) in full
        where = ", ".join(f"{k}={v}" for k, v in update.fov_metadata.items())
        assert _flat(f"{where}: {update.n_objects} nuclei (read in") in full


def test_the_view_and_the_stop_caution_are_unchanged_by_their_parts():
    setup = _nucleus_setup(acquisition_order_col="order", stage_cols=("stage_x", "stage_y"))
    update = _updates(setup, *_well())[7]
    entries = _target_decisions(update.results, update.order_check)
    assert decision_view(update.results, update.order_check) == "; ".join(e.text for e in entries)
    assert entries[-1].component is None and entries[-1].decision is Decision.STOP
    for check in (None, update.order_check):
        for rare in (False, True):
            joined = "; ".join(stop_caution_parts(check, rare=rare))
            assert stop_caution(check, rare=rare) == joined
    assert stop_caution(None, rare=True) == f"{RANDOM_SAMPLE_ASSUMPTION}; it {RARE_ASSUMPTION}"
