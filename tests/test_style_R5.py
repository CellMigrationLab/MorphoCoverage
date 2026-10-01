"""Readable reports, step R5: the live view (``LiveUpdate.summary``).

Short (the default): one block per field, its headline marked ✅ stop, ⚠️ stop
the order check flags, ⏳ continue (the live view's own marker) or ➖ not
decided; one table of the rules; at a stop each caveat once. Full: every
``explain`` line and every ``decision_view`` entry, one per line with a marker.
The text is a pure function of the update.
"""

import dataclasses

import numpy as np
import pandas as pd
import pytest

from morphocoverage import FixedFOVStrategy
from morphocoverage._style import width
from morphocoverage.objectives import RareTarget
from morphocoverage.order_check import RANDOM_SAMPLE_ASSUMPTION, RARE_ASSUMPTION
from morphocoverage.results import (
    Decision,
    _distinct,
    _target_decisions,
    decision_view,
    stop_caution,
    stop_caution_parts,
)
from morphocoverage.strategies import RareIncidenceStrategy, ScalarPrecisionStrategy
from morphocoverage.workflows import LiveTables, Setup, live_session
from morphocoverage.workflows.live import print_update

READOUT = "intensity"
INDEPENDENT = "treats observations as independent draws: assumes a spatially uniform sample"


def _flat(text: str) -> str:
    return " ".join(text.split())


def _well(*, drift: float = 0.0, n_fields: int = 16, rare_rows: int = 0, seed: int = 5):
    """A 4-column well in raster order; the readout drifts by ``drift`` per field."""
    rng = np.random.default_rng(seed)
    objects, fovs = [], []
    for rank in range(n_fields):
        row, column = divmod(rank, 4)
        name = f"f{rank:02d}.tif"
        values = 100.0 * (1 + drift * rank) * np.exp(0.1 * rng.standard_normal(30))
        rare = np.zeros(30, bool)
        if rare_rows and row >= n_fields // 4 - rare_rows:
            rare[::4] = True
        objects.append(
            pd.DataFrame(
                {
                    "condition": "MOCK",
                    "biological_repeat": "1A",
                    "image_file": name,
                    READOUT: values,
                }
            ).assign(rare=rare)
        )
        fovs.append(
            {
                "condition": "MOCK",
                "biological_repeat": "1A",
                "image_file": name,
                "order": rank,
                "stage_x": 500.0 * column,
                "stage_y": 500.0 * row,
            }
        )
    return objects, pd.DataFrame(fovs)


def _updates(setup, objects, fovs):
    """Feed the fields one at a time, each with its objects, as an acquisition does."""
    live = LiveTables(live_session(setup))
    updates = []
    for step in range(len(fovs)):
        updates += live.update(pd.concat(objects[: step + 1]), fovs.iloc[: step + 1])
    return updates


def _setup(**extra):
    rules = (
        FixedFOVStrategy(8),
        ScalarPrecisionStrategy(READOUT, 0.05, "object", "upper_95"),
        ScalarPrecisionStrategy(READOUT, 0.05, "fov", "upper_95"),
    )
    return Setup(readout=READOUT, observation_unit="nucleus", rules=rules, **extra)


def test_the_short_block_marks_continue_and_stop_and_names_what_is_missing():
    updates = _updates(_setup(), *_well())
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


def test_caveats_are_stated_once_at_a_stop_and_wait_while_the_objective_continues():
    updates = _updates(_setup(), *_well())
    stop = updates[7].summary()
    flat = _flat(stop)
    assert flat.count(RANDOM_SAMPLE_ASSUMPTION) == 1
    assert flat.count(INDEPENDENT) == 1 and "Mean-precision rule, per nucleus, 5% treats" in flat
    assert flat.count("one standard error of 5% of the mean, not a ±5% bound") == 1
    # No order check ran here: that is a caution about these fields (⚠️); the rules'
    # independent draws and what an averaging CV is are standing assumptions.
    assert stop.count("⚠️") == 1
    assert "⚠️ The acquisition order was not checked, so every stop treats" in stop
    standing = _flat(stop.split("Assumptions:")[1])
    assert INDEPENDENT in standing and "not a ±5% bound" in standing
    # While the objective says continue no stop is acted on: its caveats wait (the full
    # form states them beside every rule that says stop, as explain does).
    early = updates[0]
    assert "⚠️" not in early.summary()
    assert INDEPENDENT in _flat(early.summary(detail="full"))


def test_a_stop_the_order_check_flags_is_marked_to_look_at():
    setup = _setup(acquisition_order_col="order", stage_cols=("stage_x", "stage_y"))
    updates = _updates(setup, *_well(drift=0.02))
    flagged = [u for u in updates if u.order_check.flagged and u.summary().count("stop:")]
    assert flagged
    short = _flat(flagged[0].summary())
    assert short.startswith("⚠️ field ") and "but the order check flags these fields" in short
    assert short.count("(a flag, not a veto") == 1
    calm = [u for u in updates if not u.order_check.flagged and "— stop:" in u.summary()]
    assert all(u.summary().startswith("✅") for u in calm)


def test_a_target_without_a_rule_reads_not_decided_and_a_rare_stop_says_where_it_looked():
    target = RareTarget(
        fov_incidence=0.2, detection_probability=0.8, min_fov_occurrences=1, phenotype="rare"
    )
    objects, fovs = _well(rare_rows=1)
    rules = (
        ScalarPrecisionStrategy(READOUT, 0.05, "fov", "upper_95"),
        RareIncidenceStrategy(label_col="rare", prior_a=1.0, prior_b=1.0, max_risk=0.3),
    )
    common = {"readout": READOUT, "rare": target, "rare_label": "rare"}
    alone = _updates(Setup(rules=rules[:1], **common), objects, fovs)[-1].summary()
    assert alone.startswith(
        "➖ field 16 · MOCK · repeat 1A — not decided: no rule decides the rare"
    )
    assert "rare phenotype ➖ no rule decides it" in _flat(alone)
    staged = Setup(rules=rules, stage_cols=("stage_x", "stage_y"), **common)
    updates = _updates(staged, objects, fovs)
    both = [u for u in updates if all(r.rule_decision is Decision.STOP for r in u.results)]
    assert both
    short = _flat(both[0].summary())
    assert "rare phenotype ✅ stop" in short
    assert "The rare-phenotype stop: the fields so far cover the stage at" in short
    assert short.count(RARE_ASSUMPTION) == 1 and short.count(RANDOM_SAMPLE_ASSUMPTION) == 1


def test_the_full_block_keeps_every_explain_line_and_every_view_entry():
    setup = _setup(acquisition_order_col="order", stage_cols=("stage_x", "stage_y"))
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
    setup = _setup(acquisition_order_col="order", stage_cols=("stage_x", "stage_y"))
    update = _updates(setup, *_well())[7]
    entries = _target_decisions(update.results, update.order_check)
    assert decision_view(update.results, update.order_check) == "; ".join(e.text for e in entries)
    assert entries[-1].component is None and entries[-1].decision is Decision.STOP
    for check in (None, update.order_check):
        for rare in (False, True):
            joined = "; ".join(stop_caution_parts(check, rare=rare))
            assert stop_caution(check, rare=rare) == joined
    assert stop_caution(None, rare=True) == f"{RANDOM_SAMPLE_ASSUMPTION}; it {RARE_ASSUMPTION}"


def test_the_text_is_a_pure_function_of_the_update_and_fits_the_page():
    first = _updates(_setup(), *_well())
    second = _updates(_setup(), *_well())
    for a, b in zip(first, second, strict=True):
        a, b = (dataclasses.replace(u, read_seconds=0.0) for u in (a, b))
        for detail in ("short", "full"):
            assert a.summary(detail) == b.summary(detail)
        assert all(width(line) <= 100 for line in a.summary().splitlines())


def test_detail_is_checked_and_print_update_passes_it(capsys):
    update = _updates(_setup(), *_well(n_fields=2))[-1]
    with pytest.raises(ValueError, match="short"):
        update.summary(detail="long")
    print_update(update, detail="full")
    assert capsys.readouterr().out.strip() == update.summary("full")
    print_update(update)
    assert capsys.readouterr().out.strip() == update.summary()
