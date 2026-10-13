"""Tests of ``order_check``: the acquisition-order check beside every stop."""

from __future__ import annotations

from pathlib import Path

from _helpers_results import _flat, _nucleus_setup, _updates, _well
from _helpers_strategies import READOUT, _live, _overall_stop, _raster_well
from scenarios import ORDER_COL, SCENARIOS

from morphocoverage import validate_input_tables
from morphocoverage._caveats import is_assumption
from morphocoverage.benchmark import AcquisitionOrder
from morphocoverage.evaluator import iter_analysis_groups, restrict_input
from morphocoverage.objectives import RareTarget
from morphocoverage.order_check import (
    NOT_CHECKED,
    RANDOM_SAMPLE_ASSUMPTION,
    RARE_ASSUMPTION,
    OrderCheck,
    check_acquisition_order,
    order_caveat,
)
from morphocoverage.results import Decision
from morphocoverage.sequential import _Acquisition, _in_acquisition_order
from morphocoverage.strategies import RareIncidenceStrategy, ScalarPrecisionStrategy
from morphocoverage.sufficiency import sufficiency_report
from morphocoverage.workflows import Setup, live_session


def test_an_order_check_that_flags_nothing_names_its_threshold_a21():
    check = OrderCheck("area", 20, "order", ("x", "y"), 0.4, 0.037, 0.1, 0.84)
    said = check.caution()
    assert "no trend with acquisition order or spatial autocorrelation flagged at the " in said
    assert "Bonferroni threshold of 0.025 (Spearman p = 0.037; Moran's I permutation p" in said
    assert "no spatial autocorrelation (" not in said


def test_p2_1_live_stops_carry_the_order_caution_and_are_never_vetoed():
    objects, fovs = _raster_well(drift=0.01)
    declared = Setup(
        readout=READOUT, acquisition_order_col="order", stage_cols=("stage_x", "stage_y")
    )
    updates = _live(declared, objects, fovs)
    flagged = [u for u in updates if _overall_stop(u) and u.order_check.trend]
    assert flagged, "the drift must be flagged at some stop"
    summary = " ".join(flagged[0].summary(detail="full").split())
    assert "caution: the field means of intensity trend with acquisition order" in summary
    assert "(a flag, not a veto" in summary
    # The short block states the caution once, and marks the stop to look at (⚠️).
    short = " ".join(flagged[0].summary().split())
    assert short.startswith("⚠️ field ") and "stop: both rules say enough" in short
    assert short.count("the field means of intensity trend with acquisition order") == 1
    assert "(a flag, not a veto" in short
    # The flag changes no decision: the same rules without columns decide the same.
    plain = _live(Setup(readout=READOUT), objects, fovs)
    assert [[r.without_timing() for r in u.results] for u in updates] == [
        [r.without_timing() for r in u.results] for u in plain
    ]
    # Without either column, every stop states the assumption it rests on.
    stops = [u for u in plain if _overall_stop(u)]
    assert stops and all(u.order_check is None for u in plain)
    assert all(RANDOM_SAMPLE_ASSUMPTION in " ".join(u.summary().split()) for u in stops)
    assert all(RANDOM_SAMPLE_ASSUMPTION in " ".join(u.summary("full").split()) for u in stops)
    # The declaration says so too, and recommends random or dispersed orders.
    line = [s for s in Setup(readout=READOUT).declaration() if s.startswith("Acquisition order")]
    assert RANDOM_SAMPLE_ASSUMPTION in line[0] and "random or dispersed order" in line[0]


def test_p2_2_rare_stop_states_where_the_fields_have_looked():
    objects, fovs = _raster_well(drift=0.0, rare_rows=2)
    rules = (
        ScalarPrecisionStrategy(READOUT, 0.05, "fov", "upper_95"),
        RareIncidenceStrategy(label_col="rare", prior_a=1.0, prior_b=1.0, max_risk=0.3),
    )
    target = RareTarget(
        fov_incidence=0.2, detection_probability=0.8, min_fov_occurrences=1, phenotype="rare"
    )
    setup = Setup(
        readout=READOUT,
        rare=target,
        rare_label="rare",
        rules=rules,
        acquisition_order_col="order",
        stage_cols=("stage_x", "stage_y"),
    )
    updates = _live(setup, objects, fovs)
    rare_stops = [u for u in updates if u.results[1].rule_decision.value == "stop" and u.n_objects]
    early = [
        u
        for u in rare_stops
        if not objects.loc[
            objects["image_file"].isin(
                [v.fov_metadata["image_file"] for v in updates[: updates.index(u) + 1]]
            ),
            "rare",
        ].any()
    ]
    assert early, "the raster must stop the rare rule before it reaches the rare rows"
    line = early[0].results[1].explain(early[0].order_check)
    assert "the fields so far cover the stage at stage_x 0 to 2000, stage_y 0" in line
    assert RARE_ASSUMPTION in line
    # In the report, the share of the stage extent the recorded order had covered.
    report = sufficiency_report(
        objects,
        fovs,
        feature_cols=[READOUT],
        strategies=[rules[1]],
        objective=setup.objective(),
        orders=[AcquisitionOrder.recorded()],
        acquisition_order_col="order",
        readout=READOUT,
        stage_cols=("stage_x", "stage_y"),
    )
    answer = report.enough["answer"].iloc[0]
    # Stopped after the first 5 of 36 fields, one row of a 6 x 6 grid: 5 / 36.
    assert "the 5 fields up to the recorded stop covered 14% of the stage extent" in answer


def test_p3_4_order_check_is_a_pure_function_of_the_fields_seen():
    # Live and replayed prefixes give the same check, at every step of a contract
    # scenario with stage columns (neighbour_correlation), and the permutation
    # p-value does not depend on the order in which the same fields arrived.
    scenario = SCENARIOS["neighbour_correlation"]()
    stage = ("stage_x", "stage_y")
    order_col = ORDER_COL
    setup = Setup(readout="feature_a", acquisition_order_col=order_col, stage_cols=tuple(stage))
    session = live_session(setup)
    data = validate_input_tables(
        scenario.objects, fovs=scenario.fovs, feature_cols=list(setup.columns)
    )
    for group, group_data in iter_analysis_groups(data):
        acquisition = _Acquisition(group_data, _in_acquisition_order(group, group_data, order_col))
        for step, (_, fov) in enumerate(acquisition.ordered.iterrows(), 1):
            mine = scenario.objects["image_file"].eq(fov["image_file"])
            for column in group:
                mine &= scenario.objects[column].eq(fov[column])
            results = session.add_fov(scenario.objects.loc[mine].copy(), fov.to_dict())
            live = session.order_check(results[0].group)
            replayed = check_acquisition_order(
                acquisition.prefix(step), "feature_a", order_col=order_col, stage_cols=stage
            )
            assert live == replayed, (group, step)
        whole = acquisition.prefix(acquisition.n_fovs)
        shuffled = restrict_input(whole, whole.objects, whole.fovs.iloc[::-1])
        a = check_acquisition_order(whole, "feature_a", stage_cols=stage)
        b = check_acquisition_order(shuffled, "feature_a", stage_cols=stage)
        assert (a.moran_i, a.moran_p) == (b.moran_i, b.moran_p)
    # The calibration check no longer claims to reveal neighbour correlation.
    source = Path(__file__).resolve().parents[1] / "src" / "morphocoverage" / "calibration.py"
    assert "one thing the check can reveal" not in source.read_text()


INDEPENDENT = "treats observations as independent draws: assumes a spatially uniform sample"


def test_caveats_are_stated_once_at_a_stop_and_wait_while_the_objective_continues():
    updates = _updates(_nucleus_setup(), *_well())
    stop = updates[7].summary()
    flat = _flat(stop)
    assert flat.count(RANDOM_SAMPLE_ASSUMPTION) == 1
    assert flat.count(INDEPENDENT) == 1 and "Mean-precision rule, per nucleus, 5% treats" in flat
    assert flat.count("the standard error is 5% of the mean, not a ±5% bound") == 1
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


def test_a_stop_rests_on_an_assumption_only_when_the_order_check_ran_and_found_nothing():
    clear = "order check: no trend with acquisition order (Spearman p = 0.4) over 20 fields; " + (
        f"still {RANDOM_SAMPLE_ASSUMPTION}"
    )
    assert is_assumption(order_caveat(clear))
    assert "every stop still treats" in order_caveat(clear)
    flagged = "caution: the field means of a trend with acquisition order (...)"
    assert not is_assumption(order_caveat(flagged)) and order_caveat(flagged) == flagged
    # No check run: a missing declaration is a caution about these data, and says so.
    unchecked = order_caveat(RANDOM_SAMPLE_ASSUMPTION)
    assert not is_assumption(unchecked) and unchecked.startswith(NOT_CHECKED)
    assert not is_assumption(order_caveat("not tested: fewer than 4 fields with a mean"))
