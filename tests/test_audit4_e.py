"""Audit 4, group E: acquisition order (Q2, D1, P2-4, P2-14).

Every stop says what it rests on: the order check of the fields seen so far
(a trend of the readout with acquisition order, alike neighbours on the
stage, the stage extent a rare-state stop covered), or without order and
stage columns the assumption that the fields are a random sample of the
well's positions. The check flags; it never changes a decision.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scenarios import ORDER_COL, SCENARIOS

from morphocoverage import OnlineSession, validate_input_tables
from morphocoverage.benchmark import AcquisitionOrder
from morphocoverage.evaluator import iter_analysis_groups, restrict_input
from morphocoverage.objectives import ObjectiveConfig, RareTarget, SamplingObjective
from morphocoverage.order_check import (
    RANDOM_SAMPLE_ASSUMPTION,
    RARE_ASSUMPTION,
    check_acquisition_order,
)
from morphocoverage.results import decision_view
from morphocoverage.sequential import _Acquisition, _in_acquisition_order
from morphocoverage.strategies import RareIncidenceStrategy, ScalarPrecisionStrategy
from morphocoverage.sufficiency import sufficiency_report
from morphocoverage.workflows import LiveTables, Setup, live_session

READOUT = "intensity"
GRID = 6


def _serpentine(grid: int) -> list[int]:
    order = []
    for row in range(grid):
        columns = range(grid) if row % 2 == 0 else range(grid - 1, -1, -1)
        order += [row * grid + c for c in columns]
    return order


def _raster_well(*, drift: float, rare_rows: int = 0, seed: int = 3, objects_per_field: int = 30):
    """A 6 x 6 well in serpentine raster order; the readout drifts by ``drift`` per field.

    With ``rare_rows``, objects of the last rows carry a rare label (every
    fourth object), which the raster reaches last.
    """
    rng = np.random.default_rng(seed)
    object_rows, fov_rows = [], []
    for rank, position in enumerate(_serpentine(GRID)):
        row, column = divmod(position, GRID)
        name = f"f{position:02d}.tif"
        mean = 100.0 * (1 + drift * rank)
        values = mean * np.exp(0.1 * rng.standard_normal(objects_per_field))
        rare = np.zeros(objects_per_field, bool)
        if rare_rows and row >= GRID - rare_rows:
            rare[::4] = True
        object_rows.append(
            pd.DataFrame(
                {
                    "condition": "MOCK",
                    "biological_repeat": "R1",
                    "image_file": name,
                    READOUT: values,
                    "rare": rare,
                }
            )
        )
        fov_rows.append(
            {
                "condition": "MOCK",
                "biological_repeat": "R1",
                "image_file": name,
                "order": rank,
                "stage_x": 500.0 * column,
                "stage_y": 500.0 * row,
            }
        )
    return pd.concat(object_rows, ignore_index=True), pd.DataFrame(fov_rows)


def _live(setup, objects, fovs):
    return LiveTables(live_session(setup)).update(objects, fovs)


def _overall_stop(update) -> bool:
    return "objective): stop" in decision_view(update.results)


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


def test_p4_2_report_stop_lines_carry_the_order_check():
    objects, fovs = _raster_well(drift=0.01)
    rules = [ScalarPrecisionStrategy(READOUT, 0.05, "fov", "upper_95")]
    common = {
        "feature_cols": [READOUT],
        "strategies": rules,
        "objective": ObjectiveConfig(SamplingObjective.POPULATION),
        "orders": [AcquisitionOrder.recorded()],
        "acquisition_order_col": "order",
    }
    checked = sufficiency_report(objects, fovs, readout=READOUT, **common)
    answer = checked.enough["answer"].iloc[0]
    assert checked.enough["sampled_enough"].iloc[0] == "yes"  # flagged, not vetoed
    assert "Caution: the field means of intensity trend with acquisition order" in answer
    plain = sufficiency_report(objects, fovs, **common).enough["answer"].iloc[0]
    assert f"The stop {RANDOM_SAMPLE_ASSUMPTION}." in plain


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


def test_p2_4_object_rule_line_states_its_assumption():
    objects, fovs = _raster_well(drift=0.0)
    session = OnlineSession(
        strategies=[
            ScalarPrecisionStrategy(READOUT, 0.05, "object", "upper_95"),
            ScalarPrecisionStrategy(READOUT, 0.05, "fov", "upper_95"),
        ],
        feature_cols=[READOUT],
        objective=ObjectiveConfig(SamplingObjective.POPULATION),
    )
    first = fovs.iloc[0]
    rows = objects[objects["image_file"].eq(first["image_file"])]
    per_object, per_field = session.add_fov(rows, first.to_dict())
    clause = "treats observations as independent draws: assumes a spatially uniform sample"
    assert clause in per_object.explain()
    assert clause not in per_field.explain()


def test_p2_14_rule_deciding_one_target_reads_decides_only():
    objects, fovs = _raster_well(drift=0.0, rare_rows=2)
    setup = Setup(
        readout=READOUT,
        rare=RareTarget(
            fov_incidence=0.2, detection_probability=0.8, min_fov_occurrences=1, phenotype="rare"
        ),
        rare_label="rare",
        rules=(
            ScalarPrecisionStrategy(READOUT, 0.05, "fov", "upper_95"),
            RareIncidenceStrategy(label_col="rare", prior_a=1.0, prior_b=1.0, max_risk=0.3),
        ),
    )
    last = _live(setup, objects, fovs)[-1]
    population, rare = (r.explain() for r in last.results)
    assert population.startswith("stop (decides the population summary only): averaging_cv")
    assert rare.startswith("stop (decides the rare phenotype only): rare_incidence_risk")
    assert "unsupported" not in population + rare
