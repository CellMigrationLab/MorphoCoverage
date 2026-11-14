"""Tests for the retrospective sufficiency report."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from _helpers_retrospective import _two_group_tables
from _helpers_strategies import (
    PERMUTED_NINE,
    READOUT,
    _flat,
    _named_scenario_report,
    _raster_well,
    _report_both_orders,
)
from _helpers_sufficiency import _novel_acquisition
from _helpers_tables import _four_cell_tables
from _helpers_workflows import _scenario_report
from scenarios import FEATURE_COLS, ORDER_COL, SCENARIOS

import morphocoverage.sufficiency as sufficiency_module
from morphocoverage import (
    FixedFOVStrategy,
    FixedObjectStrategy,
    ObjectiveConfig,
    SamplingObjective,
    ScalarPrecisionStrategy,
    run_strategies,
    sufficiency_text,
)
from morphocoverage._caveats import is_assumption
from morphocoverage._style import CAUTION
from morphocoverage._wording import fields as fields_text
from morphocoverage.benchmark import AcquisitionOrder
from morphocoverage.bounds import upper_bound
from morphocoverage.order_check import RANDOM_SAMPLE_ASSUMPTION
from morphocoverage.policies import EffortPolicy, ThresholdPolicy
from morphocoverage.retrospective import microscopist_report
from morphocoverage.strategies import DescriptorPlateauStrategy, SupportNoveltyEstimator
from morphocoverage.strategies.base import SamplingStrategy
from morphocoverage.strategies.baselines import FixedAreaStrategy, ImagedAreaEstimator
from morphocoverage.strategies.rare_planning import fovs_to_detect
from morphocoverage.sufficiency import CalibrationSettings, sufficiency_report
from morphocoverage.sufficiency_short import _novelty_lines, met_count, rule_tallies
from morphocoverage.sufficiency_tables import _quantile, is_effort_rule
from morphocoverage.sufficiency_text import _next_fov_answer, novelty_definition
from morphocoverage.transforms import Standardize

POPULATION = ObjectiveConfig(SamplingObjective.POPULATION)
PERMUTED = [AcquisitionOrder.permuted(seed) for seed in range(5)]


def _report(scenario, strategies, orders=PERMUTED, **kwargs):
    built = SCENARIOS[scenario]()
    return sufficiency_report(
        built.objects,
        built.fovs,
        feature_cols=list(FEATURE_COLS),
        strategies=strategies,
        objective=POPULATION,
        orders=orders,
        **kwargs,
    )


def test_verdicts_match_the_strategies_on_the_full_acquisition():
    built = SCENARIOS["homogeneous"]()
    strategies = [FixedFOVStrategy(5), ScalarPrecisionStrategy("feature_a", 0.001, "fov", "point")]
    report = _report("homogeneous", strategies)
    direct = run_strategies(
        built.objects,
        fovs=built.fovs,
        strategies=strategies,
        feature_cols=list(FEATURE_COLS),
        objective=POPULATION,
    )
    assert list(report.final["verdict"]) == [r.decision.value for r in direct]
    assert list(report.final["n_fovs"]) == [20, 20]
    assert report.final["q.averaging_cv"].iloc[1] == pytest.approx(
        direct[1].evidence.value("averaging_cv")
    )


def test_stop_points_and_spare_fovs_follow_the_replay():
    stops = _report("homogeneous", [FixedFOVStrategy(6)]).stops.iloc[0]
    assert (stops["acquired_fovs"], stops["orders"], stops["never_stopped"]) == (20, 5, 0)
    assert stops["stop_fov_median"] == stops["stop_fov_p10"] == stops["stop_fov_p90"] == 6
    assert stops["spare_fovs_median"] == 14


def test_criteria_never_met_are_counted_and_have_no_stop_point():
    stops = _report("homogeneous", [FixedFOVStrategy(10**6)]).stops.iloc[0]
    assert stops["never_stopped"] == 5
    assert stops["stop_fov_median"] is None and stops["spare_fovs_median"] is None
    final = _report("homogeneous", [FixedFOVStrategy(10**6)]).final.iloc[0]
    assert final["verdict"] == "continue"


def test_variants_of_one_strategy_stay_apart():
    report = _report("homogeneous", [FixedFOVStrategy(3), FixedFOVStrategy(9)])
    assert report.stops["config_id"].nunique() == 2
    assert sorted(report.stops["stop_fov_median"]) == [3, 9]
    assert set(report.final["strategy"]) == {"fixed_fov"}


def test_analysis_groups_are_reported_separately():
    report = _report("condition_repeat_groups", [FixedFOVStrategy(2)])
    assert len(report.final) == len(report.stops) == 4
    assert {"condition", "biological_repeat"} <= set(report.stops.columns)
    assert set(report.stops["acquired_fovs"]) == {8}


def test_the_recorded_order_needs_an_order_column():
    with pytest.raises(ValueError, match="acquisition_order_col"):
        _report("homogeneous", [FixedFOVStrategy(3)], orders=[AcquisitionOrder.recorded()])
    recorded = _report(
        "homogeneous",
        [FixedFOVStrategy(3)],
        orders=[AcquisitionOrder.recorded()],
        acquisition_order_col=ORDER_COL,
    )
    assert recorded.stops["orders"].iloc[0] == 1


@pytest.mark.parametrize("strategies, orders", [([], PERMUTED), ([FixedFOVStrategy(3)], [])])
def test_empty_inputs_are_refused(strategies, orders):
    with pytest.raises(ValueError):
        _report("homogeneous", strategies, orders=orders)


@pytest.mark.parametrize("groups", [[], ["condition", "biological_repeat"]])
@pytest.mark.parametrize("novelty", [None, SupportNoveltyEstimator(0.5, 1, None, 2, 3)])
def test_an_acquisition_without_fovs_is_refused_whatever_its_groups(groups, novelty):
    """#128: zero FOV rows raised a KeyError or 'No objects to concatenate' with group columns."""
    fovs = pd.DataFrame({c: pd.Series([], dtype=str) for c in ["position_id", *groups]})
    objects = fovs.assign(a=pd.Series([], dtype=float))
    with pytest.raises(ValueError, match="at least one acquired FOV"):
        sufficiency_report(
            objects,
            fovs,
            feature_cols=["a"],
            strategies=[] if novelty else [FixedFOVStrategy(3)],
            objective=POPULATION,
            orders=PERMUTED,
            novelty=novelty,
            fov_col="position_id",
        )


def test_a_rules_standing_assumption_is_not_called_a_caution():
    """#139: the plateau rule's standing assumption read "Caution:" in the answer column."""
    from morphocoverage._caveats import assumption
    from morphocoverage.strategies import DescriptorPlateauStrategy
    from morphocoverage.sufficiency_text import _enough_answer

    rule = DescriptorPlateauStrategy("feature_a", 0.05, 3, ORDER_COL)
    answer = _report("homogeneous", [rule]).enough["answer"].iloc[0]
    assert "Caution" not in answer and " The tolerance bounds how much the descriptors" in answer
    row = {
        "strategy": "custom",
        "config_id": "c",
        "policies": "",
        "n_fovs": 5,
        "verdict": "continue",
        "warning_list": ("measured here", assumption("true of every dataset")),
    }
    assert _enough_answer(row, []).endswith(" Caution: measured here. True of every dataset.")


NOVELTY = SupportNoveltyEstimator(0.5, 1, None, 2, 3)


def _novelty_report(strategies=(), **kwargs):
    objects, fovs = _novel_acquisition()
    return sufficiency_report(
        objects,
        fovs,
        feature_cols=["a", "b"],
        strategies=list(strategies),
        objective=POPULATION,
        orders=PERMUTED,
        novelty=NOVELTY,
        **kwargs,
    )


def test_next_fov_chance_counts_fovs_with_enough_new_cells():
    report = _novelty_report()
    row = report.next_fov.iloc[0]
    # Two of 13 acquired FOVs (one empty) hold a 3-cell state no other FOV shows;
    # the single odd cell does not count.
    assert (row["fovs_imaged"], row["fovs_with_new"]) == (13, 2)
    assert row["chance_new"] == pytest.approx(2 / 13)
    assert row["chance_new_at_most"] == pytest.approx(upper_bound(2, 13, 0.95))
    assert pd.isna(row["extra_fovs_if_nothing_new"])
    assert report.enough.empty and report.calibration.empty


def test_a_target_beyond_the_projection_range_is_said_not_a_crash():
    """#136: a target 100000 more FOVs cannot reach left None, and the answer raised TypeError."""
    report = _novelty_report(novelty_target=1e-6)
    row = report.next_fov.iloc[0]
    assert pd.isna(row["extra_fovs_to_rule_out_target"])
    assert pd.isna(row["extra_fovs_if_nothing_new"])
    assert row["answer"].count("more than 100000 more FOVs") == 2
    assert "more than 100000 more fields" in " ".join(report.summary("full").split())


def test_new_observations_lists_only_cells_of_new_fovs_without_features():
    new = _novelty_report().new_observations
    assert sorted(new["image_file"].unique()) == ["f3", "f7"] and len(new) == 6
    assert not {"a", "b"} & set(new.columns) and "label_id" in new.columns


def test_a_target_gives_fovs_still_needed_and_a_criterion():
    report = _novelty_report(novelty_target=0.2)
    row = report.next_fov.iloc[0]
    needed = int(row["extra_fovs_if_nothing_new"])
    assert upper_bound(2, 13 + needed, 0.95) <= 0.2 < upper_bound(2, 12 + needed, 0.95)
    # 2 of 13 is below 20%, so the bound falls below it at the current rate too, later.
    assert row["extra_fovs_at_current_rate"] >= needed
    criterion = report.enough.iloc[0]
    assert criterion["criterion"] == "support_novelty"
    assert criterion["settings"] == "lofo_novel_fov_upper <= 0.2"
    assert criterion["sampled_enough"] == "no"


def test_summary_answers_both_questions_in_words():
    report = _novelty_report(
        [FixedFOVStrategy(5)],
        novelty_target=0.2,
        calibration=CalibrationSettings((6,), 10, 0),
    )
    # The sentences, one per group and rule, are the answer columns.
    answers = "\n".join([*report.enough["answer"], *report.next_fov["answer"]])
    assert "fixed_fov" in answers and "its target was reached by the 13 acquired FOVs" in answers
    assert "Chance: 15% (nominal 95% upper bound" in answers and "2 of 13 FOVs" in answers
    assert "Check on this dataset: predicting from 6 random FOVs" in answers
    # The summaries give both answers: the rules in a table, the next field in a few lines.
    for detail in ("short", "full"):
        text = " ".join(report.summary(detail=detail).split())
        assert "Each rule by its own threshold" in text and "One more field:" in text
        assert "fixed 5 fields" in text and "target reached" in text
    # The exact definition of new is in the full form; the short one says it in one line.
    short = " ".join(report.summary().split())
    assert "'New': at least 3 similar cells that no other field shows" in short
    full = " ".join(report.summary(detail="full").split())
    assert "'New' means a cluster of at least 3 similar observations" in full
    assert "with no other FOV within distance 0.5" in full
    assert "chance 15% (nominal 95% bound" in full and "2 of 13 fields" in full
    assert "Check on this dataset: from 6 random fields" in full
    assert list(report.calibration["fovs_imaged"]) == [6]


def test_readable_criteria_state_the_threshold_direction():
    report = _report("homogeneous", [ScalarPrecisionStrategy("feature_a", 0.001, "fov", "point")])
    assert report.enough["settings"].iloc[0] == "averaging_cv <= 0.001"
    assert report.enough["sampled_enough"].iloc[0] in {"yes", "no"}
    assert report.next_fov.empty and report.new_observations.empty


@pytest.mark.parametrize(
    "kwargs",
    [
        {"strategies": [], "novelty": None},
        {"novelty": None, "novelty_target": 0.1},
        {"novelty_target": 1.0},
        {"novelty": None, "calibration": CalibrationSettings((5,), 5, 0)},
        {"novelty": "support_novelty"},
    ],
)
def test_inconsistent_questions_are_refused(kwargs):
    objects, fovs = _novel_acquisition()
    settings = {"strategies": [FixedFOVStrategy(3)], "novelty": NOVELTY, **kwargs}
    with pytest.raises((TypeError, ValueError)):
        sufficiency_report(
            objects,
            fovs,
            feature_cols=["a", "b"],
            objective=POPULATION,
            orders=PERMUTED,
            **settings,
        )


def test_the_report_states_what_the_fovs_rule_out():
    # Rule of three: a state absent from all k FOVs is in at most
    # 1 - 0.05 ** (1 / k) of FOVs at 95%, whatever else was new (entry 3 of
    # docs/preliminary_findings.md shows why this must be said explicitly).
    report = _novelty_report(novelty_target=0.2)
    row = report.next_fov.iloc[0]
    assert row["unseen_state_max_frequency"] == pytest.approx(1 - 0.05 ** (1 / 13))
    needed = int(row["extra_fovs_to_rule_out_target"])
    assert upper_bound(0, 13 + needed, 0.95) <= 0.2 < upper_bound(0, 12 + needed, 0.95)
    answer = report.next_fov["answer"].iloc[0]
    assert "is in at most 21% of FOVs (95% confidence, about 3 / 13)" in answer
    assert "Rarer states can remain." in answer
    assert "To rule out unseen states in more than 20% of FOVs: at least 1 more FOV" in answer
    assert "To bring the nominal 95% bound for the next FOV below 20%" in answer
    short = " ".join(report.summary().split())
    assert "is in at most 21% of fields (95% confidence); rarer states can remain" in short
    # The projections are in the full form; the short form points there.
    assert "To rule out unseen states" not in short
    assert "or bring the next-field bound below 20%, is in the full report" in short
    full = " ".join(report.summary(detail="full").split())
    assert "Not yet seen: at most 21% of fields (95% confidence, about 3 / 13)" in full
    assert "at least 1 more field with nothing new to rule out unseen states" in full
    # Without a target the statement is still made, without the extra FOVs.
    plain = _novelty_report()
    assert pd.isna(plain.next_fov["extra_fovs_to_rule_out_target"].iloc[0])
    assert "Not yet seen:" in plain.summary() and "To rule out" not in plain.summary()


def test_novelty_cols_keep_the_definition_of_new_apart_from_the_readout():
    # The rules read the readout; "new" is judged on the declared features only, and the
    # definition names them. Without novelty_cols, novelty runs on every feature column.
    built = SCENARIOS["condition_repeat_groups"]()
    built.objects["readout"] = built.objects["feature_a"] * 3 + 40
    novelty = SupportNoveltyEstimator(1.0, 1, None, 5, 1)
    rule = ScalarPrecisionStrategy("readout", 0.05, "fov", "upper_95")
    common = {
        "objective": POPULATION,
        "orders": PERMUTED,
        "novelty": novelty,
        "novelty_target": 0.2,
    }
    split = sufficiency_report(
        built.objects,
        built.fovs,
        feature_cols=["readout", *FEATURE_COLS],
        novelty_cols=list(FEATURE_COLS),
        strategies=[rule],
        **common,
    )
    novelty_only = sufficiency_report(
        built.objects, built.fovs, feature_cols=list(FEATURE_COLS), strategies=[], **common
    )
    rule_only = sufficiency_report(
        built.objects,
        built.fovs,
        feature_cols=["readout"],
        strategies=[rule],
        objective=POPULATION,
        orders=PERMUTED,
    )
    pd.testing.assert_frame_equal(split.next_fov, novelty_only.next_fov)
    pd.testing.assert_frame_equal(
        split.stops[split.stops["strategy"] == rule.name].reset_index(drop=True),
        rule_only.stops.reset_index(drop=True),
    )
    assert set(split.stops["strategy"]) == {rule.name, *novelty_only.stops["strategy"]}
    # The same rows, order and dtypes as one run over the same rules would give.
    combined = sufficiency_report(
        built.objects,
        built.fovs,
        feature_cols=list(FEATURE_COLS),
        strategies=[ScalarPrecisionStrategy("feature_a", 0.05, "fov", "upper_95")],
        **common,
    )
    for table, name in (("final", "strategy"), ("stops", "strategy"), ("enough", "criterion")):
        mine, reference = getattr(split, table), getattr(combined, table)
        assert list(mine[name]) == list(reference[name]), table
        assert list(mine["condition"]) == list(reference["condition"]), table
        assert mine.dtypes.to_dict() == reference.dtypes.to_dict(), table
    assert "(feature_a and feature_b)" in split.summary(detail="full")
    with pytest.raises(ValueError, match="novelty_cols"):
        sufficiency_report(
            built.objects,
            built.fovs,
            feature_cols=["readout"],
            novelty_cols=["feature_a"],
            strategies=[rule],
            **common,
        )


def test_the_rule_counts_say_how_many_groups_were_judged() -> None:
    # B19: "0 of 40 (1 not defined)" hid 12 groups with too few fields to judge.
    tally = {"met": 0, "groups": 40, "undefined": 1, "waiting": 12}
    assert met_count(tally) == "0 of 27 judged; 12 too few fields; 1 undefined"
    assert met_count({"met": 3, "groups": 4, "undefined": 0, "waiting": 0}) == "3 of 4"
    assert met_count({"met": 0, "groups": 4, "undefined": 0, "waiting": 4}, "group") == (
        "none judged; 4 too few fields"
    )
    # e01: three fields per group, too few for the per-field bootstrap bound in every group.
    text = _scenario_report("e01_two_repeats").summary("full")
    row = next(line for line in text.splitlines() if "mean-precision rule, per field, 5%  " in line)
    assert "none judged; 4 too few fields" in row
    assert " 0 of 4 " not in row


def test_e9_one_replayed_order_is_singular():
    objects, fovs = _two_group_tables()
    report = sufficiency_report(
        objects,
        fovs,
        feature_cols=["a"],
        strategies=[FixedFOVStrategy(3)],
        objective=POPULATION,
        orders=[AcquisitionOrder.permuted(0)],
    )
    answers = " ".join(report.enough["answer"])
    assert "in the random order, first met after" in answers and "1 orders" not in answers
    for detail in ("short", "full"):
        assert "1 orders" not in report.summary(detail=detail)
    assert "1 random order: median" in report.summary(detail="full")


def test_e5_effort_rules_are_recognised_by_their_structure():
    area = FixedAreaStrategy(50_000.0, "area_um2")
    assert is_effort_rule(area) and is_effort_rule(FixedObjectStrategy(10))
    assert is_effort_rule(FixedFOVStrategy(3))
    # An effort policy under any name is an effort rule; a threshold on a quantity
    # whose settings read like one is not.
    assert is_effort_rule(SamplingStrategy([], [EffortPolicy("n_fovs", 3)], name="my_rule"))
    lookalike = SamplingStrategy(
        [ImagedAreaEstimator("area_um2")], [ThresholdPolicy("imaged_area", 3.0)], name="n_area"
    )
    assert not is_effort_rule(lookalike)
    report = _report_both_orders("unequal_fov_area", [area])
    answer = report.enough["answer"].iloc[0]
    assert answer.startswith("fixed imaged area 50000 (fixed_area: imaged_area >= 50000; config ")
    assert "): its target was reached" in answer
    short = " ".join(report.summary().split())  # the short form wraps its lines
    assert "Fixed imaged area 50000: a fixed amount of effort reaching its target" in short


def test_e9_new_fovs_are_worded_by_the_support_that_makes_them_new():
    one = SupportNoveltyEstimator(1.0, 1, None, 3, 1)
    two = SupportNoveltyEstimator(1.0, 2, None, 3, 1)
    assert "with no other FOV within distance 1" in novelty_definition(one)
    assert "fewer than 1 other" not in novelty_definition(one)
    assert "with fewer than 2 other FOVs within distance 1" in novelty_definition(two)
    row = {
        "chance_new": 0.1,
        "uninformative": None,
        "chance_new_at_most": 0.3,
        "confidence": 0.95,
        "fovs_with_new": 3,
        "fovs_imaged": 30,
        "unseen_state_max_frequency": 0.1,
    }
    assert "3 of 30 FOVs showed a state that no other FOV shows." in _next_fov_answer(
        row, None, None, 1
    )
    assert "3 of 30 FOVs showed a state that fewer than 2 other FOVs show." in (
        _next_fov_answer(row, None, None, 2)
    )


def test_e17_the_docstring_gives_the_stop_as_a_median_and_range():
    assert "a median of 11 FOVs, up to 23" in sufficiency_module.__doc__


def test_e18_a_rule_that_cannot_be_judged_yet_says_why():
    built = SCENARIOS["condition_repeat_groups"]()
    fovs = built.fovs
    small = fovs[(fovs.condition == "control") & (fovs.biological_repeat == "R1")].image_file.iloc[
        2:
    ]
    fovs = fovs[~fovs.image_file.isin(small)]
    objects = built.objects[built.objects.image_file.isin(fovs.image_file)]
    report = sufficiency_report(
        objects,
        fovs,
        feature_cols=list(FEATURE_COLS),
        strategies=[FixedFOVStrategy(3)],
        objective=POPULATION,
        orders=PERMUTED_NINE[:1],
        novelty=SupportNoveltyEstimator(3.0, 1, None, 3, 3),
        novelty_target=0.3,
    )
    answers = report.enough.set_index("criterion")["answer"]
    novelty = answers["support_novelty"].iloc[0]
    assert "no, not met by the 2 acquired FOVs (it cannot be judged yet: fewer than 3" in novelty
    assert "cannot be judged" not in answers["fixed_fov"].iloc[0]


def test_j12_sufficiency_text_names_its_shared_private_names():
    assert "Shared-private: ``_policies``" in sufficiency_text.__doc__


def test_j8_the_pooled_group_label_and_feature_list_use_the_shared_wording():
    novelty = SupportNoveltyEstimator(1.0, 1, None, 3, 1)
    text = novelty_definition(novelty, ["a", "b", "c", "d", "e"])
    assert "on 5 features (a, b, c and 2 more)" in text
    report = _report_both_orders("homogeneous", [FixedFOVStrategy(3)], group_cols=[])
    # The pooled group is named once, by the shared style (``_style.group_label``).
    assert "\n\nall fields (pooled) — 20 fields acquired:\n" in report.summary(detail="full")


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


def _row(**extra):
    row = {
        "chance_new": 0.02,
        "uninformative": None,
        "chance_new_at_most": 0.12,
        "confidence": 0.95,
        "fovs_with_new": 0,
        "fovs_imaged": 24,
        "unseen_state_max_frequency": upper_bound(0, 24, 0.95),
        "fovs_without_observations": 0,
        "extra_fovs_if_nothing_new": 34,
        "extra_fovs_at_current_rate": 40,
        "extra_fovs_to_rule_out_target": 34,
        "min_novel_observations": 3,
    }
    return pd.Series({**row, **extra})


def test_p6_6_states_below_m_objects_per_field_are_said_never_to_count():
    text = _next_fov_answer(_row(), None, None)
    assert (
        "A state carried by fewer than 3 similar observations per FOV is never counted, "
        "however many FOVs hold it." in text
    )
    assert "never counted" not in _next_fov_answer(_row(min_novel_observations=1), None, None)


def _wells_table():
    rng = np.random.default_rng(4)
    rows, fields = [], []
    for well in ("A", "B", "C"):
        for j in range(8):
            name = f"{well}{j}"
            fields.append(
                {
                    "condition": "c",
                    "biological_repeat": "R1",
                    "technical_repeat": well,
                    "position_id": name,
                }
            )
            for x, y in rng.normal(size=(15, 2)):
                rows.append(
                    {"condition": "c", "biological_repeat": "R1", "technical_repeat": well}
                    | {"position_id": name, "a": x, "b": y}
                )
    return pd.DataFrame(rows), pd.DataFrame(fields)


def test_p3_8_a_group_pooling_technical_repeats_bounds_their_share():
    objects, fovs = _wells_table()
    report = sufficiency_report(
        objects,
        fovs,
        feature_cols=["a", "b"],
        strategies=[],
        objective=POPULATION,
        orders=[AcquisitionOrder.permuted(0)],
        novelty=SupportNoveltyEstimator(3.0, 1, None, 5, 3),
        group_cols=["condition", "biological_repeat"],
    )
    row = report.next_fov.iloc[0]
    assert row["technical_repeats"] == 3
    assert row["unseen_state_max_technical_share"] == pytest.approx(upper_bound(0, 3, 0.95))
    assert "Fields of one technical repeat are not independent draws" in row["answer"]
    per_well = sufficiency_report(
        objects,
        fovs,
        feature_cols=["a", "b"],
        strategies=[],
        objective=POPULATION,
        orders=[AcquisitionOrder.permuted(0)],
        novelty=SupportNoveltyEstimator(3.0, 1, None, 5, 3),
    )
    assert "technical repeat are not" not in " ".join(per_well.next_fov["answer"])


def test_p3_4_p4_9_the_next_field_answer_names_its_assumptions():
    calibrated = {"predicted": 0.04, "actual": 0.05, "bound_held": 0.95, "fovs_imaged": 10}
    text = _next_fov_answer(_row(), 0.05, {**calibrated, "draws": 20})
    assert "it treats FOVs as independent, representative draws" in text
    assert "neighbouring fields acquired in a raster" in text
    assert "The bound holds at a number of FOVs fixed in advance, not at the first FOV" in text
    assert "this check cannot reveal neighbour correlation" in text


def test_p9_8_sufficiency_report_runs_its_replays_on_n_jobs_with_identical_answers():
    built = SCENARIOS["homogeneous"]()
    arguments = {
        "feature_cols": list(FEATURE_COLS),
        "strategies": [ScalarPrecisionStrategy("feature_a", 0.05, "fov", "upper_95")],
        "objective": ObjectiveConfig(SamplingObjective.POPULATION),
        "orders": [AcquisitionOrder.permuted(s) for s in range(3)],
    }
    serial = sufficiency_report(built.objects, built.fovs, **arguments)
    parallel = sufficiency_report(built.objects, built.fovs, n_jobs=2, **arguments)
    pd.testing.assert_frame_equal(serial.stops, parallel.stops)
    assert serial.summary() == parallel.summary()
    assert "stop_fov_median_mc_se" in serial.stops.columns


DISTRIBUTION = ObjectiveConfig(SamplingObjective.DISTRIBUTION)


def test_e9_stop_counts_are_never_interpolated_or_rounded_away():
    assert _quantile(np.array([2.0, 3.0]), 0.9) == 3.0
    assert _quantile(np.array([]), 0.9) is None
    assert fields_text(2.5) == "2.5 fields" and fields_text(6.0) == "6 fields"


def _few_report(confidence, min_other_fovs):
    objects, fovs = _novel_acquisition(n_fovs=40, state_fovs=(3, 4, 20, 21), outlier_fovs=())
    # Two states, each shared by two neighbouring FOVs (3 and 4, 20 and 21): recurrent,
    # not new, under min_other_fovs=1 (a state in one FOV only is new, not "seen in few").
    shifted = objects["image_file"].isin(["f4", "f21"]) & (objects["a"] > 15)
    objects.loc[shifted, ["a", "b"]] -= 5.0
    return sufficiency_report(
        objects,
        fovs,
        feature_cols=["a", "b"],
        strategies=[],
        objective=DISTRIBUTION,
        orders=[AcquisitionOrder.permuted(0)],
        novelty=SupportNoveltyEstimator(0.5, min_other_fovs, None, 2, 3, confidence=confidence),
        few_fovs=3,
    )


def test_e3_e6_e7_rare_state_sentences_follow_the_settings():
    report = _few_report(0.8, 1)
    row, answer = report.next_fov.iloc[0], report.next_fov["answer"].iloc[0]
    # E6: the zero-success bound at 80% is about 1.6 / k, not 3 / k.
    assert "about 1.6 / 41" in answer and "3 / 41" not in answer
    # E7: with min_other_fovs=1 a recurrent state is in 2 or more FOVs.
    assert "(a state seen in 2 to 3 FOVs;" in answer
    # E3: the pooled incidence is not the incidence of each state.
    assert row["fovs_to_see_fewest"] == fovs_to_detect(2 / 41, 0.8, 1)
    assert row["fovs_to_see_fewest"] > row["fovs_to_see_again"]
    assert f"one seen in only 2 FOVs needs about {row['fovs_to_see_fewest']}" in answer
    answer = _few_report(0.95, 2).next_fov["answer"].iloc[0]
    assert "about 3 / 41" in answer and "(a state seen in 3 FOVs;" in answer


def test_e8_fields_without_observations_are_counted_when_novelty_is_not_estimable():
    objects, fovs = _novel_acquisition()
    objects.loc[0, "a"] = np.nan
    report = sufficiency_report(
        objects,
        fovs,
        feature_cols=["a", "b"],
        strategies=[],
        objective=POPULATION,
        orders=PERMUTED,
        novelty=SupportNoveltyEstimator(0.5, 1, None, 2, 3),
    )
    row = report.next_fov.iloc[0]
    assert pd.isna(row["chance_new"])
    assert row["fovs_without_observations"] == 1


def test_sufficiency_summary_works_without_a_novelty_definition():
    # With group columns and no novelty, the next-FOV table is empty and has no
    # group columns; summary() raised KeyError('condition').
    built = SCENARIOS["homogeneous"]()
    report = sufficiency_report(
        built.objects,
        built.fovs,
        feature_cols=list(FEATURE_COLS),
        strategies=[FixedFOVStrategy(5)],
        objective=POPULATION,
        orders=[AcquisitionOrder.permuted(0)],
    )
    text = report.summary(detail="full")
    assert "control · repeat R1 — " in text
    assert "Analysis groups: condition · biological repeat." in text
    assert "Each rule by its own threshold" in text and "One more field" not in text
    assert "One more field" not in report.summary()


def test_the_unseen_state_bound_names_the_fovs_without_observations():
    # 3 fields with cells and 10 empty ones: the "about 3 / 13" count is right
    # for representative draws, but the reader must see that 10 could not
    # have shown anything.
    built = SCENARIOS["homogeneous"]()
    fovs = built.fovs.iloc[:13].copy()
    objects = built.objects[built.objects["image_file"].isin(fovs["image_file"].iloc[:3])]
    report = sufficiency_report(
        objects,
        fovs,
        feature_cols=list(FEATURE_COLS),
        strategies=[],
        objective=POPULATION,
        orders=[AcquisitionOrder.permuted(0)],
        novelty=SupportNoveltyEstimator(np.sqrt(2), 1, Standardize(), 2, 3),
    )
    assert report.next_fov["fovs_without_observations"].iloc[0] == 10
    answer = report.next_fov["answer"].iloc[0]
    assert "about 3 / 13" in answer
    assert "The count includes 10 FOVs without observations, which cannot show a state." in answer
    full = " ".join(report.summary(detail="full").split())  # both forms wrap their lines
    assert "about 3 / 13" in full and "13 fields (10 fields without observations)" in full
    short = " ".join(report.summary().split())
    assert "include fields without observations, which cannot show a state" in short


def test_a_fixed_rule_reports_a_reached_target_and_a_threshold_rule_a_met_threshold():
    from morphocoverage import ScalarPrecisionStrategy

    built = SCENARIOS["homogeneous"]()
    report = sufficiency_report(
        built.objects,
        built.fovs,
        feature_cols=list(FEATURE_COLS),
        strategies=[
            FixedFOVStrategy(5),
            ScalarPrecisionStrategy("feature_a", 0.05, "fov", "point"),
        ],
        objective=POPULATION,
        orders=[AcquisitionOrder.permuted(0)],
    )
    answers = report.enough.set_index("criterion")["answer"]
    assert "its target was reached by the 20 acquired FOVs" in answers["fixed_fov"]
    assert "yes, its threshold was met by the 20 acquired FOVs" in answers["scalar_precision_fov"]
    assert "yes, met by" not in report.summary()


def test_a_state_seen_in_two_neighbouring_fovs_is_reported_as_seen_in_few():
    # rare_clustered: 10 rare cells in each of two adjacent FOVs of 40. With one
    # supporting FOV they cover each other, so the next-FOV chance is 0 and the
    # state went unmentioned; few_fovs=3 names the two fields.
    built = SCENARIOS["rare_clustered"]()
    novelty = SupportNoveltyEstimator(np.sqrt(2), 1, Standardize(), 2, 3)
    report = sufficiency_report(
        built.objects,
        built.fovs,
        feature_cols=list(FEATURE_COLS),
        strategies=[],
        objective=POPULATION,
        orders=[AcquisitionOrder.permuted(0)],
        novelty=novelty,
        few_fovs=3,
    )
    row = report.next_fov.iloc[0]
    assert row["fovs_with_new"] == 0 and row["fovs_with_state_in_few"] == 2
    assert (
        "Seen in few FOVs: 2 of 40 FOVs hold a state that other FOVs show too, but fewer "
        "than 3 of them" in row["answer"]
    )
    full = " ".join(report.summary(detail="full").split())  # both forms wrap their lines
    assert "Seen in few fields: 2 of 40 fields hold a state seen in 2 to 3 fields" in full
    short = " ".join(report.summary().split())
    assert "look at those fields before calling the sample settled" in short
    with pytest.raises(ValueError, match="few_fovs must exceed"):
        sufficiency_report(
            built.objects,
            built.fovs,
            feature_cols=list(FEATURE_COLS),
            strategies=[],
            objective=POPULATION,
            orders=[AcquisitionOrder.permuted(0)],
            novelty=novelty,
            few_fovs=1,
        )


def test_a_state_seen_in_few_fovs_gets_a_planning_line_from_its_incidence():
    from morphocoverage.strategies.rare_planning import fovs_to_detect

    built = SCENARIOS["rare_clustered"]()
    novelty = SupportNoveltyEstimator(np.sqrt(2), 1, Standardize(), 2, 3)
    report = sufficiency_report(
        built.objects,
        built.fovs,
        feature_cols=list(FEATURE_COLS),
        strategies=[],
        objective=POPULATION,
        orders=[AcquisitionOrder.permuted(0)],
        novelty=novelty,
        few_fovs=3,
    )
    row = report.next_fov.iloc[0]
    assert row["few_state_incidence"] == pytest.approx(2 / 40)
    assert 0 < row["few_state_incidence_low"] < 0.05 < row["few_state_incidence_high"] < 0.2
    assert row["fovs_to_see_again"] == fovs_to_detect(0.05, 0.95, 1) == 59
    assert row["fovs_to_see_again_low_end"] == fovs_to_detect(
        row["few_state_incidence_low"], 0.95, 1
    )
    answer = row["answer"]
    assert "Planning for such a state: these FOVs together are 5% of FOVs (95% interval" in answer
    assert "image about 59 independent representative FOVs, up to" in answer
    assert "Fields that cluster the state need more." in answer
    full = " ".join(report.summary(detail="full").split())  # both forms wrap their lines
    assert "together 5% of fields (95% interval" in full
    assert "about 59 independent representative fields, up to" in full
    assert "Fields that cluster a state need more fields than the planning numbers." in full
    short = " ".join(report.summary().split())
    # The recurrence projection is in the full form; the short form keeps the actionable line.
    assert "takes about 59 independent representative fields" not in short
    assert "look at those fields before calling the sample settled" in short


def test_sufficiency_report_refusals():
    objects, fovs = _four_cell_tables()
    asked = {
        "feature_cols": ["area"],
        "strategies": [FixedFOVStrategy(2)],
        "objective": POPULATION,
        "orders": [AcquisitionOrder.permuted(0)],
    }
    with pytest.raises(TypeError) as error:
        sufficiency_report(objects, fovs, **asked, novelty_cols=[])
    assert str(error.value) == "novelty_cols must be a non-empty sequence of column names"
    with pytest.raises(ValueError) as error:
        sufficiency_report(objects, fovs, **asked, few_fovs=3)
    assert str(error.value) == "few_fovs needs a novelty definition"
    with pytest.raises(ValueError) as error:
        sufficiency_report(objects, fovs.assign(_report_row_order=0), **asked)
    assert str(error.value) == "the FOV table already has a column '_report_row_order'"


def test_short_novelty_lines_with_pooled_technical_repeats():
    frame = pd.DataFrame(
        {
            "chance_new": [0.1, 0.2],
            "reason": [None, None],
            "uninformative": [None, None],
            "confidence": [0.95, 0.95],
            "chance_new_at_most": [0.3, 0.4],
            "fovs_with_new": [0, 0],
            "unseen_state_max_frequency": [0.2, 0.25],
            "unseen_state_max_technical_share": [0.5, np.nan],
            "min_novel_observations": [1, 1],
            "fovs_without_observations": [0, 0],
            "condition": ["A", "B"],
        }
    )
    report = SimpleNamespace(
        next_fov=frame,
        group_cols=("condition",),
        observation_unit="cell",
        observation_unit_declared=True,
    )
    lines, cautions = _novelty_lines(report, None)
    assert lines == [
        (
            "One more field: no field showed a state counted as new, in both groups; nominal 95% "
            "bound for the next field 30 to 40%."
        ),
        (
            "Not yet seen: a state no field has shown is in at most 20 to 25% of fields (95% "
            "confidence); rarer states can remain."
        ),
        (
            "Pooled technical repeats (1 of 2 groups): a state none of them showed is in at most "
            "50% of technical repeats (95% confidence)."
        ),
    ]
    assert cautions == [
        (
            "the next-field bound is approximate: the leave-one-field-out outcomes share their "
            "reference fields"
        ),
        (
            "the novelty bounds treat fields as independent, representative draws: neighbouring "
            "fields acquired in a raster over patchy states make them too low"
        ),
        (
            "fields of one technical repeat are not independent draws: where a group pools "
            "technical repeats, the field bound is optimistic"
        ),
    ]
    assert [is_assumption(c) for c in cautions] == [True, True, False]


def test_next_fov_answer_when_states_in_few_fields_are_not_estimable():
    row = {
        "chance_new": 0.1,
        "reason": None,
        "uninformative": None,
        "confidence": 0.95,
        "chance_new_at_most": 0.3,
        "fovs_with_new": 1,
        "fovs_imaged": 10,
        "fovs_without_observations": 0,
        "min_novel_observations": 1,
        "unseen_state_max_frequency": 0.26,
        "few_fovs": 3,
        "fovs_with_state_in_few": np.nan,
        "few_reason": "too few fields",
        "technical_repeats": None,
    }
    assert _next_fov_answer(row, None, None) == (
        "Chance: 10% (nominal 95% upper bound 30%, approximate: the leave-one-FOV-out outcomes "
        "share their reference FOVs, and it treats FOVs as independent, representative draws, "
        "so neighbouring fields acquired in a raster over patchy states make it too low); 1 of "
        "10 FOVs showed a state that no other FOV shows.\nNot yet seen: a state this definition "
        "detects but no FOV has shown is in at most 26% of FOVs (95% confidence, about 3 / 10), "
        "if FOVs are independent, representative draws. Rarer states can remain.\nSeen in few "
        "FOVs: not estimable (too few fields)."
    )


def test_a_rules_standing_warning_does_not_count_as_a_caution_about_the_data():
    report = _named_scenario_report(
        "homogeneous",
        [DescriptorPlateauStrategy("feature_a", 50.0, 3, "acquisition_order")],
        readout="feature_a",
        stage_cols=("stage_x", "stage_y"),
    )
    warnings = [w for texts in report.final["warning_list"] for w in texts]
    assert warnings and all(is_assumption(w) for w in warnings)
    assert all(t["warned"] == 0 for t in rule_tallies(report))
    section = microscopist_report(sufficiency=report).short["enough"]
    tolerance = "the tolerance bounds how much the descriptors changed"
    assert _flat(section).count(tolerance.capitalize()[:30]) + _flat(section).count(tolerance) == 1
    assert tolerance not in "\n".join(
        line for line in section.splitlines() if line.lstrip().startswith(CAUTION)
    )
