"""Audit 3, group E: normalisation, the sufficiency report and the statistics it rests on."""

from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd
import pytest
from scenarios import FEATURE_COLS, ORDER_COL, SCENARIOS
from scipy import stats

import morphocoverage.sufficiency as sufficiency_module
from morphocoverage import (
    FixedFOVStrategy,
    FixedObjectStrategy,
    ObjectiveConfig,
    SamplingObjective,
    ScalarPrecisionStrategy,
    sufficiency_text,
)
from morphocoverage._wording import percent
from morphocoverage.benchmark import AcquisitionOrder
from morphocoverage.normalize import control_means, normalize_to_control
from morphocoverage.policies import EffortPolicy, ThresholdPolicy
from morphocoverage.statistics import (
    _interval,
    bootstrap_fields,
    bootstrap_nested,
    cluster_size,
    field_sample,
    fields_holding,
    jcb_required_objects,
    nested_variance_components,
    object_weighted_mean_se,
)
from morphocoverage.strategies import SupportNoveltyEstimator
from morphocoverage.strategies.base import SamplingStrategy
from morphocoverage.strategies.baselines import FixedAreaStrategy, ImagedAreaEstimator
from morphocoverage.sufficiency import sufficiency_report
from morphocoverage.sufficiency_tables import is_effort_rule
from morphocoverage.sufficiency_text import _next_fov_answer, novelty_definition
from morphocoverage.workflows import Setup

POPULATION = ObjectiveConfig(SamplingObjective.POPULATION)
PERMUTED = [AcquisitionOrder.permuted(seed) for seed in range(9)]
BOTH = [AcquisitionOrder.recorded(), *PERMUTED]


def _report(scenario, strategies, orders=BOTH, **kwargs):
    built = SCENARIOS[scenario]()
    return sufficiency_report(
        built.objects,
        built.fovs,
        feature_cols=list(FEATURE_COLS),
        strategies=strategies,
        objective=POPULATION,
        orders=orders,
        acquisition_order_col=ORDER_COL,
        **kwargs,
    )


def _signed_feature():
    """A log-ratio centred near 0 whose control mean is positive in both repeats."""
    rng = np.random.default_rng(2)
    rows, fields = [], []
    for repeat in ("r1", "r2"):
        for condition, centre in (("MOCK", 0.05), ("TREAT", 0.30)):
            for f in range(6):
                name = f"{condition}_{repeat}_{f}"
                fields.append({"image_file": name, "condition": condition, "repeat": repeat})
                rows += [
                    {"image_file": name, "condition": condition, "repeat": repeat, "x": v}
                    for v in rng.normal(centre, 0.3, 40)
                ]
    return pd.DataFrame(rows), pd.DataFrame(fields)


# --- statistics (foundations, verified here) ------------------------------------------------


def test_e1_empty_wells_do_not_narrow_the_nested_interval():
    rng = np.random.default_rng(0)
    filled = [rng.normal(10, 2, 30) for _ in range(6)]
    empty = [np.array([]) for _ in range(3)]
    alone = bootstrap_fields(filled, np.mean, n_boot=400, rng=np.random.default_rng(1))
    with_empty = bootstrap_nested(
        [filled, empty, empty], np.mean, n_boot=400, rng=np.random.default_rng(1)
    )
    assert with_empty == alone
    assert with_empty[2] - with_empty[1] > 0


def test_e2_intervals_are_widened_for_few_units():
    draws = np.random.default_rng(0).normal(0, 1, 20_000)
    for units in (3, 5, 10):
        _, low, high = _interval(0.0, draws, units, 0.95)
        raw_low, raw_high = np.percentile(draws, [2.5, 97.5])
        t_over_z = stats.t.ppf(0.975, units - 1) / stats.norm.ppf(0.975)
        widen = math.sqrt(units / (units - 1)) * t_over_z
        assert low == pytest.approx(raw_low * widen) and high == pytest.approx(raw_high * widen)


def test_e10_the_ratio_se_docstring_compares_relative_spreads():
    doc = object_weighted_mean_se.__doc__
    assert "relative to the mean, 0.013" in doc and "0.044" in doc
    rng = np.random.default_rng(3)
    sizes = [3, 5, 80, 120, 4, 6, 150, 2, 0, 0, 0, 0]
    centres = [10, 10, 13, 13, 10, 10, 13, 10, 10, 10, 10, 10]
    fields = [rng.normal(m, 2, n) for n, m in zip(sizes, centres, strict=True)]
    sums = np.array([f.sum() for f in fields])
    counts = np.array([len(f) for f in fields], float)
    mean, se, _ = object_weighted_mean_se(sums, counts)
    assert se / mean == pytest.approx(0.013, abs=0.0005)
    filled = np.flatnonzero(counts > 0)
    picks = filled[np.random.default_rng(0).integers(0, len(filled), (20_000, len(filled)))]
    spread = np.std(sums[picks].sum(1) / counts[picks].sum(1)) / mean
    assert spread == pytest.approx(0.044, abs=0.003)


def test_e11_variance_components_refuse_designs_they_cannot_estimate():
    values = np.arange(8, dtype=float)
    with pytest.raises(ValueError, match="do not nest"):
        nested_variance_components(
            values, [np.array([0, 0, 0, 0, 1, 1, 1, 1]), np.array([0, 1, 0, 1, 0, 1, 0, 1])]
        )
    with pytest.raises(ValueError, match="finite"):
        nested_variance_components(np.array([1.0, np.inf, 2.0]), [np.array([0, 0, 1])])
    with pytest.raises(ValueError, match="cannot be told apart"):
        nested_variance_components(values, [np.arange(8)])
    assert "can be negative" in nested_variance_components.__doc__


def test_e12_cluster_size_weights_crowded_fields():
    assert cluster_size(np.array([2, 0, 4])) == pytest.approx((4 + 16) / 6)
    assert cluster_size(np.array([5, 5, 5])) == 5
    assert math.isnan(cluster_size(np.array([0, 0])))


def test_e13_the_nested_bootstrap_resamples_the_top_level_only():
    # However the fields of a unit are split into lower levels, only whole top units
    # are drawn, so the interval is the same.
    rng = np.random.default_rng(0)
    units = [[rng.normal(u, 1, 20) for _ in range(4)] for u in range(5)]
    merged = [[np.concatenate(unit)] for unit in units]
    first = bootstrap_nested(units, np.mean, n_boot=300, rng=np.random.default_rng(4))
    second = bootstrap_nested(merged, np.mean, n_boot=300, rng=np.random.default_rng(4))
    assert first == pytest.approx(second)


def test_e14_small_helpers_validate_their_arguments():
    rng = np.random.default_rng(0)
    fields = [np.arange(3.0), np.arange(4.0)]
    assert field_sample(fields, 0, rng).size == 0
    assert fields_holding(fields, 0, rng) == 0
    with pytest.raises(ValueError):
        field_sample(fields, -1, rng)
    with pytest.raises(ValueError):
        fields_holding(fields, -1, rng)
    for arguments in ((float("nan"), 0.05), (0.1, 0.0), (-0.2, 0.05), (0.1, -0.05)):
        with pytest.raises(ValueError):
            jcb_required_objects(*arguments)


def test_e19_percent_rounds_half_up():
    assert percent(0.125) == "13%" and percent(0.135) == "14%"


# --- the sufficiency report ---------------------------------------------------------------


def test_e3_the_median_stop_says_how_many_orders_met_it():
    report = _report(
        "neighbour_correlation",
        [ScalarPrecisionStrategy("feature_a", 0.01, "fov", "point")],
        orders=PERMUTED,
    )
    answer = report.enough["answer"].iloc[0]
    stops = report.stops.iloc[0]
    met = int(stops["orders"] - stops["never_stopped"])
    assert 0 < met < 9
    assert f"in {met} of 9 random orders, first met after" in answer
    assert "never in" not in answer and "over 9 orders" not in answer


def test_e4_recorded_and_random_orders_are_summarised_apart():
    report = _report("neighbour_correlation", [FixedFOVStrategy(4)])
    stops = report.stops
    assert list(stops["order_kind"]) == ["recorded", "permuted"]
    assert list(stops["orders"]) == [1, 9]
    answer = report.enough["answer"].iloc[0]
    assert "in the recorded order, first met after 4 FOVs" in answer
    assert "in all 9 random orders, first met after 4 FOVs" in answer
    row = report.enough.iloc[0]
    assert row["order_kind"] == "permuted" and row["orders"] == 9
    recorded_only = _report(
        "neighbour_correlation", [FixedFOVStrategy(4)], orders=[AcquisitionOrder.recorded()]
    )
    assert recorded_only.enough["order_kind"].iloc[0] == "recorded"


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
    report = _report("unequal_fov_area", [area])
    assert (
        "fixed_area (imaged_area >= 50000): its target was reached"
        in (report.enough["answer"].iloc[0])
    )


def test_e6_effort_rules_say_target_reached_not_yes():
    report = _report(
        "homogeneous",
        [FixedFOVStrategy(3), ScalarPrecisionStrategy("feature_a", 0.5, "fov", "point")],
    )
    enough = report.enough.set_index("criterion")["sampled_enough"]
    assert enough["fixed_fov"] == "target reached"
    assert enough["scalar_precision_fov"] == "yes"
    assert not pd.isna(report.enough.set_index("criterion")["fovs_beyond_enough"]["fixed_fov"])


def test_e8_one_fov_is_singular():
    answer = _report("homogeneous", [FixedFOVStrategy(1)]).enough["answer"].iloc[0]
    assert "first met after 1 FOV;" in answer and "1 FOVs" not in answer


def test_e9_new_fovs_are_worded_by_the_support_that_makes_them_new():
    one = SupportNoveltyEstimator(1.0, 1, None, 3, 1)
    two = SupportNoveltyEstimator(1.0, 2, None, 3, 1)
    assert "with no other FOV within distance 1" in novelty_definition(one)
    assert "fewer than 1 other" not in novelty_definition(one)
    assert "with fewer than 2 other FOVs within distance 1" in novelty_definition(two)
    row = {
        "chance_new": 0.1,
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


def test_e15_few_fovs_accepts_numpy_integers():
    novelty = SupportNoveltyEstimator(1.0, 1, None, 3, 1)
    plain = _report("rare_clustered", [], orders=PERMUTED[:1], novelty=novelty, few_fovs=3)
    numpy = _report(
        "rare_clustered", [], orders=PERMUTED[:1], novelty=novelty, few_fovs=np.int64(3)
    )
    pd.testing.assert_frame_equal(plain.next_fov, numpy.next_fov)
    with pytest.raises(TypeError):
        _report("rare_clustered", [], orders=PERMUTED[:1], novelty=novelty, few_fovs=True)


def test_e17_the_docstring_gives_the_stop_as_a_median_and_range():
    assert "a median of 11 FOVs, up to 23" in sufficiency_module.__doc__


def test_e18_a_rule_that_cannot_be_judged_yet_says_why():
    built = SCENARIOS["condition_repeat_groups"]()
    fovs = built.fovs
    small = fovs[(fovs.condition == "control") & (fovs.repeat == "R1")].image_file.iloc[2:]
    fovs = fovs[~fovs.image_file.isin(small)]
    objects = built.objects[built.objects.image_file.isin(fovs.image_file)]
    report = sufficiency_report(
        objects,
        fovs,
        feature_cols=list(FEATURE_COLS),
        strategies=[FixedFOVStrategy(3)],
        objective=POPULATION,
        orders=PERMUTED[:1],
        novelty=SupportNoveltyEstimator(3.0, 1, None, 3, 3),
        novelty_target=0.3,
    )
    answers = report.enough.set_index("criterion")["answer"]
    novelty = answers["support_novelty"].iloc[0]
    assert "no, not met by the 2 acquired FOVs (it cannot be judged yet: fewer than 3" in novelty
    assert "cannot be judged" not in answers["fixed_fov"].iloc[0]


# --- normalisation ------------------------------------------------------------------------


def test_e7_a_control_with_negative_values_is_normalised_with_a_warning():
    objects, fovs = _signed_feature()
    with pytest.warns(UserWarning, match="negative values of 'x'"):
        normalised = normalize_to_control(
            objects, fovs, feature_cols=["x"], control={"condition": "MOCK"}
        )
    assert "not a multiple of the control" in normalised.attrs["normalization_note"]
    means = control_means(objects, fovs, feature_cols=["x"], control={"condition": "MOCK"})
    assert list(means["negative_features"]) == [("x",), ("x",)]
    setup = Setup(readout="x", control={"condition": "MOCK"}, normalize="control_mean")
    assert "Caution: the control objects hold negative values of 'x'" in (
        setup.normalized_sentence(fovs, objects)
    )
    # A positive feature is normalised without a word.
    positive = objects.assign(x=objects["x"] + 5)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        quiet = normalize_to_control(
            positive, fovs, feature_cols=["x"], control={"condition": "MOCK"}
        )
    assert "normalization_note" not in quiet.attrs
    assert "Caution" not in setup.normalized_sentence(fovs, positive)


def test_e16_control_objects_counts_the_values_the_mean_rests_on():
    objects, fovs = _signed_feature()
    objects = objects.assign(x=objects["x"] + 5)
    first = objects.index[(objects.condition == "MOCK") & (objects.repeat == "r1")][:3]
    objects.loc[first, "x"] = np.nan
    means = control_means(objects, fovs, feature_cols=["x"], control={"condition": "MOCK"})
    assert list(means["control_objects"]) == [240 - 3, 240]


# --- structure (J items in these files) ---------------------------------------------------


def test_j12_sufficiency_text_names_its_shared_private_names():
    assert "Shared-private: ``_policies``" in sufficiency_text.__doc__


def test_j8_the_pooled_group_label_and_feature_list_use_the_shared_wording():
    novelty = SupportNoveltyEstimator(1.0, 1, None, 3, 1)
    text = novelty_definition(novelty, ["a", "b", "c", "d", "e"])
    assert "on 5 features (a, b, c and 2 more)" in text
    report = _report("homogeneous", [FixedFOVStrategy(3)], group_cols=[])
    assert report.summary().startswith("all FOVs (pooled):")
