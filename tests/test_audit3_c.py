"""Regressions for the third audit, group C (scalar precision, JCB planning, descriptor
baselines, FOV bootstrap), one test per finding (docs/audit/README.md)."""

import itertools
import json
import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from morphocoverage import (
    ObjectiveConfig,
    SamplingObjective,
    evaluate_acquisition_prefixes,
    plan_objects_to_average,
    run_strategies,
    statistics,
    validate_input_tables,
)
from morphocoverage.benchmark import AcquisitionOrder
from morphocoverage.sampling_unit import object_rules
from morphocoverage.strategies import (
    DescriptorPlateauStrategy,
    DescriptorSpreadStrategy,
    FOVBootstrapEstimator,
    FOVBootstrapStrategy,
    ScalarPrecisionStrategy,
    _averaging_cv,
    base,
    baselines,
    fov_bootstrap,
    jcb_planning,
    scalar_precision,
    stabilization,
)
from morphocoverage.strategies._averaging_cv import _field_size_check
from morphocoverage.strategies.scalar_precision import fov_averaging_cv, required_fovs
from morphocoverage.sufficiency import sufficiency_report
from morphocoverage.transforms import PCA

POPULATION = ObjectiveConfig(SamplingObjective.POPULATION)
DISTRIBUTION = ObjectiveConfig(SamplingObjective.DISTRIBUTION)
PHENOTYPE = ObjectiveConfig(SamplingObjective.PHENOTYPE)


def _acquisition(values_per_fov, feature="area"):
    """Objects and an FOV table (with an acquisition order) from values per FOV."""
    rows = [
        {"position_id": f"p{j}", feature: value}
        for j, values in enumerate(values_per_fov)
        for value in values
    ]
    objects = pd.DataFrame(rows, columns=["position_id", feature])
    fovs = pd.DataFrame(
        {
            "position_id": [f"p{j}" for j in range(len(values_per_fov))],
            "acq": np.arange(1.0, len(values_per_fov) + 1),
        }
    )
    return objects, fovs


def _words(doc):
    """A docstring with its line breaks and indentation as single spaces."""
    return " ".join(doc.split())


def _run(objects, fovs, strategies, objective=POPULATION, feature="area", **kw):
    return run_strategies(
        objects,
        fovs=fovs,
        strategies=strategies,
        feature_cols=[feature],
        objective=objective,
        fov_col="position_id",
        **kw,
    )


def _replay(objects, fovs, strategies, incremental=False, feature="area"):
    return evaluate_acquisition_prefixes(
        objects,
        fovs,
        strategies=strategies,
        feature_cols=[feature],
        acquisition_order_col="acq",
        objective=POPULATION,
        fov_col="position_id",
        incremental=incremental,
    )


def test_c1_a_feature_constant_so_far_is_not_yet_enough_data_in_every_variant():
    rng = np.random.default_rng(0)
    # Two saturated fields (255, the camera maximum), then fields that vary widely.
    values = [[255.0] * 6, [255.0] * 5] + [list(rng.uniform(50, 255, 8)) for _ in range(4)]
    objects, fovs = _acquisition(values)
    strategies = [
        ScalarPrecisionStrategy("area", 0.05, unit, sd)
        for unit in ("object", "fov")
        for sd in ("point", "upper_95")
    ]
    full = _replay(objects, fovs, strategies)
    incremental = _replay(objects, fovs, strategies, incremental=True)
    for a, b in zip(full, incremental, strict=True):
        qa, qb = (r.result.evidence.quantities["averaging_cv"] for r in (a, b))
        assert qa.availability == qb.availability and a.result.decision == b.result.decision
        if a.step <= 2:
            unit = a.result.parameters["resampling_unit"]
            assert qa.availability == "insufficient_data", (a.step, a.result.strategy)
            if unit == "object":
                assert qa.reason == "no spread among the objects so far"
            elif a.step == 2:  # one FOV with objects is too few for the FOV unit anyway
                assert qa.reason == "no spread among the FOV means so far"
            assert a.result.decision == "continue"
    # Once the fields differ, every variant has a CV again.
    assert all(r.result.evidence.value("averaging_cv") > 0 for r in full if r.step == 3)
    # The observation unit names the rows in the reason.
    [cells] = _run(
        *_acquisition([[5.0, 5.0], [5.0]]),
        [ScalarPrecisionStrategy("area", 0.05, "object", "point")],
        observation_unit="cell",
    )
    assert cells.evidence.quantities["averaging_cv"].reason == "no spread among the cells so far"


def _field_size_acquisition():
    """Crowded fields hold small objects, sparse ones large objects (a density effect)."""
    sizes = [150, 120, 100, 90, 5, 4, 3, 3, 2, 2, 140, 3]
    means = [5, 5.5, 6, 6, 20, 21, 22, 20, 21, 23, 5, 22]
    rng = np.random.default_rng(2)
    return _acquisition([list(np.abs(rng.normal(m, 0.5, n))) for n, m in zip(sizes, means)])


def _summary(objects, fovs, strategies, objective=POPULATION):
    report = sufficiency_report(
        objects,
        fovs,
        feature_cols=["area"],
        strategies=strategies,
        objective=objective,
        orders=[AcquisitionOrder.recorded()],
        acquisition_order_col="acq",
        fov_col="position_id",
    )
    # The sentences per group and rule (the answer column), then the short summary, whose
    # wrapped lines are joined so that a caveat reads as one string.
    return "\n".join(report.enough["answer"]) + "\n" + " ".join(report.summary().split())


def test_c2_the_sufficiency_sentence_carries_the_rules_own_warnings():
    objects, fovs = _field_size_acquisition()
    text = _summary(objects, fovs, [ScalarPrecisionStrategy("area", 0.1, "fov", "point")])
    assert "Caution: FOV size and FOV mean are related (Spearman rho -0." in text
    assert "point SD, in the group: FOV size and FOV mean are related (Spearman rho -0." in text
    # The paper's advice for a between-object CV above 50% is printed too.
    rng = np.random.default_rng(1)
    wide = _acquisition([list(rng.exponential(10.0, 20)) for _ in range(6)])
    text = _summary(*wide, [ScalarPrecisionStrategy("area", 0.05, "object", "point")])
    assert "Caution: the SD between objects exceeds 50% CV" in text
    assert "point SD, in the group: the SD between objects exceeds 50% CV" in text


def test_c3_the_object_unit_jcs_rule_states_its_random_object_assumption():
    rng = np.random.default_rng(1)
    objects, fovs = _acquisition([list(rng.normal(10, 2, 15)) for _ in range(8)])
    text = _summary(
        objects,
        fovs,
        [
            DescriptorSpreadStrategy("area", 5.0, "object", 50, 0),
            DescriptorSpreadStrategy("area", 5.0, "fov", 50, 0),
        ],
        objective=DISTRIBUTION,
    )
    lines = {line.split(" (")[0].strip(" -"): line for line in text.splitlines() if "(" in line}
    clause = "assumes a spatially uniform sample"
    assert clause in lines["JCS spread, per object, ≤5"]
    assert clause not in lines["JCS spread, per field, ≤5"]
    # Stated once in the short summary, for the object-unit rule only.
    assert "JCS spread, per object, ≤5 treats objects as independent draws: it assumes" in text
    assert "JCS spread, per field, ≤5 treats" not in text


def test_c4_text_and_boolean_order_columns_are_refused_like_the_acquisition_order():
    rng = np.random.default_rng(4)
    objects, fovs = _acquisition([list(rng.normal(10, 2, 5)) for _ in range(12)])
    pilot = ScalarPrecisionStrategy("area", 0.02, "object", "point", pilot_fovs=3, order_col="o")
    plateau = DescriptorPlateauStrategy("area", 0.5, 3, "o")
    for strategy in (pilot, plateau):
        # "1".."12" read from file names: as text, "10" would sort before "2".
        text = fovs.assign(o=(fovs["acq"].astype(int)).astype(str))
        with pytest.raises(ValueError, match="text would sort '10' before '2'"):
            _run(objects, text, [strategy])
        with pytest.raises(ValueError, match="holds bool values"):
            _run(objects, fovs.assign(o=fovs["acq"] > 6), [strategy])
        with pytest.raises(TypeError, match="categorical"):
            _run(objects, fovs.assign(o=pd.Categorical(fovs["acq"])), [strategy])
        # Numbers and times are an order.
        _run(objects, fovs.assign(o=fovs["acq"]), [strategy])
        times = pd.Timestamp("2026-01-01") + pd.to_timedelta(fovs["acq"], unit="min")
        _run(objects, fovs.assign(o=times), [strategy])
    # The replay refuses the same column with the same rule.
    with pytest.raises(ValueError, match="text would sort '10' before '2'"):
        evaluate_acquisition_prefixes(
            objects,
            fovs.assign(acq=fovs["acq"].astype(int).astype(str)),
            strategies=[pilot.__class__("area", 0.02, "object", "point")],
            feature_cols=["area"],
            acquisition_order_col="acq",
            objective=POPULATION,
            fov_col="position_id",
        )


def test_c5_categorical_sample_columns_plan_only_the_samples_in_the_pilot(monkeypatch):
    # pandas 2 groups categoricals over every combination of categories unless told not to.
    groupby = pd.DataFrame.groupby

    def pandas2_groupby(self, *args, **kwargs):
        kwargs.setdefault("observed", False)
        return groupby(self, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "groupby", pandas2_groupby)
    rng = np.random.default_rng(0)
    rows = [
        {"condition": c, "repeat": r, "position_id": f"{c}{r}{j}", "area": rng.normal(10, 2)}
        for c, r in (("control", "r1"), ("control", "r2"), ("treated", "r1"))
        for j in range(3)
        for _ in range(20)
    ]
    pilot = pd.DataFrame(rows).astype({"condition": "category", "repeat": "category"})
    plan = plan_objects_to_average(
        pilot,
        feature="area",
        target_cv=0.05,
        sd_estimate="point",
        sample_cols=["condition", "repeat"],
        control={"condition": "control"},
        fov_col="position_id",
    )
    assert plan.n_samples == 3
    assert ("treated", "r2") not in plan.pilot_samples


def test_c6_a_transform_that_can_never_be_fitted_is_not_applicable_from_the_first_fov():
    rng = np.random.default_rng(1)
    values = [list(rng.normal(10, 2, 5)) for _ in range(6)]
    objects, fovs = _acquisition(values)
    objects["other"] = rng.normal(0, 1, len(objects))

    def replay(strategy):
        return evaluate_acquisition_prefixes(
            objects,
            fovs,
            strategies=[strategy],
            feature_cols=["area", "other"],
            acquisition_order_col="acq",
            objective=POPULATION,
            fov_col="position_id",
        )

    never = replay(FOVBootstrapStrategy("centroid", 0.05, PCA(3), 50, 0, 2))
    for step in never:
        quantity = step.result.evidence.quantities["fov_bootstrap_centroid_max_se"]
        assert quantity.availability == "not_applicable"
        assert "can never be fitted on 2 features" in quantity.reason
        assert step.result.decision is None
    # A transform that can be fitted once more FOVs arrive is not yet enough data.
    later = replay(FOVBootstrapStrategy("centroid", 5.0, PCA(2), 50, 0, 2))
    first = later[0].result.evidence.quantities["fov_bootstrap_centroid_max_se"]
    assert first.availability == "insufficient_data"
    assert later[-1].result.evidence.quantities["fov_bootstrap_centroid_max_se"].value > 0


def test_c7_the_field_size_p_value_is_exact_for_few_fovs():
    counts = np.array([2.0, 3.0, 4.0, 5.0])
    found = _field_size_check(np.array([1.0, 1.1, 1.2, 1.3]) * counts, counts)
    # A perfect rank order of 4 FOVs: 2 of the 24 pairings are as extreme, not p = 0.
    assert found["field_size_correlation"] == pytest.approx(1.0)
    assert found["field_size_correlation_p"] == pytest.approx(2 / 24)
    assert "field_size_warning" not in found
    # The exact p-value is the share of pairings at least as extreme, ties included.
    sizes = np.array([3.0, 3.0, 7.0, 9.0, 12.0, 20.0])
    means = np.array([2.0, 1.5, 1.4, 1.1, 1.1, 0.7])
    rho = stats.spearmanr(sizes, means).statistic
    pairings = [stats.spearmanr(sizes, p).statistic for p in itertools.permutations(means)]
    exact = np.mean(np.abs(pairings) >= abs(rho) - 1e-9)
    found = _field_size_check(means * sizes, sizes)
    assert found["field_size_correlation_p"] == pytest.approx(exact)
    # Under independence the warning rate stays at or below its 5% level.
    rng = np.random.default_rng(0)
    for k in (4, 5, 6):
        warned = 0
        for _ in range(1000):
            c, m = rng.integers(1, 100, k).astype(float), rng.normal(10, 1, k)
            warned += "field_size_warning" in _field_size_check(m * c, c)
        assert warned / 1000 <= 0.05


def _bootstrap_to_analytic(values_per_fov):
    objects, fovs = _acquisition(values_per_fov)
    data = validate_input_tables(objects, fovs=fovs, feature_cols=["area"], fov_col="position_id")
    boot = FOVBootstrapEstimator("centroid", None, 40_000, 0, 2).estimate(
        data, objective=POPULATION
    )
    sums = np.array([sum(v) for v in values_per_fov])
    counts = np.array([len(v) for v in values_per_fov], dtype=float)
    analytic = fov_averaging_cv(sums, counts)
    # The raw bootstrap SE: the reported one is widened for few FOVs (audit 4, P4-4).
    raw = boot.value("fov_bootstrap_centroid_se") / boot.diagnostics["small_sample_widening"]
    return raw / (analytic["cv_mean"] * analytic["mean"])


def test_c8_the_bootstrap_to_ratio_se_factor_holds_only_when_every_fov_holds_objects():
    rng = np.random.default_rng(5)
    # Equal object counts: the ratio estimator is then linear in the FOV means.
    filled = [list(rng.normal(10 + rng.normal(), 2, 10)) for _ in range(8)]
    assert _bootstrap_to_analytic(filled) == pytest.approx(math.sqrt(7 / 8), rel=0.02)
    half_empty = [values if j % 2 == 0 else [] for j, values in enumerate(filled)]
    # With 4 of 8 FOVs empty the ratio is not sqrt((k' - 1) / k') = 0.87: it is larger.
    assert _bootstrap_to_analytic(half_empty) > math.sqrt(3 / 4) * 1.05
    doc = fov_bootstrap.__doc__
    assert "When every visible FOV holds observations" in _words(doc)
    assert "With empty FOVs it does not" in _words(doc)


def test_c9_the_pilot_plan_is_not_claimed_to_stop_exactly_at_required_fovs():
    assert "reaches the target exactly" not in _words(scalar_precision.__doc__)
    # At a rounding boundary the projection needs one FOV more than required_fovs says.
    assert required_fovs(0.05, 2, 0.01) == 50
    assert 0.05 * math.sqrt(2 / 50) > 0.01 >= 0.05 * math.sqrt(2 / 51)
    assert "one FOV later" in required_fovs.__doc__


def test_c10_numpy_integers_are_accepted_and_stored_as_python_ints():
    strategies = [
        DescriptorPlateauStrategy("area", 0.1, np.int64(3), "acq"),
        DescriptorSpreadStrategy("area", 0.1, "fov", np.int64(100), np.int64(0)),
        FOVBootstrapStrategy("centroid", 0.1, None, np.int32(100), np.int64(0), np.int64(3)),
        FOVBootstrapStrategy("subspace", 0.1, None, 100, 0, 3, None, np.int64(1)),
    ]
    for strategy in strategies:
        text = json.dumps(strategy.parameters)  # provenance is JSON: no numpy integer left
        assert "int64" not in text
        for estimator in strategy.estimators:
            json.dumps(estimator.parameters)
    assert type(strategies[0].window) is int and type(strategies[2].min_fovs) is int
    with pytest.raises(ValueError, match="random_state"):
        DescriptorSpreadStrategy("area", 0.1, "fov", 100, -1)
    with pytest.raises(TypeError, match="n_resamples"):
        DescriptorSpreadStrategy("area", 0.1, "fov", True, 0)


def test_c11_the_fov_unit_rule_names_the_rows_it_averages():
    rng = np.random.default_rng(4)
    objects, fovs = _acquisition([list(np.abs(rng.normal(10, 8, 12))) for _ in range(8)])
    [result] = _run(
        objects,
        fovs,
        [ScalarPrecisionStrategy("area", 0.05, "fov", "point")],
        observation_unit="patch",
    )
    diagnostics = result.evidence.diagnostics
    assert diagnostics["scalar_parameter"] == "mean of 'area' over patches"
    assert diagnostics["warning"].startswith("the SD between patches exceeds 50% CV")
    assert "cell" not in diagnostics["warning"]


def _pilot():
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "condition": np.repeat(["DMSO", "drug"], 40),
            "repeat": "r1",
            "well": np.tile(np.repeat(["A1", "A2"], 20), 2),
            "position_id": np.repeat([f"p{i}" for i in range(8)], 10),
            "area": rng.normal(10, 2, 80),
        }
    )


def test_c12_a_control_column_or_value_the_pilot_lacks_is_named():
    def plan(control):
        return plan_objects_to_average(
            _pilot(),
            feature="area",
            target_cv=0.05,
            sd_estimate="point",
            sample_cols=["condition", "repeat", "well"],
            control=control,
            fov_col="position_id",
        )

    with pytest.raises(ValueError, match=r"control columns \['treatment'\] are not columns"):
        plan({"treatment": "DMSO"})
    with pytest.raises(ValueError, match="the pilot holds condition: 'DMSO' and 'drug'"):
        plan({"condition": "dmso"})
    assert plan({"condition": "DMSO"}).n_control_samples == 2


def test_c13_the_spreadsheet_sd_is_kept_and_its_small_pilot_bias_is_documented():
    pilot = _pilot()
    plan = plan_objects_to_average(
        pilot,
        feature="area",
        target_cv=0.05,
        sd_estimate="point",
        sample_cols=["condition", "repeat", "well"],
        control={"condition": "DMSO"},
        fov_col="position_id",
    )
    keys = ["condition", "repeat", "well"]
    centred = pilot["area"] - pilot.groupby(keys)["area"].transform("mean")
    control_mean = pilot.loc[pilot["condition"] == "DMSO", "area"].mean()
    # ddof=1 over all objects after centring, as the paper's spreadsheet computes it.
    assert plan.cv == pytest.approx(np.std(centred, ddof=1) / control_mean)
    doc = jcb_planning.__doc__
    assert "(n_objects - n_samples) / (n_objects - 1)" in _words(doc)
    assert "3 of 3, 76% and 90%" in _words(doc)


def test_c14_marcotti_rules_answer_population_and_distribution_with_two_quantities():
    rng = np.random.default_rng(1)
    objects, fovs = _acquisition([list(np.abs(rng.normal(10, 2, 15))) for _ in range(12)])
    rules = [
        DescriptorPlateauStrategy("area", 5.0, 2, "acq"),
        DescriptorSpreadStrategy("area", 5.0, "fov", 50, 0),
        DescriptorSpreadStrategy("area", 5.0, "object", 50, 0),
    ]
    for objective in (POPULATION, DISTRIBUTION, PHENOTYPE):
        for result in _run(objects, fovs, rules, objective=objective):
            assert result.decision in ("stop", "continue"), (objective, result.strategy)
    [plateau, spread, _] = _run(objects, fovs, rules)
    for result, measure in ((plateau, "range"), (spread, "spread")):
        quantities = result.evidence.quantities
        assert quantities[f"max_location_{measure}"].component == "population"
        assert quantities[f"max_scale_{measure}"].component == "distribution"
        # The earlier quantity keeps its meaning: the largest over every descriptor.
        combined = quantities[f"max_descriptor_{measure}"]
        assert combined.component == "distribution"
        assert combined.value == max(
            quantities[f"max_location_{measure}"].value, quantities[f"max_scale_{measure}"].value
        )
    # The rule's own verdict is unchanged: stop when every descriptor is within tolerance.
    for tolerance in (0.01, 0.2, 0.5, 1.0, 5.0):
        rule = DescriptorSpreadStrategy("area", tolerance, "fov", 50, 0)
        [result] = _run(objects, fovs, [rule])
        expected = result.evidence.value("max_descriptor_spread") <= tolerance
        assert (result.rule_decision == "stop") == expected
    # Location descriptors alone answer the population summary only, as before.
    location = DescriptorSpreadStrategy("area", 5.0, "fov", 50, 0, ("mean", "median"))
    [result] = _run(objects, fovs, [location], objective=DISTRIBUTION)
    assert result.decision == "unsupported"
    assert set(result.evidence.quantities) == {"max_descriptor_spread", "max_location_spread"}


def test_c15_degenerate_pca_resamples_are_not_counted_again_as_undefined():
    # Two one-object FOVs: no resample is empty, and half repeat one FOV (no spread).
    objects = pd.DataFrame({"position_id": ["p0", "p1"], "a": [1.0, 2.0], "b": [0.5, 0.1]})
    fovs = pd.DataFrame({"position_id": ["p0", "p1"]})
    data = validate_input_tables(objects, fovs=fovs, feature_cols=["a", "b"], fov_col="position_id")
    estimator = FOVBootstrapEstimator("eigenspectrum", None, 2000, 0, 2)
    diagnostics = estimator.estimate(data, objective=DISTRIBUTION).diagnostics
    assert diagnostics["degenerate_resamples"] > 800
    assert diagnostics["undefined_resamples"] == 0


def test_c16_the_plateau_window_counts_fovs_with_objects_and_missing_values_are_stated():
    rng = np.random.default_rng(1)
    values = [list(rng.normal(10, 2, 8)) if j % 2 == 0 else [] for j in range(10)]
    objects, fovs = _acquisition(values)
    [result] = _run(objects, fovs, [DescriptorPlateauStrategy("area", 100.0, 2, "acq")])
    assert result.evidence.diagnostics["fovs_in_window"] == [5, 7, 9]
    assert "``window`` counts FOVs that added objects" in _words(
        stabilization.DescriptorPlateauStrategy.__doc__
    )
    objects.loc[objects.index[3], "area"] = np.nan
    rules = [
        DescriptorPlateauStrategy("area", 1.0, 2, "acq"),
        DescriptorSpreadStrategy("area", 1.0, "fov", 20, 0),
    ]
    for result, estimator in zip(
        _run(objects, fovs, rules), (r.estimators[0] for r in rules), strict=True
    ):
        assert all(q.availability == "not_applicable" for q in result.evidence.quantities.values())
        assert "Missing values in the feature" in _words(type(estimator).__doc__)


def test_c17_the_paper_descriptors_are_computed_once_in_statistics():
    assert stabilization.describe is statistics.describe
    assert object_rules.describe is statistics.describe
    assert stabilization.PAPER_DESCRIPTORS is statistics.PAPER_DESCRIPTORS


def test_j12_j16_modules_declare_their_public_and_shared_private_names():
    for module in (
        base,
        baselines,
        scalar_precision,
        stabilization,
        fov_bootstrap,
        jcb_planning,
        _averaging_cv,
    ):
        assert module.__all__, module.__name__
        for name in module.__all__:
            assert not name.startswith("_") and hasattr(module, name), (module.__name__, name)
    assert "Shared-private: ``_field_size_check``" in _averaging_cv.__doc__
    assert "Shared-private: ``_as_ints``" in base.__doc__
