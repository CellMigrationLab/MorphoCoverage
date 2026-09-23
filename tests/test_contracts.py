"""Contract tests for the AGENTS.md §9 scenario list.

Three layers:

1. The §9 list and this module stay in sync: every scenario id in AGENTS.md has
   a fixture or a named framework contract, and vice versa.
2. Each fixture actually has the property it claims, so the contracts below are
   not tested against hollow data.
3. The framework and every strategy in ``STRATEGIES_UNDER_CONTRACT`` satisfy the
   contracts on every scenario.

These tests check correct handling, not scientific performance. Whether a
strategy stops at the right time on these data is a benchmarking question.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scenarios import FEATURE_COLS, ORDER_COL, SCENARIOS

from morphocoverage import (
    Decision,
    DescriptorPlateauStrategy,
    DescriptorSpreadStrategy,
    Estimator,
    Evidence,
    FixedFOVStrategy,
    FixedObjectStrategy,
    ObjectiveConfig,
    RareTarget,
    SamplingObjective,
    SamplingStrategy,
    ScalarPrecisionStrategy,
    evaluate_acquisition_prefixes,
    outcomes,
    run_strategies,
    transforms,
    validate_input_tables,
)
from morphocoverage.benchmark import BenchmarkDataset, BenchmarkRegistry, run_benchmark
from morphocoverage.evaluator import iter_analysis_groups, objects_in_fovs, restrict_input

POPULATION = ObjectiveConfig(SamplingObjective.POPULATION)
_RARE = RareTarget(
    object_prevalence=0.01,
    detection_probability=0.95,
    min_fov_occurrences=2,
    phenotype="true_state column",
)
OBJECTIVES = [
    POPULATION,
    ObjectiveConfig(SamplingObjective.RARE_DISCOVERY, rare=_RARE),
    ObjectiveConfig(SamplingObjective.BALANCED, rare=_RARE),
]

# Every strategy added to the package must be added here, configured with
# parameters that make it run on small data.
STRATEGIES_UNDER_CONTRACT: list[SamplingStrategy] = [
    FixedFOVStrategy(5),
    FixedObjectStrategy(100),
    ScalarPrecisionStrategy("feature_a", 0.05, "object", "point"),
    ScalarPrecisionStrategy("feature_a", 0.05, "fov", "point"),
    ScalarPrecisionStrategy(
        "feature_a", 0.05, "object", "upper_95", pilot_fovs=3, order_col=ORDER_COL
    ),
    DescriptorPlateauStrategy("feature_a", 0.1, 3, ORDER_COL),
    DescriptorSpreadStrategy("feature_a", 0.3, "object", 20, 0),
    DescriptorSpreadStrategy("feature_a", 0.3, "fov", 20, 0),
]

# Every outcome metric added to the package must be added here.
METRICS_UNDER_CONTRACT: list[outcomes.OutcomeMetric] = [
    outcomes.RelativeMeanError("feature_a"),
    outcomes.RandomCellRelativeError("feature_a"),
    outcomes.RareStateDetected("true_state", 1),
]
METRIC_IDS = [metric.name for metric in METRICS_UNDER_CONTRACT]

# Every feature transform added to the package must be added here.
TRANSFORMS_UNDER_CONTRACT: list[transforms.FeatureTransform] = [
    transforms.Standardize(),
    transforms.PCA(1),
    transforms.Chain([transforms.Standardize(), transforms.PCA(2)]),
]
TRANSFORM_IDS = [transform.name for transform in TRANSFORMS_UNDER_CONTRACT]

# §9 items that are properties of the framework rather than datasets.
FRAMEWORK_CONTRACTS = {
    "baseline_comparison": "test_fixed_cell_and_fixed_fov_baselines_are_distinguishable",
    "deterministic_under_seed": "test_strategy_results_are_deterministic",
    "no_future_fov_leakage": "test_replay_exposes_no_future_fovs",
}

SCENARIO_IDS = sorted(SCENARIOS)
STRATEGY_IDS = [strategy.name for strategy in STRATEGIES_UNDER_CONTRACT]


def _replay(scenario, strategies, objective=POPULATION, **kwargs):
    return evaluate_acquisition_prefixes(
        scenario.objects,
        scenario.fovs,
        strategies=strategies,
        feature_cols=list(FEATURE_COLS),
        acquisition_order_col=ORDER_COL,
        objective=objective,
        **kwargs,
    )


def _without_runtime(results):
    return [replace(item, result=replace(item.result, compute_seconds=None)) for item in results]


class _ProbeEstimator(Estimator):
    """Records everything an estimator could observe about its input."""

    name = "probe"

    def estimate(self, data, *, objective):
        key = list(data.fov_key)
        levels: set[object] = set()
        for table in (data.objects, data.fovs):
            for column in table.columns:
                if isinstance(table[column].dtype, pd.CategoricalDtype):
                    levels |= set(table[column].cat.categories)
        return Evidence(
            diagnostics={
                "visible_fovs": list(data.fovs.loc[:, key].itertuples(index=False, name=None)),
                "objects_per_fov": Counter(
                    data.objects.loc[:, key].itertuples(index=False, name=None)
                ),
                "category_levels": levels,
                "feature_matrix": data.feature_matrix(),
                "fov_columns": list(data.fovs.columns),
            },
        )


def _probe():
    return SamplingStrategy([_ProbeEstimator()])


# --- 1. The §9 list and the tests stay in sync ---------------------------------


def _agents_section_9_ids() -> set[str]:
    text = (Path(__file__).resolve().parents[1] / "AGENTS.md").read_text()
    section = re.search(r"^## 9\..*?(?=^## 10\.)", text, flags=re.DOTALL | re.MULTILINE)
    assert section, "AGENTS.md §9 not found"
    items = [line for line in section.group(0).splitlines() if line.startswith("- ")]
    return {match for line in items for match in re.findall(r"`([a-z0-9_]+)`", line)}


def test_agents_section_9_matches_contract_tests():
    listed = _agents_section_9_ids()
    implemented = set(SCENARIOS) | set(FRAMEWORK_CONTRACTS)
    assert listed == implemented, (
        f"only in AGENTS.md: {sorted(listed - implemented)}; "
        f"only in tests: {sorted(implemented - listed)}"
    )


def test_every_package_strategy_is_under_contract():
    # AGENTS.md section 9: a strategy added to the package must be added to
    # STRATEGIES_UNDER_CONTRACT, and so must every exported estimator and
    # policy, through some strategy. Enforced here rather than by convention.
    import morphocoverage
    from morphocoverage import strategies

    def exported(module, base):
        return {
            obj
            for obj in vars(module).values()
            if isinstance(obj, type) and issubclass(obj, base) and obj is not base
        }

    covered = {type(strategy) for strategy in STRATEGIES_UNDER_CONTRACT}
    for strategy in STRATEGIES_UNDER_CONTRACT:
        covered |= {type(part) for part in (*strategy.estimators, *strategy.policies)}
    required = (
        exported(strategies, SamplingStrategy)
        | exported(strategies, Estimator)
        | exported(morphocoverage, morphocoverage.DecisionPolicy)
    )
    missing = sorted(cls.__name__ for cls in required - covered)
    assert not missing, f"missing from STRATEGIES_UNDER_CONTRACT: {missing}"


def test_framework_contracts_name_existing_tests():
    for test_name in FRAMEWORK_CONTRACTS.values():
        assert callable(globals().get(test_name)), test_name


# --- 2. Fixtures have the properties they claim --------------------------------


@pytest.mark.parametrize("name", SCENARIO_IDS)
def test_scenario_builders_are_deterministic_and_seed_sensitive(name):
    first, second, other = SCENARIOS[name](), SCENARIOS[name](), SCENARIOS[name](seed=1)
    pd.testing.assert_frame_equal(first.objects, second.objects)
    pd.testing.assert_frame_equal(first.fovs, second.fovs)
    assert not np.allclose(
        first.objects.loc[:, FEATURE_COLS].to_numpy(),
        other.objects.loc[:, FEATURE_COLS].to_numpy(),
    )


@pytest.mark.parametrize(
    "name", ["rare_prevalence_10pct", "rare_prevalence_1pct", "rare_prevalence_0p1pct"]
)
def test_rare_prevalence_is_exact(name):
    scenario = SCENARIOS[name]()
    assert scenario.objects["true_state"].mean() == scenario.truth["object_prevalence"]


def test_uniform_and_clustered_rare_states_differ_only_in_fov_incidence():
    # Same cell prevalence, 10x different FOV incidence: the distinction the
    # rare-discovery objective has to be able to express.
    uniform, clustered = SCENARIOS["rare_uniform"](), SCENARIOS["rare_clustered"]()
    for scenario in (uniform, clustered):
        objects = scenario.objects
        assert objects["true_state"].mean() == 0.01
        rare_fovs = objects.loc[objects["true_state"].eq(1), "image_file"].nunique()
        assert rare_fovs / len(scenario.fovs) == scenario.truth["fov_incidence"]
    assert (uniform.truth["fov_incidence"], clustered.truth["fov_incidence"]) == (0.5, 0.05)
    assert clustered.truth["first_rare_step"] == 31


def test_multimodal_modes_are_well_separated_within_every_fov():
    objects = SCENARIOS["multimodal"]().objects
    per_fov = objects.groupby(["image_file", "true_state"]).size().unstack()
    assert (per_fov == 15).all().all()
    means = objects.groupby("true_state")[list(FEATURE_COLS)].mean()
    assert np.linalg.norm(means.loc[1] - means.loc[0]) > 10


def test_neighbouring_fovs_are_more_similar_than_random_pairs():
    scenario = SCENARIOS["neighbour_correlation"]()
    fov_means = scenario.objects.groupby("image_file", sort=True)["feature_a"].mean().to_numpy()
    adjacent = np.abs(np.diff(fov_means.reshape(4, 10), axis=1)).mean()
    shuffled = np.random.default_rng(0).permutation(fov_means)
    random_pairs = np.abs(np.diff(shuffled)).mean()
    assert adjacent < random_pairs / 2


def test_batch_shift_is_present_between_repeats():
    scenario = SCENARIOS["batch_shift"]()
    means = scenario.objects.groupby("repeat")["feature_a"].mean()
    assert abs((means["R2"] - means["R1"]) - scenario.truth["shift"]) < 0.5


def test_unequal_scenarios_actually_vary():
    counts = SCENARIOS["unequal_objects_per_fov"]().objects.groupby("image_file").size()
    assert (counts.min(), counts.max()) == (5, 200)
    area = SCENARIOS["unequal_fov_area"]().fovs["area_um2"]
    assert area.nunique() == 2


# --- 3. Framework and strategy contracts on every scenario ---------------------


@pytest.mark.parametrize("strategy", STRATEGIES_UNDER_CONTRACT, ids=STRATEGY_IDS)
@pytest.mark.parametrize("name", SCENARIO_IDS)
def test_counts_match_the_acquisition_record(name, strategy):
    scenario = SCENARIOS[name]()
    results = run_strategies(
        scenario.objects,
        fovs=scenario.fovs,
        strategies=[strategy],
        feature_cols=list(FEATURE_COLS),
        objective=POPULATION,
    )
    expected_fovs = scenario.fovs.groupby(["condition", "repeat"]).size()
    expected_objects = scenario.objects.groupby(["condition", "repeat"]).size()
    assert len(results) == len(expected_fovs)
    for result in results:
        group = (result.group["condition"], result.group["repeat"])
        assert result.n_fovs == expected_fovs[group]
        assert result.n_objects == expected_objects.get(group, 0)


@pytest.mark.parametrize("name", SCENARIO_IDS)
def test_replay_reveals_whole_fovs_in_acquisition_order(name):
    scenario = SCENARIOS[name]()
    key = ["condition", "repeat", "image_file"]
    all_objects_per_fov = Counter(
        scenario.objects.loc[:, key].itertuples(index=False, name=None)
    )
    for item in _replay(scenario, [_probe()]):
        group = item.result.group
        ordered = (
            scenario.fovs[
                scenario.fovs["condition"].eq(group["condition"])
                & scenario.fovs["repeat"].eq(group["repeat"])
            ]
            .sort_values(ORDER_COL)
            .loc[:, key]
        )
        expected = list(ordered.iloc[: item.step].itertuples(index=False, name=None))
        diagnostics = item.result.diagnostics
        assert diagnostics["visible_fovs"] == expected
        # Every object of every revealed FOV, and nothing else: FOVs are never
        # partially revealed.
        assert diagnostics["objects_per_fov"] == Counter(
            {fov: all_objects_per_fov[fov] for fov in expected if all_objects_per_fov[fov]}
        )
        assert item.result.n_fovs == item.step


@pytest.mark.parametrize("name", SCENARIO_IDS)
def test_replay_exposes_no_future_fovs(name):
    scenario = SCENARIOS[name]()
    # Categorical identifiers are the easiest way for withheld FOVs to leak.
    objects, fovs = scenario.objects.copy(), scenario.fovs.copy()
    levels = sorted(fovs["image_file"].unique())
    objects["image_file"] = pd.Categorical(objects["image_file"], categories=levels)
    fovs["image_file"] = pd.Categorical(fovs["image_file"], categories=levels)
    for item in _replay(replace(scenario, objects=objects, fovs=fovs), [_probe()]):
        visible_names = {fov[-1] for fov in item.acquired_fovs}
        diagnostics = item.result.diagnostics
        assert {fov[-1] for fov in diagnostics["visible_fovs"]} == visible_names
        assert {fov[-1] for fov in diagnostics["objects_per_fov"]} <= visible_names
        assert diagnostics["category_levels"] <= visible_names | {
            item.result.group["condition"],
            item.result.group["repeat"],
        }


@pytest.mark.parametrize("name", SCENARIO_IDS)
def test_only_explicit_features_reach_the_strategy(name):
    scenario = SCENARIOS[name]()
    [result] = run_strategies(
        scenario.objects,
        fovs=scenario.fovs,
        strategies=[_probe()],
        feature_cols=list(FEATURE_COLS),
        objective=POPULATION,
        group_cols=[],
    )
    matrix = result.diagnostics["feature_matrix"]
    assert matrix.shape == (len(scenario.objects), len(FEATURE_COLS))
    # true_state, label_id and stage coordinates are numeric but are not phenotype.
    np.testing.assert_array_equal(matrix, scenario.objects.loc[:, FEATURE_COLS].to_numpy())


def test_zero_object_fovs_count_as_acquisition_effort():
    scenario = SCENARIOS["zero_object_fovs"]()
    results = _replay(scenario, [_probe()])
    assert [item.result.n_fovs for item in results] == list(range(1, 21))
    assert results[0].result.n_objects == 0  # the first acquired FOV is empty
    assert results[-1].result.n_objects == len(scenario.objects)
    last_visible = {fov[-1] for fov in results[-1].result.diagnostics["visible_fovs"]}
    assert set(scenario.truth["empty_fovs"]) <= last_visible


def test_fov_design_metadata_is_available_to_strategies():
    scenario = SCENARIOS["unequal_fov_area"]()
    for item in _replay(scenario, [_probe()]):
        assert {"area_um2", "stage_x", "stage_y"} <= set(item.result.diagnostics["fov_columns"])


def test_duplicated_objects_add_objects_but_not_fovs():
    # Re-measuring the same FOV is not new acquisition: counts keep the two apart.
    scenario = SCENARIOS["duplicated_objects"]()
    first = _replay(scenario, [_probe()])[0].result
    assert (first.n_fovs, first.n_objects) == (1, 60)


@pytest.mark.parametrize("strategy", STRATEGIES_UNDER_CONTRACT, ids=STRATEGY_IDS)
@pytest.mark.parametrize("name", ["condition_repeat_groups", "batch_shift"])
def test_groups_are_independent(name, strategy):
    scenario = SCENARIOS[name]()
    target = scenario.objects["condition"].eq("control") & scenario.objects["repeat"].eq("R1")
    perturbed = scenario.objects.copy()
    perturbed.loc[~target, list(FEATURE_COLS)] *= -100.0
    perturbed = perturbed[target | (perturbed.index % 2 == 0)]  # also drop other groups' objects

    def control_r1(objects):
        return [
            item
            for item in _without_runtime(_replay(replace(scenario, objects=objects), [strategy]))
            if item.result.group == {"condition": "control", "repeat": "R1"}
        ]

    assert control_r1(scenario.objects) == control_r1(perturbed)


def test_reused_image_names_stay_separate_groups_and_fovs():
    scenario = SCENARIOS["condition_repeat_groups"]()
    results = run_strategies(
        scenario.objects,
        fovs=scenario.fovs,
        strategies=[FixedFOVStrategy(1)],
        feature_cols=list(FEATURE_COLS),
        objective=POPULATION,
    )
    assert len(results) == scenario.truth["n_groups"]
    assert {r.n_fovs for r in results} == {scenario.truth["fovs_per_group"]}
    [pooled] = run_strategies(
        scenario.objects,
        fovs=scenario.fovs,
        strategies=[FixedFOVStrategy(1)],
        feature_cols=list(FEATURE_COLS),
        objective=POPULATION,
        group_cols=[],
    )
    assert pooled.n_fovs == scenario.truth["n_groups"] * scenario.truth["fovs_per_group"]


@pytest.mark.parametrize("strategy", STRATEGIES_UNDER_CONTRACT, ids=STRATEGY_IDS)
@pytest.mark.parametrize("name", SCENARIO_IDS)
def test_strategy_results_are_deterministic(name, strategy):
    scenario = SCENARIOS[name]()
    assert _without_runtime(_replay(scenario, [strategy])) == _without_runtime(
        _replay(SCENARIOS[name](), [strategy])
    )


@pytest.mark.parametrize("objective", OBJECTIVES, ids=lambda o: o.objective.value)
@pytest.mark.parametrize("strategy", STRATEGIES_UNDER_CONTRACT, ids=STRATEGY_IDS)
@pytest.mark.parametrize("name", ["rare_clustered", "zero_object_fovs"])
def test_decisions_answer_only_the_components_the_objective_requires(name, strategy, objective):
    # A strategy never stops for a component it has no evidence about: a
    # missing required component is UNSUPPORTED, and STOP needs every required
    # component to stop (balanced needs both population and rare).
    scenario = SCENARIOS[name]()
    for item in _replay(scenario, [strategy], objective=objective):
        result = item.result
        required = [c.value for c in objective.required_components]
        assert set(required) <= set(result.component_decisions)
        decisions = [result.component_decisions[c] for c in required]
        if Decision.UNSUPPORTED in decisions:
            assert result.decision is Decision.UNSUPPORTED
        elif None in decisions:
            assert result.decision is None
        else:
            stop = all(d is Decision.STOP for d in decisions)
            assert result.decision is (Decision.STOP if stop else Decision.CONTINUE)


def test_fixed_cell_and_fixed_fov_baselines_are_distinguishable():
    # With 5-200 objects per FOV, a cell-count rule and an FOV-count rule stop at
    # different points, and the result reports both kinds of effort so the
    # difference is visible to the benchmark. The JCB-style scalar-precision and
    # Marcotti-style descriptor-stabilization baselines run through the same
    # replay and report the same effort fields.
    scenario = SCENARIOS["unequal_objects_per_fov"]()
    precision_cells = ScalarPrecisionStrategy("feature_a", 0.005, "object", "point")
    precision_fovs = ScalarPrecisionStrategy("feature_a", 0.005, "fov", "point")
    plateau = DescriptorPlateauStrategy("feature_a", 0.1, 3, ORDER_COL)
    spread = DescriptorSpreadStrategy("feature_a", 0.5, "object", 50, 0)
    results = _replay(
        scenario,
        [
            FixedFOVStrategy(10),
            FixedObjectStrategy(500),
            precision_cells,
            precision_fovs,
            plateau,
            spread,
        ],
    )

    def first_stop(strategy_name):
        return next(
            item.result
            for item in results
            if item.result.strategy == strategy_name and item.result.decision == "stop"
        )

    # FOVs 0, 5, 10, ... hold 200 objects; all others hold 5.
    by_fov, by_cells = first_stop("fixed_fov"), first_stop("fixed_objects")
    assert (by_fov.n_fovs, by_fov.n_objects) == (10, 2 * 200 + 8 * 5)
    assert (by_cells.n_fovs, by_cells.n_objects) == (11, 3 * 200 + 8 * 5)

    for name in (precision_cells.name, precision_fovs.name):
        stop = first_stop(name)
        assert stop.n_fovs >= 1 and stop.n_objects >= 1
        assert stop.parameters["target_cv"] == 0.005
        assert stop.evidence.value("averaging_cv") <= 0.005

    for strategy, quantity in (
        (plateau, "max_descriptor_range"),
        (spread, "max_descriptor_spread"),
    ):
        stop = first_stop(strategy.name)
        assert stop.n_fovs >= 1 and stop.n_objects >= 1
        assert stop.evidence.value(quantity) <= strategy.tolerance


# --- 4. Outcome metric contracts -----------------------------------------------


def test_every_package_outcome_metric_is_under_contract():
    exported = {
        obj
        for obj in vars(outcomes).values()
        if isinstance(obj, type)
        and issubclass(obj, outcomes.OutcomeMetric)
        and obj is not outcomes.OutcomeMetric
    }
    covered = {type(metric) for metric in METRICS_UNDER_CONTRACT}
    missing = sorted(cls.__name__ for cls in exported - covered)
    assert not missing, f"outcome metrics missing from METRICS_UNDER_CONTRACT: {missing}"


def _group_prefixes(scenario):
    data = validate_input_tables(
        scenario.objects, fovs=scenario.fovs, feature_cols=list(FEATURE_COLS)
    )
    for _, full in iter_analysis_groups(data):
        ordered = full.fovs.sort_values(ORDER_COL)
        for k in sorted({1, 2, len(ordered) // 2, len(ordered)}):
            fovs = ordered.iloc[:k]
            yield full, restrict_input(full, objects_in_fovs(full, fovs), fovs)


@pytest.mark.parametrize("metric", METRICS_UNDER_CONTRACT, ids=METRIC_IDS)
@pytest.mark.parametrize("name", SCENARIO_IDS)
def test_outcome_metrics_are_finite_deterministic_and_leave_inputs_alone(name, metric):
    # Outcomes are compared across strategies, so a NaN or an input edited in
    # place would corrupt every later comparison.
    for full, stopped in _group_prefixes(SCENARIOS[name]()):
        before = (full.objects.copy(), stopped.objects.copy())
        value = metric.evaluate(full, stopped)
        assert value is None or (isinstance(value, float) and math.isfinite(value))
        assert metric.evaluate(full, stopped) == value
        pd.testing.assert_frame_equal(full.objects, before[0])
        pd.testing.assert_frame_equal(stopped.objects, before[1])


def test_outcomes_depend_on_the_stop_not_on_the_strategy():
    # Two different rules that stop at the same prefix get identical outcomes.
    scenario = SCENARIOS["homogeneous"]()
    dataset = BenchmarkDataset(
        "homogeneous",
        "development",
        lambda: (scenario.objects, scenario.fovs),
        FEATURE_COLS,
        ORDER_COL,
    )
    per_fov = scenario.objects.groupby("image_file").size().iloc[0]
    by_fovs, by_objects = run_benchmark(
        BenchmarkRegistry([dataset]),
        "homogeneous",
        strategies=[FixedFOVStrategy(4), FixedObjectStrategy(int(4 * per_fov))],
        metrics=METRICS_UNDER_CONTRACT[:2],
        objective=POPULATION,
    )
    assert by_fovs.stop_step == by_objects.stop_step == 4
    assert by_fovs.config_id != by_objects.config_id
    assert by_fovs.outcomes == by_objects.outcomes


# --- 5. Feature transform contracts --------------------------------------------


def test_every_package_transform_is_under_contract():
    exported = {
        obj
        for obj in vars(transforms).values()
        if isinstance(obj, type)
        and issubclass(obj, transforms.FeatureTransform)
        and obj is not transforms.FeatureTransform
    }
    covered = {type(transform) for transform in TRANSFORMS_UNDER_CONTRACT}
    for transform in TRANSFORMS_UNDER_CONTRACT:
        if isinstance(transform, transforms.Chain):
            covered |= {type(step) for step in transform.transforms}
    missing = sorted(cls.__name__ for cls in exported - covered)
    assert not missing, f"transforms missing from TRANSFORMS_UNDER_CONTRACT: {missing}"


class _TransformProbe(Estimator):
    """Fits a transform on what it is given and reports the fit and the output."""

    name = "transform_probe"

    def __init__(self, transform):
        self.transform = transform

    def estimate(self, data, *, objective):
        fitted = self.transform.fit(data)
        output = fitted.apply(data).tolist() if fitted.fitted else None
        return Evidence(diagnostics={"transform": fitted.provenance(), "output": output})


@pytest.mark.parametrize("transform", TRANSFORMS_UNDER_CONTRACT, ids=TRANSFORM_IDS)
@pytest.mark.parametrize("name", SCENARIO_IDS)
def test_transforms_are_unaffected_by_future_fovs(name, transform):
    # AGENTS.md section 3: preprocessing for prospective evaluation must not
    # use future FOVs. Altering every FOV after step k must leave the fitted
    # state and the transformed features at steps <= k unchanged.
    scenario = SCENARIOS[name]()
    probe = [SamplingStrategy([_TransformProbe(transform)])]
    original = _replay(scenario, probe)
    order = scenario.fovs.set_index(["condition", "repeat", "image_file"])[ORDER_COL]
    cutoff = int(order.median())
    objects = scenario.objects.copy()
    key = pd.MultiIndex.from_frame(objects[["condition", "repeat", "image_file"]])
    future = order.reindex(key).to_numpy() > cutoff
    objects.loc[future, list(FEATURE_COLS)] = objects.loc[future, list(FEATURE_COLS)] * 7.0 + 50.0
    altered = _replay(replace(scenario, objects=objects), probe)
    assert len(original) == len(altered)
    compared = 0
    for before, after in zip(original, altered, strict=True):
        assert before.acquired_fovs == after.acquired_fovs
        latest = order[before.acquired_fovs[-1]]
        if latest <= cutoff:
            assert before.result.diagnostics == after.result.diagnostics
            compared += 1
    assert compared > 0
    if future.any():
        # The alteration is real: the full-acquisition fits differ.
        assert original[-1].result.diagnostics != altered[-1].result.diagnostics


@pytest.mark.parametrize("transform", TRANSFORMS_UNDER_CONTRACT, ids=TRANSFORM_IDS)
def test_fitted_transforms_are_json_provenance(transform):
    import json

    scenario = SCENARIOS["multimodal"]()
    for item in _replay(scenario, [SamplingStrategy([_TransformProbe(transform)])]):
        record = item.result.diagnostics["transform"]
        assert json.loads(json.dumps(record)) == record
        assert record["config_id"] == transform.config_id
        assert record["n_fit_fovs"] == item.step
