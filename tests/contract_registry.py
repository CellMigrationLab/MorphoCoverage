"""What the contract tests run on: every packaged strategy, metric and transform.

Adding a strategy, outcome metric or feature transform to the package means
adding it here (AGENTS.md section 9); coverage tests fail otherwise. The
contract tests themselves are in ``test_contracts*.py``.
"""

from dataclasses import replace

from scenarios import FEATURE_COLS, ORDER_COL, SCENARIOS

from morphocoverage import (
    DescriptorPlateauStrategy,
    DescriptorSpreadStrategy,
    FixedFOVStrategy,
    FixedObjectStrategy,
    FixedObservationStrategy,
    ObjectiveConfig,
    PlannedCellCountStrategy,
    RareTarget,
    SamplingObjective,
    SamplingStrategy,
    ScalarPrecisionStrategy,
    evaluate_acquisition_prefixes,
    multivariate_outcomes,
    outcomes,
    plan_cells_to_average,
    transforms,
)
from morphocoverage.strategies import FOVBootstrapStrategy

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


def _pilot_plan():
    """A JCB cell plan from a pilot repeat that no scenario evaluates."""
    pilot = SCENARIOS["homogeneous"]()
    return plan_cells_to_average(
        pilot.objects.assign(repeat="PILOT"),
        fovs=pilot.fovs.assign(repeat="PILOT"),
        feature="feature_a",
        target_cv=0.01,
        sd_estimate="upper_95",
        sample_cols=("condition", "repeat"),
        control={"condition": "control"},
    )


# Every strategy added to the package must be added here, configured with
# parameters that make it run on small data.
STRATEGIES_UNDER_CONTRACT: list[SamplingStrategy] = [
    FixedFOVStrategy(5),
    FixedObjectStrategy(100),
    FixedObservationStrategy(100),
    ScalarPrecisionStrategy("feature_a", 0.05, "object", "point"),
    ScalarPrecisionStrategy("feature_a", 0.05, "fov", "point"),
    ScalarPrecisionStrategy(
        "feature_a", 0.05, "object", "upper_95", pilot_fovs=3, order_col=ORDER_COL
    ),
    DescriptorPlateauStrategy("feature_a", 0.1, 3, ORDER_COL),
    DescriptorSpreadStrategy("feature_a", 0.3, "object", 20, 0),
    DescriptorSpreadStrategy("feature_a", 0.3, "fov", 20, 0),
    PlannedCellCountStrategy(_pilot_plan()),
    FOVBootstrapStrategy("centroid", 0.05, transforms.Standardize(), 30, 0, 3),
    FOVBootstrapStrategy("quantiles", 0.1, None, 20, 0, 3, (0.1, 0.5, 0.9)),
]

# Every outcome metric added to the package must be added here.
METRICS_UNDER_CONTRACT: list[outcomes.OutcomeMetric] = [
    outcomes.RelativeMeanError("feature_a"),
    outcomes.RandomCellRelativeError("feature_a"),
    outcomes.RareStateDetected("true_state", 1),
    multivariate_outcomes.CentroidError(FEATURE_COLS, True),
    multivariate_outcomes.CovarianceError(FEATURE_COLS, True),
    multivariate_outcomes.QuantileError(FEATURE_COLS, True, (0.05, 0.5, 0.95)),
    multivariate_outcomes.FrechetError(FEATURE_COLS, False),
    multivariate_outcomes.EnergyError(FEATURE_COLS, True, 200, 0),
    multivariate_outcomes.MMDError(FEATURE_COLS, True, 200, 0, 1.0),
    multivariate_outcomes.SlicedWassersteinError(FEATURE_COLS, True, 20, 0),
]
METRIC_IDS = [metric.name for metric in METRICS_UNDER_CONTRACT]

# Every feature transform added to the package must be added here.
TRANSFORMS_UNDER_CONTRACT: list[transforms.FeatureTransform] = [
    transforms.Standardize(),
    transforms.PCA(1),
    transforms.Chain([transforms.Standardize(), transforms.PCA(2)]),
]
TRANSFORM_IDS = [transform.name for transform in TRANSFORMS_UNDER_CONTRACT]

SCENARIO_IDS = sorted(SCENARIOS)
STRATEGY_IDS = [strategy.name for strategy in STRATEGIES_UNDER_CONTRACT]


def replay(scenario, strategies, objective=POPULATION, **kwargs):
    """Replay a scenario in its recorded acquisition order."""
    return evaluate_acquisition_prefixes(
        scenario.objects,
        scenario.fovs,
        strategies=strategies,
        feature_cols=list(FEATURE_COLS),
        acquisition_order_col=ORDER_COL,
        objective=objective,
        **kwargs,
    )


def without_timing(results):
    """Replay results with time measurements removed, for exact comparison."""
    return [replace(item, result=item.result.without_timing()) for item in results]
