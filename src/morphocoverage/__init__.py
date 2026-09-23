"""MorphoCoverage: FOV-aware phenotypic sampling analysis."""

from .evidence import Availability, Component, Direction, Evidence, Quantity
from .framework import run_strategies
from .objectives import ObjectiveConfig, RareTarget, SamplingObjective
from .online import OnlineSession
from .policies import DecisionPolicy, Effort, EffortPolicy, ThresholdPolicy, redecide
from .results import Decision, StrategyResult
from .schema import ValidatedInput, validate_input_tables
from .sequential import SequentialResult, evaluate_acquisition_prefixes
from .strategies import (
    CellCountPlan,
    DescriptorPlateauEstimator,
    DescriptorPlateauStrategy,
    DescriptorSpreadEstimator,
    DescriptorSpreadStrategy,
    Estimator,
    FixedFOVStrategy,
    FixedObjectStrategy,
    FixedObservationStrategy,
    PlannedCellCountEstimator,
    PlannedCellCountStrategy,
    SamplingStrategy,
    ScalarPrecisionEstimator,
    ScalarPrecisionStrategy,
    plan_cells_to_average,
)

__all__ = [
    "Availability",
    "CellCountPlan",
    "Component",
    "Decision",
    "DecisionPolicy",
    "DescriptorPlateauEstimator",
    "DescriptorPlateauStrategy",
    "DescriptorSpreadEstimator",
    "DescriptorSpreadStrategy",
    "Direction",
    "Effort",
    "EffortPolicy",
    "Estimator",
    "Evidence",
    "FixedFOVStrategy",
    "FixedObjectStrategy",
    "FixedObservationStrategy",
    "ObjectiveConfig",
    "OnlineSession",
    "PlannedCellCountEstimator",
    "PlannedCellCountStrategy",
    "Quantity",
    "RareTarget",
    "SamplingObjective",
    "SamplingStrategy",
    "ScalarPrecisionEstimator",
    "ScalarPrecisionStrategy",
    "SequentialResult",
    "StrategyResult",
    "ThresholdPolicy",
    "ValidatedInput",
    "evaluate_acquisition_prefixes",
    "plan_cells_to_average",
    "redecide",
    "run_strategies",
    "validate_input_tables",
]
