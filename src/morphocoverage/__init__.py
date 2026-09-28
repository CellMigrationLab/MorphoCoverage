"""MorphoCoverage: FOV-aware phenotypic sampling analysis."""

from . import _deprecated
from .data_check import DataCheck, Problem, check_tables, validate_input_tables
from .evidence import Availability, Component, Direction, Evidence, Quantity
from .framework import run_strategies
from .objectives import ObjectiveConfig, RareTarget, SamplingObjective
from .online import OnlineSession
from .planes import planes_as_features
from .policies import DecisionPolicy, Effort, EffortPolicy, ThresholdPolicy, redecide
from .results import Decision, StrategyResult
from .schema import ValidatedInput
from .sequential import SequentialResult, evaluate_acquisition_prefixes
from .strategies import (
    DescriptorPlateauEstimator,
    DescriptorPlateauStrategy,
    DescriptorSpreadEstimator,
    DescriptorSpreadStrategy,
    Estimator,
    FixedFOVStrategy,
    FixedObjectStrategy,
    FixedObservationStrategy,
    ObjectCountPlan,
    PlannedObjectCountEstimator,
    PlannedObjectCountStrategy,
    SamplingStrategy,
    ScalarPrecisionEstimator,
    ScalarPrecisionStrategy,
    plan_objects_to_average,
)

__all__ = [
    "Availability",
    "Component",
    "DataCheck",
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
    "ObjectCountPlan",
    "ObjectiveConfig",
    "OnlineSession",
    "PlannedObjectCountEstimator",
    "PlannedObjectCountStrategy",
    "Problem",
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
    "check_tables",
    "evaluate_acquisition_prefixes",
    "plan_objects_to_average",
    "planes_as_features",
    "redecide",
    "run_strategies",
    "validate_input_tables",
]


# Former object names, resolved with a DeprecationWarning (see morphocoverage._deprecated).
__getattr__ = _deprecated.module_getattr(
    __name__,
    globals(),
    (
        "plan_cells_to_average",
        "CellCountPlan",
        "PlannedCellCountEstimator",
        "PlannedCellCountStrategy",
    ),
)
