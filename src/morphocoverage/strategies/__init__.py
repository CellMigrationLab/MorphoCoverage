"""Interchangeable sampling strategies and the estimators they are built from."""

from .base import Estimator, EstimatorState, SamplingStrategy
from .baselines import (
    FixedFOVStrategy,
    FixedObjectStrategy,
    FixedObservationStrategy,
)
from .jcb_planning import (
    CellCountPlan,
    PlannedCellCountEstimator,
    PlannedCellCountStrategy,
    plan_cells_to_average,
)
from .scalar_precision import ScalarPrecisionEstimator, ScalarPrecisionStrategy
from .stabilization import (
    DescriptorPlateauEstimator,
    DescriptorPlateauStrategy,
    DescriptorSpreadEstimator,
    DescriptorSpreadStrategy,
)

__all__ = [
    "CellCountPlan",
    "DescriptorPlateauEstimator",
    "DescriptorPlateauStrategy",
    "DescriptorSpreadEstimator",
    "DescriptorSpreadStrategy",
    "Estimator",
    "EstimatorState",
    "FixedFOVStrategy",
    "FixedObjectStrategy",
    "FixedObservationStrategy",
    "PlannedCellCountEstimator",
    "PlannedCellCountStrategy",
    "SamplingStrategy",
    "ScalarPrecisionEstimator",
    "ScalarPrecisionStrategy",
    "plan_cells_to_average",
]
