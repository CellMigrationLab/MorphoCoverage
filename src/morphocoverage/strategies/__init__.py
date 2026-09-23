"""Interchangeable sampling strategies and the estimators they are built from."""

from .base import Estimator, EstimatorState, SamplingStrategy
from .baselines import (
    FixedFOVStrategy,
    FixedObjectStrategy,
    FixedObservationStrategy,
)
from .fov_bootstrap import FOVBootstrapEstimator, FOVBootstrapStrategy
from .jcb_planning import (
    CellCountPlan,
    PlannedCellCountEstimator,
    PlannedCellCountStrategy,
    plan_cells_to_average,
)
from .rare_planning import FOVIncidencePlanEstimator, FOVIncidencePlanStrategy
from .scalar_precision import ScalarPrecisionEstimator, ScalarPrecisionStrategy
from .stabilization import (
    DescriptorPlateauEstimator,
    DescriptorPlateauStrategy,
    DescriptorSpreadEstimator,
    DescriptorSpreadStrategy,
)
from .support_novelty import SupportNoveltyEstimator, SupportNoveltyStrategy

__all__ = [
    "CellCountPlan",
    "DescriptorPlateauEstimator",
    "DescriptorPlateauStrategy",
    "DescriptorSpreadEstimator",
    "DescriptorSpreadStrategy",
    "Estimator",
    "EstimatorState",
    "FOVBootstrapEstimator",
    "FOVBootstrapStrategy",
    "FOVIncidencePlanEstimator",
    "FOVIncidencePlanStrategy",
    "FixedFOVStrategy",
    "FixedObjectStrategy",
    "FixedObservationStrategy",
    "PlannedCellCountEstimator",
    "PlannedCellCountStrategy",
    "SamplingStrategy",
    "ScalarPrecisionEstimator",
    "ScalarPrecisionStrategy",
    "SupportNoveltyEstimator",
    "SupportNoveltyStrategy",
    "plan_cells_to_average",
]
