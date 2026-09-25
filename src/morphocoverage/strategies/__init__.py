"""Interchangeable sampling strategies and the estimators they are built from."""

from .. import _deprecated
from .base import Estimator, EstimatorState, SamplingStrategy
from .baselines import (
    FixedAreaStrategy,
    FixedFOVStrategy,
    FixedObjectStrategy,
    FixedObservationStrategy,
    ImagedAreaEstimator,
)
from .fov_bootstrap import FOVBootstrapEstimator, FOVBootstrapStrategy
from .incidence_coverage import IncidenceCoverageEstimator, IncidenceCoverageStrategy
from .jcb_planning import (
    ObjectCountPlan,
    PlannedObjectCountEstimator,
    PlannedObjectCountStrategy,
    plan_objects_to_average,
)
from .rare_incidence import RareIncidenceEstimator, RareIncidenceStrategy
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
    "FixedAreaStrategy",
    "FixedFOVStrategy",
    "FixedObjectStrategy",
    "FixedObservationStrategy",
    "ImagedAreaEstimator",
    "IncidenceCoverageEstimator",
    "IncidenceCoverageStrategy",
    "ObjectCountPlan",
    "PlannedObjectCountEstimator",
    "PlannedObjectCountStrategy",
    "RareIncidenceEstimator",
    "RareIncidenceStrategy",
    "SamplingStrategy",
    "ScalarPrecisionEstimator",
    "ScalarPrecisionStrategy",
    "SupportNoveltyEstimator",
    "SupportNoveltyStrategy",
    "plan_objects_to_average",
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
