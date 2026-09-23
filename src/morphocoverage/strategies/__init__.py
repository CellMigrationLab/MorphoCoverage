"""Interchangeable sampling strategies and the estimators they are built from."""

from .base import Estimator, EstimatorState, SamplingStrategy
from .baselines import FixedFOVStrategy, FixedObjectStrategy
from .scalar_precision import ScalarPrecisionEstimator, ScalarPrecisionStrategy
from .stabilization import (
    DescriptorPlateauEstimator,
    DescriptorPlateauStrategy,
    DescriptorSpreadEstimator,
    DescriptorSpreadStrategy,
)

__all__ = [
    "DescriptorPlateauEstimator",
    "DescriptorPlateauStrategy",
    "DescriptorSpreadEstimator",
    "DescriptorSpreadStrategy",
    "Estimator",
    "EstimatorState",
    "FixedFOVStrategy",
    "FixedObjectStrategy",
    "SamplingStrategy",
    "ScalarPrecisionEstimator",
    "ScalarPrecisionStrategy",
]
