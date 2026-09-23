"""Interchangeable sampling strategies."""

from .base import SamplingStrategy
from .baselines import FixedFOVStrategy, FixedObjectStrategy
from .scalar_precision import ScalarPrecisionStrategy
from .stabilization import DescriptorPlateauStrategy, DescriptorSpreadStrategy

__all__ = [
    "DescriptorPlateauStrategy",
    "DescriptorSpreadStrategy",
    "SamplingStrategy",
    "FixedFOVStrategy",
    "FixedObjectStrategy",
    "ScalarPrecisionStrategy",
]
