"""Interchangeable sampling strategies."""

from .base import SamplingStrategy
from .baselines import FixedFOVStrategy, FixedObjectStrategy
from .scalar_precision import ScalarPrecisionStrategy

__all__ = [
    "SamplingStrategy",
    "FixedFOVStrategy",
    "FixedObjectStrategy",
    "ScalarPrecisionStrategy",
]
