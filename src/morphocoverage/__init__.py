"""MorphoCoverage: FOV-aware phenotypic sampling analysis."""

from .framework import run_strategies
from .objectives import ObjectiveConfig, SamplingObjective
from .results import Decision, StrategyResult
from .schema import ValidatedInput, validate_input_tables
from .sequential import SequentialResult, evaluate_acquisition_prefixes
from .strategies import (
    DescriptorPlateauStrategy,
    DescriptorSpreadStrategy,
    FixedFOVStrategy,
    FixedObjectStrategy,
    SamplingStrategy,
    ScalarPrecisionStrategy,
)

__all__ = [
    "Decision",
    "DescriptorPlateauStrategy",
    "DescriptorSpreadStrategy",
    "FixedFOVStrategy",
    "FixedObjectStrategy",
    "ObjectiveConfig",
    "SamplingObjective",
    "SamplingStrategy",
    "ScalarPrecisionStrategy",
    "SequentialResult",
    "StrategyResult",
    "ValidatedInput",
    "evaluate_acquisition_prefixes",
    "run_strategies",
    "validate_input_tables",
]
