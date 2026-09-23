"""Base interfaces for interchangeable MorphoCoverage strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Mapping, Any

from ..objectives import ObjectiveConfig
from ..results import StrategyResult
from ..schema import ValidatedInput


class SamplingStrategy(ABC):
    """Interface implemented by every sampling/completeness strategy."""

    name: str

    @property
    def parameters(self) -> Mapping[str, Any]:
        """Return serializable strategy parameters for provenance."""
        return {}

    @abstractmethod
    def evaluate(
        self,
        data: ValidatedInput,
        *,
        objective: ObjectiveConfig,
    ) -> StrategyResult:
        """Evaluate the current acquisition state."""
        raise NotImplementedError
