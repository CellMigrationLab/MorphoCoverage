"""Base interfaces for interchangeable MorphoCoverage strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Mapping, Any

from ..objectives import ObjectiveConfig
from ..results import StrategyResult
from ..schema import ValidatedInput


class SamplingStrategy(ABC):
    """Interface implemented by every sampling/completeness strategy.

    A strategy instance is configuration only. The framework calls ``evaluate``
    on a fresh deep copy for every analysis group and acquisition prefix, so
    state stored on the instance never carries between calls. ``evaluate``
    receives one resolved analysis group; it must use only the rows it is
    given and should leave framework-owned result fields (counts, group,
    runtime) unset.
    """

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
