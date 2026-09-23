"""Canonical strategy result types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .objectives import ObjectiveConfig


@dataclass(frozen=True)
class StrategyResult:
    """One strategy evaluation at one acquisition state."""

    strategy: str
    objective: ObjectiveConfig
    score: float | None
    decision: str | None = None
    predicted_remaining: float | None = None
    runtime_seconds: float | None = None
    n_fovs: int | None = None
    n_objects: int | None = None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    parameters: Mapping[str, Any] = field(default_factory=dict)
