"""Canonical strategy result types."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .objectives import ObjectiveConfig


class Decision(str, Enum):
    """Allowed continue/stop recommendations; any other value is rejected."""

    CONTINUE = "continue"
    STOP = "stop"


@dataclass(frozen=True)
class StrategyResult:
    """One strategy evaluation for one analysis group and acquisition state.

    ``n_fovs``, ``n_objects``, ``group`` and ``runtime_seconds`` are owned by the
    framework and filled after the strategy returns. Strategies may leave them
    unset; a strategy-supplied count that disagrees with the acquisition record
    is rejected.
    """

    strategy: str
    objective: ObjectiveConfig
    score: float | None
    decision: Decision | None = None
    predicted_remaining: float | None = None
    runtime_seconds: float | None = None
    n_fovs: int | None = None
    n_objects: int | None = None
    group: Mapping[str, Any] = field(default_factory=dict)
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    parameters: Mapping[str, Any] = field(default_factory=dict)
