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

    ``n_fovs``, ``n_objects``, ``group``, ``feature_cols``, ``fov_key`` and
    ``runtime_seconds`` are owned by the framework and filled after the
    strategy returns. Strategies leave them unset; a strategy-supplied count
    that disagrees with the acquisition record is rejected.

    ``runtime_seconds`` is the wall-clock time of the strategy's own
    ``evaluate`` call only. It excludes input copying, prefix construction and
    other framework overhead, and it is a full recomputation, not an
    incremental update, so it is not yet an end-to-end smart-microscopy
    latency.

    ``feature_cols`` and ``fov_key`` record the phenotype representation and
    the FOV identity the result was computed on, as AGENTS.md section 3
    requires; strategy-specific settings (preprocessing, metric, thresholds)
    belong in ``parameters``.
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
    feature_cols: tuple[str, ...] = ()
    fov_key: tuple[str, ...] = ()
