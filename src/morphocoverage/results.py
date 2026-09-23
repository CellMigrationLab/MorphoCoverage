"""Canonical strategy result types."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .evidence import Evidence
from .objectives import ObjectiveConfig


class Decision(str, Enum):
    """Continue/stop recommendations; any other value is rejected.

    ``UNSUPPORTED`` means the strategy has no evidence about a component the
    objective requires, for example a population-precision rule asked about
    rare-state discovery. It is never a disguised ``STOP``.
    """

    CONTINUE = "continue"
    STOP = "stop"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class StrategyResult:
    """One strategy evaluation for one analysis group and acquisition state.

    Built by the framework, never by a strategy: estimators supply
    ``evidence``, decision policies supply ``decision`` and
    ``component_decisions``, and the framework fills everything else from the
    acquisition record.

    - ``config_id`` identifies the estimators and policies exactly; ``strategy``
      is a human-readable name that several configurations may share.
    - ``component_decisions`` maps each objective component
      (``"population"``, ``"rare"``) that has a policy, or that the objective
      requires, to its decision. ``decision`` combines the required ones.
    - ``compute_seconds`` is the wall-clock time of the estimators' ``estimate``
      calls only. It excludes input copying, prefix construction and other
      framework overhead, and it is a full recomputation, not an incremental
      update, so it is not yet an end-to-end smart-microscopy latency.
      ``runtime_seconds`` is an alias.
    - ``feature_cols`` and ``fov_key`` record the phenotype representation and
      the FOV identity, as AGENTS.md section 3 requires; method settings
      (preprocessing, metric, thresholds) are in ``parameters`` and
      ``configuration``.
    """

    strategy: str
    config_id: str
    objective: ObjectiveConfig
    evidence: Evidence
    decision: Decision | None
    component_decisions: Mapping[str, Decision | None] = field(default_factory=dict)
    compute_seconds: float | None = None
    n_fovs: int | None = None
    n_objects: int | None = None
    group: Mapping[str, Any] = field(default_factory=dict)
    parameters: Mapping[str, Any] = field(default_factory=dict)
    configuration: Mapping[str, Any] = field(default_factory=dict)
    feature_cols: tuple[str, ...] = ()
    fov_key: tuple[str, ...] = ()

    @property
    def runtime_seconds(self) -> float | None:
        """Alias of ``compute_seconds``."""
        return self.compute_seconds

    @property
    def diagnostics(self) -> Mapping[str, Any]:
        """The estimators' diagnostics."""
        return self.evidence.diagnostics
