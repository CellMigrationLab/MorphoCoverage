"""The incremental path of the JCB scalar-precision rule (``scalar_precision``).

Per-FOV sufficient statistics, merged at each update, so a live session pays
for the new objects and the acquired FOVs only.

Shared-private: ``_ScalarPrecisionState`` is created by
``ScalarPrecisionEstimator.start``; it is private to the package and may
change with it.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from .._identity import key_labels
from ..evidence import Evidence
from ..schema import ValidatedInput, fov_positions
from ._averaging_cv import (
    _central_sums,
    _FovSummary,
    _merged_moments,
    _missing_reason,
    _object_cv_from_moments,
)
from .base import EstimatorState

if TYPE_CHECKING:
    from .scalar_precision import ScalarPrecisionEstimator


class _ScalarPrecisionState(EstimatorState):
    """Per-FOV sufficient statistics; each update costs O(new objects + acquired FOVs).

    The estimate from these statistics equals full recomputation up to
    floating-point rounding: the object mean, variance and third and fourth
    moments (for the shape check) are merged per FOV instead of computed over
    all values at once.
    """

    def __init__(self, estimator: ScalarPrecisionEstimator, unit: str) -> None:
        self.estimator = estimator
        self.unit = unit  # the group's observation unit, fixed for the session
        self.fovs: list[_FovSummary] = []
        self.order: pd.Series | None = None  # order_col of every FOV, in arrival order

    def _precision(self, fovs: list[_FovSummary]) -> dict[str, Any] | tuple[str, str]:
        """The estimate from per-FOV summaries, as the full path computes it from values."""
        sums = np.array([fov.total for fov in fovs], dtype=float)
        counts = np.array([fov.count for fov in fovs], dtype=float)

        def cells() -> dict[str, float]:
            """Cell-level CVs from the merged moments."""
            n, mean, m2, _, _ = _merged_moments(fovs)
            return _object_cv_from_moments(n, mean, math.sqrt(m2 / (n - 1)))

        def moments() -> tuple[int, float, float, float]:
            """The objects' central sums from the merged moments."""
            n, _, m2, m3, m4 = _merged_moments(fovs)
            return n, m2, m3, m4

        gaps = [fov for fov in fovs if fov.first_missing is not None]
        missing = None
        if gaps:
            missing = _missing_reason(gaps[0].name, gaps[0].first_missing, len(gaps), self.unit)
        return self.estimator._precision_from(
            sums,
            counts,
            has_negative=any(fov.has_negative for fov in fovs),
            cells=cells,
            unit=self.unit,
            moments=moments,
            missing=missing,
        )

    def update(self, new: ValidatedInput) -> Evidence:
        """Summarize the new FOV(s) and recompute the estimate from the summaries."""
        estimator = self.estimator
        estimator._check_order_col(new.fovs)
        values = new.feature_values(estimator.feature)
        positions = fov_positions(new, new.fovs)
        names = key_labels(new.fovs, list(new.fov_key))
        for j in range(len(new.fovs)):
            members = values[positions == j]
            mean = float(members.mean()) if len(members) else 0.0
            _, m2, m3, m4 = _central_sums(members)
            gap = np.isnan(members)
            self.fovs.append(
                _FovSummary(
                    count=len(members),
                    total=float(members.sum()),
                    mean=mean,
                    m2=m2,
                    has_negative=bool((members < 0).any()),
                    m3=m3,
                    m4=m4,
                    name=names[j],
                    first_missing=int(np.argmax(gap)) + 1 if gap.any() else None,
                )
            )
        if estimator.pilot_fovs is not None:
            arrived = new.fovs[estimator.order_col].reset_index(drop=True)
            before = [] if self.order is None else [self.order]
            self.order = pd.concat([*before, arrived], ignore_index=True)
        current = self._precision(self.fovs)
        pilot = None
        if estimator.pilot_fovs is not None and len(self.fovs) >= estimator.pilot_fovs:
            # The full recomputation's sort (pandas, stable: ties by arrival).
            ordered = self.order.sort_values(kind="stable").index[: estimator.pilot_fovs]
            pilot = self._precision([self.fovs[i] for i in ordered])
        n_objects = sum(fov.count for fov in self.fovs)
        return estimator._evidence(current, pilot, n_objects, len(self.fovs), self.unit)
