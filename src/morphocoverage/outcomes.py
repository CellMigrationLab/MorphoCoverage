"""Retrospective outcome metrics: what stopping actually cost, judged from the full acquisition.

An outcome metric is a function of (full acquisition of one analysis group,
the prefix at which a strategy stopped) -> value. It never sees a strategy's
evidence or decision, so every strategy is judged by the same criteria and not
by its own score (docs/benchmarking.md). Metrics are retrospective by design:
they use FOVs the strategy never saw, and must never be used inside an
estimator.

The full acquisition is an empirical reference reservoir, not biological
ground truth: a metric measures agreement with what the complete recorded
acquisition shows.

Metrics return a float, or ``None`` when the outcome is undefined (for example
a relative error of a zero mean); never NaN.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .provenance import class_identity, config_hash
from .schema import ValidatedInput, fov_positions


class OutcomeMetric(ABC):
    """Scores a stop against the full acquisition of the same analysis group.

    A metric instance is configuration only and must be deterministic and
    strategy-independent: ``evaluate`` receives the full group and the stop
    prefix, nothing else.
    """

    name: str

    @property
    def parameters(self) -> Mapping[str, Any]:
        """Serializable settings that define the metric, for provenance."""
        return {}

    @property
    def config_id(self) -> str:
        """Stable hash of the metric class and its parameters."""
        return config_hash(class_identity(type(self)), self.parameters)

    @abstractmethod
    def evaluate(self, full: ValidatedInput, stopped: ValidatedInput) -> float | None:
        """Outcome of stopping at ``stopped``, judged against ``full``; ``None`` if undefined."""
        raise NotImplementedError


def srs_relative_rmse(variance: float, mean: float, n: float, total: float) -> float:
    """Expected relative RMSE of the mean of ``n`` cells drawn at random from ``total``.

    Simple random sampling without replacement, with finite-population
    correction: ``sqrt(variance / n * (1 - n / total)) / |mean|``. This is the
    error cell-number rules assume, used by ``data/crick_idr/fov_vs_cells.py``.
    """
    return math.sqrt(variance / n * (1 - n / total)) / abs(mean)


def _feature(feature: object) -> str:
    if not isinstance(feature, str):
        raise TypeError("feature must be one column name")
    return feature


@dataclass(frozen=True)
class RelativeMeanError(OutcomeMetric):
    """``|mean(stop) - mean(full)| / |mean(full)|`` of one feature's observation mean.

    The realized error of the population summary at the stop. ``None`` when
    the stop prefix has no observations, either acquisition contains missing
    values for the selected feature, or the full mean is zero. With FOVs as the
    acquisition unit this is the quantity whose spread across acquisition
    orders gives the whole-FOV error of ``fov_vs_cells.py``.
    """

    feature: str

    def __post_init__(self) -> None:
        _feature(self.feature)

    @property
    def name(self) -> str:
        """``relative_mean_error:<feature>``."""
        return f"relative_mean_error:{self.feature}"

    @property
    def parameters(self) -> dict[str, Any]:
        """The feature."""
        return {"feature": self.feature}

    def evaluate(self, full: ValidatedInput, stopped: ValidatedInput) -> float | None:
        """Relative error of the stop mean against the full mean."""
        reference = full.feature_values(self.feature)
        values = stopped.feature_values(self.feature)
        if not len(values) or not len(reference):
            return None
        if np.isnan(values).any() or np.isnan(reference).any():
            return None
        target = float(reference.mean())
        if target == 0:
            return None
        return abs(float(values.mean()) - target) / abs(target)


@dataclass(frozen=True)
class RandomCellRelativeError(OutcomeMetric):
    """Expected relative RMSE had the stop's number of cells been drawn at random.

    The comparator that cell-number rules assume: the same number of objects
    as the stop prefix, drawn at random from the full group (simple random
    sampling with finite-population correction, ``srs_relative_rmse``).
    Averaged over acquisition orders, the ratio of ``RelativeMeanError``'s RMSE
    to this value is the error inflation from acquiring whole FOVs. ``None``
    with fewer than 2 objects in the full group or none at the stop, or a zero
    full mean.
    """

    feature: str

    def __post_init__(self) -> None:
        _feature(self.feature)

    @property
    def name(self) -> str:
        """``random_cell_relative_error:<feature>``."""
        return f"random_cell_relative_error:{self.feature}"

    @property
    def parameters(self) -> dict[str, Any]:
        """The feature."""
        return {"feature": self.feature}

    def evaluate(self, full: ValidatedInput, stopped: ValidatedInput) -> float | None:
        """Expected random-cell error at the stop's object count."""
        reference = full.feature_values(self.feature)
        n = len(stopped.objects)
        if len(reference) < 2 or n == 0:
            return None
        if np.isnan(reference).any():
            return None
        mean = float(reference.mean())
        if mean == 0:
            return None
        return srs_relative_rmse(float(reference.var(ddof=1)), mean, n, len(reference))


@dataclass(frozen=True)
class RareStateDetected(OutcomeMetric):
    """1.0 if the stop prefix contains a labelled state in ``min_fov_occurrences`` FOVs, else 0.0.

    ``label_col`` is an object column that labels the state independently of
    any strategy (booleans, or integers 0/1); it is metadata and never needs to
    be a feature. ``None`` when the full acquisition itself does not reach
    ``min_fov_occurrences`` FOVs with the state, since no stop could then
    detect it.
    """

    label_col: str
    min_fov_occurrences: int

    def __post_init__(self) -> None:
        if not isinstance(self.label_col, str):
            raise TypeError("label_col must be one column name")
        if isinstance(self.min_fov_occurrences, bool) or not isinstance(
            self.min_fov_occurrences, int
        ):
            raise TypeError("min_fov_occurrences must be an integer count of FOVs")
        if self.min_fov_occurrences < 1:
            raise ValueError("min_fov_occurrences must be at least 1")

    @property
    def name(self) -> str:
        """``rare_state_detected:<label_col>:<min_fov_occurrences>``."""
        return f"rare_state_detected:{self.label_col}:{self.min_fov_occurrences}"

    @property
    def parameters(self) -> dict[str, Any]:
        """The label column and the number of FOVs required."""
        return {"label_col": self.label_col, "min_fov_occurrences": self.min_fov_occurrences}

    def _fovs_with_state(self, data: ValidatedInput) -> int:
        if self.label_col not in data.objects.columns:
            raise ValueError(f"label_col {self.label_col!r} is not a column of the object table")
        labels = data.objects[self.label_col]
        if not (pd.api.types.is_bool_dtype(labels) or set(labels.unique()) <= {0, 1}):
            raise ValueError(
                f"label_col {self.label_col!r} must hold booleans or 0/1, got "
                f"{sorted(map(str, labels.unique()))[:5]}"
            )
        positions = fov_positions(data, data.fovs)[labels.to_numpy(dtype=bool)]
        return len(np.unique(positions[positions >= 0]))

    def evaluate(self, full: ValidatedInput, stopped: ValidatedInput) -> float | None:
        """Whether the stop prefix already shows the state in enough FOVs."""
        if self._fovs_with_state(full) < self.min_fov_occurrences:
            return None
        return float(self._fovs_with_state(stopped) >= self.min_fov_occurrences)
