"""Retrospective outcome metrics: what stopping actually cost, judged from the full acquisition.

An outcome metric is a function of (full acquisition of one analysis group,
the prefix at which a strategy stopped) -> value. It never sees a strategy's
evidence or decision, so every strategy is judged by the same criteria and not
by its own score (docs/optimise/benchmarking.md). Metrics are retrospective by design:
they use FOVs the strategy never saw, and must never be used inside an
estimator.

The full acquisition is an empirical reference reservoir, not biological
ground truth: a metric measures agreement with what the complete recorded
acquisition shows.

Metrics return a float, or ``None`` when the outcome is undefined (for example
a relative error of a zero mean); never NaN.

Every metric takes an optional keyword ``label``: when given it is the
metric's ``name`` (the key of its value in benchmark records) and enters its
parameters, so two settings of one metric can be told apart; without it the
name and ``config_id`` are the defaults.

Shared-private: ``_check_label`` (used by ``multivariate_outcomes``).
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .objectives import fov_occurrences
from .provenance import class_identity, config_hash
from .schema import ValidatedInput
from .statistics import mean_is_zero, srs_relative_rmse
from .strategies.rare_incidence import _positive_fovs


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

    @property
    def min_observations(self) -> int:
        """The fewest observations a stop needs for the metric to be computed (default 1).

        ``OutcomeCriterion.assess`` counts a stop with fewer observations as
        not preserved, as it does an empty one, when the reference itself can
        be judged: too few observations to compute the metric is too few to
        preserve it. A covariance, for example, needs 2.
        """
        return 1

    def trained(self, training: ValidatedInput) -> OutcomeMetric:
        """The metric with any fitted state fitted on ``training`` alone: the fields of a
        group that are not held back, when held-back fields judge a stop
        (``sampling_reliability``, ``replay_acquisitions``), so no held-back field enters a
        fit (owner, 2026-10-06). The metric itself when it has no fitted state (the default)."""
        return self

    @abstractmethod
    def evaluate(self, full: ValidatedInput, stopped: ValidatedInput) -> float | None:
        """Outcome of stopping at ``stopped``, judged against ``full``; ``None`` if undefined."""
        raise NotImplementedError


def _feature(feature: object) -> str:
    """Require a single column name."""
    if not isinstance(feature, str):
        raise TypeError("feature must be one column name")
    return feature


def _check_label(label: object) -> None:
    """``None`` or a non-empty string: a metric's own name."""
    if label is not None and (not isinstance(label, str) or not label):
        raise TypeError("label must be a non-empty string (the metric's name) or None")


def _named(label: str | None, default: str) -> str:
    """The metric's name: its label when given, else its default name."""
    return default if label is None else label


def _labelled(label: str | None, parameters: dict[str, Any]) -> dict[str, Any]:
    """``parameters`` with the label, when one is given (the default ``config_id`` otherwise)."""
    return parameters if label is None else {**parameters, "label": label}


def _stop_within(full: ValidatedInput, stopped: ValidatedInput) -> bool:
    """Whether every FOV of the stop is a FOV of the reference (the stop drawn from it)."""
    if tuple(stopped.fov_key) != tuple(full.fov_key):
        return False
    key = list(full.fov_key)
    reference = set(zip(*(full.fovs[column].tolist() for column in key)))
    return all(fov in reference for fov in zip(*(stopped.fovs[column].tolist() for column in key)))


@dataclass(frozen=True)
class RelativeMeanError(OutcomeMetric):
    """``|mean(stop) - mean(full)| / |mean(full)|`` of one feature's observation mean.

    The realized error of the population summary at the stop. ``None`` when
    the stop prefix has no observations, either acquisition contains missing
    values for the selected feature, or the full mean is zero (up to rounding: a
    mean of 1e-17 from values of about 0.1 is a zero one). With FOVs as the
    acquisition unit this is the quantity whose spread across acquisition
    orders gives the whole-FOV error of ``fov_vs_objects.py``.
    """

    feature: str
    label: str | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        _feature(self.feature)
        _check_label(self.label)

    @property
    def name(self) -> str:
        """``label`` when given, else ``relative_mean_error:<feature>``."""
        return _named(self.label, f"relative_mean_error:{self.feature}")

    @property
    def parameters(self) -> dict[str, Any]:
        """The feature, and the label when given."""
        return _labelled(self.label, {"feature": self.feature})

    def evaluate(self, full: ValidatedInput, stopped: ValidatedInput) -> float | None:
        """Relative error of the stop mean against the full mean."""
        reference = full.feature_values(self.feature)
        values = stopped.feature_values(self.feature)
        if not len(values) or not len(reference):
            return None
        if np.isnan(values).any() or np.isnan(reference).any():
            return None
        target = float(reference.mean())
        if mean_is_zero(target, reference):
            return None
        return abs(float(values.mean()) - target) / abs(target)


@dataclass(frozen=True)
class RandomObjectRelativeError(OutcomeMetric):
    """Expected relative RMSE had the stop's number of objects been drawn at random.

    The comparator that object-number rules assume: the same number of objects
    as the stop prefix, drawn at random from the reference's objects.
    Averaged over acquisition orders, the ratio of ``RelativeMeanError``'s RMSE
    to this value is the error inflation from acquiring whole FOVs. ``None``
    with fewer than 2 objects in the reference or none at the stop, or a zero
    reference mean (up to rounding).

    When the reference contains the stop (the full group, the default), the
    stop's objects are a sample without replacement from that finite
    population: simple random sampling with finite-population correction,
    ``sqrt(s^2 / n * (1 - n / N)) / |mean|`` (``srs_relative_rmse``). When it
    does not (a held-out reservoir, whose FOVs the stop never sees), the stop's
    mean estimates the population the reference samples, and there is no
    finite population to correct for: ``sqrt(s^2 / n) / |mean|``, for any
    ``n``, including more objects than the reference holds.
    """

    feature: str
    label: str | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        _feature(self.feature)
        _check_label(self.label)

    @property
    def name(self) -> str:
        """``label`` when given, else ``random_object_relative_error:<feature>``."""
        return _named(self.label, f"random_object_relative_error:{self.feature}")

    @property
    def parameters(self) -> dict[str, Any]:
        """The feature, and the label when given."""
        return _labelled(self.label, {"feature": self.feature})

    def evaluate(self, full: ValidatedInput, stopped: ValidatedInput) -> float | None:
        """Expected random-object error at the stop's object count."""
        reference = full.feature_values(self.feature)
        n = len(stopped.objects)
        if len(reference) < 2 or n == 0:
            return None
        if np.isnan(reference).any():
            return None
        mean = float(reference.mean())
        if mean_is_zero(mean, reference):
            return None
        within = _stop_within(full, stopped) and n <= len(reference)
        # An infinite population is no finite-population correction.
        population = len(reference) if within else math.inf
        return srs_relative_rmse(float(reference.var(ddof=1)), mean, n, population)


@dataclass(frozen=True)
class RareStateDetected(OutcomeMetric):
    """1.0 if the stop prefix contains a labelled state in ``min_fov_occurrences`` FOVs, else 0.0.

    ``label_col`` is an object column that labels the state independently of
    any strategy (booleans, or integers 0/1); it is metadata and never needs to
    be a feature. FOVs holding the state are counted as ``RareIncidenceEstimator``
    counts them. ``None`` when the full acquisition itself does not reach
    ``min_fov_occurrences`` FOVs with the state, since no stop could then
    detect it, or when a label is missing in either acquisition (the
    estimator reports its risk as not applicable then).
    """

    label_col: str
    min_fov_occurrences: int
    label: str | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        if not isinstance(self.label_col, str):
            raise TypeError("label_col must be one column name")
        occurrences = fov_occurrences(self.min_fov_occurrences)
        object.__setattr__(self, "min_fov_occurrences", occurrences)
        _check_label(self.label)

    @property
    def name(self) -> str:
        """``label`` when given, else ``rare_state_detected:<label_col>:<min_fov_occurrences>``."""
        default = f"rare_state_detected:{self.label_col}:{self.min_fov_occurrences}"
        return _named(self.label, default)

    @property
    def parameters(self) -> dict[str, Any]:
        """The label column, the number of FOVs required, and the label when given."""
        settings = {"label_col": self.label_col, "min_fov_occurrences": self.min_fov_occurrences}
        return _labelled(self.label, settings)

    def evaluate(self, full: ValidatedInput, stopped: ValidatedInput) -> float | None:
        """Whether the stop prefix already shows the state in enough FOVs."""
        in_full = _positive_fovs(full, self.label_col)
        if in_full is None or in_full < self.min_fov_occurrences:
            return None
        at_stop = _positive_fovs(stopped, self.label_col)
        return None if at_stop is None else float(at_stop >= self.min_fov_occurrences)


__all__ = ["OutcomeMetric", "RandomObjectRelativeError", "RareStateDetected", "RelativeMeanError"]
