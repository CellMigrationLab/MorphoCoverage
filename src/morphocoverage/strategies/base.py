"""Base interfaces for interchangeable MorphoCoverage strategies.

A strategy is a combination of estimators, which report evidence about the
visible acquisition, and decision policies, which turn that evidence into
continue/stop. Strategy families implement ``Estimator``; the framework runs
the estimators, applies the policies and builds the result.

Shared-private: ``_as_ints`` is used by ``stabilization`` and ``fov_bootstrap``
to store integer settings given as numpy integers as Python ints; it is private
to the package and may change with it. Both resample FOVs in an order that the
table's row order cannot change (``schema.canonical_fields``).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from typing import Any

import pandas as pd

from .._wording import count
from ..evidence import Evidence
from ..objectives import ObjectiveConfig
from ..policies import DecisionPolicy, describe, strategy_config_id
from ..provenance import class_identity, config_hash
from ..schema import ValidatedInput


class EstimatorState(ABC):
    """Running state of an incremental estimator within one analysis group.

    Created by ``Estimator.start`` for one group and fed each newly revealed
    FOV in acquisition order. ``update`` must return the same evidence as
    ``Estimator.estimate`` on the whole visible prefix (to floating-point
    rounding); a contract test checks this on every scenario. Full
    recomputation remains the reference implementation.
    """

    @abstractmethod
    def update(self, new: ValidatedInput) -> Evidence:
        """Add the newly revealed FOV (its FOV row and its objects) and return the evidence."""
        raise NotImplementedError


class ConstantState(EstimatorState):
    """Incremental state for evidence that does not depend on the revealed FOVs."""

    def __init__(self, evidence: Evidence) -> None:
        self.evidence = evidence

    def update(self, new: ValidatedInput) -> Evidence:
        """The same evidence as at the start."""
        return self.evidence


class Estimator(ABC):
    """Computes evidence from the visible acquisition of one analysis group.

    An estimator instance is configuration only. The framework calls
    ``estimate`` on a fresh deep copy, with its own copy of the input, for
    every analysis group and acquisition prefix, so state stored on the
    instance never carries between calls. An estimator may read the
    objective's targets (for example a ``RareTarget``) but never decides:
    thresholds belong to decision policies.
    """

    name: str

    @property
    def parameters(self) -> Mapping[str, Any]:
        """Serializable settings that define the estimator, for provenance."""
        return {}

    @property
    def config_id(self) -> str:
        """Stable hash of the estimator class and its parameters."""
        return config_hash(class_identity(type(self)), self.parameters)

    @abstractmethod
    def estimate(self, data: ValidatedInput, *, objective: ObjectiveConfig) -> Evidence:
        """Evidence about the current acquisition state."""
        raise NotImplementedError

    def start(
        self, template: ValidatedInput, *, objective: ObjectiveConfig
    ) -> EstimatorState | None:
        """Optional incremental support; ``None`` (the default) means full recomputation.

        ``template`` describes one analysis group with no FOVs revealed yet:
        empty tables with the group's columns and the explicit features.
        """
        return None


class SamplingStrategy:
    """Estimators plus the decision policies applied to their evidence.

    Compose one directly, ``SamplingStrategy([estimator], [policy], name=...)``,
    or subclass it and override ``estimators``, ``policies``, ``name`` and
    ``parameters``, as the packaged baselines do to keep their published
    parameterization. ``config_id`` depends on the estimators' and policies'
    ids only, not on names, so two strategies with the same id are the same
    method.
    """

    def __init__(
        self,
        estimators: Iterable[Estimator] = (),
        policies: Iterable[DecisionPolicy] = (),
        *,
        name: str | None = None,
    ) -> None:
        self._estimators = tuple(estimators)
        self._policies = tuple(policies)
        for estimator in self._estimators:
            if not isinstance(estimator, Estimator):
                raise TypeError(f"{estimator!r} is not an Estimator")
        for policy in self._policies:
            if not isinstance(policy, DecisionPolicy):
                raise TypeError(f"{policy!r} is not a DecisionPolicy")
        if not self._estimators and not self._policies:
            raise ValueError("a strategy needs at least one estimator or policy")
        self._name = name or "+".join(
            [e.name for e in self._estimators] + [p.name for p in self._policies]
        )

    @property
    def name(self) -> str:
        """Human-readable name; not unique across configurations (see ``config_id``)."""
        return self._name

    @property
    def estimators(self) -> tuple[Estimator, ...]:
        """Estimators run on the visible data."""
        return self._estimators

    @property
    def policies(self) -> tuple[DecisionPolicy, ...]:
        """Policies applied to the combined evidence."""
        return self._policies

    @property
    def configuration(self) -> dict[str, Any]:
        """Name, type, config_id and parameters of every estimator and policy."""
        return {
            "estimators": [describe(e) for e in self.estimators],
            "policies": [describe(p) for p in self.policies],
        }

    @property
    def parameters(self) -> Mapping[str, Any]:
        """Serializable strategy parameters for provenance."""
        return self.configuration

    @property
    def config_id(self) -> str:
        """Stable hash of the estimators' and policies' configurations."""
        return strategy_config_id(
            [e.config_id for e in self.estimators], [p.config_id for p in self.policies]
        )


def unsupported_observation_unit(
    data: ValidatedInput,
    allowed: Iterable[str],
    *,
    method: str,
) -> str | None:
    """Why ``method`` does not apply to these rows, or ``None`` if it does.

    Generic MorphoCoverage inputs may contain cells, nuclei, patches, slices,
    embeddings or FOV summaries. A method derived for cell/object sampling must
    not reinterpret other rows as objects. It reports its quantities as
    ``not_applicable`` with this reason, so a benchmark or live session keeps
    running and the decision is undefined rather than an error.
    """
    allowed_units = tuple(sorted(allowed))
    if data.observation_unit in allowed_units:
        return None
    return (
        f"{method} is defined for individual objects ({', '.join(allowed_units)}), and these "
        f"rows are of another unit ({data.observation_unit})"
    )


def _as_ints(instance: object, *names: str) -> None:
    """Store a frozen dataclass's named integer settings as Python ints (``None`` kept).

    Checked settings may be numpy integers (``np.int64(3)``); provenance must
    be JSON, so they are stored as ``int``.
    """
    for name in names:
        value = getattr(instance, name)
        if value is not None:
            object.__setattr__(instance, name, int(value))


def check_order_col(
    fovs: pd.DataFrame, order_col: str, purpose: str, argument: str = "order_col"
) -> None:
    """Refuse an order column that is missing, incomplete, categorical or not numbers or times.

    A field with no order value has no place in the order: sorting would put it
    last without saying so, and the full and the incremental evaluation could
    place it differently. Text is refused rather than sorted as text, where
    ``"10"`` comes before ``"2"``; booleans are not an order either. The replay
    checks its acquisition order with it too (``sequential.REPLAY_PURPOSE``; owner,
    2026-10-12), so a pilot or a plateau window follows the order a replay would.
    A categorical column is refused (``TypeError``): a
    newly revealed FOV carries only its own category (the framework hides the
    categories of FOVs not yet seen), so its category order cannot be kept FOV
    by FOV. ``purpose`` names what needs the order and ``argument`` the caller's argument
    that named the column, for the messages (owner, 2026-10-12).
    """
    if order_col not in fovs.columns:
        raise ValueError(f"{argument} {order_col!r} is not a column of the FOV table")
    values = fovs[order_col]
    missing = int(values.isna().sum())
    if missing:
        raise ValueError(
            f"{argument} {order_col!r} has missing values; {purpose} needs an order value for "
            f"every FOV ({count(missing, 'FOV row')} without one)"
        )
    dtype, types = values.dtype, pd.api.types
    if isinstance(dtype, pd.CategoricalDtype):
        raise TypeError(
            f"{argument} {order_col!r} is categorical, and its category order is lost when FOVs "
            f"arrive one at a time; give {purpose} its order as numbers or times (for an "
            "ordered categorical, its .cat.codes)"
        )
    numeric = types.is_numeric_dtype(dtype) and not types.is_bool_dtype(dtype)
    if not (numeric or types.is_datetime64_any_dtype(dtype) or types.is_timedelta64_dtype(dtype)):
        raise ValueError(
            f"{argument} {order_col!r} holds {dtype} values, not numbers or date-times; text "
            f"would sort '10' before '2'. Convert it first, for example "
            f"fovs[{order_col!r}] = pd.to_numeric(fovs[{order_col!r}]) or pd.to_datetime(...)."
        )


__all__ = [
    "ConstantState",
    "Estimator",
    "EstimatorState",
    "SamplingStrategy",
    "check_order_col",
    "unsupported_observation_unit",
]
