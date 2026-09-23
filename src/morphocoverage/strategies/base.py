"""Base interfaces for interchangeable MorphoCoverage strategies.

A strategy is a combination of estimators, which report evidence about the
visible acquisition, and decision policies, which turn that evidence into
continue/stop. Strategy families implement ``Estimator``; the framework runs
the estimators, applies the policies and builds the result.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from typing import Any

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
        self._name = name or "+".join(
            [e.name for e in self._estimators] + [p.name for p in self._policies]
        )
        if not self._name:
            raise ValueError("a strategy needs at least one estimator or policy")

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


def require_observation_unit(
    data: ValidatedInput,
    allowed: Iterable[str],
    *,
    method: str,
) -> None:
    """Require an estimator's scientifically valid observation semantics.

    Generic MorphoCoverage inputs may contain cells, nuclei, patches, slices,
    embeddings or FOV summaries. A method derived for cell/object sampling must
    refuse incompatible row semantics rather than silently reinterpret them.
    """
    allowed_units = tuple(allowed)
    if data.observation_unit not in allowed_units:
        raise ValueError(
            f"{method} requires observation_unit in {allowed_units}, "
            f"got {data.observation_unit!r}"
        )
