"""Leakage-safe feature transforms, fitted on the visible acquisition only.

Multivariate estimators usually rescale or project features before measuring
distances. If each estimator did this its own way, fitting on data it should
not see and unfair comparisons would both become likely. A
``FeatureTransform`` is declared as an estimator parameter and fitted inside
``estimate``, on the ``ValidatedInput`` the estimator receives. The framework
only ever gives an estimator the visible prefix of one analysis group, so a
fitted transform cannot depend on future FOVs or other groups. A contract
test checks this for every transform.

The fitted state (means, scales, components) is JSON-serializable and belongs
in the estimator's diagnostics (``FittedTransform.provenance()``), so the
phenotype geometry of every result can be reproduced (AGENTS.md section 3).

What this cannot detect: features normalized or projected over the whole
dataset upstream, before they reach MorphoCoverage. Record that as provenance.

A transform that cannot be fitted on the visible data (too few observations, a
feature with zero spread) returns a fitted transform with a ``reason``.
Estimators should then report their quantities as insufficient data rather
than guess a scale.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .provenance import class_identity, config_hash
from .schema import ValidatedInput


class FeatureTransform(ABC):
    """A feature-space transform fitted on visible data only.

    Subclasses implement ``fit_matrix`` and ``apply_matrix`` on a float
    matrix with one row per phenotype observation and one column per named input.
    A transform instance is configuration only.
    """

    name: str

    @property
    def parameters(self) -> Mapping[str, Any]:
        """Serializable settings that define the transform, for provenance."""
        return {}

    @property
    def config_id(self) -> str:
        """Stable hash of the transform class and its parameters."""
        return config_hash(class_identity(type(self)), self.parameters)

    def describe(self) -> dict[str, Any]:
        """Name, type, config_id and parameters, for an estimator's ``parameters``."""
        return {
            "name": self.name,
            "type": class_identity(type(self)),
            "config_id": self.config_id,
            "parameters": dict(self.parameters),
        }

    @abstractmethod
    def fit_matrix(
        self, matrix: np.ndarray, names: tuple[str, ...]
    ) -> tuple[dict[str, Any], tuple[str, ...]] | str:
        """Fitted state and output names, or the reason the transform cannot be fitted."""
        raise NotImplementedError

    @abstractmethod
    def apply_matrix(self, state: Mapping[str, Any], matrix: np.ndarray) -> np.ndarray:
        """Apply a fitted state to a matrix with the same input columns."""
        raise NotImplementedError

    def fit(self, data: ValidatedInput) -> FittedTransform:
        """Fit on the explicit features of ``data``: the visible prefix of one group."""
        names = tuple(data.feature_cols)
        fitted = self.fit_matrix(data.feature_matrix(), names)
        common = {
            "transform": self,
            "input_features": names,
            "n_fit_observations": len(data.objects),
            "n_fit_fovs": len(data.fovs),
        }
        if isinstance(fitted, str):
            return FittedTransform(**common, reason=fitted)
        state, outputs = fitted
        return FittedTransform(**common, state=state, output_names=outputs)


@dataclass(frozen=True)
class FittedTransform:
    """A transform with its state fitted on one visible acquisition."""

    transform: FeatureTransform
    input_features: tuple[str, ...]
    n_fit_observations: int
    n_fit_fovs: int
    state: Mapping[str, Any] = field(default_factory=dict)
    output_names: tuple[str, ...] = ()
    reason: str | None = None

    @property
    def n_fit_objects(self) -> int:
        """Backward-compatible alias for n_fit_observations."""
        return self.n_fit_observations

    @property
    def fitted(self) -> bool:
        """Whether the transform could be fitted; see ``reason`` otherwise."""
        return self.reason is None

    def apply(self, data: ValidatedInput) -> np.ndarray:
        """Transformed features of ``data``, which must have the fitted input features."""
        if not self.fitted:
            raise ValueError(f"transform {self.transform.name!r} was not fitted: {self.reason}")
        if tuple(data.feature_cols) != self.input_features:
            raise ValueError(
                f"transform fitted on features {list(self.input_features)} cannot be "
                f"applied to {list(data.feature_cols)}"
            )
        return self.transform.apply_matrix(self.state, data.feature_matrix())

    def provenance(self) -> dict[str, Any]:
        """The transform, its fitted state and what it was fitted on, for diagnostics."""
        record = {
            **self.transform.describe(),
            "input_features": list(self.input_features),
            "n_fit_observations": self.n_fit_observations,
            "n_fit_objects": self.n_fit_objects,
            "n_fit_fovs": self.n_fit_fovs,
        }
        if self.fitted:
            record["fitted_state"] = dict(self.state)
            record["output_names"] = list(self.output_names)
        else:
            record["reason"] = self.reason
        return record


def _floats(values: np.ndarray) -> list:
    """Plain Python floats (lists), so fitted state is JSON-serializable."""
    return np.asarray(values, dtype=float).tolist()


@dataclass(frozen=True)
class Standardize(FeatureTransform):
    """Centre each feature on its visible mean and divide by its visible SD (``ddof=1``).

    Not fitted with fewer than 2 observations, or if a feature has zero SD on the
    visible data: dividing by zero or by an arbitrary constant would silently
    define the geometry.
    """

    name: str = "standardize"

    def fit_matrix(
        self, matrix: np.ndarray, names: tuple[str, ...]
    ) -> tuple[dict[str, Any], tuple[str, ...]] | str:
        """Visible mean and SD per feature."""
        if np.isnan(matrix).any():
            return "missing feature values; choose or implement an explicit missing-data policy"
        if len(matrix) < 2:
            return "fewer than 2 observations"
        mean = matrix.mean(axis=0)
        sd = matrix.std(axis=0, ddof=1)
        constant = [name for name, value in zip(names, sd, strict=True) if not value > 0]
        if constant:
            return f"zero SD on the visible data for {constant}"
        return {"mean": _floats(mean), "sd": _floats(sd)}, names

    def apply_matrix(self, state: Mapping[str, Any], matrix: np.ndarray) -> np.ndarray:
        """``(x - mean) / sd``."""
        return (matrix - np.asarray(state["mean"])) / np.asarray(state["sd"])


@dataclass(frozen=True)
class PCA(FeatureTransform):
    """Project the centred visible features onto their first ``n_components`` principal axes.

    Components come from the SVD of the centred visible matrix. Each
    component's sign is fixed so that its largest-magnitude loading is
    positive, so the projection is deterministic. Not fitted unless there are
    more observations than components and at least as many features as components.
    The explained-variance ratios are recorded in the fitted state.
    """

    n_components: int

    def __post_init__(self) -> None:
        if isinstance(self.n_components, bool) or not isinstance(self.n_components, int):
            raise TypeError("n_components must be an integer")
        if self.n_components < 1:
            raise ValueError("n_components must be at least 1")

    @property
    def name(self) -> str:
        """``pca<n_components>``."""
        return f"pca{self.n_components}"

    @property
    def parameters(self) -> dict[str, Any]:
        """The number of components."""
        return {"n_components": self.n_components}

    def fit_matrix(
        self, matrix: np.ndarray, names: tuple[str, ...]
    ) -> tuple[dict[str, Any], tuple[str, ...]] | str:
        """Visible mean, leading components and their explained-variance ratios."""
        if np.isnan(matrix).any():
            return "missing feature values; choose or implement an explicit missing-data policy"
        n_observations, n_features = matrix.shape
        if n_features < self.n_components:
            return f"{n_features} features cannot give {self.n_components} components"
        if n_observations <= self.n_components:
            return f"{n_observations} observations cannot define {self.n_components} components"
        mean = matrix.mean(axis=0)
        _, singular, vt = np.linalg.svd(matrix - mean, full_matrices=False)
        total = float((singular**2).sum())
        if total == 0:
            return "the visible features have no variance"
        components = vt[: self.n_components]
        signs = np.sign(components[np.arange(self.n_components), np.abs(components).argmax(1)])
        components = components * signs[:, None]
        state = {
            "mean": _floats(mean),
            "components": _floats(components),
            "explained_variance_ratio": _floats(singular[: self.n_components] ** 2 / total),
        }
        return state, tuple(f"pc{i + 1}" for i in range(self.n_components))

    def apply_matrix(self, state: Mapping[str, Any], matrix: np.ndarray) -> np.ndarray:
        """Scores on the fitted components."""
        return (matrix - np.asarray(state["mean"])) @ np.asarray(state["components"]).T


class Chain(FeatureTransform):
    """Transforms applied in order, each fitted on the previous one's visible output."""

    def __init__(self, transforms: Iterable[FeatureTransform]) -> None:
        self.transforms: tuple[FeatureTransform, ...] = tuple(transforms)
        if not self.transforms:
            raise ValueError("a chain needs at least one transform")
        for transform in self.transforms:
            if not isinstance(transform, FeatureTransform):
                raise TypeError(f"{transform!r} is not a FeatureTransform")

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Chain) and self.transforms == other.transforms

    def __hash__(self) -> int:
        return hash(self.transforms)

    def __repr__(self) -> str:
        return f"Chain({list(self.transforms)!r})"

    @property
    def name(self) -> str:
        """The step names joined by ``>``."""
        return ">".join(t.name for t in self.transforms)

    @property
    def parameters(self) -> dict[str, Any]:
        """The steps, with their own ids."""
        return {"steps": [t.describe() for t in self.transforms]}

    def fit_matrix(
        self, matrix: np.ndarray, names: tuple[str, ...]
    ) -> tuple[dict[str, Any], tuple[str, ...]] | str:
        """Fit each step on the output of the previous one."""
        states: list[dict[str, Any]] = []
        for transform in self.transforms:
            fitted = transform.fit_matrix(matrix, names)
            if isinstance(fitted, str):
                return f"{transform.name}: {fitted}"
            state, names = fitted
            states.append(state)
            matrix = transform.apply_matrix(state, matrix)
        return {"steps": states}, names

    def apply_matrix(self, state: Mapping[str, Any], matrix: np.ndarray) -> np.ndarray:
        """Apply every fitted step in order."""
        steps: Sequence[Mapping[str, Any]] = state["steps"]
        for transform, step in zip(self.transforms, steps, strict=True):
            matrix = transform.apply_matrix(step, matrix)
        return matrix
