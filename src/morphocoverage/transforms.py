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
than guess a scale. A feature counts as constant when it varies by less than
``CONSTANT_RELATIVE_SPREAD`` (1e-7) of its magnitude, the rounding of values
stored in single precision: rounding noise would otherwise be standardised to
full weight and decide the geometry (``constant_columns``).

``RobustStandardize`` centres on the median and scales by the MAD, so a few
extreme objects (clumps, mis-segmented debris) cannot widen the scale and
hide a genuinely new state elsewhere, as they widen a mean and SD.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ._checks import is_integer
from ._frozen import owned
from ._robust_sd import CONSTANT_RELATIVE_SPREAD, constant_columns
from ._wording import listing
from .provenance import class_identity, config_hash, describe
from .schema import ValidatedInput
from .statistics import MAD_TO_SD


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
        return describe(self)

    def output_width(self, n_inputs: int) -> int | None:
        """How many columns the transform gives for ``n_inputs``, known without fitting.

        Lets an estimator tell a summary these features can never define (a PCA
        summary of one standardized feature) from one the transform cannot fit
        yet. ``None`` when unknown, or when no fit can ever succeed.
        """
        return None

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


@owned("state")
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

    def output_width(self, n_inputs: int) -> int:
        """One output per feature."""
        return n_inputs

    def fit_matrix(
        self, matrix: np.ndarray, names: tuple[str, ...]
    ) -> tuple[dict[str, Any], tuple[str, ...]] | str:
        """Visible mean and SD per feature."""
        missing = missing_values_reason(matrix, names)
        if missing:
            return f"{missing}; choose or implement an explicit missing-data policy"
        if len(matrix) < 2:
            return "fewer than 2 observations"
        mean = matrix.mean(axis=0)
        sd = matrix.std(axis=0, ddof=1)
        constant = _constant_reason(matrix, names)
        if constant:
            return constant
        return {"mean": _floats(mean), "sd": _floats(sd)}, names

    def apply_matrix(self, state: Mapping[str, Any], matrix: np.ndarray) -> np.ndarray:
        """``(x - mean) / sd``."""
        return (matrix - np.asarray(state["mean"])) / np.asarray(state["sd"])


def missing_values_reason(matrix: np.ndarray, names: Sequence[str]) -> str | None:
    """Which features hold missing values, and in how many observations, or None:
    ``missing values in linearity_mean (1 observation) and entropy_mean (3)``."""
    counts = np.isnan(matrix).sum(axis=0)
    missing = [(name, int(n)) for name, n in zip(names, counts, strict=True) if n]
    if not missing:
        return None
    first, *others = missing
    named = [f"{first[0]} ({first[1]} observation{'' if first[1] == 1 else 's'})"]
    named += [f"{name} ({n})" for name, n in others]
    return "missing values in " + listing(named)


def _constant_reason(matrix: np.ndarray, names: tuple[str, ...]) -> str | None:
    """Why the features cannot be standardised (the constant ones, named), or None."""
    flags = constant_columns(matrix)
    constant = [name for name, flag in zip(names, flags, strict=True) if flag]
    if not constant:
        return None
    return (
        f"zero SD on the visible data for {constant} (constant, or varying by at most "
        f"{CONSTANT_RELATIVE_SPREAD:g} of the value: rounding noise); leave "
        f"{'it' if len(constant) == 1 else 'them'} out of the features"
    )


@dataclass(frozen=True)
class RobustStandardize(FeatureTransform):
    """Centre each feature on its visible median and divide by its visible MAD (as an SD).

    The scale is ``1.4826 * median(|x - median(x)|)``, which estimates the SD
    of normal data, so a distance means about what it means after
    ``Standardize``; but a few extreme objects (clumps, mis-segmented debris,
    well under half of them) move neither the centre nor the scale, whereas
    they widen an SD and pull every other object, a genuinely new state
    included, towards the rest. A feature whose MAD is 0 although it varies
    (more than half of its values equal, as for counts) is scaled by its SD
    instead; the fitted state records which features were
    (``sd_scaled``). Not fitted with fewer than 2 observations, missing
    values, or a feature that is constant (``constant_columns``).
    """

    name: str = "robust_standardize"

    def output_width(self, n_inputs: int) -> int:
        """One output per feature."""
        return n_inputs

    def fit_matrix(
        self, matrix: np.ndarray, names: tuple[str, ...]
    ) -> tuple[dict[str, Any], tuple[str, ...]] | str:
        """Visible median and MAD-based scale per feature (the SD where the MAD is 0)."""
        missing = missing_values_reason(matrix, names)
        if missing:
            return f"{missing}; choose or implement an explicit missing-data policy"
        if len(matrix) < 2:
            return "fewer than 2 observations"
        constant = _constant_reason(matrix, names)
        if constant:
            return constant
        median = np.median(matrix, axis=0)
        scale = MAD_TO_SD * np.median(np.abs(matrix - median), axis=0)
        sd = matrix.std(axis=0, ddof=1)
        # A MAD at rounding level is no scale: such a feature is scaled by its SD.
        no_mad = scale <= CONSTANT_RELATIVE_SPREAD * np.maximum(np.abs(median), sd)
        scale = np.where(no_mad, sd, scale)
        state = {
            "median": _floats(median),
            "scale": _floats(scale),
            "sd_scaled": [name for name, flag in zip(names, no_mad, strict=True) if flag],
        }
        return state, names

    def apply_matrix(self, state: Mapping[str, Any], matrix: np.ndarray) -> np.ndarray:
        """``(x - median) / scale``."""
        return (matrix - np.asarray(state["median"])) / np.asarray(state["scale"])


@dataclass(frozen=True)
class PCA(FeatureTransform):
    """Project the centred visible features onto their first ``n_components`` principal axes.

    Components come from the SVD of the centred visible matrix. Each
    component's sign is fixed so that its largest-magnitude loading is
    positive, so the projection is deterministic. Not fitted unless there are
    more observations than components, at least as many features as components,
    and the centred features span at least ``n_components`` dimensions (a
    component beyond their rank would be an arbitrary direction of rounding
    noise). The rank is numerical, as ``numpy.linalg.matrix_rank`` counts it: a
    singular value is rounding noise below ``max(n, d) * eps`` of the largest
    one, or of the features' magnitude (centring a feature of mean ``mu``
    leaves rounding of about ``eps * |mu|`` in every value). Features on very
    different scales therefore keep their components; only a dimension at
    floating-point rounding level is refused.
    The explained-variance ratios are recorded in the fitted state.
    """

    n_components: int

    def __post_init__(self) -> None:
        if not is_integer(self.n_components):
            raise TypeError("n_components must be an integer")
        if self.n_components < 1:
            raise ValueError("n_components must be at least 1")
        # A numpy integer is stored as a Python int: the same name and config_id.
        object.__setattr__(self, "n_components", int(self.n_components))

    @property
    def name(self) -> str:
        """``pca<n_components>``."""
        return f"pca{self.n_components}"

    @property
    def parameters(self) -> dict[str, Any]:
        """The number of components."""
        return {"n_components": self.n_components}

    def output_width(self, n_inputs: int) -> int | None:
        """``n_components``, or ``None`` when fewer features can never give them."""
        return self.n_components if n_inputs >= self.n_components else None

    def fit_matrix(
        self, matrix: np.ndarray, names: tuple[str, ...]
    ) -> tuple[dict[str, Any], tuple[str, ...]] | str:
        """Visible mean, leading components and their explained-variance ratios."""
        missing = missing_values_reason(matrix, names)
        if missing:
            return f"{missing}; choose or implement an explicit missing-data policy"
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
        magnitude = max(float(singular[0]), math.sqrt(n_observations) * float(np.abs(mean).max()))
        rounding = max(n_observations, n_features) * np.finfo(float).eps * magnitude
        if singular[self.n_components - 1] <= rounding:
            return (
                f"the visible features span fewer than {self.n_components} dimensions "
                "(collinear features, up to floating-point rounding); fewer components or "
                "other features are needed"
            )
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


@dataclass(frozen=True)
class ColumnWeights(FeatureTransform):
    """Multiply each feature by a fixed, declared weight: nothing is fitted.

    Fixed weights, applied after a standardisation:
    ``Chain([RobustStandardize(), ColumnWeights(weights)])``. Not fitted when the number of
    features is not the number of weights.
    """

    weights: tuple[float, ...]

    def __post_init__(self) -> None:
        values = tuple(float(w) for w in self.weights)
        if not values or not all(np.isfinite(w) and w > 0 for w in values):
            raise ValueError("weights must be one positive finite number per feature")
        object.__setattr__(self, "weights", values)

    @property
    def name(self) -> str:
        """``column_weights``."""
        return "column_weights"

    @property
    def parameters(self) -> dict[str, Any]:
        """The weights, in the order of the features."""
        return {"weights": list(self.weights)}

    def output_width(self, n_inputs: int) -> int | None:
        """One output per feature, when there is one weight per feature."""
        return n_inputs if n_inputs == len(self.weights) else None

    def fit_matrix(
        self, matrix: np.ndarray, names: tuple[str, ...]
    ) -> tuple[dict[str, Any], tuple[str, ...]] | str:
        """Nothing to fit: only the number of features is checked."""
        if matrix.shape[1] != len(self.weights):
            return f"{matrix.shape[1]} features for {len(self.weights)} weights"
        return {}, names

    def apply_matrix(self, state: Mapping[str, Any], matrix: np.ndarray) -> np.ndarray:
        """Each feature times its weight."""
        return matrix * np.asarray(self.weights)


@dataclass(frozen=True)
class BlockWeights(FeatureTransform):
    """Weigh each block of features (each explainable feature table) as multiple factor
    analysis does, fitted on the visible observations (owner, 2026-10-06): the features of a
    block are multiplied by ``sqrt(n - 1)`` over its first singular value (each block centred
    on its visible means), so each block's first axis has variance 1 whatever the order of the
    blocks and a table of many correlated columns does not outweigh one of few
    (``feature_tables.mfa_weights``; audit 8). Applied after a standardisation:
    ``Chain([RobustStandardize(), BlockWeights(blocks)])``. A feature of no block keeps weight
    1. Not fitted with fewer than 2 observations or missing values.

    ``inertia_radius=True`` (the report's novelty) also scales the weighted space by
    ``sqrt(d / I)``, ``d`` the features and ``I`` the weighted space's total inertia (the sum
    of its columns' variances), so a radius declared for ``d`` standardised features,
    ``sqrt(d)``, reads as ``sqrt(I)`` in the weighted space (owner, 2026-10-06, audit 8).
    """

    blocks: tuple[tuple[str, ...], ...]
    inertia_radius: bool = False

    def __post_init__(self) -> None:
        blocks = tuple(tuple(str(c) for c in block) for block in self.blocks)
        if not blocks or any(not block for block in blocks):
            raise ValueError("blocks must be one or more non-empty tuples of feature names")
        object.__setattr__(self, "blocks", blocks)

    @property
    def name(self) -> str:
        """``block_weights``."""
        return "block_weights"

    @property
    def parameters(self) -> dict[str, Any]:
        """The blocks, each its feature names, and whether the radius follows the inertia."""
        return {
            "blocks": [list(block) for block in self.blocks],
            "inertia_radius": self.inertia_radius,
        }

    def output_width(self, n_inputs: int) -> int:
        """One output per feature."""
        return n_inputs

    def fit_matrix(
        self, matrix: np.ndarray, names: tuple[str, ...]
    ) -> tuple[dict[str, Any], tuple[str, ...]] | str:
        """Each feature's weight on the visible observations."""
        from .feature_tables import mfa_weights  # feature_tables reads the schema

        missing = missing_values_reason(matrix, names)
        if missing:
            return f"{missing}; choose or implement an explicit missing-data policy"
        if len(matrix) < 2:
            return "fewer than 2 observations"
        blocks = {str(i): block for i, block in enumerate(self.blocks)}
        weights = mfa_weights(matrix, [str(n) for n in names], blocks)
        if self.inertia_radius:
            inertia = float((np.var(matrix, axis=0, ddof=1) * np.square(weights)).sum())
            if inertia > 0:
                weights = weights * np.sqrt(matrix.shape[1] / inertia)
        return {"weights": _floats(weights)}, names

    def apply_matrix(self, state: Mapping[str, Any], matrix: np.ndarray) -> np.ndarray:
        """Each feature times its fitted weight."""
        return matrix * np.asarray(state["weights"])


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

    def output_width(self, n_inputs: int) -> int | None:
        """The last step's width, each step taking the previous one's; ``None`` if any is."""
        width: int | None = n_inputs
        for transform in self.transforms:
            width = transform.output_width(width)
            if width is None:
                return None
        return width

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


__all__ = [
    "PCA",
    "BlockWeights",
    "Chain",
    "ColumnWeights",
    "FeatureTransform",
    "FittedTransform",
    "RobustStandardize",
    "Standardize",
    "constant_columns",
    "missing_values_reason",
]
