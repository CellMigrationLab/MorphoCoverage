"""Leave-one-FOV-out support novelty at a declared phenotype resolution.

Described in ``docs/common/architecture.md``. Are
observations still arriving in regions of phenotype space that the other
FOVs do not cover? At resolution ``epsilon`` an observation is covered when
observations from at least ``min_other_fovs`` *other* FOVs lie within
``epsilon`` (Euclidean, in the transformed feature space). Holding out each
visible FOV in turn estimates how much of a new representative FOV would be
novel:

    lofo_novel_fraction = (novel observations) / (observations), pooled over
    the visible FOVs, each judged against all the others.

This is the leave-one-FOV-out estimate of ``1 - C_epsilon``, the chance that
an observation from another FOV falls outside the current support. At the FOV
level,

    lofo_novel_fov_fraction = (FOVs showing a new state) / (visible FOVs),

estimates the probability that one more representative FOV shows something
outside the current support at this resolution. A FOV shows a **new state**
when at least ``min_novel_observations`` of its novel observations form one
cluster: they are linked by chains of steps of at most ``epsilon`` between
novel observations of that FOV (single linkage at the same resolution). So
``m`` new objects must resemble each other, and scattered outliers (debris,
segmentation errors) do not add up to a phenotype however many there are.
With ``min_novel_observations=1`` any novel observation counts. When every visible FOV
with observations holds exactly one (one cell per image), a cluster of ``m`` in one FOV
cannot occur, so the estimator, ``novel_fovs_against`` and ``novel_observation_flags`` read
the definition on fields (``per_field``): an observation is new when no other FOV holds one
within ``epsilon``, a state seen once (owner, 2026-10-06, plan v2), the same in the report,
``optimise``, ``benchmark`` and a live session. Acquired FOVs
without observations count in the denominator, since the next FOV may be
empty too.

With few FOVs that fraction is uninformative: two homogeneous FOVs give 0,
yet say little about a third. ``lofo_novel_fov_upper`` is the one-sided
Clopper-Pearson upper bound at nominal level ``confidence`` on the same
probability. The leave-one-FOV-out outcomes are not independent, so its
coverage is approximate and is checked empirically by
``morphocoverage.calibration`` (``morphocoverage.bounds``). It is 0.78 for 0
novel FOVs of 2 at a nominal 95%, and is the quantity to stop on when few
FOVs have been seen. Support is counted in
FOVs, not objects: a state seen only in one FOV stays novel however many objects
it has there, and support from another FOV counts once however many of its
objects are nearby, so a dense FOV cannot cover what a sparse FOV showing the
same state would not.

Support mirrors the new-state rule: another FOV supports an observation
only with at least ``min_novel_observations`` of its own observations within
``epsilon`` of it. A state must recur as a group of similar objects to be
covered, so a single stray object (debris, a mis-segmented object) in another
FOV cannot cancel a real new state, just as scattered outliers cannot make
one. The two rules are not symmetric, though: support counts objects in a
ball of radius ``epsilon`` around the observation, while a new state is a
single-linkage chain, which can reach far beyond ``epsilon``. A sparse state
strung out at almost ``epsilon`` spacing can therefore form a new state in
each of two FOVs while no object of either has ``min_novel_observations``
objects of the other within ``epsilon``, and it is counted new in both
although it recurs (five objects 0.9 apart at ``epsilon = 1`` with
``min_novel_observations = 3``, repeated half a step along in a second FOV).
With ``min_novel_observations=1`` one nearby observation is enough, and the
two rules agree.

Single linkage can chain: unsupported objects spaced at most ``epsilon``
apart form one state however far the chain extends. A continuous run of
unseen objects is treated as one new state.

Weighting of the held-out side differs between the quantities:

- ``lofo_novel_fraction`` pools observations, so each FOV counts in
  proportion to its objects: two dense, mutually supporting FOVs of 1,000
  objects and eight one-object FOVs each in its own unsupported state give
  8 / 2,008 = 0.004, although 8 of 10 FOVs are wholly novel. It estimates
  the chance that a random *object* of a new FOV is novel, not a
  representative-FOV quantity, and a dense homogeneous FOV can make it
  look settled.
- ``lofo_mean_fov_novel_fraction`` averages the novel fraction per FOV over
  the FOVs with observations (0.8 above): each FOV counts once.
- ``lofo_novel_fov_fraction`` and its bound count FOVs showing a new state
  (each FOV once, empty FOVs in the denominator), the quantity for "would
  one more FOV show something new?".

Recurrence across FOVs is evidence that a state is common only when the FOVs
are representative, exchangeable draws. Adjacent, spatially correlated FOVs
showing the same local patch support each other with ``min_other_fovs=1``
although they may be one occurrence; raise ``min_other_fovs`` or use FOVs far
enough apart when neighbours are correlated.

Choices and limits:

- The estimate uses ``k - 1`` FOVs for each held-out FOV, so it slightly
  overstates the novelty of the full visible set.
- It is resolution-dependent by construction: ``epsilon`` and
  ``min_other_fovs`` define the estimand and have no defaults. Density varies
  across phenotype space, so one global ``epsilon`` is coarse where
  phenotypes are dense and fine where they are sparse.
- In high dimensions nearest-neighbour distances concentrate, and a single
  ``epsilon`` either covers everything or nothing; use a transform that
  reduces the dimension.
- With a ``transform``, each held-out FOV is judged in a geometry fitted on
  the other FOVs only (a nested holdout), as a genuinely new FOV is by
  ``novel_fovs_against``. Fitting once on all visible FOVs would let an
  extreme FOV widen the standardization that judges it: three FOVs near 0
  and one at 100 put it about 2 SD away instead of about 1,000. The
  diagnostics record the fit on all visible FOVs and the number of folds.
  A mean-and-SD fit is still widened by extreme objects in other FOVs (20
  debris objects at +60 SD in two FOVs hid a 15-object state at +6 SD in
  every run); ``transforms.RobustStandardize`` (median and MAD) is not.
- It only sees states that have appeared: a phenotype absent from every
  visible FOV contributes nothing.
- Every held-out FOV needs ``min_other_fovs`` other FOVs with observations to
  be supported at all, so the quantities are insufficient data below
  ``max(min_fovs, min_other_fovs + 1)`` FOVs with observations (else every
  observation would be novel by construction).
- A feature that varies in only one FOV (a count that is 0 everywhere else)
  cannot be standardized without that FOV, so its fold, and the estimate,
  are insufficient data; the reason names the feature and the FOV. Leave such
  a feature out, or use no transform.
- Zero-observation FOVs add no observations to judge; they still count as
  acquisition effort.

The counting lives in ``support_geometry``. Cost: ``scipy.spatial.cKDTree``
queries. With ``min_other_fovs=1`` and ``min_novel_observations=1``, counting
ball queries (about 10 ms for a 1,000-object well); otherwise a dense distance
matrix for small groups, or bounded nearest-neighbour queries per FOV (about
0.5 s for 6,700 objects in 168 FOVs).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, TypeVar

import numpy as np

from .._checks import is_integer
from .._wording import listing
from ..bounds import check_confidence, upper_bound
from ..evidence import Component, Direction, Evidence, Quantity
from ..objectives import ObjectiveConfig
from ..policies import ThresholdPolicy
from ..schema import ValidatedInput, fov_positions
from ..transforms import FeatureTransform, constant_columns, missing_values_reason
from .base import Estimator, SamplingStrategy
from .support_geometry import new_state_members, novel_against, novel_observations

QUANTITY = "lofo_novel_fraction"
FOV_QUANTITY = "lofo_novel_fov_fraction"
UPPER_QUANTITY = "lofo_novel_fov_upper"
MEAN_FOV_QUANTITY = "lofo_mean_fov_novel_fraction"
QUANTITIES = (QUANTITY, MEAN_FOV_QUANTITY, FOV_QUANTITY, UPPER_QUANTITY)
_UNITS = {
    QUANTITY: "fraction of observations",
    MEAN_FOV_QUANTITY: "mean fraction of observations per FOV",
    FOV_QUANTITY: "fraction of FOVs (probability)",
    UPPER_QUANTITY: "probability (upper confidence bound)",
}


def _check(
    epsilon: float, min_other_fovs: int, min_fovs: int, transform: FeatureTransform | None
) -> None:
    """Validate the settings shared by the estimator and the strategy."""
    if isinstance(epsilon, bool) or not (float(epsilon) > 0 and np.isfinite(epsilon)):
        raise ValueError("epsilon must be a positive, finite distance")
    for value, label, minimum in (
        (min_other_fovs, "min_other_fovs", 1),
        (min_fovs, "min_fovs", 2),
    ):
        if not is_integer(value):
            raise TypeError(f"{label} must be an integer")
        if value < minimum:
            raise ValueError(f"{label} must be at least {minimum}")
    if transform is not None and not isinstance(transform, FeatureTransform):
        raise TypeError("transform must be a FeatureTransform or None (raw feature units)")


_Novelty = TypeVar("_Novelty", "SupportNoveltyEstimator", "SupportNoveltyStrategy")


def per_field(novelty: _Novelty) -> _Novelty:
    """The definition of new on fields of one observation each (owner, 2026-10-06, plan v2):
    a new state's ``min_novel_observations`` similar observations cannot sit in one field, so
    an observation is new when no other field holds one within ``epsilon`` (a state seen
    once): ``min_other_fovs`` 1 and ``min_novel_observations`` 1, the radius, transform and
    confidence kept. Unchanged when the definition already needs one observation."""
    if novelty.min_novel_observations <= 1:
        return novelty
    fewest = max(novelty.min_fovs, 2)
    return replace(novelty, min_other_fovs=1, min_novel_observations=1, min_fovs=fewest)


def one_per_field(data: ValidatedInput) -> bool:
    """Whether every field of ``data`` with observations holds exactly one."""
    positions = fov_positions(data, data.fovs)
    counts = np.bincount(positions[positions >= 0])
    return bool(len(counts)) and int(counts.max()) == 1


def _check_novel(min_novel_observations: int) -> None:
    """A positive integer number of novel observations per FOV."""
    value = min_novel_observations
    if not is_integer(value) or value < 1:
        raise ValueError("min_novel_observations must be a positive integer")


def _as_ints(settings: object, *names: str) -> None:
    """Store numpy integer settings of a frozen dataclass as Python ints (JSON provenance)."""
    for name in names:
        object.__setattr__(settings, name, int(getattr(settings, name)))


def _fov_names(data: ValidatedInput) -> list[str]:
    """Each FOV's identifier, by position in ``data.fovs``, for messages."""
    return [str(value) for value in data.fovs[data.fov_key[-1]].tolist()]


def _fold_failure(
    matrix: np.ndarray, own: np.ndarray, names: tuple[str, ...], fov: str, reason: str
) -> str:
    """Why a leave-one-FOV-out fold could not be fitted, naming a feature varying only there."""
    only_here = constant_columns(matrix[~own]) & ~constant_columns(matrix)
    varying = [f"{name!r}" for name, flag in zip(names, only_here, strict=True) if flag]
    failed = "the transform could not be fitted without one FOV"
    if not varying:
        return f"{failed} ({fov}): {reason}"
    if len(varying) == 1:
        subject = f"feature {varying[0]} varies"
    else:
        subject = f"features {listing(varying)} vary"
    return (
        f"{failed}: {subject} only in FOV {fov}, and the other FOVs alone give {reason}; "
        "leave the feature out or use no transform"
    )


def _meta(name: str) -> dict[str, Any]:
    """Direction, units and component of a novelty quantity.

    Whether a field shows a state no other field supports is a property of
    the distribution of phenotypes, not of its summary: distribution evidence.
    """
    return {
        "direction": Direction.LOWER_IS_BETTER,
        "units": _UNITS[name],
        "component": Component.DISTRIBUTION,
    }


def _unavailable(kind: str, reason: str, diagnostics: dict[str, Any]) -> Evidence:
    """Every quantity insufficient or not applicable, with the reason."""
    make = Quantity.insufficient if kind == "insufficient" else Quantity.not_applicable
    quantities = {name: make(reason, **_meta(name)) for name in QUANTITIES}
    return Evidence(quantities, {**diagnostics, "undefined_reason": reason})


@dataclass(frozen=True)
class SupportNoveltyEstimator(Estimator):
    """Leave-one-FOV-out fraction of observations outside the other FOVs' ``epsilon``-support.

    Reports ``lofo_novel_fraction`` (object-weighted),
    ``lofo_mean_fov_novel_fraction`` (FOV-weighted), ``lofo_novel_fov_fraction``
    and its upper bound (distribution evidence, lower is better). See the module
    docstring for the estimands, their weighting and their limits.
    """

    epsilon: float
    min_other_fovs: int
    transform: FeatureTransform | None
    min_fovs: int
    min_novel_observations: int = 1
    confidence: float = 0.95
    name: str = "support_novelty"

    def __post_init__(self) -> None:
        _check(self.epsilon, self.min_other_fovs, self.min_fovs, self.transform)
        _check_novel(self.min_novel_observations)
        _as_ints(self, "min_other_fovs", "min_fovs", "min_novel_observations")
        object.__setattr__(self, "epsilon", float(self.epsilon))
        object.__setattr__(self, "confidence", check_confidence(self.confidence))

    @property
    def _minimum_fovs(self) -> int:
        """``max(min_fovs, min_other_fovs + 1)``: a held-out FOV needs ``min_other_fovs`` others."""
        return max(self.min_fovs, self.min_other_fovs + 1)

    @property
    def parameters(self) -> dict[str, Any]:
        """The resolution, neighbour count, transform and minimum FOVs."""
        return {
            "epsilon": self.epsilon,
            "min_other_fovs": self.min_other_fovs,
            "transform": None if self.transform is None else self.transform.describe(),
            "min_fovs": self.min_fovs,
            "min_novel_observations": self.min_novel_observations,
            "confidence": self.confidence,
        }

    def estimate(self, data: ValidatedInput, *, objective: ObjectiveConfig) -> Evidence:
        """The leave-one-FOV-out novel fractions of observations and of FOVs; on fields of
        one observation each, by ``per_field``'s definition, as every workflow judges it."""
        if self.min_novel_observations > 1 and one_per_field(data):
            return per_field(self).estimate(data, objective=objective)
        matrix = data.feature_matrix()
        missing = missing_values_reason(matrix, data.feature_cols)
        if missing:
            return _unavailable("not_applicable", missing, {})
        positions = fov_positions(data, data.fovs)
        diagnostics: dict[str, Any] = {"fovs_with_observations": len(np.unique(positions))}
        reason = None
        if self.transform is not None:
            # Fitted on all visible FOVs for provenance and the fit check only:
            # each leave-one-FOV-out fold refits it on the other FOVs.
            fitted = self.transform.fit(data)
            diagnostics["transform"] = fitted.provenance()
            if not fitted.fitted:
                reason = f"the transform could not be fitted: {fitted.reason}"
        minimum = self._minimum_fovs
        if diagnostics["fovs_with_observations"] < minimum:
            reason = f"fewer than {minimum} visible FOVs with observations"
            if minimum > self.min_fovs:
                reason += (
                    f" (an observation needs {self.min_other_fovs} other FOVs to be supported, "
                    f"min_other_fovs={self.min_other_fovs})"
                )
        if reason is not None:
            return _unavailable("insufficient", reason, diagnostics)

        lofo = self._lofo(matrix, positions, tuple(data.feature_cols), _fov_names(data))
        if isinstance(lofo, str):
            return _unavailable("insufficient", lofo, diagnostics)
        novel, members = lofo
        if self.transform is not None:
            diagnostics["transform_folds"] = len(np.unique(positions))
        k = len(data.fovs)
        novel_per_fov = np.bincount(positions, weights=novel, minlength=k)
        counts = np.bincount(positions, minlength=k)
        per_fov = novel_per_fov / np.maximum(counts, 1)
        observed = counts > 0
        novel_fov_count = int(np.unique(positions[members]).size)
        upper = upper_bound(novel_fov_count, k, self.confidence)
        diagnostics.update(
            {
                "novel_observations": int(novel.sum()),
                "novel_fovs": novel_fov_count,
                "fovs_with_novel_observations": int(np.unique(positions[novel]).size),
                "mean_fov_novel_fraction": float(per_fov[observed].mean()),
            }
        )
        quantities = {
            QUANTITY: Quantity.available(float(novel.mean()), **_meta(QUANTITY)),
            MEAN_FOV_QUANTITY: Quantity.available(
                float(per_fov[observed].mean()), **_meta(MEAN_FOV_QUANTITY)
            ),
            FOV_QUANTITY: Quantity.available(novel_fov_count / k, **_meta(FOV_QUANTITY)),
            UPPER_QUANTITY: Quantity.available(upper, **_meta(UPPER_QUANTITY)),
        }
        return Evidence(quantities, diagnostics)

    def novel_fovs_against(self, visible: ValidatedInput, new: ValidatedInput) -> np.ndarray | None:
        """Whether each FOV of ``new`` would count as novel after the FOVs of ``visible``.

        The same definition the estimate uses, applied to FOVs not yet seen: a
        FOV of ``new`` is novel when at least ``min_novel_observations`` of its
        observations have fewer than ``min_other_fovs`` FOVs of ``visible``
        within ``epsilon`` and form one cluster at ``epsilon`` (a new state).
        The transform is fitted on ``visible`` only. FOVs
        without observations are not novel. Returns ``None`` for missing values
        in ``visible`` or ``new``, too few FOVs (fewer than ``min_fovs``, or
        than ``min_other_fovs``, with observations in ``visible``), or a
        transform that cannot be fitted on ``visible``. It does not refit the
        leave-one-FOV-out folds, so it can return flags where ``estimate`` on
        ``visible`` is insufficient because one fold could not be fitted;
        callers check ``estimate`` first. Both inputs must have the same feature
        columns in the same order, the coordinates the distances are taken in.
        """
        if self.min_novel_observations > 1 and one_per_field(visible):
            return per_field(self).novel_fovs_against(visible, new)
        if tuple(new.feature_cols) != tuple(visible.feature_cols):
            raise ValueError(
                f"new has feature columns {list(new.feature_cols)}, visible "
                f"{list(visible.feature_cols)}: novelty needs the same columns in the same order"
            )
        reference = visible.feature_matrix()
        queries = new.feature_matrix()
        if np.isnan(reference).any() or np.isnan(queries).any():
            return None
        positions = fov_positions(visible, visible.fovs)
        # Every visible FOV is another FOV for a new one.
        if len(np.unique(positions)) < max(self.min_fovs, self.min_other_fovs):
            return None
        if self.transform is not None:
            fitted = self.transform.fit(visible)
            if not fitted.fitted:
                return None
            reference, queries = fitted.apply(visible), fitted.apply(new)
        flags = novel_against(
            reference,
            positions,
            queries,
            self.epsilon,
            self.min_other_fovs,
            self.min_novel_observations,
        )
        new_positions = fov_positions(new, new.fovs)
        members = new_state_members(
            queries, new_positions, flags, self.epsilon, self.min_novel_observations
        )
        return np.isin(np.arange(len(new.fovs)), new_positions[members])

    def novel_observation_flags(
        self, data: ValidatedInput, *, new_states_only: bool = False
    ) -> np.ndarray | None:
        """Leave-one-FOV-out novelty of each observation of ``data``, or ``None``.

        The per-observation flags behind the estimate, for inspecting which
        objects were counted as new; ``None`` when the estimate is unavailable.
        With ``new_states_only``, only the novel observations that form a new
        state (a cluster of at least ``min_novel_observations``) are flagged. On fields of
        one observation each, by ``per_field``'s definition.
        """
        if self.min_novel_observations > 1 and one_per_field(data):
            return per_field(self).novel_observation_flags(data, new_states_only=new_states_only)
        matrix = data.feature_matrix()
        positions = fov_positions(data, data.fovs)
        minimum = self._minimum_fovs
        if np.isnan(matrix).any() or len(np.unique(positions)) < minimum:
            return None
        if self.transform is not None and not self.transform.fit(data).fitted:
            return None
        lofo = self._lofo(matrix, positions, tuple(data.feature_cols), _fov_names(data))
        if isinstance(lofo, str):
            return None
        novel, members = lofo
        return members if new_states_only else novel

    def new_state_fields(
        self, matrix: np.ndarray, positions: np.ndarray, names: tuple[str, ...]
    ) -> np.ndarray | None:
        """The fields (``positions`` values) of ``matrix`` holding a new state, judged as
        ``novel_observation_flags`` judges them, or ``None`` when not estimable: for a
        simulation of the definition itself (``novelty_null``)."""
        counts = np.bincount(positions)
        if self.min_novel_observations > 1 and int(counts.max()) == 1:
            return per_field(self).new_state_fields(matrix, positions, names)
        if len(np.unique(positions)) < self._minimum_fovs:
            return None
        lofo = self._lofo(matrix, positions, names, [str(p) for p in range(len(counts))])
        return None if isinstance(lofo, str) else np.unique(positions[lofo[1]])

    def _lofo(
        self,
        matrix: np.ndarray,
        positions: np.ndarray,
        names: tuple[str, ...],
        fov_names: list[str],
    ) -> tuple[np.ndarray, np.ndarray] | str:
        """Leave-one-FOV-out novel observations and new-state members, or why not.

        Without a transform, the raw features serve every fold. With
        one, each held-out FOV is judged in a geometry fitted on the other FOVs
        only, as a genuinely new FOV would be (``novel_fovs_against``): an extreme
        FOV must not widen the standardization or rotate the PCA that decides
        whether it is extreme. ``fov_names`` name the FOVs (by position) when a
        fold cannot be fitted.
        """
        m = self.min_novel_observations
        if self.transform is None:
            novel = novel_observations(matrix, positions, self.epsilon, self.min_other_fovs, m)
            return novel, new_state_members(matrix, positions, novel, self.epsilon, m)
        novel = np.zeros(len(matrix), dtype=bool)
        members = np.zeros(len(matrix), dtype=bool)
        for fov in np.unique(positions):
            own = positions == fov
            fitted = self.transform.fit_matrix(matrix[~own], names)
            if isinstance(fitted, str):
                return _fold_failure(matrix, own, names, fov_names[fov], fitted)
            state = fitted[0]
            reference = self.transform.apply_matrix(state, matrix[~own])
            queries = self.transform.apply_matrix(state, matrix[own])
            flags = novel_against(
                reference, positions[~own], queries, self.epsilon, self.min_other_fovs, m
            )
            novel[own] = flags
            members[own] = new_state_members(
                queries, np.zeros(len(queries), dtype=np.int64), flags, self.epsilon, m
            )
        return novel, members


@dataclass(frozen=True)
class SupportNoveltyStrategy(SamplingStrategy):
    """Stop when a leave-one-FOV-out novel fraction is at most ``tolerance``.

    ``SupportNoveltyEstimator`` plus ``ThresholdPolicy`` on ``quantity``: by
    default the upper bound on the chance that one more FOV shows a new state
    (``lofo_novel_fov_upper``, as ``Setup.novelty_rule`` uses), or the
    fraction of FOVs showing one, the FOV-weighted mean fraction or the
    object-weighted fraction of novel observations (``lofo_novel_fraction``,
    the default until audit 4: two dense homogeneous FOVs make it look
    settled while sparse FOVs each show a state of their own). The
    resolution, support and minimum FOVs have no defaults. A tolerance of 0 is
    refused for the upper bound, which is positive for any number of FOVs.
    The bound holds at a number of FOVs fixed in advance: stopping at the
    first FOV where it passes is premature more often than its level says.
    """

    epsilon: float
    tolerance: float
    min_other_fovs: int
    transform: FeatureTransform | None
    min_fovs: int
    min_novel_observations: int = 1
    quantity: str = UPPER_QUANTITY
    confidence: float = 0.95
    name: str = "support_novelty"

    def __post_init__(self) -> None:
        _check(self.epsilon, self.min_other_fovs, self.min_fovs, self.transform)
        _check_novel(self.min_novel_observations)
        _as_ints(self, "min_other_fovs", "min_fovs", "min_novel_observations")
        if self.quantity not in QUANTITIES:
            raise ValueError(f"quantity must be one of {QUANTITIES}")
        check_confidence(self.confidence)
        if not 0 <= self.tolerance < 1:
            raise ValueError("tolerance must be a fraction in [0, 1)")
        if self.quantity == UPPER_QUANTITY and self.tolerance == 0:
            raise ValueError(
                f"tolerance 0 on {UPPER_QUANTITY} is never reached: the upper confidence "
                "bound stays above 0 however many FOVs are seen; choose a positive tolerance"
            )

    @property
    def parameters(self) -> dict[str, Any]:
        """The estimator's settings and the tolerance."""
        return {
            **self.estimators[0].parameters,
            "quantity": self.quantity,
            "tolerance": self.tolerance,
        }

    @property
    def estimators(self) -> tuple[SupportNoveltyEstimator]:
        """The support-novelty estimator with this strategy's settings."""
        return (
            SupportNoveltyEstimator(
                self.epsilon,
                self.min_other_fovs,
                self.transform,
                self.min_fovs,
                self.min_novel_observations,
                self.confidence,
            ),
        )

    @property
    def policies(self) -> tuple[ThresholdPolicy]:
        """``quantity <= tolerance``."""
        return (ThresholdPolicy(self.quantity, self.tolerance),)


__all__ = [
    "SupportNoveltyEstimator",
    "SupportNoveltyStrategy",
    "new_state_members",
    "novel_against",
    "novel_observations",
    "one_per_field",
    "per_field",
]
