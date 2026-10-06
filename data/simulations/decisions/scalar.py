"""Scalar features on a ``design.Layout``: the outcome generator of modules P1, P2 and P4.

Each feature is a monotone transform of a latent normal value ``L = H + R`` of variance 1:
``H`` the hierarchical part (repeat, well and field effects with variance shares
``icc_repeat``, ``icc_well``, ``icc_field``; a plate effect of SD ``plate_sd`` with
``sessions="plates"``), ``R`` the cell's residual (normal, or a scaled t with 3 degrees of
freedom for ``t3``; multiplied by ``heteroscedastic`` in the treated conditions). The
transform sets the distribution (``DISTRIBUTIONS``):

- ``normal`` ``level (1 + cv L)``; ``t3`` the same with t3 residuals; ``lognormal``
  ``level exp(s L - s^2/2)``, ``s = sqrt(log(1 + cv^2))``; ``gamma`` a lognormal
  hierarchical part times a gamma residual of CV ``cv`` (skewed, heavy right tail);
- ``zero_inflated`` counts: Poisson of a lognormal rate, a share ``zero_share`` set to 0;
- ``bounded`` a proportion ``expit(logit(0.3) + L)``;
- ``background`` (background-subtracted, about 16% negative) ``sd (L + 1)``;
  ``near_zero`` ``max(0, sd (L + 0.25))`` (a control mean near 0, many zeros);
  ``saturated`` the normal values censored at one SD above the control mean.

A condition's effect ``delta`` is in units of the total SD: with ``scale="ratio"`` the
values are multiplied by ``exp(delta s)`` (a fold change), with ``"difference"``
``delta cv level`` is added (``bounded``: ``delta`` added on the logit scale either way).
``effect_sd`` spreads the effect over repeats (a condition-by-repeat interaction),
``extreme_repeat`` adds to the last repeat's effect, ``faulty_well`` shifts one treated well
of the first repeat. Acquisition, never biology: ``gap`` multiplies the values of session
``s1`` (``Design.sessions``), ``segmentation_loss`` loses treated cells with a chance rising
with their value.

The truth (``ScalarData.truth``) is computed, not simulated: the expected value of every
condition at its mean effect, by Gauss-Hermite quadrature over ``L``, with informative
cluster size tilting the field effect for the cell-weighted and well-balanced estimands and
not for the field-balanced one; then the ``ratio`` (log of the ratio to the control) and the
``difference``. Acquisition effects (gap, segmentation loss, the faulty well, the extreme
repeat) are not in the truth: they are what a stress design shows. ``conditional`` holds,
per well, the mean of that well's population given its realised repeat, well, plate and
effect (the target of the sequential rules of P4), cell-weighted and field-balanced.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any

import numpy as np
import pandas as pd
from numpy.polynomial.hermite_e import hermegauss
from scipy.special import expit, logit
from scipy.stats import norm

from morphocoverage.metadata import CONDITION_COLUMN, REPEAT_COLUMN

from .design import Design, Layout, drop_mnar, layout, tables

DISTRIBUTIONS = (
    "normal",
    "t3",
    "lognormal",
    "gamma",
    "zero_inflated",
    "bounded",
    "background",
    "near_zero",
    "saturated",
)
SCALES = ("ratio", "difference")
ESTIMANDS = ("cell", "field", "well")
_NODES, _WEIGHTS = hermegauss(48)
_WEIGHTS = _WEIGHTS / _WEIGHTS.sum()
BOUNDED_BASE = 0.3


@dataclass(frozen=True)
class Scalar:
    """The outcome of one scalar feature (or ``features`` of them; see the module)."""

    distribution: str = "normal"
    level: float = 100.0
    cv: float = 0.3
    icc_repeat: float = 0.2
    icc_well: float = 0.05
    icc_field: float = 0.1
    effect: float = 0.0
    effects: tuple[tuple[str, float], ...] = ()
    scale: str = "ratio"
    effect_sd: float = 0.0
    extreme_repeat: float = 0.0
    faulty_well: float = 0.0
    heteroscedastic: float = 1.0
    gap: float = 1.0
    plate_sd: float = 0.0
    zero_share: float = 0.3
    segmentation_loss: float = 0.0
    features: int = 1
    nonnull_share: float = 1.0
    correlation: float = 0.0
    duplicates: int = 0

    def __post_init__(self) -> None:
        if self.distribution not in DISTRIBUTIONS:
            raise ValueError(f"distribution must be one of {DISTRIBUTIONS}")
        if self.scale not in SCALES:
            raise ValueError(f"scale must be one of {SCALES}")
        if self.icc_repeat + self.icc_well + self.icc_field >= 1:
            raise ValueError("the variance shares of the levels must sum to less than 1")

    @property
    def hierarchical(self) -> float:
        """The variance share of the repeat, well and field effects together."""
        return self.icc_repeat + self.icc_well + self.icc_field

    def effect_of(self, condition: str, control: str) -> float:
        """The mean effect of ``condition`` (0 for the control)."""
        if condition == control:
            return 0.0
        return float(dict(self.effects).get(condition, self.effect))


def scalar_from(params: dict[str, Any]) -> Scalar:
    """A ``Scalar`` from the outcome keys of ``params`` (the others ignored)."""
    names = {f.name for f in fields(Scalar)}
    given = {k: v for k, v in params.items() if k in names}
    if isinstance(given.get("effects"), (list, dict)):
        items = given["effects"].items() if isinstance(given["effects"], dict) else given["effects"]
        given["effects"] = tuple((str(c), float(v)) for c, v in items)
    return Scalar(**given)


@dataclass(frozen=True)
class ScalarData:
    """One simulated dataset: the MorphoBricks tables, the feature names, the truth by
    ``(feature, condition, estimand, scale)`` and the per-well ``conditional`` means."""

    objects: pd.DataFrame
    fovs: pd.DataFrame
    features: tuple[str, ...]
    truth: dict[tuple[str, str, str, str], float]
    conditional: pd.DataFrame
    layout: Layout
    nonnull: tuple[str, ...] = field(default_factory=tuple)


def _sigma(spec: Scalar) -> float:
    return float(np.sqrt(np.log1p(spec.cv**2)))


def _transform(
    spec: Scalar, h: np.ndarray, r: np.ndarray, delta: np.ndarray, rng: np.random.Generator | None
) -> np.ndarray:
    """The values of the distribution for hierarchical parts ``h``, residuals ``r`` and
    effects ``delta`` (see the module); ``rng`` None gives the expected value of the random
    parts that are not normal (gamma residual, Poisson, zero inflation)."""
    s, sd, level = _sigma(spec), spec.cv * spec.level, spec.level
    latent = h + r
    mult = np.exp(delta * s) if spec.scale == "ratio" else np.ones_like(delta)
    add = delta * sd if spec.scale == "difference" else np.zeros_like(delta)
    dist = spec.distribution
    if dist in ("normal", "t3"):
        return level * (1 + spec.cv * latent) * mult + add
    if dist == "lognormal":
        return level * np.exp(s * latent - s**2 / 2) * mult + add
    if dist == "gamma":
        shape = 1.0 / spec.cv**2
        g = 1.0 if rng is None else rng.gamma(shape, 1.0 / shape, size=h.shape)
        return level * np.exp(s * h - s**2 * spec.hierarchical / 2) * g * mult + add
    if dist == "zero_inflated":
        rate = np.maximum(0.0, level * np.exp(s * latent - s**2 / 2) * mult + add)
        if rng is None:
            return (1 - spec.zero_share) * rate
        return rng.poisson(rate) * (rng.random(rate.shape) >= spec.zero_share)
    if dist == "bounded":
        return expit(logit(BOUNDED_BASE) + latent + delta)
    if dist == "background":
        return sd * (latent + 1) * mult + add
    if dist == "near_zero":
        return np.maximum(0.0, sd * (latent + 0.25) * mult + add)
    # saturated: censored one SD above the control mean
    return np.minimum(level * (1 + spec.cv), level * (1 + spec.cv * latent) * mult + add)


def expected(spec: Scalar, mean: np.ndarray, variance: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """The expected value when the latent part read by the transform is normal with
    ``mean`` and ``variance`` (arrays of one shape), by Gauss-Hermite quadrature; for
    ``gamma`` that part is ``h`` alone, its residual having mean 1."""
    mean, variance = np.asarray(mean, dtype=float), np.asarray(variance, dtype=float)
    delta = np.broadcast_to(np.asarray(delta, dtype=float), mean.shape)
    points = mean[..., None] + np.sqrt(variance)[..., None] * _NODES
    zeros = np.zeros_like(points)
    values = _transform(spec, points, zeros, np.broadcast_to(delta[..., None], points.shape), None)
    return (values * _WEIGHTS).sum(axis=-1)


def _truth(
    spec: Scalar, design: Design, names: tuple[str, ...], shares: np.ndarray
) -> dict[tuple[str, str, str, str], float]:
    """The truth of every feature, condition, estimand and scale (see the module)."""
    control = design.conditions[0]
    out: dict[tuple[str, str, str, str], float] = {}
    for name, share, carries in zip(names, shares, _carriers(spec, names), strict=True):
        means: dict[tuple[str, str], float] = {}
        for condition in design.conditions:
            delta = spec.effect_of(condition, control) if carries else 0.0
            het = spec.heteroscedastic if condition != control else 1.0
            residual = (1 - spec.hierarchical) * het**2
            hier = spec.hierarchical + spec.plate_sd**2
            variance = hier if spec.distribution == "gamma" else hier + residual
            tilt = design.informative_size * np.sqrt(share * spec.icc_field)
            for estimand, shift in (("cell", tilt), ("field", 0.0), ("well", tilt)):
                means[condition, estimand] = float(expected(spec, shift, variance, delta))
        for condition in design.conditions[1:]:
            for estimand in ESTIMANDS:
                treated, base = means[condition, estimand], means[control, estimand]
                ratio = np.log(treated / base) if treated > 0 and base > 0 else np.nan
                out[name, condition, estimand, "ratio"] = float(ratio)
                out[name, condition, estimand, "difference"] = treated - base
    return out


def _carriers(spec: Scalar, names: tuple[str, ...]) -> list[bool]:
    """Which features carry the effect: the first ``nonnull_share`` of them."""
    carrying = round(spec.nonnull_share * len(names))
    return [i < carrying for i in range(len(names))]


def _level_effects(
    frame: pd.DataFrame, share: float, rng: np.random.Generator
) -> dict[str, np.ndarray]:
    """Per field, the standard normal repeat, well and field effects of one feature: the
    layout's (shared, so informative cluster size reaches it) mixed with its own at
    ``share``."""
    out = {}
    for level, key in (("z_repeat", "biological_repeat"), ("z_well", "well"), ("z_field", None)):
        codes = (
            np.arange(len(frame))
            if key is None
            else np.unique(frame[key].to_numpy(), return_inverse=True)[1]
        )
        own = rng.standard_normal(codes.max() + 1 if len(codes) else 0)[codes]
        out[level] = np.sqrt(share) * frame[level].to_numpy() + np.sqrt(1 - share) * own
    return out


def _deltas(
    spec: Scalar, frame: pd.DataFrame, control: str, rng: np.random.Generator
) -> np.ndarray:
    """Per field, the realised effect: the mean effect, its spread over repeats and the
    extreme last repeat."""
    conditions = frame[CONDITION_COLUMN].to_numpy()
    repeats = frame[REPEAT_COLUMN].to_numpy()
    base = np.array([spec.effect_of(c, control) for c in conditions])
    treated = conditions != control
    keys = pd.Series(repeats).astype(str) + "\x1f" + pd.Series(conditions).astype(str)
    codes = np.unique(keys.to_numpy(), return_inverse=True)[1]
    spread = spec.effect_sd * rng.standard_normal(codes.max() + 1 if len(codes) else 0)[codes]
    last = repeats == max(set(repeats), key=lambda r: int(str(r)[1:]))
    return np.where(treated, base + spread + spec.extreme_repeat * last, 0.0)


def generate(design: Design, spec: Scalar, rng: np.random.Generator) -> ScalarData:
    """One dataset of ``spec`` on a fresh layout of ``design`` (see the module)."""
    lay = layout(design, rng)
    frame = lay.fields
    control = design.conditions[0]
    rows = lay.cells()
    names = tuple(f"x{j}" for j in range(spec.features))
    shares = np.array([1.0] + [spec.correlation] * (spec.features - 1))
    treated = frame["treated"].to_numpy()
    plate = np.zeros(len(frame))
    if spec.plate_sd > 0:
        codes = np.unique(frame["session"].to_numpy(), return_inverse=True)[1]
        plate = spec.plate_sd * rng.standard_normal(codes.max() + 1)[codes]
    faulty = np.zeros(len(frame))
    if spec.faulty_well:
        wells = frame.loc[treated & (frame[REPEAT_COLUMN] == "R1"), "well"]
        if len(wells):
            faulty = spec.faulty_well * (frame["well"] == wells.iloc[0]).to_numpy()
    delta_field = _deltas(spec, frame, control, rng)
    carriers = _carriers(spec, names)
    values, conditional, first_latent, field_score = {}, [], None, None
    het = np.where(treated, spec.heteroscedastic, 1.0)
    for name, share, carries in zip(names, shares, carriers, strict=True):
        z = _level_effects(frame, share, rng)
        fixed = (
            np.sqrt(spec.icc_repeat) * z["z_repeat"]
            + np.sqrt(spec.icc_well) * z["z_well"]
            + plate
            + faulty
        )
        h = fixed + np.sqrt(spec.icc_field) * z["z_field"]
        if spec.distribution == "t3":
            r = rng.standard_t(3, size=len(rows)) / np.sqrt(3.0)
        else:
            r = rng.standard_normal(len(rows))
        r = r * np.sqrt(1 - spec.hierarchical) * het[rows]
        delta = delta_field if carries else np.zeros(len(frame))
        values[name] = _transform(spec, h[rows], r, delta[rows], rng)
        if spec.gap != 1.0 and spec.distribution != "bounded":
            values[name] = (
                values[name] * np.where(frame["session"].to_numpy() == "s1", spec.gap, 1)[rows]
            )
        if first_latent is None:
            first_latent, field_score = h[rows] + r, h + delta
        conditional.append(_conditional(spec, design, frame, name, share, fixed, delta, het))
    for k in range(spec.duplicates):
        noise = 0.05 * spec.cv * spec.level * rng.standard_normal(len(rows))
        values[f"x0_copy{k + 1}"] = values["x0"] + noise
    table = pd.DataFrame(values)
    keep_fields = np.ones(len(frame), dtype=bool)
    if design.missing == "mnar" and design.missing_rate > 0:
        keep_fields = drop_mnar(frame, field_score, design.missing_rate, rng)
    keep_cells = keep_fields[rows]
    if spec.segmentation_loss > 0:
        lost = rng.random(len(rows)) < spec.segmentation_loss * norm.cdf(first_latent)
        keep_cells &= ~(lost & treated[rows])
    kept = Layout(design, frame[keep_fields].reset_index(drop=True))
    objects, fovs = tables(kept, table[keep_fields[rows]], keep_cells[keep_fields[rows]])
    truth = _truth(spec, design, names, shares)
    nonnull = tuple(n for n, c in zip(names, carriers, strict=True) if c)
    cond = pd.concat(conditional, ignore_index=True) if conditional else pd.DataFrame()
    return ScalarData(objects, fovs, tuple(table.columns), truth, cond, kept, nonnull)


def _conditional(
    spec: Scalar,
    design: Design,
    frame: pd.DataFrame,
    name: str,
    share: float,
    fixed: np.ndarray,
    delta: np.ndarray,
    het: np.ndarray,
) -> pd.DataFrame:
    """Per well, the mean of its population given its realised repeat, well, plate and
    effect: over the field effect (tilted for the cell-weighted mean) and the residual."""
    first = frame.drop_duplicates("well").index.to_numpy()
    residual = 0.0 if spec.distribution == "gamma" else (1 - spec.hierarchical)
    variance = spec.icc_field + residual * het[first] ** 2
    tilt = design.informative_size * np.sqrt(share * spec.icc_field)
    cell = expected(spec, fixed[first] + tilt, variance, delta[first])
    field_mean = expected(spec, fixed[first], variance, delta[first])
    return pd.DataFrame(
        {
            "feature": name,
            CONDITION_COLUMN: frame[CONDITION_COLUMN].to_numpy()[first],
            REPEAT_COLUMN: frame[REPEAT_COLUMN].to_numpy()[first],
            "well": frame["well"].to_numpy()[first],
            "cell": cell,
            "field": field_mean,
        }
    )


__all__ = [
    "DISTRIBUTIONS",
    "ESTIMANDS",
    "SCALES",
    "Scalar",
    "ScalarData",
    "expected",
    "generate",
    "scalar_from",
]
