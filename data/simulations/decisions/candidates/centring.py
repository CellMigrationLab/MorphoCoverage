"""Embedding centring per biological repeat, P3's second question: remove the repeat's batch
and keep the biology (maps, populations; never a test).

Each function returns the values with a vector subtracted per repeat, except the
location-scale references, which also rescale:

- ``coordinate_median``: today's package rule (``workflows._feature_space``): the median, over
  the conditions in every repeat, of their per-condition coordinatewise medians. Not
  rotation-equivariant (a coordinatewise median is not);
- ``centroid``: the equal-weighted mean of those conditions' centroids (equivariant);
- ``geometric_median``: the geometric median of those conditions' cells, each condition
  weighing the same (Weiszfeld's algorithm; equivariant);
- ``vector_model``: the additive model ``y_rc = alpha_c + b_r`` fitted by least squares on
  every condition centroid of every repeat, ``b_r`` subtracted (equivariant; handles a
  connected incomplete design);
- ``location_scale``: each coordinate centred and scaled per repeat (ComBat without its
  empirical Bayes shrinkage; a coordinatewise reference, not equivariant);
- ``harmony``: Harmony (``harmonypy``, the ``validation`` extra; GPL-3.0, used only here),
  a limited reference.
"""

from __future__ import annotations

import numpy as np

from .fixed_block import Layout


def _shared(condition: np.ndarray, repeat: np.ndarray) -> list[str]:
    """The conditions held in every repeat."""
    held = [set(condition[repeat == r]) for r in np.unique(repeat)]
    return sorted(set.intersection(*held)) if held else []


def _subtract(values: np.ndarray, repeat: np.ndarray, centres: dict) -> np.ndarray:
    out = values.copy()
    for r, centre in centres.items():
        out[repeat == r] -= centre
    return out


def coordinate_median(values: np.ndarray, condition: np.ndarray, repeat: np.ndarray) -> np.ndarray:
    """Today's rule: the median of the per-condition coordinatewise medians."""
    shared = _shared(condition, repeat)
    centres = {
        r: np.median(
            np.vstack(
                [np.median(values[(repeat == r) & (condition == c)], axis=0) for c in shared]
            ),
            axis=0,
        )
        for r in np.unique(repeat)
    }
    return _subtract(values, repeat, centres)


def centroid(values: np.ndarray, condition: np.ndarray, repeat: np.ndarray) -> np.ndarray:
    """The equal-weighted mean of the shared conditions' centroids."""
    shared = _shared(condition, repeat)
    centres = {
        r: np.mean([values[(repeat == r) & (condition == c)].mean(axis=0) for c in shared], axis=0)
        for r in np.unique(repeat)
    }
    return _subtract(values, repeat, centres)


def _weiszfeld(points: np.ndarray, weights: np.ndarray, iterations: int = 200) -> np.ndarray:
    """The weighted geometric median of ``points`` (Weiszfeld's iterations)."""
    centre = np.average(points, axis=0, weights=weights)
    for _ in range(iterations):
        distance = np.maximum(np.linalg.norm(points - centre, axis=1), 1e-12)
        w = weights / distance
        updated = (w[:, None] * points).sum(axis=0) / w.sum()
        if np.linalg.norm(updated - centre) <= 1e-10 * max(1.0, np.linalg.norm(centre)):
            return updated
        centre = updated
    return centre


def geometric_median(values: np.ndarray, condition: np.ndarray, repeat: np.ndarray) -> np.ndarray:
    """The geometric median of the shared conditions' cells, conditions weighted equally."""
    shared = _shared(condition, repeat)
    centres = {}
    for r in np.unique(repeat):
        rows = (repeat == r) & np.isin(condition, shared)
        sizes = {c: max(1, int(((repeat == r) & (condition == c)).sum())) for c in shared}
        weights = np.array([1.0 / sizes[c] for c in condition[rows]])
        centres[r] = _weiszfeld(values[rows], weights)
    return _subtract(values, repeat, centres)


def vector_model(values: np.ndarray, condition: np.ndarray, repeat: np.ndarray) -> np.ndarray:
    """``y_rc = alpha_c + b_r`` on the condition centroids, ``b_r`` subtracted; the repeats'
    effects are centred to mean 0, so the overall level stays."""
    presence = {r: list(dict.fromkeys(condition[repeat == r])) for r in dict.fromkeys(repeat)}
    layout = Layout.from_presence(presence)
    y = np.vstack([values[(repeat == r) & (condition == c)].mean(axis=0) for r, c in layout.cells])
    beta = np.linalg.pinv(layout.matrix()) @ y
    offset = len(layout.conditions)
    effects = np.vstack([np.zeros(values.shape[1]), beta[offset:]])
    effects -= effects.mean(axis=0)
    return _subtract(values, repeat, dict(zip(layout.repeats, effects, strict=True)))


def location_scale(values: np.ndarray, condition: np.ndarray, repeat: np.ndarray) -> np.ndarray:
    """Each coordinate centred and scaled per repeat, back to the pooled scale."""
    out = values.copy()
    pooled = values.std(axis=0, ddof=1)
    for r in np.unique(repeat):
        rows = repeat == r
        sd = values[rows].std(axis=0, ddof=1)
        out[rows] = (values[rows] - values[rows].mean(axis=0)) / np.where(sd > 0, sd, 1) * pooled
    return out


def harmony(values: np.ndarray, condition: np.ndarray, repeat: np.ndarray) -> np.ndarray:
    """Harmony's corrected values (``harmonypy``); ImportError without it."""
    import harmonypy
    import pandas as pd

    meta = pd.DataFrame({"repeat": repeat.astype(str)})
    result = harmonypy.run_harmony(values, meta, ["repeat"], verbose=False)
    corrected = np.asarray(result.Z_corr)
    return corrected.T if corrected.shape[0] != len(values) else corrected


CENTRINGS = {
    "coordinate_median": coordinate_median,
    "centroid": centroid,
    "geometric_median": geometric_median,
    "vector_model": vector_model,
    "location_scale": location_scale,
    "harmony": harmony,
}

__all__ = [
    "CENTRINGS",
    "centroid",
    "coordinate_median",
    "geometric_median",
    "harmony",
    "location_scale",
    "vector_model",
]
