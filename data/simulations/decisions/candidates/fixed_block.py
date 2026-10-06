"""The contrast layer's candidate engine: fixed-block model on per-repeat condition summaries,
with estimability by algebra and its explanation by the design's graph (P0, P2).

The model: ``y_rc = mu + alpha_c + beta_r + e_rc``, one value per condition ``c`` held in
biological repeat ``r`` (its summary: a repeat mean, a log mean), the repeat a fixed block
(3-6 repeats are too few to estimate a repeat variance), ``e_rc`` independent with one
variance. A contrast ``sum_c w_c alpha_c`` (weights summing to 0) is

- **directly paired** when at least ``MIN_REPEATS`` repeats hold every condition of the
  contrast: the paired t-test on ``sum_c w_c y_rc`` over those repeats, as today;
- **model-estimated** when it is estimable: its coefficient vector lies in the row space of
  the design matrix ``X`` (``estimable``: the part of it outside the span of ``X``'s right
  singular vectors with singular value above ``TOLERANCE`` times the largest is below
  ``TOLERANCE``), the residual degrees of freedom ``n - rank(X)`` at least 1; the least
  squares estimate, its SE from the pooled residual variance, a t-test and interval on those
  degrees of freedom;
- **not estimable** otherwise ("cannot be separated from the batch with this design").

The graph explains it: conditions are linked when a repeat holds both (``components``, the
package's sub-experiments); a contrast is estimable exactly when, within every connected
component, its weights sum to 0 (``graph_estimable``). P0 checks the algebra against the
graph on random designs.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from morphocoverage.sampling_unit._experiments import experiments

MIN_REPEATS = 3
TOLERANCE = 1e-9
NOT_ESTIMABLE = "cannot be separated from the batch with this design"


@dataclass(frozen=True)
class Layout:
    """The cells of the model: each ``(repeat, condition)`` held, and the orders used for
    the reference coding (the first condition and repeat absorbed in ``mu``)."""

    cells: tuple[tuple[Any, Any], ...]
    conditions: tuple[Any, ...]
    repeats: tuple[Any, ...]

    @classmethod
    def from_presence(cls, presence: Mapping[Any, Sequence[Any]]) -> Layout:
        """From each repeat's conditions."""
        cells = tuple((r, c) for r, held in presence.items() for c in held)
        conditions = tuple(dict.fromkeys(c for _, c in cells))
        return cls(cells, conditions, tuple(presence))

    def matrix(self) -> np.ndarray:
        """The design matrix: intercept, condition and repeat dummies (reference coding)."""
        c_index = {c: i for i, c in enumerate(self.conditions)}
        r_index = {r: i for i, r in enumerate(self.repeats)}
        width = 1 + (len(self.conditions) - 1) + (len(self.repeats) - 1)
        x = np.zeros((len(self.cells), width))
        x[:, 0] = 1.0
        for row, (r, c) in enumerate(self.cells):
            if c_index[c] > 0:
                x[row, c_index[c]] = 1.0
            if r_index[r] > 0:
                x[row, len(self.conditions) - 1 + r_index[r]] = 1.0
        return x

    def contrast(self, weights: Mapping[Any, float]) -> np.ndarray:
        """The coefficient vector of ``sum_c w_c alpha_c`` (a condition not held anywhere
        cannot be part of an estimable contrast: ``KeyError``)."""
        unknown = set(weights) - set(self.conditions)
        if unknown:
            raise KeyError(f"conditions held in no repeat: {sorted(map(str, unknown))}")
        vector = np.zeros(1 + len(self.conditions) - 1 + len(self.repeats) - 1)
        for i, c in enumerate(self.conditions):
            if i > 0:
                vector[i] = weights.get(c, 0.0)
        return vector


def estimable(x: np.ndarray, contrast: np.ndarray, tolerance: float = TOLERANCE) -> bool:
    """Whether ``contrast`` lies in the row space of ``x`` (see the module)."""
    _, singular, vt = np.linalg.svd(x, full_matrices=False)
    kept = vt[singular > tolerance * max(singular.max(initial=0.0), 1.0)]
    residual = contrast - kept.T @ (kept @ contrast)
    return bool(np.linalg.norm(residual) <= tolerance * max(1.0, np.linalg.norm(contrast)))


def components(presence: Mapping[Any, Sequence[Any]]) -> dict[Any, int]:
    """Each condition's connected component: conditions linked by a shared repeat (the
    package's ``_experiments.experiments``)."""
    held: dict[Any, set[Any]] = {}
    for repeat, conditions in presence.items():
        for condition in conditions:
            held.setdefault(condition, set()).add(repeat)
    return experiments(held)


def graph_estimable(presence: Mapping[Any, Sequence[Any]], weights: Mapping[Any, float]) -> bool:
    """Whether the weights sum to 0 within every connected component (see the module)."""
    found = components(presence)
    if any(c not in found and w != 0 for c, w in weights.items()):
        return False
    totals: dict[int, float] = {}
    for condition, weight in weights.items():
        if condition in found:
            totals[found[condition]] = totals.get(found[condition], 0.0) + weight
    return all(abs(t) <= TOLERANCE for t in totals.values())


def regime(
    presence: Mapping[Any, Sequence[Any]],
    weights: Mapping[Any, float],
    min_repeats: int = MIN_REPEATS,
) -> str:
    """``direct``, ``model`` or ``not estimable`` for one contrast (see the module)."""
    involved = [c for c, w in weights.items() if w != 0]
    shared = [r for r, held in presence.items() if set(involved) <= set(held)]
    if len(shared) >= min_repeats:
        return "direct"
    layout = Layout.from_presence(presence)
    if any(c not in layout.conditions for c in involved):
        return "not estimable"
    x = layout.matrix()
    if not estimable(x, layout.contrast(weights)):
        return "not estimable"
    rank = np.linalg.matrix_rank(x)
    return "model" if len(layout.cells) - rank >= 1 else "not estimable"


def _direct(values: pd.DataFrame, weights: Mapping[Any, float], alpha: float) -> dict[str, Any]:
    """The paired t-test of the contrast over the repeats holding all its conditions."""
    involved = [c for c, w in weights.items() if w != 0]
    wide = values.pivot_table(index="repeat", columns="condition", values="value")
    wide = wide[involved].dropna()
    combined = sum(weights[c] * wide[c] for c in involved).to_numpy(dtype=float)
    n = len(combined)
    estimate = float(combined.mean())
    se = float(combined.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return _t_result("direct", estimate, se, n - 1, alpha)


def _t_result(name: str, estimate: float, se: float, df: int, alpha: float) -> dict[str, Any]:
    """The t-test and interval of an estimate."""
    if not (np.isfinite(se) and se > 0 and df >= 1):
        return {
            "regime": name,
            "estimate": estimate,
            "se": se,
            "df": df,
            "p": np.nan,
            "lo": np.nan,
            "hi": np.nan,
        }
    t = estimate / se
    half = stats.t.ppf(1 - alpha / 2, df) * se
    return {
        "regime": name,
        "estimate": estimate,
        "se": se,
        "df": df,
        "p": float(2 * stats.t.sf(abs(t), df)),
        "lo": estimate - half,
        "hi": estimate + half,
    }


def fit_model(
    values: pd.DataFrame, weights: Mapping[Any, float], alpha: float = 0.05
) -> dict[str, Any]:
    """The fixed-block estimate of the contrast from ``values`` (columns ``repeat``,
    ``condition``, ``value``: one row per cell held), whatever the regime."""
    presence = values.groupby("repeat", sort=False)["condition"].agg(list).to_dict()
    layout = Layout.from_presence(presence)
    lookup = values.set_index(["repeat", "condition"])["value"]
    y = np.array([lookup[cell] for cell in layout.cells], dtype=float)
    x = layout.matrix()
    vector = layout.contrast(weights)
    if not estimable(x, vector):
        return {
            "regime": "not estimable",
            "estimate": np.nan,
            "se": np.nan,
            "df": 0,
            "p": np.nan,
            "lo": np.nan,
            "hi": np.nan,
            "reason": NOT_ESTIMABLE,
        }
    pinv = np.linalg.pinv(x)
    beta = pinv @ y
    rank = np.linalg.matrix_rank(x)
    df = len(y) - rank
    residual = y - x @ beta
    variance = float(residual @ residual / df) if df > 0 else np.nan
    se = float(np.sqrt(variance * vector @ (pinv @ pinv.T) @ vector)) if df > 0 else np.nan
    return _t_result("model", float(vector @ beta), se, df, alpha)


def contrast(
    values: pd.DataFrame,
    weights: Mapping[Any, float],
    alpha: float = 0.05,
    min_repeats: int = MIN_REPEATS,
) -> dict[str, Any]:
    """One contrast through the three regimes: direct, else model, else not estimable."""
    presence = values.groupby("repeat", sort=False)["condition"].agg(list).to_dict()
    found = regime(presence, weights, min_repeats)
    if found == "direct":
        return _direct(values, weights, alpha)
    if found == "model":
        return fit_model(values, weights, alpha)
    return {
        "regime": "not estimable",
        "estimate": np.nan,
        "se": np.nan,
        "df": 0,
        "p": np.nan,
        "lo": np.nan,
        "hi": np.nan,
        "reason": NOT_ESTIMABLE,
    }


__all__ = [
    "MIN_REPEATS",
    "NOT_ESTIMABLE",
    "Layout",
    "components",
    "contrast",
    "estimable",
    "fit_model",
    "graph_estimable",
    "regime",
]
