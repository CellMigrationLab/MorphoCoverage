"""The frontier's resampling of independent units (private; ``frontier`` uses it).

``_bootstrap`` resamples the declared independent units with replacement and
scores every family in every resample with the same "from here on" rule as
the point estimate (``_from_here_on``); ``_saved_interval`` turns the draws
into the widened, approximate 95% interval of the fields saved, withheld
below ``FEW_UNITS`` units. Shared-private: every name is used by
``retrospective.frontier``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from ..statistics import _interval
from .criteria import Preservation
from .frontier_text import BASELINE
from .protocol_text import FEW_UNITS


def _from_here_on(risks: np.ndarray, alpha: float) -> int | None:
    """Index of the first setting (by effort) from which every later risk is within alpha.

    The same "from here on" rule as the protocol's field count: a lucky dip in
    a noisy curve, followed by worse settings, is not a safe choice. Settings
    whose risk could not be judged (NaN) neither start nor break the run.
    """
    chosen = None
    for i, risk in enumerate(risks):
        if np.isnan(risk):
            continue
        if risk <= alpha + 1e-12:
            chosen = i if chosen is None else chosen
        else:
            chosen = None
    return chosen


def _unit_ids(runs: pd.DataFrame, unit_cols: Sequence[str]) -> pd.Series:
    if not unit_cols:
        return pd.Series("all", index=runs.index)
    return runs[list(unit_cols)].astype(str).agg(" / ".join, axis=1)


def _bootstrap(
    runs: pd.DataFrame,
    unit_cols: Sequence[str],
    alphas: Sequence[float],
    n_boot: int,
    rng: np.random.Generator,
) -> dict[tuple[str, float], np.ndarray]:
    """Fields saved per family and alpha in each resample of independent units."""
    runs = runs.assign(
        unit=_unit_ids(runs, unit_cols),
        defined=runs["status"] != Preservation.UNDEFINED.value,
        failed=runs["status"] == Preservation.NOT_PRESERVED.value,
    )
    units = sorted(runs["unit"].unique())
    if len(units) < 2:
        return {}
    configs = list(runs.groupby(["family", "setting"], sort=False).groups)
    index = {u: i for i, u in enumerate(units)}
    shape = (len(configs), len(units))
    count, effort, defined, failed = (np.zeros(shape) for _ in range(4))
    for c, ((name, setting), own) in enumerate(runs.groupby(["family", "setting"], sort=False)):
        per_unit = own.groupby("unit")[["effort", "defined", "failed"]].agg(["sum", "count"])
        for unit, row in per_unit.iterrows():
            u = index[unit]
            count[c, u] = row[("effort", "count")]
            effort[c, u] = row[("effort", "sum")]
            defined[c, u] = row[("defined", "sum")]
            failed[c, u] = row[("failed", "sum")]
    picks = rng.integers(0, len(units), size=(n_boot, len(units)))
    weights = np.zeros((n_boot, len(units)))
    np.add.at(weights, (np.arange(n_boot)[:, None], picks), 1.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_effort = (weights @ effort.T) / (weights @ count.T)
        risk = (weights @ failed.T) / (weights @ defined.T)
    families = {}
    for c, (name, _) in enumerate(configs):
        families.setdefault(name, []).append(c)
    saved = {}
    for alpha in alphas:
        needed = {}
        for name, columns in families.items():
            cols = np.array(columns)
            best = np.full(n_boot, np.nan)
            for b in range(n_boot):
                # Settings in the order of their effort in this resample.
                order = np.argsort(mean_effort[b, cols], kind="stable")
                index = _from_here_on(risk[b, cols][order], alpha)
                if index is not None:
                    best[b] = mean_effort[b, cols][order][index]
            needed[name] = best
        for name in families:
            if name != BASELINE:
                saved[(name, alpha)] = needed[BASELINE] - needed[name]
    return saved


def _saved_interval(
    point: float, draws: np.ndarray, units: int
) -> tuple[float | None, float | None]:
    """The widened 95% interval of the fields saved, or ``(None, None)`` below ``FEW_UNITS``.

    ``statistics._interval`` widens each half-width of the percentile interval
    by ``sqrt(k / (k - 1)) t_{k-1} / z`` for the ``k`` units resampled.
    """
    if units < FEW_UNITS:
        return None, None
    _, low, high = _interval(float(point), draws, units, 0.95)
    if not (np.isfinite(low) and np.isfinite(high)):
        return None, None
    return float(low), float(high)


__all__ = ["_bootstrap", "_from_here_on", "_saved_interval", "_unit_ids"]
