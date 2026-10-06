"""Per-repeat condition summaries under the three estimands of P1, and the comparators that
read the cells or fields directly.

- ``cell``: the mean of every cell of the condition in the repeat (the package's default,
  the owner's estimand: each cell weighs the same);
- ``field``: the mean of the field means (each field with cells weighs the same);
- ``well``: the mean of the well means, a well's mean over its cells (each recorded well
  weighs the same; without a well column, the condition in the repeat is one well, so this
  is the cell-weighted mean).

``moderated_t`` is the moderated paired t-test across a family of features (limma-style
empirical Bayes, Smyth 2004): each feature's variance of its per-repeat values shrunk toward
a prior fitted on every feature of the family, so 3 repeats that agree by chance on one
feature do not give it a tiny p (audit 10, owner, 2026-10-06: a P1 candidate, added before
any confirmatory run).

``mixed_model`` is the random-intercept comparator (statsmodels, the ``validation`` extra):
field means with a fixed condition effect and random repeat, repeat-by-condition and,
when recorded, well intercepts; ``pooled_cells`` the reference that ignores the hierarchy
(Welch's t on the cells pooled, the anti-conservative reading the paired test replaces).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from scipy import special, stats

from morphocoverage.metadata import CONDITION_COLUMN, REPEAT_COLUMN, TECHNICAL_REPEAT_COLUMN

FIELD = "position_id"
ESTIMANDS = ("cell", "field", "well")


def summaries(objects: pd.DataFrame, feature: str, estimand: str) -> pd.DataFrame:
    """One row per repeat and condition: ``repeat``, ``condition``, ``value`` (the mean
    under ``estimand``) and ``cells``."""
    keys = [REPEAT_COLUMN, CONDITION_COLUMN]
    frame = objects.dropna(subset=[feature])
    if estimand == "cell":
        grouped = frame.groupby(keys)[feature].agg(["mean", "size"])
    elif estimand in ("field", "well"):
        unit = FIELD if estimand == "field" else TECHNICAL_REPEAT_COLUMN
        if unit not in frame:
            grouped = frame.groupby(keys)[feature].agg(["mean", "size"])
        else:
            units = frame.groupby([*keys, unit])[feature].agg(["mean", "size"]).reset_index()
            grouped = units.groupby(keys).agg(mean=("mean", "mean"), size=("size", "sum"))
    else:
        raise ValueError(f"estimand must be one of {ESTIMANDS}")
    out = grouped.reset_index().rename(
        columns={
            REPEAT_COLUMN: "repeat",
            CONDITION_COLUMN: "condition",
            "mean": "value",
            "size": "cells",
        }
    )
    return out


def paired_values(
    table: pd.DataFrame, condition: str, control: str, log: bool
) -> tuple[np.ndarray, list[Any]]:
    """Per repeat holding both, the log ratio (``log``) or the difference of the condition's
    value to the control's, and the repeats."""
    wide = table.pivot_table(index="repeat", columns="condition", values="value")
    if condition not in wide or control not in wide:
        return np.array([]), []
    both = wide[[condition, control]].dropna()
    if log:
        if (both <= 0).any().any():
            return np.array([]), []
        values = np.log(both[condition] / both[control]).to_numpy()
    else:
        values = (both[condition] - both[control]).to_numpy()
    return values.astype(float), list(both.index)


def t_summary(values: np.ndarray, alpha: float = 0.05) -> dict[str, float]:
    """The one-sample t-test of ``values`` against 0: estimate, p and the t interval."""
    n = len(values)
    if n < 2 or np.ptp(values) == 0:
        return {
            "estimate": float(values.mean()) if n else np.nan,
            "p": np.nan,
            "lo": np.nan,
            "hi": np.nan,
            "n": n,
        }
    se = values.std(ddof=1) / np.sqrt(n)
    half = stats.t.ppf(1 - alpha / 2, n - 1) * se
    p = 2 * stats.t.sf(abs(values.mean() / se), n - 1)
    mean = float(values.mean())
    return {"estimate": mean, "p": float(p), "lo": mean - half, "hi": mean + half, "n": n}


def _trigamma_inverse(x: float) -> float:
    """The ``y`` with ``trigamma(y) = x``, by Newton's method (limma's ``trigammaInverse``)."""
    if x > 1e7:
        return 1 / np.sqrt(x)
    if x < 1e-6:
        return 1 / x
    y = 0.5 + 1 / x
    for _ in range(50):
        tri = float(special.polygamma(1, y))
        step = tri * (1 - tri / x) / float(special.polygamma(2, y))
        y += step
        if -step / y < 1e-8:
            break
    return float(y)


def variance_prior(variances: np.ndarray, dfs: np.ndarray) -> tuple[float, float]:
    """The prior degrees of freedom ``d0`` and variance ``s0^2`` of a family's sample
    variances (each on ``dfs`` degrees of freedom), by Smyth's (2004) moments of their logs;
    ``(0, nan)`` with fewer than 3 usable variances, ``(inf, s0^2)`` when they vary no more
    than sampling alone would (every feature then shares one variance)."""
    keep = np.isfinite(variances) & (variances > 0) & (dfs > 0)
    s2, d = variances[keep], dfs[keep]
    if len(s2) < 3:
        return 0.0, float("nan")
    e = np.log(s2) - special.digamma(d / 2) + np.log(d / 2)
    mean = float(e.mean())
    spread = float(((e - mean) ** 2).sum() / (len(e) - 1) - special.polygamma(1, d / 2).mean())
    if spread <= 0:
        return float("inf"), float(np.exp(mean))
    d0 = 2 * _trigamma_inverse(spread)
    return d0, float(np.exp(mean + special.digamma(d0 / 2) - np.log(d0 / 2)))


def moderated_t(
    family: Mapping[str, np.ndarray], alpha: float = 0.05
) -> dict[str, dict[str, float]]:
    """The moderated paired t-test of each feature's per-repeat values (``family``: feature
    -> values, log ratios or differences against the control) against 0: its variance
    shrunk toward the family's prior (``variance_prior``), ``d0 + n - 1`` degrees of
    freedom; estimate, p and the t interval per feature. With fewer than 3 features (no
    prior) it is the paired t-test."""
    names = [k for k, v in family.items() if len(v) >= 2]
    variances = np.array([np.var(family[k], ddof=1) for k in names], dtype=float)
    dfs = np.array([len(family[k]) - 1 for k in names], dtype=float)
    d0, s0 = variance_prior(variances, dfs)
    out = {}
    for name, s2, d in zip(names, variances, dfs, strict=True):
        values = np.asarray(family[name], dtype=float)
        mean, n = float(values.mean()), len(values)
        if d0 == 0 or not np.isfinite(s0):
            post, df = s2, d
        elif np.isinf(d0):
            post, df = s0, np.inf
        else:
            post, df = (d0 * s0 + d * s2) / (d0 + d), d0 + d
        se = float(np.sqrt(post / n)) if post > 0 else float("nan")
        if not se > 0:
            out[name] = {"estimate": mean, "p": np.nan, "lo": np.nan, "hi": np.nan, "n": n}
            out[name]["prior_df"] = float(d0)
            continue
        quantile = stats.norm if np.isinf(df) else stats.t(df)
        half = float(quantile.ppf(1 - alpha / 2)) * se
        p = 2 * float(quantile.sf(abs(mean / se)))
        out[name] = {"estimate": mean, "p": p, "lo": mean - half, "hi": mean + half, "n": n}
        out[name]["prior_df"] = float(d0)
    return out


def pooled_cells(
    objects: pd.DataFrame, feature: str, condition: str, control: str
) -> dict[str, float]:
    """Welch's t on every cell of the two conditions pooled (the reference that ignores the
    repeats, wells and fields)."""
    first = objects.loc[objects[CONDITION_COLUMN] == condition, feature].dropna()
    second = objects.loc[objects[CONDITION_COLUMN] == control, feature].dropna()
    test = stats.ttest_ind(first, second, equal_var=False)
    return {"estimate": float(first.mean() - second.mean()), "p": float(test.pvalue)}


def mixed_model(
    objects: pd.DataFrame, feature: str, condition: str, control: str, log: bool
) -> dict[str, float]:
    """The random-intercept comparator on field means (see the module); the condition's
    coefficient on the log scale when ``log``. Needs statsmodels."""
    import warnings

    import statsmodels.formula.api as smf

    keys = [REPEAT_COLUMN, CONDITION_COLUMN]
    wells = TECHNICAL_REPEAT_COLUMN in objects
    frame = objects[objects[CONDITION_COLUMN].isin([condition, control])]
    by = [*keys, TECHNICAL_REPEAT_COLUMN, FIELD] if wells else [*keys, FIELD]
    fields = frame.groupby(by)[feature].mean().reset_index()
    fields["y"] = np.log(fields[feature]) if log else fields[feature]
    fields["treated"] = (fields[CONDITION_COLUMN] == condition).astype(float)
    fields["block"] = fields[REPEAT_COLUMN].astype(str) + "_" + fields[CONDITION_COLUMN].astype(str)
    components = {"block": "0 + C(block)"}
    if wells:
        fields["well"] = fields["block"] + "_" + fields[TECHNICAL_REPEAT_COLUMN].astype(str)
        if fields["well"].nunique() > fields["block"].nunique():
            components["well"] = "0 + C(well)"
    model = smf.mixedlm("y ~ treated", fields, groups=fields[REPEAT_COLUMN], vc_formula=components)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = model.fit(reml=True, method="lbfgs")
    estimate, se = float(result.params["treated"]), float(result.bse["treated"])
    p = float(result.pvalues["treated"])
    return {
        "estimate": estimate,
        "p": p,
        "lo": estimate - 1.96 * se,
        "hi": estimate + 1.96 * se,
        "converged": float(bool(result.converged)),
    }


__all__ = [
    "ESTIMANDS",
    "mixed_model",
    "moderated_t",
    "paired_values",
    "pooled_cells",
    "summaries",
    "t_summary",
    "variance_prior",
]
