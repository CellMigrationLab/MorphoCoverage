"""Per-repeat condition summaries under the three estimands of P1, and the comparators that
read the cells or fields directly.

- ``cell``: the mean of every cell of the condition in the repeat (the package's default,
  the owner's estimand: each cell weighs the same);
- ``field``: the mean of the field means (each field with cells weighs the same);
- ``well``: the mean of the well means, a well's mean over its cells (each recorded well
  weighs the same; without a well column, the condition in the repeat is one well, so this
  is the cell-weighted mean).

``mixed_model`` is the random-intercept comparator (statsmodels, the ``validation`` extra):
field means with a fixed condition effect and random repeat, repeat-by-condition and,
when recorded, well intercepts; ``pooled_cells`` the reference that ignores the hierarchy
(Welch's t on the cells pooled, the anti-conservative reading the paired test replaces).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

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


__all__ = ["ESTIMANDS", "mixed_model", "paired_values", "pooled_cells", "summaries", "t_summary"]
