"""Beyond the mean: does each feature's distribution differ from the control's?

The tests of question 9 compare means, so a condition that splits its cells into two
populations, or narrows them, with the mean unchanged reads as "no effect". With a
phenotypic fingerprint declared, the report also compares the distributions (owner,
2026-10-02), per biological repeat against the control of the same repeat, on the cells
pooled over each side's technical repeats (as ``feature_discovery`` pools them):

- **the spread**: the condition's SD over the control's (its log is tested);
- **the quantiles**: the shift of the 10%, 50% and 90% quantiles, in control SDs.

- **a split into two groups**: Sarle's bimodality coefficient of each side, per repeat
  (``statistics.bimodality_coefficient``); a condition splits its cells (``splits``) when
  its coefficient is above 5/9 and its control's is not, in at least half of its repeats. A
  skewed feature reaches 5/9 too, so the control's own shape is the reference; this is
  described, not tested.

Each statistic is tested across the biological repeats like a mean (a one-sample t-test
on its per-repeat values, ``MIN_REPEATS`` repeats or more), its p-values Benjamini-Hochberg
corrected within each sub-experiment over every feature, condition and statistic of this
comparison (a family of its own, beside the means'). A feature and condition differ
beyond the mean when one of its statistics has ``q <= alpha``. A feature without spread in
a control is not compared.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .._wording import count
from ..metadata import MORPHOBRICKS_GROUP_COLUMNS, TECHNICAL_REPEAT_COLUMN
from ..statistics import BIMODAL, bimodality_coefficient
from ._common import groups, validated
from ._experiments import corrected, experiments
from ._units import contrasts_of, describe_groups, resolve_columns
from .contrast_repeats import MIN_REPEATS, _paired_t
from .discovery import _pooled

QUANTILES = (0.1, 0.5, 0.9)
# The statistics compared, in the order the report gives them.
STATISTICS = ("spread", "q10", "q50", "q90")
ALPHA = 0.05
SHAPE_COLUMNS = ["feature", "condition", "repeat", "bimodality", "control_bimodality"]
SPLIT_COLUMNS = [
    "feature",
    "condition",
    "repeats",
    "split_repeats",
    "bimodality",
    "control_bimodality",
]


@dataclass(frozen=True)
class DistributionContrast:
    """One row per feature, condition and statistic (``tests``), with its per-repeat values."""

    tests: pd.DataFrame
    settings: dict[str, Any]
    shapes: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=SHAPE_COLUMNS))

    def splits(self) -> pd.DataFrame:
        """The feature and condition pairs whose cells split into two groups where the
        control's do not (see the module): the repeats that show it, of how many, and the
        median coefficient of each side over those repeats."""
        shapes = self.shapes
        if shapes.empty:
            return pd.DataFrame(columns=SPLIT_COLUMNS)
        shows = (shapes["bimodality"] > BIMODAL) & ~(shapes["control_bimodality"] > BIMODAL)
        rows = []
        for (feature, condition), block in shapes.assign(shows=shows).groupby(
            ["feature", "condition"], sort=False
        ):
            showing = block[block["shows"]]
            if len(showing) and 2 * len(showing) >= len(block):
                rows.append(
                    {
                        "feature": feature,
                        "condition": condition,
                        "repeats": len(block),
                        "split_repeats": len(showing),
                        "bimodality": float(showing["bimodality"].median()),
                        "control_bimodality": float(showing["control_bimodality"].median()),
                    }
                )
        return pd.DataFrame(rows, columns=SPLIT_COLUMNS)

    def differing(self) -> pd.DataFrame:
        """The feature and condition pairs whose distribution differs (some ``q <= alpha``),
        with the statistics that do."""
        passing = self.tests[self.tests["q"] <= self.settings["alpha"]]
        return (
            passing.groupby(["condition", "feature"], sort=False)["statistic"]
            .agg(list)
            .reset_index()
        )

    @property
    def n_tests(self) -> int:
        """How many statistics were tested (a p-value)."""
        return int(self.tests["p"].notna().sum())


def _statistics(treated: np.ndarray, control: np.ndarray) -> dict[str, float] | None:
    """The spread (log SD ratio) and the quantile shifts in control SDs, or ``None``
    without spread in the control."""
    sd = np.std(control, ddof=1) if len(control) > 1 else 0.0
    if not np.isfinite(sd) or sd == 0 or len(treated) < 2:
        return None
    spread = np.std(treated, ddof=1) / sd
    shifts = (np.quantile(treated, QUANTILES) - np.quantile(control, QUANTILES)) / sd
    values = {"spread": float(np.log(spread)) if spread > 0 else np.nan}
    values.update({f"q{round(q * 100)}": float(s) for q, s in zip(QUANTILES, shifts, strict=True)})
    return values


def _rows(feature: str, condition: Any, side: dict[str, Any]) -> list[dict[str, Any]]:
    per_repeat = {}
    for repeat, treated in side["by_t"].items():
        values = _statistics(treated, side["by_c"][repeat])
        if values is not None:
            per_repeat[repeat] = values
    rows = []
    for statistic in STATISTICS:
        values = np.array([v[statistic] for v in per_repeat.values()], dtype=float)
        tested = len(values) >= MIN_REPEATS and np.isfinite(values).all()
        rows.append(
            {
                "feature": feature,
                "condition": condition,
                "control_condition": side["control_condition"],
                "statistic": statistic,
                "repeats": len(values),
                "values": values,
                "mean": float(np.mean(values)) if len(values) else np.nan,
                "low": float(np.min(values)) if len(values) else np.nan,
                "high": float(np.max(values)) if len(values) else np.nan,
                "lower": int((values < 0).sum()),
                "higher": int((values > 0).sum()),
                "p": _paired_t(values) if tested else np.nan,
            }
        )
    return rows


def distribution_contrast(
    objects: pd.DataFrame,
    fovs: pd.DataFrame,
    *,
    features: Sequence[str],
    control: Mapping[str, Any],
    condition_col: str = MORPHOBRICKS_GROUP_COLUMNS[0],
    repeat_col: str | None = MORPHOBRICKS_GROUP_COLUMNS[1],
    technical_col: str | None = TECHNICAL_REPEAT_COLUMN,
    alpha: float = ALPHA,
    fov_col: str | None = None,
    group_cols: Sequence[str] | None = None,
    observation_unit: str | None = None,
) -> DistributionContrast:
    """Compare each feature's distribution with the control's (see the module); arguments
    follow ``feature_discovery``."""
    features = list(dict.fromkeys(features))
    data = validated(objects, fovs, features, fov_col, group_cols, observation_unit)
    repeat_col, technical_col = resolve_columns(
        data, condition_col, repeat_col, technical_col, control
    )
    all_groups = groups(data)
    described = describe_groups(all_groups, condition_col, repeat_col, control)
    contrasts = contrasts_of(described)
    rows, held, shapes = [], {}, []
    for feature in features:
        for condition, side in _pooled(
            all_groups, described, contrasts, feature, technical_col
        ).items():
            held.setdefault(condition, set()).update(side["by_t"])
            rows.extend(_rows(feature, condition, side))
            shapes += [
                {
                    "feature": feature,
                    "condition": condition,
                    "repeat": repeat,
                    "bimodality": bimodality_coefficient(treated),
                    "control_bimodality": bimodality_coefficient(side["by_c"][repeat]),
                }
                for repeat, treated in side["by_t"].items()
            ]
    columns = ["feature", "condition", "control_condition", "statistic", "repeats", "values"]
    columns += ["mean", "low", "high", "lower", "higher", "p", "q", "experiment"]
    tests = pd.DataFrame(rows).reindex(columns=columns)
    if len(tests):
        tests["experiment"] = tests["condition"].map(experiments(held))
        tests["q"] = corrected(tests["p"], tests["experiment"])
    settings = {
        "features": features,
        "control": dict(control),
        "alpha": alpha,
        "observation_unit": data.observation_unit,
        "repeat_col": repeat_col,
        "families": int(tests["experiment"].nunique()) if len(tests) else 0,
    }
    return DistributionContrast(tests, settings, pd.DataFrame(shapes, columns=SHAPE_COLUMNS))


def spread_words(row: Mapping[str, Any]) -> str:
    """``SD 0.73x the control (0.70-0.78)``: the spread on the ratio scale."""
    mean, low, high = (float(np.exp(row[k])) for k in ("mean", "low", "high"))
    return f"SD {mean:.2f}x the control ({low:.2f}-{high:.2f})"


def shift_words(row: Mapping[str, Any]) -> str:
    """``10% quantile +0.36 SD (+0.29 to +0.51)``."""
    name = f"{row['statistic'][1:]}% quantile"
    return f"{name} {row['mean']:+.2f} SD ({row['low']:+.2f} to {row['high']:+.2f})"


def tests_words(result: DistributionContrast) -> str:
    """The correction family counted: ``40 tests``, ``40 tests in each of 2 sub-experiments``."""
    families = result.settings["families"]
    per = result.n_tests // max(families, 1)
    if families <= 1:
        return count(per, "test")
    return f"{count(per, 'test')} in each of {count(families, 'sub-experiment')}"


__all__ = [
    "QUANTILES",
    "SHAPE_COLUMNS",
    "SPLIT_COLUMNS",
    "STATISTICS",
    "DistributionContrast",
    "distribution_contrast",
    "shift_words",
    "spread_words",
    "tests_words",
]
