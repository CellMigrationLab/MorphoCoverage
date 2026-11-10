"""Beyond the mean: does each feature's distribution differ from the control's?

The tests of question 9 compare means, so a condition that splits its cells into two
populations, or narrows them, with the mean unchanged reads as "no effect". With a
phenotypic fingerprint declared, the report also compares the distributions (owner,
2026-10-02), per biological repeat against the control of the same repeat:

- **the spread**: the condition's SD over the control's (its log is tested);
- **the quantiles**: the shift of the 10%, 50% and 90% quantiles beyond the shift of the
  mean, in control SDs, so a condition that only moves its mean shifts no quantile (audit 10,
  owner, 2026-10-06; before, the quantile shifts held the mean's shift and a pure shift read
  "differs beyond the mean").

Every statistic reads the feature on its scale, the one its tests across repeats read
(``statistics.log_scale``: declared in ``scales``, else the log when every observation is
positive): on the log of the values, a change by a factor moves the mean alone.

Both are read on **field-balanced subsamples** (owner, 2026-10-06, audit 8): per repeat,
each side keeps the same number of fields, the smaller side's (drawn at random among its
fields, its technical repeats pooled), and the same number of cells in each field, the
smallest field's of either side, at most ``CELLS_CAP`` (drawn at random in each field); the
statistics are averaged over ``DRAWS`` such draws, seeded from the package's seeds (one
draw when nothing is left out). Pooled cells read the field design as an effect: a control
imaged in one field of 60 cells against three fields of 20 has a smaller spread, field to
field variation left out, and its quantiles shift with the number of cells.

- **a split into two groups**: Sarle's bimodality coefficient of each side, per repeat
  (``statistics.bimodality_coefficient``); a condition's cells may split in two
  (``splits``) when its coefficient is above 5/9 without a heavy tail (excess kurtosis at
  most ``HEAVY_TAIL``: a skewed single-peaked feature reaches 5/9 too) and its control's
  is below 5/9, in at least half of its repeats that can be assessed. Described, not
  tested. A repeat cannot be assessed when a coefficient is undefined (fewer than 4
  observations on a side, or no spread): it leaves the denominator and is named
  (``1 repeat not assessed: 3 cells``), and a condition with no repeat assessed is not
  assessed (``not_assessed``; owner, 2026-10-05, #301).

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

from .._checks import check_seed, is_number
from .._control import check_comparisons
from .._frozen import owned
from .._seeds import group_rng
from .._style import low_to_high
from .._wording import count, listing, plural
from ..metadata import MORPHOBRICKS_GROUP_COLUMNS, REPEAT_COLUMN, TECHNICAL_REPEAT_COLUMN
from ..schema import observation_words
from ..statistics import (
    ALPHA,
    BIMODAL,
    bimodality_coefficient,
    excess_kurtosis,
    log_scale,
    paired_t,
)
from ._common import groups, validated
from ._experiments import corrected, declared, experiments, family_noun
from ._units import (
    contrasts_of,
    control_label,
    describe_groups,
    resolve_columns,
    side_units,
    sides,
)
from .contrast_repeats import MIN_REPEATS

QUANTILES = (0.1, 0.5, 0.9)
# The field-balanced draws whose statistics are averaged, and the most cells kept a field.
DRAWS = 50
CELLS_CAP = 100
# The statistics compared, in the order the report gives them.
STATISTICS = ("spread", "q10", "q50", "q90")
SHAPE_COLUMNS = [
    "feature",
    "condition",
    REPEAT_COLUMN,
    "bimodality",
    "kurtosis",
    "control_bimodality",
    "observations",
    "control_observations",
]
# The excess kurtosis above which a sample has a heavy tail rather than two groups.
HEAVY_TAIL = 2.0
SPLIT_COLUMNS = [
    "feature",
    "condition",
    "repeats",
    "assessed",
    "split_repeats",
    "bimodality",
    "control_bimodality",
    "not_assessed",
]
# Fewer observations than this on a side give no bimodality coefficient.
_MIN_SHAPE = 4


def _assessable(shapes: pd.DataFrame) -> pd.Series:
    """Whether each repeat's split can be assessed: both coefficients and the kurtosis
    defined (4 observations or more a side, with spread)."""
    return (
        shapes["bimodality"].notna()
        & shapes["kurtosis"].notna()
        & shapes["control_bimodality"].notna()
    )


def _counted(numbers: list[int], unit: str, side: str = "") -> str:
    """``3 cells``, ``3 and 2 cells``, ``1 control cell``."""
    word = unit if len(numbers) == 1 and numbers[0] == 1 else plural(unit)
    return f"{listing(numbers, len(numbers))} {side}{word}"


def _reasons(rows: pd.DataFrame, unit: str) -> str:
    """``3 and 2 cells``: why each repeat's split cannot be assessed (the observations of the
    side with fewer than 4, else no spread)."""
    few, few_control, flat, flat_control = [], [], 0, 0
    for row in rows.to_dict("records"):
        if row["observations"] < _MIN_SHAPE:
            few.append(int(row["observations"]))
        elif row["control_observations"] < _MIN_SHAPE:
            few_control.append(int(row["control_observations"]))
        elif pd.isna(row["bimodality"]) or pd.isna(row["kurtosis"]):
            flat += 1
        else:
            flat_control += 1
    reasons = []
    if few:
        reasons.append(_counted(few, unit))
    if few_control:
        reasons.append(_counted(few_control, unit, "control "))
    if flat:
        reasons.append("no spread")
    if flat_control:
        reasons.append("no spread in the control")
    return listing(reasons, len(reasons))


def unassessed_words(rows: pd.DataFrame, unit: str = "cell") -> str:
    """``1 repeat not assessed: 3 cells``: the repeats whose split cannot be assessed and why;
    empty for none."""
    if rows.empty:
        return ""
    return f"{count(len(rows), 'repeat')} not assessed: {_reasons(rows, unit)}"


@owned("tests", "settings", "shapes")
@dataclass(frozen=True)
class DistributionContrast:
    """One row per feature, condition and statistic (``tests``), with its per-repeat values."""

    tests: pd.DataFrame
    settings: dict[str, Any]
    shapes: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=SHAPE_COLUMNS))

    def splits(self) -> pd.DataFrame:
        """The feature and condition pairs whose cells split into two groups where the
        control's do not (see the module): the repeats that show it, of the repeats
        ``assessed`` (``repeats`` counts them all), the median coefficient of each side over
        those that show it, and the repeats ``not_assessed``, in words."""
        shapes = self.shapes
        if shapes.empty:
            return pd.DataFrame(columns=SPLIT_COLUMNS)
        unit = observation_words(self.settings)[0]
        # Two groups, not a long tail: a skewed single-peaked feature (an exponential, a
        # lognormal) also passes 5/9, but with a heavy tail (excess kurtosis above 2, the
        # package's heavy-tail threshold, ``strategies._averaging_cv``); the control's
        # coefficient must be known and below 5/9.
        shows = (
            (shapes["bimodality"] > BIMODAL)
            & (shapes["kurtosis"] <= HEAVY_TAIL)
            & (shapes["control_bimodality"] <= BIMODAL)
        )
        marked = shapes.assign(shows=shows, assessable=_assessable(shapes))
        rows = []
        for (feature, condition), block in marked.groupby(["feature", "condition"], sort=False):
            # A repeat that cannot be assessed is no evidence either way: out of the
            # denominator, and named (owner, 2026-10-05).
            assessed = block[block["assessable"]]
            showing = assessed[assessed["shows"]]
            if len(showing) and 2 * len(showing) >= len(assessed):
                rows.append(
                    {
                        "feature": feature,
                        "condition": condition,
                        "repeats": len(block),
                        "assessed": len(assessed),
                        "split_repeats": len(showing),
                        "bimodality": float(showing["bimodality"].median()),
                        "control_bimodality": float(showing["control_bimodality"].median()),
                        "not_assessed": unassessed_words(block[~block["assessable"]], unit),
                    }
                )
        return pd.DataFrame(rows, columns=SPLIT_COLUMNS)

    def not_assessed(self) -> pd.DataFrame:
        """The feature and condition pairs whose split no repeat can assess: their repeats and
        why (``reasons``: ``3, 2 and 3 cells``)."""
        columns = ["feature", "condition", "repeats", "reasons"]
        shapes = self.shapes
        if shapes.empty:
            return pd.DataFrame(columns=columns)
        unit = observation_words(self.settings)[0]
        rows = [
            {
                "feature": feature,
                "condition": condition,
                "repeats": len(block),
                "reasons": _reasons(block, unit),
            }
            for (feature, condition), block in shapes.assign(
                assessable=_assessable(shapes)
            ).groupby(["feature", "condition"], sort=False)
            if not block["assessable"].any()
        ]
        return pd.DataFrame(rows, columns=columns)

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


def _drawn(fields: list[np.ndarray], n: int, m: int, draws: int, rng: Any) -> np.ndarray:
    """``draws`` subsamples of ``n`` of ``fields`` and ``m`` values of each, one per row."""
    cells = np.stack([f[np.argsort(rng.random((draws, len(f))), axis=1)[:, :m]] for f in fields])
    chosen = np.argsort(rng.random((draws, len(fields))), axis=1)[:, :n]
    return cells[chosen, np.arange(draws)[:, None]].reshape(draws, n * m)


def _statistics(
    treated: list[np.ndarray], control: list[np.ndarray], rng: Any, log: bool = False
) -> dict[str, float] | None:
    """The spread (log SD ratio) and the quantile shifts beyond the mean's, in control SDs,
    on field-balanced subsamples of the fields ``treated`` and ``control`` (see the module),
    on the log of the values with ``log``, averaged over the draws; ``None`` without two
    values a side or spread in the control."""
    if log:
        treated, control = [np.log(f) for f in treated], [np.log(f) for f in control]
    n = min(len(treated), len(control))
    m = min(CELLS_CAP, *(len(f) for f in [*treated, *control])) if n else 0
    if n * m < 2:
        return None
    whole = all(len(f) == m for f in [*treated, *control])
    draws = 1 if whole and len(treated) == len(control) == n else DRAWS
    t, c = _drawn(treated, n, m, draws, rng), _drawn(control, n, m, draws, rng)
    sd = c.std(axis=1, ddof=1)
    kept = np.isfinite(sd) & (sd > 0)
    if not kept.any():
        return None
    t, c, sd = t[kept], c[kept], sd[kept]
    ratio = t.std(axis=1, ddof=1) / sd
    moved = t.mean(axis=1) - c.mean(axis=1)
    shifts = (np.quantile(t, QUANTILES, axis=1) - np.quantile(c, QUANTILES, axis=1) - moved) / sd
    values = {"spread": float(np.log(ratio).mean()) if (ratio > 0).all() else np.nan}
    values.update(
        {f"q{round(q * 100)}": float(s.mean()) for q, s in zip(QUANTILES, shifts, strict=True)}
    )
    return values


def _fields(
    all_groups: list, described: list, contrasts: list, feature: str, technical_col: str | None
) -> dict[Any, dict[str, Any]]:
    """Per condition (or comparison), the fields holding values of ``feature`` of each side
    per repeat (``by_t``, ``by_c``: lists of arrays) and its control's label."""
    pooled: dict[Any, dict[str, Any]] = {}
    for repeat, condition in contrasts:
        treated, controls = sides(described, repeat, condition)
        if not controls:
            continue
        found = [
            side_units(all_groups, feature, chosen, technical_col)[0]
            for chosen in (treated, controls)
        ]
        t_fields, c_fields = ([f for u in units for f in u if len(f)] for units in found)
        if not t_fields or not c_fields:
            continue
        side = pooled.setdefault(
            condition,
            {"by_t": {}, "by_c": {}, "control_condition": control_label(described, controls)},
        )
        side["by_t"][repeat], side["by_c"][repeat] = t_fields, c_fields
    return pooled


def _rows(
    feature: str, condition: Any, side: dict[str, Any], random_state: int, log: bool = False
) -> list[dict[str, Any]]:
    per_repeat = {}
    for repeat, treated in side["by_t"].items():
        key = {"feature": feature, "condition": condition, REPEAT_COLUMN: repeat}
        rng = group_rng(random_state, {**key, "test": "beyond the mean"})
        values = _statistics(treated, side["by_c"][repeat], rng, log)
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
                "p": paired_t(values) if tested else np.nan,
            }
        )
    return rows


def distribution_contrast(
    objects: pd.DataFrame,
    fovs: pd.DataFrame,
    *,
    features: Sequence[str],
    control: Mapping[str, Any] | None = None,
    condition_col: str = MORPHOBRICKS_GROUP_COLUMNS[0],
    repeat_col: str | None = MORPHOBRICKS_GROUP_COLUMNS[1],
    technical_col: str | None = TECHNICAL_REPEAT_COLUMN,
    alpha: float = ALPHA,
    fov_col: str | None = None,
    group_cols: Sequence[str] | None = None,
    observation_unit: str | None = None,
    comparisons: Sequence[tuple[Any, Any]] | None = None,
    random_state: int = 0,
    scales: Mapping[str, str] | None = None,
    families: Mapping[Any, Any] | None = None,
) -> DistributionContrast:
    """Compare each feature's distribution with the control's (see the module); arguments
    follow ``feature_discovery`` (``comparisons``: each condition against its reference;
    ``random_state`` seeds the field-balanced draws; ``scales``: each feature's declared
    scale, else the automatic rule; ``families``: each condition's declared experiment)."""
    random_state = check_seed(random_state)
    if comparisons is not None:
        comparisons = check_comparisons(comparisons)
    features = list(dict.fromkeys(features))
    if not is_number(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    data = validated(objects, fovs, features, fov_col, group_cols, observation_unit)
    repeat_col, technical_col = resolve_columns(
        data, condition_col, repeat_col, technical_col, control
    )
    all_groups = groups(data)
    described = describe_groups(all_groups, condition_col, repeat_col, control, comparisons)
    contrasts = contrasts_of(described)
    rows, held, shapes, logs = [], {}, [], {}
    for feature in features:
        scale = (scales or {}).get(feature)
        logs[feature] = log_scale(data.feature_values(feature), scale, repr(feature))
        for condition, side in _fields(
            all_groups, described, contrasts, feature, technical_col
        ).items():
            held.setdefault(condition, set()).update(side["by_t"])
            rows.extend(_rows(feature, condition, side, random_state, logs[feature]))
            # The split into two groups reads every observation, pooled.
            for repeat, fields in side["by_t"].items():
                cells, controls = np.concatenate(fields), np.concatenate(side["by_c"][repeat])
                shapes.append(
                    {
                        "feature": feature,
                        "condition": condition,
                        REPEAT_COLUMN: repeat,
                        "bimodality": bimodality_coefficient(cells),
                        "kurtosis": excess_kurtosis(cells),
                        "control_bimodality": bimodality_coefficient(controls),
                        "observations": len(cells),
                        "control_observations": len(controls),
                    }
                )
    columns = ["feature", "condition", "control_condition", "statistic", "repeats", "values"]
    columns += ["mean", "low", "high", "lower", "higher", "p", "q", "experiment"]
    tests = pd.DataFrame(rows).reindex(columns=columns)
    if len(tests):
        tests["experiment"] = tests["condition"].map(declared(experiments(held), families))
        tests["q"] = corrected(tests["p"], tests["experiment"])
    settings = {
        "features": features,
        "control": dict(control or {}),
        "comparisons": list(comparisons or []),
        "alpha": alpha,
        "observation_unit": data.observation_unit,
        "observation_unit_declared": data.observation_unit_declared,
        "repeat_col": repeat_col,
        "families": int(tests["experiment"].nunique()) if len(tests) else 0,
        "declared_experiments": bool(families),
        "random_state": random_state,
        "draws": DRAWS,
        "cells_cap": CELLS_CAP,
        "log_scale": logs,
    }
    return DistributionContrast(tests, settings, pd.DataFrame(shapes, columns=SHAPE_COLUMNS))


def spread_words(row: Mapping[str, Any]) -> str:
    """``SD 0.73x the control (0.70 to 0.78)``: the spread on the ratio scale."""
    mean, low, high = (float(np.exp(row[k])) for k in ("mean", "low", "high"))
    return f"SD {mean:.2f}x the control ({low_to_high(f'{low:.2f}', f'{high:.2f}')})"


def shift_words(row: Mapping[str, Any]) -> str:
    """``10% quantile +0.36 SD (+0.29 to +0.51)``."""
    name = f"{row['statistic'][1:]}% quantile"
    spread = low_to_high(f"{row['low']:+.2f}", f"{row['high']:+.2f}")
    return f"{name} {row['mean']:+.2f} SD ({spread})"


def tests_words(result: DistributionContrast) -> str:
    """The correction family counted: ``40 tests``, ``40 tests in each of 2 sub-experiments``,
    ``16 to 48 tests in each of 2 sub-experiments`` when their sizes differ."""
    tests = result.tests
    sizes = tests.loc[tests["p"].notna()].groupby("experiment").size() if len(tests) else []
    if len(sizes) <= 1:
        return count(result.n_tests, "test")
    low, high = int(min(sizes)), int(max(sizes))
    per = count(low, "test") if low == high else f"{low} to {high} tests"
    return f"{per} in each of {len(sizes)} {family_noun(result.settings, plural=True)}"


__all__ = [
    "CELLS_CAP",
    "DRAWS",
    "QUANTILES",
    "SHAPE_COLUMNS",
    "SPLIT_COLUMNS",
    "STATISTICS",
    "DistributionContrast",
    "distribution_contrast",
    "shift_words",
    "spread_words",
    "tests_words",
    "unassessed_words",
]
