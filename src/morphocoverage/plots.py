"""PlotsOfData-style plots of single features, with the field as the sampling unit.

Postma and Goedhart's PlotsOfData shows every value as a point beside a
summary and its 95% confidence interval; SuperPlots keep the biological
repeats apart. These plots do the same for one feature per figure: within
each condition the biological repeats (the repeat column) sit side by side,
every object is a jittered point coloured by technical repeat (the
MorphoBricks ``technical_repeat`` column, a well or a dish, else the group
column below the repeat; never a biological repeat), and each biological
repeat's summary (median by default, or mean) carries its interval from
resampling that repeat's technical repeats with replacement, each keeping its
whole fields (``statistics.bootstrap_nested``; whole fields alone when there
is one technical repeat), never objects, so the interval reflects how the
data were sampled: a well-to-well shift widens it instead of passing for
field-to-field noise. Acquired fields without observations take part, as in
``statistics.bootstrap_fields``, when the FOV table places them in the
cluster.

Limits keep the figures legible and are arguments: at most ``max_conditions``
conditions per figure (more are split over several figures), at most
``max_repeats`` biological repeats side by side (more are pooled into one
cluster per condition, and the title says so, each technical repeat staying
within its biological repeat; the interval then resamples the biological
repeats first, and within each drawn repeat its technical repeats, then
their fields, as ``bootstrap_nested`` does for one repeat), at most
``max_technical`` colours (up to 8; more technical repeats of a biological
repeat share one grey), at most ``max_features`` features per call (more
raise, so the caller chooses). A column named explicitly that the tables do
not hold raises; only the default ``condition`` and ``repeat`` columns may be
absent. Every cluster draws its jitter and resamples from its own generator,
seeded from ``random_state``, the feature, the condition and the repeat, so
a figure does not change with the other features or conditions of the call.
Figures are ``matplotlib.figure.Figure`` objects built without pyplot, so
they display in a notebook and save with ``savefig`` without touching global
state.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ._seeds import group_rng
from .data_check import validate_input_tables
from .metadata import MORPHOBRICKS_GROUP_COLUMNS, TECHNICAL_REPEAT_COLUMN
from .sampling_unit._common import fov_values
from .schema import ValidatedInput, fov_positions
from .statistics import bootstrap_nested

MAX_CONDITIONS = 12
MAX_REPEATS = 8
MAX_TECHNICAL = 8
MAX_FEATURES = 24
SUMMARIES: dict[str, Callable[[np.ndarray], float]] = {"median": np.median, "mean": np.mean}
# No grey in the palette: grey marks the technical repeats beyond it.
PALETTE = ("#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860", "#DA8BC3", "#64B5CD")
GREY = "0.6"


@dataclass(frozen=True)
class _Layout:
    """How the tables split into clusters: the columns used and their distinct values."""

    condition_col: str | None
    repeat_col: str | None
    technical_col: str | None
    conditions: list[Any]
    repeats: list[Any]
    pooled_repeats: bool
    max_repeats: int


def _field_metadata(data: ValidatedInput, columns: Sequence[str | None]) -> pd.DataFrame:
    """The layout's columns for every acquired field, from the FOV table.

    A column only the observations carry is taken from each field's first
    observation; a field without observations then has no value there and
    cannot be placed in a cluster.
    """
    fields = data.fovs.reset_index(drop=True)
    positions = fov_positions(data, data.fovs)
    meta = pd.DataFrame(index=fields.index)
    for column in dict.fromkeys(c for c in columns if c is not None):
        if column in fields.columns:
            meta[column] = fields[column].to_numpy()
        else:
            first = pd.Series(data.objects[column].to_numpy()).groupby(positions).first()
            meta[column] = first.reindex(meta.index).to_numpy()
    return meta


def _column(table: pd.DataFrame, column: str | None, default: str | None, name: str) -> str | None:
    """``column`` when the table has it, ``None`` when it is an absent default; else raise."""
    if column is None or column in table.columns:
        return column
    if column == default:
        return None
    raise ValueError(
        f"{name} {column!r} is not a column of the tables; name an existing column, or pass "
        f"{name}=None"
    )


def _layout(
    data: ValidatedInput,
    condition_col: str | None,
    repeat_col: str | None,
    technical_col: str | None,
    max_repeats: int,
) -> _Layout:
    """Resolve the condition, repeat and technical columns and their values."""
    table = data.objects
    condition_col = _column(table, condition_col, MORPHOBRICKS_GROUP_COLUMNS[0], "condition_col")
    repeat_col = _column(table, repeat_col, MORPHOBRICKS_GROUP_COLUMNS[1], "repeat_col")
    if technical_col is None:
        # The MorphoBricks technical repeat, else the first analysis-group column
        # below condition and repeat (a well, say).
        below = [c for c in data.group_cols if c not in (condition_col, repeat_col)]
        if TECHNICAL_REPEAT_COLUMN in table.columns:
            technical_col = TECHNICAL_REPEAT_COLUMN
        else:
            technical_col = below[0] if below else None
    else:
        technical_col = _column(table, technical_col, None, "technical_col")
    conditions = list(dict.fromkeys(table[condition_col].tolist())) if condition_col else [None]
    repeats = list(dict.fromkeys(table[repeat_col].tolist())) if repeat_col else [None]
    pooled = len(repeats) > max_repeats
    return _Layout(
        condition_col,
        repeat_col,
        technical_col,
        conditions,
        [None] if pooled else repeats,
        pooled,
        max_repeats,
    )


def _ordinal(n: int) -> str:
    """``1st``, ``2nd``, ``3rd``, ``4th``... for legend labels."""
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _mask(table: pd.DataFrame, column: str | None, value: Any) -> np.ndarray:
    """The rows holding ``value`` in ``column``; every row when either is absent."""
    if column is None or value is None:
        return np.ones(len(table), dtype=bool)
    return table[column].to_numpy() == value


def _draw_cluster(
    ax: Axes,
    x: float,
    width: float,
    data: ValidatedInput,
    rows: np.ndarray,
    values: np.ndarray,
    fields: tuple[pd.DataFrame, list[np.ndarray], np.ndarray],
    layout: _Layout,
    colours: Sequence[str],
    statistic: Callable[[np.ndarray], float],
    n_boot: int,
    confidence: float,
    rng: np.random.Generator,
) -> None:
    """One cluster (a condition and biological repeat): technical-repeat points, summary, interval.

    ``rows`` is the boolean mask of the cluster's observations; ``fields``
    holds every acquired field's metadata and values, and the boolean mask of
    the cluster's fields, empty ones included.
    """
    table = data.objects
    meta, per_field, in_cluster = fields
    # A technical repeat lies within its biological repeat: when repeats are pooled
    # into one cluster, well "B02" of two repeats is two units, not one.
    repeat_col = layout.repeat_col if layout.pooled_repeats else None
    technical_col = layout.technical_col

    def keys(frame: pd.DataFrame, mask: np.ndarray) -> list[tuple[Any, Any]]:
        n = int(mask.sum())
        return list(
            zip(
                frame.loc[mask, repeat_col] if repeat_col else [None] * n,
                frame.loc[mask, technical_col] if technical_col else [None] * n,
                strict=True,
            )
        )

    # Units in the order their observations come, then units whose fields are all empty.
    replicates = list(dict.fromkeys([*keys(table, rows), *keys(meta, in_cluster)]))
    half = 0.4 * width
    by_repeat: dict[Any, list[list[np.ndarray]]] = {}
    ranks: dict[Any, int] = {}
    for repeat, replicate in replicates:
        # Colours by rank within the biological repeat, as the legend says.
        rank = ranks[repeat] = ranks.get(repeat, -1) + 1
        unit_fields = in_cluster & _mask(meta, repeat_col, repeat)
        unit_fields &= _mask(meta, technical_col, replicate)
        by_repeat.setdefault(repeat, []).append([per_field[i] for i in np.flatnonzero(unit_fields)])
        in_unit = rows & _mask(table, repeat_col, repeat) & _mask(table, technical_col, replicate)
        y = values[in_unit]
        y = y[~np.isnan(y)]
        if not len(y):
            continue
        colour = colours[rank] if rank < len(colours) else GREY
        jitter = rng.uniform(-half, half, size=len(y))
        ax.scatter(x + jitter, y, s=6, alpha=0.35, color=colour, linewidths=0, rasterized=True)
    repeats = list(by_repeat.values())
    if len(repeats) > 1:
        point, low, high = _bootstrap_repeats(
            repeats, statistic, n_boot=n_boot, rng=rng, confidence=confidence
        )
    else:
        point, low, high = bootstrap_nested(
            repeats[0], statistic, n_boot=n_boot, rng=rng, confidence=confidence
        )
    if np.isfinite(low) and np.isfinite(high):
        # A thick bar with caps behind the summary line, so a narrow interval stays visible.
        ax.vlines(x, low, high, color="black", lw=7, alpha=0.3, zorder=2)
        ax.hlines([low, high], x - 0.3 * width, x + 0.3 * width, color="black", lw=1, zorder=3)
    if np.isfinite(point):
        ax.hlines(point, x - half, x + half, color="black", lw=2, zorder=4)


def _bootstrap_repeats(
    repeats: Sequence[Sequence[Sequence[np.ndarray]]],
    statistic: Callable[[np.ndarray], float],
    *,
    n_boot: int,
    rng: np.random.Generator,
    confidence: float,
) -> tuple[float, float, float]:
    """``statistic`` of pooled biological repeats, and its interval from resampling them first.

    ``repeats`` holds, per biological repeat, its technical repeats, each a
    list of its fields' values. Each resample draws as many repeats as there
    are, with replacement, and within each drawn repeat does what
    ``statistics.bootstrap_nested`` does for one: its technical repeats with
    replacement, each keeping its fields, or its fields with replacement when
    it has a single technical repeat. A repeat-to-repeat shift then widens the
    interval instead of passing for field-to-field noise (resampling only the
    fields of pooled repeats covered the mean 60% of the time for a nominal
    95%). Resamples holding no object are left out of the percentiles; NaN
    bounds when fewer than two fields hold objects.
    """
    levels = [
        [[np.asarray(f, dtype=float) for f in unit] for unit in r if len(unit)] for r in repeats
    ]
    levels = [r for r in levels if r]
    fields = [f for r in levels for unit in r for f in unit]
    pooled = np.concatenate(fields) if fields else np.array([])
    point = float(statistic(pooled)) if len(pooled) else float("nan")
    if sum(len(f) > 0 for f in fields) < 2:
        return point, float("nan"), float("nan")
    draws = np.empty(n_boot)
    for b in range(n_boot):
        parts = []
        for r in rng.integers(0, len(levels), size=len(levels)):
            units = levels[r]
            if len(units) == 1:
                chosen = rng.integers(0, len(units[0]), size=len(units[0]))
                parts += [units[0][i] for i in chosen]
            else:
                for u in rng.integers(0, len(units), size=len(units)):
                    parts += units[u]
        sample = np.concatenate(parts) if parts else np.array([])
        draws[b] = statistic(sample) if len(sample) else np.nan
    defined = draws[np.isfinite(draws)]
    if len(defined) < 2:
        return point, float("nan"), float("nan")
    tail = 100 * (1 - confidence) / 2
    low, high = np.percentile(defined, [tail, 100 - tail])
    return point, float(low), float(high)


def _technical_count(table: pd.DataFrame, layout: _Layout) -> int:
    """The most technical repeats any condition-and-biological-repeat cluster holds."""
    if layout.technical_col is None:
        return 0
    # Colours are ranked within each biological repeat, pooled or not, so the count is
    # per repeat: well names of different repeats are different wells.
    keys = [c for c in (layout.condition_col, layout.repeat_col) if c is not None]
    if not keys:
        return int(table[layout.technical_col].nunique())
    return int(table.groupby(keys, sort=False)[layout.technical_col].nunique().max())


def _legend(ax: Axes, layout: _Layout, colours: Sequence[str], n_technical: int) -> None:
    """Colours by technical-repeat rank, or a note when there are too many to colour."""
    if layout.technical_col is None:
        return
    if n_technical <= len(colours):
        handles = [
            ax.scatter(
                [],
                [],
                color=colours[rank],
                s=20,
                label=f"{_ordinal(rank + 1)} {layout.technical_col}",
            )
            for rank in range(n_technical)
        ]
        ax.legend(
            handles=handles,
            frameon=False,
            fontsize=8,
            title=(
                f"technical repeat ({layout.technical_col}), in order within each biological repeat"
            ),
            title_fontsize=8,
        )
    else:
        ax.text(
            0.01,
            0.99,
            f"up to {n_technical} technical repeats ({layout.technical_col}) per biological "
            f"repeat: the first {len(colours)} coloured in order, the others grey",
            transform=ax.transAxes,
            va="top",
            fontsize=8,
        )


def _title(feature: str, part: str, layout: _Layout, unit: str, summary: str, level: float) -> str:
    """Three lines: what is plotted and how it is split, what a point is, what the interval is."""
    repeats = ""
    if layout.pooled_repeats:
        repeats = (
            f"; biological repeats ({layout.repeat_col}) pooled, more than the "
            f"{layout.max_repeats} shown side by side"
        )
    elif layout.repeat_col:
        repeats = f"; biological repeats ({layout.repeat_col}) side by side"
    coloured, resampled = "", "whole fields"
    if layout.technical_col:
        coloured = f", coloured by technical repeat ({layout.technical_col})"
        resampled = "technical repeats, then their whole fields"
    if layout.pooled_repeats:
        resampled = f"biological repeats, then their {resampled}"
    return (
        f"{feature}{part}{repeats}\nevery {unit} a point{coloured}\n"
        f"{summary} with its {level:.0%} interval from resampling {resampled}"
    )


def plots_of_data(
    objects: pd.DataFrame,
    fovs: pd.DataFrame,
    *,
    features: Sequence[str],
    condition_col: str | None = MORPHOBRICKS_GROUP_COLUMNS[0],
    repeat_col: str | None = MORPHOBRICKS_GROUP_COLUMNS[1],
    technical_col: str | None = None,
    summary: str = "median",
    confidence: float = 0.95,
    n_boot: int = 1000,
    random_state: int = 0,
    max_conditions: int = MAX_CONDITIONS,
    max_repeats: int = MAX_REPEATS,
    max_technical: int = MAX_TECHNICAL,
    max_features: int = MAX_FEATURES,
    fov_col: str | None = None,
    group_cols: Sequence[str] | None = None,
    observation_unit: str | None = None,
    value_note: str | None = None,
) -> dict[str, list[Figure]]:
    """One or more figures per feature (see the module); keys are the feature names.

    ``features`` must be columns of the tables and at most ``max_features``.
    ``repeat_col`` holds the biological repeats, drawn side by side.
    ``technical_col`` names the technical-repeat column (a well, a dish;
    never a biological repeat); by default the MorphoBricks
    ``technical_repeat`` column, else the first analysis-group column below
    condition and repeat, or none. Without a condition column every object
    is one group; without a repeat column each condition is one cluster.
    ``condition_col`` and ``repeat_col`` may name absent columns only at
    their defaults; any other absent column raises.
    ``value_note`` is added to the value axis label, for example
    ``"relative to control"`` for normalised values.
    """
    features = list(features)
    if not features:
        raise ValueError("features must name at least one column")
    if len(features) > max_features:
        raise ValueError(
            f"{len(features)} features asked for, more than max_features={max_features}: "
            "choose the features to plot"
        )
    if summary not in SUMMARIES:
        raise ValueError(f"summary must be one of {sorted(SUMMARIES)}")
    for name, value in (
        ("n_boot", n_boot),
        ("max_conditions", max_conditions),
        ("max_repeats", max_repeats),
        ("max_technical", max_technical),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if max_technical > len(PALETTE):
        raise ValueError(
            f"max_technical can be at most {len(PALETTE)}, the colours that can be told apart"
        )
    data = validate_input_tables(
        objects,
        fovs=fovs,
        feature_cols=features,
        fov_col=fov_col,
        group_cols=group_cols,
        observation_unit=observation_unit,
    )
    table = data.objects
    layout = _layout(data, condition_col, repeat_col, technical_col, max_repeats)
    colours = PALETTE[: min(max_technical, len(PALETTE))]
    n_technical = _technical_count(table, layout)
    meta = _field_metadata(data, [layout.condition_col, layout.repeat_col, layout.technical_col])
    statistic = SUMMARIES[summary]
    n_repeats = len(layout.repeats)
    width = 0.8 / n_repeats
    figures: dict[str, list[Figure]] = {}
    for feature in features:
        values = data.feature_values(feature)
        per_field = fov_values(data, values)  # every acquired field, empty ones included
        figures[feature] = []
        for start in range(0, len(layout.conditions), max_conditions):
            chunk = layout.conditions[start : start + max_conditions]
            figure = Figure(figsize=(max(6.5, 0.9 * len(chunk) * n_repeats + 2.0), 4.6))
            ax = figure.add_subplot(111)
            repeat_labels: list[tuple[float, str]] = []
            for index, condition in enumerate(chunk):
                in_condition = _mask(table, layout.condition_col, condition)
                fields_in_condition = _mask(meta, layout.condition_col, condition)
                for j, repeat in enumerate(layout.repeats):
                    rows = in_condition & _mask(table, layout.repeat_col, repeat)
                    if not rows.any():
                        continue
                    in_cluster = fields_in_condition & _mask(meta, layout.repeat_col, repeat)
                    x = index - 0.4 + width * (j + 0.5)
                    # A generator per cluster: no figure depends on the others of the call.
                    rng = group_rng(
                        random_state,
                        {"feature": feature, "condition": condition, "repeat": repeat},
                    )
                    _draw_cluster(
                        ax,
                        x,
                        width,
                        data,
                        rows,
                        values,
                        (meta, per_field, in_cluster),
                        layout,
                        colours,
                        statistic,
                        n_boot,
                        confidence,
                        rng,
                    )
                    if repeat is not None:
                        repeat_labels.append((x, str(repeat)))
            many = len(chunk) > 4
            ax.set_xticks(range(len(chunk)))
            ax.set_xticklabels(
                ["all fields" if c is None else str(c) for c in chunk],
                rotation=30 if many else 0,
                ha="right" if many else "center",
            )
            for x, label in repeat_labels:
                ax.annotate(
                    label,
                    (x, 1.0),
                    xycoords=("data", "axes fraction"),
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color="0.35",
                )
            ax.set_ylabel(feature if value_note is None else f"{feature} ({value_note})")
            part = (
                f" (conditions {start + 1} to {start + len(chunk)} of {len(layout.conditions)})"
                if len(layout.conditions) > max_conditions
                else ""
            )
            ax.set_title(
                _title(feature, part, layout, data.observation_unit, summary, confidence),
                fontsize=9,
                loc="left",
                pad=14,
            )
            _legend(ax, layout, colours, n_technical)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            figure.tight_layout()
            figures[feature].append(figure)
    return figures


__all__ = [
    "MAX_CONDITIONS",
    "MAX_FEATURES",
    "MAX_REPEATS",
    "MAX_TECHNICAL",
    "SUMMARIES",
    "plots_of_data",
]
