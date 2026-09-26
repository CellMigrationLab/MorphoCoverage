"""PlotsOfData-style plots of single features, with the field as the sampling unit.

Postma and Goedhart's PlotsOfData shows every value as a point beside a
summary and its 95% confidence interval; SuperPlots keep the biological
repeats apart. These plots do the same for one feature per figure: within
each condition the biological repeats (the repeat column) sit side by side,
every object is a jittered point coloured by technical repeat (the
MorphoBricks ``technical_repeat`` column, a well or a dish, else the group
column below the repeat; never a biological repeat), and each biological
repeat's summary (median by default, or mean) carries its interval from
resampling that repeat's technical repeats and then their whole fields
(``statistics.bootstrap_nested``; whole fields alone when there is one
technical repeat), never objects, so the interval reflects how the data were
sampled: a well-to-well shift widens it instead of passing for
field-to-field noise.

Limits keep the figures legible and are arguments: at most ``max_conditions``
conditions per figure (more are split over several figures), at most
``max_repeats`` biological repeats side by side (more are pooled into one
cluster per condition, and the title says so), at most ``max_technical``
colours (more technical repeats share one grey), at most ``max_features`` features per
call (more raise, so the caller chooses). Figures are
``matplotlib.figure.Figure`` objects built without pyplot, so they display
in a notebook and save with ``savefig`` without touching global state.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .metadata import MORPHOBRICKS_GROUP_COLUMNS, TECHNICAL_REPEAT_COLUMN
from .schema import ValidatedInput, fov_positions, validate_input_tables
from .statistics import bootstrap_nested

MAX_CONDITIONS = 12
MAX_REPEATS = 8
MAX_TECHNICAL = 8
MAX_FEATURES = 24
SUMMARIES: dict[str, Callable[[np.ndarray], float]] = {"median": np.median, "mean": np.mean}
PALETTE = ("#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860", "#DA8BC3", "#8C8C8C")
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


def _fields_of(data: ValidatedInput, rows: np.ndarray, values: np.ndarray) -> list[np.ndarray]:
    """The values of ``rows`` (a boolean mask over observations) split by acquired field."""
    positions = fov_positions(data, data.fovs)
    chosen = rows & ~np.isnan(values)
    return [values[chosen & (positions == p)] for p in np.unique(positions[rows])]


def _present(table: pd.DataFrame, column: str | None) -> str | None:
    """``column`` when the table has it, else ``None``."""
    return column if column is not None and column in table.columns else None


def _layout(
    data: ValidatedInput,
    condition_col: str | None,
    repeat_col: str | None,
    technical_col: str | None,
    max_repeats: int,
) -> _Layout:
    """Resolve the condition, repeat and technical columns and their values."""
    table = data.objects
    condition_col, repeat_col = _present(table, condition_col), _present(table, repeat_col)
    if technical_col is None:
        # The MorphoBricks technical repeat, else the first analysis-group column
        # below condition and repeat (a well, say).
        below = [c for c in data.group_cols if c not in (condition_col, repeat_col)]
        if TECHNICAL_REPEAT_COLUMN in table.columns:
            technical_col = TECHNICAL_REPEAT_COLUMN
        else:
            technical_col = below[0] if below else None
    technical_col = _present(table, technical_col)
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
    layout: _Layout,
    colours: Sequence[str],
    statistic: Callable[[np.ndarray], float],
    n_boot: int,
    confidence: float,
    rng: np.random.Generator,
) -> None:
    """One cluster (a condition and biological repeat): technical-repeat points, summary, interval.

    ``rows`` is the boolean mask of the cluster's observations.
    """
    table = data.objects
    replicates = (
        list(dict.fromkeys(table.loc[rows, layout.technical_col].tolist()))
        if layout.technical_col
        else [None]
    )
    half = 0.4 * width
    units = []
    for rank, replicate in enumerate(replicates):
        in_unit = rows & _mask(table, layout.technical_col, replicate)
        units.append(_fields_of(data, in_unit, values))
        y = values[in_unit]
        y = y[~np.isnan(y)]
        if not len(y):
            continue
        colour = colours[rank] if rank < len(colours) else GREY
        jitter = rng.uniform(-half, half, size=len(y))
        ax.scatter(x + jitter, y, s=6, alpha=0.35, color=colour, linewidths=0, rasterized=True)
    point, low, high = bootstrap_nested(
        units, statistic, n_boot=n_boot, rng=rng, confidence=confidence
    )
    if np.isfinite(low) and np.isfinite(high):
        # A thick bar with caps behind the summary line, so a narrow interval stays visible.
        ax.vlines(x, low, high, color="black", lw=7, alpha=0.3, zorder=2)
        ax.hlines([low, high], x - 0.3 * width, x + 0.3 * width, color="black", lw=1, zorder=3)
    if np.isfinite(point):
        ax.hlines(point, x - half, x + half, color="black", lw=2, zorder=4)


def _technical_count(table: pd.DataFrame, layout: _Layout) -> int:
    """The most technical repeats any condition-and-biological-repeat cluster holds."""
    if layout.technical_col is None:
        return 0
    # Pooled repeats form one cluster per condition, so they do not split the count.
    repeat_col = None if layout.pooled_repeats else layout.repeat_col
    keys = [c for c in (layout.condition_col, repeat_col) if c is not None]
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
            f"{n_technical} technical repeats ({layout.technical_col}): more than "
            f"{len(colours)}, not coloured",
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
) -> dict[str, list[Figure]]:
    """One or more figures per feature (see the module); keys are the feature names.

    ``features`` must be columns of the tables and at most ``max_features``.
    ``repeat_col`` holds the biological repeats, drawn side by side.
    ``technical_col`` names the technical-repeat column (a well, a dish;
    never a biological repeat); by default the MorphoBricks
    ``technical_repeat`` column, else the first analysis-group column below
    condition and repeat, or none. Without a condition column every object
    is one group; without a repeat column each condition is one cluster.
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
    rng = np.random.default_rng(random_state)
    statistic = SUMMARIES[summary]
    n_repeats = len(layout.repeats)
    width = 0.8 / n_repeats
    figures: dict[str, list[Figure]] = {}
    for feature in features:
        values = data.feature_values(feature)
        figures[feature] = []
        for start in range(0, len(layout.conditions), max_conditions):
            chunk = layout.conditions[start : start + max_conditions]
            figure = Figure(figsize=(max(6.5, 0.9 * len(chunk) * n_repeats + 2.0), 4.6))
            ax = figure.add_subplot(111)
            repeat_labels: list[tuple[float, str]] = []
            for index, condition in enumerate(chunk):
                in_condition = _mask(table, layout.condition_col, condition)
                for j, repeat in enumerate(layout.repeats):
                    rows = in_condition & _mask(table, layout.repeat_col, repeat)
                    if not rows.any():
                        continue
                    x = index - 0.4 + width * (j + 0.5)
                    _draw_cluster(
                        ax,
                        x,
                        width,
                        data,
                        rows,
                        values,
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
            ax.set_ylabel(feature)
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
