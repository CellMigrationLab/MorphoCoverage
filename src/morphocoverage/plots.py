"""PlotsOfData-style plots of single features, with the field as the sampling unit.

Postma and Goedhart's PlotsOfData shows every value as a point beside a
summary and its 95% confidence interval; SuperPlots keep the biological
repeats apart. These plots do the same for one feature per figure: within
each condition the biological repeats (the repeat column) sit side by side,
every object is a jittered point coloured by technical repeat (the
MorphoBricks ``technical_repeat`` column, a well or a dish, or the column
``technical_col`` names; never a biological repeat), and each biological
repeat's summary (median by default, or mean) carries its interval from
resampling that repeat's whole technical repeats, each with its fields as
acquired (``statistics.bootstrap_nested``, Ren et al. 2010; whole fields
alone when there is one technical repeat), never objects, so the interval
reflects how the data were sampled: a well-to-well shift widens it instead of
passing for field-to-field noise. A cluster that pools several biological
repeats resamples the repeats alone, each with all the fields of all its
technical repeats as acquired (``bootstrap_nested`` with the repeats as the
units): resampling the technical repeats again within each drawn repeat would
count their variance twice. Acquired fields without observations take part,
as in ``statistics.bootstrap_fields``, when the FOV table places them in the
cluster. The title says how many units a cluster's interval resamples, and
warns when one has only 2 or 3, which no resampling turns into a reliable
interval, or a single technical repeat, whose interval shows field-to-field
variation only.

Every column of FOV identity that the figure does not place (condition,
repeat or technical repeat) is pooled within a cluster, never merged: with
``repeat_col=None`` on several repeats that reuse well names, well ``A`` of
each repeat stays its own technical repeat, and the repeats are resampled as
pooled repeats, which the title says. An analysis-group column (``group_cols``)
that the figure does not place raises instead, since the figure would pool
the groups the caller asked to keep apart: name it as ``condition_col``,
``repeat_col`` or ``technical_col``, or leave it out of ``group_cols``.

Limits keep the figures legible and are arguments: at most ``max_conditions``
conditions per figure (more are split over several figures), at most
``max_repeats`` biological repeats side by side (more are pooled into one
cluster per condition, and the title says so), at most ``max_technical``
colours (up to 8; more technical repeats of a biological repeat share one
grey), at most ``max_features`` features per call (more raise, so the caller
chooses). A column named explicitly that the tables do not hold raises; only
the default ``condition`` and ``biological_repeat`` columns may be absent. The layout's
columns are read from the FOV table (or, when only the observations carry
one, from each field's observations, which must agree); a field with no
value in one of them raises, since it cannot be placed. Every cluster draws
its jitter and resamples from its own generator, seeded from
``random_state``, the feature, the condition and the repeat, so a figure does
not change with the other features or conditions of the call. Figures are
``matplotlib.figure.Figure`` objects built without pyplot, so they display in
a notebook and save with ``savefig`` without touching global state.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ._checks import is_integer
from ._former_repeat import one_warning, read_former
from ._seeds import group_rng
from ._wording import ordinal, rate
from .data_check import validate_input_tables
from .metadata import (
    MORPHOBRICKS_GROUP_COLUMNS,
    REPEAT_COLUMN,
    TECHNICAL_REPEAT_COLUMN,
    empty_values,
    sorted_labels,
)
from .schema import ValidatedInput, fov_positions, fov_values
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
    """How the tables split into clusters: the columns used and their distinct values.

    ``pooled_cols`` are the columns of FOV identity above the technical
    repeat that a cluster pools (the repeat column when there are too many
    repeats to show, identity columns the figure does not place): the
    cluster's top-level units, resampled as wholes.
    """

    condition_col: str | None
    repeat_col: str | None
    technical_col: str | None
    conditions: list[Any]
    repeats: list[Any]
    pooled_repeats: bool
    max_repeats: int
    pooled_cols: tuple[str, ...] = ()


def _field_metadata(data: ValidatedInput, columns: Sequence[str | None]) -> pd.DataFrame:
    """The layout's columns for every acquired field, from the FOV table.

    A column only the observations carry is taken from each field's
    observations, which must agree; a field without observations then has no
    value there and cannot be placed in a cluster. A missing value in the
    FOV table, or in the observations, raises: that field cannot be placed.
    """
    fields = data.fovs.reset_index(drop=True)
    positions = fov_positions(data, data.fovs)
    meta = pd.DataFrame(index=fields.index)
    for column in dict.fromkeys(c for c in columns if c is not None):
        source = fields[column] if column in fields.columns else data.objects[column]
        missing = int(empty_values(source).sum())
        if missing:
            table = "FOV table" if column in fields.columns else "observations"
            raise ValueError(
                f"{missing} rows of the {table} have no {column!r} value, so their fields "
                "cannot be placed in the figure; fill it in"
            )
        if column in fields.columns:
            meta[column] = fields[column].to_numpy()
            continue
        by_field = pd.Series(data.objects[column].to_numpy()).groupby(positions)
        if (by_field.nunique() > 1).any():
            raise ValueError(
                f"{column!r} varies within a field's observations; a field lies in one "
                "cluster, so give each field one value (in the FOV table)"
            )
        meta[column] = by_field.first().reindex(meta.index).to_numpy()
    return meta


def _column(columns: set[str], column: str | None, default: str | None, name: str) -> str | None:
    """``column`` when the tables have it, ``None`` when it is an absent default; else raise."""
    if column is None or column in columns:
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
    control: Any = None,
) -> _Layout:
    """Resolve the condition, repeat and technical columns and their values.

    The technical-repeat column is ``technical_col``, else the MorphoBricks
    ``technical_repeat`` column when present, never another column. An
    analysis-group column the layout does not place raises. Conditions and
    repeats are drawn in sorted order, the ``control`` condition first, never
    in the order of the table's rows.
    """
    columns = set(data.objects.columns) | set(data.fovs.columns)
    # condition_col=None or repeat_col=None asks for that column to be pooled.
    pooled_on_request = [
        default
        for default, argument in zip(MORPHOBRICKS_GROUP_COLUMNS[:2], (condition_col, repeat_col))
        if argument is None
    ]
    condition_col = _column(columns, condition_col, MORPHOBRICKS_GROUP_COLUMNS[0], "condition_col")
    repeat_col = _column(columns, repeat_col, MORPHOBRICKS_GROUP_COLUMNS[1], "repeat_col")
    if technical_col is None:
        technical_col = TECHNICAL_REPEAT_COLUMN if TECHNICAL_REPEAT_COLUMN in columns else None
    else:
        technical_col = _column(columns, technical_col, None, "technical_col")
    placed = (condition_col, repeat_col, technical_col)
    unplaced = [c for c in data.group_cols if c not in (*placed, *pooled_on_request)]
    if unplaced:
        raise ValueError(
            f"group_cols {list(data.group_cols)} hold {unplaced}, which the figure does not "
            "place, so its clusters would pool those groups; name the column as condition_col, "
            "repeat_col or technical_col, or leave it out of group_cols to pool it"
        )
    meta = _field_metadata(data, placed)
    observed = meta.iloc[fov_positions(data, data.fovs)]
    conditions = sorted_labels(observed[condition_col], control) if condition_col else [None]
    repeats = sorted_labels(observed[repeat_col]) if repeat_col else [None]
    pooled = len(repeats) > max_repeats
    identity = [c for c in data.fov_key if c != data.fov_col and c not in placed]
    pooled_cols = ([repeat_col] if pooled else []) + identity
    return _Layout(
        condition_col,
        repeat_col,
        technical_col,
        conditions,
        [None] if pooled else repeats,
        pooled,
        max_repeats,
        tuple(pooled_cols),
    )


def _mask(table: pd.DataFrame, column: str | None, value: Any) -> np.ndarray:
    """The rows holding ``value`` in ``column``; every row when either is absent."""
    if column is None or value is None:
        return np.ones(len(table), dtype=bool)
    return table[column].to_numpy() == value


def _unit_keys(frame: pd.DataFrame, mask: np.ndarray, layout: _Layout) -> list[tuple[Any, Any]]:
    """``(top-level unit, technical repeat)`` of each row of ``frame`` in ``mask``.

    The top-level unit is the tuple of the layout's pooled columns (empty
    when nothing is pooled), so a technical repeat is identified within it:
    well ``B02`` of two pooled repeats is two technical repeats.
    """
    rows = frame.loc[mask]
    tops = list(zip(*(rows[c] for c in layout.pooled_cols), strict=True))
    tops = tops if layout.pooled_cols else [()] * len(rows)
    technical = rows[layout.technical_col] if layout.technical_col else [None] * len(rows)
    return list(zip(tops, technical, strict=True))


def _unit_mask(frame: pd.DataFrame, layout: _Layout, top: tuple, replicate: Any) -> np.ndarray:
    """The rows of ``frame`` in one ``(top-level unit, technical repeat)``."""
    mask = _mask(frame, layout.technical_col, replicate)
    for column, value in zip(layout.pooled_cols, top, strict=True):
        mask &= frame[column].to_numpy() == value
    return mask


def _draw_cluster(
    ax: Axes,
    x: float,
    width: float,
    rows: np.ndarray,
    values: np.ndarray,
    fields: tuple[pd.DataFrame, pd.DataFrame, list[np.ndarray], np.ndarray],
    layout: _Layout,
    colours: Sequence[str],
    statistic: Callable[[np.ndarray], float],
    n_boot: int,
    confidence: float,
    rng: np.random.Generator,
) -> None:
    """One cluster (a condition and biological repeat): technical-repeat points, summary, interval.

    ``rows`` is the boolean mask of the cluster's observations; ``fields``
    holds every acquired field's metadata, every observation's (its field's),
    every field's values, and the boolean mask of the cluster's fields, empty
    ones included.
    """
    meta, observed, per_field, in_cluster = fields
    # Units in the order their observations come, then units whose fields are all empty.
    units = dict.fromkeys(
        [*_unit_keys(observed, rows, layout), *_unit_keys(meta, in_cluster, layout)]
    )
    half = 0.4 * width
    by_top: dict[tuple, list[list[np.ndarray]]] = {}
    for top, replicate in units:
        # Colours by rank within the top-level unit (the biological repeat), as the legend says.
        rank = len(by_top.setdefault(top, []))
        unit_fields = in_cluster & _unit_mask(meta, layout, top, replicate)
        by_top[top].append([per_field[i] for i in np.flatnonzero(unit_fields)])
        y = values[rows & _unit_mask(observed, layout, top, replicate)]
        y = y[~np.isnan(y)]
        if not len(y):
            continue
        colour = colours[rank] if rank < len(colours) else GREY
        jitter = rng.uniform(-half, half, size=len(y))
        ax.scatter(x + jitter, y, s=6, alpha=0.35, color=colour, linewidths=0, rasterized=True)
    tops = [t for t in by_top.values() if any(len(f) for unit in t for f in unit)]
    if len(tops) > 1:
        # Pooled biological repeats: the repeats alone are resampled, each with all the
        # fields of all its technical repeats as acquired (Ren et al. 2010).
        units_resampled = [[f for unit in top for f in unit] for top in tops]
    else:
        units_resampled = tops[0] if tops else [[]]
    point, low, high = bootstrap_nested(
        units_resampled, statistic, n_boot=n_boot, rng=rng, confidence=confidence
    )
    if np.isfinite(low) and np.isfinite(high):
        # A thick bar with caps behind the summary line, so a narrow interval stays visible.
        ax.vlines(x, low, high, color="black", lw=7, alpha=0.3, zorder=2)
        ax.hlines([low, high], x - 0.3 * width, x + 0.3 * width, color="black", lw=1, zorder=3)
    if np.isfinite(point):
        ax.hlines(point, x - half, x + half, color="black", lw=2, zorder=4)


def _cluster_units(observed: pd.DataFrame, layout: _Layout) -> list[tuple[int, list[int]]]:
    """Per cluster: its top-level units with observations, and each one's technical repeats."""
    shown = [layout.condition_col] + ([] if layout.pooled_repeats else [layout.repeat_col])
    cluster_cols = [c for c in shown if c is not None]
    top_cols = [*cluster_cols, *layout.pooled_cols]
    clusters = observed.groupby(cluster_cols, sort=False) if cluster_cols else [((), observed)]
    result = []
    for _, cluster in clusters:
        tops = cluster.groupby(top_cols, sort=False) if top_cols else [((), cluster)]
        technical = [
            int(top[layout.technical_col].nunique()) if layout.technical_col else 0
            for _, top in tops
        ]
        result.append((len(technical), technical))
    return result


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
                label=f"{ordinal(rank + 1)} {layout.technical_col}",
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


def _span(counts: Sequence[int]) -> str:
    """``3`` or ``2 to 5``."""
    return f"{min(counts)}" if min(counts) == max(counts) else f"{min(counts)} to {max(counts)}"


def _pooled_name(layout: _Layout) -> str:
    """The pooled top-level units in words: ``biological repeats (repeat)`` or their columns."""
    if len(layout.pooled_cols) == 1 and layout.pooled_cols[0] in (
        layout.repeat_col,
        REPEAT_COLUMN,
    ):
        return f"biological repeats ({layout.pooled_cols[0]})"
    return f"units of {'/'.join(layout.pooled_cols)}"


def _title(
    feature: str,
    part: str,
    layout: _Layout,
    unit: str,
    summary: str,
    level: float,
    clusters: Sequence[tuple[int, list[int]]] = (),
) -> str:
    """Three lines: what is plotted and how it is split, what a point is, what the interval is.

    ``clusters`` holds, per cluster, its top-level units and each one's
    technical repeats (``_cluster_units``): the interval line says how many
    units are resampled, and a fourth line warns when a cluster has only 2 or
    3, which no resampling turns into a reliable interval, or a single
    technical repeat, whose interval resamples its fields and so shows
    field-to-field variation only (it covered the repeat's median about half
    the time with a well SD above the field SD).
    """
    repeats = ""
    if layout.pooled_repeats:
        repeats = (
            f"; biological repeats ({layout.repeat_col}) pooled, more than the "
            f"{layout.max_repeats} shown side by side"
        )
    elif layout.repeat_col:
        repeats = f"; biological repeats ({layout.repeat_col}) side by side"
    unshown = [
        c for c in layout.pooled_cols if not (layout.pooled_repeats and c == layout.repeat_col)
    ]
    if unshown:
        repeats += f"; {'/'.join(unshown)} not shown apart, pooled within each cluster"
    by_top = [n for n, _ in clusters if n > 1]
    single = [technical[0] for n, technical in clusters if n == 1]
    coloured, phrases, caveat = "", [], ""
    if layout.technical_col:
        coloured = f", coloured by technical repeat ({layout.technical_col})"
    if by_top:
        phrases.append(
            f"whole {_pooled_name(layout)}, each with all its fields as acquired "
            f"({_span(by_top)} per cluster)"
        )
        if any(n in (2, 3) for n in by_top):
            caveat = f"\nwith only 2-3 {_pooled_name(layout)} a cluster's interval is not reliable"
    if single or not by_top:
        if layout.technical_col:
            span = f"{_span(single)} per cluster; " if single else ""
            phrases.append(
                f"whole technical repeats with their fields as acquired ({span}whole fields "
                "where there is one)"
            )
            if any(n in (2, 3) for n in single):
                caveat = "\nwith only 2-3 technical repeats a cluster's interval is not reliable"
            if any(n == 1 for n in single):
                caveat += (
                    "\none technical repeat: the interval shows field-to-field variation "
                    "only, not well-to-well"
                )
        else:
            phrases.append("whole fields")
    resampled = " or, in a cluster of one, ".join(phrases)
    return (
        f"{feature}{part}{repeats}\nevery {unit} a point{coloured}\n"
        f"{summary} with its {rate(level)} interval from resampling {resampled}{caveat}"
    )


@one_warning
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
    control: Mapping[str, Any] | None = None,
) -> dict[str, list[Figure]]:
    """One or more figures per feature (see the module); keys are the feature names.

    ``features`` must be columns of the tables and at most ``max_features``.
    ``repeat_col`` holds the biological repeats, drawn side by side.
    ``technical_col`` names the technical-repeat column (a well, a dish;
    never a biological repeat), in the FOV table or the observations; by
    default the MorphoBricks ``technical_repeat`` column, else none (another
    column is never taken for it). Without a condition column every object
    is one group; without a repeat column each condition is one cluster.
    ``condition_col`` and ``repeat_col`` may name absent columns only at
    their defaults; any other absent column raises, and so does an
    analysis-group column none of the three names.
    ``value_note`` is added to the value axis label, for example
    ``"relative to control"`` for normalised values. Conditions are drawn in
    sorted order, the ``control`` condition (``{"condition": "MOCK"}``) first.
    """
    (repeat_col,) = read_former(repeat_col=repeat_col)  # the former name ``repeat``
    if not (features := list(features)):
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
        if not is_integer(value) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    n_boot, max_conditions = int(n_boot), int(max_conditions)
    max_repeats, max_technical = int(max_repeats), int(max_technical)
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
    first = dict(control or {}).get(condition_col) if condition_col is not None else None
    layout = _layout(data, condition_col, repeat_col, technical_col, max_repeats, first)
    colours = PALETTE[:max_technical]
    meta = _field_metadata(
        data,
        [layout.condition_col, layout.repeat_col, layout.technical_col, *layout.pooled_cols],
    )
    observed = meta.iloc[fov_positions(data, data.fovs)].reset_index(drop=True)
    clusters = _cluster_units(observed, layout)
    n_technical = max((n for _, technical in clusters for n in technical), default=0)
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
                in_condition = _mask(observed, layout.condition_col, condition)
                fields_in_condition = _mask(meta, layout.condition_col, condition)
                for j, repeat in enumerate(layout.repeats):
                    rows = in_condition & _mask(observed, layout.repeat_col, repeat)
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
                        rows,
                        values,
                        (meta, observed, per_field, in_cluster),
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
                _title(feature, part, layout, data.observation_unit, summary, confidence, clusters),
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
