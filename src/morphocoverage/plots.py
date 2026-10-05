"""PlotsOfData-style plots of single features, with the field as the sampling unit.

Postma and Goedhart's PlotsOfData shows every value as a point beside a
summary and its 95% confidence interval; SuperPlots keep the biological
repeats apart. These plots do the same for one feature per figure: within
each condition the biological repeats (the repeat column) sit side by side,
every object is a jittered point in its condition's colour (``_palette``, the same as the
HTML report's), shaped by technical repeat (the
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
cluster. The legend (at the bottom) says how many units a cluster's interval resamples, and
warns when one has only 2 or 3, which no resampling turns into a reliable
interval, or a single technical repeat, whose interval shows field-to-field
variation only.

Every column of FOV identity that the figure does not place (condition,
repeat or technical repeat) is pooled within a cluster, never merged: with
``repeat_col=None`` on several repeats that reuse well names, well ``A`` of
each repeat stays its own technical repeat, and the repeats are resampled as
pooled repeats, which the legend says. An analysis-group column (``group_cols``)
that the figure does not place raises instead, since the figure would pool
the groups the caller asked to keep apart: name it as ``condition_col``,
``repeat_col`` or ``technical_col``, or leave it out of ``group_cols``.

Limits keep the figures legible and are arguments: at most ``max_conditions``
conditions per figure (more are split evenly over several figures), at most
``max_repeats`` biological repeats side by side (more are pooled into one
cluster per condition, and the legend says so), at most ``max_technical``
shapes (up to 8; more technical repeats of a biological repeat share small
dots), at most ``max_features`` features per call (more raise, so the caller
chooses). A column named explicitly that the tables do not hold raises; only
the default ``condition`` and ``biological_repeat`` columns may be absent. The layout's
columns are read from the FOV table (or, when only the observations carry
one, from each field's observations, which must agree); a field with no
value in one of them raises, since it cannot be placed. Every cluster draws
its jitter and resamples from its own generator, seeded from
``random_state``, the feature, the condition and the repeat, so a figure does
not change with the other features or conditions of the call. Figures are
``matplotlib.figure.Figure`` objects built without pyplot, so they display in
a notebook and save with ``savefig`` without touching global state. Each carries
``cluster_summaries``: per cluster drawn, its condition, repeat (``None`` when pooled),
x position and width, and its summary with the interval (``summary``, ``low``, ``high``),
and ``caption``, what it shows (one clause per line), so another rendering (the HTML
report's) draws the same numbers. The caption is the figure's legend, at its bottom
(``_figure_layout``); its title is the feature.
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
from ._figure_layout import sentences, set_legend
from ._former_repeat import one_warning, read_former
from ._palette import (
    COLOURS,
    FONT,
    GREY,
    INK,
    MARKERS,
    MUTED,
    OTHER_MARKER,
    condition_palette,
    style_axes,
)
from ._plots_text import legend, title
from ._seeds import group_rng
from ._wording import shortened
from .data_check import validate_input_tables
from .metadata import (
    MORPHOBRICKS_GROUP_COLUMNS,
    TECHNICAL_REPEAT_COLUMN,
    empty_values,
    sorted_labels,
)
from .schema import ValidatedInput, fov_positions, fov_values, observation_words
from .statistics import bootstrap_nested

MAX_CONDITIONS = 12
MAX_REPEATS = 8
MAX_TECHNICAL = 8
MAX_FEATURES = 24
# The longest condition name written in full under the plot.
NAME_CHARACTERS = 30
# The x axis: one slot per biological repeat, a gap between conditions, and the inches a
# slot takes.
GAP = 0.8
INCHES_PER_SLOT = 0.62
SUMMARIES: dict[str, Callable[[np.ndarray], float]] = {"median": np.median, "mean": np.mean}
# One colour per condition (``_palette``); the technical repeats differ by shape.
PALETTE = COLOURS


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
    value there and raises, as the validator does for a group column. A missing
    value in the FOV table, or in the observations, raises: that field cannot be placed.
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
        placed = by_field.first().reindex(meta.index)
        if placed.isna().any():
            raise ValueError(
                f"{int(placed.isna().sum())} fields without observations have no {column!r} "
                f"value, since only the observations carry it; add {column!r} to the FOV "
                "table, so fields without observations keep their place"
            )
        meta[column] = placed.to_numpy()
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
    colour: str,
    statistic: Callable[[np.ndarray], float],
    n_boot: int,
    confidence: float,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    """One cluster (a condition and biological repeat): its points in the condition's
    ``colour``, each technical repeat by shape, then the summary and interval. Returns the
    summary and its interval.

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
        # Shapes by rank within the top-level unit (the biological repeat), as the legend says.
        rank = len(by_top.setdefault(top, []))
        unit_fields = in_cluster & _unit_mask(meta, layout, top, replicate)
        by_top[top].append([per_field[i] for i in np.flatnonzero(unit_fields)])
        y = values[rows & _unit_mask(observed, layout, top, replicate)]
        y = y[~np.isnan(y)]
        if not len(y):
            continue
        marker = MARKERS[rank] if rank < len(MARKERS) else OTHER_MARKER
        jitter = rng.uniform(-half, half, size=len(y))
        ax.scatter(
            x + jitter,
            y,
            s=16,
            alpha=0.5,
            color=colour,
            marker=marker,
            linewidths=0,
            rasterized=True,
            zorder=1,
        )
    tops = [t for t in by_top.values() if any(len(f) for unit in t for f in unit)]
    if len(tops) > 1:
        # Pooled repeats: the repeats alone resampled, all their fields as acquired (Ren 2010).
        units_resampled = [[f for unit in top for f in unit] for top in tops]
    else:
        units_resampled = tops[0] if tops else [[]]
    point, low, high = bootstrap_nested(
        units_resampled, statistic, n_boot=n_boot, rng=rng, confidence=confidence
    )
    if np.isfinite(low) and np.isfinite(high):
        # The interval as an error bar with caps, so a narrow one stays visible.
        ax.vlines(x, low, high, color=INK, lw=1.8, zorder=3)
        ax.hlines([low, high], x - 0.13 * width, x + 0.13 * width, color=INK, lw=1.8, zorder=3)
    if np.isfinite(point):
        ax.hlines(point, x - 0.34 * width, x + 0.34 * width, color=INK, lw=3, zorder=4)
    return float(point), float(low), float(high)


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
    highlight: Sequence[Any] = (),
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
    Each figure colours its own conditions (``_palette.condition_palette``, the control in
    the first colour): past 12 in one figure, only the control and the ``highlight``
    conditions keep a colour, the others are grey and the legend says so. Each carries its
    colours as ``colours``.
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
    if max_technical > len(MARKERS):
        raise ValueError(
            f"max_technical can be at most {len(MARKERS)}, the shapes that can be told apart"
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
    shapes = MARKERS[:max_technical]
    meta = _field_metadata(
        data,
        [layout.condition_col, layout.repeat_col, layout.technical_col, *layout.pooled_cols],
    )
    observed = meta.iloc[fov_positions(data, data.fovs)].reset_index(drop=True)
    statistic = SUMMARIES[summary]
    figures: dict[str, list[Figure]] = {}
    for feature in features:
        values = data.feature_values(feature)
        per_field = fov_values(data, values)  # every acquired field, empty ones included
        figures[feature] = []
        # Every part of a split figure starts with the control, so each condition is drawn
        # beside the one it is compared with.
        conditions = list(layout.conditions)
        lead = conditions[:1] if len(conditions) > 1 and conditions[0] == first else []
        rest = conditions[len(lead) :]
        room = max(max_conditions - len(lead), 1)
        size = -(-len(rest) // -(-len(rest) // room))
        for start in range(0, len(rest), size):
            chunk = lead + rest[start : start + size]
            # The legend describes the clusters of this figure only, not those of its other parts.
            in_chunk = np.logical_or.reduce(
                [_mask(observed, layout.condition_col, c) for c in chunk]
            )
            clusters = _cluster_units(observed[in_chunk], layout)
            n_technical = max((n for _, technical in clusters for n in technical), default=0)
            # Each part its own colours, the control's first (owner, 2026-10-05).
            colours = condition_palette(chunk, first, highlight)
            # Each condition is as wide as the repeats it holds: a slot of 1 per repeat, a gap
            # of GAP between conditions, so repeats imaged for other conditions take no room.
            present = [
                [
                    r
                    for r in layout.repeats
                    if (
                        _mask(observed, layout.condition_col, c)
                        & _mask(observed, layout.repeat_col, r)
                    ).any()
                ]
                for c in chunk
            ]
            starts = np.cumsum([0.0] + [max(len(p), 1) + GAP for p in present[:-1]])
            span = starts[-1] + max(len(present[-1]), 1)
            figure = Figure(figsize=(max(6.5, INCHES_PER_SLOT * span + 2.0), 4.8))
            ax = figure.add_subplot(111)
            repeat_labels: list[tuple[float, str]] = []
            summaries = []
            width = 1.0
            for index, condition in enumerate(chunk):
                in_condition = _mask(observed, layout.condition_col, condition)
                fields_in_condition = _mask(meta, layout.condition_col, condition)
                for j, repeat in enumerate(present[index]):
                    rows = in_condition & _mask(observed, layout.repeat_col, repeat)
                    in_cluster = fields_in_condition & _mask(meta, layout.repeat_col, repeat)
                    x = starts[index] + j + 0.5
                    # A generator per cluster: no figure depends on the others of the call.
                    rng = group_rng(
                        random_state,
                        {"feature": feature, "condition": condition, "repeat": repeat},
                    )
                    point, low, high = _draw_cluster(
                        ax,
                        x,
                        width,
                        rows,
                        values,
                        (meta, observed, per_field, in_cluster),
                        layout,
                        colours[condition],
                        statistic,
                        n_boot,
                        confidence,
                        rng,
                    )
                    summaries.append(
                        {"condition": condition, "repeat": repeat, "x": x, "width": width}
                        | {"summary": point, "low": low, "high": high}
                    )
                    # CellPatcher's "single" names a folder level the data did not have.
                    if repeat is not None and str(repeat) != "single":
                        repeat_labels.append((x, str(repeat)))
            many = len(chunk) > 4
            ax.set_xticks([b + max(len(p), 1) / 2 for b, p in zip(starts, present, strict=True)])
            ax.set_xlim(-0.4, span + 0.4)
            # A long name is cut (the figure's data holds it whole), so the plot keeps its room.
            ax.set_xticklabels(
                ["all fields" if c is None else shortened(c, NAME_CHARACTERS) for c in chunk],
                rotation=30 if many else 0,
                ha="right" if many else "center",
            )
            # Slanted when a repeat's name is wider than its slot, so neighbours never run
            # into each other ("2026_03_022026_03_09"); about 0.075 in a character at 9.5 pt.
            longest = max((len(label) for _, label in repeat_labels), default=0)
            slot = (figure.get_size_inches()[0] - 1.3) / (span + 0.8)
            slant = 0.075 * longest > 0.9 * slot
            for x, label in repeat_labels:
                ax.annotate(
                    label,
                    (x, 1.0),
                    xycoords=("data", "axes fraction"),
                    xytext=(-3, 3) if slant else (0, 0),
                    textcoords="offset points",
                    ha="left" if slant else "center",
                    va="bottom",
                    rotation=40 if slant else 0,
                    rotation_mode="anchor",
                    fontsize=9.5,
                    color=MUTED,
                )
            ax.set_ylabel(feature if value_note is None else f"{feature} ({value_note})")
            shown = len(chunk) - len(lead)
            part = (
                f" (conditions {start + 1} to {start + shown} of {len(rest)}"
                + (", with the control)" if lead else ")")
                if len(rest) > size
                else ""
            )
            words = title(
                feature, part, layout, observation_words(data)[0], summary, confidence, clusters
            )
            # The feature on top; what the figure shows is its legend, at the bottom.
            pad = 22 + (3.5 * longest if slant else 0)  # above the slanted names
            ax.set_title(f"{feature}{part}", fontsize=13, weight="bold", loc="left", pad=pad)
            legend(ax, layout, shapes, n_technical)
            style_axes(ax)
            ax.tick_params(axis="x", labelsize=FONT + 1)
            figure.tight_layout()
            greyed = sum(colour == GREY for colour in colours.values())
            if greyed:
                words += (
                    f"\nin colour the control and the highlighted conditions; the other "
                    f"{greyed} in grey"
                )
            figure.cluster_summaries = pd.DataFrame(summaries)
            figure.caption = words
            figure.colours = colours
            set_legend(figure, [sentences(words)])
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
