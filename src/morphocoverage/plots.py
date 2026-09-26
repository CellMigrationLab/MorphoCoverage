"""PlotsOfData-style plots of single features, with the field as the sampling unit.

Postma and Goedhart's PlotsOfData shows every value as a point beside a
summary and its 95% confidence interval; SuperPlots colour the points by
biological repeat. These plots do the same for one feature per figure: every
object is a jittered point coloured by repeat, the summary (median by
default, or mean) is drawn per condition, and its interval comes from
resampling whole fields (``statistics.bootstrap_fields``), not objects, so the
interval reflects how the data were sampled.

Limits keep the figures legible and are arguments: at most ``max_conditions``
conditions per figure (more are split over several figures), at most
``max_repeats`` repeat colours (more repeats share one grey and the legend
says so), at most ``max_features`` features per call (more raise, so the
caller chooses). Figures are ``matplotlib.figure.Figure`` objects, built
without pyplot, so they display in a notebook and save with ``savefig``
without touching global state.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from .metadata import MORPHOBRICKS_GROUP_COLUMNS
from .schema import ValidatedInput, fov_positions, unit_plural, validate_input_tables
from .statistics import bootstrap_fields

MAX_CONDITIONS = 12
MAX_REPEATS = 8
MAX_FEATURES = 24
SUMMARIES = {"median": np.median, "mean": np.mean}
PALETTE = ("#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860", "#DA8BC3", "#8C8C8C")
GREY = "0.6"


def _fields_of(data: ValidatedInput, rows: np.ndarray, values: np.ndarray) -> list[np.ndarray]:
    """The values of ``rows`` (a boolean mask over observations) split by acquired field."""
    positions = fov_positions(data, data.fovs)
    chosen = rows & ~np.isnan(values)
    fields = []
    for position in np.unique(positions[rows]):
        fields.append(values[chosen & (positions == position)])
    return fields


def plots_of_data(
    objects: pd.DataFrame,
    fovs: pd.DataFrame,
    *,
    features: Sequence[str],
    condition_col: str | None = MORPHOBRICKS_GROUP_COLUMNS[0],
    repeat_col: str | None = MORPHOBRICKS_GROUP_COLUMNS[1],
    summary: str = "median",
    confidence: float = 0.95,
    n_boot: int = 1000,
    random_state: int = 0,
    max_conditions: int = MAX_CONDITIONS,
    max_repeats: int = MAX_REPEATS,
    max_features: int = MAX_FEATURES,
    fov_col: str | None = None,
    group_cols: Sequence[str] | None = None,
    observation_unit: str | None = None,
) -> dict[str, list[Figure]]:
    """One or more figures per feature (see the module); keys are the feature names.

    ``features`` must be columns of the tables and at most ``max_features``.
    Without a condition column every object is one group; without a repeat
    column the points share one colour.
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
    for name, value in (("n_boot", n_boot), ("max_conditions", max_conditions)):
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
    condition_col = condition_col if condition_col in table.columns else None
    repeat_col = repeat_col if repeat_col in table.columns else None
    conditions = (
        list(dict.fromkeys(table[condition_col].tolist())) if condition_col else ["all fields"]
    )
    repeats = list(dict.fromkeys(table[repeat_col].tolist())) if repeat_col else [None]
    coloured = len(repeats) <= max_repeats
    colours = {r: (PALETTE[i % len(PALETTE)] if coloured else GREY) for i, r in enumerate(repeats)}
    rng = np.random.default_rng(random_state)
    statistic = SUMMARIES[summary]
    words = unit_plural(data.observation_unit)
    figures: dict[str, list[Figure]] = {}
    for feature in features:
        values = data.feature_values(feature)
        figures[feature] = []
        for start in range(0, len(conditions), max_conditions):
            chunk = conditions[start : start + max_conditions]
            figure = Figure(figsize=(max(4.0, 1.1 * len(chunk) + 2.0), 4.0))
            ax = figure.add_subplot(111)
            for x, condition in enumerate(chunk):
                in_condition = (
                    table[condition_col].to_numpy() == condition
                    if condition_col
                    else np.ones(len(table), dtype=bool)
                )
                for repeat in repeats:
                    rows = in_condition & (
                        table[repeat_col].to_numpy() == repeat if repeat_col else True
                    )
                    y = values[rows]
                    y = y[~np.isnan(y)]
                    if not len(y):
                        continue
                    jitter = rng.uniform(-0.3, 0.3, size=len(y))
                    ax.scatter(
                        x + jitter,
                        y,
                        s=6,
                        alpha=0.35,
                        color=colours[repeat],
                        linewidths=0,
                        rasterized=True,
                    )
                point, low, high = bootstrap_fields(
                    _fields_of(data, in_condition, values),
                    statistic,
                    n_boot=n_boot,
                    rng=rng,
                    confidence=confidence,
                )
                if np.isfinite(point):
                    ax.hlines(point, x - 0.4, x + 0.4, color="black", lw=2, zorder=3)
                if np.isfinite(low) and np.isfinite(high):
                    ax.add_patch(
                        _box(x, low, high),
                    )
            ax.set_xticks(range(len(chunk)))
            ax.set_xticklabels([str(c) for c in chunk], rotation=30, ha="right")
            ax.set_ylabel(feature)
            part = (
                f" (conditions {start + 1} to {start + len(chunk)} of {len(conditions)})"
                if len(conditions) > max_conditions
                else ""
            )
            ax.set_title(
                f"{feature}{part}: every {data.observation_unit} a point, {summary} with its "
                f"{confidence:.0%} interval from resampling whole fields",
                fontsize=9,
            )
            if repeat_col:
                if coloured:
                    handles = [
                        ax.scatter([], [], color=colours[r], s=20, label=f"{repeat_col}={r}")
                        for r in repeats
                    ]
                    ax.legend(handles=handles, frameon=False, fontsize=8, title=None)
                else:
                    ax.text(
                        0.01,
                        0.99,
                        f"{len(repeats)} repeats (more than {max_repeats}): {words} not coloured "
                        "by repeat",
                        transform=ax.transAxes,
                        va="top",
                        fontsize=8,
                    )
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            figure.tight_layout()
            figures[feature].append(figure)
    return figures


def _box(x: float, low: float, high: float) -> Rectangle:
    """The interval as a translucent box behind the summary line."""
    return Rectangle(
        (x - 0.4, low), 0.8, high - low, facecolor="black", alpha=0.12, edgecolor="none", zorder=2
    )


__all__ = ["MAX_CONDITIONS", "MAX_FEATURES", "MAX_REPEATS", "SUMMARIES", "plots_of_data"]
