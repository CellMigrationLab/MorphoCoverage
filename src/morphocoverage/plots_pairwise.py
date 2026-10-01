"""Every pair of conditions for one feature, as the lab's Plot&Stats notebook shows them
under each plot: the effect size and the test, in one matrix.

- ``pairwise_contrasts``: for every pair of conditions, Cohen's d on the observations
  (``statistics.cohen_d``, the row's condition minus the column's: signed, so lower and
  higher read apart) and the paired t-test on the biological repeats' means, paired within
  each repeat that imaged both conditions (on the log ratios when every mean is positive,
  else on the differences, as the discovery's second gate does). A pair needs
  ``MIN_REPEATS`` such repeats to be tested; one imaged in different repeats gets d only,
  since comparing means from different repeats would read a batch as an effect. The p-values
  are Bonferroni-corrected over the pairs tested. Pairs without the control are not the
  declared comparisons: they describe, as the notebook's do.
- ``add_pairwise``: the matrix drawn under a ``plots.plots_of_data`` figure, the figure made
  taller for it: the lower-left triangle holds d (blue lower, red higher, full colour at
  ``D_CAP``), the upper-right triangle the corrected p (green and bold at or below
  ``ALPHA``, darker the smaller; grey above it), "–" where a pair could not be tested,
  and † on a d whose conditions were imaged in different repeats (the difference between
  batches is in it).

Cohen's d treats every observation as an independent draw, so it describes the size of a
difference, not its evidence; the paired test is the evidence across repeats, with little
power on three.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from .data_check import validate_input_tables
from .metadata import CONDITION_COLUMN, REPEAT_COLUMN, sorted_labels
from .sampling_unit.contrast_repeats import MIN_REPEATS, _paired_t
from .schema import fov_positions
from .statistics import cohen_d

ALPHA = 0.05
D_CAP = 2.0
PAIR_COLUMNS = [
    "feature",
    "first",
    "second",
    "d",
    "observations",
    "second_observations",
    "shared_repeats",
    "paired_values",
    "p",
    "p_bonferroni",
    "status",
]
# The colours of d (lower, higher, the middle) and of p (strongest, weakest).
LOWER, HIGHER, MIDDLE = (42, 120, 214), (227, 73, 72), (240, 239, 236)
P_STRONG, P_WEAK = (46, 125, 50), (240, 239, 236)


def _paired(first: dict, second: dict) -> tuple[float, str, int]:
    shared = [r for r in first if r in second]
    if len(shared) < MIN_REPEATS:
        return np.nan, "", len(shared)
    a = np.array([np.mean(first[r]) for r in shared])
    b = np.array([np.mean(second[r]) for r in shared])
    if (a > 0).all() and (b > 0).all():
        return _paired_t(np.log(a / b)), "log ratios", len(shared)
    return _paired_t(a - b), "differences", len(shared)


def pairwise_contrasts(
    objects: pd.DataFrame,
    fovs: pd.DataFrame,
    *,
    feature: str,
    conditions: Sequence[Any] | None = None,
    control: Any = None,
    fov_col: str | None = None,
    group_cols: Sequence[str] | None = None,
    observation_unit: str | None = None,
) -> pd.DataFrame:
    """One row per pair of ``conditions`` (all, the ``control`` first, by default); see the
    module. ``first`` and ``second`` follow the order of the conditions."""
    data = validate_input_tables(
        objects,
        fovs=fovs,
        feature_cols=[feature],
        fov_col=fov_col,
        group_cols=group_cols,
        observation_unit=observation_unit,
    )
    meta = data.fovs.iloc[fov_positions(data, data.fovs)].reset_index(drop=True)
    if CONDITION_COLUMN not in meta:
        return pd.DataFrame(columns=PAIR_COLUMNS)
    values = data.feature_values(feature)
    keep = ~np.isnan(values)
    frame = pd.DataFrame(
        {
            "condition": meta[CONDITION_COLUMN].to_numpy()[keep],
            "repeat": meta[REPEAT_COLUMN].to_numpy()[keep] if REPEAT_COLUMN in meta else 0,
            "value": values[keep],
        }
    )
    order = (
        list(conditions) if conditions is not None else sorted_labels(frame["condition"], control)
    )
    by_condition = {c: block for c, block in frame.groupby("condition", sort=False)}
    rows = []
    for first, second in combinations(order, 2):
        a, b = by_condition.get(first), by_condition.get(second)
        if a is None or b is None:
            continue
        repeats_a = {r: g.to_numpy() for r, g in a.groupby("repeat", sort=False)["value"]}
        repeats_b = {r: g.to_numpy() for r, g in b.groupby("repeat", sort=False)["value"]}
        p, paired_values, shared = _paired(repeats_a, repeats_b)
        if not shared:
            status = "not imaged in the same repeats"
        elif not np.isfinite(p) and shared < MIN_REPEATS:
            status = (
                f"{shared} repeat{'s' if shared > 1 else ''} with both, fewer than {MIN_REPEATS}"
            )
        elif not np.isfinite(p):
            status = "the repeat means do not vary"
        else:
            status = "ok"
        rows.append(
            {
                "feature": feature,
                "first": first,
                "second": second,
                "d": cohen_d(a["value"].to_numpy(), b["value"].to_numpy()),
                "observations": len(a),
                "second_observations": len(b),
                "shared_repeats": shared,
                "paired_values": paired_values,
                "p": p,
                "status": status,
            }
        )
    table = pd.DataFrame(rows, columns=[c for c in PAIR_COLUMNS if c != "p_bonferroni"])
    tested = int(np.isfinite(table["p"].astype(float)).sum()) if len(table) else 0
    table["p_bonferroni"] = (table["p"].astype(float) * tested).clip(upper=1.0)
    return table[PAIR_COLUMNS]


def _mix(start: tuple, end: tuple, share: float) -> tuple[float, float, float]:
    return tuple((s + (e - s) * share) / 255 for s, e in zip(start, end, strict=True))


def d_colour(d: float) -> tuple[float, float, float]:
    """The colour of ``d``: the middle at 0, blue lower, red higher, full at ``D_CAP``."""
    if not np.isfinite(d):
        return _mix(MIDDLE, MIDDLE, 0)
    return _mix(MIDDLE, LOWER if d < 0 else HIGHER, min(abs(d), D_CAP) / D_CAP)


def p_shade(p: float) -> float:
    """How strongly a corrected p is shaded: 0 above ``ALPHA`` (so a p that does not pass is
    never coloured as if it did), then from 0.3 at ``ALPHA`` to 1 at 0.001."""
    if not np.isfinite(p) or p > ALPHA:
        return 0.0
    top = -np.log10(ALPHA)
    return 0.3 + 0.7 * min((-np.log10(max(p, 1e-3)) - top) / (3.0 - top), 1.0)


def p_colour(p: float) -> tuple[float, float, float]:
    """The colour of a corrected p (``p_shade``): grey when it does not pass, green when it
    does, darker the smaller."""
    return _mix(P_WEAK, P_STRONG, p_shade(p))


def format_p(p: float) -> str:
    """A p-value in a cell: three significant digits, ``<0.001`` below."""
    return "<0.001" if p < 0.001 else f"{p:.2g}" if p < 0.01 else f"{p:.2f}"


def _ink(colour: tuple[float, float, float]) -> str:
    return "white" if sum(colour) / 3 < 0.55 else "#0b0b0b"


def add_pairwise(figure: Figure, table: pd.DataFrame, conditions: Sequence[Any]) -> None:
    """Draw the matrix of ``table`` (``pairwise_contrasts``) for ``conditions`` under the
    figure's plot, the figure made taller for it; nothing with fewer than two conditions."""
    conditions = list(conditions)
    n = len(conditions)
    if n < 2 or table.empty:
        return
    labels = [str(c) for c in conditions]
    cell = 0.55 if n <= 8 else max(0.3, 4.4 / n)
    label_w = min(2.8, 0.11 * max(len(t) for t in labels) + 0.3)
    extra = cell * n + 0.4 + label_w * 0.6
    width, height = figure.get_size_inches()
    total = height + extra
    figure.set_size_inches(width, total)
    for ax in figure.axes:
        box = ax.get_position()
        ax.set_position(
            [box.x0, (box.y0 * height + extra) / total, box.width, box.height * height / total]
        )
    left = label_w + 0.2
    ax = figure.add_axes(
        [left / width, (label_w * 0.6) / total, cell * n / width, cell * n / total]
    )
    lookup = {(r.first, r.second): r for r in table.itertuples()}
    font = 9.5 if n <= 8 else 7.5
    for i, row in enumerate(conditions):
        for j, column in enumerate(conditions):
            if i == j:
                ax.add_patch(_square(j, i, (0.97, 0.97, 0.96)))
                continue
            pair = lookup.get((row, column)) or lookup.get((column, row))
            if pair is None:
                continue
            if i > j:
                # Lower left: d of the row's condition minus the column's.
                d = pair.d if pair.first == row else -pair.d
                colour = d_colour(d)
                text = "–" if not np.isfinite(d) else f"{d:+.2f}"
                if np.isfinite(d) and pair.shared_repeats == 0:
                    text += "†"  # different repeats: the batch is in d too
                weight = "normal"
            else:
                p = pair.p_bonferroni
                colour = p_colour(p)
                text = "–" if not np.isfinite(p) else format_p(p)
                weight = "bold" if np.isfinite(p) and p <= ALPHA else "normal"
            ax.add_patch(_square(j, i, colour))
            ax.text(
                j + 0.5,
                i + 0.5,
                text,
                ha="center",
                va="center",
                fontsize=font,
                color=_ink(colour),
                weight=weight,
            )
    ax.set_xlim(0, n)
    ax.set_ylim(n, 0)
    ax.set_xticks(np.arange(n) + 0.5)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=font + 0.5)
    ax.set_yticks(np.arange(n) + 0.5)
    ax.set_yticklabels(labels, fontsize=font + 0.5)
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    tested = int(np.isfinite(table["p"].astype(float)).sum())
    note = (
        "Every pair of conditions. Lower left: Cohen's d on the observations, the row's "
        "condition minus the column's (blue lower, red higher). Upper right: the paired "
        f"t-test on the biological repeats' means, Bonferroni over the {tested} pairs tested "
        f"(bold at p ≤ {ALPHA:g}); – : fewer than {MIN_REPEATS} repeats imaged both; † : "
        "imaged in different repeats, so d holds the batch too. d describes a size; the "
        "paired test is the evidence across repeats."
    )
    x = (left + cell * n + 0.35) / width
    if x < 0.8:
        figure.text(
            x,
            (label_w * 0.6 + cell * n) / total,
            _wrapped(note, width * (1 - x) * 11),
            fontsize=9,
            va="top",
            ha="left",
            color="0.3",
        )


def _square(x: float, y: float, colour: tuple[float, float, float]):
    from matplotlib.patches import Rectangle

    return Rectangle((x + 0.03, y + 0.03), 0.94, 0.94, facecolor=colour, edgecolor="none")


def _wrapped(text: str, width: float) -> str:
    import textwrap

    return "\n".join(textwrap.wrap(text, max(30, int(width))))


__all__ = [
    "ALPHA",
    "D_CAP",
    "PAIR_COLUMNS",
    "add_pairwise",
    "d_colour",
    "format_p",
    "p_colour",
    "p_shade",
    "pairwise_contrasts",
]
