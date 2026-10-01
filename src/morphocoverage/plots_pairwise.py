"""Every pair of conditions for one feature, as the lab's Plot&Stats notebook shows them
under each plot: the effect size and the test, in one matrix.

- ``pairwise_contrasts``: for every pair of conditions, Cohen's d on the observations
  (``statistics.cohen_d``, the row's condition minus the column's: signed, so lower and
  higher read apart) and the randomization test on it, as Plot&Stats runs it: the condition
  labels shuffled over every observation of the two conditions, pooled over the repeats
  (``contrast_cells.cell_randomization``, ``N_PERMUTATIONS`` shuffles). The p-values are
  Benjamini-Hochberg corrected over the pairs (``q``), as the feature discovery corrects
  its gate. The paired t-test on the repeats' means is kept beside it (``p_paired``, for
  pairs imaged in ``MIN_REPEATS`` shared repeats), not drawn: on three repeats it has
  little power. A pair imaged in different repeats is tested too, but its difference holds
  the batch between them: it is marked. Pairs without the control are not the declared
  comparisons: they describe, as the notebook's do.
- ``add_pairwise``: the matrix drawn under a ``plots.plots_of_data`` figure, the figure made
  taller for it: the lower-left triangle holds d (blue lower, red higher, full colour at
  ``D_CAP``), the upper-right triangle the q-value (green and bold at or below ``ALPHA``,
  darker the smaller; grey above it), and † where the two conditions were imaged in
  different repeats.

The randomization test treats every observation as an independent draw: it finds
differences between these cells, and when the difference varies between repeats it passes
more often than ``ALPHA`` without an effect ("Do the conditions differ?" gives that chance
for the readout); the paired test holds its rate but needs the repeats.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from ._palette import MUTED
from ._seeds import group_rng
from .data_check import validate_input_tables
from .metadata import CONDITION_COLUMN, REPEAT_COLUMN, sorted_labels
from .sampling_unit.contrast_cells import cell_randomization
from .sampling_unit.contrast_repeats import MIN_REPEATS, _paired_t
from .schema import fov_positions
from .statistics import benjamini_hochberg

ALPHA = 0.05
D_CAP = 2.0
N_PERMUTATIONS = 1999
PAIR_NOTE = (
    "Every pair of conditions. Lower left: Cohen's d on the observations, the row's "
    "condition minus the column's (blue lower, red higher, full colour at |d| ≥ 2). Upper "
    "right: the randomization test on d, every observation of the two conditions shuffled "
    "(as Plot&Stats), Benjamini-Hochberg corrected over the pairs (q; green at q ≤ 0.05, "
    "darker the smaller). It treats every observation as an independent draw, so when the repeats "
    "differ it passes more often than 5% without an effect. † : imaged in different "
    "repeats, so the batch is in the difference. Pairs without the control describe; the "
    "declared comparisons are against the control."
)
PAIR_COLUMNS = [
    "feature",
    "first",
    "second",
    "d",
    "observations",
    "second_observations",
    "shared_repeats",
    "p",
    "q",
    "p_paired",
    "paired_values",
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
    n_permutations: int = N_PERMUTATIONS,
    random_state: int = 0,
) -> pd.DataFrame:
    """One row per pair of ``conditions`` (all, the ``control`` first, by default); see the
    module. ``first`` and ``second`` follow the order of the conditions; each pair's shuffles
    are seeded from ``random_state``, the feature and the pair."""
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
        p_paired, paired_values, shared = _paired(repeats_a, repeats_b)
        rng = group_rng(random_state, {"feature": feature, "first": first, "second": second})
        d, p = cell_randomization(a["value"].to_numpy(), b["value"].to_numpy(), n_permutations, rng)
        if not np.isfinite(p):
            status = "one value everywhere"
        elif not shared:
            status = "imaged in different repeats"
        else:
            status = "ok"
        rows.append(
            {
                "feature": feature,
                "first": first,
                "second": second,
                "d": d,
                "observations": len(a),
                "second_observations": len(b),
                "shared_repeats": shared,
                "p": p,
                "p_paired": p_paired,
                "paired_values": paired_values,
                "status": status,
            }
        )
    table = pd.DataFrame(rows, columns=[c for c in PAIR_COLUMNS if c != "q"])
    table["q"] = benjamini_hochberg(table["p"].to_numpy(float)) if len(table) else []
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
    cell = 0.72 if n <= 8 else max(0.4, 5.6 / n)
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
    font = 11 if n <= 8 else 8.5
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
                q = pair.q
                colour = p_colour(q)
                text = "–" if not np.isfinite(q) else format_p(q)
                if np.isfinite(q) and pair.shared_repeats == 0:
                    text += "†"
                weight = "bold" if np.isfinite(q) and q <= ALPHA else "normal"
            ax.add_patch(_square(j, i, colour))
            ax.text(
                j + 0.5,
                i + 0.5,
                text,
                ha="center",
                va="center",
                fontsize=font if len(text) < 6 else font - 1.5,
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
    note = PAIR_NOTE
    x = (left + cell * n + 0.35) / width
    if x < 0.8:
        figure.text(
            x,
            (label_w * 0.6 + cell * n) / total,
            _wrapped(note, width * (1 - x) * 11),
            fontsize=9.5,
            va="top",
            ha="left",
            color=MUTED,
        )


def _square(x: float, y: float, colour: tuple[float, float, float]):
    from matplotlib.patches import FancyBboxPatch

    return FancyBboxPatch(
        (x + 0.05, y + 0.05),
        0.9,
        0.9,
        boxstyle="round,pad=0,rounding_size=0.08",
        facecolor=colour,
        edgecolor="none",
    )


def _wrapped(text: str, width: float) -> str:
    import textwrap

    return "\n".join(textwrap.wrap(text, max(30, int(width))))


__all__ = [
    "ALPHA",
    "D_CAP",
    "N_PERMUTATIONS",
    "PAIR_COLUMNS",
    "PAIR_NOTE",
    "add_pairwise",
    "d_colour",
    "format_p",
    "p_colour",
    "p_shade",
    "pairwise_contrasts",
]
