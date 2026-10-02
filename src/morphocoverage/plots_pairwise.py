"""Every pair of conditions for one feature, as the lab's Plot&Stats notebook shows them
under each plot: the effect size and the test, in one matrix.

- ``pairwise_contrasts``: for every pair of conditions, Cohen's d on the observations
  (``statistics.cohen_d``, the row's condition minus the column's: signed, so lower and
  higher read apart) and the randomization test on it, as Plot&Stats runs it: the condition
  labels shuffled over the observations of the two conditions (``contrast_cells.
  cell_randomization``, ``N_PERMUTATIONS`` shuffles), taken from the repeats that imaged
  both, so each condition meets the control of its own repeats (as the reading does). Two
  conditions imaged in different repeats (two sub-experiments) get their d, marked †, and
  no test: the batch between their repeats would be in it. The p-values are
  Benjamini-Hochberg corrected over the pairs of the figure (``q``); the report's tables
  correct over every feature and condition, and the figure's note says so. The paired
  t-test on the repeats' means is kept beside it (``p_paired``, for pairs imaged in
  ``MIN_REPEATS`` shared repeats). Pairs without the control are not the declared
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
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from ._figure_layout import grow_bottom, panel_letter, remove_legend, sentences, set_legend
from ._palette import MUTED
from ._seeds import group_rng
from .data_check import validate_input_tables
from .metadata import CONDITION_COLUMN, REPEAT_COLUMN, sorted_labels
from .sampling_unit.contrast_cells import cell_randomization
from .sampling_unit.contrast_repeats import MIN_REPEATS
from .schema import fov_positions
from .statistics import benjamini_hochberg, cohen_d, paired_t, repeat_values

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.patches import FancyBboxPatch

ALPHA = 0.05
D_CAP = 2.0
N_PERMUTATIONS = 1999
# The panels' titles, with what each is good for (owner, 2026-10-02).
PANEL_TITLES = {
    "B": "d · randomization test: discovery, false positives when repeats differ",
    "C": "paired t-test, repeat means: decides, conservative with few repeats",
}
# What each panel is good for, in a line (owner, 2026-10-02).
PAIR_READING = (
    "Good for discovery, but prone to false positives when the repeats differ: every "
    "observation counts as an independent draw."
)
PAIRED_READING = (
    "The test that decides: it holds its false-positive rate when the repeats differ, but "
    "it is conservative, more so with few repeats (with 3, only large, consistent effects "
    "reach 0.05)."
)
PAIR_NOTE = (
    f"{PAIR_READING} Every pair of conditions; the row and the column of the control answer "
    "the question, the other squares describe. Lower left: Cohen's d, the row's condition "
    "minus the column's (blue lower, red higher, full colour at |d| ≥ 2). Upper right: the "
    "randomization test on d, as Plot&Stats runs it, on the repeats that imaged both "
    "conditions (so each condition meets the control of its own repeats), adjusted over the "
    "pairs of this figure (Benjamini-Hochberg; green at ≤ 0.05, darker the smaller); the "
    "tables adjust over every feature and condition, so their values differ. †: the two "
    "conditions were imaged in different repeats (two sub-experiments): d describes, and "
    "they are not tested, since the difference between their repeats would be in it."
)
PAIRED_NOTE = (
    f"{PAIRED_READING} The paired t-test on the biological repeats' means: each condition's "
    "mean paired with the other's in every repeat that imaged both (on the log ratios when "
    "every mean is positive). Lower left: its p-value, unadjusted; upper right: adjusted "
    "over the pairs of this figure (Benjamini-Hochberg; the tables adjust over every feature "
    "and condition). Green at ≤ 0.05, darker the smaller; – : fewer than 3 repeats imaged "
    "both."
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
    "q_paired",
    "paired_values",
    "status",
]
# The colours of d (lower, higher, the middle) and of p (strongest, weakest).
LOWER, HIGHER, MIDDLE = (42, 120, 214), (227, 73, 72), (240, 239, 236)
P_STRONG, P_WEAK = (46, 125, 50), (240, 239, 236)


# The values the paired test compares, as the figure's notes name them.
_VALUE_WORDS = {"normalised": "log ratios", "raw": "differences"}


def _paired(first: dict, second: dict) -> tuple[float, str, int]:
    """The paired t-test of two conditions on the repeats they share, the values it
    compares, and how many repeats they share (no test below ``MIN_REPEATS``)."""
    shared = [r for r in first if r in second]
    if len(shared) < MIN_REPEATS:
        return np.nan, "", len(shared)
    a = np.array([np.mean(first[r]) for r in shared])
    b = np.array([np.mean(second[r]) for r in shared])
    values, compared, _ = repeat_values(a, b)
    return paired_t(compared), _VALUE_WORDS[values], len(shared)


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
        # Compared on the repeats that imaged both (each condition against the control of
        # its own repeats): pooling repeats one side was not imaged in reads a batch as an
        # effect. Two conditions with no repeat in common (two sub-experiments) get their d,
        # described (†), and no test: the batch between their repeats would be in it.
        common = set(repeats_a) & set(repeats_b)
        if common:
            a = a[a["repeat"].isin(common)]
            b = b[b["repeat"].isin(common)]
            rng = group_rng(random_state, {"feature": feature, "first": first, "second": second})
            d, p = cell_randomization(
                a["value"].to_numpy(), b["value"].to_numpy(), n_permutations, rng
            )
            status = "ok" if np.isfinite(p) else "one value everywhere"
        else:
            d, p = cohen_d(a["value"].to_numpy(), b["value"].to_numpy()), np.nan
            status = "imaged in different repeats"
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
    table = pd.DataFrame(rows, columns=[c for c in PAIR_COLUMNS if c not in ("q", "q_paired")])
    for p_column, q_column in (("p", "q"), ("p_paired", "q_paired")):
        table[q_column] = benjamini_hochberg(table[p_column].to_numpy(float)) if len(table) else []
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
    """A p-value in a cell: ``<0.001``, two significant digits below 0.01, four decimals
    near 0.05 (so a p either side of it never reads as 0.050), three below 0.1, else two."""
    if p < 0.001:
        return "<0.001"
    if 0.045 <= p < 0.055:
        return f"{p:.4f}"
    return f"{p:.2g}" if p < 0.01 else f"{p:.3f}" if p < 0.1 else f"{p:.2f}"


def _ink(colour: tuple[float, float, float]) -> str:
    return "white" if sum(colour) / 3 < 0.55 else "#0b0b0b"


def _cells_d(pair: Any, row: Any) -> tuple[str, tuple, str]:
    """Panel B, lower left: d of the row's condition minus the column's."""
    d = pair.d if pair.first == row else -pair.d
    text = "–" if not np.isfinite(d) else f"{d:+.2f}"
    if np.isfinite(d) and pair.shared_repeats == 0:
        text += "†"  # different repeats: the batch is in d too
    return text, d_colour(d), "normal"


def _cells_p(value: float) -> tuple[str, tuple, str]:
    """A p- or q-value cell: green at or below ``ALPHA``, "–" when there is none (a pair
    imaged in different repeats is not tested: its † is on its d)."""
    if not np.isfinite(value):
        return "–", p_colour(np.nan), "normal"
    return format_p(value), p_colour(value), "bold" if value <= ALPHA else "normal"


def _matrix(
    ax: Axes, conditions: list[Any], table: pd.DataFrame, panel: str, *, ylabels: bool
) -> None:
    """Draw panel ``"B"`` (d and the randomization q) or ``"C"`` (the paired p and q)."""
    n = len(conditions)
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
            if panel == "B":
                cell = _cells_d(pair, row) if i > j else _cells_p(float(pair.q))
            else:
                cell = _cells_p(float(pair.p_paired if i > j else pair.q_paired))
            text, colour, weight = cell
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
    labels = [str(c) for c in conditions]
    ax.set_xlim(0, n)
    ax.set_ylim(n, 0)
    ax.set_xticks(np.arange(n) + 0.5)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=font + 0.5)
    ax.set_yticks(np.arange(n) + 0.5)
    ax.set_yticklabels(labels if ylabels else [], fontsize=font + 0.5)
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    title = PANEL_TITLES[panel]
    ax.set_title(title.replace(": ", "\n", 1), fontsize=9.5, loc="left", color=MUTED, pad=8)


def add_pairwise(figure: Figure, table: pd.DataFrame, conditions: Sequence[Any]) -> None:
    """Panels B (d and the randomization test) and C (the paired t-test on the repeats'
    means) under the figure's plot, side by side when the figure is wide enough, else one
    under the other; the legend says what each shows. Nothing with fewer than two
    conditions."""
    conditions = list(conditions)
    n = len(conditions)
    if n < 2 or table.empty:
        return
    caption = getattr(figure, "caption", "")
    remove_legend(figure)  # written again under the new panels
    cell = 0.72 if n <= 8 else max(0.4, 5.6 / n)
    label_w = min(2.8, 0.11 * max(len(str(c)) for c in conditions) + 0.3)
    side = cell * n
    width = figure.get_size_inches()[0]
    beside = label_w + 0.3 + 2 * side + 0.7 <= width - 0.2
    row_h = side + label_w * 0.6 + 0.75  # the cells, the slanted labels, the panel title
    extra = row_h if beside else 2 * row_h
    grow_bottom(figure, extra)
    total = figure.get_size_inches()[1]
    left = label_w + 0.3
    bottom_b = (extra - side - 0.55) / total
    ax_b = figure.add_axes([left / width, bottom_b, side / width, side / total])
    if beside:
        ax_c = figure.add_axes([(left + side + 0.7) / width, bottom_b, side / width, side / total])
    else:
        ax_c = figure.add_axes([left / width, bottom_b - row_h / total, side / width, side / total])
    _matrix(ax_b, conditions, table, "B", ylabels=True)
    _matrix(ax_c, conditions, table, "C", ylabels=not beside)
    panel_letter(figure.axes[0], "A", pad=22)
    panel_letter(ax_b, "B", pad=8)
    panel_letter(ax_c, "C", pad=8)
    set_legend(figure, [sentences(caption), PAIR_NOTE, PAIRED_NOTE])


def _square(x: float, y: float, colour: tuple[float, float, float]) -> FancyBboxPatch:
    from matplotlib.patches import FancyBboxPatch

    return FancyBboxPatch(
        (x + 0.05, y + 0.05),
        0.9,
        0.9,
        boxstyle="round,pad=0,rounding_size=0.08",
        facecolor=colour,
        edgecolor="none",
    )


__all__ = [
    "ALPHA",
    "D_CAP",
    "N_PERMUTATIONS",
    "PAIRED_NOTE",
    "PAIR_COLUMNS",
    "PAIR_NOTE",
    "add_pairwise",
    "d_colour",
    "format_p",
    "p_colour",
    "p_shade",
    "pairwise_contrasts",
]
