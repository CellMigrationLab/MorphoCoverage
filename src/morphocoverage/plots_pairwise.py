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
  comparisons: they describe, as the notebook's do. With several controls, the normalised
  values divide each condition by its own control, so only two conditions normalised to the
  same control are compared there; the other pairs are marked ≠, "different references: not
  compared" (owner, 2026-10-05).
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

import textwrap
from collections.abc import Mapping, Sequence
from itertools import combinations
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from ._checks import as_count, check_seed
from ._figure_layout import grow_bottom, panel_letter, remove_legend, sentences, set_legend
from ._palette import MUTED
from ._seeds import group_rng
from ._wording import shortened
from .data_check import validate_input_tables
from .metadata import CONDITION_COLUMN, REPEAT_COLUMN, sorted_labels
from .sampling_unit.contrast_cells import cell_randomization
from .sampling_unit.contrast_repeats import MIN_REPEATS
from .schema import fov_positions
from .statistics import ALPHA, benjamini_hochberg, cohen_d, log_scale, paired_t, repeat_values

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.patches import FancyBboxPatch

D_CAP = 2.0
N_PERMUTATIONS = 1999
# The longest condition name a matrix writes in full: longer ones are cut with an ellipsis
# (the pairs' CSV holds them whole), so the slanted labels stay clear of the legend.
LABEL_CHARACTERS = 24
# The panels' titles, with what each is good for (owner, 2026-10-02).
PANEL_TITLES = {
    "B": "d · randomization test: discovery, false positives when repeats differ",
    "C": "paired t-test, repeat means: decides, conservative with few repeats",
}
# Which test each triangle of a panel shows (owner, 2026-10-02): the lower left below the
# diagonal, the upper right above it.
PANEL_KEYS = {
    "B": "◣ lower left: Cohen's d · ◥ upper right: adjusted p",
    "C": "◣ lower left: p · ◥ upper right: adjusted over this figure only",
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
# Several controls (owner, 2026-10-05): each condition is normalised to its own control, so
# in the normalised values only two conditions normalised to the same control are compared.
APART = "different references: not compared"
APART_MARK = "≠"
PAIR_NOTE_SEVERAL = PAIR_NOTE.replace(
    "the row and the column of the control answer the question",
    "the pair of a condition and its own control answers the question",
).replace(
    "they are not tested, since the difference between their repeats would be in it.",
    "they are not tested, since the difference between their repeats would be in it. "
    f"{APART_MARK}: in the normalised values, two conditions normalised to different "
    "controls, so not compared there; the raw values compare them.",
)
PAIRED_NOTE = (
    f"{PAIRED_READING} The paired t-test on the biological repeats' means: each condition's "
    "mean paired with the other's in every repeat that imaged both (on the log ratios when "
    "every value of the feature is positive). Lower left: its p-value, unadjusted; upper "
    "right: adjusted over the pairs of this figure only (Benjamini-Hochberg); the verdict's "
    "q is adjusted over every feature and condition of its sub-experiment, so the two can "
    "fall either side of 0.05. Green at ≤ 0.05, darker the smaller; – : fewer than 3 "
    "repeats imaged both."
)


def verdict_note(verdicts: Mapping[Any, float]) -> str:
    """The verdict's q of each condition against its control, the feature's family (audit
    10, owner, 2026-10-06), so panel C's q, adjusted over the figure only, is never read as
    the verdict: "Verdict q (the feature's family): KD 0.074; KD2 –."; empty without any."""
    if not verdicts:
        return ""
    parts = [
        f"{condition} {format_p(q) if np.isfinite(q) else '–'}" for condition, q in verdicts.items()
    ]
    return f"Verdict q (the feature's family): {'; '.join(parts)}."


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


def _paired(first: dict, second: dict, log: bool | None = None) -> tuple[float, str, int]:
    """The paired t-test of two conditions on the repeats they share, on the feature's scale
    ``log`` (``statistics.log_scale``), the values it compares, and how many repeats they
    share (no test below ``MIN_REPEATS``)."""
    shared = [r for r in first if r in second]
    if len(shared) < MIN_REPEATS:
        return np.nan, "", len(shared)
    a = np.array([np.mean(first[r]) for r in shared])
    b = np.array([np.mean(second[r]) for r in shared])
    values, compared, _ = repeat_values(a, b, log)
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
    references: Mapping[Any, Any] | None = None,
    scale: str | None = None,
) -> pd.DataFrame:
    """One row per pair of ``conditions`` (all, the ``control`` first, by default); see the
    module. ``first`` and ``second`` follow the order of the conditions; each pair's shuffles
    are seeded from ``random_state``, the feature and the pair. ``references``, each
    condition's own control when the values are normalised to it (several controls): a pair
    of conditions normalised to different controls gets no d and no test (``APART``).
    ``scale``: the feature's declared scale (``Setup.scales``), else the automatic rule."""
    # Zero shuffles would give every pair p = 1 as if the test had run.
    n_permutations = as_count(n_permutations, "n_permutations")
    random_state = check_seed(random_state)
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
    log = log_scale(frame["value"].to_numpy(), scale, repr(feature))
    rows = []
    for first, second in combinations(order, 2):
        a, b = by_condition.get(first), by_condition.get(second)
        if a is None or b is None:
            continue
        if references is not None and references.get(first) != references.get(second):
            rows.append(_apart(feature, first, second, a, b))
            continue
        rows.append(_pair(feature, (first, second), a, b, log, n_permutations, random_state))
    table = pd.DataFrame(rows, columns=[c for c in PAIR_COLUMNS if c not in ("q", "q_paired")])
    for p_column, q_column in (("p", "q"), ("p_paired", "q_paired")):
        table[q_column] = benjamini_hochberg(table[p_column].to_numpy(float)) if len(table) else []
    return table[PAIR_COLUMNS]


def _pair(
    feature: str,
    pair: tuple[Any, Any],
    a: pd.DataFrame,
    b: pd.DataFrame,
    log: bool,
    n_permutations: int,
    random_state: int,
) -> dict:
    """One pair of conditions compared: d and the cell-level test on the repeats that imaged
    both, the paired test on the repeat means."""
    first, second = pair
    repeats_a = {r: g.to_numpy() for r, g in a.groupby("repeat", sort=False)["value"]}
    repeats_b = {r: g.to_numpy() for r, g in b.groupby("repeat", sort=False)["value"]}
    p_paired, paired_values, shared = _paired(repeats_a, repeats_b, log)
    # Compared on the repeats that imaged both (each condition against the control of
    # its own repeats): pooling repeats one side was not imaged in reads a batch as an
    # effect. Two conditions with no repeat in common (two sub-experiments) get their d,
    # described (†), and no test: the batch between their repeats would be in it.
    common = set(repeats_a) & set(repeats_b)
    if common:
        a = a[a["repeat"].isin(common)]
        b = b[b["repeat"].isin(common)]
        rng = group_rng(random_state, {"feature": feature, "first": first, "second": second})
        d, p = cell_randomization(a["value"].to_numpy(), b["value"].to_numpy(), n_permutations, rng)
        status = "ok" if np.isfinite(p) else "one value everywhere"
    else:
        d, p = cohen_d(a["value"].to_numpy(), b["value"].to_numpy()), np.nan
        status = "imaged in different repeats"
    return {
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


def _apart(feature: str, first: Any, second: Any, a: pd.DataFrame, b: pd.DataFrame) -> dict:
    """A pair normalised to different controls: counted, not compared (``APART``)."""
    shared = len(set(a["repeat"]) & set(b["repeat"]))
    return {
        "feature": feature,
        "first": first,
        "second": second,
        "d": np.nan,
        "observations": len(a),
        "second_observations": len(b),
        "shared_repeats": shared,
        "p": np.nan,
        "p_paired": np.nan,
        "paired_values": "",
        "status": APART,
    }


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
    if pair.status == APART:
        return APART_MARK, d_colour(np.nan), "normal"
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


def _title(panel: str, side: float) -> str:
    """A panel's title and the key of its triangles, wrapped to its width (``side`` inches at
    9.5 pt), so it never runs into the panel beside it."""
    width = max(16, int(side / 0.075))
    # One line per triangle, so a glyph never parts from its words.
    parts = [*PANEL_TITLES[panel].split(": ", 1), *PANEL_KEYS[panel].split(" · ")]
    return "\n".join(line for part in parts for line in textwrap.wrap(part, width))


def _matrix(
    ax: Axes,
    conditions: list[Any],
    table: pd.DataFrame,
    panel: str,
    *,
    ylabels: bool,
    side: float,
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
    labels = [shortened(c, LABEL_CHARACTERS) for c in conditions]
    ax.set_xlim(0, n)
    ax.set_ylim(n, 0)
    ax.set_xticks(np.arange(n) + 0.5)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=font + 0.5)
    ax.set_yticks(np.arange(n) + 0.5)
    ax.set_yticklabels(labels if ylabels else [], fontsize=font + 0.5)
    ax.tick_params(length=0)
    # Each cell reads its row's condition against its column's.
    ax.set_xlabel("against", fontsize=font, color=MUTED, labelpad=4)
    if ylabels:
        ax.set_ylabel("condition", fontsize=font, color=MUTED, labelpad=4)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(_title(panel, side), fontsize=9.5, loc="left", color=MUTED, pad=8)


def add_pairwise(
    figure: Figure,
    table: pd.DataFrame,
    conditions: Sequence[Any],
    several: bool = False,
    verdicts: Mapping[Any, float] | None = None,
) -> None:
    """Panels B (d and the randomization test) and C (the paired t-test on the repeats'
    means) under the figure's plot, side by side when the figure is wide enough, else one
    under the other; the legend says what each shows (``several``: with several controls,
    ``PAIR_NOTE_SEVERAL``) and, with ``verdicts`` (each condition's verdict q against its
    control, the feature's family), those q beside panel C's (``verdict_note``). Nothing
    with fewer than two conditions."""
    conditions = list(conditions)
    n = len(conditions)
    if n < 2 or table.empty:
        return
    caption = getattr(figure, "caption", "")
    remove_legend(figure)  # written again under the new panels
    cell = 0.72 if n <= 8 else max(0.4, 5.6 / n)
    # The labels and, beside them, the axis title ("condition", "against").
    longest = max(len(shortened(c, LABEL_CHARACTERS)) for c in conditions)
    label_w = min(2.8, 0.11 * longest + 0.3) + 0.25
    side = cell * n
    width = figure.get_size_inches()[0]
    beside = label_w + 0.3 + 2 * side + 0.7 <= width - 0.2
    # The cells, the slanted labels and the panel title (two lines, more when it wraps).
    lines = max(_title(p, side).count("\n") + 1 for p in ("B", "C"))
    # Above the cells, the panel title (0.16 inch a wrapped line) and a gap that clears panel
    # A's tick labels; below them, the slanted labels and "against". With two conditions the
    # cells are narrow, the title wraps to many lines, and the old fixed gap let it overprint
    # panel A's labels (audit A7).
    above = 0.16 * lines + 0.45
    below = label_w * 0.6 + 0.4
    row_h = side + above + below
    extra = row_h if beside else 2 * row_h
    grow_bottom(figure, extra)
    total = figure.get_size_inches()[1]
    left = label_w + 0.3
    bottom_b = (extra - above - side) / total
    ax_b = figure.add_axes([left / width, bottom_b, side / width, side / total])
    if beside:
        ax_c = figure.add_axes([(left + side + 0.7) / width, bottom_b, side / width, side / total])
    else:
        ax_c = figure.add_axes([left / width, bottom_b - row_h / total, side / width, side / total])
    _matrix(ax_b, conditions, table, "B", ylabels=True, side=side)
    _matrix(ax_c, conditions, table, "C", ylabels=not beside, side=side)
    panel_letter(figure.axes[0], "A", pad=22)
    panel_letter(ax_b, "B", pad=8)
    panel_letter(ax_c, "C", pad=8)
    note = PAIR_NOTE_SEVERAL if several else PAIR_NOTE
    paired = " ".join(filter(None, [PAIRED_NOTE, verdict_note(verdicts or {})]))
    set_legend(figure, [sentences(caption), note, paired])


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
    "APART",
    "APART_MARK",
    "D_CAP",
    "N_PERMUTATIONS",
    "PAIRED_NOTE",
    "PAIR_COLUMNS",
    "PAIR_NOTE",
    "PAIR_NOTE_SEVERAL",
    "PANEL_KEYS",
    "add_pairwise",
    "d_colour",
    "format_p",
    "p_colour",
    "p_shade",
    "pairwise_contrasts",
    "verdict_note",
]
