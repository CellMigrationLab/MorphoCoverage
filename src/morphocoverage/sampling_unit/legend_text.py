"""What to write in a figure legend for the readout (question 9, ``legend_box``).

One paragraph per readout, from the numbers the report already holds (no new test): the
design in lab words (n biological repeats, wells per condition, observations per well),
what the figure draws, the test behind "significant across repeats" with its correction
family counted (the conditions of each sub-experiment), and per condition the change in the
feature's own terms, Cohen's d and q. With one biological repeat it says that the result is
one experiment, described, not tested (owner, 2026-10-01).
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from .._wording import count, listing, p_value
from ..schema import unit_plural
from .contrast_cells import ALPHA
from .reading_text import change_words


def _design(contrast: Any) -> str:
    """``3 independent biological repeats, one well per condition in each (11 to 27 cells a
    well)``."""
    table = contrast.table
    repeat_col = contrast.settings.get("repeat_col")
    total = int(table[repeat_col].nunique()) if repeat_col in table else 1
    # Per condition: a sub-experiment's own repeats, not every repeat of the report.
    repeats = (
        int(table.groupby("condition")[repeat_col].nunique().max()) if repeat_col in table else 1
    )
    wells = pd.concat([table["technical"], table["control_technical"]])
    per = "one well per condition in each" if (wells == 1).all() else "wells per condition: "
    if not (wells == 1).all():
        per += f"{int(wells.min())} to {int(wells.max())}"
    cells = pd.concat([table["observations"], table["control_observations"]]).astype(int)
    units = unit_plural(contrast.settings["observation_unit"])
    spread = f"{cells.min()}" if cells.min() == cells.max() else f"{cells.min()} to {cells.max()}"
    each = "a well" if (wells == 1).all() else "a condition and repeat"
    n = f"n = {repeats} independent biological repeat" + ("" if repeats == 1 else "s")
    if total > repeats:
        n += f" per sub-experiment ({total} in all)"
    return f"{n}, {per} ({spread} {units} {each})"


def _families(across: pd.DataFrame, features: int | None = None) -> str:
    """The correction family, counted: ``2 tests (siCCT8#5 and siCCT8#8 against siCTRL)``
    per sub-experiment."""
    tested = across[across["p_paired_t_normalised"].notna() | across["p_paired_t"].notna()]
    if features and features > 1:
        return f"over {features} features and their conditions"
    groups = list(tested.groupby("experiment", sort=False))
    sizes = sorted({len(block) for _, block in groups})
    if len(groups) == 1:
        block = groups[0][1]
        control = block["control_condition"].iloc[0]
        names = listing(list(block["condition"].astype(str)), 4)
        return f"{count(len(block), 'test')}: {names} against {control}"
    size = f"{sizes[0]}" if len(sizes) == 1 else f"{sizes[0]} to {sizes[-1]}"
    return f"{size} tests in each of {len(groups)} sub-experiments, each with its own control"


def _condition(row: pd.Series, contrast: Any) -> str:
    q, p = row["q_paired_t_normalised"], row["p_paired_t_normalised"]
    if pd.isna(q):
        q, p = row["q_paired_t"], row["p_paired_t"]
    parts = [change_words(contrast, row["condition"])]
    if pd.notna(row["d_mean"]):
        parts.append(f"d {row['d_mean']:+.2f}")
    if pd.notna(q):
        parts.append(f"q {p_value(q)} (p {p_value(p)})")
    return f"{row['condition']}: " + ", ".join(x for x in parts if x)


def legend_box(contrast: Any, unit_words: str) -> str | None:
    """The figure-legend paragraph for the readout (see the module), or ``None`` without a
    contrast. ``unit_words`` names the observations (``cells``)."""
    if contrast is None or not len(getattr(contrast, "table", ())):
        return None
    feature = contrast.settings["feature"]
    across = getattr(contrast, "repeats", None)
    drawn = (
        f"Points: {unit_words}, coloured by condition; bars: the median with its 95% interval "
        "(the tests use means)."
    )
    if across is None or not len(across) or across["p_paired_t"].isna().all():
        readings = [
            f"{c}: {change_words(contrast, c)}" for c in dict.fromkeys(contrast.table["condition"])
        ]
        return (
            f"For a figure legend ({feature}): {_design(contrast)}. {drawn} With fewer than 3 "
            "biological repeats there is no test across repeats: describe the result as one "
            "experiment (" + "; ".join(r for r in readings if not r.endswith(": ")) + "), and "
            "repeat it before testing; the cell-level screen is exploratory."
        )
    rows = "; ".join(_condition(row, contrast) for _, row in across.iterrows())
    families = _families(across, contrast.settings.get("family_features"))
    return (
        f"For a figure legend ({feature}): {_design(contrast)}. {drawn} Statistics: paired "
        "t-test on the repeat means (values normalised to each repeat's control, as log "
        f"ratios), Benjamini-Hochberg within each sub-experiment ({families}); "
        f"significant at q ≤ {ALPHA:g}; d: Cohen's d, the mean of the per-repeat values. "
        f"{rows}. Report the change and q; the cell-level screen is exploratory."
    )


__all__ = ["legend_box"]
