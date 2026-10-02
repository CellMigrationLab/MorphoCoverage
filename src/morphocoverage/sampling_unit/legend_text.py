"""What to write in a figure legend for a plotted feature (question 9, ``figure_legend``).

One paragraph per plotted feature, under its figure on the page (owner, 2026-10-02), from
the numbers the report already holds (no new test), each one the page's table across
repeats gives for that feature and condition: the design in lab words (n biological
repeats, wells per condition, observations per well), what the figure draws, the test
behind "significant across repeats" with its correction family counted, and per condition
the change (the fold change the test compares, or the difference of the means), the mean
per-repeat Cohen's d, the adjusted p and p. With fewer than 3 biological repeats it says
that the result is one experiment, described, not tested (owner, 2026-10-01).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .._wording import change, count, listing, p_value
from ..schema import unit_plural
from ..statistics import repeat_values
from .contrast_cells import ALPHA


def _design(contrast: Any, rows: pd.DataFrame) -> str:
    """``3 independent biological repeats, one well per condition in each (11 to 27 cells a
    well)``: the repeats and the observations from the feature's ``rows`` (the page's table
    across repeats), the wells from the design (the readout's ``contrast``)."""
    table = contrast.table
    repeat_col = contrast.settings.get("repeat_col")
    total = int(table[repeat_col].nunique()) if repeat_col in table else 1
    # Per condition: a sub-experiment's own repeats, not every repeat of the report.
    repeats = int(rows["repeats"].max())
    wells = pd.concat([table["technical"], table["control_technical"]]).astype(int)
    if (wells == 1).all():
        per = "one well per condition in each"
    elif wells.min() == wells.max():
        per = f"{wells.min()} wells per condition in each"
    else:
        per = f"{wells.min()} to {wells.max()} wells per condition in each"
    if {"repeat_observations_min", "repeat_observations_max"} <= set(rows):
        low = int(rows["repeat_observations_min"].min())
        high = int(rows["repeat_observations_max"].max())
    else:
        cells = pd.concat([table["observations"], table["control_observations"]]).astype(int)
        low, high = int(cells.min()), int(cells.max())
    units = unit_plural(contrast.settings["observation_unit"])
    spread = f"{low}" if low == high else f"{low} to {high}"
    each = "a well" if (wells == 1).all() else "a condition and repeat"
    n = f"n = {repeats} independent biological repeat" + ("" if repeats == 1 else "s")
    if total > repeats:
        n += f" per sub-experiment ({total} in all)"
    return f"{n}, {per} ({spread} {units} {each})"


def _families(across: pd.DataFrame, features: int | None = None) -> str:
    """The correction family, counted: ``2 tests (siCCT8#5 and siCCT8#8 against siCTRL)``
    per sub-experiment."""
    if features and features > 1:
        return f"over {features} features and their conditions"
    tested = across[across["p_paired_t_normalised"].notna() | across["p_paired_t"].notna()]
    groups = list(tested.groupby("experiment", sort=False))
    sizes = sorted({len(block) for _, block in groups})
    if len(groups) == 1:
        block = groups[0][1]
        control = block["control_condition"].iloc[0]
        names = listing(list(block["condition"].astype(str)), 4)
        return f"{count(len(block), 'test')}: {names} against {control}"
    size = f"{sizes[0]}" if len(sizes) == 1 else f"{sizes[0]} to {sizes[-1]}"
    return f"{size} tests in each of {len(groups)} sub-experiments, each with its own control"


def _values(rows: pd.DataFrame) -> str:
    """What the paired test compared: the log ratios, the differences, or each where it
    applies."""
    kinds = set(rows["values"])
    if kinds == {"normalised"}:
        return "values normalised to each repeat's control, as log ratios"
    if kinds == {"raw"}:
        return "on the differences of the repeat means, a mean not being positive"
    return (
        "values normalised to each repeat's control as log ratios, the differences of the "
        "means where a mean is not positive"
    )


def _condition(row: Any) -> str:
    """``siCCT8#5: change ×0.77, d -0.57, adjusted p 0.26 (p 0.21)``, as the table has them."""
    parts = [f"change {change(row.change, row.values)}"] if pd.notna(row.change) else []
    if pd.notna(row.d_per_repeat_mean):
        parts.append(f"d {row.d_per_repeat_mean:+.2f}")
    if pd.notna(row.adjusted_p):
        parts.append(f"adjusted p {p_value(row.adjusted_p)} (p {p_value(row.p)})")
    return f"{row.condition}: " + ", ".join(parts)


def described_rows(contrast: Any) -> pd.DataFrame:
    """Without a table across repeats (one repeat, one feature): the readout's rows for its
    legend from the per-repeat table, the change and the mean d as the paired test would
    read them, no p."""
    table = getattr(contrast, "table", pd.DataFrame())
    if not len(table) or "mean" not in table:
        return pd.DataFrame()
    repeat_col = contrast.settings.get("repeat_col")
    rows = []
    for condition, block in table.groupby("condition", sort=False):
        block = block[block["mean"].notna() & block["control_mean"].notna()]
        if not len(block):
            continue
        values, _, fold = repeat_values(block["mean"], block["control_mean"])
        d = block["d_objects"].to_numpy(float)
        d = d[np.isfinite(d)]
        rows.append(
            {
                "condition": condition,
                "repeats": int(block[repeat_col].nunique()) if repeat_col in block else 1,
                "values": values,
                "change": fold,
                "d_per_repeat_mean": float(d.mean()) if len(d) else np.nan,
                "p": np.nan,
                "adjusted_p": np.nan,
            }
        )
    return pd.DataFrame(rows)


def figure_legend(contrast: Any, rows: pd.DataFrame, unit_words: str) -> str | None:
    """The figure-legend paragraph for one feature (see the module), or ``None`` without a
    contrast or rows. ``rows`` are the feature's rows of the page's table across repeats
    (``condition``, ``repeats``, ``values``, ``change``, ``d_per_repeat_mean``, ``p``,
    ``adjusted_p``, and when known the fewest and most observations of a side in one
    repeat), so every number is the table's; ``unit_words`` names the observations."""
    if contrast is None or not len(getattr(contrast, "table", ())) or not len(rows):
        return None
    drawn = (
        f"Points: {unit_words}, coloured by condition; bars: the median with its 95% interval "
        "(the tests use means)."
    )
    design = _design(contrast, rows)
    if rows["p"].isna().all():
        readings = [_condition(r) for r in rows.itertuples() if pd.notna(r.change)]
        described = f" ({'; '.join(readings)})" if readings else ""
        return (
            f"{design}. {drawn} With fewer than 3 biological repeats there is no test across "
            f"repeats: describe the result as one experiment{described}, and repeat it before "
            "testing; the cell-level screen is exploratory."
        )
    across = getattr(contrast, "repeats", None)
    features = contrast.settings.get("family_features")
    families = _families(across, features) if across is not None and len(across) else ""
    family = f" ({families})" if families else ""
    conditions = "; ".join(_condition(r) for r in rows.itertuples())
    return (
        f"{design}. {drawn} Statistics: paired t-test on the repeat means ({_values(rows)}), "
        f"Benjamini-Hochberg within each sub-experiment{family}; significant at adjusted "
        f"p ≤ {ALPHA:g}; d: Cohen's d, the mean of the per-repeat values. {conditions}. "
        "Report the change and the adjusted p; the cell-level screen is exploratory."
    )


__all__ = ["described_rows", "figure_legend"]
