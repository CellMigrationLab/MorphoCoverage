"""The words around the page's maps (private; ``_html_maps``): the comparison of the feature
tables under the maps, what the line under each map says, the centred view, and each table's
share of the combined map's axes.
"""

from __future__ import annotations

import html
from typing import Any

from .._wording import count, percent
from ..feature_tables import EXPLAINABLE
from ..separation import K, Separation

# The values views, as their buttons name them.
VALUE_WORDS = {"raw": "Raw", "normalised": "Normalised", "centred": "Centred per repeat"}
# The Normalisation menu: the normalised view first, then the raw values, the centred last.
# The methods behind the Normalised button, named by what they do.
METHOD_WORDS = {
    "normalised": "Divided by the control mean",
    "centred": "Minus the repeat median",
}
# The button when a method is alone.
METHOD_NAMES = {"normalised": "Normalised", "centred": "Centred per repeat"}
VIEW_ORDER = {"normalised": 0, "raw": 1, "centred": 2}
VALUE_HELP = {
    "raw": "The values as measured, no correction.",
    "normalised": (
        "Each explainable feature divided by the mean of its control in the same biological "
        "repeat (a fold change); other feature tables are left as measured."
    ),
    "centred": (
        "Each column of an other feature table (an embedding) minus a median of its biological "
        "repeat, which removes a shift shared by the whole repeat; the explainable features are "
        "not shown in this view."
    ),
}

# What the line under each map says (``separation``), once under the maps.
NEIGHBOURS = (
    f"Under each map: of each observation's {K} nearest neighbours among the other observations "
    "of its biological repeat, never from its own field (nor its own well, when its condition "
    "has another well in the repeat), the share that have its condition, against that "
    "condition's share of those it could have, which is what chance gives; the median over "
    "the repeats, so a batch between repeats cannot pass for a separation. Where a condition "
    f"has fewer than {K} such observations in a repeat, the neighbours are as many as the "
    "fewest it has, so a perfect separation reads 100%. Within a condition, the same with "
    "the repeat, always on the raw values: how far the repeats separate, a batch effect. A "
    "description, not a test."
)

# With one condition, instead of ``NEIGHBOURS``: no line under the maps (audit 10, C29).
ONE_CONDITION = (
    "No separation of the conditions under the maps: the data hold one condition, so there "
    "is nothing to separate."
)

# What the centred view removes, said once (owner, 2026-10-06).
CENTRED = (
    "Centred: each column of an other feature table minus a median of its biological repeat, "
    "which removes a shift shared by a whole repeat, a batch, and keeps the differences "
    "within it."
)


def centred_note(line: str = "") -> str:
    """``CENTRED`` and the centring used (``_feature_space.centring_line``: which conditions
    set the centre, the repeats left uncentred; owner, 2026-10-06, audit 8)."""
    return " ".join(p for p in (CENTRED, line) if p)


def combined_words(columns: dict[str, int]) -> str:
    """The combined map's columns in words: ``17 explainable features and 768 columns of 2
    other tables, standardised, each table divided by its first singular value``."""
    explainable = columns.get(EXPLAINABLE, 0)
    others = {n: c for n, c in columns.items() if n != EXPLAINABLE}
    tables = f"{len(others)} other tables" if len(others) > 1 else f"the table {next(iter(others))}"
    return (
        f"{count(explainable, 'explainable feature')} and "
        f"{count(sum(others.values()), 'column')} of {tables}, standardised, each table "
        "divided by its first singular value"
    )


def share_lines(shares: dict[str, list[float]], explained: tuple[float, float]) -> list[str]:
    """Each shown axis of the combined map with each table's share of it: ``PC1: explainable
    42%, DINOv2 58%.`` (an axis that holds no variance left out)."""
    lines = []
    for axis in range(2):
        if explained[axis] <= 0:
            continue
        named = ", ".join(f"{name} {percent(values[axis])}" for name, values in shares.items())
        lines.append(f"PC{axis + 1}: {named}.")
    return lines


def _row(r: Any, name: str) -> str:
    """One row of the comparison: a space's columns, its scores by condition and repeat."""
    by_repeat = (
        f'<td class="num">{percent(r.repeat_share)}</td>'
        f'<td class="num">{percent(r.repeat_chance)}</td>'
        if r.conditions
        else '<td class="num">–</td><td class="num">–</td>'
    )
    return (
        f'<tr><td>{html.escape(name)}</td><td class="num">{r.features}</td>'
        f'<td class="num">{percent(r.share)}</td><td class="num">{percent(r.chance)}</td>'
        f'<td class="num">{r.repeats}</td>{by_repeat}</tr>'
    )


def comparison_table(found: Separation | None) -> str:
    """The feature tables side by side under the maps, the most separating first, the
    combined space apart (it joins the others, so it is not ranked among them), and the
    tables where the repeats separate more than the conditions, said plainly (owner,
    2026-10-06)."""
    if found is None or not found.spaces["repeats"].any():
        return ""
    ranked, apart = found.ranked()
    rows = "".join(_row(r, str(r.space)) for r in ranked.itertuples())
    rows += "".join(
        _row(r, f"{r.space}: all the tables at once, not ranked") for r in apart.itertuples()
    )
    batch = [str(name) for name in found.spaces["space"] if found.repeats_first(str(name))]
    said = (
        "<p>The repeats separate more than the conditions in "
        f"{html.escape(', '.join(batch))}: there, a batch between repeats is larger than the "
        "difference between conditions.</p>"
        if batch
        else ""
    )
    return (
        "<p><strong>How far the conditions and the repeats separate in each feature table"
        "</strong></p>"
        '<div class="table-wrap"><table><thead><tr><th>feature table</th><th>columns</th>'
        "<th>neighbours sharing the condition</th><th>by chance</th><th>repeats</th>"
        "<th>sharing the repeat, within a condition</th><th>by chance</th></tr>"
        f"</thead><tbody>{rows}</tbody></table></div>{said}"
    )


__all__ = [
    "CENTRED",
    "METHOD_NAMES",
    "METHOD_WORDS",
    "NEIGHBOURS",
    "VALUE_WORDS",
    "combined_words",
    "comparison_table",
    "share_lines",
]
