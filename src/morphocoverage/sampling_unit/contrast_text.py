"""The words of "Do the conditions differ, and how confidently?" (question 9), short and full.

``contrast_summary(contrast, detail)`` writes a ``ConditionContrast``:

- **short** (the default): what is compared in one sentence; with biological
  repeats kept apart, one table with a row per condition across the repeats
  (the direction in each repeat, the per-repeat d, the sign-flip and paired
  t-tests on the repeat means, ``contrast_repeats``); one table with a row per
  contrast within a repeat (at most ``SHORT_CONTRASTS`` rows, else a pointer to
  the full form) (fields, observations and technical repeats on each side,
  Cohen's d with observations, fields and technical repeats as the unit, the
  randomization test), one table with the JCS effect size's same-sign shares
  over the draws (the range over the contrasts per sample size), then each
  caveat once: why a contrast has no test, an unstable d, what the test and
  the draws cannot say;
- **full**: the same comparison with every number: the counts and effect
  sizes of each contrast, each test with its unit, assignments and
  Bonferroni factor, the JCS effect size of every draw size with its median
  and interquartile band, and every caveat once, naming the contrasts it
  concerns.

``contrast_parts(contrast, indent)`` gives the short form as ``(lines,
cautions)``, so the report can print the caveats of question 9 once.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .._caveats import assumption, caveat_lines
from .._style import SINGLE_LEVEL, among, check_detail, span, table, wrap
from .._wording import column_word, count, listing, number, percent
from ..metadata import REPEAT_COLUMN
from ..schema import unit_plural
from ._units import FIELD, NOT_INDEPENDENT, UNKNOWN_WELLS_ADVICE, opposite_signs
from .cells_text import cells_parts
from .object_rules import RANDOM_OBJECTS, WHOLE_FOVS
from .reading_text import readings_parts
from .reagents_text import reagents_parts
from .repeat_plan_text import plan_parts
from .repeats_text import _range, _repeat_head, _repeat_rows, _repeats, repeat_cautions

NO_CONTROL = "no control group in"
# Units a side at or below which a d on unit means is not computed: from so few means a d
# can be very large by chance, so it could not help.
FEW_UNITS_D = 3
P_ALPHA = 0.05  # the p-value at which contrasts of opposite signs are flagged
# Per-repeat contrasts printed in the short form when the across-repeat table is there.
SHORT_CONTRASTS = 12
_MEANS_ONLY = (
    "The test and the effect sizes compare means: a difference confined to a subpopulation of "
    "{words} or to the shape of the distribution is not tested (see the unusual {words} and "
    "the rare label)."
)
_DECLARED = (
    "whole fields, treated as independent draws as declared (fields_independent; if they came "
    "from a few wells or dishes, the p-value tests those wells, not the condition)"
)


def _short_statistic(value: float) -> str:
    """As ``statistic``, in a table cell: ``inf`` for a perfect separation."""
    if np.isinf(value):
        return "inf" if value > 0 else "-inf"
    return "–" if np.isnan(value) else f"{value:.2f}"


def _name(row: dict[str, Any], repeat_col: str | None) -> str:
    """``LATS1 vs MOCK, repeat 1A``: the contrast, and its repeat when repeats are kept
    apart."""
    name = f"{row['condition']} vs {row['control_condition']}"
    # CellPatcher's "single" names a folder level the data did not have: no repeat to name.
    if not repeat_col or str(row[REPEAT_COLUMN]) == SINGLE_LEVEL:
        return name
    return f"{name}, repeat {row[REPEAT_COLUMN]}"


def _head(contrast: Any, short: bool) -> str:
    """What is compared: the feature, the control, the repeats, the units."""
    settings = contrast.settings
    control, repeat_col, shared = settings["control"], settings["repeat_col"], settings.get("unit")
    within = f", within each {column_word(repeat_col)}" if repeat_col else ""
    unit = settings["observation_unit"]
    if short:
        label = " · ".join(str(v) for v in control.values())
        units = (
            f"{unit_plural(unit)}, fields and technical repeats (tech.)"
            if shared
            else f"{unit_plural(unit)} and fields"
        )
        return (
            f"{settings['feature']}, each condition against the control ({label}){within}; the "
            f"fields of a condition's analysis groups are pooled. Cohen's d with {units} as "
            "the unit:"
        )
    return (
        f"Feature {settings['feature']}, each condition against the control {control}{within}. "
        "The fields of a condition's analysis groups are pooled for this comparison. Objects "
        "are compared as Plot&Stats does and, since a microscope acquires them through fields, "
        "with the field as the unit"
        + (
            f" and, where each side has at least two, with the technical repeat ({shared}) as "
            "the unit."
            if shared
            else "."
        )
    )


def _test_cell(row: dict[str, Any]) -> str:
    if str(row["status"]).startswith(NO_CONTROL):
        return "no control fields"
    if row["status"] != "ok":
        return "no test"
    return f"{row['p_value']:.3g} ({row['p_bonferroni']:.3g})"


def _one_per_field(contrast: Any) -> bool:
    """Whether every compared field holds one observation (fields and observations coincide)."""
    table = contrast.table
    counted = table["fields"] > 0
    return bool(counted.any()) and bool(
        (table.loc[counted, "observations"] == table.loc[counted, "fields"]).all()
        and (
            table.loc[counted, "control_observations"] == table.loc[counted, "control_fields"]
        ).all()
    )


def _contrast_rows(contrast: Any, full: bool) -> tuple[list[list[str]], list[str]]:
    """One row per contrast: counts, effect sizes and (short) the test; and its columns."""
    settings = contrast.settings
    repeat_col, shared = settings["repeat_col"], settings.get("unit")
    words = unit_plural(settings["observation_unit"])
    with_technical = shared and contrast.table["d_technical"].notna().any()
    # One observation per field: the d on field means is the d on observations.
    by_field = not _one_per_field(contrast)
    technical = "tech. repeats" if full else "tech."
    columns = ["contrast", "fields", words, *([technical] if shared else [])]
    columns += [f"d {words}", *(["d fields"] if by_field else [])]
    columns += [f"d {technical}"] if with_technical else []
    unit = settings["observation_unit"]
    columns += [f"JCS, every {unit}"] if full else ["p (Bonferroni)"]
    rows = []
    for row in contrast.table.to_dict("records"):
        cells = [
            _name(row, repeat_col),
            f"{int(row['fields'])} vs {int(row['control_fields'])}",
            f"{int(row['observations'])} vs {int(row['control_observations'])}",
        ]
        if shared:
            cells.append(f"{int(row['technical'])} vs {int(row['control_technical'])}")
        cells.append(_short_statistic(row["d_objects"]))
        if by_field:
            cells.append(_short_statistic(row["d_fields"]))
        if with_technical:
            cells.append(_short_statistic(row["d_technical"]))
        cells.append(_short_statistic(row["effect_full"]) if full else _test_cell(row))
        rows.append(cells)
    return rows, columns


def across_tested(contrast: Any) -> bool:
    """Whether some condition has a test across the biological repeats."""
    repeats = _repeats(contrast)
    return bool(len(repeats) and repeats["p_paired_t"].notna().any())


# The per-repeat reason that the test across repeats answers: one well or dish a side.
ONE_UNIT = "only one technical repeat"


def design_limited(status: str) -> bool:
    """Whether a contrast is untested because of the design (one or few technical repeats or
    fields a side), not because something is missing."""
    return status.startswith((ONE_UNIT, "fewer than 2 fields", "too few technical repeats"))


def _untested(contrast: Any) -> list[str]:
    """Why contrasts have no test, once per reason, with the contrasts it concerns. A test the
    design cannot run (one or few technical repeats a side) is a note, not a caution: it is
    the usual design, and the test across repeats, when there is one, answers the question."""
    repeat_col = contrast.settings["repeat_col"]
    table_ = contrast.table
    n = len(table_)
    lines: list[str] = []
    untested = table_[table_["status"] != "ok"]
    limited = untested["status"].map(design_limited).astype(bool)
    if limited.any() and across_tested(contrast):
        lines.append(
            assumption(
                "Within a repeat the conditions are described by their effect sizes; they are "
                "tested across repeats (the first table)."
            )
        )
    for status, rows in untested.groupby("status", sort=False):
        names = [_name(r, repeat_col) for r in rows.to_dict("records")]
        which = among(len(rows), n, "contrast")
        named = "" if len(rows) == n else f" ({listing(names, 3)})"
        text = f"no well-level test within a repeat in {which}{named}: {status}"
        if design_limited(status):
            if not across_tested(contrast):
                lines.append(assumption(f"{text[0].upper()}{text[1:]}."))
            continue
        lines.append(text)
        if status == NOT_INDEPENDENT:
            lines.append(UNKNOWN_WELLS_ADVICE)
    return lines


def _cautions(contrast: Any) -> list[str]:
    """Every caveat of the contrast, once each (see the module)."""
    settings = contrast.settings
    words = unit_plural(settings["observation_unit"])
    cautions = _untested(contrast)
    sizes = contrast.table[["d_objects", "d_fields", "d_technical"]].to_numpy(dtype=float)
    if np.isinf(sizes).any():
        cautions.append("an infinite d is a perfect separation: no variation within either side")
    if ((contrast.table["permutation_unit"] == FIELD) & contrast.table["p_value"].notna()).any():
        cautions.append(assumption(f"the tests over {_DECLARED}"))
    note = settings.get("unit_note")
    if note:
        cautions.append(f"Note: {note}")
    passing = contrast.table[contrast.table["p_value"] <= P_ALPHA]
    cautions += opposite_signs(
        passing,
        ["condition"],
        settings["repeat_col"],
        lambda key: f"{key[0]} differs from the control at p <= {P_ALPHA:g}",
    )
    cautions.append(assumption(_MEANS_ONLY.format(words=words)))
    if not contrast.effects.empty:
        whole = contrast.effects[contrast.effects["sampling"] == WHOLE_FOVS]["resampled"]
        resampled = whole.dropna().unique().tolist()
        if resampled:
            partly = (
                "; where a side has fewer than two, they resample its fields only"
                if whole.isna().any()
                else ""
            )
            cautions.append(
                assumption(
                    "The whole-field draws take the fields of technical repeats "
                    f"({listing(resampled)}) drawn with replacement, so their same-sign share "
                    f"includes well-to-well variation{partly}; the random-object draws treat "
                    f"{words} as independent draws, so their same-sign share is not a confidence "
                    "in the effect."
                )
            )
        else:
            cautions.append(
                assumption(
                    f"The draws resample these fields and {words}, not wells or repeats: a "
                    "same-sign share describes these fields and is not a confidence in the "
                    "effect."
                )
            )
    return cautions


def _draws_short(contrast: Any, indent: int) -> list[str]:
    """The JCS effect size over the contrasts, and its same-sign shares per draw size."""
    words = unit_plural(contrast.settings["observation_unit"])
    full = contrast.table["effect_full"].astype(float)
    full = full[np.isfinite(full)]
    if full.empty:
        return []
    n = len(contrast.table)
    head = (
        f"JCS effect size (treated mean minus control mean, over the control SD) from every "
        f"{contrast.settings['observation_unit']}: {_range(full)}"
        + (f" over the {n} contrasts" if n > 1 else "")
    )
    effects = contrast.effects
    if effects.empty:
        return [wrap(f"{head}.", indent, hang=indent + 2)]
    head += f"; in draws of a fixed number of {words}, the share of draws with that sign" + (
        " (lowest to highest contrast):" if n > 1 else ":"
    )
    rows = []
    for size, block in effects.groupby("n_objects", sort=True):
        by = {s: b for s, b in block.groupby("sampling")}
        cells = [str(int(size))]
        for sampling in (RANDOM_OBJECTS, WHOLE_FOVS):
            shares = by[sampling]["same_sign"].dropna() if sampling in by else pd.Series([])
            cells.append(span(shares, "percent") if len(shares) else "–")
        taken = by[WHOLE_FOVS]["fields_treated"].dropna() if WHOLE_FOVS in by else []
        cells.append(f"{span(taken)} fields" if len(taken) else "–")
        rows.append(cells)
    columns = [words, f"random {words}", "whole fields", "whole fields took"]
    return [wrap(head, indent, hang=indent + 2), table(rows, columns, indent=indent + 2)]


def contrast_parts(contrast: Any, indent: int = 0) -> tuple[list[str], list[str]]:
    """The short form of the contrast under ``indent``, and its caveats (see the module)."""
    lines = [wrap(_head(contrast, short=True), indent, hang=indent + 2)]
    if contrast.table.empty:
        reason = contrast.settings.get("reason", "No condition to compare with the control.")
        return [*lines, wrap(reason, indent, hang=indent + 2)], []
    across = not _repeats(contrast).empty
    cell_lines, cell_notes = cells_parts(contrast, indent)
    reading = readings_parts(contrast, indent)
    if not across:
        # A pilot (one repeat): the cell-level test is the one that responds, so it leads.
        lines = reading + cell_lines + lines
    else:
        lines = [*reading, wrap(_repeat_head(contrast), indent, hang=indent + 2)]
        rows, columns = _repeat_rows(contrast)
        lines.append(table(rows, columns, indent=indent + 2))
        lines += reagents_parts(contrast, indent)[0] + cell_lines
        lines.append(
            wrap(f"Within each repeat: {_head(contrast, short=True)}", indent, hang=indent + 2)
        )
    rows, columns = _contrast_rows(contrast, full=False)
    within_untested = bool((contrast.table["status"] != "ok").all())
    if across and within_untested:
        # Nothing was tested within a repeat: the per-repeat d are in question 2 and the
        # across-repeat table, so the table would repeat them.
        lines[-1] = wrap(
            "No test within a single repeat: each condition has one well there, and the "
            "cells of one well are not independent replicates (the cell-level screen above "
            "treats them as such, and is exploratory). The per-repeat d are in question 2, "
            "and every number in the full report (report_full.txt).",
            indent,
            hang=indent + 2,
        )
    elif across and len(rows) > SHORT_CONTRASTS:
        lines[-1] = wrap(
            f"Within each repeat: {count(len(rows), 'contrast')}, one per condition and "
            "repeat, with Cohen's d on observations and on fields and the randomization test, "
            "are in the full report (report_full.txt).",
            indent,
            hang=indent + 2,
        )
    else:
        lines.append(table(rows, columns, indent=indent + 2))
    tests = _tests_short(contrast)
    lines += [wrap(tests, indent, hang=indent + 2)] if tests else []
    # Across repeats the reading and the tables answer; the JCS draws stay in the full report.
    lines += [] if across else _draws_short(contrast, indent)
    planned, plan_notes = plan_parts(contrast, indent)
    lines += planned
    side_notes = (reagents_parts(contrast, indent)[1] if across else []) + cell_notes
    return lines, [
        *repeat_cautions(contrast, full=False),
        *side_notes,
        *_cautions(contrast),
        *plan_notes,
    ]


def _tests_short(contrast: Any) -> str | None:
    """How the tested contrasts were tested, in one sentence: the unit shuffled, the fewest
    distinct assignments, exact or shuffled, the Bonferroni factor."""
    tested = contrast.table[contrast.table["status"] == "ok"]
    if tested.empty:
        return None
    units = sorted(set(tested["permutation_unit"]))
    over = listing(
        [
            "whole fields (declared independent draws)"
            if unit == FIELD
            else f"whole technical repeats ({unit})"
            for unit in units
        ]
    )
    by_unit = tested[tested["permutation_unit"] != FIELD]
    fewest = (
        f", as few as {int(by_unit['distinct_assignments'].min())} distinct assignments"
        if len(by_unit)
        else ""
    )
    exact = (
        "every assignment enumerated"
        if tested["p_exact"].all()
        else (f"{contrast.settings['n_permutations']} shuffles where the assignments are more")
    )
    n = int(contrast.table["p_value"].notna().sum())
    return (
        f"Randomization tests over {over}{fewest} ({exact}); p Bonferroni-corrected over "
        f"{count(n, 'comparison')}."
    )


def _test_sentence(contrast: Any, row: dict[str, Any], n_contrasts: int) -> str:
    """The randomization-test sentence of one tested contrast."""
    if row["permutation_unit"] == FIELD:
        shuffled = _DECLARED
    else:
        shuffled = (
            f"whole technical repeats ({row['permutation_unit']}; only "
            f"{int(row['distinct_assignments'])} distinct assignments exist)"
        )
    how = (
        "every assignment of the condition labels enumerated, so p is exact"
        if row["p_exact"]
        else f"{contrast.settings['n_permutations']} shuffles of the condition labels"
    )
    return (
        f"{_name(row, contrast.settings['repeat_col'])}: randomization test over {shuffled}, "
        f"{how}: p = {row['p_value']:.3g} (Bonferroni over "
        f"{count(n_contrasts, 'comparison')}: {row['p_bonferroni']:.3g})."
    )


def _draw_rows(contrast: Any) -> tuple[list[list[str]], list[str]]:
    """Per contrast and draw size: the median (IQR) and same-sign share of each sampling."""
    words = unit_plural(contrast.settings["observation_unit"])
    repeat_col = contrast.settings["repeat_col"]
    effects = contrast.effects
    resampled = effects["resampled"].notna().any() if "resampled" in effects else False
    rows = []
    for row in contrast.table.to_dict("records"):
        same = (
            effects[REPEAT_COLUMN].isna()
            if pd.isna(row[REPEAT_COLUMN])
            else effects[REPEAT_COLUMN] == row[REPEAT_COLUMN]
        )
        own = effects[same & (effects["condition"] == row["condition"])]
        for size, block in own.groupby("n_objects", sort=True):
            by = block.set_index("sampling")
            cells = [_name(row, repeat_col), str(int(size))]
            for sampling in (RANDOM_OBJECTS, WHOLE_FOVS):
                if sampling not in by.index:
                    cells.append("–")
                    continue
                r = by.loc[sampling]
                cell = f"{r['median']:.2f} ({r['q25']:.2f} to {r['q75']:.2f})"
                if not pd.isna(r["same_sign"]):
                    cell += f", {percent(r['same_sign'])}"
                cells.append(cell)
            whole = by.loc[WHOLE_FOVS] if WHOLE_FOVS in by.index else None
            taken = None if whole is None else whole["fields_treated"]
            cells.append("–" if taken is None or pd.isna(taken) else f"about {number(taken)}")
            if resampled:
                unit = None if whole is None else whole["resampled"]
                cells.append(unit if isinstance(unit, str) else "fields only")
            rows.append(cells)
    columns = ["contrast", words, f"random {words}", "whole fields", "fields"]
    return rows, [*columns, *(["resampled"] if resampled else [])]


def _full(contrast: Any) -> str:
    """Every number of the contrast, its tests and its draws (see the module)."""
    lines = [wrap(_head(contrast, short=False), 0, hang=2)]
    if contrast.table.empty:
        lines.append(contrast.settings.get("reason", "No condition to compare with the control."))
        return "\n".join(lines)
    lines += readings_parts(contrast, 0)
    if not _repeats(contrast).empty:
        rows, columns = _repeat_rows(contrast, full=True)
        lines += [wrap(_repeat_head(contrast), 0, hang=2), table(rows, columns)]
        lines += reagents_parts(contrast, 0)[0]
        lines += cells_parts(contrast, 0, full=True)[0]
        lines.append(wrap("Within each repeat:", 0, hang=2))
    else:
        lines += cells_parts(contrast, 0, full=True)[0]
    rows, columns = _contrast_rows(contrast, full=True)
    lines.append(table(rows, columns))
    tested = contrast.table[contrast.table["status"] == "ok"]
    n_contrasts = int(contrast.table["p_value"].notna().sum())
    lines += [
        wrap(_test_sentence(contrast, row, n_contrasts), 2, hang=4)
        for row in tested.to_dict("records")
    ]
    if not contrast.effects.empty:
        words = unit_plural(contrast.settings["observation_unit"])
        lines.append(
            wrap(
                "JCS effect size (treated mean minus control mean, over the control SD) from "
                f"draws of {words}: median (IQR), and the share of draws with the sign of the "
                f"effect from every {contrast.settings['observation_unit']}; fields: the mean "
                "number of acquired fields the whole-field draws took on the treated side.",
                0,
                hang=2,
            )
        )
        draws, names = _draw_rows(contrast)
        lines.append(table(draws, names))
    planned, plan_notes = plan_parts(contrast, 0, full=True)
    lines += planned
    agree_notes = reagents_parts(contrast, 0)[1]
    cell_notes = cells_parts(contrast, 0, full=True)[1]
    cautions = [
        *repeat_cautions(contrast),
        *agree_notes,
        *cell_notes,
        *_cautions(contrast),
        *plan_notes,
    ]
    lines += caveat_lines(cautions, indent=0, detail="full")
    return "\n".join(lines)


def contrast_summary(contrast: Any, detail: str = "short") -> str:
    """The contrast in a table per contrast and per draw size (short) or with every number
    (full); see the module."""
    if check_detail(detail) == "full":
        return _full(contrast)
    lines, cautions = contrast_parts(contrast)
    return "\n".join([*lines, *caveat_lines(cautions, indent=0)])


__all__ = [
    "FEW_UNITS_D",
    "NO_CONTROL",
    "P_ALPHA",
    "SHORT_CONTRASTS",
    "across_tested",
    "contrast_parts",
    "contrast_summary",
    "design_limited",
    "repeat_cautions",
]
