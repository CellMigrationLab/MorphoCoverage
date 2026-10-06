"""The whole-table test in the report (private; ``retrospective``, ``_document``, the page).

Every other feature table is also tested whole, beside the per-feature tests of the
explainable features (owner, 2026-10-06; ``sampling_unit.whole_table``): for each condition
against its own control, does its centre in the table move away from the control's, the same
way in every biological repeat? In a report without explainable features it is the only
test of "Do the conditions differ?". Here:

- ``table_pairs``: the comparisons tested, each condition against its own control: the main
  rows of the contrast table with several controls; its rows across experiments are tested
  feature by feature only, as a difference of log ratios has no meaning in an embedding
  (said once, ``ACROSS_NOT_TESTED``);
- ``whole_table_in_report``: the tests, ``None`` without other tables or a comparison;
- ``whole_table_lines``: the words, a block per table, a line per comparison;
- ``with_whole_table``: the block in "Do the conditions differ?", the headline naming the
  significant shifts (so "At a glance" mentions them), the answer itself when there are no
  explainable features;
- ``whole_table_html``: the page's block, with its data as ``WHOLE_CSV``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from .._control import VS
from .._style import Status, wrap
from .._wording import count, listing, p_value
from ..metadata import CONDITION_COLUMN, REPEAT_COLUMN
from ..sampling_unit.contrast_repeats import TREND_REPEATS
from ..sampling_unit.whole_table import (
    ALPHA,
    MIN_REPEATS,
    ONE_EACH,
    WholeTable,
    whole_table_tests,
)

if TYPE_CHECKING:
    from ..retrospective import MicroscopistReport
    from .setup import Setup

WHOLE_CSV = "tests_whole_table.csv"
ACROSS_NOT_TESTED = (
    "The rows across experiments of the contrast table are tested feature by feature only: a "
    "difference of log ratios has no meaning in a table such as an embedding."
)
# The same without explainable features: nothing tests those rows (audit 8B).
ACROSS_NO_FEATURES = (
    "The rows across experiments of the contrast table are not tested: a difference of log "
    "ratios has no meaning in a table such as an embedding, and there are no explainable "
    "features to test them on."
)


def table_pairs(setup: Setup, fovs: pd.DataFrame) -> list[tuple[Any, Any]]:
    """Each condition and its own control, as the main verdicts compare them."""
    if setup.comparisons is not None:
        return [tuple(pair) for pair in setup.comparisons]
    control = (setup.single_control or {}).get(CONDITION_COLUMN)
    if control is None or CONDITION_COLUMN not in fovs:
        return []
    conditions = [c for c in dict.fromkeys(fovs[CONDITION_COLUMN]) if c != control]
    return [(condition, control) for condition in conditions]


def whole_table_in_report(
    objects: pd.DataFrame, fovs: pd.DataFrame, setup: Setup
) -> WholeTable | None:
    """The whole-table tests of every other table (see the module), on the raw values (a
    table is never divided by a control); ``None`` without tables, repeats or a comparison."""
    if not setup.feature_tables or REPEAT_COLUMN not in fovs:
        return None
    pairs = table_pairs(setup, fovs)
    if not pairs:
        return None
    result = whole_table_tests(
        objects,
        fovs,
        tables=dict(setup.feature_tables),
        pairs=pairs,
        random_state=setup.random_state,
        **setup.common,
    )
    table = setup.contrast_table
    across = table is not None and bool((table["kind"] != "own_control").any())
    said = ACROSS_NOT_TESTED if setup.columns else ACROSS_NO_FEATURES
    result.settings["across_not_tested"] = said if across else ""
    return result


def _line(row: Any) -> str:
    """One comparison in words (see the module)."""
    name = f"{row.condition}{VS}{row.control}"
    if row.status != "tested" and not row.repeats:
        return f"- {name}: not tested, no biological repeat holds both."
    if row.status == ONE_EACH:
        return f"- {name}: not tested, one observation a side and repeat, no spread to scale on."
    if row.status == "tested":
        verdict = (
            f"shifted the same way in {row.same_way} of {count(row.repeats, 'repeat')}, "
            f"p {p_value(row.p_repeats)} (q {p_value(row.q_repeats)}) across repeats"
        )
    else:
        verdict = f"not tested across repeats, {count(row.repeats, 'repeat')}"
    apart = f"; the centres {row.effect:.2f} within-group SDs apart" if pd.notna(row.effect) else ""
    return f"- {name}: {verdict}{apart}."


def _shuffled(result: WholeTable) -> str:
    """The units the exploratory p shuffles, in words: ``fields``, ``technical repeats``."""
    names = [str(n) for n in dict.fromkeys(result.tests["shuffled"].dropna())] or ["field"]
    words = {"field": "fields", "technical_repeat": "technical repeats"}
    return " or ".join(words.get(n, f"{n} groups") for n in names)


def whole_table_lines(result: WholeTable, indent: int = 2) -> list[str]:
    """A block per table, a line per comparison, then how the tests read (see the module)."""
    tests = result.tests
    lines = [
        wrap(
            "The whole tables, each condition against its own control on all of a table's "
            "columns at once:",
            indent,
        )
    ]
    for name, rows in tests.groupby("table", sort=False):
        columns = result.settings["tables"][name]
        left = result.settings["left_out"].get(name, 0)
        head = f"{name} ({count(columns, 'column')}"
        head += f"; {count(left, 'observation')} with a missing value left out):" if left else "):"
        lines.append(wrap(head, indent))
        lines += [wrap(_line(row), indent, hang=indent + 2) for row in rows.itertuples()]
    permutations = int(result.settings["permutations"])
    how = (
        "Across repeats, the verdict: each comparison's columns scaled on the spread of its "
        "two sides within their repeats; per repeat, the shift of the condition's centre from "
        "its control's (each field's observations averaged first), projected on the other "
        "repeats' mean shift; a one-sided paired t-test that the projections are positive, "
        "corrected for the repeats they share by an asymptotic high-dimensional correction, "
        f"not an exact one, from {MIN_REPEATS} repeats; q within each "
        "table and sub-experiment. Within-group SDs: the typical distance of an observation "
        "from its own group's centre. The CSV also gives an exploratory p with the labels of "
        f"the {_shuffled(result)} shuffled within each repeat ({permutations} permutations), "
        "never part of the verdict: it ignores any well effect the data do not name."
    )
    lines.append(wrap(how, indent))
    if result.settings.get("across_not_tested"):
        lines.append(wrap(result.settings["across_not_tested"], indent))
    return lines


def _significant(result: WholeTable) -> pd.DataFrame:
    """The rows significant across repeats (q at most ``ALPHA``)."""
    return result.tests[result.tests["q_repeats"] <= ALPHA]


def whole_headline(result: WholeTable) -> tuple[Status, str]:
    """The status and one-line answer of the whole-table tests: ✅ when a shift is
    significant across repeats, ⚠️ otherwise, ➖ when none could be tested."""
    tests = result.tests
    significant = _significant(result)
    if len(significant):
        parts = []
        for name, rows in significant.groupby("table", sort=False):
            named = ", ".join(
                f"{r.condition} (q {p_value(r.q_repeats)})" for r in rows.itertuples()
            )
            parts.append(f"the whole {name} table shifted across repeats in {named}")
        return Status.OK, "; ".join(parts)
    tested = tests[tests["status"] == "tested"]
    if len(tests) and tests["status"].eq(ONE_EACH).all():
        return Status.NOT_ASSESSED, "no whole-table test: one observation a side and repeat"
    if not len(tested):
        fewest = int(tests["repeats"].max()) if len(tests) else 0
        return Status.NOT_ASSESSED, (
            f"no whole-table test: at most {count(fewest, 'repeat')} hold a condition and its "
            f"control, and the test needs {MIN_REPEATS}"
        )
    # The trend from TREND_REPEATS repeats only (owner, 2026-10-06, audit 8): with fewer,
    # every repeat goes the same way without an effect too often (25% with 3).
    every = tested[(tested["same_way"] == tested["repeats"]) & (tested["repeats"] >= TREND_REPEATS)]
    if len(every):
        names = listing([str(c) for c in dict.fromkeys(every["condition"])], 4)
        return Status.CAUTION, (
            f"no whole-table shift significant across repeats; {names} shifted the same way in "
            "every repeat"
        )
    return Status.CAUTION, "no whole-table shift significant across repeats"


def with_whole_table(
    document: MicroscopistReport, result: WholeTable | None, explainable: bool
) -> MicroscopistReport:
    """The report with the whole-table tests in "Do the conditions differ?" (see the module);
    ``explainable`` says whether per-feature tests answer it beside them."""
    if result is None or not len(result.tests):
        return document
    from ._document import with_text

    lines = whole_table_lines(result)
    text = "\n".join(line.removeprefix("  ") for line in lines)
    status, answer = whole_headline(result)
    shown = answer if not explainable or status is Status.OK else None
    document = with_text(
        document, "conditions", text, status=status, headline=shown, short_text="\n".join(lines)
    )
    if status is Status.OK and document.statuses.get("conditions") is not Status.PROBLEM:
        document = replace(document, statuses={**document.statuses, "conditions": Status.OK})
    return document


def whole_table_html(
    result: WholeTable | None, folder: Path, render: Callable[[str], str], buttons: Callable
) -> str:
    """The page's block: the lines (``render`` makes them HTML) and the CSV's button."""
    if result is None or not len(result.tests):
        return ""
    path = folder / WHOLE_CSV
    result.tests.to_csv(path, index=False, float_format="%.6g")
    files = buttons([(f"figures/{WHOLE_CSV}", "Data (CSV)")])
    return (
        "<h3>The whole tables</h3>"
        + render("\n".join(whole_table_lines(result, indent=0)))
        + f'<p class="tools">{files}</p>'
    )


__all__ = [
    "WHOLE_CSV",
    "table_pairs",
    "whole_headline",
    "whole_table_html",
    "whole_table_in_report",
    "whole_table_lines",
    "with_whole_table",
]
