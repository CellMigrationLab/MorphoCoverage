""" "Did I image enough?" on the HTML page (private; ``html_report``).

The owner's layout (2026-10-02, "one sentence + table"): the answer in one sentence (with
fewer than 3 biological repeats, first that no difference can be called significant), the
declared rare phenotype and fingerprint in a line each, then a compact table, one row per
feature: the comparisons shown within single repeats and what would show the rest. Then one
feature at a time, as tabs in the order of "Do the conditions differ?" (significant across
repeats first): its comparisons, its table within each repeat and the precision of each
well's own mean; the rare phenotype and the fingerprint have their own tabs after the
features. The card ends with the assumptions of these answers, once. What every tab would
repeat (what a comparison is, that it is per repeat, what each well's own mean is for) is
said once, in the card's "?" (``_html_sections.enough_intro``).

With one feature, the same: the answer, the comparison table once (every comparison), the
precision line, the assumptions. The stopping rules, the repeats a new experiment would
need and what optimise would plan are not in this card: they are the page's last section,
"Plan the next experiment" (``_html_plan``).
"""

from __future__ import annotations

import html
from collections import Counter
from typing import Any

import pandas as pd

from .._caveats import caveat_lines
from .._style import NOT_ASSESSED, table
from .._wording import count, joined_clauses, listing, plural
from ..metadata import REPEAT_COLUMN
from ..retrospective.report_resolved import (
    SHORT_ROWS,
    _needing,
    _table,
    _unassessed,
    between_wells,
    has_resolution,
    resolved_parts,
)
from ..retrospective.report_status import group_word
from ..sampling_unit.contrast_repeats import MIN_REPEATS
from ..sampling_unit.contrast_sampling import MIN_FIELDS, MORE_WELLS, PRACTICAL
from ..sampling_unit.repeat_plan_text import field_words
from ..sufficiency_short import (
    field_bound_assumption,
    met_count,
    rule_tallies,
    uniformity_assumption,
)
from ._enough_features import ENOUGH, NOT_ASSESSABLE, NOT_YET, PARTLY, FeatureEnough
from ._enough_targets import (
    NO_TARGET,
    assessable_reason,
    fingerprint_lines,
    no_target_judged,
    rare_lines,
)
from ._feature_space import table_of
from ._html_render import text_html
from ._html_sections import _resolution
from ._html_tabs import tabs_html

_MARKS = {ENOUGH: "✅", PARTLY: "⚠️", NOT_YET: "⚠️", NOT_ASSESSABLE: NOT_ASSESSED}
# The summary table's rows shown before "Show all" (the style hides the others).
_FIRST_ROWS = 20
# Named in full up to this many; beyond, the first few and how many more.
_NAMED = 7


def _rows(contrast: Any) -> pd.DataFrame:
    """Every comparison, those without an interval included (``report_resolved``)."""
    return _table(contrast)


def repeats_seen(contrast: Any) -> int:
    """The most biological repeats any condition is compared with the control in."""
    rows = getattr(contrast, "table", None)
    if rows is None or rows.empty or REPEAT_COLUMN not in rows:
        return 0
    return int(rows.groupby("condition")[REPEAT_COLUMN].nunique().max())


def few_repeats(repeats: int) -> str:
    """With fewer than ``MIN_REPEATS`` biological repeats: that none can be significant."""
    if not 0 < repeats < MIN_REPEATS:
        return ""
    return (
        f"With {count(repeats, 'biological repeat')}, no difference can be called significant: "
        f"the test across repeats needs at least {MIN_REPEATS}."
    )


def _where(repeats: int) -> str:
    return "Within the biological repeat" if repeats == 1 else "Within each biological repeat"


def _named(names: list[str], others: bool) -> str:
    if others:
        return f"the other {count(len(names), 'feature')}"
    return listing(names, len(names) if len(names) <= _NAMED else 5)


def _rest(rows: pd.DataFrame, fields: str) -> str:
    """What the comparisons not shown are: ``most of the others are too small ...``."""
    unshown = int((rows["comparisons"] - rows["shown"]).sum())
    small, need = int(rows["too_small"].sum()), int(rows["needing"].sum())
    wells = int(rows["between_wells"].sum()) if "between_wells" in rows else 0
    beyond = f"too small to show even with {PRACTICAL} times the {fields}"
    if not unshown:
        return ""
    unassessed = int(rows["unassessed"].sum()) if "unassessed" in rows else 0
    said_of = (
        (small, f"are {beyond}"),
        (need, f"would show with more {fields}"),
        (wells, f"need {MORE_WELLS}, not more {fields}"),
        (unassessed, f"are not assessable, with fewer than {MIN_FIELDS} {fields} a side"),
    )
    for part, said in said_of:
        if part == unshown:
            return f"; the others {said}"
        if 2 * part > unshown:
            return f"; most of the others {said}"
    return ""


def answer_lines(features: FeatureEnough, fields: str, repeats: int) -> list[str]:
    """The answer over the features: with fewer than 3 repeats that none can be significant,
    then one sentence (see the module)."""
    rows = features.table
    total = len(rows)
    every = list(rows.loc[rows["shown"] == rows["comparisons"], "feature"].astype(str))
    some = list(rows.loc[(rows["shown"] > 0) & (rows["shown"] < rows["comparisons"]), "feature"])
    none = list(rows.loc[rows["shown"] == 0, "feature"].astype(str))
    head = f"{_where(repeats)}, the {fields} imaged show the difference from the control"
    one = bool((rows["comparisons"] == 1).all())
    if len(every) == total:
        everyone = (
            str(every[0])
            if total == 1
            else "both features"
            if total == 2
            else f"all {total} features"
        )
        body = f" for {everyone}" if one else f" in every comparison, for {everyone}"
    elif every:
        groups = [
            ("for" if one else "in every comparison for", every),
            ("in some for", [str(f) for f in some]),
            ("not for" if one else "in none for", none),
        ]
        groups = [(words, names) for words, names in groups if names]
        last = len(groups) - 1
        body = " " + joined_clauses(
            [f"{words} {_named(names, i == last)}" for i, (words, names) in enumerate(groups)]
        )
    else:
        comparisons = rows["comparisons"].astype(int)
        of = (
            str(comparisons.iloc[0])
            if comparisons.nunique() == 1
            else f"{comparisons.min()}-{comparisons.max()}"
        )
        low, high = int(rows["shown"].min()), int(rows["shown"].max())
        if not high:
            body = " in no comparison, for any feature"
        elif not low:
            body = f" in at most {high} of {of} comparisons for any feature"
        elif low == high:
            body = f" in {high} of {of} comparisons for every feature"
        else:
            body = f" in {low} to {high} of {of} comparisons for each feature"
        body += _rest(rows, fields)
    return [line for line in (few_repeats(repeats), f"{head}{body}.") if line]


def _needs(contrast: Any, fields: str, repeats: int = 0, project: bool = True) -> str:
    """``1 needs about 31 cells a side in its repeat; 1 too small even with 4 times the
    cells``, how many comparisons each count covers (with one biological repeat, no repeat
    to name); ``–`` when every one is shown."""
    rows = _rows(contrast)
    needing = _needing(rows)
    parts = []
    if len(needing) and project:
        counts = sorted(Counter(int(v) for v in needing["fields_to_resolve"]).items())
        (first, n), more = counts[0], counts[1:]
        verb = "needs" if n == 1 else "need"
        if len(counts) > 3:
            # Many conditions: a range, not a run-on list of counts.
            items = [f"{len(needing)} need about {first}-{counts[-1][0]}"]
        else:
            items = [f"{n} {verb} about {first}", *(f"{k} about {v}" for v, k in more)]
        their = "its" if len(needing) == 1 else "their"
        where = "" if repeats == 1 else f" in {their} repeat"
        parts.append(f"{listing(items, len(items))} {fields} a side{where}")
    elif len(needing):  # counted, not projected: the counts add up (audit A4)
        parts.append(f"{len(needing)} not projected (no candidate difference)")
    small = int(rows["too_small"].sum())
    if small:
        parts.append(f"{small} too small even with {PRACTICAL} times the {fields}")
    wells = int(between_wells(rows).sum())
    if wells:
        parts.append(f"{wells} {'needs' if wells == 1 else 'need'} {MORE_WELLS}")
    if _unassessed(rows):
        parts.append(f"{_unassessed(rows)} not assessable: fewer than {MIN_FIELDS} {fields} a side")
    return "; ".join(parts) or "–"


def compact_table(features: FeatureEnough, fields: str, repeats: int = 0) -> str:
    """One row per feature: the comparisons shown, and what would show the rest."""
    rows = features.table
    same = rows["comparisons"].nunique() == 1
    shown = (
        f"shown, of {count(int(rows['comparisons'].iloc[0]), 'comparison')}" if same else "shown"
    )
    cells = [
        [
            r["feature"],
            str(r["shown"]) if same else f"{r['shown']} of {r['comparisons']}",
            # No projection for a feature no test flags (audit 10, C25).
            _needs(features.contrasts[r["feature"]], fields, repeats, r.get("flagged", True)),
        ]
        for r in rows.to_dict("records")
    ]
    return table(cells, ["feature", shown, "to show the rest"], indent=0)


def own_mean_line(feature: str, tally: dict[str, Any] | None, word: str) -> str:
    """Whether each well's own mean of ``feature`` meets the mean-precision rule."""
    if not tally:
        return ""
    met, groups = tally["met"], tally["groups"]
    line = (
        f"Enough to pin down each {word}'s own mean of {feature}? The {tally['label']} was met "
        f"in {met_count(tally, word)}"
    )
    return line + (": enough." if met == groups else ".")


def _paragraphs(lines: list[str]) -> str:
    return "".join(f"<p>{html.escape(line)}</p>" for line in lines if line)


def _panel(contrast: Any, feature: str, tally: Any, word: str, repeats: int) -> str:
    """One feature's tab: its comparisons, its own mean, its table within each repeat."""
    lines = [*_resolution(contrast, repeats == 1), own_mean_line(feature, tally, word)]
    rows, _ = resolved_parts(contrast, indent=0, where=False)
    return f'<div class="reading">{_paragraphs(lines)}</div>' + text_html("\n".join(rows))


def _targets(result: Any, word: str, field: str) -> tuple[list[str], list[tuple[str, str, str]]]:
    """The declared rare phenotype and fingerprint: a line each for the top, a tab each."""
    setup, results = result.setup, result.results
    clustering = results.get("rare_clustering") if setup.rare is not None else None
    precision = results.get("fingerprint_precision")
    lines, tabs = [], []
    if clustering is not None and len(clustering.table):
        line, tab = _rare_target(clustering, setup, word, field)
        lines.append(line)
        tabs.append(tab)
    if precision is not None:
        criteria = list(dict.fromkeys(precision["criterion"]))
        lines += [
            _criterion_line(precision, criterion, len(criteria), word) for criterion in criteria
        ]
        if no_target_judged(precision, setup):
            lines.append(NO_TARGET)
        tabs += _fingerprint_tab(precision, setup, word, field)
    return lines, tabs


def _rare_target(
    clustering: Any, setup: Any, word: str, field: str
) -> tuple[str, tuple[str, str, str]]:
    """The rare phenotype's line for the top (in how many groups it was seen) and its tab."""
    rows = clustering.table
    k = setup.rare.min_fov_occurrences
    seen = rows["fovs_with_state"] >= k
    name = setup.rare_label or setup.rare.phenotype
    where = (
        f"in every {word}"
        if seen.all()
        else f"in {int(seen.sum())} of {count(len(rows), word)}"
        + _seen_by_condition(seen, rows, word)
    )
    line = f"The rare phenotype, {name}, was seen in at least {count(k, field)} {where}."
    text = "\n".join(rare_lines(clustering, setup, word, field))
    return line, ("rare phenotype", "✅" if seen.all() else "⚠️", text_html(text))


def _seen_by_condition(seen: pd.Series, rows: pd.DataFrame, word: str) -> str:
    """Where the rare phenotype was seen, condition by condition (``: every a well, ...``),
    or nothing when no condition saw it."""
    parts = []
    for condition, block in seen.groupby(rows["condition"], sort=False):
        if block.all():
            parts.append(("every", str(condition)))
        elif block.any():
            parts.append((f"{int(block.sum())} of {len(block)}", str(condition)))
    full = [c for kind, c in parts if kind == "every"]
    said = [f"every {listing(full, len(full))} {word}"] if full else []
    said += [f"{kind} {c} {plural(word)}" for kind, c in parts if kind != "every"]
    return f": {joined_clauses(said)}" if said else ""


def _criterion_line(precision: pd.DataFrame, criterion: Any, criteria: int, word: str) -> str:
    """The fingerprint line of one criterion (of ``criteria``): not assessable, not judged,
    or within its tolerance in how many groups."""
    judged = precision[precision["met"].notna()]
    own = judged[judged["criterion"] == criterion]
    rows = precision[precision["criterion"] == criterion]
    table_name = rows["table"].iloc[0] if "table" in rows else None
    # With other feature tables, each table its own verdict (owner, 2026-10-06).
    named = (
        f" of {table_name}"
        if isinstance(table_name, str)
        else ("" if criteria == 1 else f", {criterion},")
    )
    reason = assessable_reason(rows)
    what = rows["noun"].iloc[0] if "noun" in rows else "fingerprint"
    if reason is not None:  # a wide table and small groups (owner, 2026-10-06)
        return f"The {what}{named} is not assessable: {reason}."
    if not len(own):
        return f"The {what}{named} is not judged in any {word}."
    met = int(own["met"].sum())
    every = (precision["criterion"] == criterion).sum() == len(own)
    within = (
        f"in every {word}"
        if met == len(own) and every
        else f"in {met} of {count(len(own), word)}" + ("" if every else " judged")
    )
    return f"The {what}{named} is within its tolerance {within}."


def _fingerprint_tab(
    precision: pd.DataFrame, setup: Any, word: str, field: str
) -> list[tuple[str, str, str]]:
    """The fingerprint's tab, none when there is nothing to show."""
    text = "\n".join(
        fingerprint_lines(precision, word, plural(field), held_back=False, unit=setup.unit_word)
    )
    if not text:
        return []
    judged = precision[precision["met"].notna()]
    # No group judged: not assessed, never a ✅ over an empty set.
    mark = "✅" if judged["met"].astype(bool).all() else "⚠️"
    mark = mark if len(judged) else NOT_ASSESSED
    return [("fingerprint", mark, text_html(text))]


def _assumptions(contrast: Any, sufficiency: Any, labels: set[str]) -> str:
    """The assumptions of this card's answers, once: the comparison and its projection, and
    the mean-precision rules' (those of the stopping rules and of novelty are theirs)."""
    notes = list(resolved_parts(contrast)[1])
    if sufficiency is not None and not sufficiency.final.empty:
        final = sufficiency.final
        unit = getattr(sufficiency, "observation_unit", None) or "object"
        for tally in rule_tallies(sufficiency):
            own = final[final["config_id"].eq(tally["config_id"])]
            if tally["label"] not in labels:
                continue
            if own["assumes_spatial_uniformity"].astype(bool).any():
                notes.append(uniformity_assumption(tally["label"], unit))
            if bound := field_bound_assumption(tally["label"], own):
                notes.append(bound)
    return text_html("\n".join(caveat_lines(notes, indent=0)))


def _precision_tally(sufficiency: Any) -> dict[str, Any] | None:
    """The readout's first mean-precision rule defined for these observations."""
    tallies = rule_tallies(sufficiency) if sufficiency is not None else []
    return next((t for t in tallies if t["data_rule"] and t["undefined"] < t["groups"]), None)


def _features_body(result: Any, features: FeatureEnough, annotate: Any) -> str:
    contrast = next(iter(features.contrasts.values()))
    field, fields = field_words(contrast)
    word = group_word(result.results.get("sufficiency"))
    repeats = repeats_seen(contrast)
    targets, target_tabs = _targets(result, word, field)
    top = _paragraphs([*answer_lines(features, fields, repeats), *targets])
    # With explainable tables, the features under their table (owner, 2026-10-06).
    owner = table_of(result.setup)
    order = {name: i for i, name in enumerate(dict.fromkeys(owner.values()))}
    rows = features.table.to_dict("records")
    rows = sorted(rows, key=lambda r: order.get(owner.get(r["feature"]), 0))
    items = [
        (
            r["feature"],
            _MARKS[r["verdict"]],
            annotate(
                _panel(
                    features.contrasts[r["feature"]],
                    r["feature"],
                    features.precision.get(r["feature"]),
                    word,
                    repeats,
                )
            ),
        )
        for r in rows
    ]
    # Its last column wraps (the style), so the table fits the card.
    shown = f'<div class="compact">{text_html(compact_table(features, fields, repeats))}</div>'
    if len(rows) > _FIRST_ROWS:
        # Hundreds of features: the first rows, sortable over them all, and a button.
        shown = (
            f'<div class="long-table">{shown}<button type="button" class="show-all" '
            f'data-rows="{_FIRST_ROWS}" aria-expanded="false">Show all {len(rows)} features'
            "</button></div>"
        )
    tabs = [*items, *((n, m, annotate(t)) for n, m, t in target_tabs)]
    pointer = f"Each tab gives the comparisons and how precise each {word}'s own mean is."
    labels = {r for r in features.table["precision_rule"] if r}
    return (
        f'<div class="reading">{annotate(top)}</div>{shown}<h3>One feature at a time</h3>'
        f'<p class="note">{html.escape(pointer)}</p>'
        + tabs_html(
            "enough",
            tabs,
            "Features",
            [owner.get(r["feature"], "") for r in rows] + [""] * len(target_tabs)
            if owner
            else None,
        )
        + annotate(_assumptions(contrast, result.results.get("sufficiency"), labels))
    )


def _single_answer(contrast: Any, fields: str, repeats: int) -> str:
    rows = _rows(contrast)
    n, resolved = len(rows), int(rows["resolved"].sum())
    head = f"{_where(repeats)}, the {fields} imaged show the difference from the control"
    if resolved == n:
        return f"{head} in every comparison, all {n}." if n > 1 else f"{head}."
    parts = [f"{head} in {resolved} of {count(n, 'comparison')}"]
    needing = _needing(rows)
    if len(needing):
        values = needing["fields_to_resolve"].astype(float)
        low, high = int(values.min()), int(values.max())
        about = f"{low}" if low == high else f"{low} to {high}"
        their = "its" if len(needing) == 1 else "their"
        parts.append(
            f"{len(needing)} would show with about {about} {fields} a side in {their} repeat"
        )
    small = int(rows["too_small"].sum())
    if small:
        verb = "is" if small == 1 else "are"
        parts.append(
            f"{small} {verb} too small to show even with {PRACTICAL} times the {fields}, which "
            "is not evidence of no effect"
        )
    wells = int(between_wells(rows).sum())
    if wells:
        parts.append(f"{wells} that more {fields} will not resolve: {MORE_WELLS}")
    if _unassessed(rows):
        parts.append(
            f"{_unassessed(rows)} not assessable, with fewer than {MIN_FIELDS} {fields} a side"
        )
    return "; ".join(parts) + "."


def _single_body(result: Any, annotate: Any) -> str:
    contrast = result.results["contrast"]
    sufficiency = result.results.get("sufficiency")
    fields = field_words(contrast)[1]
    repeats = repeats_seen(contrast)
    word = group_word(sufficiency)
    top = [few_repeats(repeats), _single_answer(contrast, fields, repeats)]
    tally = _precision_tally(sufficiency)
    rows, _ = resolved_parts(contrast, indent=0, every=True)
    labels = {tally["label"]} if tally else set()
    return annotate(
        f'<div class="reading">{_paragraphs(top)}</div>'
        + text_html("\n".join(rows))
        + f'<div class="reading">'
        f"{_paragraphs([own_mean_line(result.setup.readout, tally, word)])}</div>"
        + _assumptions(contrast, sufficiency, labels)
    )


def enough_body(result: Any, annotate: Any) -> str | None:
    """The card's body (see the module); ``annotate`` links the glossary's words. ``None``
    without a comparison with the control to answer from (the text report's answer, the
    stopping rules, is then shown as it is)."""
    results = getattr(result, "results", {}) or {}
    features = results.get("enough_features")
    if features is not None:
        return _features_body(result, features, annotate)
    if has_resolution(results.get("contrast")):
        return _single_body(result, annotate)
    return None


def enough_help(result: Any) -> dict[str, Any]:
    """The design the card's "?" is written for (``intros_for``'s keywords)."""
    results = getattr(result, "results", {}) or {}
    features = results.get("enough_features")
    contrast = results.get("contrast")
    if features is not None:
        contrast = next(iter(features.contrasts.values()))
    listed = features is not None and any(
        len(_rows(c)) > SHORT_ROWS for c in features.contrasts.values()
    )
    return {
        "single_repeat": repeats_seen(contrast) == 1 if contrast is not None else False,
        "group": group_word(results.get("sufficiency")),
        "listed": listed,
        "compared": features is not None or has_resolution(contrast),
    }


__all__ = [
    "answer_lines",
    "compact_table",
    "enough_body",
    "enough_help",
    "few_repeats",
    "own_mean_line",
    "repeats_seen",
]
