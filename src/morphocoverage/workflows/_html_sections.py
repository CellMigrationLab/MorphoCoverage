"""The HTML report's reading order and its plain-language pieces (private; ``html_report``).

- ``ORDER`` and ``TITLES``: the sections in the order a user acts on them, the data and the
  fields to check first, then the result, then how far the sampling supports it; titled in
  words, without the question numbers of the text report (whose 4, 5, 7 and 8 are the
  planning questions of ``optimise``).
- ``INTROS``: what each section is for, in one sentence, printed under its title.
- ``hidden``: a section whose answer is "not applicable" for this design is left out of the
  page (its reason stays in "What this dataset cannot tell me").
- ``what_checked``: what the run checked, in short lines, instead of the declaration's
  sentences (kept under "Full detail").
- ``enough_reading``: "Did I image enough?" answered in plain sentences, with what to do.
- ``unnumbered``: the text report's question numbers replaced by these titles.
- ``heatmap``: Cohen's d of every feature and condition against the control, as an SVG, with
  the discovery's gates marked.
"""

from __future__ import annotations

import html
import re
from typing import Any

import numpy as np

from .._style import Status, rule_label
from .._wording import count, listing, p_value, rate
from ..retrospective.report import QUESTIONS, SHORT_QUESTIONS
from ..retrospective.report_status import group_word
from ..sampling_unit.contrast_sampling import PRACTICAL
from ..sampling_unit.reading_text import CONFIRMED, condition_readings
from ..sampling_unit.repeat_plan_text import field_words
from ..schema import resolve_observation_unit
from ..sufficiency_short import rule_tallies

ORDER = ("overview", "inspect", "conditions", "enough", "profile", "unit", "controls")
TITLES = {
    "overview": "The data",
    "inspect": "Fields to check (quality control)",
    "conditions": "Do the conditions differ?",
    "enough": "Did I image enough?",
    "profile": "What limits the sampling?",
    "unit": "Why count fields, and not only cells?",
    "controls": "Can these features detect what I care about?",
}
INTROS = {
    "overview": "What the tables hold, and whether they read cleanly (missing values, copies).",
    "inspect": (
        "Fields that move a summary strongly, or that the copy checks name. Each may be an "
        "artefact (out of focus, debris, a field exported twice) or real biology: open their "
        "images before trusting the answers below."
    ),
    "conditions": (
        "Each condition against the control: the reading per condition first, then the "
        "figures, then the tests behind them."
    ),
    "enough": (
        "Whether the fields imaged are enough: to tell each condition from the control within "
        "a repeat, and to pin each group's mean down to the precision that was asked."
    ),
    "profile": (
        "What limits the sampling: how much the fields differ from each other, unusual cells, "
        "and whether rare states sit in a few fields."
    ),
    "unit": (
        "Whether the cells of one field behave like independent draws; when they do not, the "
        "field is the unit to count, and more cells per field help little."
    ),
    "controls": (
        "Whether the features can see a known change: a positive control (a condition known "
        "to change the phenotype) against the negative control."
    ),
}
# The heatmap's diverging colours (lower than the control, higher), and its neutral middle.
LOWER, HIGHER, MIDDLE = (42, 120, 214), (227, 73, 72), (240, 239, 236)
D_CAP = 2.0


def hidden(status: Status | None, headline: str) -> bool:
    """A section that only says this design cannot answer it."""
    return status is Status.NOT_ASSESSED and headline.lower().startswith("not applicable")


def _rules(setup: Any) -> str:
    try:
        unit = resolve_observation_unit(setup.observation_unit)[0]
        rules = [rule_label(rule, unit) or rule.name for rule in setup.stopping_rules()]
    except (AttributeError, TypeError, ValueError):
        return ""
    return listing(rules, 4)


def what_checked(setup: Any) -> list[str]:
    """What the run checked, one short line each (the declaration in words)."""
    unit = setup.unit_words
    lines = [f"Readout: {setup.readout}; its mean over all {unit} is the summary compared."]
    features = tuple(setup.columns)
    if len(features) > 1:
        lines.append(f"Features: {count(len(features), 'feature')} ({listing(list(features), 6)}).")
    control = (setup.control or {}).get("condition")
    if control is not None:
        lines.append(
            f"Negative control: {control}; every condition is compared with it, within each "
            "biological repeat."
        )
    if setup.positive is not None:
        lines.append(
            f"Positive control: {dict(setup.positive.selector).get('condition', '')}; it checks "
            "that the features see a change."
        )
    tolerances = " and ".join(rate(t) for t in setup.tolerances)
    lines.append(
        f"Precision asked: the standard error of the mean of {setup.readout} at most "
        f"{tolerances} of the mean."
    )
    if setup.fingerprint:
        named = "; ".join(c.description or c.name for c in setup.fingerprint)
        lines.append(f"Fingerprint: {named}.")
    lines.append(
        "Values: divided by each repeat's control mean."
        if setup.normalize
        else "Values: as measured (the tests between conditions also use them normalised)."
    )
    rules = _rules(setup)
    if rules:
        lines.append(f"Stopping rules checked: {rules}.")
    missing = []
    if setup.positive is None:
        missing.append("a positive control")
    if not setup.fingerprint:
        missing.append("a phenotypic fingerprint")
    if setup.rare is None:
        missing.append("a rare population")
    if setup.acquisition_order_col is None and setup.stage_cols is None:
        missing.append("the acquisition order and stage positions")
    if missing:
        lines.append(f"Not declared, so not checked: {listing(missing, 6)}.")
    return lines


def _repeats_advice(contrast: Any, fields: str) -> str:
    """What to do across repeats: the repeats the plan asks for the conditions not yet
    confirmed (``repeat_plan``, at the report's correction)."""
    plan = getattr(contrast, "plan", None)
    readings = condition_readings(contrast)
    if plan is None or plan.empty or readings.empty:
        return ""
    open_ = set(readings.loc[readings["verdict"] != CONFIRMED, "condition"])
    if not open_:
        return "What to do: every condition is confirmed across repeats; nothing more is needed."
    rows = plan[plan["condition"].isin(open_)]
    needed = rows["repeats_needed"].astype(float)
    finite = needed[np.isfinite(needed)]
    if finite.empty:
        return ""
    low, high = int(finite.min()), int(finite.max())
    about = f"{low}" if low == high else f"{low} to {high}"
    tail = (
        f"; more {fields} per repeat help little where the repeats differ"
        if (rows["limited_by"] == "between repeats").any()
        else ""
    )
    return (
        f"What to do: to confirm the {count(len(open_), 'condition')} not yet confirmed, plan "
        f"about {about} biological repeats in all (the planning table in "
        f"\u201c{TITLES['conditions']}\u201d){tail}."
    )


def _resolution(contrast: Any) -> list[str]:
    rows = getattr(contrast, "table", None)
    if rows is None or rows.empty or "d_low" not in rows:
        return []
    rows = rows[rows["d_low"].notna()]
    if rows.empty:
        return []
    _, fields = field_words(contrast)
    resolved = int(rows["resolved"].sum())
    small = int(rows["too_small"].sum())
    needing = rows[~rows["resolved"] & ~rows["too_small"]]
    verdict = "Yes" if needing.empty and not small else ("Partly" if resolved else "Not yet")
    lines = [
        (
            f"{verdict}: within one biological repeat at a time, the {fields} imaged show the "
            f"difference from the control in {resolved} of {count(len(rows), 'comparison')} "
            "(a condition in one repeat; shown: its 95% interval of d excludes 0)."
        )
    ]
    if len(needing):
        counts = needing["fields_to_resolve"].astype(float)
        low, high = int(counts.min()), int(counts.max())
        about = f"{low}" if low == high else f"{low} to {high}"
        lines.append(
            f"In {count(len(needing), 'comparison')} the interval still includes 0: about "
            f"{about} {fields} a side in that repeat would show a difference of that size."
        )
    if small:
        lines.append(
            f"In {count(small, 'comparison')} the difference is so small that even {PRACTICAL} "
            f"times the {fields} would not show it within one repeat."
        )
    lines.append(
        "This is per repeat. Whether the conditions differ across the repeats is answered in "
        f"\u201c{TITLES['conditions']}\u201d, which does not need each repeat to show it alone."
    )
    advice = _repeats_advice(contrast, fields)
    if advice:
        lines.append(advice)
    return lines


def _precision(sufficiency: Any, readout: str) -> list[str]:
    data = [t for t in rule_tallies(sufficiency) if t["data_rule"]] if sufficiency else []
    if not data:
        return []
    word = group_word(sufficiency)
    # A rule undefined everywhere (one per object, on tracks) says nothing about precision.
    defined = [t for t in data if t["undefined"] < t["groups"]]
    lines = [
        f"The {t['label']} is not defined for these observations, so it is not read."
        for t in data
        if t["undefined"] == t["groups"]
    ]
    if not defined:
        return lines
    first = defined[0]
    met, groups = first["met"], first["groups"]
    line = f"Each group's own mean of {readout}: the {first['label']} was met in {met} of " + (
        count(groups, word) + "."
    )
    if met < groups:
        line += (
            " Where it is not met, that mean is known less precisely than asked: it matters "
            "when the value of a mean is reported, less when conditions are compared."
        )
    return [line, *lines]


def enough_reading(result: Any) -> list[str]:
    """ "Did I image enough?" in plain sentences: the verdict, what it means, what to do."""
    results = getattr(result, "results", {}) or {}
    lines = _resolution(results.get("contrast"))
    return lines + _precision(results.get("sufficiency"), result.setup.readout)


def _numbered() -> dict[str, str]:
    """Each numbered title of the text report, to its title here (unnumbered)."""
    names = {}
    for key, title in [*QUESTIONS, *SHORT_QUESTIONS.items()]:
        plain = re.sub(r"^(\d\.|Foundation[.:])\s*", "", title)
        names[title] = TITLES.get(key, plain[:1].upper() + plain[1:])
    return names


_NUMBERED = _numbered()
_TITLE = re.compile("|".join(re.escape(t) for t in sorted(_NUMBERED, key=len, reverse=True)))
_NUMBER_KEYS = {title.split(".")[0]: key for key, title in QUESTIONS if title[:1].isdigit()}
_QUESTION = re.compile(r"\b[Qq]uestion (\d)\b")
_FOUNDATION = re.compile(r"\b(?:the )?Foundation\b")


def unnumbered(markup: str) -> str:
    """``markup`` with the text report's numbered titles and "question N" given the titles of
    this page, which has no numbers."""
    markup = _TITLE.sub(lambda m: html.escape(_NUMBERED[m.group(0)]), markup)

    def title(match: re.Match) -> str:
        key = _NUMBER_KEYS.get(match.group(1))
        name = TITLES.get(key) or _NUMBERED.get(dict(QUESTIONS).get(key, ""), "")
        return f"\u201c{html.escape(name)}\u201d" if name else match.group(0)

    markup = _FOUNDATION.sub(f"\u201c{html.escape(TITLES['unit'])}\u201d", markup)
    return _QUESTION.sub(title, markup)


def _colour(d: float) -> tuple[str, str]:
    """The fill for ``d`` and a text colour that reads on it."""
    if not np.isfinite(d):
        return "#f0efec", "#52514e"
    share = min(abs(d), D_CAP) / D_CAP
    pole = LOWER if d < 0 else HIGHER
    rgb = [round(m + (p - m) * share) for m, p in zip(MIDDLE, pole, strict=True)]
    text = "#ffffff" if share > 0.55 else "#0b0b0b"
    return "#" + "".join(f"{v:02x}" for v in rgb), text


def heatmap(discovery: Any, conditions: list[Any]) -> str:
    """Cohen's d of every feature (rows, best-ranked first) and condition (columns) against
    the control, as an SVG; ● a candidate at the cell level, ✓ also confirmed across
    repeats. Empty without tests."""
    if discovery is None or discovery.tests is None or not len(discovery.tests):
        return ""
    tests = discovery.tests
    features = list(discovery.ranking["feature"])
    present = set(tests["condition"])
    columns = [c for c in conditions if c in present] or sorted(present, key=str)
    # The column labels slant up to the right: room above and beside for the longest.
    longest = max(len(str(c)) for c in columns)
    cell_w, cell_h, left = 72, 24, 10 + 7 * max(len(str(f)) for f in features)
    top = 24 + round(4.2 * longest)
    width = left + cell_w * len(columns) + 10 + round(5.8 * longest)
    height = top + cell_h * len(features) + 10
    parts = [
        (
            f'<svg class="heatmap" viewBox="0 0 {width} {height}" width="{width}" '
            'role="img" aria-label="Cohen\'s d of every feature against the control">'
        )
    ]
    for j, condition in enumerate(columns):
        x = left + j * cell_w + cell_w / 2
        parts.append(
            f'<text class="hm-col" x="{x}" y="{top - 8}" transform="rotate(-35 {x} {top - 8})">'
            f"{html.escape(str(condition))}</text>"
        )
    lookup = {(r.feature, r.condition): r for r in tests.itertuples()}
    for i, feature in enumerate(features):
        y = top + i * cell_h
        parts.append(
            f'<text class="hm-row" x="{left - 8}" y="{y + cell_h / 2 + 4}" text-anchor="end">'
            f"{html.escape(str(feature))}</text>"
        )
        for j, condition in enumerate(columns):
            row = lookup.get((feature, condition))
            d = float(row.d) if row is not None and row.d == row.d else float("nan")
            fill, ink = _colour(d)
            mark = ""
            if row is not None and bool(row.passes):
                mark = " ✓" if bool(row.confirmed) else " ●"
            shown = "–" if not np.isfinite(d) else f"{d:+.2f}"
            tip = f"{feature}, {condition}: d {shown}"
            if row is not None and np.isfinite(row.q_cells):
                tip += f"; cell-level q {p_value(row.q_cells)}"
            if row is not None and np.isfinite(row.q_paired):
                tip += f"; repeat means q {p_value(row.q_paired)}"
            x = left + j * cell_w
            parts.append(
                f'<g><title>{html.escape(tip)}</title><rect x="{x + 1}" y="{y + 1}" '
                f'width="{cell_w - 2}" height="{cell_h - 2}" rx="3" fill="{fill}"/>'
                f'<text x="{x + cell_w / 2}" y="{y + cell_h / 2 + 4}" text-anchor="middle" '
                f'fill="{ink}" class="hm-cell">{shown}{mark}</text></g>'
            )
    parts.append("</svg>")
    legend = (
        '<p class="note">Cohen\'s d against the control, '
        '<span class="hm-key" style="background:#2a78d6"></span> lower · '
        '<span class="hm-key" style="background:#e34948"></span> higher (colour full at '
        f"|d| ≥ {D_CAP:g}); ● a candidate (cell-level q ≤ 0.05), ✓ also confirmed across "
        "repeats (the paired t-test on the repeat means, q ≤ 0.05). Hover a cell for its "
        "q-values.</p>"
    )
    return f'<div class="table-wrap">{"".join(parts)}</div>{legend}'


__all__ = [
    "INTROS",
    "ORDER",
    "TITLES",
    "enough_reading",
    "heatmap",
    "hidden",
    "unnumbered",
    "what_checked",
]
