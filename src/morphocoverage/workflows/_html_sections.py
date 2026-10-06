"""The HTML report's reading order and its plain-language pieces (private; ``html_report``).

- ``ORDER`` and ``TITLES``: the sections in the order a user acts on them, the data and the
  fields to check first, then the whole dataset, its populations and whether the conditions
  differ, then how far the sampling supports it (owner, 2026-10-02); titled in
  words, without the question numbers of the text report (whose 4, 5, 7 and 8 are the
  planning questions of ``optimise``).
- ``INTROS``: what each section is for, shown by the "?" beside its title.
- ``hidden``: a section whose answer is "not applicable" for this design is left out of the
  page (its reason stays in "What this dataset cannot tell me").
- ``what_checked``: what the run checked, in short lines, instead of the declaration's
  sentences (kept under "Full detail"); ``not_declared``: what it could not check.
- ``enough_reading``: "Did I image enough?" answered in plain sentences, with what to do.
- ``unnumbered``: the text report's question numbers replaced by these titles.
- ``heatmap``: Cohen's d of every feature and condition against the control, as an SVG, with
  the discovery's gates marked.
"""

from __future__ import annotations

import html
import re
from collections.abc import Sequence
from functools import lru_cache
from typing import Any

import numpy as np

from .._style import Status, rule_label
from .._wording import count, listing, p_value, rate, shortened
from ..retrospective.report import QUESTIONS, questions_for, short_questions_for
from ..retrospective.report_resolved import _needing, _table, _unassessed, between_wells
from ..retrospective.report_status import group_word
from ..sampling_unit.contrast_sampling import MORE_WELLS, PRACTICAL
from ..sampling_unit.repeat_plan_text import field_words
from ..schema import resolve_observation_unit
from ..strategies.base import SamplingStrategy
from ..sufficiency_short import rule_tallies
from ._contrasts import controls_sentence
from ._html_render import in_units

ORDER = (
    "overview",
    "inspect",
    "dataset",
    "populations",
    "conditions",
    "enough",
    "profile",
    "unit",
    "controls",
)
TITLES = {
    "overview": "The data",
    "inspect": "Fields to check (quality control)",
    "dataset": "The whole dataset at once",
    "conditions": "Do the conditions differ?",
    "populations": "Are there populations or groups of cells?",
    "enough": "Did I image enough?",
    "profile": "What limits the sampling?",
    "unit": "Why count fields, and not only cells?",
    "controls": "Can these features detect what I care about?",
}
INTROS = {
    "overview": "What the tables hold, and whether they read cleanly (missing values, copies).",
    "inspect": (
        "Fields that move a summary strongly, hold an unusual count or a state no other field "
        "shows, or that the copy checks name. Each may be an "
        "artefact (out of focus, debris, a field exported twice) or real biology: open their "
        "images before trusting the answers below. Focus and segmentation are not checked, so "
        "a field not flagged is not thereby a good image."
    ),
    "dataset": (
        "Every cell on all the features at once, as a PCA and a UMAP: where the conditions "
        "and the populations sit, and which features set a group of cells apart. They "
        "describe, they do not test."
    ),
    "conditions": (
        "Each condition against the control: the features significant across repeats first, "
        "then every feature at once, the best-ranked plotted, then the tests behind them."
    ),
    "populations": (
        "Groups of cells that sit together on all the features, from whichever condition, "
        "and whether a condition holds more or fewer of its cells in one than the control."
    ),
    # Said here once rather than in every feature's tab (``enough_intro``).
    "enough": "",
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


def enough_intro(
    single_repeat: bool = False, group: str = "group", listed: bool = True, compared: bool = True
) -> str:
    """The "?" of "Did I image enough?": what a comparison is, that the answer is per repeat
    (not with one biological repeat, where there is nothing across repeats to answer), what
    each group's own mean is for, and, when a tab lists only the comparisons not yet shown
    (``listed``), where every one is. Without a comparison with the control (``compared``),
    what the stopping rules answer."""
    if not compared:
        return (
            f"Whether the fields imaged are enough to pin down each {group}'s own mean, and "
            "whether one more field would show something new: the stopping rules, as optimise "
            "and live run them."
        )
    where = "the biological repeat" if single_repeat else "each biological repeat"
    one = "against the control" + ("" if single_repeat else " in one repeat")
    parts = [
        (
            f"Whether the fields imaged are enough to show each condition's difference from "
            f"the control within {where}, a comparison being one condition {one}, and to pin "
            f"down each {group}'s own mean."
        )
    ]
    if not single_repeat:
        parts.append(
            "This is per repeat: whether the conditions differ across the repeats is answered "
            f"in \u201c{TITLES['conditions']}\u201d, which does not need each repeat to show it "
            "alone."
        )
    parts.append(
        f"Each {group}'s own mean is a different goal from the comparison: it matters when the "
        f"value of a mean is reported, not for comparing conditions; more fields per {group} "
        "is the lever."
    )
    if listed:
        parts.append(
            "A tab with many comparisons lists those not yet shown; every one is in the full "
            "report."
        )
    return " ".join(parts)


INTROS["enough"] = enough_intro()


def intros_for(
    units: str,
    one: str,
    *,
    single_repeat: bool = False,
    group: str = "group",
    listed: bool = True,
    compared: bool = True,
) -> dict[str, str]:
    """``INTROS`` in the report's own words for its observations (tracks, patches), the "?"
    of "Did I image enough?" for its design (``enough_intro``)."""
    intros = {**INTROS, "enough": enough_intro(single_repeat, group, listed, compared)}
    return {key: in_units(text, units, one) for key, text in intros.items()}


def titles_for(units: str, one: str) -> dict[str, str]:
    """``TITLES`` in the words of the report's observations."""
    return {key: in_units(title, units, one) for key, title in TITLES.items()}


# The heatmap's diverging colours (lower than the control, higher), and its neutral middle.
LOWER, HIGHER, MIDDLE = (42, 120, 214), (227, 73, 72), (240, 239, 236)
D_CAP = 2.0


def hidden(status: Status | None, headline: str) -> bool:
    """A section that only says this design cannot answer it."""
    return status is Status.NOT_ASSESSED and headline.lower().startswith("not applicable")


def _rules(setup: Any, undefined: Sequence[str] = ()) -> str:
    """The stopping rules in words; a rule that is not a ``SamplingStrategy`` (it has no
    policies to describe) by its name, and any error of a strategy's label raises."""
    unit = resolve_observation_unit(setup.observation_unit)[0]
    rules = [
        (rule_label(rule, unit) or rule.name)
        if isinstance(rule, SamplingStrategy)
        else str(getattr(rule, "name", type(rule).__name__))
        for rule in setup.stopping_rules()
    ]
    return listing([r for r in rules if r not in undefined], 4)


def what_checked(setup: Any, undefined: Sequence[str] = ()) -> list[str]:
    """What the run checked, one short line each (the declaration in words); a stopping rule
    in ``undefined`` (its labels) did not apply to these observations and is said apart."""
    unit = setup.unit_words
    lines = [
        f"Readout: {setup.readout}; its mean over all {unit} is the summary compared."
        if setup.readout is not None
        else "Readout: none; the other feature tables are compared whole (no explainable "
        "features, so the mean-precision rules are not applicable)."
    ]
    features = tuple(setup.columns)
    if len(features) > 1:
        lines.append(f"Features: {count(len(features), 'feature')} ({listing(list(features), 6)}).")
    control = (setup.single_control or {}).get("condition")
    several = controls_sentence(setup)
    if several:  # several controls, each with its conditions (owner, 2026-10-05)
        lines.append(f"{several}.")
    elif control is not None:
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
    if setup.mean_target:
        lines.append(
            f"Precision asked: the mean of {setup.readout} within {tolerances} of its value "
            "(judged on held-back fields by optimise; the stopping rules stop when its standard "
            "error is a set share of the mean)."
        )
    if setup.fingerprint:
        named = "; ".join(c.description or c.name for c in setup.fingerprint)
        lines.append(f"Fingerprint: {named}.")
    if several and setup.normalize:
        lines.append(
            "Values: each condition divided by its own control's mean in its repeat for the "
            "figures; levels across experiments are read on the raw values."
        )
    elif setup.readout is not None:
        lines.append(
            "Values: divided by each repeat's control mean."
            if setup.normalize
            else "Values: as measured (the tests between conditions also use them normalised)."
        )
    rules = _rules(setup, undefined)
    if rules:
        lines.append(f"Stopping rules checked: {rules}.")
    if undefined:
        lines.append(
            f"Not applicable to {unit}: {listing(list(undefined), 3)} (it counts objects such "
            "as cells), so it was not used."
        )
    missing = not_declared(setup)
    if missing:
        lines.append(f"Not declared, so not checked: {listing(missing, 6)}.")
    return lines


def not_declared(setup: Any) -> list[str]:
    """What the setup does not declare, so the report does not check."""
    missing = []
    if setup.positive is None:
        missing.append("a positive control")
    if not setup.fingerprint:
        missing.append("a phenotypic fingerprint")
    if setup.rare is None:
        missing.append("a rare population")
    if setup.acquisition_order_col is None and setup.stage_cols is None:
        missing.append("the acquisition order and stage positions")
    return missing


def _resolution(contrast: Any, single_repeat: bool = False) -> list[str]:
    """Whether the fields acquired show each comparison, and what more would show the others;
    what a comparison is and that it is per repeat are said once, in the card's "?"."""
    rows = _table(contrast)
    if rows.empty:
        return []
    _, fields = field_words(contrast)
    resolved = int(rows["resolved"].sum())
    small = int(rows["too_small"].sum())
    needing = _needing(rows)
    verdict = "Yes" if resolved == len(rows) else ("Partly" if resolved else "Not yet")
    where = "the biological repeat" if single_repeat else "each biological repeat"
    lines = [
        (
            f"Enough {fields} to compare with the control? {verdict}: within {where}, the "
            f"{fields} imaged show the difference from the control in {resolved} of "
            f"{count(len(rows), 'comparison')}."
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
            f"times the {fields} would not show it within one repeat; that is not evidence of "
            "no effect."
        )
    wells = int(between_wells(rows).sum())
    if wells:
        lines.append(
            f"In {count(wells, 'comparison')} more {fields} will not resolve it: {MORE_WELLS}."
        )
    if _unassessed(rows):
        lines.append(
            f"In {count(_unassessed(rows), 'comparison')} there are too few {fields} for an "
            "interval: not assessed."
        )
    return lines


def _precision(sufficiency: Any, readout: str, fields: str) -> list[str]:
    tallies = rule_tallies(sufficiency) if sufficiency else []
    data = [t for t in tallies if t["data_rule"]]
    if not data:
        return []
    word = group_word(sufficiency)
    # A rule undefined everywhere (one per object, on tracks) says nothing about precision.
    defined = [t for t in data if t["undefined"] < t["groups"]]
    lines = []
    novel = [t for t in tallies if t["label"].startswith("nothing new")]
    if defined:
        first = defined[0]
        met, groups = first["met"], first["groups"]
        line = (
            f"Enough to pin down each {word}'s own mean of {readout}? The {first['label']} "
            f"was met in {met} of {count(groups, word)}."
        )
        if met < groups:
            line += (
                " A different goal from the comparison: it matters when the value of a mean is "
                f"reported, not for comparing conditions; more {fields} per {word} is the lever."
            )
        lines.append(line)
    if novel:
        t = novel[0]
        lines.append(
            f"Enough to have seen every phenotype? '{t['label']}' was met in {t['met']} of "
            f"{count(t['groups'], word)} (the numbers below)."
        )
    return lines


def enough_reading(result: Any) -> list[str]:
    """ "Did I image enough?" in plain sentences, one question each: enough to compare with
    the control, to pin down each well's mean, to have seen every phenotype; each with its
    lever."""
    results = getattr(result, "results", {}) or {}
    contrast = results.get("contrast")
    lines = _resolution(contrast)
    fields = field_words(contrast)[1] if contrast is not None else "fields"
    return lines + _precision(results.get("sufficiency"), result.setup.readout, fields)


@lru_cache(maxsize=8)
def _numbered(units: str = "cells", one: str = "cell") -> dict[str, str]:
    """Each numbered title of the text report, to its title here (unnumbered), in the words
    of the report's observations."""
    names = {}
    titles = titles_for(units, one)
    for key, title in [*questions_for(units), *short_questions_for(units).items()]:
        plain = re.sub(r"^(\d\.|Foundation[.:])\s*", "", title)
        names[title] = titles.get(key, plain[:1].upper() + plain[1:])
    return names


@lru_cache(maxsize=8)
def _title_pattern(units: str, one: str) -> re.Pattern:
    names = _numbered(units, one)
    return re.compile("|".join(re.escape(t) for t in sorted(names, key=len, reverse=True)))


_NUMBER_KEYS = {title.split(".")[0]: key for key, title in QUESTIONS if title[:1].isdigit()}
_QUESTION = re.compile(r"\b[Qq]uestion (\d)\b")
_FOUNDATION = re.compile(r"\b(?:the )?Foundation\b")


def unnumbered(markup: str, units: str = "cells", one: str = "cell") -> str:
    """``markup`` with the text report's numbered titles and "question N" given the titles of
    this page, which has no numbers, in the words of the report's observations (``units``,
    ``one``)."""
    names, titles = _numbered(units, one), titles_for(units, one)
    markup = _title_pattern(units, one).sub(lambda m: html.escape(names[m.group(0)]), markup)
    questions = dict(questions_for(units))

    def title(match: re.Match) -> str:
        key = _NUMBER_KEYS.get(match.group(1))
        name = titles.get(key) or names.get(questions.get(key, ""), "")
        return f"\u201c{html.escape(name)}\u201d" if name else match.group(0)

    unit = titles["unit"]
    markup = _FOUNDATION.sub(f"\u201c{html.escape(unit)}\u201d", markup)
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


# The longest name the heatmap writes in full; the cell's hover and the CSV hold it whole.
HEATMAP_NAME = 30


def _label(x: float, y: float, name: Any, css: str, extra: str = "") -> str:
    """A heatmap label, cut to ``HEATMAP_NAME`` characters, the full name on hover."""
    full = html.escape(str(name))
    return (
        f'<g><title>{full}</title><text class="{css}" x="{x}" y="{y}"{extra}>'
        f"{html.escape(shortened(name, HEATMAP_NAME))}</text></g>"
    )


def heatmap(discovery: Any, conditions: list[Any], one: str = "cell") -> str:
    """Cohen's d of every feature (best-ranked first) and condition against the control, as
    an SVG at a readable size that scrolls sideways rather than shrinks: the features are the
    rows, or the conditions when there are more of them than features (owner, 2026-10-05);
    ● a candidate of the screen on every observation (``one``: the observation unit's word),
    ✓ also confirmed across repeats. Empty without tests."""
    if discovery is None or discovery.tests is None or not len(discovery.tests):
        return ""
    tests = discovery.tests
    features = list(discovery.ranking["feature"])
    present = set(tests["condition"])
    columns = [c for c in conditions if c in present] or sorted(present, key=str)
    named = tests["control_condition"].dropna() if "control_condition" in tests else []
    controls = sorted({str(c) for c in named})
    against = controls[0] if len(controls) == 1 else "its control"
    condition_title = f"condition, against {html.escape(against)}"
    feature_title = "feature, best-ranked first"
    transposed = len(columns) > len(features)
    rows, cols = (columns, features) if transposed else (features, columns)
    row_title, col_title = (
        (condition_title, feature_title) if transposed else (feature_title, condition_title)
    )
    # The column labels slant up to the right: room above and beside for the longest, and
    # for each axis's title (the rows' turned up its left side).
    longest = max(len(shortened(c, HEATMAP_NAME)) for c in cols)
    title = 22
    cell_w, cell_h = 72, 24
    left = title + 10 + 7 * max(len(shortened(r, HEATMAP_NAME)) for r in rows)
    top = title + 24 + round(4.2 * longest)
    width = left + cell_w * len(cols) + 10 + round(5.8 * longest)
    height = top + cell_h * len(rows) + 10
    middle_x = left + cell_w * len(cols) / 2
    middle_y = top + cell_h * len(rows) / 2
    parts = [
        (
            f'<svg class="heatmap" viewBox="0 0 {width} {height}" width="{width}" '
            f'height="{height}" role="img" '
            'aria-label="Cohen\'s d of every feature against the control">'
        ),
        (f'<text class="hm-axis" x="{middle_x}" y="16" text-anchor="middle">{col_title}</text>'),
        (
            f'<text class="hm-axis" x="16" y="{middle_y}" text-anchor="middle" '
            f'transform="rotate(-90 16 {middle_y})">{row_title}</text>'
        ),
    ]
    for j, name in enumerate(cols):
        x = left + j * cell_w + cell_w / 2
        parts.append(_label(x, top - 8, name, "hm-col", f' transform="rotate(-35 {x} {top - 8})"'))
    lookup = {(r.feature, r.condition): r for r in tests.itertuples()}
    for i, name in enumerate(rows):
        y = top + i * cell_h
        parts.append(_label(left - 8, y + cell_h / 2 + 4, name, "hm-row", ' text-anchor="end"'))
        for j, other in enumerate(cols):
            feature, condition = (other, name) if transposed else (name, other)
            row = lookup.get((feature, condition))
            d = float(row.d) if row is not None and row.d == row.d else float("nan")
            fill, ink = _colour(d)
            mark = ""
            if row is not None and bool(row.passes):
                mark = " ✓" if bool(row.confirmed) else " ●"
            shown = "–" if not np.isfinite(d) else f"{d:+.2f}"
            tip = f"{feature}, {condition}: d {shown}"
            if row is not None and np.isfinite(row.q_cells):
                tip += f"; {one}-level q {p_value(row.q_cells)}"
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
    # The key first: which test each value and mark comes from (owner, 2026-10-02).
    legend = (
        '<p class="note hm-legend">Each square: Cohen\'s d of the condition against the '
        f"control on every {one}, as the {one}-level screen computes it "
        '(<span class="hm-key" style="background:#2a78d6"></span> lower · '
        '<span class="hm-key" style="background:#e34948"></span> higher, colour full at '
        f"|d| ≥ {D_CAP:g}). ● a candidate: the {one}-level screen's randomization test, "
        "q ≤ 0.05, exploratory. ✓ also significant across repeats: the paired t-test on "
        "the repeat means, q ≤ 0.05. Hover a square for its q-values.</p>"
    )
    # Scrolls sideways at its own size instead of shrinking to the page (owner, 2026-10-05).
    return f'{legend}<div class="heatmap-scroll">{"".join(parts)}</div>'


__all__ = [
    "INTROS",
    "ORDER",
    "TITLES",
    "enough_intro",
    "enough_reading",
    "heatmap",
    "hidden",
    "intros_for",
    "not_declared",
    "titles_for",
    "unnumbered",
    "what_checked",
]
