"""The design scan in sentences (private; ``scan.DesignScan.summary``)."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .._style import table, wrap
from .._wording import count, listing
from ._scan_rules import fields_line, repeat_lines, technical_line, unit_words

# Repeats shown as columns of the layout table; more are summarised per condition.
MAX_REPEAT_COLUMNS = 8
MAX_OPTIONS = 8


def _title(scan: Any) -> str:
    design = scan.design
    _, words = unit_words(scan)
    parts = [f"{design['observations']} {words} in {count(design['fields'], 'field')}"]
    if design["conditions"]:
        parts.append(count(len(design["conditions"]), "condition"))
    if design["repeats"]:
        parts.append(count(len(design["repeats"]), "biological repeat"))
    marker = scan.status.marker
    return f"{marker} Design scan — " + ", ".join(parts)


def _conditions_line(scan: Any) -> str | None:
    design = scan.design
    conditions = design["conditions"]
    if not conditions:
        return None
    control, guess = design["control"], design.get("control_guess")
    controls = design.get("controls") or []
    names = []
    for condition in conditions:
        if len(controls) > 1 and condition in controls:
            names.append(f"{condition} (a control)")
        elif control is not None and condition == control:
            names.append(f"{condition} (the control)")
        elif control is None and guess is not None and condition == guess:
            names.append(f"{condition} (named like a control; not declared)")
        else:
            names.append(str(condition))
    return f"Conditions: {listing(names, 12)}."


def _targets_line(scan: Any) -> str | None:
    targets = scan.design["targets"]
    if not targets:
        return None
    shared = {t: c for t, c in targets.items() if len(c) > 1}
    if not shared:
        return f"Targets: {count(len(targets), 'target')}, one condition each."
    named = [f"{t} ({listing(c)})" for t, c in shared.items()]
    return f"Targets with several conditions: {'; '.join(named)}."


def _features_line(scan: Any) -> str:
    settings = scan.settings
    features = settings["feature_cols"]
    if not features:
        return "Features: no numeric column found."
    names = listing(features, 3)
    if settings["features_declared"]:
        return f"Features: {count(len(features), 'feature')} declared ({names})."
    return (
        f"Features: {count(len(features), 'numeric column')} read as candidates, not declared "
        f"({names}); used here for the copy checks only."
    )


def _layout_lines(scan: Any) -> list[str]:
    layout = scan.layout
    if layout.empty:
        return []
    _, words = unit_words(scan)
    if list(layout.columns) == ["observations"]:
        rows = [(c, int(n)) for c, n in layout["observations"].items()]
        return [
            f"{words.capitalize()} per condition:",
            table(rows, ["condition", words], indent=4, align="lr"),
        ]
    if len(layout.columns) <= MAX_REPEAT_COLUMNS:
        columns = ["condition", *[str(r) for r in layout.columns]]
        rows = [
            [condition, *[None if pd.isna(n) else int(n) for n in counts]]
            for condition, counts in layout.iterrows()
        ]
        return [
            f"{words.capitalize()} per condition and biological repeat:",
            table(rows, columns, indent=4, align="l" + "r" * (len(columns) - 1)),
        ]
    rows = []
    for condition, counts in layout.iterrows():
        held = counts.dropna().astype(int)
        rows.append([condition, len(held), f"{held.min()} to {held.max()}", int(held.sum())])
    return [
        f"{words.capitalize()} per condition (over its biological repeats):",
        table(rows, ["condition", "repeats", "per repeat", "total"], indent=4, align="lrrr"),
    ]


def _design_lines(scan: Any) -> list[str]:
    lines = ["What the dataset is"]
    sentences = [_conditions_line(scan), *repeat_lines(scan), technical_line(scan)]
    sentences += [fields_line(scan), _targets_line(scan), _features_line(scan)]
    # The other feature tables, checked, before "Loaded correctly?" (owner, 2026-10-05).
    sentences += [f"Other feature table {line}" for line in scan.design.get("feature_tables", [])]
    for axis, (low, high) in scan.design["planes"].items():
        sentences.append(f"{axis!r}: {low} to {high} planes per field, one field each.")
    for sentence in sentences:
        if sentence is None:
            continue
        if sentence.startswith("  "):
            lines.append(wrap(sentence.strip(), 4, hang=6))
        else:
            lines.append(wrap(sentence, 2, hang=4))
    layout = _layout_lines(scan)
    if layout:
        lines.append(wrap(layout[0], 2))
        lines.append(layout[1])
    return lines


def _finding_lines(scan: Any) -> list[str]:
    lines = ["Loaded correctly?"]
    for finding in scan.findings:
        lines.append(wrap(f"{finding.status.marker} {finding.text}", 2, hang=5))
    return lines


def _suggestion_lines(scan: Any) -> list[str]:
    if not scan.suggestions:
        return []
    lines = ["Columns that would say more"]
    lines += [wrap(f"- {text}", 2, hang=4) for text in scan.suggestions]
    return lines


def _options(question: Any) -> str:
    options = list(question.options)
    if not options:
        return ""
    shown = listing(options, MAX_OPTIONS)
    return f" Options: {shown}."


def _question_lines(scan: Any) -> list[str]:
    lines = ["Before the analysis"]
    for number, question in enumerate(scan.questions, start=1):
        suggested = f" Suggested: {question.suggested}." if question.suggested else ""
        text = f"{number}. {question.text}{suggested}"
        if question.key != "goal":
            text += _options(question)
        lines.append(wrap(text, 2, hang=5))
        lines.append(wrap(question.why, 5))
    lines.append(
        wrap(
            "Then: setup = scan.setup(readout=..., goal=...) and report(scan.objects, "
            "scan.fovs, setup).",
            2,
        )
    )
    return lines


def scan_summary(scan: Any, detail: str, questions: bool = True) -> str:
    """The scan's text: the design, the checks, the columns, the questions (``"full"`` adds
    the data check and the copy checks as they print themselves)."""
    sections = [
        [_title(scan)],
        _design_lines(scan),
        _finding_lines(scan),
        _suggestion_lines(scan),
        _question_lines(scan) if questions else [],
    ]
    if detail == "full":
        sections.append([scan.data_check.summary(detail="full")])
        if scan.integrity is not None:
            sections.append([scan.integrity.summary(detail="full")])
    return "\n\n".join("\n".join(section) for section in sections if section)


__all__ = ["scan_summary"]
