"""Questions 6, 8 and 9 of the microscopist's report in their short form, and its closing.

- ``short_section``: a question's short parts (``(lines, cautions)`` of each
  result: ``inspection_text.inspection_parts``, ``order_text.order_parts``,
  ``contrast_text.contrast_parts``, ``FeatureRanking.short_parts``) as one
  section, the caveats of all its parts pooled and stated once at its end;
- ``conditions_sections``: question 9's full parts and short section: the contrast, the
  features the discovery ranks, and the ranking over whole technical repeats only when it
  could test something (with the discovery, its "not ranked" would repeat it);
- ``limits_text``: "What this dataset cannot tell me". One line per limit, in the short
  and the full forms alike: each question not answered and each question answered in part
  points to its section, where the reason is stated, once (owner, 2026-10-06); then the
  other notes, each once, marked by what they are: ❌ a weak representation or fields the
  integrity checks name (every answer counts them as independent), ⚠️ a caution about
  these data (a missing control, no stage coordinates), the standing assumptions under
  "Assumptions:", and the run's settings as plain lines. ``pointers=False`` gives the
  reasons instead, for the page, which leaves out the sections it cannot answer.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from .._caveats import caveat_lines, once
from .._style import NOT_ASSESSED, PROBLEM, Status, check_detail, headline, wrap

ASSESSED = (
    "Every question could be answered; the assumptions each answer rests on are listed with it."
)
# Notes that change how every answer is read: a weak representation, fields that the
# integrity checks name.
_PROBLEMS = ("Weak representation", "Data integrity")


def conditions_sections(contrast: Any, ranking: Any, discovery: Any) -> tuple[list[str], str]:
    """Question 9's full parts and its short section (empty without any result); a control
    of few fields shared by conditions significant the same way named once (``⚠️``)."""
    from ..sampling_unit.contrast_text import contrast_parts
    from ..sampling_unit.reading_text import shared_control_line

    shuffled = ranking is not None and (
        discovery is None or bool(len(ranking.ranking) and ranking.ranking["tests"].any())
    )
    shown = [r for r in (contrast, discovery, ranking if shuffled else None) if r is not None]
    full = [result.summary(detail="full") for result in shown]
    if not shown:
        return full, ""
    answers = [contrast_parts(contrast, indent=2)] if contrast is not None else []
    answers += [r.short_parts(indent=2) for r in shown if r is not contrast]
    shared = shared_control_line(contrast, discovery) if contrast is not None else None
    full += caveat_lines([shared], indent=2) if shared else []
    return full, short_section(answers, *([shared] if shared else []))


def short_section(parts: Sequence[tuple[list[str], list[str]]], *cautions: str) -> str:
    """The lines of every part, then every caveat of the parts (and ``cautions``) once."""
    lines: list[str] = []
    pooled: list[str] = []
    for own, own_cautions in parts:
        lines += own
        pooled += own_cautions
    return "\n".join([*lines, *caveat_lines([*pooled, *cautions], indent=2)])


def _notes(notes: Sequence[str], detail: str = "short") -> list[str]:
    """The notes, each once: problems (❌) first, then the cautions about these data (⚠️)
    and the standing assumptions ("Assumptions:", ``_caveats``), then the settings."""
    notes = once(notes)
    problems = [n for n in notes if n.startswith(_PROBLEMS)]
    settings = [n for n in notes if n.startswith("Settings:")]
    limits = [n for n in notes if n not in problems and n not in settings]
    lines = [wrap(f"{PROBLEM} {note}", 2, hang=5) for note in problems]
    lines += caveat_lines(limits, indent=2, detail=detail)
    lines += [wrap(note, 2, hang=4) for note in settings]
    return lines


def _where(title: str) -> str:
    """``question 3``, or ``the Foundation``: where a limit's reason is stated."""
    number = title.split(".", 1)[0]
    return f"question {number}" if number.isdigit() else f"the {title.split(':', 1)[0]}"


def _pointer(title: str, headline_text: str) -> str:
    """``not assessed, see question 1``: the state of an answer, its reason left to its
    section (owner, 2026-10-06)."""
    state = re.split(r"[:;]", headline_text, maxsplit=1)[0].strip() or "not assessed"
    return f"{state}, see {_where(title)}"


def _pointed(notes: Sequence[str], titles: Sequence[tuple[str, str]]) -> list[str]:
    """The notes, a reason added to a question answered in part replaced by a pointer to its
    section, where the reason is (``workflows._document.with_reasons``)."""
    named = [title for _, title in titles]
    out = []
    for note in notes:
        title = next((t for t in named if note.startswith(f"{t} ")), None)
        out.append(note if title is None else f"{title} Answered in part: see {_where(title)}.")
    return out


def limits_text(
    report: Any,
    questions: Sequence[str] | None,
    detail: str,
    *,
    titles: Sequence[tuple[str, str]],
    needs: Mapping[str, str],
    short_titles: Mapping[str, str],
    pointers: bool = True,
) -> str:
    """The closing section of ``report`` (see the module), among ``questions`` or all."""
    full = check_detail(detail) == "full"
    chosen = [(key, title) for key, title in titles if questions is None or key in questions]
    missing = [
        (k, t)
        for k, t in chosen
        if report.inputs.get(k) == "not assessed" or report.statuses.get(k) is Status.NOT_ASSESSED
    ]
    lines = []
    for key, title in missing:
        answer = report.headlines.get(key, "")
        if pointers:  # one line per limit, its reason in its section (owner, 2026-10-06)
            line = (
                f"{NOT_ASSESSED} {title}: {_pointer(title, answer)}."
                if full
                else headline(Status.NOT_ASSESSED, short_titles[key], _pointer(title, answer))
            )
        elif full and answer and not answer.startswith("not assessed"):
            # Its own reason, never what to run (audit, 2026-10-02; audit 10, C13).
            line = f"{NOT_ASSESSED} {title}: {answer}."
        elif full and report.inputs.get(key) == "not assessed":
            line = f"{NOT_ASSESSED} {title}: not assessed; needs: {needs[key]}."
        elif full:
            # Run, but no conclusion possible (one repeat): its own answer says why.
            line = f"{NOT_ASSESSED} {title}: {report.headlines.get(key, '')}."
        else:
            answer = report.headlines.get(key) or f"not assessed; needs: {needs[key]}"
            line = headline(Status.NOT_ASSESSED, short_titles[key], answer)
        lines.append(wrap(line, 2, hang=5))
    if not missing:
        lines.append(wrap(ASSESSED, 2, hang=4))
    # A reason added to an answer given in part is printed in its section; here a pointer to
    # it, in the short and the full forms alike (owner, 2026-10-06).
    notes = _pointed(report.notes, titles) if pointers else list(report.notes)
    return "\n".join([*lines, *_notes(notes, detail)])


__all__ = [
    "ASSESSED",
    "conditions_sections",
    "limits_text",
    "short_section",
]
