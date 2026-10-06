"""The analyses the benchmark has not validated yet, marked exploratory (private; ``assemble``,
the page).

Plan v2 (owner, 2026-10-06): the confirmatory simulations cover the core first (modules P0 to
P5 of ``docs/common/validation.md``); the populations and their enrichment, beyond the mean,
the covariance fingerprint, the rare target and novelty stay exploratory in the report, on
the page and in the docs until their module validates them. Each section that answers with
one of them says so once, in one sentence that points to the validation protocol.
"""

from __future__ import annotations

from dataclasses import replace
from html import escape
from typing import TYPE_CHECKING, Any

from .._style import wrap
from .._wording import listing

if TYPE_CHECKING:
    from ..retrospective import MicroscopistReport
    from .setup import Setup

VALIDATION = "docs/common/validation.md"


def exploratory_line(names: list[str]) -> str:
    """``The covariance fingerprint and novelty are exploratory: not yet validated by the
    benchmark of docs/common/validation.md.``"""
    verb = "is" if len(names) == 1 else "are"
    text = f"{listing(names, len(names))} {verb} exploratory: not yet validated by the "
    return f"{text[:1].upper()}{text[1:]}benchmark of {VALIDATION}."


def exploratory_names(results: dict[str, Any], setup: Setup) -> dict[str, list[str]]:
    """By report section, the exploratory analyses that answer in it."""
    enough = []
    precision = results.get("fingerprint_precision")
    if precision is not None and len(precision):
        nouns = set(precision["noun"]) if "noun" in precision else set()
        covariance = nouns == {"covariance fingerprint"}
        enough.append("the covariance fingerprint" if covariance else "the fingerprint")
    if setup.rare is not None and results.get("rare_clustering") is not None:
        enough.append("the rare target")
    next_fov = getattr(results.get("sufficiency"), "next_fov", None)
    if setup.novelty is not None and next_fov is not None and len(next_fov):
        enough.append("novelty")
    names = {"enough": enough}
    controls = results.get("controls")
    if setup.positive is not None and getattr(controls, "status", "not assessed") != "not assessed":
        names["controls"] = ["the positive control's check, on the definition of new"]
    if results.get("distribution") is not None:
        names["conditions"] = ["beyond the mean"]
    if results.get("populations") is not None:
        names["populations"] = ["population enrichment"]
    return {key: value for key, value in names.items() if value}


def exploratory_html(result: Any, key: str) -> str:
    """The exploratory line of section ``key`` as a paragraph of the page, for the cards that
    do not print the section's text (the per-feature "Did I image enough?", beyond the mean);
    empty when nothing there is exploratory."""
    names = exploratory_names(getattr(result, "results", {}) or {}, result.setup).get(key)
    return f'<p class="note">{escape(exploratory_line(names))}</p>' if names else ""


def with_exploratory(
    document: MicroscopistReport, results: dict[str, Any], setup: Setup
) -> MicroscopistReport:
    """The report with one exploratory line at the end of each section answered by an analysis
    not yet validated (see the module), in the short and the full forms."""
    short, sections = dict(document.short), dict(document.sections)
    headlines = dict(document.headlines)
    for key, names in exploratory_names(results, setup).items():
        if key not in sections or document.inputs.get(key) == "not assessed":
            continue
        line = wrap(exploratory_line(names), 2, hang=4)
        sections[key] = f"{sections[key]}\n\n{line}"
        if key in short:
            short[key] = f"{short[key]}\n\n{line}"
        headlines[key] = _tagged(key, headlines.get(key, ""))
    return replace(document, short=short, sections=sections, headlines=headlines)


def _tagged(key: str, headline: str) -> str:
    """The glance line with its exploratory part marked (audit 10, C27): the positive
    control's whole answer, the "beyond the mean" part of question 9."""
    if key == "controls" and headline and "exploratory" not in headline:
        return f"{headline} (exploratory)"
    part = "beyond the mean"
    if key == "conditions" and part in headline and f"{part} (exploratory)" not in headline:
        return headline.replace(part, f"{part} (exploratory)", 1)
    return headline


__all__ = [
    "VALIDATION",
    "exploratory_html",
    "exploratory_line",
    "exploratory_names",
    "with_exploratory",
]
