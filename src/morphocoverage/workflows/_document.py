"""The assembled report of the retrospective workflows (private): parts, reasons, settings.

- ``assemble`` builds the microscopist's report from the results under the
  setup's declaration and presets (``Setup.minimum_detection``), and adds
  what ``microscopist_report`` has no argument for: the declared rare
  phenotype's clustering, the positive control's pairing and the unusual
  observations' pooled reference;
- ``with_reasons`` adds, by question, why an analysis was not assessed
  (``pooling_reason``, ``protocol_reason``, ``no_hierarchy_reasons``);
- ``report_settings`` and ``plan_settings`` state the computational settings
  and presets of a run in one sentence, which ``with_settings`` keeps among
  the closing notes of a merged report.

Shared-private: every name in ``__all__`` is used by ``workflows.retrospective``
(``pooling_reason`` also by ``workflows._shared``), not part of the public API.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import pandas as pd

from .._wording import count, fields, listing, percent
from ..metadata import REPEAT_COLUMN
from ..retrospective import MicroscopistReport, SamplingHierarchy, microscopist_report
from ..retrospective.report import QUESTIONS
from ..retrospective.report_profile import sampling_profile

if TYPE_CHECKING:
    from .setup import Setup

# Results of the report that ``microscopist_report`` does not take; ``assemble`` adds them.
_EXTRA_RESULTS = ("rare_clustering", "integrity")


def pooling_reason(analyses: str, setup: Setup, pooled: Sequence[str]) -> str:
    """Why ``analyses`` were not run: the analysis groups pool the columns they need apart."""
    named = ", ".join(repr(c) for c in pooled)
    return (
        f"{analyses} not assessed: group_cols {tuple(setup.group_cols or ())} pool {named} "
        f"(an analysis group holds several of its values), and they compare within one "
        f"{' and one '.join(pooled)}; add {named} to group_cols to assess them"
    )


def with_reasons(document: MicroscopistReport, reasons: Mapping[str, str]) -> MicroscopistReport:
    """The report with each question's reason for what it could not assess.

    A question with nothing assessed shows the reason instead of the generic
    need; one assessed in part keeps its answers, adds the reason, and the
    closing section lists it among the notes.
    """
    if not reasons:
        return document
    titles = dict(QUESTIONS)
    sections, notes = dict(document.sections), list(document.notes)
    for key, reason in reasons.items():
        sentence = f"{reason[:1].upper()}{reason[1:]}."
        if document.inputs.get(key) == "not assessed":
            sections[key] = f"  {sentence}"
        else:
            sections[key] = f"{sections[key]}\n\n  {sentence}"
            notes.append(f"{titles[key]} {sentence}")
    return replace(document, sections=sections, notes=tuple(notes))


def assemble(results: Mapping[str, Any], setup: Setup) -> MicroscopistReport:
    """The microscopist's report of ``results`` under the setup's declaration and presets.

    ``microscopist_report`` with ``Setup.minimum_detection``, then what it has
    no argument for: the declared rare phenotype's clustering in the sampling
    profile, how the positive control paired its fields, whether the
    unusual observations' reference pools repeats, and the fields the
    data-integrity checks name (in the overview, among the fields to look at
    and in the limits).
    """
    document = microscopist_report(
        **{k: v for k, v in results.items() if k not in _EXTRA_RESULTS},
        declaration=setup.declaration(),
        minimum_detection=setup.minimum_detection,
    )
    integrity = results.get("integrity")
    if integrity is not None:
        # What the tables hold includes structure to inspect before reading the answers.
        overview = document.overview or ""
        document = replace(
            document, overview="\n\n".join(p for p in (overview, integrity.summary()) if p)
        )
        document = _with_integrity(document, integrity)
    rare = results.get("rare_clustering")
    if rare is not None:
        rows = sampling_profile(None, None, None, None, clustering=rare)
        lines = "\n".join(f"{r.statement} [{r.where}; {r.source}]" for r in rows.itertuples())
        text = "\n\n".join(part for part in (lines, rare.summary()) if part)
        document = with_text(document, "profile", text, rows)
    controls = results.get("controls")
    if controls is not None and controls.status != "not assessed" and not controls.pair_cols:
        document = with_text(
            document,
            "controls",
            "Positive and reference fields were paired across the whole table (pair_cols=()): "
            "every repeat is pooled, so a shift between repeats can read as detection.",
        )
    pooled = _reference_repeats(results.get("unusual"))
    if pooled > 1:
        units = setup.unit_words
        document = with_text(
            document,
            "profile",
            f"The reference of the unusual {units} pools the control {units} of "
            f"{count(pooled, 'repeat')}: a shift shared by one repeat (a staining or plating "
            f"batch) makes its {units} read as unusual, so compare the unusual share between "
            "repeats before reading it as biology.",
        )
    return document


def _with_integrity(document: MicroscopistReport, integrity: Any) -> MicroscopistReport:
    """The report with the fields the integrity checks name among the fields to look at (6)
    and in the limits: every answer counts them as independent fields."""
    involved = integrity.fields_involved()
    if not involved:
        return document
    named = (
        f"{count(len(involved), 'field')} in {count(len(integrity.findings), 'finding')} "
        "(fields that hold another field's observations, or their own twice)"
    )
    sentence = (
        f"The data-integrity checks (in the overview) name {named}; every answer here counts "
        f"them as independent fields, so look at them first: {listing(involved, limit=5)}."
    )
    sections = dict(document.sections)
    sections["inspect"] = f"{sections['inspect']}\n\n  {sentence}"
    note = (
        f"Data integrity: the checks in the overview name {named}, and every answer counts "
        "them as independent fields."
    )
    return replace(document, sections=sections, notes=(*document.notes, note))


def _reference_repeats(unusual: Any) -> int:
    """How many repeats the unusual observations' reference groups come from (0: none)."""
    if unusual is None or "reference_groups" not in unusual.groups.columns:
        return 0
    repeats = {
        str(group[REPEAT_COLUMN])
        for groups in unusual.groups["reference_groups"]
        if isinstance(groups, list | tuple)
        for group in groups
        if isinstance(group, Mapping) and REPEAT_COLUMN in group
    }
    return len(repeats)


def with_text(
    document: MicroscopistReport, key: str, text: str, rows: pd.DataFrame | None = None
) -> MicroscopistReport:
    """The report with ``text`` added to question ``key`` (and ``rows`` to its profile).

    A question not assessed so far is assessed by it: the text replaces the
    generic need.
    """
    indented = "\n".join(f"  {line}" if line else line for line in text.splitlines())
    sections, inputs = dict(document.sections), dict(document.inputs)
    if inputs.get(key) == "not assessed":
        sections[key], inputs[key] = indented, "given"
    else:
        sections[key] = f"{sections[key]}\n\n{indented}"
    profile = document.profile
    if rows is not None and len(rows):
        profile = pd.concat([profile, rows], ignore_index=True) if len(profile) else rows
    return replace(document, sections=sections, inputs=inputs, profile=profile)


def with_settings(document: MicroscopistReport, sentences: Sequence[str]) -> MicroscopistReport:
    """The report with each part's settings sentence among its closing notes."""
    return replace(document, notes=(*document.notes, *sentences))


def _orders_text(recorded: bool, permuted: int, dispersed: int | None = None) -> str:
    """``the recorded order, 10 random and 5 dispersed orders``, ``the recorded order and 5
    random orders``."""
    if dispersed is None:
        kinds = f"{permuted} random orders"
        return f"the recorded order and {kinds}" if recorded else kinds
    kinds = f"{permuted} random and {dispersed} dispersed orders"
    return f"the recorded order, {kinds}" if recorded else kinds


def held_back_text(settings: Mapping[str, Any]) -> str:
    """The fields each group actually held back, and the ``reservoir_fraction`` asked."""
    asked = f"reservoir_fraction {percent(settings['reservoir_fraction'])}"
    sizes = settings.get("reservoir_fovs")
    if not sizes:
        return f"no field held back ({asked}: no group had enough fields)"
    low, high = sizes
    held = fields(low) if low == high else f"{low}-{high} fields"
    return f"{held} of each group held back ({asked}, rounded up to whole fields)"


def report_settings(settings: Mapping[str, Any]) -> str:
    """The report's replay settings in one sentence."""
    orders = _orders_text(settings["recorded_order"], settings["permuted_orders"])
    return (
        f"Settings: 'Did I image enough?' replays each rule in {orders} (seeded from "
        f"random_state); random_state {settings['random_state']}."
    )


def plan_settings(settings: Mapping[str, Any]) -> str:
    """The run's settings in one sentence, the provisional preset first."""
    orders = _orders_text(
        settings.get("recorded_order", False),
        settings["permuted_orders"],
        settings["dispersed_orders"],
    )
    return (
        f"Settings: a sample counts as adequately represented at k fields when "
        f"{percent(settings['protocol_threshold'])} of random acquisitions of k fields "
        f"preserved the criterion (protocol_threshold, a provisional preset); "
        f"{held_back_text(settings)}, {settings['n_splits']} hold-outs, "
        f"{settings['draws']} draws per field count, {orders}; "
        f"random_state {settings['random_state']}."
    )


def protocol_reason(
    setup: Setup, hierarchy: SamplingHierarchy, strata: Sequence[str], groups: Sequence[str]
) -> str | None:
    """Why protocol reliability cannot run on these analysis groups, or ``None``."""
    levels = [c for c in hierarchy.unit_key(hierarchy.sample) if c not in groups]
    apart = [c for c in strata if c not in groups]
    if not levels and not apart:
        return None
    named = ", ".join(repr(c) for c in [*levels, *apart])
    reason = (
        f"protocol reliability across samples is not assessed: group_cols {tuple(groups)} pool "
        f"{named}, so the reliability has no result per sample ({hierarchy.sample}"
    )
    reason += f") and per stratum ({', '.join(strata)}); " if apart else "); "
    reason += f"add {named} to group_cols"
    if apart:
        reason += " (or pass effort_strata=() to pool the strata explicitly)"
    return reason


def no_hierarchy_reasons(setup: Setup, fovs: pd.DataFrame) -> dict[str, str]:
    """Why questions 4 (across samples) and 7 were not assessed although the FOV
    table has technical repeats: ``group_cols`` leave them out."""
    technical = setup.technical_col(fovs)
    if technical is None:
        return {}  # the generic need (a technical_repeat column or a hierarchy) holds
    pooled = (
        f"group_cols {tuple(setup.group_cols or ())} leave {technical!r} out, so the "
        "technical repeats are pooled"
    )
    return {
        "effort": (
            f"not assessed: {pooled} and no sampling hierarchy is derived; add {technical!r} "
            "to group_cols, or declare hierarchy to model them anyway"
        ),
        "next_time": (
            f"protocol reliability across samples is not assessed: {pooled}, and the "
            f"reliability then has no result per technical repeat; add {technical!r} to "
            "group_cols"
        ),
    }


__all__ = [
    "assemble",
    "held_back_text",
    "no_hierarchy_reasons",
    "plan_settings",
    "pooling_reason",
    "protocol_reason",
    "report_settings",
    "with_reasons",
    "with_settings",
    "with_text",
]
