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

import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import pandas as pd

from .._caveats import ASSUMPTIONS, assumption, caveat_lines, pool_assumptions
from .._style import CAUTION, Status, worst, wrap
from .._wording import count, fields, listing, number, percent
from ..metadata import REPEAT_COLUMN
from ..retrospective import MicroscopistReport, SamplingHierarchy, microscopist_report
from ..retrospective.report import QUESTIONS, with_opening
from ..retrospective.report_profile import sampling_profile
from ..retrospective.report_sampling import states_lines
from ..retrospective.report_status import (
    RESULT_QUESTIONS,
    caveat_status,
    group_word,
    profile_status,
)
from ..sampling_unit._state_text import clustering_parts
from ..sampling_unit.repeat_plan_text import field_words
from ..schema import observation_words
from ._declaration import declaration_short
from ._distribution import with_distribution
from ._enough_features import default_precision, with_features
from ._enough_targets import with_targets
from ._label_control import with_label_check
from ._populations import with_populations

if TYPE_CHECKING:
    from .setup import Setup

# Results of the report that ``microscopist_report`` does not take; ``assemble`` adds them.
_EXTRA_RESULTS = (
    "rare_clustering",
    "integrity",
    "enough_features",
    "distribution",
    "fingerprint_precision",
    "label_control",
    "populations",
)


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
    short, headlines = dict(document.short), dict(document.headlines)
    for key, reason in reasons.items():
        sentence = f"{reason[:1].upper()}{reason[1:]}."
        if document.inputs.get(key) == "not assessed":
            sections[key] = f"  {sentence}"
            # The reason is the answer's one line: what it could not assess, and what it needs.
            headlines[key] = reason
        else:
            wrapped = wrap(sentence, 2, hang=4)
            sections[key] = f"{sections[key]}\n\n{wrapped}"
            if key in short:
                short[key] = f"{short[key]}\n\n{wrapped}"
            notes.append(f"{titles[key]} {sentence}")
    return replace(
        document, sections=sections, notes=tuple(notes), short=short, headlines=headlines
    )


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
    document = replace(document, short={**document.short, "declaration": declaration_short(setup)})
    integrity = results.get("integrity")
    if integrity is not None:
        # What the tables hold includes structure to inspect before reading the answers.
        overview = results.get("overview")
        if overview is not None:
            document = with_opening(document, overview, integrity)
        else:
            full = integrity.summary(detail="full")
            text = "\n\n".join(p for p in (document.overview, full) if p)
            document = replace(document, overview=text)
        document = _with_integrity(document, integrity)
    rare = results.get("rare_clustering")
    if rare is not None:
        rows = sampling_profile(None, None, None, None, clustering=rare)
        lines = "\n".join(f"{r.statement} [{r.where}; {r.source}]" for r in rows.itertuples())
        text = "\n\n".join(part for part in (lines, rare.summary(detail="full")) if part)
        status, glance = profile_status(rows, [rare])
        words = observation_words(rare.settings)[1]
        brief = [wrap(f"- {line}", 2, hang=4) for line in states_lines(rare, None, words)]
        brief_text = "\n".join([*brief, *caveat_lines(clustering_parts(rare, 2)[1], indent=2)])
        document = with_text(
            document, "profile", text, rows, status=status, headline=glance, short_text=brief_text
        )
    controls = results.get("controls")
    if controls is not None and controls.status != "not assessed" and not controls.pair_cols:
        paired = (
            "Positive and reference fields were paired across the whole table (pair_cols=()): "
            "every repeat is pooled, so a shift between repeats can read as detection."
        )
        document = with_text(
            document,
            "controls",
            paired,
            status=Status.CAUTION,
            short_text="\n".join(caveat_lines([paired], indent=2)),
        )
    document = _technical_repeats(document, results)
    document = replace(document, notes=(*document.notes, *coverage_notes(results, setup)))
    pooled = _reference_repeats(results.get("unusual"))
    if pooled > 1:
        units = setup.unit_words
        text = (
            f"The reference of the unusual {units} pools the control {units} of "
            f"{count(pooled, 'repeat')}: a shift shared by one repeat (a staining or plating "
            f"batch) makes its {units} read as unusual, so compare the unusual share between "
            "repeats before reading it as biology."
        )
        document = with_text(
            document,
            "profile",
            text,
            status=Status.CAUTION,
            short_text="\n".join(caveat_lines([text], indent=2)),
        )
    # Question 2 over every feature alike (the report is neutral).
    features = results.get("enough_features")
    word = group_word(results.get("sufficiency"))
    default = default_precision(setup)
    document = with_features(document, features, results.get("contrast"), word, default)
    # The rare and fingerprint targets, answered from the fields acquired.
    contrast = results.get("contrast")
    field = field_words(contrast)[0] if contrast is not None else "field"
    document = with_targets(document, results, setup, word, field)
    document = with_populations(document, results.get("populations"), setup)
    document = with_label_check(document, results.get("label_control"), results)
    # Beyond the mean, with a fingerprint declared (question 9).
    return with_distribution(document, results.get("distribution"), setup.unit_words)


def _with_integrity(document: MicroscopistReport, integrity: Any) -> MicroscopistReport:
    """The report with the fields the integrity checks name among the fields to look at (6)
    and in the limits: every answer counts them as independent fields."""
    involved = integrity.field_labels()
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
    # The short form names them once, in the overview, and points there.
    pointer = (
        f"Look first at the {count(len(involved), 'field')} the integrity checks name in the "
        "overview: every answer here counts them as independent fields."
    )
    # Appended whether or not the fields were inspected: the question stays "not assessed"
    # without an inspection, and its status asks the reader to look either way.
    sections, short = dict(document.sections), dict(document.short)
    sections["inspect"] = f"{sections['inspect']}\n\n{wrap(sentence, 2, hang=4)}"
    if "inspect" in short:
        short["inspect"] = f"{short['inspect']}\n\n{wrap(pointer, 2, hang=4)}"
    statuses = {**document.statuses, "inspect": Status.CAUTION}
    headlines = dict(document.headlines)
    headlines["inspect"] = "; ".join(
        part
        for part in (headlines.get("inspect"), f"integrity names {count(len(involved), 'field')}")
        if part
    )
    document = replace(
        document, sections=sections, short=short, statuses=statuses, headlines=headlines
    )
    note = (
        f"Data integrity: the checks in the overview name {named}, and every answer counts "
        "them as independent fields."
    )
    return replace(document, notes=(*document.notes, note))


def _between_technical_repeats(fov_rows: pd.DataFrame, technical: str, outer: list[str]) -> list:
    """Per condition and repeat with 2 or more technical repeats: the spread of their means
    against what their fields predict, and the relative SE of the mean over them.

    From the fields' means (every field with observations weighing one): the
    observed variance of the technical-repeat means against the mean over
    technical repeats of ``var(field means) / fields``, which is all the
    fields alone would give. Returns ``(ratio, relative SE)`` pairs.
    """
    rows = fov_rows.dropna(subset=["mean"])
    parts = rows.groupby(outer, sort=False, dropna=False) if outer else [((), rows)]
    found = []
    for _, own in parts:
        per = own.groupby(technical, sort=False)["mean"].agg(["mean", "var", "count"])
        per = per[per["count"] >= 2]
        if len(per) < 2:
            continue
        predicted = float((per["var"] / per["count"]).mean())
        observed = float(per["mean"].var(ddof=1))
        grand = abs(float(per["mean"].mean()))
        if predicted <= 0 or grand == 0:
            continue
        found.append(
            (math.sqrt(observed / predicted), math.sqrt(observed / len(per)) / grand, len(per))
        )
    return found


def _technical_repeats(document: MicroscopistReport, results: Mapping[str, Any]) -> Any:
    """What every "enough" leaves out about technical repeats (audit 4, P1-7 and P1-8).

    With technical repeats kept apart, question 3 gets how much their means
    differ against what their fields predict; with them pooled, question 2
    says that its groups pool them. Either way a closing note says what an
    "enough" is per.
    """
    overview, variability = results.get("overview"), results.get("variability")
    if overview is None or variability is None:
        return document
    technical = overview.settings.get("technical_col")
    layout = overview.layout
    if technical is None or "technical_repeats" not in layout.columns or not len(layout):
        return document
    per_group = layout["technical_repeats"].dropna().astype(int)
    if technical not in variability.group_cols:
        if not len(per_group) or per_group.max() < 2:
            return document
        low, high = int(per_group.min()), int(per_group.max())
        span = f"{low}" if low == high else f"{low} to {high}"
        text = (
            f"Each analysis group pools the technical repeats of its condition and repeat "
            f"({span} per group: the analysis groups {tuple(variability.group_cols)} leave "
            f"{technical!r} out), so every answer above counts their fields together, as if "
            "they came from one sample: fields of one technical repeat are not independent "
            "draws of the condition, and the variation between technical repeats is not seen "
            f"by any field count here; add {technical!r} to group_cols to judge each apart."
        )
        note = (
            f"Each 'enough' is for a group that pools technical repeats ({span} per group), "
            "so it counts fields, not independent technical repeats."
        )
        document = with_text(document, "enough", text, status=Status.CAUTION)
        return replace(document, notes=(*document.notes, note))
    outer = [c for c in variability.group_cols if c != technical]
    found = _between_technical_repeats(variability.fovs, technical, outer)
    note = assumption(
        "Each 'enough' is per technical repeat; the condition mean also varies between "
        "technical repeats, which more fields of one technical repeat cannot reduce "
        "(question 3; optimise, question 7: more fields or more technical repeats?)."
    )
    if found:
        ratios = sorted(r for r, _, _ in found)
        errors = sorted(e for _, e, _ in found)
        ratio, error = statistics.median(ratios), statistics.median(errors)
        feature = variability.settings.get("feature")
        few = min(n for _, _, n in found)
        groups = count(len(found), "condition and repeat group")
        uncertain = (
            f"; with {few} technical repeats the spread itself is uncertain" if few < 5 else ""
        )
        text = (
            f"Between technical repeats ({feature}): in {groups} with 2 or more technical "
            f"repeats, their means spread {number(ratio)} times (median; {number(ratios[0])} "
            f"to {number(ratios[-1])}) as much as their fields alone predict, and the mean over "
            f"them has a relative standard error of about {percent(error)} (median; "
            f"{percent(errors[0])} to {percent(errors[-1])}) from that spread. Every 'enough' "
            "of question 2 is per technical repeat and says nothing about this variation, "
            f"which only more technical repeats reduce (optimise, question 7){uncertain}."
        )
        brief = "\n".join(caveat_lines([text], indent=2))
        document = with_text(document, "profile", text, status=Status.CAUTION, short_text=brief)
    return replace(document, notes=(*document.notes, note))


def coverage_notes(results: Mapping[str, Any], setup: Setup) -> list[str]:
    """What the assessed questions did not read: columns, stage positions, acquisition order.

    "Every question was assessed" is not "everything was checked" (audit 4,
    P1-9): the Foundation, the stopping rules and the inspection read the
    readout only, and without stage coordinates or an acquisition order some
    checks cannot run at all.
    """
    notes = []
    others = [f for f in setup.features if f != setup.readout]
    if others and "variability" in results:
        # The analyses that ran; rules of the caller's own may read every declared column.
        found = (setup.novelty, *(results.get(k) for k in ("controls", "unusual", "ranking")))
        names = ("novelty", "the representation control", "the unusual", "the feature ranking")
        uses = [n for n, r in zip(names, found, strict=True) if r is not None]
        uses = listing([f"{n} {setup.unit_words}" if n == "the unusual" else n for n in uses])
        spread = "a heavy tail or a between-well spread in"
        rest = f"the other declared features ({listing(others)})"
        rest = f"; {rest} enter {uses}, so {spread} them" if uses else f", so {spread} {rest}"
        rules = setup.rules is None
        notes.append(
            assumption(
                f"The Foundation{', the stopping rules' if rules else ''} and the inspection's "
                f"influence and neighbour reasons read the readout ({setup.readout}) only"
                f"{rest} is not judged by {'any stopping rule' if rules else 'them'}."
            )
        )
    # Options left at their defaults: what they leave unassessed, as notes, not cautions.
    if setup.stage_cols is None:
        notes.append(
            assumption(
                "No stage coordinates: neighbouring fields were not compared, and "
                "spatial redundancy and dispersed orders were not assessed."
            )
        )
    if setup.acquisition_order_col is None:
        notes.append(
            assumption(
                "No acquisition order: the recorded order was not "
                "replayed, and the random orders treat the fields as exchangeable, so a drift "
                "or a raster along the acquisition is not seen."
            )
        )
    return notes


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
    document: MicroscopistReport,
    key: str,
    text: str,
    rows: pd.DataFrame | None = None,
    *,
    status: Status = Status.OK,
    headline: str | None = None,
    short_text: str | None = None,
) -> MicroscopistReport:
    """The report with ``text`` added to question ``key`` (and ``rows`` to its profile).

    A question not assessed so far is assessed by it: the text replaces the
    generic need, ``headline`` (or the text's first sentence) becomes its
    one-line answer and ``status`` its status. Added to an assessed answer,
    the text follows it in the full and in the short form, ``headline`` joins its
    one-line answer, and ``status``
    (``report_status``: a caution the text states) can only raise its
    status, never lower it. ``short_text``, already indented as the report
    prints it, stands for ``text`` in the short form (a ⚠️ line for a caveat).
    """
    indented = "\n".join(f"  {line}" if line else line for line in text.splitlines())
    brief = short_text if short_text is not None else indented
    sections, inputs = dict(document.sections), dict(document.inputs)
    short, statuses = dict(document.short), dict(document.statuses)
    headlines = dict(document.headlines)
    if inputs.get(key) == "not assessed":
        sections[key], inputs[key] = indented, "given"
        short.pop(key, None)
        if short_text is not None:
            short[key] = short_text
        statuses[key] = caveat_status(status, short.get(key, sections[key]), key)
        headlines[key] = headline or text.split(". ")[0].split("\n")[0].rstrip(".")
    else:
        # One "Assumptions:" block per section, at its end; a caution raises the status.
        sections[key] = pool_assumptions(f"{sections[key]}\n\n{indented}")
        if key in short:
            merged = _merged(short[key], brief) if short_text else f"{short[key]}\n\n{brief}"
            short[key] = pool_assumptions(merged)
        if headline:
            # The added answer joins the one-line answer (its parts not already in it).
            known = headlines.get(key, "")
            new = [p for p in headline.split("; ") if p and p not in known]
            headlines[key] = "; ".join([known, *new]) if known else headline
        current = statuses.get(key, Status.NOT_ASSESSED)
        if key in RESULT_QUESTIONS and status is not Status.PROBLEM:
            # Its marker says what was found; a caution added here is read in the section.
            raised = current
        else:
            raised = worst([current, status])
        statuses[key] = caveat_status(raised, short.get(key, sections[key]), key)
    profile = document.profile
    if rows is not None and len(rows):
        profile = pd.concat([profile, rows], ignore_index=True) if len(profile) else rows
    return replace(
        document,
        sections=sections,
        inputs=inputs,
        profile=profile,
        short=short,
        statuses=statuses,
        headlines=headlines,
    )


def _caution_split(text: str) -> tuple[list[str], list[str]]:
    """A short section's lines before its closing ⚠️ block, and that block's caveats (each
    with its continuation lines, three columns deeper)."""
    lines = text.splitlines()
    start, depth = None, None
    for index, line in enumerate(lines):
        stripped = line.lstrip()
        indent = len(line) - len(stripped)
        if stripped.startswith(CAUTION):
            start = index if start is None else start
            depth = indent + 3
        elif start is not None and indent != depth:
            start = None
    start = len(lines) if start is None else start
    cautions: list[str] = []
    for line in lines[start:]:
        if line.lstrip().startswith(CAUTION):
            cautions.append(line)
        else:
            cautions[-1] += "\n" + line
    return lines[:start], cautions


def _apart(text: str) -> tuple[str, list[str]]:
    """A short section without its "Assumptions:" block, and that block's lines."""
    lines = pool_assumptions(text).splitlines()
    starts = [i for i, line in enumerate(lines) if line.lstrip().startswith(ASSUMPTIONS)]
    return ("\n".join(lines[: starts[-1]]), lines[starts[-1] :]) if starts else (text, [])


def _merged(short: str, brief: str) -> str:
    """``brief`` added to a short section: its lines before the section's caveats, the caveats
    of both after them, each once, then the assumptions of both (``pool_assumptions``)."""
    short, standing = _apart(short)
    brief, more_standing = _apart(brief)
    body, cautions = _caution_split(short)
    extra, more = _caution_split(brief)
    kept = list(dict.fromkeys([*cautions, *more]))
    return pool_assumptions("\n".join([*body, *extra, *kept, *standing, *more_standing]))


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
