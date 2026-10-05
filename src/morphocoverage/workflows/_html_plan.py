"""The last section of the HTML page, "Plan the next experiment" (private; ``html_report``;
owner, 2026-10-02): the reader picks what the next experiment should get right, or what the
microscope should watch for, and gets the code, filled in with this dataset's values.

- **Want to optimise your imaging?** The mean of a feature in every well, a group of cells
  (from "Are there populations or groups of cells?") or the declared rare phenotype seen in
  enough fields of every well, the declared fingerprint: ``optimise`` on these data. A
  difference from the control is planned in repeats, not fields: the repeats a paired t-test
  on the repeat means needs (``repeat_plan``), with no code.
- **Want to focus on a specific phenotype on the microscope?** Stop each well once the mean
  of a feature is known, or once a group or the rare phenotype has been seen in enough
  fields: ``live_session`` with the rule that decides it, the acquisition labelling each
  field's cells as it writes them.

Each picker is a ``<select>`` showing one panel at a time (``_html_script``). The code reads
``result`` (the ``Report``), ``objects`` and ``fovs``, as in ``docs/workflows.md``.
"""

from __future__ import annotations

import html
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pandas as pd

from .._wording import change, count, listing, percent_fine, plural, rate
from ..data_check import validate_input_tables
from ..retrospective.report_status import group_word
from ..sampling_unit.reading_text import CONFIRMED, condition_readings
from ..sampling_unit.repeat_plan import MAX_REPEATS, POWER
from ..sampling_unit.repeat_plan_text import field_words
from ._enough_targets import held_back_note
from ._html_next import (
    _GUARD,
    RESERVOIR,
    _chosen_population,
    _labelled,
    _optimise_code,
    _repeats_line,
    fov_counts,
    pilot_conditions,
    rule_code,
    rule_labels,
)
from ._html_render import text_html
from ._populations import rule_words

PLAN_TITLE = "Plan the next experiment"
# Seeing a phenotype: the chance asked (a rule's risk is its complement) and the fields.
DETECTION = 0.95
OCCURRENCES = 3
# The folder the acquisition writes each field to, and the watcher reads.
LIVE_FOLDER = "live"


# The headings the stopping rules sit under in the text report's "Did I image enough?".
_RULES_HEADS = (
    "The stopping rules, as optimise and live run them (on the readout):",
    "The precision of each group's own mean:",
)


def _rules_note(result: Any) -> str:
    """How the stopping rules did on these fields, on the readout: the table the text report
    gives in "Did I image enough?", its assumptions left there."""
    text = result.document.short.get("enough", "")
    head = next((h for h in _RULES_HEADS if h in text), None)
    if head is None:
        return ""
    rest = text[text.index(head) + len(head) :].lstrip("\n")
    kept = []
    for line in rest.splitlines():
        if line.strip().startswith("Assumptions"):
            break
        kept.append(line)
    body = "\n".join(kept).strip("\n")
    if not body:
        return ""
    return (
        f"<p>How the stopping rules did on these fields, on the mean of "
        f"{html.escape(result.setup.readout)}:</p>" + text_html(body)
    )


def _fingerprint_note(result: Any) -> str:
    """What optimise's held-back fields can confirm of the fingerprint, per criterion."""
    precision = result.results.get("fingerprint_precision")
    if precision is None or not len(precision):
        return ""
    judged = precision[precision["met"].notna()]
    word = group_word(result.results.get("sufficiency"))
    notes = [
        held_back_note(rows, word, plural("field"))
        for _, rows in judged.groupby("criterion", sort=False)
    ]
    return "".join(f"<p>{html.escape(n)}</p>" for n in notes if n)


def _features(result: Any) -> list[str]:
    """The declared features, the readout first, then those significant across repeats in
    their rank, then the rest as declared."""
    columns = list(result.setup.columns)
    readout = result.setup.readout
    ranked: list[str] = []
    discovery = result.discovery
    if discovery is not None and discovery.tests is not None and len(discovery.tests):
        tests = discovery.tests
        confirmed = set(tests.loc[tests["confirmed"].fillna(False).astype(bool), "feature"])
        ranked = [f for f in discovery.ranking["feature"] if f in confirmed]
    order = [readout, *ranked, *columns]
    return [f for f in dict.fromkeys(order) if f in columns]


def _significant(result: Any) -> set[str]:
    discovery = result.discovery
    if discovery is None or discovery.tests is None or not len(discovery.tests):
        return set()
    tests = discovery.tests
    return set(tests.loc[tests["confirmed"].fillna(False).astype(bool), "feature"])


def _incidence(
    result: Any, objects: pd.DataFrame, fovs: pd.DataFrame, labelled: pd.Series
) -> tuple[float, str] | None:
    """The share of fields holding at least one labelled observation in the condition where
    it is rarest (the hardest well to find it in) and that condition; ``None`` when no field
    holds it."""
    data = validate_input_tables(
        objects, fovs=fovs, feature_cols=list(result.setup.columns)[:1], **result.setup.common
    )
    key = [c for c in data.fov_key if c in data.objects]
    if "condition" not in data.objects or not key:
        return None
    flags = labelled.reindex(data.objects.index).fillna(False).astype(bool)
    frame = data.objects[key].astype(str).assign(_held=flags.to_numpy())
    fields = frame.groupby(key, sort=False)["_held"].any()
    by = fields.groupby(level="condition").mean()
    by = by[by > 0]
    if by.empty:
        return None
    condition = str(by.idxmin())
    return float(by.min()), condition


def _meta_comment(fovs: pd.DataFrame) -> str:
    """The metadata a field is written with, named from this dataset's own columns."""
    names = [
        c
        for c in ("condition", "biological_repeat", "technical_repeat", "position_id", "image_file")
        if c in fovs
    ]
    return "{" + ", ".join(f'"{c}": ...' for c in names) + "}"


def _panel(sentence: str, code: str = "", extra: str = "") -> str:
    """A choice's panel: what it does, its code, and what goes with it."""
    block = f"<pre><code>{html.escape(code)}</code></pre>" if code else ""
    return f"<p>{html.escape(sentence)}</p>{extra}{block}"


def _choose(name: str, label: str, items: Sequence[tuple[str, str]]) -> str:
    """A ``<select>`` called ``name`` (an id prefix) and one panel per ``(option, markup)``,
    the first shown."""
    options = "".join(
        f'<option value="{i}"{" selected" if i == 0 else ""}>{html.escape(text)}</option>'
        for i, (text, _) in enumerate(items)
    )
    panels = "".join(
        f'<div class="choice-panel" id="{name}-panel-{i}"{"" if i == 0 else " hidden"}>'
        f"{markup}</div>"
        for i, (_, markup) in enumerate(items)
    )
    return (
        f'<div class="choose"><label for="{name}-pick">{html.escape(label)}</label>'
        f'<select id="{name}-pick">{options}</select>{panels}</div>'
    )


def _tolerances(result: Any) -> str:
    return listing([rate(t) for t in result.setup.tolerances], 4).replace(", and ", " and ")


def _mean_goal(result: Any, feature: str, counts: list[int], extra: str) -> tuple[str, str]:
    readout = result.setup.readout
    # The features significant across repeats come first (``_features``): no suffix, which
    # would read as if optimise planned that significance (ChatGPT audit A1, 2026-10-02).
    option = f"The mean of {feature} in every well"
    if not counts:
        sentence = (
            "These wells hold too few fields for optimise to replay: image more fields per "
            f"well first, then plan the fields for {feature}."
        )
        return option, _panel(sentence, extra=extra)
    sentence = (
        f"Optimise replays each well's fields against fields held back from it: how many "
        f"fields hold the mean of {feature} within {_tolerances(result)}, in “How many "
        "fields next time?”, and how reliable each stopping rule was, which says the rule "
        "to run live."
    )
    return option, _panel(sentence, _optimise_code(feature, counts, readout), extra)


def _plans(result: Any) -> list[tuple[str, Any]]:
    """Each feature's repeat plan, in the order of ``_features``."""
    features = result.results.get("enough_features")
    contrasts = getattr(features, "contrasts", None) or {}
    contrast = result.results.get("contrast")
    if not contrasts and contrast is not None:
        contrasts = {result.setup.readout: contrast}
    return [(f, contrasts[f]) for f in _features(result) if f in contrasts]


def _confirmed_pairs(result: Any) -> set[tuple[str, str]]:
    """The (feature, condition) pairs significant across repeats."""
    discovery = result.discovery
    if discovery is None or discovery.tests is None or not len(discovery.tests):
        return set()
    tests = discovery.tests
    confirmed = tests[tests["confirmed"].fillna(False).astype(bool)]
    return {(str(f), str(c)) for f, c in zip(confirmed["feature"], confirmed["condition"])}


def _plan_words(value: Any) -> str:
    if pd.isna(value):
        return "–"
    return f"more than {MAX_REPEATS}" if not np.isfinite(float(value)) else str(int(value))


def _difference_goal(result: Any) -> tuple[str, str] | None:
    """The repeats each condition not yet significant across repeats would need, per
    feature; with too few repeats, how many to plan."""
    discovery = result.discovery
    few = (
        _repeats_line(discovery)
        if discovery is not None and discovery.tests is not None and len(discovery.tests)
        else ""
    )
    option = "A difference from the control"
    pilots = pilot_conditions(discovery)
    plans = _plans(result)
    # Every condition a pilot: only how many repeats to plan. Some: the others' plan, then
    # the pilots named (stress test, 2026-10-02).
    if few and (not plans or len(pilots) == len(_repeat_conditions(discovery))):
        return option, _panel(few)
    if not plans:
        return None
    done = _confirmed_pairs(result)
    # Without a feature discovery (one feature), the readout's own verdicts.
    contrast = result.results.get("contrast")
    if not done and contrast is not None:
        readings = condition_readings(contrast)
        if len(readings):
            confirmed = readings.loc[readings["verdict"] == CONFIRMED, "condition"]
            done = {(result.setup.readout, str(c)) for c in confirmed}
    rows, shown = [], []
    for feature, contrast in plans:
        plan = getattr(contrast, "plan", None)
        if plan is None or plan.empty:
            continue
        unit = field_words(contrast)[1]
        for _, row in plan.iterrows():
            if str(row["condition"]) in pilots:
                continue
            if (feature, str(row["condition"])) in done:
                shown.append(feature)
                continue
            if pd.isna(row["repeats_needed"]):
                continue
            planned = str(row["planned_on"])
            moved = (
                change(float(row["change"]), "normalised" if planned == "log ratio" else "raw")
                if planned in ("log ratio", "difference") and pd.notna(row["change"])
                else "–"
            )
            rows.append(
                (
                    feature,
                    str(row["condition"]),
                    moved,
                    _plan_words(row["repeats_needed"]),
                    _plan_words(row["repeats_needed_twice_fields"]),
                    unit,
                )
            )
    already = list(dict.fromkeys(shown))
    known = f" Significant across repeats already: {listing(already, 6)}." if already else ""
    pilot = f"<p>{html.escape(few)}</p>" if few else ""
    if not rows:
        done_words = (
            "Every condition with 3 or more repeats is significant across repeats."
            if few
            else "Every condition is significant across repeats."
        )
        return option, _panel(f"{done_words}{known}".strip(), extra=pilot)
    unit = rows[0][5]
    head = (
        "<tr><th>feature</th><th>condition</th><th>change seen</th><th>repeats needed</th>"
        f"<th>with twice the {html.escape(unit)} per repeat</th></tr>"
    )
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(v)}</td>" for v in row[:5]) + "</tr>" for row in rows
    )
    table = (
        f'<div class="table-wrap"><table class="sortable"><thead>{head}</thead>'
        f"<tbody>{body}</tbody></table></div>"
    )
    sentence = (
        "Optimise plans each well's own measurements, not a comparison: a difference from "
        "the control is planned in biological repeats. For each condition not yet "
        "significant across repeats, the repeats a paired t-test on the repeat means needs "
        f"for {rate(POWER)} power, if the change and the variation seen here recur, each "
        f"repeat with its own control.{known}"
    )
    return option, _panel(sentence, extra=table + pilot)


def _repeat_conditions(discovery: Any) -> set[str]:
    return {str(c) for c in discovery.tests["condition"]} if discovery is not None else set()


def _group(result: Any) -> Any | None:
    """The group of "Are there populations or groups of cells?" to plan for, with a rule
    and not the declared rare label's; ``None`` otherwise."""
    populations = (getattr(result, "results", {}) or {}).get("populations")
    row = _chosen_population(populations) if populations is not None else None
    if row is None or not row["rule"]:
        return None
    if _labelled(populations, str(row.name), result.setup.rare_label):
        return None
    return row


def _target_code(name: str, incidence: float, prevalence: float | None, phenotype: str) -> str:
    share = f"object_prevalence={round(prevalence, 4)}, " if prevalence is not None else ""
    return (
        f"target = RareTarget(fov_incidence={round(incidence, 4)}, {share}"
        f"detection_probability={DETECTION},\n"
        f"                    min_fov_occurrences={OCCURRENCES}, phenotype={phenotype!r})\n"
    )


def _cleared(setup: Any, keep_rare: bool = False) -> str:
    """The setup's other declared targets, cleared so a stop waits for the one chosen."""
    parts = []
    if setup.fingerprint:
        parts.append("fingerprint=()")
    if not keep_rare and setup.rare is not None:
        parts += ["rare=None", "rare_label=None"]
    return "".join(f", {p}" for p in parts)


def _group_goals(
    result: Any, objects: pd.DataFrame | None, fovs: pd.DataFrame, counts: list[int]
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """The group's optimise and live choices: label it with its rule, then plan its fields
    or stop each well once it has been seen."""
    row = _group(result)
    if row is None or objects is None:
        return [], []
    name, terms = str(row.name), list(row["rule_terms"])
    labelled = rule_labels(objects, terms)
    found = _incidence(result, objects, fovs, labelled)
    if found is None:
        return [], []
    incidence, where = found
    prevalence = float(labelled.mean())
    phenotype = f"{name}: {row['rule']}"
    rarest = (
        f"It is planned on the condition where it is rarest, {where}: "
        f"{percent_fine(incidence)} of its fields hold it."
    )
    seen = f"{name} seen in at least {count(OCCURRENCES, 'field')} of every well"
    labels = f"objects[{name!r}] = {rule_code(terms, 'objects')}\n"
    plan_code = (
        _GUARD + "from dataclasses import replace\n"
        "from morphocoverage import RareTarget\n"
        "from morphocoverage.workflows import optimise\n\n"
        + labels
        + _target_code(name, incidence, prevalence, phenotype)
        + f"setup = replace(result.setup, rare=target, rare_label={name!r})\n"
        f"plan = optimise(objects, fovs, setup, fov_counts={counts}, "
        f"reservoir_fraction={RESERVOIR})\n"
        "print(plan.summary())"
    )
    plan_sentence = (
        f"Once {name} is validated on the images, label its {result.setup.unit_words} with "
        f"its rule, {rule_words(row, result.setup.unit_words)}, and declare it: optimise says "
        f"how many fields find it in at least {count(OCCURRENCES, 'field')} of every well, in "
        f"“How many fields next time?”. {rarest}"
    )
    optimise = [(seen, _panel(plan_sentence, plan_code if counts else ""))]
    live = [_live_rare(result, fovs, name, incidence, where, None, phenotype, terms)]
    return optimise, live


def _live_rare(
    result: Any,
    fovs: pd.DataFrame,
    name: str,
    incidence: float,
    where: str,
    prevalence: float | None,
    phenotype: str,
    terms: Sequence[Sequence[Any]] | None,
) -> tuple[str, str]:
    """Stop each well once ``name`` has been seen in ``OCCURRENCES`` fields, or once enough
    fields without it show it rarer than in ``where``."""
    setup = result.setup
    risk = round(1 - DETECTION, 4)
    label = (
        f"#   field_objects[{name!r}] = {rule_code(terms, 'field_objects')}\n"
        if terms
        else f"#   field_objects[{name!r}]: each {setup.unit_word}'s label, True or False\n"
    )
    code = (
        "from dataclasses import replace\n"
        "from morphocoverage import ObjectiveConfig, RareTarget\n"
        "from morphocoverage.strategies import RareIncidenceStrategy\n"
        "from morphocoverage.workflows import LiveFolder, live_session\n\n"
        + _target_code(name, incidence, prevalence, phenotype)
        + f"setup = replace(result.setup, rare=target, rare_label={name!r}"
        + _cleared(setup, keep_rare=True)
        + ")\n"
        f"rule = RareIncidenceStrategy({name!r}, prior_a=1.0, prior_b=1.0, max_risk={risk})\n"
        "session = live_session(setup, rules=(rule,),\n"
        '                       objective=ObjectiveConfig("rare_discovery", rare=target))\n\n'
        "# On the acquisition computer, once per completed field "
        f"(field_objects: its {setup.unit_words}):\n"
        "#   from morphocoverage.workflows import write_fov\n"
        + label
        + f"#   write_fov({LIVE_FOLDER!r}, field_objects, {_meta_comment(fovs)})\n"
        f'LiveFolder({LIVE_FOLDER!r}, session).watch(interval=1.0, until_stop="latest_group")'
    )
    labelling = (
        f"The acquisition labels each field's {setup.unit_words} with {name}'s rule as it "
        "writes them."
        if terms
        else f"The acquisition writes each {setup.unit_word}'s {name} label with its field."
    )
    sentence = (
        f"Each well stops once {name} has been seen in {count(OCCURRENCES, 'field')}, or once "
        f"enough fields without it show it in fewer than {percent_fine(incidence)} of fields, "
        f"as in {where}, where it is rarest here; the risk of stopping too early is "
        f"{rate(risk)}. {labelling}"
    )
    return f"…{name} has been seen in {count(OCCURRENCES, 'field')}", _panel(sentence, code)


def _declared_goals(
    result: Any,
    objects: pd.DataFrame | None,
    fovs: pd.DataFrame,
    counts: list[int],
    notes: dict[str, str],
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """The declared rare phenotype and fingerprint: optimise judges them as declared; live
    stops on the rare phenotype once it has been seen."""
    setup = result.setup
    optimise, live = [], []
    code = (
        (
            _GUARD + "from morphocoverage.workflows import optimise\n\n"
            f"plan = optimise(objects, fovs, result.setup, fov_counts={counts}, "
            f"reservoir_fraction={RESERVOIR})\n"
            "print(plan.summary())"
        )
        if counts
        else ""
    )
    label = setup.rare_label
    if setup.rare is not None and label is not None:
        fields = setup.rare.min_fov_occurrences
        sentence = (
            f"{label} is declared: optimise says how many fields find it in at least "
            f"{count(fields, 'field')} of every well, in “How many fields next time?”."
        )
        optimise.append(
            (
                f"{label} seen in at least {count(fields, 'field')} of every well",
                _panel(sentence, code, notes.get("rare", "")),
            )
        )
        if objects is not None and label in objects:
            flags = objects[label].fillna(False).astype(bool)
            found = _incidence(result, objects, fovs, flags)
            if found is not None:
                incidence, where = found
                declared = setup.rare.fov_incidence
                live.append(
                    _live_rare(
                        result,
                        fovs,
                        label,
                        float(declared) if declared is not None else incidence,
                        where,
                        setup.rare.object_prevalence,
                        setup.rare.phenotype or label,
                        None,
                    )
                )
    if setup.fingerprint:
        sentence = (
            "The fingerprint is declared: optimise judges each of its criteria on fields "
            "held back from each well, in “How many fields next time?”."
        )
        markup = _panel(sentence, code, notes.get("fingerprint", ""))
        optimise.append(("The fingerprint in every well", markup))
    return optimise, live


def _live_mean(result: Any, feature: str, fovs: pd.DataFrame) -> tuple[str, str]:
    setup = result.setup
    readout = "" if feature == setup.readout else f", readout={feature!r}"
    cleared = _cleared(setup)
    head = "from dataclasses import replace\n" if readout or cleared else ""
    made = (
        f"setup = replace(result.setup{readout}{cleared})\n"
        if readout or cleared
        else "setup = result.setup\n"
    )
    code = (
        head
        + "from morphocoverage.workflows import LiveFolder, live_session\n\n"
        + made
        + "\n# On the acquisition computer, once per completed field "
        f"(field_objects: its {result.setup.unit_words}):\n"
        "#   from morphocoverage.workflows import write_fov\n"
        f"#   write_fov({LIVE_FOLDER!r}, field_objects, {_meta_comment(fovs)})\n"
        f"LiveFolder({LIVE_FOLDER!r}, live_session(setup)).watch(interval=1.0, "
        'until_stop="latest_group")'
    )
    sentence = (
        f"Each well stops once its stopping rules say the mean of {feature} is known within "
        f"{rate(setup.tolerances[0])}. Optimise, above, says how reliable each rule was on "
        "these data."
    )
    return f"…the mean of {feature} is known", _panel(sentence, code)


def plan_card(
    result: Any,
    objects: pd.DataFrame | None,
    fovs: pd.DataFrame,
    annotate: Callable[[str], str],
    notes: dict[str, str] | None = None,
) -> str:
    """The section's body (see the module); ``notes`` are the report's own words to show
    with a choice, by default ``"rules"`` (how the stopping rules did, with the readout's
    mean) and ``"fingerprint"`` (what optimise's held-back fields can confirm of it)."""
    notes = (
        {"rules": _rules_note(result), "fingerprint": _fingerprint_note(result)}
        if notes is None
        else notes
    )
    counts = fov_counts(fovs, result.setup.group_cols)
    readout = result.setup.readout
    means = {
        f: _mean_goal(result, f, counts, notes.get("rules", "") if f == readout else "")
        for f in _features(result)
    }
    first = [f for f in means if f == readout or f in _significant(result)]
    group_plan, group_live = _group_goals(result, objects, fovs, counts)
    declared_plan, declared_live = _declared_goals(result, objects, fovs, counts, notes)
    difference = _difference_goal(result)
    # The readout and the features significant across repeats, the groups and declared
    # targets, the difference from the control, then the other features.
    optimise = [
        *(means[f] for f in first),
        *group_plan,
        *declared_plan,
        *([difference] if difference is not None else []),
        *(means[f] for f in means if f not in first),
    ]
    live = [*group_live, *declared_live, *(_live_mean(result, f, fovs) for f in _features(result))]
    body = (
        "<p>Pick what the next experiment should get right, or what the microscope should "
        "watch for: the code below continues the session that made this report "
        "(objects, fovs and result).</p>"
        "<h3>Want to optimise your imaging?</h3>"
        + _choose("plan-optimise", "What should the next experiment get right?", optimise)
        + "<h3>Want to focus on a specific phenotype on the microscope?</h3>"
        + _choose("plan-live", "Stop each well once…", live)
        + '<p class="note">Live decides when to stop imaging a well, not where to image.</p>'
    )
    return annotate(body)


__all__ = ["PLAN_TITLE", "plan_card"]
