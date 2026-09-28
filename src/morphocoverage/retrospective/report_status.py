"""The status and headline of every answer of the microscopist's report, written once.

Each function below reads one question's results and returns ``(Status,
headline)``: the marker of the answer (``_style``: ✅ answered and good for
the user's goal, ⚠️ answered but look at this, ❌ a problem that changes how
the answers are read or acted on, ➖ not assessed) and its answer in one
line, for the report's "At a glance" block and its section header. Nothing is
recomputed: the statuses read the result tables. The rules, question by
question:

- **What is in the dataset** (``overview_status``): the data checks
  (``check_tables``) ✅ without a warning, ⚠️ with one, ➖ not run; the
  data-integrity checks ✅ with no finding, ❌ with a finding (every answer
  counts those fields as independent), ➖ not run or not applicable. The
  worse of the two.
- **Foundation** (``unit_status``): ⚠️ when the JCB object-count rule missed
  its target with whole fields in some group, when whole-field samples never
  stayed within the random-object descriptor spread in some group, or when a
  group's variability could not be estimated; ✅ otherwise. Headline: how
  much more a mean from whole fields varies (the median design effect), and
  in how many groups the JCB rule held with whole fields.
- **1. Controls** (``controls_status``): ❌ when the declared positive
  control (the planted objects when declared) was recognized in fewer than
  ``minimum_detection`` of checks (weak representation); ⚠️ when a stratum
  could not be judged (named, with the reason), the false-alarm rate was not
  assessed or is above ``FALSE_ALARM_CAUTION``; ➖ without a control; ✅
  otherwise.
- **2. Did I image enough?** (``enough_status``): over the rules that judge
  the data (not the fixed amounts of effort), each by its own threshold: ❌
  when none was met in any group; ✅ when every one was met in every group,
  none carries a caution of its own where it was met (a non-normal readout,
  field size related to field mean), the order check flagged no group, no
  rule was met and then lost, and in the held-back check (calibration) no
  rule stopped where the criterion was not preserved in more than
  ``PREMATURE_CAUTION`` of the acquisitions of a group and order kind; ⚠️
  otherwise (met in some groups or by some rules only, or a caution). Only
  fixed amounts of effort: ⚠️ (reaching a target is not evidence).
  With a weak representation (question 1 ❌), ❌ whatever the rules say
  (``weak_status``), as for questions 4 and 5: a "nothing new" or an
  "enough" cannot exclude a phenotype like the control.
- **3. What is limiting my sampling?** (``profile_status``): ⚠️ when a
  labelled state is clustered in a few fields (p <= 0.05) in some group,
  when the unusual-object labels are unstable (refits agree on fewer than
  90% of them) in some group, or when the largest source of the variance of
  the mean is not established (few samples, an interval including zero) in
  some stratum; ✅ when something was measured and none of these holds.
  Headline: the design effect of whole fields, the clustered states, and
  where most of the variance of the mean lies.
- **4. Next time** (``next_time_status``): per declared criterion, the
  smallest field count at which every eligible sample was adequately
  represented (``protocol_reliability``); ❌ when no criterion reached that in
  any listed count, ⚠️ when some criteria did not or fewer than 5
  independent units stand behind the counts, ✅ otherwise. Reliability curves
  alone (no protocol): ✅, per group only.
- **5. What did the extra fields buy?** (``bought_status``): ⚠️ when some
  replayed acquisitions never reliably preserved everything wanted, ✅
  otherwise.
- **6. Which fields should I look at?** (``inspect_status``): ⚠️ when fields
  are listed, ✅ when none is.
- **7. More fields or more samples?** (``effort_status``): ⚠️ when the
  comparison cannot tell in some stratum (too few samples), ✅ otherwise.
- **8. Could I have sampled more efficiently?** (``order_status``): ⚠️ when
  the orders differ by more than one field in some group, or some replayed
  runs never reached the criterion; ✅ otherwise.
- **9. Do the conditions differ?** (``conditions_status``): ⚠️ when some
  contrast has no randomization test (one unit on a side); ✅ otherwise.

A text added later to a question (``workflows._document.with_text``) may raise
its status; it never lowers one. These rules are the first ones: each
question's restyle may refine its own, here.
"""

from __future__ import annotations

import statistics
from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from .._style import Status, among, worst
from .._wording import count, listing, number, percent
from ..schema import unit_plural
from ..sufficiency_short import rule_tallies
from .adequacy import ALL_CRITERIA
from .calibration import PREMATURE_CAUTION
from .report_sampling import (
    clustered_states,
    largest_sources,
    profile_answer,
    rule_held,
    skipped_strata,
    unstable_labels,
)

# Above this false-alarm rate of the declared reference, the control answer asks the reader
# to look (a provisional preset, as minimum_detection).
FALSE_ALARM_CAUTION = 0.1
# Fewer independent units than this make a protocol's field count uncertain.
FEW_UNITS = 5

Answer = tuple[Status, str]


def data_check_status(overview: Any) -> Answer:
    """The data checks of the overview: none run, no warning, or the warnings' titles."""
    if overview is None or overview.settings.get("data_checked") is False:
        return Status.NOT_ASSESSED, "data checks not run (check_tables)"
    warnings = list(overview.warnings)
    if not warnings:
        return Status.OK, "data checks: no warning"
    titles = listing([problem.title for problem in warnings], limit=2)
    return Status.CAUTION, f"data checks: {count(len(warnings), 'warning')} ({titles})"


def integrity_status(integrity: Any) -> Answer:
    """The data-integrity checks: not run, nothing found, or the fields they name."""
    if integrity is None:
        return Status.NOT_ASSESSED, "integrity not checked (integrity_checks)"
    if not integrity.findings.empty:
        fields = integrity.fields_involved()
        found = count(len(integrity.findings), "finding")
        return (
            Status.PROBLEM,
            f"integrity: {found} in {count(len(fields), 'field')}; check them first",
        )
    checks = integrity.checks
    if len(checks) and (checks["status"] == "not applicable").all():
        return Status.NOT_ASSESSED, "integrity checks not applicable"
    return Status.OK, "integrity: nothing found"


def overview_status(overview: Any, integrity: Any = None) -> Answer:
    """The opening: the worse of the data checks and the integrity checks."""
    parts = [data_check_status(overview), integrity_status(integrity)]
    return worst(s for s, _ in parts), "; ".join(text for _, text in parts)


def unit_status(variability: Any, object_rules: Any, descriptor_spread: Any) -> Answer:
    """Foundation: would object counts have held with whole fields? (see the module)."""
    if variability is None and object_rules is None and descriptor_spread is None:
        return Status.NOT_ASSESSED, "not assessed"
    status, parts = Status.OK, []
    if variability is not None:
        groups = variability.groups
        ok = groups[groups["status"] == "ok"]
        if len(ok):
            words = unit_plural(variability.settings.get("observation_unit", "object"))
            effect = statistics.median(ok["design_effect"].astype(float))
            parts.append(
                f"a mean from whole fields varies {effect:.1f}x as much as from random "
                f"{words} (median)"
            )
        if len(ok) < len(groups):
            status = Status.CAUTION
            parts.append(
                f"variability not estimated in {among(len(groups) - len(ok), len(groups))}"
            )
    if object_rules is not None:
        kept, total = rule_held(object_rules)
        if not total:
            parts.append("JCB count rule not applicable")
        elif kept == total:
            parts.append(f"the JCB count rule still held with whole fields in {among(kept, total)}")
        else:
            status = Status.CAUTION
            missed = among(total - kept, total)
            parts.append(f"the JCB count rule missed with whole fields in {missed}")
    if descriptor_spread is not None:
        rows = descriptor_spread.at_threshold
        keys = list(descriptor_spread.group_cols)
        if len(rows) and keys:
            total = len(rows[keys].drop_duplicates())
            never = rows[rows["objects_to_match"].isna()][keys].drop_duplicates()
            missed = len(never)
        else:
            total, missed = 1, int(len(rows) > 0 and bool(rows["objects_to_match"].isna().any()))
        if missed:
            status = Status.CAUTION
            parts.append(f"descriptors from whole fields never settled in {among(missed, total)}")
    return status, "; ".join(parts) or "assessed"


def _rate(controls: Any, kind: str) -> float | None:
    rates = controls.pooled.set_index("candidate")["rate"]
    return None if kind not in rates.index or pd.isna(rates[kind]) else float(rates[kind])


def controls_status(controls: Any, minimum_detection: float = 0.9) -> Answer:
    """1. Can these features detect what I care about? (see the module)."""
    if controls is None:
        return Status.NOT_ASSESSED, "no positive control declared"
    if controls.status == "not assessed":
        declared = controls.settings.get("reason") != "none"
        if declared:
            return Status.CAUTION, "a control was declared, but could not be judged"
        return Status.NOT_ASSESSED, "no positive control declared"
    words = unit_plural(controls.settings.get("observation_unit", "object"))
    planted = (controls.settings.get("control") or {}).get("planted_objects")
    kind = "planted" if planted and _rate(controls, "planted") is not None else "positive"
    detection = _rate(controls, kind)
    what = f"{planted} planted {words}" if kind == "planted" else "positive-control fields"
    if detection is None:
        return Status.CAUTION, f"{what}: not judged"
    headline = f"{what} recognized in {percent(detection)} of checks"
    if detection < float(minimum_detection):
        return Status.PROBLEM, f"weak representation: {headline}"
    false_alarm = _rate(controls, "reference")
    status = Status.OK
    if false_alarm is None:
        status, headline = Status.CAUTION, f"{headline}; false alarms not assessed"
    else:
        headline += f", false alarms in {percent(false_alarm)}"
        if false_alarm > FALSE_ALARM_CAUTION:
            status = Status.CAUTION
    if controls.settings.get("skipped_strata"):
        status = Status.CAUTION
        headline += f"; {skipped_strata(controls)}"
    return status, headline


def _premature(calibration: Any) -> float | None:
    """The largest share of stops where a criterion was not preserved, of any group and kind."""
    if calibration is None or calibration.table.empty:
        return None
    shares = calibration.table["not_preserved_at_stop"].dropna()
    return float(shares.max()) if len(shares) else None


def enough_status(sufficiency: Any, calibration: Any = None) -> Answer:
    """2. Did I image enough? (see the module)."""
    tallies = rule_tallies(sufficiency) if sufficiency is not None else []
    premature = _premature(calibration)
    if not tallies and premature is None:
        return Status.NOT_ASSESSED, "not assessed"
    late = premature is not None and premature > PREMATURE_CAUTION
    held = (
        f"held-back check: up to {percent(premature)} of stops not preserved"
        if premature is not None
        else ""
    )
    data = [t for t in tallies if t["data_rule"]]
    if not data:
        if not tallies:
            return (Status.CAUTION if late else Status.OK), held
        return Status.CAUTION, "only fixed amounts of effort, which say nothing about the data"
    groups = max(t["groups"] for t in data)
    everywhere = [t for t in data if t["met"] == t["groups"]]
    nowhere = [t for t in data if t["met"] == 0]
    if len(nowhere) == len(data):
        return Status.PROBLEM, f"no rule met in any of {among(groups, groups)}"
    parts = [
        f"every rule met in {among(groups, groups)}"
        if len(everywhere) == len(data)
        else f"{len(everywhere)} of {count(len(data), 'rule')} met in {among(groups, groups)}"
    ]
    parts += [f"{t['label']} in none" for t in nowhere if len(nowhere) <= 2]
    cautions = any(t["warned"] or t["flagged"] or t["lost"] for t in data)
    status = Status.OK if len(everywhere) == len(data) and not cautions else Status.CAUTION
    if late:
        status = Status.CAUTION
    if cautions:
        parts.append("cautions below")
    if held and late:
        parts.append(held)
    return status, "; ".join(parts)


def weak_status(answer: Answer) -> Answer:
    """Questions 2, 4 and 5 under a weak representation: ❌, the headline saying why first."""
    status, headline = answer
    return worst([status, Status.PROBLEM]), f"weak representation; {headline}"


def profile_status(
    profile: pd.DataFrame, clustering: Sequence[Any] = (), unusual: Any = None
) -> Answer:
    """3. What is limiting my sampling? (see the module)."""
    _, clustered, seen = clustered_states(clustering)
    if not len(profile) and not seen:
        return Status.NOT_ASSESSED, "not assessed"
    parts = profile_answer(profile, clustering, unusual)
    _, uncertain = largest_sources(profile)
    status = Status.CAUTION if clustered or uncertain or unstable_labels(unusual) else Status.OK
    return status, "; ".join(parts) or f"{count(len(profile), 'quantity', 'quantities')} measured"


def next_time_status(reliability: Any, protocol: Any) -> Answer:
    """4. How many fields should I image next time? (see the module)."""
    if protocol is None:
        if reliability is None:
            return Status.NOT_ASSESSED, "not assessed"
        return Status.OK, "reliability per group; no protocol across samples"
    labels = protocol.settings.get("labels", {})
    table = protocol.table
    parts, reached = [], 0
    criteria = list(dict.fromkeys(table["criterion"])) if len(table) else []
    for criterion in criteria:
        rows = table[table["criterion"].eq(criterion) & (table["eligible"] > 0)]
        every = rows[rows["adequate"] == rows["eligible"]]
        name = labels.get(criterion, criterion)
        if len(every):
            reached += 1
            parts.append(f"{name}: {count(int(every['fovs'].min()), 'field')}")
        else:
            parts.append(f"{name}: not reached")
    if not criteria:
        return Status.NOT_ASSESSED, "not assessed"
    units = int(table["units"].min()) if "units" in table and len(table) else 0
    if reached == 0:
        status = Status.PROBLEM
    elif reached < len(criteria) or units < FEW_UNITS:
        status = Status.CAUTION
    else:
        status = Status.OK
    few = f" (from {count(units, 'independent unit')})" if units < FEW_UNITS else ""
    return status, "; ".join(parts) + few


def bought_status(adequacy: Any) -> Answer:
    """5. What did the extra fields buy? (see the module)."""
    if adequacy is None:
        return Status.NOT_ASSESSED, "not assessed"
    rows = adequacy.realized[adequacy.realized["criterion"] == ALL_CRITERIA]
    if rows.empty:
        return Status.NOT_ASSESSED, "not assessed"
    sustained = rows["sustained_realized"].dropna().astype(float)
    if sustained.empty:
        return Status.CAUTION, "everything wanted was never reliably preserved"
    head = f"everything wanted reliably preserved from field {number(sustained.median())} (median)"
    if len(sustained) < len(rows):
        return Status.CAUTION, f"{head}; never in {percent(1 - len(sustained) / len(rows))} of runs"
    return Status.OK, head


def inspect_status(inspection: Any) -> Answer:
    """6. Which fields should I look at? (see the module)."""
    if inspection is None:
        return Status.NOT_ASSESSED, "not assessed"
    listed = inspection.fovs
    if listed.empty:
        return Status.OK, "no field to reopen"
    keys = [*inspection.group_cols]
    fields = listed.drop_duplicates([*keys, *[c for c in listed.columns if c == "image_file"]])
    groups = len(listed[keys].drop_duplicates()) if keys else 1
    return Status.CAUTION, f"{count(len(fields), 'field')} to reopen in {count(groups, 'group')}"


def effort_status(effort: Any) -> Answer:
    """7. More fields or more samples? (see the module)."""
    if effort is None or effort.comparisons.empty:
        return Status.NOT_ASSESSED, "not assessed"
    strata = effort.comparisons.groupby("stratum", sort=False)["resolved"].any()
    resolved, total = int(strata.sum()), len(strata)
    if resolved == total:
        return Status.OK, f"a better allocation resolved in {among(resolved, total, 'stratum')}"
    units = count(int(effort.comparisons["units"].min()), "sample")
    return Status.CAUTION, f"cannot tell in {among(total - resolved, total, 'stratum')} ({units})"


def order_status(efficiency: Any) -> Answer:
    """8. Could I have sampled more efficiently? (see the module)."""
    if efficiency is None or efficiency.table.empty:
        return Status.NOT_ASSESSED, "not assessed"
    table, keys = efficiency.table, list(efficiency.group_cols)
    parts = table.groupby(keys, sort=False, dropna=False) if keys else [((), table)]
    differ, groups = 0, 0
    for _, rows in parts:
        groups += 1
        fovs = rows["median_fovs"].dropna().astype(float)
        differ += bool(len(fovs) and fovs.max() - fovs.min() > 1)
    unreached = bool((table["reached"].fillna(0) < 1).any())
    text = (
        f"the orders differ by more than one field in {among(differ, groups)}"
        if differ
        else "the orders do not differ by more than one field"
    )
    if unreached:
        text += "; some runs never reached the criterion"
    return (Status.CAUTION if differ or unreached else Status.OK), text


def conditions_status(contrast: Any, ranking: Any) -> Answer:
    """9. Do the conditions differ, and how confidently? (see the module)."""
    if contrast is None and ranking is None:
        return Status.NOT_ASSESSED, "not assessed"
    status, parts = Status.OK, []
    if contrast is not None:
        table = contrast.table
        untested = int((table["status"] != "ok").sum()) if len(table) else 0
        parts.append(f"{count(len(table), 'contrast')}")
        if untested:
            status = Status.CAUTION
            parts.append(f"no randomization test in {untested} (too few units on a side)")
    if ranking is not None and len(ranking.ranking):
        passing = int((ranking.ranking["passing"] > 0).sum())
        parts.append(f"{count(passing, 'feature')} pass the gate")
    return status, "; ".join(parts)


def question_statuses(
    results: Mapping[str, Any], profile: pd.DataFrame, minimum_detection: float
) -> dict[str, Answer]:
    """Every question's ``(Status, headline)`` from the report's results (see the module)."""
    get = results.get
    return {
        "unit": unit_status(get("variability"), get("object_rules"), get("descriptor_spread")),
        "controls": controls_status(get("controls"), minimum_detection),
        "enough": enough_status(get("sufficiency"), get("calibration")),
        "profile": profile_status(profile, [get("clustering")], get("unusual")),
        "next_time": next_time_status(get("reliability"), get("protocol")),
        "bought": bought_status(get("adequacy")),
        "inspect": inspect_status(get("inspection")),
        "effort": effort_status(get("effort")),
        "order": order_status(get("efficiency")),
        "conditions": conditions_status(get("contrast"), get("ranking")),
    }


__all__ = [
    "FALSE_ALARM_CAUTION",
    "FEW_UNITS",
    "bought_status",
    "conditions_status",
    "controls_status",
    "data_check_status",
    "effort_status",
    "enough_status",
    "inspect_status",
    "integrity_status",
    "next_time_status",
    "order_status",
    "overview_status",
    "profile_status",
    "question_statuses",
    "unit_status",
    "weak_status",
]
