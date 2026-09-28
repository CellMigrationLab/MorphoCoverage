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
  fewest listed fields from which every sample that could be judged was
  adequately represented at every larger count, in every stratum
  (``protocol_text.served_from``); the headline gives the largest of these
  over the criteria that reached one. ❌ when no criterion reached one and
  some samples needed more fields than could be judged; ⚠️ when some criteria
  did not reach one (not reached; not shown: every unserved sample capped
  by its held-back fields; or not judged at any depth) or fewer than
  ``FEW_UNITS`` (5) independent units stand behind the counts; ✅
  otherwise. Reliability curves alone (no protocol): the fewest listed
  fields from which ``RELIABLE_SHARE`` (90%) of random acquisitions were
  enough in every group, per criterion; ❌ when no
  criterion reached it, ⚠️ otherwise, since nothing was checked across
  samples (a count that served these samples is not yet a protocol).
- **5. What did the extra fields buy?** (``bought_status``): over the
  replayed acquisitions that could be judged, ⚠️ when some never had
  everything wanted reliably confirmed by held-back fields, ✅ otherwise;
  the headline gives the median field from which it was, and the field by
  which 90% of runs had it.
- **6. Which fields should I look at?** (``inspect_status``): ⚠️ when fields
  are listed (the headline counts them, their groups and each reason), or
  when none is but some reason could not be judged in some group; ➖ when no
  reason could be judged anywhere; ✅ when none is listed and every reason
  turned on was judged.
- **7. More fields or more samples?** (``effort_status``): ⚠️ when the
  comparison cannot tell in some stratum (too few samples, or no
  allocation's interval excludes no change), ✅ when a better split of the
  same total of fields is resolved in every stratum. The headline names the
  samples per stratum and in how many strata most of the variance of the
  mean lies between samples.
- **8. Could I have sampled more efficiently?** (``order_status``): ⚠️ when
  the orders differ by more than one field in some group, or some replayed
  runs never reached the criterion; ✅ otherwise. The headline says whether
  another order than the recorded one needed fewer fields, and in how many
  groups (``order_text.group_comparison``).
- **9. Do the conditions differ?** (``conditions_status``): ⚠️ when some
  contrast has no randomization test (one technical repeat or too few fields
  on a side, wells unknown, no control), when a condition differs from the
  control at p <= 0.05 in opposite directions in different repeats, or when
  the feature ranking could test no contrast; ✅ otherwise. The headline gives
  the contrasts, the range of |d| on field means, how many tested contrasts
  differ after the Bonferroni correction and how many features pass the
  ranking's gate.
- **optimise's rule check** (``calibration_status``, P5 and P6: how reliable
  is each stopping rule?): ⚠️ when some rule stopped where a criterion was
  not preserved in held-back fields in more than ``PREMATURE_CAUTION`` of the
  acquisitions of a group and order kind, ✅ otherwise, ➖ with no replayed
  stop.

A text added later to a question (``workflows._document.with_text``) may raise
its status; it never lowers one. These rules are the first ones: each
question's restyle may refine its own, here.
"""

from __future__ import annotations

import statistics
from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from .._style import Status, among, span, worst
from .._wording import count, listing, percent
from ..sampling_unit._units import opposite_signs
from ..sampling_unit.contrast_text import P_ALPHA
from ..schema import unit_plural
from ..sufficiency_short import rule_tallies
from ._text import rule_names
from .calibration import PREMATURE_CAUTION
from .inspection_text import reason_counts, tally
from .order_text import group_comparison
from .report_sampling import (
    clustered_states,
    largest_sources,
    profile_answer,
    rule_held,
    skipped_strata,
    unstable_labels,
)
from .report_status_plan import (
    FEW_UNITS,
    RELIABLE_SHARE,
    bought_status,
    effort_status,
    next_time_status,
)

# Above this false-alarm rate of the declared reference, the control answer asks the reader
# to look (a provisional preset, as minimum_detection).
FALSE_ALARM_CAUTION = 0.1
# The reasons of question 6 in a word or two, for its headline.
_REASONS = {"count": "unusual count", "neighbours": "unlike neighbours"}

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
    if integrity.status is Status.NOT_ASSESSED:  # no check applied to these features
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


def calibration_status(calibration: Any) -> Answer:
    """``optimise``'s rule check (P5, P6): how often each rule's stop lost a criterion.

    ➖ without a calibration or with nothing replayed; ⚠️ when some rule stopped
    where a criterion was not preserved in held-back fields in more than
    ``PREMATURE_CAUTION`` of the acquisitions of a group and order kind; ✅
    otherwise. The headline names the rule with the largest share.
    """
    premature = _premature(calibration)
    if premature is None:
        return Status.NOT_ASSESSED, "not assessed (no replayed stop)"
    table = calibration.table
    worst_row = table.loc[table["not_preserved_at_stop"].astype(float).idxmax()]
    names = rule_names(calibration.settings)
    rule = names.get(worst_row["config_id"], worst_row["strategy"])
    kinds = {"recorded": "the recorded order", "permuted": "random orders"}
    kind = kinds.get(worst_row["order"], f"{worst_row['order']} orders")
    if premature > PREMATURE_CAUTION:
        text = (
            f"{rule}: in some group up to {percent(premature)} of its stops did not preserve "
            f"a criterion in held-back fields ({kind})"
        )
        return Status.CAUTION, text
    text = (
        f"at most {percent(premature)} of any rule's stops did not preserve a criterion in "
        "held-back fields"
    )
    return Status.OK, text


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


def inspect_status(inspection: Any) -> Answer:
    """6. Which fields should I look at? (see the module)."""
    if inspection is None:
        return Status.NOT_ASSESSED, "not assessed"
    settings = inspection.settings
    skipped = sum(
        len(settings.get(key, []))
        for key in ("skipped_summaries", "skipped_neighbours", "skipped_novelty")
    )
    if inspection.fovs.empty:
        if settings.get("checked") == []:
            return Status.NOT_ASSESSED, "no reason could be judged in any group"
        if skipped:
            return Status.CAUTION, "no field stands out; some reasons not judged in some groups"
        return Status.OK, "no field stands out"
    fields, groups = tally(inspection)
    reasons = ", ".join(
        f"{_REASONS.get(reason, reason)} in {n}" for reason, n in reason_counts(inspection)
    )
    return (
        Status.CAUTION,
        f"{count(fields, 'field')} to reopen in {count(groups, 'group')} ({reasons})",
    )


def order_status(efficiency: Any) -> Answer:
    """8. Could I have sampled more efficiently? (see the module)."""
    if efficiency is None or efficiency.table.empty:
        return Status.NOT_ASSESSED, "not assessed"
    compared = group_comparison(efficiency)
    n = len(compared)
    differ = [c for c in compared if c["verdict"] == "differ"]
    slower = [c for c in compared if c["recorded_slower"]]
    same = sum(c["verdict"] == "same" for c in compared)
    recorded = any("recorded" in c["medians"] for c in compared)
    if slower:
        gap = max(c["gap"] for c in slower)
        up_to = f" (up to {count(gap, 'field')})" if gap > 1 else ""
        text = f"another order needed fewer fields in {among(len(slower), n)}{up_to}"
    elif recorded:
        text = "no order beat the recorded one by more than one field"
        if differ:
            text += f"; orders differ in {among(len(differ), n)}"
    elif differ:
        text = f"the orders differ by more than one field in {among(len(differ), n)}"
    else:
        text = "no order better by more than one field"
        if same < n:
            text += f" in {among(same, n)}"
    unreached = bool((efficiency.table["reached"].fillna(0) < 1).any())
    if unreached:
        text += "; some runs never reliably enough"
    return (Status.CAUTION if differ or unreached else Status.OK), text


_UNTESTED = {
    "only one technical repeat": "one technical repeat on a side",
    "fewer than 2 fields": "too few fields on a side",
    "fields of one dish or well": "the fields' wells unknown",
    "no control group": "no control fields",
}


def _untested_words(statuses: Sequence[str]) -> str:
    """Why contrasts have no test, in a few words."""
    words = {
        next((short for start, short in _UNTESTED.items() if s.startswith(start)), s)
        for s in statuses
    }
    return words.pop() if len(words) == 1 else "too few units on a side"


def conditions_status(contrast: Any, ranking: Any) -> Answer:
    """9. Do the conditions differ, and how confidently? (see the module)."""
    if contrast is None and ranking is None:
        return Status.NOT_ASSESSED, "not assessed"
    status, parts = Status.OK, []
    if contrast is not None and len(contrast.table):
        table = contrast.table
        sizes = table["d_fields"].abs().dropna()
        size = f", |d| on field means {span(sizes)}" if len(sizes) else ""
        parts.append(f"{count(len(table), 'contrast')}{size}")
        tested = table[table["status"] == "ok"]
        untested = table[table["status"] != "ok"]
        if len(tested):
            differ = int((tested["p_bonferroni"] <= 0.05).sum())
            parts.append(f"{differ} of {len(tested)} tested differ (Bonferroni p <= 0.05)")
        if len(untested):
            status = Status.CAUTION
            which = "untested" if not len(tested) else f"{len(untested)} untested"
            parts.append(f"{which} ({_untested_words(list(untested['status']))})")
        passing = tested[tested["p_value"] <= P_ALPHA]
        if opposite_signs(passing, ["condition"], contrast.settings["repeat_col"], str):
            status = Status.CAUTION
            parts.append("opposite signs between repeats")
    elif contrast is not None:
        parts.append("no condition to compare with the control")
    if ranking is not None and len(ranking.ranking):
        passing = int(ranking.ranking["score"].notna().sum())
        parts.append(f"{passing} of {count(len(ranking.ranking), 'feature')} pass the gate")
        if not ranking.ranking["tests"].any():
            status = Status.CAUTION
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
    "RELIABLE_SHARE",
    "bought_status",
    "calibration_status",
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
