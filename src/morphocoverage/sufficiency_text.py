"""The sentences of the sufficiency report: how each answer is worded.

``sufficiency`` computes the tables; this module turns their rows into the
plain sentences a microscopist reads, so the wording can be read and changed
without touching the computation. Every sentence names the criterion, its
threshold and the assumptions it rests on, and never calls a sample adequate
for a target it did not measure.

Shared-private: ``_policies`` (``sufficiency_tables``), ``_enough_answer`` and
``_next_fov_answer`` (``sufficiency``).
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from ._style import rule_label
from .bounds import EXTRA_TRIALS_LIMIT
from ._wording import count, listing, number, percent
from .evidence import Direction
from .results import StrategyResult
from .strategies.support_novelty import SupportNoveltyEstimator
from .transforms import RobustStandardize, Standardize


def _policy_text(policy: Mapping[str, Any], result: StrategyResult) -> str:
    """One decision policy as a readable rule, such as ``averaging_cv <= 0.05``."""
    parameters = policy["parameters"]
    quantity = result.evidence.quantities.get(parameters.get("quantity"))
    if quantity is not None and set(parameters) == {"quantity", "threshold"}:
        better = "<=" if quantity.direction is Direction.LOWER_IS_BETTER else ">="
        return f"{parameters['quantity']} {better} {parameters['threshold']:g}"
    if {"effort", "target"} <= set(parameters):
        return f"{parameters['effort']} >= {parameters['target']:g}"
    return f"{policy['name']} {json.dumps(parameters, sort_keys=True)}"


def _policies(result: StrategyResult) -> str:
    """The decision policies and their settings, compactly."""
    policies = result.configuration.get("policies", [])
    return "; ".join(_policy_text(policy, result) for policy in policies)


def _stops_clause(stops: Mapping[str, Any]) -> str:
    """Where one kind of replayed order (recorded or random) first met the criterion.

    The median and its 10-90% range are over the orders that met it, so the
    clause says how many did: "in 2 of 10 random orders, first met after 14.5
    FOVs", never a median that reads as if every order had stopped.
    """
    orders = int(stops["orders"])
    met = orders - int(stops["never_stopped"])
    recorded = stops.get("order_kind") == "recorded"
    kind = "recorded order" if recorded else "random order"
    where = f"the {kind}" if orders == 1 else None
    if not met:
        return f"not met at any point in {where or count(orders, kind)}"
    first = f"first met after {count(stops['stop_fov_median'], 'FOV')}"
    if met > 1:
        se = stops.get("stop_fov_median_mc_se")
        se = "" if se is None or pd.isna(se) else f"; Monte Carlo SE {number(se)}"
        first += (
            f" (median{se}; 10-90%: {number(stops['stop_fov_p10'])}-"
            f"{number(stops['stop_fov_p90'])})"
        )
    share = f"all {count(orders, kind)}" if met == orders else f"{met} of {count(orders, kind)}"
    return f"in {where or share}, {first}"


def _enough_answer(final: Mapping[str, Any], stops: Sequence[Mapping[str, Any]]) -> str:
    """One criterion's answer to 'did we sample enough?' as a sentence.

    ``stops`` holds one row per kind of replayed order (the recorded order, the
    random orders), each given in its own clause: pooled, a recorded order that
    drifts would hide among the random ones.
    """
    criterion = rule_label(final, detail="full")
    acquired = final["n_fovs"]
    if pd.isna(final["verdict"]):
        return f"{criterion}: undefined for these data ({final['undefined_reason']})."
    # A fixed amount of effort says its target was reached, not that the sample was
    # enough; read from the rule's structure (sufficiency_tables.is_effort_rule).
    effort_rule = bool(final.get("effort_rule", False))
    if final["verdict"] == "continue":
        head = f"{criterion}: no, not met by the {acquired} acquired FOVs"
        reason = final.get("insufficient_reason")
        if isinstance(reason, str) and reason:
            head += f" (it cannot be judged yet: {reason})"
    elif effort_rule:
        head = f"{criterion}: its target was reached by the {acquired} acquired FOVs"
    else:
        head = f"{criterion}: yes, its threshold was met by the {acquired} acquired FOVs"
    if final.get("assumes_spatial_uniformity"):
        head += " (treats observations as independent draws: assumes a spatially uniform sample)"
    # The rule's own caveats (FOV size related to FOV mean, SD above 50% CV) follow its answer.
    warnings = final.get("warnings")
    caution = f" Caution: {warnings}." if isinstance(warnings, str) and warnings else ""
    ever_met = any(int(row["orders"]) > int(row["never_stopped"]) for row in stops)
    # Beside every stop, what it rests on: the order check, or the random-sample assumption.
    caution += _order_sentence(final) if final["verdict"] == "stop" or ever_met else ""
    if not stops:
        return f"{head}.{caution}"
    clauses = "; ".join(_stops_clause(row) for row in stops)
    lost = (
        "; it was met and then lost as more FOVs arrived"
        if final["verdict"] == "continue" and ever_met
        else ""
    )
    return f"{head}; {clauses}{lost}.{caution}"


def _order_sentence(final: Mapping[str, Any]) -> str:
    """The order caution of a rule's stop (``sufficiency_tables``) as a sentence, or nothing."""
    text = final.get("order_caution")
    if not isinstance(text, str) or not text:
        return ""
    if text.startswith("treats"):
        return f" The stop {text}."
    return f" {text[0].upper()}{text[1:]}."


def novelty_definition(
    novelty: SupportNoveltyEstimator, feature_cols: Sequence[str] | None = None
) -> str:
    """What counts as a FOV showing something new, in words; with ``feature_cols``,
    also the columns it is judged on (the phenotype geometry's provenance).

    It says how compact a state must be: its observations must be linked by
    steps within the distance, and after a standardization two observations
    of a state that varies by one SD per feature sit about ``sqrt(2 d)`` apart
    for ``d`` features, so a state must be tighter than that to count.
    """
    m = novelty.min_novel_observations
    space = "raw feature units" if novelty.transform is None else novelty.transform.name
    cells = (
        "at least 1 observation"
        if m == 1
        else f"a cluster of at least {m} similar observations (each linked to another by a "
        "step within the distance, so a more spread-out state is not counted)"
    )
    text = f"{cells} with {_other_fovs(novelty.min_other_fovs)} within distance "
    text += f"{novelty.epsilon:g} ({space})"
    if feature_cols:
        names = list(feature_cols)
        text += f", on {count(len(names), 'feature')} ({listing(names, limit=3)})"
        if m > 1 and isinstance(novelty.transform, (Standardize, RobustStandardize)):
            text += (
                f"; two observations of a state that varies by one SD per feature sit about "
                f"{math.sqrt(2 * len(names)):.1f} apart"
            )
    return text


def _other_fovs(min_other_fovs: int) -> str:
    """``no other FOV`` (for 1), ``fewer than 3 other FOVs``: the support that makes a state new."""
    if min_other_fovs == 1:
        return "no other FOV"
    return f"fewer than {count(min_other_fovs, 'other FOV')}"


def _more_fovs(n: int) -> str:
    """``1 more FOV``, ``3 more FOVs``."""
    return count(n, "more FOV")


def _several_states(row: Mapping[str, Any], fewest: int) -> str:
    """When the FOVs seen in few could hold two or more recurrent states, each one's need."""
    if int(row["fovs_with_state_in_few"]) < 2 * fewest:
        return ""
    return (
        " They may hold several different states, each rarer: one seen in only "
        f"{fewest} FOVs needs about {int(row['fovs_to_see_fewest'])}."
    )


def _next_fov_answer(
    row: Mapping[str, Any],
    target: float | None,
    calibrated: Mapping[str, Any] | None,
    min_other_fovs: int = 1,
) -> str:
    """Question 2 for one group, as lines of text.

    ``min_other_fovs`` is the novelty definition's: a FOV counts as new when a
    state in it is shown by fewer than that many other FOVs (none, for 1).
    Every bound names what it assumes: FOVs that are independent,
    representative draws (neighbouring fields of a raster over patchy states
    are not, and fields of one technical repeat are not independent of each
    other), and a bound that holds at a number of FOVs fixed in advance, not
    at the first FOV where it passes.
    """
    if pd.isna(row["chance_new"]):
        return f"Not estimable: {row['reason']}."
    confidence = percent(row["confidence"])
    shows = "shows" if min_other_fovs == 1 else "show"
    chance = (
        f"Chance: {percent(row['chance_new'])} (nominal {confidence} upper bound "
        f"{percent(row['chance_new_at_most'])}, approximate: the leave-one-FOV-out outcomes "
        f"share their reference FOVs, and it treats FOVs as independent, representative "
        f"draws, so neighbouring fields acquired in a raster over patchy states make it too "
        f"low); {row['fovs_with_new']} of {row['fovs_imaged']} FOVs "
        f"showed a state that {_other_fovs(min_other_fovs)} {shows}."
    )
    empty = int(row.get("fovs_without_observations", 0) or 0)
    counted = (
        f" The count includes {count(empty, 'FOV')} without observations, "
        "which cannot show a state."
        if empty
        else ""
    )
    m = row.get("min_novel_observations")
    sparse = (
        f" A state carried by fewer than {int(m)} similar observations per FOV is never "
        "counted, however many FOVs hold it."
        if m is not None and not pd.isna(m) and int(m) > 1
        else ""
    )
    # The zero-success bound is about -ln(1 - confidence) / k: 3 / k at 95%.
    successes = -math.log(1 - row["confidence"])
    unseen = (
        f"Not yet seen: a state this definition detects but no FOV has shown is in at most "
        f"{percent(row['unseen_state_max_frequency'])} of FOVs ({confidence} confidence, about "
        f"{successes:.2g} / {row['fovs_imaged']}), if FOVs are independent, representative "
        f"draws. Rarer states can remain.{sparse}{counted}"
    )
    lines = [chance, unseen, *_technical_repeats_line(row, confidence)]
    few = row.get("few_fovs")
    if few is not None and not pd.isna(few):
        seen_in_few = row["fovs_with_state_in_few"]
        if pd.isna(seen_in_few):
            lines.append(f"Seen in few FOVs: not estimable ({row['few_reason']}).")
        else:
            fewest = int(row["fewest_recurrent_fovs"])
            span = f"{fewest} FOVs" if fewest == int(few) else f"{fewest} to {int(few)} FOVs"
            lines.append(
                f"Seen in few FOVs: {int(seen_in_few)} of {row['fovs_imaged']} FOVs hold a "
                f"state that other FOVs show too, but fewer than {int(few)} of them (a state "
                f"seen in {span}; FOVs counted as new above are not included). Such a state "
                "is not new to the next FOV but is not established either; it counts as "
                "recurrent above, so look at those FOVs before calling the sample settled."
            )
            if int(seen_in_few):
                lines.append(
                    f"Planning for such a state: these FOVs together are "
                    f"{percent(row['few_state_incidence'])} of FOVs ({confidence} interval "
                    f"{percent(row['few_state_incidence_low'])}-"
                    f"{percent(row['few_state_incidence_high'])}); to see a state that "
                    f"common at least once with {confidence} probability, image about "
                    f"{int(row['fovs_to_see_again'])} independent representative FOVs, up "
                    f"to {int(row['fovs_to_see_again_low_end'])} at the low end of the "
                    f"interval.{_several_states(row, fewest)} Fields that cluster the state "
                    "need more."
                )
    if target is not None:
        extra = row["extra_fovs_to_rule_out_target"]
        rule_out = f"To rule out unseen states in more than {percent(target)} of FOVs"
        lines.append(
            f"{rule_out}: more than {_more_fovs(EXTRA_TRIALS_LIMIT)} with nothing new."
            if pd.isna(extra)
            else f"{rule_out}: already reached."
            if extra == 0
            else f"{rule_out}: at least {_more_fovs(extra)} with nothing new."
        )
    if target is not None:
        goal = f"To bring the nominal {confidence} bound for the next FOV below {percent(target)}"
        if row["chance_new_at_most"] <= target:
            lines.append(f"{goal}: already reached.")
        else:
            best = row["extra_fovs_if_nothing_new"]
            rate = row["extra_fovs_at_current_rate"]
            at_rate = (
                "never at the current rate of new FOVs"
                if pd.isna(rate)
                else f"{_more_fovs(rate)} if new FOVs keep appearing at the current rate"
            )
            least = (
                f"more than {_more_fovs(EXTRA_TRIALS_LIMIT)}"
                if pd.isna(best)
                else f"at least {_more_fovs(best)}"
            )
            lines.append(f"{goal}: {least} if none shows anything new; {at_rate}.")
        lines.append(
            f"The bound holds at a number of FOVs fixed in advance, not at the first FOV where "
            f"it passes: stopping there is premature more often than "
            f"{percent(1 - row['confidence'])} of the time."
        )
    if calibrated is not None:
        predicted, actual = percent(calibrated["predicted"]), percent(calibrated["actual"])
        held = percent(calibrated["bound_held"])
        lines.append(
            f"Check on this dataset: predicting from {int(calibrated['fovs_imaged'])} random FOVs, "
            f"predicted {predicted}, actual {actual}; "
            f"the bound held in {held} of {int(calibrated['draws'])} draws (random subsets "
            "break the acquisition order and the neighbours, so this check cannot reveal "
            "neighbour correlation)."
        )
    return "\n".join(lines)


def _technical_repeats_line(row: Mapping[str, Any], confidence: str) -> list[str]:
    """When the group pools technical repeats, the unseen-state bound over them (P3-8).

    Fields of one technical repeat (a well, a dish) share its preparation, so
    a state held by some technical repeats is clustered in their fields: the
    per-FOV bound above then fails far more often than its level, while ``n``
    technical repeats none of which showed the state bound its share of
    technical repeats as ``n`` independent draws.
    """
    n = row.get("technical_repeats")
    share = row.get("unseen_state_max_technical_share")
    if n is None or pd.isna(n) or int(n) < 2 or share is None or pd.isna(share):
        return []
    return [
        (
            f"Fields of one technical repeat are not independent draws: this group pools "
            f"{int(n)} technical repeats, so the FOV bound above is optimistic; a state none "
            f"of them showed is in at most {percent(share)} of technical repeats "
            f"({confidence} confidence, {int(n)} independent technical repeats)."
        )
    ]


__all__ = ["novelty_definition"]
