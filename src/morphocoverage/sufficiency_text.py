"""The sentences of the sufficiency report: how each answer is worded.

``sufficiency`` computes the tables; this module turns their rows into the
plain sentences a microscopist reads, so the wording can be read and changed
without touching the computation. Every sentence names the criterion, its
threshold and the assumptions it rests on, and never calls a sample adequate
for a target it did not measure.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from ._wording import percent, rows_of
from .evidence import Direction
from .results import StrategyResult
from .strategies.support_novelty import SupportNoveltyEstimator


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


def _enough_answer(final: Mapping[str, Any], stops: Mapping[str, Any]) -> str:
    """One criterion's answer to 'did we sample enough?' as a sentence."""
    criterion = f"{final['strategy']} ({final['policies']})"
    acquired, orders = final["n_fovs"], stops["orders"]
    if pd.isna(final["verdict"]):
        return f"{criterion}: undefined for these data ({final['undefined_reason']})."
    # A fixed count says its target was reached, not that the sample was enough.
    effort_rule = " >= " in str(final["policies"]) and str(final["policies"]).startswith("n_")
    if final["verdict"] == "continue":
        head = f"{criterion}: no, not met by the {acquired} acquired FOVs"
    elif effort_rule:
        head = f"{criterion}: its target was reached by the {acquired} acquired FOVs"
    else:
        head = f"{criterion}: yes, its threshold was met by the {acquired} acquired FOVs"
    if final.get("assumes_spatial_uniformity"):
        head += " (treats observations as independent draws: assumes a spatially uniform sample)"
    if pd.isna(stops["stop_fov_median"]):
        return f"{head}; not met at any point in {orders} replayed orders."
    first = (
        f"first met after {stops['stop_fov_median']:g} FOVs "
        f"(10-90%: {stops['stop_fov_p10']:g}-{stops['stop_fov_p90']:g}) over {orders} orders"
    )
    never = stops["never_stopped"]
    missing = f", never in {never}" if never else ""
    lost = (
        "; it was met and then lost as more FOVs arrived" if final["verdict"] == "continue" else ""
    )
    return f"{head}; {first}{missing}{lost}."


def novelty_definition(
    novelty: SupportNoveltyEstimator, feature_cols: Sequence[str] | None = None
) -> str:
    """What counts as a FOV showing something new, in words; with ``feature_cols``,
    also the columns it is judged on (the phenotype geometry's provenance)."""
    m, others = novelty.min_novel_observations, novelty.min_other_fovs
    space = "raw feature units" if novelty.transform is None else novelty.transform.name
    cells = (
        "at least 1 observation"
        if m == 1
        else f"a cluster of at least {m} similar observations (linked within the distance)"
    )
    text = (
        f"{cells} with fewer than {others} other FOV{'s' if others > 1 else ''} "
        f"within distance {novelty.epsilon:g} ({space})"
    )
    if feature_cols:
        names = list(feature_cols)
        shown = ", ".join(names[:3]) + (f" and {len(names) - 3} more" if len(names) > 3 else "")
        text += f", on {len(names)} feature{'s' if len(names) > 1 else ''} ({shown})"
    return text


def _more_fovs(count: int) -> str:
    """``1 more FOV``, ``3 more FOVs``."""
    return f"{count} more FOV" if count == 1 else f"{count} more FOVs"


def _several_states(row: Mapping[str, Any], fewest: int) -> str:
    """When the FOVs seen in few could hold two or more recurrent states, each one's need."""
    if int(row["fovs_with_state_in_few"]) < 2 * fewest:
        return ""
    return (
        " They may hold several different states, each rarer: one seen in only "
        f"{fewest} FOVs needs about {int(row['fovs_to_see_fewest'])}."
    )


def _next_fov_answer(
    row: Mapping[str, Any], target: float | None, calibrated: Mapping[str, Any] | None
) -> str:
    """Question 2 for one group, as lines of text."""
    if pd.isna(row["chance_new"]):
        return f"Not estimable: {row['reason']}."
    confidence = percent(row["confidence"])
    chance = (
        f"Chance: {percent(row['chance_new'])} (nominal {confidence} upper bound "
        f"{percent(row['chance_new_at_most'])}, approximate: the leave-one-FOV-out outcomes "
        f"share their reference FOVs); {row['fovs_with_new']} of {row['fovs_imaged']} FOVs "
        "showed something the others did not."
    )
    empty = int(row.get("fovs_without_observations", 0) or 0)
    counted = (
        f" The count includes {empty} FOV{'s' if empty > 1 else ''} without observations, "
        "which cannot show a state."
        if empty
        else ""
    )
    # The zero-success bound is about -ln(1 - confidence) / k: 3 / k at 95%.
    successes = -math.log(1 - row["confidence"])
    unseen = (
        f"Not yet seen: a state this definition detects but no FOV has shown is in at most "
        f"{percent(row['unseen_state_max_frequency'])} of FOVs ({confidence} confidence, about "
        f"{successes:.2g} / {row['fovs_imaged']}), if FOVs are independent, representative "
        f"draws. Rarer states can remain.{counted}"
    )
    lines = [chance, unseen]
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
                f"state that fewer than {int(few)} other FOVs show. A state seen in {span} "
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
            f"{rule_out}: already reached."
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
            lines.append(
                f"{goal}: at least {_more_fovs(best)} if none shows anything new; {at_rate}."
            )
    if calibrated is not None:
        predicted, actual = percent(calibrated["predicted"]), percent(calibrated["actual"])
        held = percent(calibrated["bound_held"])
        lines.append(
            f"Check on this dataset: predicting from {int(calibrated['fovs_imaged'])} random FOVs, "
            f"predicted {predicted}, actual {actual}; "
            f"the bound held in {held} of {int(calibrated['draws'])} draws."
        )
    return "\n".join(lines)


def report_summary(
    enough: pd.DataFrame,
    next_fov: pd.DataFrame,
    group_cols: Sequence[str],
    novelty_definition: str | None,
) -> str:
    """Both answers in plain sentences, one block per analysis group."""
    blocks = []
    keys = list(group_cols)
    tables = [table[keys] for table in (enough, next_fov) if not table.empty]
    groups = pd.concat(tables).drop_duplicates() if tables else pd.DataFrame()
    # Without group columns there is one pooled group, which has no key.
    rows = list(groups.itertuples(index=False, name=None)) if keys else [()] * bool(tables)
    for values in rows:
        group = dict(zip(keys, values, strict=True))
        label = ", ".join(f"{k}={v}" for k, v in group.items()) or "all FOVs (pooled)"
        lines = [f"{label}:"]
        # A question that was not asked has an empty table without the group columns.
        criteria = rows_of(enough, group) if not enough.empty else enough
        if len(criteria):
            lines.append(
                "  Did each criterion say enough? (its own threshold, not a held-out check)"
            )
            lines += [f"  - {answer}" for answer in criteria["answer"]]
        outlook = rows_of(next_fov, group) if not next_fov.empty else next_fov
        if len(outlook):
            lines.append("  Would one more FOV show something new?")
            lines += [f"  - {line}" for line in outlook["answer"].iloc[0].split("\n")]
        blocks.append("\n".join(lines))
    if novelty_definition:
        blocks.append(f"'New' means {novelty_definition}.")
    return "\n\n".join(blocks)


__all__ = ["novelty_definition", "report_summary"]
