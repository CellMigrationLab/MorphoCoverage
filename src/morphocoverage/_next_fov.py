"""Question 2 of the sufficiency report (private): would one more FOV show something new?

Per analysis group, the chance that the next field holds a new state and the fields it
would take to claim it below a target (``_next_fov_tables``), the observations counted as
new, the fields holding a state seen in only a few other fields (``few_fovs``), the
shuffle calibration and each group's answer (``_next_fov_part``).

Shared-private: ``_NoveltyQuestion`` and ``_next_fov_part``, used by ``sufficiency``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any, NamedTuple

import numpy as np
import pandas as pd

from ._wording import rows_of
from .bounds import extra_trials_for_bound, interval, upper_bound
from .calibration import calibrate_next_fov_novelty
from .evaluator import iter_analysis_groups
from .evidence import Availability, Evidence, Quantity
from .objectives import ObjectiveConfig, SamplingObjective
from .schema import ValidatedInput, fov_positions, technical_repeat_columns
from .statistics import fovs_to_detect
from .strategies.support_novelty import FOV_QUANTITY, UPPER_QUANTITY, SupportNoveltyEstimator
from .sufficiency_null import mark_uninformative
from .sufficiency_text import _next_fov_answer

_POPULATION = ObjectiveConfig(SamplingObjective.POPULATION)
# The count columns of the table: integers, missing as <NA>.
_COUNT_COLUMNS = (
    "fovs_with_new",
    "extra_fovs_if_nothing_new",
    "extra_fovs_at_current_rate",
    "extra_fovs_to_rule_out_target",
    "technical_repeats",
)


class _NoveltyQuestion(NamedTuple):
    """Question 2 as asked: the definition of new, the target, the user's "few" fields and
    the calibration settings (``sufficiency.CalibrationSettings``), each possibly ``None``
    but the definition."""

    novelty: SupportNoveltyEstimator
    target: float | None
    few_fovs: int | None
    calibration: Any


def _next_fov_part(
    objects: pd.DataFrame,
    fovs: pd.DataFrame,
    novelty_data: ValidatedInput,
    question: _NoveltyQuestion,
    novelty_columns: dict[str, Any],
    group_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """The next-FOV table with each group's answer, the observations counted as new and the
    calibration (empty unless asked)."""
    novelty, target = question.novelty, question.target
    next_fov, new_observations = _next_fov_tables(novelty_data, novelty, target, question.few_fovs)
    next_fov = mark_uninformative(next_fov, novelty_data, novelty)
    calibrated = pd.DataFrame()
    if question.calibration is not None:
        calibrated = calibrate_next_fov_novelty(
            objects,
            fovs,
            novelty=novelty,
            fov_counts=question.calibration.fov_counts,
            draws=question.calibration.draws,
            random_state=question.calibration.random_state,
            **novelty_columns,
        )
    next_fov["answer"] = [
        _next_fov_answer(
            row,
            target,
            _largest_calibrated(calibrated, row, group_cols),
            novelty.min_other_fovs,
        )
        for _, row in next_fov.iterrows()
    ]
    return next_fov, new_observations, calibrated


def _largest_calibrated(
    calibrated: pd.DataFrame, row: Mapping[str, Any], group_cols: list[str]
) -> pd.Series | None:
    """The group's calibration row with the most FOVs and usable draws, or None."""
    if calibrated.empty:
        return None
    rows = rows_of(calibrated, {c: row[c] for c in group_cols})
    rows = rows[rows["draws"] > 0]
    return None if rows.empty else rows.sort_values("fovs_imaged").iloc[-1]


def _next_fov_tables(
    data: ValidatedInput,
    novelty: SupportNoveltyEstimator,
    target: float | None,
    few_fovs: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Question 2 per group, and the observations counted as new."""
    rows, flagged = [], []
    features = set(data.feature_cols)
    for group, group_data in iter_analysis_groups(data):
        evidence = novelty.estimate(group_data, objective=_POPULATION)
        chance, upper = evidence.quantities[FOV_QUANTITY], evidence.quantities[UPPER_QUANTITY]
        row = _group_row(group, group_data, novelty, evidence)
        if chance.availability is not Availability.AVAILABLE:
            row.update(
                dict.fromkeys(("fovs_with_new", "chance_new", "chance_new_at_most"), None),
                extra_fovs_if_nothing_new=None,
                extra_fovs_at_current_rate=None,
                unseen_state_max_frequency=None,
                extra_fovs_to_rule_out_target=None,
                reason=chance.reason,
            )
            rows.append(row)
            continue
        new_fovs = int(evidence.diagnostics["novel_fovs"])
        _add_chances(row, new_fovs, (chance, upper), len(group_data.fovs), novelty, target)
        row.update(_seen_in_few(novelty, group_data, few_fovs))
        rows.append(row)
        # Only the objects of new states, so scattered outliers stay out when
        # several similar new objects are required.
        counted = novelty.novel_observation_flags(group_data, new_states_only=True)
        if counted is not None and counted.any():
            keep = [c for c in group_data.objects.columns if c not in features]
            flagged.append(group_data.objects.loc[counted, keep])
    new_observations = pd.concat(flagged, ignore_index=True) if flagged else pd.DataFrame()
    table = pd.DataFrame(rows)
    for column in _COUNT_COLUMNS:
        table[column] = table[column].astype("Int64")  # counts, missing as <NA>
    return table, new_observations


def _group_row(
    group: Mapping[str, Any],
    group_data: ValidatedInput,
    novelty: SupportNoveltyEstimator,
    evidence: Evidence,
) -> dict[str, Any]:
    """A group's row before its chances: the fields imaged, those without observations,
    the novelty definition's settings and the technical repeats."""
    k = len(group_data.fovs)
    # The estimator's own count when it ran; otherwise counted here, so fields
    # without observations are reported even when novelty is not estimable.
    with_observations = evidence.diagnostics.get("fovs_with_observations")
    if with_observations is None:
        positions = fov_positions(group_data, group_data.fovs)
        with_observations = len(np.unique(positions[positions >= 0]))
    with_observations = int(with_observations)
    return {
        **group,
        "fovs_imaged": k,
        "fovs_without_observations": k - with_observations,
        "confidence": novelty.confidence,
        "min_novel_observations": novelty.min_novel_observations,
        "min_other_fovs": novelty.min_other_fovs,
        "technical_repeats": _technical_repeats(group_data),
    }


def _add_chances(
    row: dict[str, Any],
    new_fovs: int,
    chances: tuple[Quantity, Quantity],
    k: int,
    novelty: SupportNoveltyEstimator,
    target: float | None,
) -> None:
    """Add to ``row`` the chance of a new state in the next field and its upper bound, the
    fields needed to claim it below ``target``, and the bounds on a state never seen."""
    chance, upper = chances
    row.update(
        {
            "fovs_with_new": new_fovs,
            "chance_new": chance.value,
            "chance_new_at_most": upper.value,
            "reason": None,
        }
    )
    for column, keep_rate in (
        ("extra_fovs_if_nothing_new", False),
        ("extra_fovs_at_current_rate", True),
    ):
        row[column] = (
            None
            if target is None
            else extra_trials_for_bound(
                new_fovs, k, target, novelty.confidence, keep_rate=keep_rate
            )
        )
    # A state absent from all k FOVs has a per-FOV frequency below the
    # zero-success bound 1 - (1 - confidence) ** (1 / k), about
    # -ln(1 - confidence) / k (3 / k at 95%, the rule of three), whatever
    # else was new.
    row["unseen_state_max_frequency"] = upper_bound(0, k, novelty.confidence)
    # Pooled technical repeats cluster a state in their fields (P3-8): they are
    # the independent draws, and none of them showed the state.
    technical = row["technical_repeats"]
    many = technical is not None and technical > 1
    row["unseen_state_max_technical_share"] = (
        upper_bound(0, technical, novelty.confidence) if many else None
    )
    row["extra_fovs_to_rule_out_target"] = (
        None
        if target is None
        else extra_trials_for_bound(0, k, target, novelty.confidence, keep_rate=False)
    )


def _technical_repeats(group_data: ValidatedInput) -> int | None:
    """The technical repeats (each with its condition and repeat) of the group's fields."""
    columns = technical_repeat_columns(group_data.fovs)
    return None if columns is None else len(group_data.fovs[columns].drop_duplicates())


def _seen_in_few(
    novelty: SupportNoveltyEstimator, group_data: ValidatedInput, few_fovs: int | None
) -> dict[str, Any]:
    """FOVs holding a state that some, but fewer than ``few_fovs``, other FOVs show; or why not.

    Only states that are not new: an observation counts when it belongs to a
    state new under ``min_other_fovs=few_fovs`` but is itself not new under the
    novelty definition, so the FOVs already reported as new (a state in one
    FOV) are not counted again here, nor pooled into the planning incidence.
    """
    if few_fovs is None:
        return {}
    looser = replace(novelty, min_other_fovs=few_fovs)
    evidence = looser.estimate(group_data, objective=_POPULATION)
    quantity = evidence.quantities[FOV_QUANTITY]
    if quantity.availability is not Availability.AVAILABLE:
        return {"few_fovs": few_fovs, "fovs_with_state_in_few": None, "few_reason": quantity.reason}
    in_few = looser.novel_observation_flags(group_data, new_states_only=True)
    new = novelty.novel_observation_flags(group_data)
    if in_few is None or new is None:
        reason = "the leave-one-FOV-out flags of the observations are not available"
        return {"few_fovs": few_fovs, "fovs_with_state_in_few": None, "few_reason": reason}
    positions = fov_positions(group_data, group_data.fovs)
    seen = int(np.unique(positions[in_few & ~new]).size)
    k = len(group_data.fovs)
    # The fewest FOVs a state can be in and still count as recurrent (not new).
    fewest = novelty.min_other_fovs + 1
    row = {
        "few_fovs": few_fovs,
        "fovs_with_state_in_few": seen,
        "few_reason": None,
        "fewest_recurrent_fovs": fewest,
    }
    if seen:
        # Planning for a state seen in ``seen`` of ``k`` independent, representative
        # FOVs: its incidence with a two-sided Clopper-Pearson interval, and the
        # binomial number of FOVs to see it at least once with probability
        # ``confidence`` at the observed incidence and at the interval's low end.
        # The model was checked against the planted-state control
        # (docs/preliminary_findings.md, entry 3; tests/test_crick_idr.py).
        low, high = interval(seen, k, novelty.confidence)
        row.update(
            {
                "few_state_incidence": seen / k,
                "few_state_incidence_low": low,
                "few_state_incidence_high": high,
                "fovs_to_see_again": fovs_to_detect(seen / k, novelty.confidence, 1)
                if seen < k
                else 1,
                "fovs_to_see_again_low_end": fovs_to_detect(low, novelty.confidence, 1),
                # These FOVs may hold several different states, each rarer than all of
                # them together; one seen in only ``fewest`` FOVs needs this many.
                "fovs_to_see_fewest": fovs_to_detect(min(fewest, k) / k, novelty.confidence, 1)
                if fewest < k
                else 1,
            }
        )
    return row
