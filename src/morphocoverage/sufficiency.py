"""Here is an analysed dataset: did we sample enough, and would one more FOV show something new?

``sufficiency_report`` answers two questions for each analysis group
(condition x repeat by default) of a finished acquisition, given the
observation table and the FOV table of every acquired position:

1. **Did we sample enough?** (``enough``) For each criterion (a strategy with
   its thresholds): is it met by all the acquired FOVs, after how many FOVs
   was it first met when the FOVs are replayed in the declared acquisition
   orders, and so how many FOVs were imaged beyond that point? A criterion not
   met by the acquired FOVs says more FOVs were needed by that criterion.
2. **Would one more FOV show something new?** (``next_fov``, when a
   ``novelty`` definition is given) The estimated chance that one more
   representative FOV shows a phenotype not seen in the other FOVs, with a
   nominal upper bound (a Clopper-Pearson bound on leave-one-FOV-out
   outcomes, which are not independent, so its coverage is approximate and
   is checked by ``calibration``), and how many more FOVs would be needed before the
   bound could fall below ``novelty_target``. It also says what the FOVs rule
   out: a state not yet seen in any of ``k`` representative FOVs is present in
   at most ``1 - (1 - confidence) ** (1 / k)`` of FOVs, about ``3 / k`` at 95%
   (``unseen_state_max_frequency``), with the extra FOVs needed to rule out
   states above ``novelty_target``. A small chance for the next FOV is not
   evidence that no rare state remains: in the idr0028 frequency control
   (``docs/preliminary_findings.md``, entry 3) stopping at a 25% bound (after
   a median of 11 FOVs, up to 23) missed a state in 5% of FOVs in almost half
   the acquisitions that had it. ``new_observations`` lists the
   observations counted as new so they can be inspected in the images, and
   ``calibration`` (optional) checks the prediction against this dataset by
   shuffling (``morphocoverage.calibration``).

``summary()`` puts both answers into a short summary (one table of the
rules across groups, each caveat once; ``sufficiency_short``) and
``summary(detail="full")`` into every number, one block per group, each caveat
once (``sufficiency_full``); the sentences, one per group and rule, are the
``answer`` columns of the tables (``sufficiency_text``), and the criteria
tables live in ``sufficiency_tables``.

Reading it:

- Every answer holds only for the stated criterion, thresholds and
  definition of new. Thresholds without a published meaning are the user's
  choice, so compare several (pass the same strategy with different
  thresholds) rather than read one verdict as a fact.
- "New" is new at the chosen resolution in the chosen features: a cluster of
  at least ``min_novel_observations`` similar observations (linked within
  ``epsilon``) with fewer than ``min_other_fovs`` other FOVs within
  ``epsilon``, so scattered outliers do not count. Look at
  ``new_observations`` to see whether that is biology (a rare object state) or
  technique (clumped debris, a focus problem). Too fine a resolution calls
  almost every FOV new.
- With few FOVs the bound stays high whatever the data look like: with no
  new FOV among ``k``, the 95% bound is ``1 - 0.05 ** (1 / k)``, so claiming
  "below 10%" needs at least 29 FOVs.
- Random orders treat FOVs as exchangeable; the recorded order keeps drift
  and neighbour effects. Declare both to see the difference: the stops table
  has one row per ``order_kind`` and each answer gives the recorded and the
  random orders in separate clauses, never pooled into one median.
- A fixed-effort rule (a number of FOVs or objects, an imaged area) reports
  ``sampled_enough = "target reached"``, not ``"yes"``: reaching a fixed
  amount of effort says nothing about what the data show.
- This is self-assessment of one dataset. It says what each criterion
  concludes, not that the criterion is right; that needs held-out deep data
  (``docs/benchmarking.md``). The calibration check is the part that tests a
  prediction against this dataset itself.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd

from ._checks import is_integer
from ._former_repeat import one_warning
from ._parallel import check_n_jobs
from ._style import check_detail
from ._wording import rows_of
from .benchmark import (
    AcquisitionOrder,
)
from .bounds import extra_trials_for_bound, interval, upper_bound
from .calibration import calibrate_next_fov_novelty
from .data_check import validate_input_tables
from .evaluator import iter_analysis_groups
from .evidence import Availability
from .metadata import CONDITION_COLUMN, REPEAT_COLUMN, TECHNICAL_REPEAT_COLUMN
from .objectives import ObjectiveConfig, SamplingObjective
from .schema import ValidatedInput, fov_positions
from .statistics import fovs_to_detect
from .strategies.base import SamplingStrategy
from .strategies.support_novelty import (
    FOV_QUANTITY,
    UPPER_QUANTITY,
    SupportNoveltyEstimator,
    SupportNoveltyStrategy,
)
from .sufficiency_full import full_summary
from .sufficiency_short import short_summary
from .sufficiency_tables import criteria_tables, enough_table, group_major
from .sufficiency_text import (
    _next_fov_answer,
    novelty_definition,
)

_ROW_ORDER = "_report_row_order"
_POPULATION = ObjectiveConfig(SamplingObjective.POPULATION)


@dataclass(frozen=True)
class CalibrationSettings:
    """How to check the new-phenotype prediction by shuffling; see ``morphocoverage.calibration``.

    ``fov_counts`` are the numbers of FOVs to predict from, ``draws`` the
    random subsets per count and group, ``random_state`` the seed.
    """

    fov_counts: tuple[int, ...]
    draws: int
    random_state: int


@dataclass(frozen=True)
class SufficiencyReport:
    """The answers, and the detailed tables behind them; see the module docstring.

    - ``enough``: did we sample enough? One row per group and criterion.
    - ``next_fov``: would one more FOV show something new? One row per group
      (empty without a ``novelty`` definition).
    - ``new_observations``: the observations counted as new, with their FOV
      and metadata columns, for inspection.
    - ``calibration``: predicted versus actual chance of a new phenotype, per
      group and number of FOVs (empty unless requested).
    - ``final``: every criterion's decision and evidence (``q.<name>``) on the
      full acquisition. ``stops``: the stop-FOV distribution over orders, one
      row per group, criterion and ``order_kind`` (recorded, permuted).
    """

    enough: pd.DataFrame
    next_fov: pd.DataFrame
    new_observations: pd.DataFrame
    calibration: pd.DataFrame
    final: pd.DataFrame
    stops: pd.DataFrame
    group_cols: tuple[str, ...] = ()
    novelty_definition: str | None = None
    observation_unit: str | None = None
    novelty_target: float | None = None
    observation_unit_declared: bool = True

    def summary(self, detail: str = "short") -> str:
        """Both answers: ``"short"``, one table of the rules across groups, the novelty answers
        in a few lines and each caveat once (``sufficiency_short``); ``"full"``, every number,
        one block per analysis group, each caveat once (``sufficiency_full``). The sentences
        per group and rule are the ``answer`` columns of ``enough`` and ``next_fov``."""
        if check_detail(detail) == "short":
            return short_summary(self)
        return full_summary(self)


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


def _technical_repeats(group_data: ValidatedInput) -> int | None:
    """The technical repeats (each with its condition and repeat) of the group's fields."""
    fovs = group_data.fovs
    if TECHNICAL_REPEAT_COLUMN not in fovs.columns:
        return None
    columns = [c for c in (CONDITION_COLUMN, REPEAT_COLUMN, TECHNICAL_REPEAT_COLUMN) if c in fovs]
    return len(fovs[columns].drop_duplicates())


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
        k = len(group_data.fovs)
        # The estimator's own count when it ran; otherwise counted here, so fields
        # without observations are reported even when novelty is not estimable.
        with_observations = evidence.diagnostics.get("fovs_with_observations")
        if with_observations is None:
            positions = fov_positions(group_data, group_data.fovs)
            with_observations = len(np.unique(positions[positions >= 0]))
        with_observations = int(with_observations)
        row = {
            **group,
            "fovs_imaged": k,
            "fovs_without_observations": k - with_observations,
            "confidence": novelty.confidence,
            "min_novel_observations": novelty.min_novel_observations,
            "technical_repeats": _technical_repeats(group_data),
        }
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
    counts = (
        "fovs_with_new",
        "extra_fovs_if_nothing_new",
        "extra_fovs_at_current_rate",
        "extra_fovs_to_rule_out_target",
        "technical_repeats",
    )
    for column in counts:
        table[column] = table[column].astype("Int64")  # counts, missing as <NA>
    return table, new_observations


def _check_distinct(strategies: Sequence[SamplingStrategy], added: bool) -> None:
    """Refuse a criterion passed twice: its rows could not be told apart in the tables."""
    seen: dict[tuple[str, str], int] = {}
    for position, strategy in enumerate(strategies):
        key = (strategy.name, strategy.config_id)
        if key in seen:
            automatic = added and position == len(strategies) - 1
            raise ValueError(
                f"the criterion {strategy.name!r} (config_id {strategy.config_id}) is "
                + (
                    "the one novelty_target adds: drop it from strategies, or drop novelty_target"
                    if automatic
                    else "passed twice in strategies: pass each criterion once"
                )
            )
        seen[key] = position


def _check_report_inputs(
    strategies: Sequence[SamplingStrategy],
    orders: Sequence[AcquisitionOrder],
    novelty: SupportNoveltyEstimator | None,
    target: float | None,
    calibration: CalibrationSettings | None,
) -> None:
    """Validate the combination of questions asked."""
    if not strategies and novelty is None:
        raise ValueError("give at least one strategy or a novelty definition")
    if not orders or not all(isinstance(order, AcquisitionOrder) for order in orders):
        raise ValueError("orders must be a non-empty sequence of AcquisitionOrder")
    if novelty is not None and not isinstance(novelty, SupportNoveltyEstimator):
        raise TypeError("novelty must be a SupportNoveltyEstimator")
    if target is not None:
        if novelty is None:
            raise ValueError("novelty_target needs a novelty definition")
        if isinstance(target, bool) or not 0 < target < 1:
            raise ValueError("novelty_target must be a probability between 0 and 1")
    if calibration is not None:
        if novelty is None:
            raise ValueError("calibration needs a novelty definition")
        if not isinstance(calibration, CalibrationSettings):
            raise TypeError("calibration must be CalibrationSettings")


@one_warning
def sufficiency_report(
    objects: pd.DataFrame,
    fovs: pd.DataFrame,
    *,
    feature_cols: Sequence[str],
    strategies: Iterable[SamplingStrategy],
    objective: ObjectiveConfig,
    orders: Sequence[AcquisitionOrder],
    novelty: SupportNoveltyEstimator | None = None,
    novelty_target: float | None = None,
    calibration: CalibrationSettings | None = None,
    few_fovs: int | None = None,
    acquisition_order_col: str | None = None,
    fov_col: str | None = None,
    group_cols: Sequence[str] | None = None,
    observation_unit: str | None = None,
    novelty_cols: Sequence[str] | None = None,
    readout: str | None = None,
    stage_cols: Sequence[str] | None = None,
    n_jobs: int = 1,
) -> SufficiencyReport:
    """Did we sample enough, and would one more FOV show something new? Per analysis group.

    ``fovs`` must list every acquired FOV, including FOVs without
    observations. ``strategies`` are the criteria for "enough". ``orders``
    are the replay orders, for example ``[AcquisitionOrder.recorded()] +
    [AcquisitionOrder.permuted(s) for s in range(20)]``; the recorded order
    needs ``acquisition_order_col``, and without that column only permuted
    orders are allowed, since the table's row order is not an acquisition
    record.

    ``few_fovs`` (with ``novelty``) also reports how many FOVs hold a state
    that other FOVs show too, but fewer than ``few_fovs`` of them (FOVs
    already counted as new are left out): a state recurring in two
    neighbouring FOVs is not new to the next FOV, yet is not established
    either, and would otherwise go unmentioned. It is the user's choice of
    "few" and has no default.

    ``novelty`` defines what counts as new and turns on the second question.
    With ``novelty_target``, the report also says how many more FOVs are
    needed before the chance of something new can be claimed below the target,
    and adds that claim as a criterion for "enough" (a ``SupportNoveltyStrategy``
    on the upper bound; a criterion passed twice, or equal to that one, is
    refused). ``calibration`` checks the prediction on this
    dataset by shuffling. ``novelty_cols`` (a subset of ``feature_cols``) are
    the columns "new" is judged on, when they differ from the columns the
    rules read: for example the rules on a readout, novelty on the features it
    was defined for. By default novelty uses every column of ``feature_cols``.
    ``readout`` with ``acquisition_order_col`` or ``stage_cols`` adds the order
    check (``order_check``) beside every stop in the answers.
    ``n_jobs`` replays the rules' analysis groups in that many worker processes
    (``-1``: every CPU), the dominant cost with a novelty criterion on large
    tables; results are identical for every value (``run_benchmark``).
    Other arguments are those of ``run_strategies``.
    """
    check_n_jobs(n_jobs)
    strategies, orders = list(strategies), list(orders)
    _check_report_inputs(strategies, orders, novelty, novelty_target, calibration)
    if novelty_cols is not None:
        if isinstance(novelty_cols, str) or not novelty_cols:
            raise TypeError("novelty_cols must be a non-empty sequence of column names")
        outside = [c for c in novelty_cols if c not in list(feature_cols)]
        if outside or novelty is None:
            raise ValueError(
                "novelty_cols must be columns of feature_cols, with a novelty definition; "
                f"not in feature_cols: {outside}"
            )
    if few_fovs is not None:
        if novelty is None:
            raise ValueError("few_fovs needs a novelty definition")
        if not is_integer(few_fovs):  # numpy integers too; bool is refused
            raise TypeError("few_fovs must be an integer number of FOVs")
        if few_fovs <= novelty.min_other_fovs:
            raise ValueError("few_fovs must exceed the novelty definition's min_other_fovs")
        few_fovs = int(few_fovs)
    novelty_rules: list[SamplingStrategy] = []
    if novelty is not None and novelty_target is not None:
        novelty_rules.append(
            SupportNoveltyStrategy(
                novelty.epsilon,
                novelty_target,
                novelty.min_other_fovs,
                novelty.transform,
                novelty.min_fovs,
                novelty.min_novel_observations,
                UPPER_QUANTITY,
                novelty.confidence,
            )
        )
    _check_distinct([*strategies, *novelty_rules], added=bool(novelty_rules))
    declared_order = acquisition_order_col
    if acquisition_order_col is None:
        if any(order.kind == "recorded" for order in orders):
            raise ValueError("the recorded order needs acquisition_order_col")
        if _ROW_ORDER in fovs.columns:
            raise ValueError(f"the FOV table already has a column {_ROW_ORDER!r}")
        # Only a tie-breaking base for permutations, never replayed as recorded.
        fovs = fovs.assign(**{_ROW_ORDER: np.arange(len(fovs))})
        acquisition_order_col = _ROW_ORDER

    columns = {
        "feature_cols": list(feature_cols),
        "fov_col": fov_col,
        "group_cols": None if group_cols is None else list(group_cols),
        "observation_unit": observation_unit,
    }
    data = validate_input_tables(objects, fovs=fovs, **columns)
    resolved_groups = list(data.group_cols)
    # Novelty reads only its own columns; the rules read all of feature_cols.
    novelty_columns = (
        columns if novelty_cols is None else {**columns, "feature_cols": list(novelty_cols)}
    )
    novelty_data = (
        data
        if novelty_cols is None
        else validate_input_tables(objects, fovs=fovs, **novelty_columns)
    )
    if novelty_cols is None:
        strategies += novelty_rules
        runs = [(strategies, columns)]
    else:
        runs = [(strategies, columns), (novelty_rules, novelty_columns)]

    final, stops, enough = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    parts = [
        criteria_tables(
            objects,
            fovs,
            rules,
            objective,
            orders,
            acquisition_order_col,
            run_columns,
            resolved_groups,
            readout=readout,
            order_col=declared_order,
            stage_cols=stage_cols,
            n_jobs=n_jobs,
        )
        for rules, run_columns in runs
        if rules
    ]
    if parts:
        # The order and dtypes of a single run: final group by group, stops strategy
        # by strategy (a block of undefined stops alone would read as objects).
        final = group_major(
            pd.concat([part[0] for part in parts], ignore_index=True), resolved_groups
        ).infer_objects()
        stops = pd.concat([part[1] for part in parts], ignore_index=True).infer_objects()
        enough = enough_table(final, stops, resolved_groups)

    next_fov, new_observations, calibrated = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    if novelty is not None:
        next_fov, new_observations = _next_fov_tables(
            novelty_data, novelty, novelty_target, few_fovs
        )
        if calibration is not None:
            calibrated = calibrate_next_fov_novelty(
                objects,
                fovs,
                novelty=novelty,
                fov_counts=calibration.fov_counts,
                draws=calibration.draws,
                random_state=calibration.random_state,
                **novelty_columns,
            )
        next_fov["answer"] = [
            _next_fov_answer(
                row,
                novelty_target,
                _largest_calibrated(calibrated, row, resolved_groups),
                novelty.min_other_fovs,
            )
            for _, row in next_fov.iterrows()
        ]
    return SufficiencyReport(
        enough=enough,
        next_fov=next_fov,
        new_observations=new_observations,
        calibration=calibrated,
        final=final,
        stops=stops,
        group_cols=tuple(resolved_groups),
        novelty_definition=(
            None
            if novelty is None
            else novelty_definition(novelty, novelty_columns["feature_cols"])
        ),
        observation_unit=data.observation_unit,
        novelty_target=novelty_target,
        observation_unit_declared=data.observation_unit_declared,
    )


def _largest_calibrated(
    calibrated: pd.DataFrame, row: Mapping[str, Any], group_cols: list[str]
) -> pd.Series | None:
    """The group's calibration row with the most FOVs and usable draws, or None."""
    if calibrated.empty:
        return None
    rows = rows_of(calibrated, {c: row[c] for c in group_cols})
    rows = rows[rows["draws"] > 0]
    return None if rows.empty else rows.sort_values("fovs_imaged").iloc[-1]


__all__ = [
    "CalibrationSettings",
    "SufficiencyReport",
    "novelty_definition",
    "sufficiency_report",
]
