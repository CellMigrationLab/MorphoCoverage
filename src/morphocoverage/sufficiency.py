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
  (``docs/optimise/benchmarking.md``). The calibration check is the part that tests a
  prediction against this dataset itself.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np
import pandas as pd

from ._acquisition import acquisition_identity
from ._checks import is_integer
from ._frozen import owned
from ._next_fov import _next_fov_part, _NoveltyQuestion
from ._parallel import check_n_jobs
from ._style import check_detail
from .benchmark import (
    AcquisitionOrder,
)
from .data_check import validate_input_tables
from .metadata import column_names
from .objectives import ObjectiveConfig
from .schema import ValidatedInput
from .strategies.base import SamplingStrategy
from .strategies.support_novelty import (
    UPPER_QUANTITY,
    SupportNoveltyEstimator,
    SupportNoveltyStrategy,
)
from .sufficiency_full import full_summary
from .sufficiency_short import short_summary
from .sufficiency_tables import criteria_tables, enough_table, group_major
from .sufficiency_text import (
    novelty_definition,
)

_ROW_ORDER = "_report_row_order"


@dataclass(frozen=True)
class CalibrationSettings:
    """How to check the new-phenotype prediction by shuffling; see ``morphocoverage.calibration``.

    ``fov_counts`` are the numbers of FOVs to predict from, ``draws`` the
    random subsets per count and group, ``random_state`` the seed.
    """

    fov_counts: tuple[int, ...]
    draws: int
    random_state: int


@owned("enough", "next_fov", "new_observations", "calibration", "final", "stops")
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
    acquisition: str | None = None

    def summary(self, detail: str = "short") -> str:
        """Both answers: ``"short"``, one table of the rules across groups, the novelty answers
        in a few lines and each caveat once (``sufficiency_short``); ``"full"``, every number,
        one block per analysis group, each caveat once (``sufficiency_full``). The sentences
        per group and rule are the ``answer`` columns of ``enough`` and ``next_fov``."""
        if check_detail(detail) == "short":
            return short_summary(self)
        return full_summary(self)


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
    feature_cols, group_cols, novelty_cols = _read_columns(
        feature_cols, group_cols, novelty_cols, novelty
    )
    few_fovs = _checked_few_fovs(few_fovs, novelty)
    novelty_rules = _novelty_rules(novelty, novelty_target)
    _check_distinct([*strategies, *novelty_rules], added=bool(novelty_rules))
    declared_order = acquisition_order_col
    fovs, acquisition_order_col = _with_order_base(fovs, acquisition_order_col, orders)
    columns = {
        "feature_cols": list(feature_cols),
        "fov_col": fov_col,
        "group_cols": None if group_cols is None else list(group_cols),
        "observation_unit": observation_unit,
    }
    data, novelty_data, novelty_columns = _validated(objects, fovs, columns, novelty_cols)
    resolved_groups = list(data.group_cols)
    if novelty_cols is None:
        strategies += novelty_rules
        runs = [(strategies, columns)]
    else:
        runs = [(strategies, columns), (novelty_rules, novelty_columns)]
    replay = _Replay(
        objective, orders, acquisition_order_col, readout, declared_order, stage_cols, n_jobs
    )
    final, stops, enough = _enough_tables(objects, fovs, runs, replay, resolved_groups)
    next_fov, new_observations, calibrated = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    if novelty is not None:
        question = _NoveltyQuestion(novelty, novelty_target, few_fovs, calibration)
        next_fov, new_observations, calibrated = _next_fov_part(
            objects, fovs, novelty_data, question, novelty_columns, resolved_groups
        )
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
        acquisition=acquisition_identity(data),
    )


def _validated(
    objects: pd.DataFrame,
    fovs: pd.DataFrame,
    columns: dict[str, Any],
    novelty_cols: Sequence[str] | None,
) -> tuple[ValidatedInput, ValidatedInput, dict[str, Any]]:
    """The tables validated for the rules and for novelty, and novelty's column arguments;
    ``ValueError`` for a FOV table without rows."""
    data = validate_input_tables(objects, fovs=fovs, **columns)
    if data.fovs.empty:
        raise ValueError("the FOV table has no rows: the report needs at least one acquired FOV")
    # Novelty reads only its own columns; the rules read all of feature_cols.
    novelty_columns = (
        columns if novelty_cols is None else {**columns, "feature_cols": list(novelty_cols)}
    )
    novelty_data = (
        data
        if novelty_cols is None
        else validate_input_tables(objects, fovs=fovs, **novelty_columns)
    )
    return data, novelty_data, novelty_columns


def _read_columns(
    feature_cols: Any, group_cols: Any, novelty_cols: Any, novelty: Any
) -> tuple[Any, Any, Any]:
    """The column arguments read once (each may be a generator); ``novelty_cols`` checked
    to be columns of ``feature_cols``, with a novelty definition."""
    feature_cols = column_names(feature_cols, "feature_cols")  # read once: may be a generator
    group_cols = column_names(group_cols, "group_cols")
    novelty_cols = column_names(novelty_cols, "novelty_cols")
    if novelty_cols is not None:
        if not novelty_cols:
            raise TypeError("novelty_cols must be a non-empty sequence of column names")
        outside = [c for c in novelty_cols if c not in feature_cols]
        if outside or novelty is None:
            raise ValueError(
                "novelty_cols must be columns of feature_cols, with a novelty definition; "
                f"not in feature_cols: {outside}"
            )
    return feature_cols, group_cols, novelty_cols


def _checked_few_fovs(few_fovs: Any, novelty: SupportNoveltyEstimator | None) -> int | None:
    """``few_fovs`` as an integer above the novelty definition's ``min_other_fovs``."""
    if few_fovs is None:
        return None
    if novelty is None:
        raise ValueError("few_fovs needs a novelty definition")
    if not is_integer(few_fovs):  # numpy integers too; bool is refused
        raise TypeError("few_fovs must be an integer number of FOVs")
    if few_fovs <= novelty.min_other_fovs:
        raise ValueError("few_fovs must exceed the novelty definition's min_other_fovs")
    return int(few_fovs)


def _novelty_rules(
    novelty: SupportNoveltyEstimator | None, target: float | None
) -> list[SamplingStrategy]:
    """The claim "nothing new below the target" as a criterion for "enough", with a novelty
    definition and a target; none otherwise."""
    if novelty is None or target is None:
        return []
    return [
        SupportNoveltyStrategy(
            novelty.epsilon,
            target,
            novelty.min_other_fovs,
            novelty.transform,
            novelty.min_fovs,
            novelty.min_novel_observations,
            UPPER_QUANTITY,
            novelty.confidence,
        )
    ]


def _with_order_base(
    fovs: pd.DataFrame, acquisition_order_col: str | None, orders: Sequence[AcquisitionOrder]
) -> tuple[pd.DataFrame, str]:
    """The FOV table and the order column the replays read: without a declared order, the
    table's row order as a tie-breaking base for permutations (never the recorded order)."""
    if acquisition_order_col is not None:
        return fovs, acquisition_order_col
    if any(order.kind == "recorded" for order in orders):
        raise ValueError("the recorded order needs acquisition_order_col")
    if _ROW_ORDER in fovs.columns:
        raise ValueError(f"the FOV table already has a column {_ROW_ORDER!r}")
    # Only a tie-breaking base for permutations, never replayed as recorded.
    return fovs.assign(**{_ROW_ORDER: np.arange(len(fovs))}), _ROW_ORDER


class _Replay(NamedTuple):
    """How the rules are replayed: the objective, the orders and their column, and the
    order check's readout, declared order column and stage columns; the worker count."""

    objective: ObjectiveConfig
    orders: list[AcquisitionOrder]
    acquisition_order_col: str
    readout: str | None
    order_col: str | None
    stage_cols: Sequence[str] | None
    n_jobs: int


def _enough_tables(
    objects: pd.DataFrame,
    fovs: pd.DataFrame,
    runs: list[tuple[list[SamplingStrategy], dict[str, Any]]],
    replay: _Replay,
    resolved_groups: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Question 1: every rule on the full acquisition (``final``), its stops over the
    orders and the "enough" table; empty without a rule."""
    final, stops, enough = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    parts = [
        criteria_tables(
            objects,
            fovs,
            rules,
            replay.objective,
            replay.orders,
            replay.acquisition_order_col,
            run_columns,
            resolved_groups,
            readout=replay.readout,
            order_col=replay.order_col,
            stage_cols=replay.stage_cols,
            n_jobs=replay.n_jobs,
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
    return final, stops, enough


__all__ = [
    "CalibrationSettings",
    "SufficiencyReport",
    "novelty_definition",
    "sufficiency_report",
]
