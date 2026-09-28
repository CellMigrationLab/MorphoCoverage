"""Replay acquisitions against a fixed held-back reservoir: decisions and outcomes per prefix.

The shared engine of stopping-rule calibration (P6 of ``docs/questions.md``) and the
adequacy points (R2, P4). Per analysis group and outer split:

1. hold out a fixed reservoir of whole FOVs (or spatial blocks) with
   ``ReservoirSplit``, exactly as ``sampling_reliability`` does for the same
   ``random_state`` and split (both seed a split from ``random_state`` and the
   split number together, ``_seeds.split_seed``, so another ``random_state``
   holds back other fields in every split); the rest is the stream;
2. for each acquisition order of the stream (recorded, or seeded random
   permutations), reveal whole FOVs one at a time and record every strategy's
   decision after each prefix, with the same estimators and policies a live
   session would run;
3. for the same prefixes, assess every ``OutcomeCriterion`` against the
   reservoir. These outcomes depend on the prefix only, never on a strategy.

Decisions use only the visible prefix (the prospective path); outcomes use the
hidden reservoir (retrospective scoring). The two tables are kept apart so
that "the rule said enough" and "it was enough" are never mixed.

Permutations are drawn per group and split with ``benchmark.permuted_fovs``
(the benchmark's permutation), seeded from ``random_state``, the group's
identity, the order's seed and the split, so adding a group or a split never
changes another's orders, and neither does the row order of the tables.
Every prefix lists its observations in acquisition order (table order within
each FOV), as a live session receives them.

A dispersed order is fixed by its start; with more dispersed seeds than
stream fields, a seed whose start repeats an earlier seed's start in the same
split would replay the same acquisition again and weigh it twice in every
median and share, so it is replayed once.

A group too small to hold out a reservoir and keep a stream (a single FOV,
or a single block with ``block_col``) is skipped by the rule of
``sampling_reliability``, and listed with the reason in
``ReplayResult.skipped``; the other groups are replayed as usual. The
adequacy, calibration and efficiency summaries name the skipped groups.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .._checks import as_count, check_seed
from .._parallel import check_n_jobs, map_units
from .._seeds import group_identity
from ..benchmark import AcquisitionOrder, permuted_fovs
from ..data_check import validate_input_tables
from ..evaluator import iter_analysis_groups
from ..objectives import ObjectiveConfig
from ..schema import ValidatedInput
from ..sequential import (
    _Acquisition,
    _check_order_values,
    _in_acquisition_order,
    _interleave,
    _replay_strategies,
    _strategy_units,
)
from ..strategies.base import SamplingStrategy
from .criteria import OutcomeCriterion
from .reliability import _reservoir_split, _skip_reason
from .stage import dispersed_order, stage_travel

DEFAULT_PERMUTATIONS = 10

_RUN = ("split", "order_id", "order", "order_seed")
DECISION_COLUMNS = (*_RUN, "strategy", "config_id", "fovs", "decision")
OUTCOME_COLUMNS = (*_RUN, "fovs", "criterion", "value", "status")
STREAM_COLUMNS = (*_RUN, "stream_fovs", "reservoir_fovs")


@dataclass(frozen=True)
class ReplayResult:
    """Decisions and outcomes after every prefix of every replayed acquisition.

    - ``decisions``: one row per group, split, order, strategy and prefix
      length (``fovs``); ``decision`` is the rule's own verdict
      (``StrategyResult.rule_decision``): ``"stop"``, ``"continue"`` or
      ``None`` (undefined).
    - ``outcomes``: one row per group, split, order, prefix length and
      criterion, with the metric ``value`` and ``status`` (``preserved``,
      ``not_preserved``, ``undefined``) against that split's reservoir.
    - ``streams``: one row per group, split and order with the stream and
      reservoir sizes.

    - ``paths`` (with ``stage_cols``): one row per group, split, order and
      prefix length, with the cumulative stage ``travel`` from the first FOV.
    - ``skipped``: one row per analysis group that was not replayed, with the
      ``reason`` (too few FOVs or blocks to hold out a reservoir).

    One replayed acquisition is identified by the group, ``split`` and
    ``order_id``; ``order`` and ``order_seed`` give the order's kind and seed.
    """

    decisions: pd.DataFrame
    outcomes: pd.DataFrame
    streams: pd.DataFrame
    group_cols: tuple[str, ...]
    settings: Mapping[str, Any] = field(default_factory=dict)
    paths: pd.DataFrame = field(default_factory=pd.DataFrame)
    skipped: pd.DataFrame = field(default_factory=pd.DataFrame)


def default_orders(acquisition_order_col: str | None) -> list[AcquisitionOrder]:
    """The recorded order when there is one, plus ``DEFAULT_PERMUTATIONS`` random orders."""
    recorded = [AcquisitionOrder.recorded()] if acquisition_order_col is not None else []
    return recorded + [AcquisitionOrder.permuted(seed) for seed in range(DEFAULT_PERMUTATIONS)]


def strategy_label(strategy: SamplingStrategy) -> str:
    """The strategy's name with its scalar settings, e.g. ``fixed_fov (target_fovs=5)``."""
    settings = ", ".join(
        f"{key}={value:g}" if isinstance(value, float) else f"{key}={value}"
        for key, value in strategy.parameters.items()
        if isinstance(value, int | float | str) and not isinstance(value, bool)
    )
    return f"{strategy.name} ({settings})" if settings else strategy.name


def order_id(order: AcquisitionOrder) -> str:
    """A text label of one order: ``recorded`` or ``permuted-<seed>``."""
    return order.kind if order.seed is None else f"{order.kind}-{order.seed}"


def _dispersed_start(
    group: Mapping[str, Any], order: AcquisitionOrder, split: int, n_fovs: int
) -> int:
    """The first field of a dispersed order: its only free choice.

    Stepping it by the seed (from a group- and split-specific offset) makes
    seeds below the stream length give distinct orders, never repeated ones.
    """
    return (order.seed + group_identity(group) + split) % n_fovs


def _ordered_stream(
    group: Mapping[str, Any],
    stream: ValidatedInput,
    order: AcquisitionOrder,
    order_col: str | None,
    split: int,
    stage_cols: Sequence[str] | None,
    random_state: int,
) -> pd.DataFrame:
    """The stream's FOV rows in the order to replay."""
    if order.kind == "recorded":
        return _in_acquisition_order(group, stream, order_col)
    if order.kind == "dispersed":
        xy = stream.fovs[list(stage_cols)].to_numpy(dtype=float)
        start = _dispersed_start(group, order, split, len(xy))
        return stream.fovs.iloc[dispersed_order(xy, start)]
    return permuted_fovs(stream.fovs, stream.fov_key, random_state, group, order.seed, split)


def _check(
    strategies: Sequence[SamplingStrategy],
    criteria: Sequence[OutcomeCriterion],
    orders: Sequence[AcquisitionOrder] | None,
    acquisition_order_col: str | None,
    n_splits: int,
    random_state: int,
    min_fovs: int,
    stage_cols: Sequence[str] | None,
) -> tuple[list[SamplingStrategy], list[OutcomeCriterion], list[AcquisitionOrder]]:
    """The strategies, criteria and orders as lists, after refusing any invalid argument.

    ``orders=None`` becomes ``default_orders(acquisition_order_col)``.
    """
    strategies = list(strategies)
    if not strategies or not all(isinstance(s, SamplingStrategy) for s in strategies):
        raise TypeError("strategies must be a non-empty sequence of SamplingStrategy")
    if len({s.config_id for s in strategies}) != len(strategies):
        raise ValueError("strategies must be distinct configurations")
    criteria = list(criteria)
    if not criteria or not all(isinstance(c, OutcomeCriterion) for c in criteria):
        raise TypeError("criteria must be a non-empty sequence of OutcomeCriterion")
    if len({c.name for c in criteria}) != len(criteria):
        raise ValueError("criteria must have distinct names")
    orders = default_orders(acquisition_order_col) if orders is None else list(orders)
    if not orders or not all(isinstance(o, AcquisitionOrder) for o in orders):
        raise TypeError("orders must be a non-empty sequence of AcquisitionOrder")
    if len(set(orders)) != len(orders):
        raise ValueError("orders must be distinct")
    if acquisition_order_col is None and any(o.kind == "recorded" for o in orders):
        raise ValueError("the recorded order needs acquisition_order_col")
    if stage_cols is None and any(o.kind == "dispersed" for o in orders):
        raise ValueError("a dispersed order needs stage_cols")
    as_count(n_splits, "n_splits")
    as_count(min_fovs, "min_fovs")
    check_seed(random_state)
    return strategies, criteria, orders


def _runs(
    group: Mapping[str, Any],
    group_data: ValidatedInput,
    orders: Sequence[AcquisitionOrder],
    split_settings: tuple[float, str | None, int, int],
    acquisition_order_col: str | None,
    stage_cols: Sequence[str] | None,
) -> Iterator[tuple[dict[str, Any], ValidatedInput, pd.DataFrame, ValidatedInput]]:
    """Every replayed acquisition of one group: its label, stream order and reservoir.

    Reservoirs and orders are fixed here, before any run is handed out, so a
    run's result does not depend on where it is computed.
    """
    reservoir_fraction, block_col, n_splits, random_state = split_settings
    for split in range(n_splits):
        splitter = _reservoir_split(float(reservoir_fraction), random_state, split, block_col)
        stream, reservoir = splitter.split(group, group_data)
        starts: set[int] = set()
        for order in orders:
            if order.kind == "dispersed":
                start = _dispersed_start(group, order, split, len(stream.fovs))
                if start in starts:
                    continue  # the same acquisition as an earlier dispersed seed
                starts.add(start)
            ordered = _ordered_stream(
                group, stream, order, acquisition_order_col, split, stage_cols, random_state
            )
            where = {
                **group,
                "split": split,
                "order_id": order_id(order),
                "order": order.kind,
                "order_seed": order.seed,
            }
            yield where, stream, ordered, reservoir


def _decide(
    where: Mapping[str, Any],
    group: Mapping[str, Any],
    acquisition: _Acquisition | tuple,
    strategies: Sequence[SamplingStrategy],
    objective: ObjectiveConfig,
    min_fovs: int,
    incremental: bool,
    keep_results: bool = False,
) -> list[tuple[int, dict[str, Any]]]:
    """``(step, row)`` of every strategy's decision after every prefix of one run.

    With ``keep_results`` each row also holds the ``StrategyResult`` (without
    timings), so thresholds can be swept afterwards with ``redecide``.
    """
    if not isinstance(acquisition, _Acquisition):
        acquisition = _Acquisition(*acquisition)
    rows = []
    for step, result in _replay_strategies(
        strategies,
        group,
        acquisition,
        objective=objective,
        min_fovs=min_fovs,
        incremental=incremental,
    ):
        # What the rule itself said (its own components), not the objective's verdict.
        decision = None if result.rule_decision is None else result.rule_decision.value
        row = {
            **where,
            "strategy": result.strategy,
            "config_id": result.config_id,
            "fovs": step,
            "decision": decision,
        }
        if keep_results:
            row["result"] = result.without_timing()
        rows.append((step, row))
    return rows


def _score(
    where: Mapping[str, Any],
    acquisition: _Acquisition | tuple,
    reservoir: ValidatedInput,
    criteria: Sequence[OutcomeCriterion],
    stage_cols: Sequence[str] | None,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Outcomes, stream sizes and stage path of one run; independent of every strategy."""
    if not isinstance(acquisition, _Acquisition):
        acquisition = _Acquisition(*acquisition)
    ordered = acquisition.ordered
    streams = [{**where, "stream_fovs": acquisition.n_fovs, "reservoir_fovs": len(reservoir.fovs)}]
    paths = []
    if stage_cols is not None:
        travel = stage_travel(ordered[list(stage_cols)].to_numpy(dtype=float))
        paths = [{**where, "fovs": i + 1, "travel": float(t)} for i, t in enumerate(travel)]
    outcomes = []
    for step in range(1, acquisition.n_fovs + 1):
        prefix = acquisition.prefix(step)
        for criterion in criteria:
            status, value = criterion.assess(reservoir, prefix)
            outcomes.append(
                {
                    **where,
                    "fovs": step,
                    "criterion": criterion.name,
                    "value": value,
                    "status": status.value,
                }
            )
    return outcomes, streams, paths


def _replay_group(
    group: Mapping[str, Any],
    group_data: ValidatedInput,
    orders: Sequence[AcquisitionOrder],
    split_settings: tuple[float, str | None, int, int],
    acquisition_order_col: str | None,
    stage_cols: Sequence[str] | None,
    strategies: Sequence[SamplingStrategy],
    criteria: Sequence[OutcomeCriterion],
    objective: ObjectiveConfig,
    min_fovs: int,
    incremental: bool,
    keep_results: bool = False,
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    """Decisions, outcomes, streams and stage paths of every run of one analysis group."""
    decisions, outcomes, streams, paths = [], [], [], []
    for where, stream, ordered, reservoir in _runs(
        group, group_data, orders, split_settings, acquisition_order_col, stage_cols
    ):
        # One acquisition serves the decisions and the outcomes of this run.
        acquisition = _Acquisition(stream, ordered)
        decided = _decide(
            where, group, acquisition, strategies, objective, min_fovs, incremental, keep_results
        )
        decisions += [row for _, row in decided]
        scored = _score(where, acquisition, reservoir, criteria, stage_cols)
        for collected, rows in zip((outcomes, streams, paths), scored, strict=True):
            collected += rows
    return decisions, outcomes, streams, paths


def _run_unit(kind: str, *args: Any) -> Any:
    units = {"group": _replay_group, "decide": _decide, "score": _score}
    return units[kind](*args)


def replay_acquisitions(
    objects: pd.DataFrame,
    fovs: pd.DataFrame,
    *,
    feature_cols: Sequence[str],
    strategies: Sequence[SamplingStrategy],
    criteria: Sequence[OutcomeCriterion],
    objective: ObjectiveConfig,
    reservoir_fraction: float,
    orders: Sequence[AcquisitionOrder] | None = None,
    acquisition_order_col: str | None = None,
    n_splits: int = 5,
    random_state: int = 0,
    min_fovs: int = 1,
    incremental: bool = False,
    keep_results: bool = False,
    block_col: str | None = None,
    stage_cols: tuple[str, str] | None = None,
    stage_unit: str = "µm",
    fov_col: str | None = None,
    group_cols: Sequence[str] | None = None,
    observation_unit: str | None = None,
    n_jobs: int = 1,
) -> ReplayResult:
    """Decisions and held-back outcomes after every prefix of every replayed acquisition.

    ``reservoir_fraction`` is the share of each group's FOVs (or blocks) held
    back per split, a design choice with no default. ``orders`` defaults to
    the recorded order (when ``acquisition_order_col`` is given) plus
    ``DEFAULT_PERMUTATIONS`` random orders; a permuted order is seeded from
    ``random_state``, the group, its own seed and the split, so another
    ``random_state`` gives other random orders as well as other reservoirs.
    ``n_splits``, ``random_state`` and ``incremental`` are computational
    settings; decisions are the same with or without ``incremental``. A group
    too small to split is skipped and listed in ``skipped`` with its reason.
    ``keep_results`` stores each ``StrategyResult`` in a ``result`` column of
    the decisions, for threshold sweeps
    (``morphocoverage.retrospective.frontier.sweep``). Strategies are evaluated from
    ``min_fovs`` FOVs on; outcomes from the first FOV. ``fovs`` lists every
    acquired FOV, including FOVs without observations. ``stage_cols`` (x, y
    columns of the FOV table, in ``stage_unit``) enables dispersed orders and
    records the stage path of every replayed order. ``n_jobs`` spreads the
    replayed acquisitions (group, split and order) over that many worker
    processes (``-1``: every CPU), each strategy apart when they are too few
    to keep every worker busy; results are identical for every value
    (bit for bit when BLAS is single-threaded; see ``morphocoverage._parallel``).
    """
    check_n_jobs(n_jobs)
    if not isinstance(objective, ObjectiveConfig):
        raise TypeError("objective must be an ObjectiveConfig, for example ObjectiveConfig()")
    strategies, criteria, orders = _check(
        strategies,
        criteria,
        orders,
        acquisition_order_col,
        n_splits,
        random_state,
        min_fovs,
        stage_cols,
    )
    # numpy integers are accepted and recorded as the Python values they hold.
    n_splits, min_fovs, random_state = int(n_splits), int(min_fovs), int(random_state)
    if acquisition_order_col is not None and acquisition_order_col in feature_cols:
        raise ValueError("acquisition order cannot also be a phenotype feature")
    data = validate_input_tables(
        objects,
        fovs=fovs,
        feature_cols=feature_cols,
        fov_col=fov_col,
        group_cols=group_cols,
        observation_unit=observation_unit,
    )
    if acquisition_order_col is not None:
        if acquisition_order_col not in data.fovs.columns:
            raise ValueError(f"fovs is missing required columns: {[acquisition_order_col]}")
        _check_order_values(data.fovs, acquisition_order_col)
    if stage_cols is not None:
        if len(stage_cols) != 2:
            raise TypeError("stage_cols must be two column names (x, y)")
        absent = [c for c in stage_cols if c not in data.fovs.columns]
        if absent:
            raise ValueError(f"stage columns missing from the FOV table: {absent}")
        if data.fovs[list(stage_cols)].isna().any().any():
            raise ValueError("stage coordinates contain missing values")
        if np.isinf(data.fovs[list(stage_cols)].to_numpy(dtype=float)).any():
            raise ValueError("stage coordinates must be finite: correct or drop the infinite ones")
        if any(c in feature_cols for c in stage_cols):
            raise ValueError("stage coordinates cannot also be phenotype features")
    split_settings = (float(reservoir_fraction), block_col, n_splits, random_state)
    # Units are as coarse as keeps every worker busy, since each sends its data
    # once: whole groups; else one unit per run (group, split and order) and
    # one scoring its outcomes, which no strategy affects; else, when runs are
    # fewer than workers too, every strategy apart. Decisions of strategies
    # replayed apart are put back in the order of one joint replay, so the
    # tables are identical for every n_jobs.
    groups, skipped = _replayable(data, float(reservoir_fraction), block_col)
    settings_of_run = (orders, split_settings, acquisition_order_col, stage_cols)
    replay_settings = (objective, min_fovs, incremental, keep_results)
    decisions, outcomes, streams, paths = [], [], [], []
    if len(groups) >= check_n_jobs(n_jobs):
        jobs = [
            ("group", group, group_data, *settings_of_run, strategies, criteria, *replay_settings)
            for group, group_data in groups
        ]
        for part in map_units(_run_unit, jobs, n_jobs):
            for collected, rows in zip((decisions, outcomes, streams, paths), part, strict=True):
                collected += rows
    else:
        units = _strategy_units(strategies, n_jobs, len(groups) * n_splits * len(orders))
        jobs, layout = [], []
        for group, group_data in groups:
            for where, stream, ordered, reservoir in _runs(group, group_data, *settings_of_run):
                # A run sent to a worker is built there, from the stream and its order.
                run = (stream, ordered)
                jobs += [("decide", where, group, run, part, *replay_settings) for part in units]
                jobs.append(("score", where, run, reservoir, criteria, stage_cols))
                layout.append(len(units))
        parts = iter(map_units(_run_unit, jobs, n_jobs))
        for n_units in layout:
            decided = [next(parts) for _ in range(n_units)]
            decisions += [row for _, row in _interleave(decided)]
            for collected, rows in zip((outcomes, streams, paths), next(parts), strict=True):
                collected += rows
    keys = list(data.group_cols)
    settings = {
        "criteria": [c.parameters for c in criteria],
        "labels": {c.name: c.label for c in criteria},
        "strategies": {s.config_id: s.configuration for s in strategies},
        "strategy_names": {s.config_id: strategy_label(s) for s in strategies},
        "strategy_kinds": {s.config_id: s.name for s in strategies},
        "objective": objective.objective.value,
        "orders": [{"kind": o.kind, "seed": o.seed} for o in orders],
        "reservoir_fraction": float(reservoir_fraction),
        "block_col": block_col,
        "n_splits": n_splits,
        "random_state": random_state,
        "min_fovs": min_fovs,
        "acquisition_order_col": acquisition_order_col,
        "stage_cols": None if stage_cols is None else list(stage_cols),
        "stage_unit": stage_unit,
    }
    return ReplayResult(
        decisions=pd.DataFrame(
            decisions,
            columns=[*keys, *DECISION_COLUMNS, *(("result",) if keep_results else ())],
        ),
        outcomes=pd.DataFrame(outcomes, columns=[*keys, *OUTCOME_COLUMNS]),
        streams=pd.DataFrame(streams, columns=[*keys, *STREAM_COLUMNS]),
        group_cols=tuple(keys),
        settings=settings,
        paths=pd.DataFrame(paths, columns=[*keys, *_RUN, "fovs", "travel"]),
        skipped=pd.DataFrame(skipped, columns=[*keys, "fovs", "reason"]),
    )


def _replayable(
    data: ValidatedInput, reservoir_fraction: float, block_col: str | None
) -> tuple[list[tuple[dict[str, Any], ValidatedInput]], list[dict[str, Any]]]:
    """The groups that can hold out a reservoir and keep a stream, and the others with why.

    The skip rule of ``sampling_reliability`` (``reliability._skip_reason``):
    a group with a single FOV (or block) cannot be split. It is skipped with
    its reason rather than failing every other group; with no group left the
    replay raises, since there would be nothing to report.
    """
    groups, skipped = [], []
    for group, group_data in iter_analysis_groups(data):
        reason = _skip_reason(group_data, reservoir_fraction, block_col)
        if reason is None:
            groups.append((group, group_data))
        else:
            skipped.append({**group, "fovs": len(group_data.fovs), "reason": reason})
    if not groups:
        raise ValueError(
            f"no analysis group has two or more {'FOVs' if block_col is None else 'blocks'} "
            f"to hold out a reservoir and keep a stream: {skipped[:3]}"
        )
    return groups, skipped


__all__ = [
    "DECISION_COLUMNS",
    "DEFAULT_PERMUTATIONS",
    "OUTCOME_COLUMNS",
    "STREAM_COLUMNS",
    "ReplayResult",
    "default_orders",
    "order_id",
    "replay_acquisitions",
    "strategy_label",
]
