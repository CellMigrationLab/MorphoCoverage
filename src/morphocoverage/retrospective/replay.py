"""Replay acquisitions against a fixed held-back reservoir: decisions and outcomes per prefix.

The shared engine of stopping-rule calibration (roadmap Step 2) and the
adequacy points (Step 3). Per analysis group and outer split:

1. hold out a fixed reservoir of whole FOVs (or spatial blocks) with
   ``ReservoirSplit``, exactly as ``sampling_reliability`` does for the same
   ``random_state`` and split; the rest is the stream;
2. for each acquisition order of the stream (recorded, or seeded random
   permutations), reveal whole FOVs one at a time and record every strategy's
   decision after each prefix, with the same estimators and policies a live
   session would run;
3. for the same prefixes, assess every ``OutcomeCriterion`` against the
   reservoir. These outcomes depend on the prefix only, never on a strategy.

Decisions use only the visible prefix (the prospective path); outcomes use the
hidden reservoir (retrospective scoring). The two tables are kept apart so
that "the rule said enough" and "it was enough" are never mixed.

Permutations are drawn per group and split, reproducibly from the order's seed,
so adding a group or a split never changes another's orders.
"""

from __future__ import annotations

import json
import zlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..benchmark import AcquisitionOrder
from ..evaluator import iter_analysis_groups
from ..objectives import ObjectiveConfig
from ..reservoir import ReservoirSplit
from ..schema import ValidatedInput, validate_input_tables
from ..sequential import _Acquisition, _in_acquisition_order, _replay_full, _replay_incremental
from ..strategies.base import SamplingStrategy
from .criteria import OutcomeCriterion

DEFAULT_PERMUTATIONS = 10

_RUN = ("split", "order_id", "order", "order_seed")
DECISION_COLUMNS = (*_RUN, "strategy", "config_id", "fovs", "decision")
OUTCOME_COLUMNS = (*_RUN, "fovs", "criterion", "value", "status")
STREAM_COLUMNS = (*_RUN, "stream_fovs", "reservoir_fovs")


@dataclass(frozen=True)
class ReplayResult:
    """Decisions and outcomes after every prefix of every replayed acquisition.

    - ``decisions``: one row per group, split, order, strategy and prefix
      length (``fovs``); ``decision`` is ``"stop"``, ``"continue"``,
      ``"unsupported"`` or ``None`` (undefined).
    - ``outcomes``: one row per group, split, order, prefix length and
      criterion, with the metric ``value`` and ``status`` (``preserved``,
      ``not_preserved``, ``undefined``) against that split's reservoir.
    - ``streams``: one row per group, split and order with the stream and
      reservoir sizes.

    One replayed acquisition is identified by the group, ``split`` and
    ``order_id``; ``order`` and ``order_seed`` give the order's kind and seed.
    """

    decisions: pd.DataFrame
    outcomes: pd.DataFrame
    streams: pd.DataFrame
    group_cols: tuple[str, ...]
    settings: Mapping[str, Any] = field(default_factory=dict)


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


def _identity(group: Mapping[str, Any]) -> int:
    return zlib.crc32(json.dumps(sorted(group.items()), default=str).encode())


def _ordered_stream(group, stream: ValidatedInput, order, order_col, split) -> pd.DataFrame:
    """The stream's FOV rows in the order to replay."""
    if order.kind == "recorded":
        return _in_acquisition_order(group, stream, order_col)
    rng = np.random.default_rng([order.seed, _identity(group), split])
    return stream.fovs.iloc[rng.permutation(len(stream.fovs))]


def _check(strategies, criteria, orders, acquisition_order_col, n_splits, random_state, min_fovs):
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
    for value, label in ((n_splits, "n_splits"), (min_fovs, "min_fovs")):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{label} must be a positive integer")
    if isinstance(random_state, bool) or not isinstance(random_state, int):
        raise TypeError("random_state must be an integer seed")
    return strategies, criteria, orders


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
    block_col: str | None = None,
    fov_col: str | None = None,
    group_cols: Sequence[str] | None = None,
    observation_unit: str | None = None,
) -> ReplayResult:
    """Decisions and held-back outcomes after every prefix of every replayed acquisition.

    ``reservoir_fraction`` is the share of each group's FOVs (or blocks) held
    back per split, a design choice with no default. ``orders`` defaults to
    the recorded order (when ``acquisition_order_col`` is given) plus
    ``DEFAULT_PERMUTATIONS`` random orders. ``n_splits``, ``random_state``
    and ``incremental`` are computational settings; decisions are the same
    with or without ``incremental``. Strategies are evaluated from
    ``min_fovs`` FOVs on; outcomes from the first FOV. ``fovs`` lists every
    acquired FOV, including FOVs without observations.
    """
    strategies, criteria, orders = _check(
        strategies, criteria, orders, acquisition_order_col, n_splits, random_state, min_fovs
    )
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
        if data.fovs[acquisition_order_col].isna().any():
            raise ValueError("acquisition order contains missing values")
    replay = _replay_incremental if incremental else _replay_full
    decisions, outcomes, streams = [], [], []
    for group, group_data in iter_analysis_groups(data):
        for split in range(n_splits):
            splitter = ReservoirSplit(float(reservoir_fraction), random_state + split, block_col)
            stream, reservoir = splitter.split(group, group_data)
            for order in orders:
                ordered = _ordered_stream(group, stream, order, acquisition_order_col, split)
                acquisition = _Acquisition(stream, ordered)
                where = {
                    **group,
                    "split": split,
                    "order_id": order_id(order),
                    "order": order.kind,
                    "order_seed": order.seed,
                }
                streams.append(
                    {
                        **where,
                        "stream_fovs": acquisition.n_fovs,
                        "reservoir_fovs": len(reservoir.fovs),
                    }
                )
                for step, result in replay(
                    strategies, group, acquisition, objective=objective, min_fovs=min_fovs
                ):
                    decision = None if result.decision is None else result.decision.value
                    decisions.append(
                        {
                            **where,
                            "strategy": result.strategy,
                            "config_id": result.config_id,
                            "fovs": step,
                            "decision": decision,
                        }
                    )
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
    keys = list(data.group_cols)
    settings = {
        "criteria": [c.parameters for c in criteria],
        "labels": {c.name: c.label for c in criteria},
        "strategies": {s.config_id: s.configuration for s in strategies},
        "strategy_names": {s.config_id: strategy_label(s) for s in strategies},
        "objective": objective.objective.value,
        "orders": [{"kind": o.kind, "seed": o.seed} for o in orders],
        "reservoir_fraction": float(reservoir_fraction),
        "block_col": block_col,
        "n_splits": n_splits,
        "random_state": random_state,
        "min_fovs": min_fovs,
        "acquisition_order_col": acquisition_order_col,
    }
    return ReplayResult(
        decisions=pd.DataFrame(decisions, columns=[*keys, *DECISION_COLUMNS]),
        outcomes=pd.DataFrame(outcomes, columns=[*keys, *OUTCOME_COLUMNS]),
        streams=pd.DataFrame(streams, columns=[*keys, *STREAM_COLUMNS]),
        group_cols=tuple(keys),
        settings=settings,
    )


__all__ = [
    "DEFAULT_PERMUTATIONS",
    "ReplayResult",
    "default_orders",
    "order_id",
    "replay_acquisitions",
    "strategy_label",
]
