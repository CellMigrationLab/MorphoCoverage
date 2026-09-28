"""The criteria tables of the sufficiency report: each rule on the full acquisition and its stops.

``criteria_tables`` runs every strategy on the whole acquisition of each
analysis group (the rule's own verdict, its evidence and its settings) and
replays the declared acquisition orders to record where each rule first
stopped, separately for the recorded order and the random (permuted) orders
(``order_kind``): the recorded order keeps drift and neighbour effects that
random orders remove, so the two are never pooled into one distribution.
``sufficiency`` turns the two tables into the answer to "did each criterion
say enough?".
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from .benchmark import (
    AcquisitionOrder,
    BenchmarkDataset,
    BenchmarkRegistry,
    records_to_frame,
    run_benchmark,
)
from .evidence import Availability
from .framework import run_strategies
from .objectives import ObjectiveConfig
from .policies import EffortPolicy
from .results import StrategyResult
from .strategies.base import SamplingStrategy
from .sufficiency_text import _policies


def is_effort_rule(strategy: SamplingStrategy) -> bool:
    """Whether ``strategy`` stops at a fixed amount of acquisition effort (FOVs, objects, area).

    Read from the rule's structure, never from its name or its settings text:
    every policy is an ``EffortPolicy``, or the strategy declares
    ``effort_rule = True`` (the fixed-object and fixed-area baselines, whose
    policies are not ``EffortPolicy``). Such a rule's "stop" says its target
    was reached, not that the data showed the sample to be enough.
    """
    if getattr(strategy, "effort_rule", False) is True:
        return True
    policies = strategy.policies
    return bool(policies) and all(isinstance(policy, EffortPolicy) for policy in policies)


def _insufficient_reason(result: StrategyResult) -> str | None:
    """Why the rule cannot decide yet: the reason of a quantity it reads that lacks data."""
    named = {
        policy.get("parameters", {}).get("quantity")
        for policy in result.configuration.get("policies", [])
    }
    for name, quantity in result.evidence.quantities.items():
        if name in named and quantity.availability is Availability.INSUFFICIENT_DATA:
            return quantity.reason
    return None


def _warnings(diagnostics: Mapping[str, Any]) -> str | None:
    """The caveats a rule's evidence carries (``field_size_warning``, ``warning``), joined.

    The FOV-unit JCB rule warns when FOV size and FOV mean are related (its
    CV can then be several times too small) and the JCB rule when the SD
    exceeds 50% CV; its pilot's own warnings are marked as the pilot's.
    """
    found = [diagnostics.get(key) for key in ("field_size_warning", "warning")]
    pilot = diagnostics.get("pilot")
    if isinstance(pilot, Mapping):
        found += [
            f"pilot: {pilot[key]}" for key in ("field_size_warning", "warning") if pilot.get(key)
        ]
    found = [str(text) for text in found if text]
    return "; ".join(found) if found else None


def _final_row(
    result: StrategyResult, group_cols: Sequence[str], effort_rule: bool = False
) -> dict[str, Any]:
    """One row of the full-acquisition table."""
    diagnostics = result.evidence.diagnostics
    row = {column: result.group.get(column) for column in group_cols}
    row.update(
        {
            "strategy": result.strategy,
            "config_id": result.config_id,
            "policies": _policies(result),
            "effort_rule": effort_rule,
            "n_fovs": result.n_fovs,
            "n_observations": result.n_observations,
            # The rule's own verdict, by its own threshold, whatever the objective asks.
            "verdict": None if result.rule_decision is None else result.rule_decision.value,
            "undefined_reason": diagnostics.get("undefined_reason"),
            # A "continue" only because a quantity it reads has too little data says so.
            "insufficient_reason": _insufficient_reason(result),
            # The object-unit JCB and JCS rules treat objects as independent draws; the
            # sentence says so, since that is what makes a stop after one FOV possible.
            "assumes_spatial_uniformity": bool(
                diagnostics.get("assumes_spatial_uniformity", False)
                or diagnostics.get("assumes_random_object_sampling", False)
            ),
            "warnings": _warnings(diagnostics),
        }
    )
    for name, quantity in result.evidence.quantities.items():
        available = quantity.availability is Availability.AVAILABLE
        row[f"q.{name}"] = quantity.value if available else None
    return row


def _quantile(values: np.ndarray, q: float) -> float | None:
    """The ``q`` quantile as one of the values (``inverted_cdf``), or ``None`` without values."""
    return float(np.quantile(values, q, method="inverted_cdf")) if len(values) else None


def _stop_summary(records: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Stop-FOV distribution per group, configuration and order kind."""
    rows = []
    for key, runs in records.groupby(keys, sort=False, dropna=False):
        stopped = runs[runs["stopped"]]
        stop = stopped["n_fovs"].to_numpy(dtype=float)
        spare = (stopped["total_fovs"] - stopped["n_fovs"]).to_numpy(dtype=float)
        row = dict(zip(keys, key if isinstance(key, tuple) else (key,), strict=True))
        row.update(
            {
                "acquired_fovs": int(runs["total_fovs"].iloc[0]),
                "orders": len(runs),
                "never_stopped": int((~runs["stopped"]).sum()),
                "stop_fov_median": float(np.median(stop)) if len(stop) else None,
                # Observed stop counts, never interpolated between runs.
                "stop_fov_p10": _quantile(stop, 0.1),
                "stop_fov_p90": _quantile(stop, 0.9),
                "spare_fovs_median": float(np.median(spare)) if len(spare) else None,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def group_major(table: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Rows group by group in first-seen order, keeping each group's row order."""
    if not group_cols or table.empty:
        return table
    key = table[group_cols].astype(str).agg("\x1f".join, axis=1)
    rank = {value: i for i, value in enumerate(dict.fromkeys(key))}
    order = key.map(rank).sort_values(kind="stable").index
    return table.loc[order].reset_index(drop=True)


def criteria_tables(
    objects: pd.DataFrame,
    fovs: pd.DataFrame,
    strategies: Sequence[SamplingStrategy],
    objective: ObjectiveConfig,
    orders: Sequence[AcquisitionOrder],
    acquisition_order_col: str | None,
    columns: Mapping[str, Any],
    group_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Decisions on the full acquisition, and the stop-FOV distribution over orders.

    The stops table has one row per group, criterion and ``order_kind``
    (``"recorded"``, ``"permuted"``, in the order declared), each kind
    summarised on its own.
    """
    results = run_strategies(
        objects, fovs=fovs, strategies=strategies, objective=objective, **columns
    )
    effort = {(rule.name, rule.config_id): is_effort_rule(rule) for rule in strategies}
    final = pd.DataFrame(
        [
            _final_row(result, group_cols, effort[(result.strategy, result.config_id)])
            for result in results
        ]
    )
    dataset = BenchmarkDataset(
        name="sufficiency_report",
        role="development",
        load=lambda: (objects, fovs),
        feature_cols=tuple(columns["feature_cols"]),
        acquisition_order_col=acquisition_order_col,
        fov_col=columns["fov_col"],
        group_cols=None if columns["group_cols"] is None else tuple(columns["group_cols"]),
        observation_unit=columns["observation_unit"],
    )
    records = records_to_frame(
        run_benchmark(
            BenchmarkRegistry([dataset]),
            dataset.name,
            strategies=strategies,
            metrics=[],
            objective=objective,
            orders=orders,
            # The verdict above is the rule's own; its stop points must be too.
            stop_on="rule",
        )
    )
    records = records.rename(columns={"order": "order_kind"})
    group_keys = [f"group.{column}" for column in group_cols]
    stops = _stop_summary(records, [*group_keys, "strategy", "config_id", "order_kind"])
    stops = stops.rename(columns={f"group.{c}": c for c in group_cols})
    return final, stops


__all__ = ["criteria_tables", "group_major", "is_effort_rule"]
