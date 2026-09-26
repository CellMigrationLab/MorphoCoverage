"""The criteria tables of the sufficiency report: each rule on the full acquisition and its stops.

``criteria_tables`` runs every strategy on the whole acquisition of each
analysis group (the rule's own verdict, its evidence and its settings) and
replays the declared acquisition orders to record where each rule first
stopped; ``sufficiency`` turns the two tables into the answer to "did each
criterion say enough?".
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
from .results import StrategyResult
from .strategies.base import SamplingStrategy
from .sufficiency_text import _policies


def _final_row(result: StrategyResult, group_cols: Sequence[str]) -> dict[str, Any]:
    """One row of the full-acquisition table."""
    row = {column: result.group.get(column) for column in group_cols}
    row.update(
        {
            "strategy": result.strategy,
            "config_id": result.config_id,
            "policies": _policies(result),
            "n_fovs": result.n_fovs,
            "n_observations": result.n_observations,
            # The rule's own verdict, by its own threshold, whatever the objective asks.
            "verdict": None if result.rule_decision is None else result.rule_decision.value,
            "undefined_reason": result.evidence.diagnostics.get("undefined_reason"),
            # The object-unit JCB rule treats objects as independent draws; the
            # sentence says so, since that is what makes a stop after one FOV possible.
            "assumes_spatial_uniformity": bool(
                result.evidence.diagnostics.get("assumes_spatial_uniformity", False)
            ),
        }
    )
    for name, quantity in result.evidence.quantities.items():
        available = quantity.availability is Availability.AVAILABLE
        row[f"q.{name}"] = quantity.value if available else None
    return row


def _stop_summary(records: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Stop-FOV distribution per group and configuration."""
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
                "stop_fov_p10": float(np.quantile(stop, 0.1)) if len(stop) else None,
                "stop_fov_p90": float(np.quantile(stop, 0.9)) if len(stop) else None,
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
    """Decisions on the full acquisition, and the stop-FOV distribution over orders."""
    results = run_strategies(
        objects, fovs=fovs, strategies=strategies, objective=objective, **columns
    )
    final = pd.DataFrame([_final_row(result, group_cols) for result in results])
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
        )
    )
    group_keys = [f"group.{column}" for column in group_cols]
    stops = _stop_summary(records, [*group_keys, "strategy", "config_id"])
    stops = stops.rename(columns={f"group.{c}": c for c in group_cols})
    return final, stops


__all__ = ["criteria_tables", "group_major"]
