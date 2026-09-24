"""Was an analysed dataset sampled enough? A retrospective report across methods.

Given a finished acquisition (observation table plus the FOV table of every
acquired position), ``sufficiency_report`` answers two questions for each
analysis group (condition x repeat by default) and each strategy:

1. **Verdict on the full acquisition.** Is the strategy's criterion met by
   all the FOVs that were acquired? ``stop`` means sufficient by that
   criterion, ``continue`` means not yet, and an empty verdict means the
   criterion is undefined for these data (the reason is reported).
   ``final`` also lists every evidence quantity, including the probability
   that another FOV would show a new phenotype and its upper bound where the
   strategy reports them.
2. **Stop point.** Replaying the FOVs in each declared acquisition order,
   after how many FOVs would the strategy first have stopped? ``stops``
   summarizes this over orders; ``spare_fovs`` is how many acquired FOVs lay
   beyond the stop (negative values do not occur: a strategy that never
   stops is counted in ``never_stopped``).

Reading it:

- Every verdict holds only for that strategy's criterion and thresholds.
  Thresholds without a published meaning are the user's choice, so compare
  several (pass the same strategy with different thresholds) rather than
  read one verdict as a fact. Rows are keyed by ``config_id``, so variants
  that share a name stay apart.
- Disagreement between strategies is informative, for example a cell-count
  rule satisfied after one FOV while FOV-based rules need several.
- Random orders treat FOVs as exchangeable; the recorded order keeps drift
  and neighbour effects. Declare both to see the difference.
- Upper confidence bounds need enough FOVs to go low at all: with no novel
  FOV among ``k``, the 95% bound is ``1 - 0.05 ** (1 / k)``, so asserting
  "below 10%" needs at least 29 FOVs whatever the data look like.
- This is retrospective self-assessment of one dataset. It says what each
  criterion concludes, not that the criterion is right; that needs the
  benchmark on held-out deep data (``docs/benchmarking.md``).
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

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

_ROW_ORDER = "_report_row_order"


@dataclass(frozen=True)
class SufficiencyReport:
    """The report tables; see the module docstring for how to read them.

    - ``final``: one row per group and strategy configuration, with the
      verdict on the full acquisition, effort, thresholds and every evidence
      quantity (``q.<name>``; missing, shown as NaN, when the strategy does
      not report it or it is unavailable, with the reason in
      ``undefined_reason``).
    - ``stops``: one row per group and configuration, summarizing the stop
      FOV over the declared acquisition orders.
    """

    final: pd.DataFrame
    stops: pd.DataFrame


def _policies(result: StrategyResult) -> str:
    """The decision policies and their settings, compactly."""
    policies = result.configuration.get("policies", [])
    return "; ".join(f"{p['name']} {json.dumps(p['parameters'], sort_keys=True)}" for p in policies)


def _final_row(result: StrategyResult, group_cols: Sequence[str]) -> dict:
    """One row of the full-acquisition table."""
    row = {column: result.group.get(column) for column in group_cols}
    row.update(
        {
            "strategy": result.strategy,
            "config_id": result.config_id,
            "policies": _policies(result),
            "n_fovs": result.n_fovs,
            "n_observations": result.n_observations,
            "verdict": None if result.decision is None else result.decision.value,
            "undefined_reason": result.evidence.diagnostics.get("undefined_reason"),
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


def sufficiency_report(
    objects: pd.DataFrame,
    fovs: pd.DataFrame,
    *,
    feature_cols: Sequence[str],
    strategies: Iterable[SamplingStrategy],
    objective: ObjectiveConfig,
    orders: Sequence[AcquisitionOrder],
    acquisition_order_col: str | None = None,
    fov_col: str | None = None,
    group_cols: Sequence[str] | None = None,
    observation_unit: str | None = None,
) -> SufficiencyReport:
    """Retrospective sufficiency of one analysed dataset under several strategies.

    ``fovs`` must list every acquired FOV, including FOVs without
    observations. ``orders`` are the replay orders to summarize, for example
    ``[AcquisitionOrder.recorded()] + [AcquisitionOrder.permuted(s) for s in
    range(20)]``; the recorded order needs ``acquisition_order_col``. Without
    that column only permuted orders are allowed, since the table's row order
    is not an acquisition record. Other arguments are those of
    ``run_strategies``.
    """
    strategies = list(strategies)
    orders = list(orders)
    if not strategies:
        raise ValueError("give at least one strategy")
    if not orders or not all(isinstance(order, AcquisitionOrder) for order in orders):
        raise ValueError("orders must be a non-empty sequence of AcquisitionOrder")
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
    results = run_strategies(
        objects, fovs=fovs, strategies=strategies, objective=objective, **columns
    )
    resolved_groups = sorted({column for result in results for column in result.group})
    final = pd.DataFrame([_final_row(result, resolved_groups) for result in results])

    dataset = BenchmarkDataset(
        name="sufficiency_report",
        role="development",
        load=lambda: (objects, fovs),
        feature_cols=tuple(feature_cols),
        acquisition_order_col=acquisition_order_col,
        fov_col=fov_col,
        group_cols=None if group_cols is None else tuple(group_cols),
        observation_unit=observation_unit,
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
    group_keys = [f"group.{column}" for column in resolved_groups]
    stops = _stop_summary(records, [*group_keys, "strategy", "config_id"])
    stops = stops.rename(columns={f"group.{c}": c for c in resolved_groups})
    return SufficiencyReport(final=final, stops=stops)


__all__ = ["SufficiencyReport", "sufficiency_report"]
