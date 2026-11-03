"""The criteria tables of the sufficiency report: each rule on the full acquisition and its stops.

``criteria_tables`` runs every strategy on the whole acquisition of each
analysis group (the rule's own verdict, its evidence and its settings) and
replays the declared acquisition orders to record where each rule first
stopped, separately for the recorded order and the random (permuted) orders
(``order_kind``): the recorded order keeps drift and neighbour effects that
random orders remove, so the two are never pooled into one distribution.
``enough_table`` turns the two tables into the answer to "did each criterion
say enough?", one row per group and criterion.

With a ``readout``, each group's fields are also checked for a trend of the
readout's field means with the declared acquisition order and for spatial
autocorrelation between stage neighbours (``order_check``); every row of the
full-acquisition table carries, in ``order_caution``, the sentence a stop of
that rule rests on: the check's caution (for a rare-phenotype rule, with the
share of the stage extent the recorded order had covered at its stop), or
without a check the assumption that the fields are a random sample of the
well's positions.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from ._caveats import assumption, is_assumption
from ._identity import typed_labels
from ._seeds import group_key
from ._style import rule_label
from ._wording import rows_of
from .benchmark import (
    AcquisitionOrder,
    BenchmarkDataset,
    BenchmarkRegistry,
    records_to_frame,
    run_benchmark,
)
from .data_check import validate_input_tables
from .evaluator import iter_analysis_groups, restrict_input
from .evidence import Availability
from .framework import run_strategies
from .objectives import ObjectiveConfig
from .order_check import OrderCheck, check_acquisition_order
from .policies import EffortPolicy
from .results import StrategyResult, stop_caution
from .schema import resolve_observation_unit
from .statistics import median_standard_error
from .strategies.base import SamplingStrategy
from .sufficiency_text import _enough_answer, _policies


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


def _warning_list(diagnostics: Mapping[str, Any]) -> tuple[str, ...]:
    """The caveats a rule's evidence carries (``field_size_warning``, ``warning``), one by one.

    The FOV-unit mean-precision rule warns when FOV size and FOV mean are related (its
    CV can then be several times too small) and the mean-precision rule when the SD
    exceeds 50% CV; its pilot's own warnings are marked as the pilot's.
    """
    found = [diagnostics.get(key) for key in ("field_size_warning", "warning")]
    pilot = diagnostics.get("pilot")
    if isinstance(pilot, Mapping):
        found += [
            _kept(pilot[key], f"pilot: {pilot[key]}")
            for key in ("field_size_warning", "warning")
            if pilot.get(key)
        ]
    # A standing assumption a rule states beside every value keeps its kind (_style).
    return tuple(text if isinstance(text, str) else str(text) for text in found if text)


def _kept(text: Any, new: str) -> str:
    """``new`` with the kind of ``text``: an assumption stays one (``_style.Assumption``)."""
    return assumption(new) if is_assumption(text) else new


def _warnings(diagnostics: Mapping[str, Any]) -> str | None:
    """The caveats of ``_warning_list``, joined."""
    found = _warning_list(diagnostics)
    return "; ".join(found) if found else None


def _final_row(
    result: StrategyResult,
    group_cols: Sequence[str],
    effort_rule: bool = False,
    rule: str | None = None,
) -> dict[str, Any]:
    """One row of the full-acquisition table; ``rule`` is the rule's readable name."""
    diagnostics = result.evidence.diagnostics
    row = {column: result.group.get(column) for column in group_cols}
    row.update(
        {
            "strategy": result.strategy,
            "rule": rule or result.strategy,
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
            # One by one, so the short summary can state each kind once (``sufficiency_short``).
            "warning_list": _warning_list(diagnostics),
            "decides_rare": result.decides_rare,
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
                # How far the median would move under another seed (a Monte Carlo error).
                "stop_fov_median_mc_se": median_standard_error(stop),
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
    # Typed labels: group 1 is not group "1" (#420); no identifier holds the separator.
    labels = {c: typed_labels(table[c]) for c in group_cols}
    key = pd.DataFrame(labels, index=table.index).agg("\x1f".join, axis=1)
    rank = {value: i for i, value in enumerate(dict.fromkeys(key))}
    order = key.map(rank).sort_values(kind="stable").index
    return table.loc[order].reset_index(drop=True)


def _order_checks(
    objects: pd.DataFrame,
    fovs: pd.DataFrame,
    columns: Mapping[str, Any],
    readout: str,
    order_col: str | None,
    stage_cols: Sequence[str] | None,
) -> dict[str, OrderCheck]:
    """The order check of every group's fields, in the recorded order when there is one."""
    data = validate_input_tables(objects, fovs=fovs, **{**columns, "feature_cols": [readout]})
    checks = {}
    for group, group_data in iter_analysis_groups(data):
        ordered = group_data.fovs
        if order_col is not None:
            ordered = ordered.sort_values(order_col, kind="stable")
        acquired = restrict_input(group_data, group_data.objects, ordered)
        checks[group_key(group)] = check_acquisition_order(
            acquired, readout, order_col=order_col, stage_cols=stage_cols
        )
    return checks


def _recorded_stop(
    stops: pd.DataFrame, row: Mapping[str, Any], group_cols: list[str]
) -> int | None:
    """Where the recorded order first met this rule's criterion, or ``None``."""
    if stops.empty:
        return None
    keys = {**{c: row[c] for c in group_cols}, "config_id": row["config_id"]}
    mine = stops[stops["order_kind"].eq("recorded")]
    for column, value in keys.items():
        mine = mine[mine[column].eq(value)]
    if len(mine) != 1 or int(mine["orders"].iloc[0]) == int(mine["never_stopped"].iloc[0]):
        return None
    return int(mine["stop_fov_median"].iloc[0])


def _with_order_cautions(
    final: pd.DataFrame,
    stops: pd.DataFrame,
    checks: Mapping[str, OrderCheck],
    group_cols: list[str],
) -> pd.DataFrame:
    """``final`` with ``order_caution``, what a stop of each rule rests on (see the module),
    and ``order_flagged``, whether the order check flagged the group's fields."""
    cautions, flagged = [], []
    for _, row in final.iterrows():
        check = checks.get(group_key({c: row[c] for c in group_cols}))
        rare = bool(row["decides_rare"])
        if check is None:
            cautions.append(stop_caution(None, rare=rare))
        else:
            stop = _recorded_stop(stops, row, group_cols) if rare else None
            cautions.append(check.caution(rare=rare, stop_after=stop))
        flagged.append(check is not None and check.flagged)
    return final.assign(order_caution=cautions, order_flagged=flagged)


def criteria_tables(
    objects: pd.DataFrame,
    fovs: pd.DataFrame,
    strategies: Sequence[SamplingStrategy],
    objective: ObjectiveConfig,
    orders: Sequence[AcquisitionOrder],
    acquisition_order_col: str | None,
    columns: Mapping[str, Any],
    group_cols: list[str],
    *,
    readout: str | None = None,
    order_col: str | None = None,
    stage_cols: Sequence[str] | None = None,
    n_jobs: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Decisions on the full acquisition, and the stop-FOV distribution over orders.

    The stops table has one row per group, criterion and ``order_kind``
    (``"recorded"``, ``"permuted"``, in the order declared), each kind
    summarised on its own. ``readout`` turns on the order check of each
    group's fields, with the declared ``order_col`` (``None`` when the orders
    replayed rest on no recorded order) and ``stage_cols``; the final table's
    ``order_caution`` says what each rule's stop rests on (see the module).
    summarised on its own. ``n_jobs`` replays the groups in worker processes
    (``run_benchmark``).
    """
    results = run_strategies(
        objects, fovs=fovs, strategies=strategies, objective=objective, **columns
    )
    effort = {(rule.name, rule.config_id): is_effort_rule(rule) for rule in strategies}
    unit = resolve_observation_unit(columns["observation_unit"])[0]
    names = {(rule.name, rule.config_id): rule_label(rule, unit) for rule in strategies}
    final = pd.DataFrame(
        [
            _final_row(
                result,
                group_cols,
                effort[(result.strategy, result.config_id)],
                names[(result.strategy, result.config_id)],
            )
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
            n_jobs=n_jobs,
        )
    )
    records = records.rename(columns={"order": "order_kind"})
    group_keys = [f"group.{column}" for column in group_cols]
    stops = _stop_summary(records, [*group_keys, "strategy", "config_id", "order_kind"])
    stops = stops.rename(columns={f"group.{c}": c for c in group_cols})
    checks = (
        {}
        if readout is None or (order_col is None and stage_cols is None)
        else _order_checks(objects, fovs, columns, readout, order_col, stage_cols)
    )
    return _with_order_cautions(final, stops, checks, group_cols), stops


def enough_table(final: pd.DataFrame, stops: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Question 1, one row per group and criterion.

    ``sampled_enough`` is ``"yes"`` when a rule on the data met its threshold,
    ``"target reached"`` when a fixed-effort rule (FOVs, objects, area) reached
    its target, ``"no"`` or ``"undefined"``. The stop columns summarise the
    random orders when any were replayed, otherwise the recorded order
    (``order_kind`` says which); ``answer`` states each kind apart, and the
    stops table holds both.
    """
    keys = [*group_cols, "strategy", "config_id"]
    rows = []
    for _, row in final.iterrows():
        own = rows_of(stops, {column: row[column] for column in keys}) if len(stops) else stops
        kinds = [part for _, part in own.iterrows()]
        by_kind = {part["order_kind"]: part for part in kinds}
        summary = by_kind.get("permuted", by_kind.get("recorded"))
        verdict = row["verdict"]
        stop_word = "target reached" if row.get("effort_rule", False) else "yes"
        enough = {"stop": stop_word, "continue": "no"}.get(verdict, "undefined")
        stop = {} if summary is None else summary
        rows.append(
            {
                **{column: row[column] for column in group_cols},
                "criterion": row["strategy"],
                "rule": row.get("rule", row["strategy"]),
                "settings": row["policies"],
                "config_id": row["config_id"],
                "fovs_acquired": row["n_fovs"],
                "sampled_enough": enough,
                "enough_after_fovs": stop.get("stop_fov_median", np.nan),
                "enough_after_p10": stop.get("stop_fov_p10", np.nan),
                "enough_after_p90": stop.get("stop_fov_p90", np.nan),
                # FOVs imaged beyond the point of enough exist only when the acquired
                # FOVs are enough: a criterion met early and lost has none.
                "fovs_beyond_enough": (
                    stop.get("spare_fovs_median", np.nan) if verdict == "stop" else np.nan
                ),
                "order_kind": stop.get("order_kind"),
                "orders": stop.get("orders", np.nan),
                "orders_never_enough": stop.get("never_stopped", np.nan),
                "answer": _enough_answer(row, kinds),
            }
        )
    return pd.DataFrame(rows)


__all__ = ["criteria_tables", "enough_table", "group_major", "is_effort_rule"]
