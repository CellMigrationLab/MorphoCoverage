"""The parts of ``replay_acquisitions`` that need nothing of ``replay`` itself (private): one
run's decisions and its scores, the groups that can be replayed, and the checks of the tables.

Split from ``replay`` to keep that module under 600 lines; the replay is described there.
Shared-private: used by ``retrospective.replay``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .._stage_coordinates import stage_matrix, stage_names
from ..evaluator import iter_analysis_groups
from ..objectives import ObjectiveConfig
from ..schema import ValidatedInput
from ..sequential import REPLAY_PURPOSE, _Acquisition, _replay_strategies
from ..strategies.base import SamplingStrategy, check_order_col
from ._run_identity import RUN_FINGERPRINT, run_fingerprint
from .criteria import OutcomeCriterion
from .reliability import _skip_reason
from .stage import stage_travel


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
    fingerprint = run_fingerprint(where, ordered, reservoir.fovs, acquisition.data.fov_key)
    streams = [
        {
            **where,
            "stream_fovs": acquisition.n_fovs,
            "reservoir_fovs": len(reservoir.fovs),
            RUN_FINGERPRINT: fingerprint,
        }
    ]
    paths = []
    if stage_cols is not None:
        travel = stage_travel(ordered[list(stage_cols)].to_numpy(dtype=float))
        paths = [{**where, "fovs": i + 1, "travel": float(t)} for i, t in enumerate(travel)]
    outcomes = []
    # A fitted outcome (a wide table's components) is fitted on the stream, never on the
    # held-back fields that judge it (owner, 2026-10-06).
    criteria = [criterion.trained(acquisition.data) for criterion in criteria]
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


def _check_tables(
    data: ValidatedInput,
    feature_cols: Sequence[str],
    acquisition_order_col: str | None,
    stage_cols: tuple[str, str] | None,
) -> tuple[str, str] | None:
    """Raise when the order column or the stage columns cannot be read; the stage columns
    checked (None without them)."""
    if acquisition_order_col is not None:
        if acquisition_order_col not in data.fovs.columns:
            raise ValueError(f"fovs is missing required columns: {[acquisition_order_col]}")
        check_order_col(data.fovs, acquisition_order_col, REPLAY_PURPOSE)
    if stage_cols is not None:
        stage_cols = stage_names(stage_cols)
        stage_matrix(data.fovs, stage_cols)  # real, finite, complete: before any order is made
        if any(c in feature_cols for c in stage_cols):
            raise ValueError("stage coordinates cannot also be phenotype features")
    return stage_cols


__all__ = ["_check_tables", "_decide", "_replayable", "_score"]
