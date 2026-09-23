"""Strategy execution utilities."""

from __future__ import annotations

from dataclasses import replace
from time import perf_counter
from typing import Iterable

import pandas as pd

from .objectives import ObjectiveConfig
from .results import StrategyResult
from .schema import ValidatedInput
from .strategies.base import SamplingStrategy


def _slice_group(
    data: ValidatedInput,
    group_values: tuple[object, ...],
) -> ValidatedInput:
    if not data.group_cols:
        return data

    object_mask = pd.Series(True, index=data.objects.index)
    fov_mask = pd.Series(True, index=data.fovs.index)
    for column, value in zip(data.group_cols, group_values, strict=True):
        object_mask &= data.objects[column].eq(value)
        fov_mask &= data.fovs[column].eq(value)

    return ValidatedInput(
        objects=data.objects.loc[object_mask].copy(),
        fovs=data.fovs.loc[fov_mask].copy(),
        feature_cols=data.feature_cols,
        fov_col=data.fov_col,
        group_cols=data.group_cols,
    )


def iter_analysis_groups(
    data: ValidatedInput,
) -> Iterable[tuple[dict[str, object], ValidatedInput]]:
    """Yield independent analysis groups, preserving zero-object FOV groups."""
    if not data.group_cols:
        yield {}, data
        return

    unique_groups = data.fovs.loc[:, list(data.group_cols)].drop_duplicates()
    for row in unique_groups.itertuples(index=False, name=None):
        values = tuple(row)
        group = dict(zip(data.group_cols, values, strict=True))
        yield group, _slice_group(data, values)


def evaluate_strategy(
    strategy: SamplingStrategy,
    data: ValidatedInput,
    *,
    objective: ObjectiveConfig,
    group: dict[str, object] | None = None,
) -> StrategyResult:
    """Run one strategy and attach measured wall-clock runtime and group identity."""
    start = perf_counter()
    result = strategy.evaluate(data, objective=objective)
    runtime_seconds = perf_counter() - start

    if result.strategy != strategy.name:
        raise ValueError(
            f"strategy returned name {result.strategy!r}, expected {strategy.name!r}"
        )
    return replace(
        result,
        runtime_seconds=runtime_seconds,
        group=dict(group or {}),
    )


def evaluate_strategies(
    strategies: Iterable[SamplingStrategy],
    data: ValidatedInput,
    *,
    objective: ObjectiveConfig,
) -> list[StrategyResult]:
    """Run all strategies independently within every resolved analysis group."""
    strategy_list = list(strategies)
    results: list[StrategyResult] = []
    for group, group_data in iter_analysis_groups(data):
        for strategy in strategy_list:
            results.append(
                evaluate_strategy(
                    strategy,
                    group_data,
                    objective=objective,
                    group=group,
                )
            )
    return results
