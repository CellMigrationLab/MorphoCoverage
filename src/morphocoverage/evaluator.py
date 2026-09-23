"""Strategy execution utilities."""

from __future__ import annotations

import copy
from collections.abc import Iterable
from dataclasses import replace
from time import perf_counter

import pandas as pd

from .objectives import ObjectiveConfig
from .results import Decision, StrategyResult
from .schema import ValidatedInput, fov_positions
from .strategies.base import SamplingStrategy


def _visible_table(table: pd.DataFrame) -> pd.DataFrame:
    """Return a copy exposing nothing about rows that were removed.

    Categorical columns keep their full category list after row filtering,
    which would reveal withheld FOV identifiers (other groups, future
    acquisitions); the original index would reveal how many rows were
    withheld between visible ones.
    """
    visible = table.reset_index(drop=True)
    for column in visible.columns:
        if isinstance(visible[column].dtype, pd.CategoricalDtype):
            visible[column] = visible[column].cat.remove_unused_categories()
    return visible


def _isolated_input(data: ValidatedInput) -> ValidatedInput:
    """A deep copy of the input tables for one strategy call."""
    return replace(data, objects=data.objects.copy(deep=True), fovs=data.fovs.copy(deep=True))


def restrict_input(
    data: ValidatedInput,
    objects: pd.DataFrame,
    fovs: pd.DataFrame,
) -> ValidatedInput:
    """Build a ValidatedInput over a subset of already validated rows."""
    return ValidatedInput(
        objects=_visible_table(objects),
        fovs=_visible_table(fovs),
        feature_cols=data.feature_cols,
        fov_col=data.fov_col,
        group_cols=data.group_cols,
        fov_key=data.fov_key,
    )


def objects_in_fovs(data: ValidatedInput, fovs: pd.DataFrame) -> pd.DataFrame:
    """Return the objects belonging to the given FOV rows, matched on the full FOV key."""
    return data.objects.loc[fov_positions(data, fovs) >= 0]


def iter_analysis_groups(
    data: ValidatedInput,
) -> Iterable[tuple[dict[str, object], ValidatedInput]]:
    """Yield independent analysis groups, preserving zero-object FOV groups."""
    if not data.group_cols:
        yield {}, restrict_input(data, data.objects, data.fovs)
        return

    group_cols = list(data.group_cols)
    unique_groups = data.fovs.loc[:, group_cols].drop_duplicates()
    for values in unique_groups.itertuples(index=False, name=None):
        group = dict(zip(group_cols, values, strict=True))
        fov_mask = pd.Series(True, index=data.fovs.index)
        for column, value in group.items():
            fov_mask &= data.fovs[column].eq(value)
        group_fovs = data.fovs.loc[fov_mask]
        yield group, restrict_input(data, objects_in_fovs(data, group_fovs), group_fovs)


def _framework_count(reported: int | None, actual: int, name: str, strategy: str) -> int:
    if reported is not None and reported != actual:
        raise ValueError(
            f"strategy {strategy!r} reported {name}={reported}, but the acquisition "
            f"record has {actual}; {name} is framework-owned and should be left unset"
        )
    return actual


def evaluate_strategy(
    strategy: SamplingStrategy,
    data: ValidatedInput,
    *,
    objective: ObjectiveConfig,
    group: dict[str, object] | None = None,
) -> StrategyResult:
    """Run one strategy and attach framework-owned counts, provenance and runtime.

    Each call runs on a fresh copy of the configured strategy and on its own
    deep copy of the input tables, so neither strategy state nor in-place edits
    to the data can flow to another strategy, analysis group or acquisition
    prefix. Framework-owned fields are computed from the original input.
    ``evaluate`` is therefore a pure full recomputation on the data it receives.
    Copying happens outside the timed region.
    """
    fresh = copy.deepcopy(strategy)
    isolated = _isolated_input(data)
    start = perf_counter()
    result = fresh.evaluate(isolated, objective=objective)
    runtime_seconds = perf_counter() - start

    if not isinstance(result, StrategyResult):
        raise TypeError(f"strategy {strategy.name!r} must return a StrategyResult")
    if result.strategy != strategy.name:
        raise ValueError(
            f"strategy returned name {result.strategy!r}, expected {strategy.name!r}"
        )
    decision = result.decision
    if decision is not None:
        try:
            decision = Decision(decision)
        except ValueError:
            allowed = [d.value for d in Decision]
            raise ValueError(
                f"strategy {strategy.name!r} returned decision {decision!r}; "
                f"expected one of {allowed} or None"
            ) from None

    return replace(
        result,
        decision=decision,
        runtime_seconds=runtime_seconds,
        group=dict(group or {}),
        feature_cols=data.feature_cols,
        fov_key=data.fov_key,
        n_fovs=_framework_count(result.n_fovs, len(data.fovs), "n_fovs", strategy.name),
        n_objects=_framework_count(
            result.n_objects, len(data.objects), "n_objects", strategy.name
        ),
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
