"""Strategy execution utilities."""

from __future__ import annotations

import copy
from collections.abc import Iterable
from dataclasses import replace
from time import perf_counter
from typing import Any

import pandas as pd

from .evidence import Evidence, Quantity
from .objectives import ObjectiveConfig
from .policies import Effort, decide
from .results import StrategyResult
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


def _estimate(
    strategy: SamplingStrategy, data: ValidatedInput, objective: ObjectiveConfig
) -> tuple[Evidence, float]:
    """Run every estimator in isolation and merge their evidence."""
    quantities: dict[str, Quantity] = {}
    diagnostics: dict[str, Any] = {}
    compute_seconds = 0.0
    for estimator in strategy.estimators:
        fresh = copy.deepcopy(estimator)
        isolated = _isolated_input(data)
        start = perf_counter()
        evidence = fresh.estimate(isolated, objective=objective)
        compute_seconds += perf_counter() - start
        if not isinstance(evidence, Evidence):
            raise TypeError(f"estimator {estimator.name!r} must return Evidence")
        for kind, merged, new in (
            ("quantity", quantities, evidence.quantities),
            ("diagnostic", diagnostics, evidence.diagnostics),
        ):
            clash = sorted(set(merged) & set(new))
            if clash:
                raise ValueError(
                    f"estimators of strategy {strategy.name!r} both report {kind} {clash}; "
                    "give them distinct names"
                )
            merged.update(new)
    return Evidence(quantities, diagnostics), compute_seconds


def evaluate_strategy(
    strategy: SamplingStrategy,
    data: ValidatedInput,
    *,
    objective: ObjectiveConfig,
    group: dict[str, object] | None = None,
) -> StrategyResult:
    """Run one strategy and build its result with framework-owned counts and provenance.

    Each estimator runs on a fresh copy of itself and on its own deep copy of
    the input tables, so neither estimator state nor in-place edits to the
    data can flow to another estimator, analysis group or acquisition prefix.
    ``estimate`` is therefore a pure full recomputation on the data it
    receives. Effort (``n_fovs``, ``n_objects``) is computed from the original
    input and given to the policies. Copying happens outside the timed region.
    """
    evidence, compute_seconds = _estimate(strategy, data, objective)
    effort = Effort(n_fovs=len(data.fovs), n_objects=len(data.objects))
    decision, component_decisions = decide(
        evidence, effort, copy.deepcopy(strategy.policies), objective
    )
    return StrategyResult(
        strategy=strategy.name,
        config_id=strategy.config_id,
        objective=objective,
        evidence=evidence,
        decision=decision,
        component_decisions=component_decisions,
        compute_seconds=compute_seconds,
        n_fovs=effort.n_fovs,
        n_objects=effort.n_objects,
        group=dict(group or {}),
        parameters=dict(strategy.parameters),
        configuration=strategy.configuration,
        feature_cols=data.feature_cols,
        fov_key=data.fov_key,
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
