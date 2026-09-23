"""Strategy execution utilities."""

from __future__ import annotations

from dataclasses import replace
from time import perf_counter
from typing import Iterable

from .objectives import ObjectiveConfig
from .results import StrategyResult
from .schema import ValidatedInput
from .strategies.base import SamplingStrategy


def evaluate_strategy(
    strategy: SamplingStrategy,
    data: ValidatedInput,
    *,
    objective: ObjectiveConfig,
) -> StrategyResult:
    """Run one strategy and attach measured wall-clock runtime."""
    start = perf_counter()
    result = strategy.evaluate(data, objective=objective)
    runtime_seconds = perf_counter() - start

    if result.strategy != strategy.name:
        raise ValueError(
            f"strategy returned name {result.strategy!r}, expected {strategy.name!r}"
        )
    return replace(result, runtime_seconds=runtime_seconds)


def evaluate_strategies(
    strategies: Iterable[SamplingStrategy],
    data: ValidatedInput,
    *,
    objective: ObjectiveConfig,
) -> list[StrategyResult]:
    """Run multiple interchangeable strategies on exactly the same input."""
    return [
        evaluate_strategy(strategy, data, objective=objective)
        for strategy in strategies
    ]
