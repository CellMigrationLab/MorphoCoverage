"""Strategy execution utilities."""

from __future__ import annotations

import copy
from collections.abc import Callable, Iterable
from dataclasses import replace
from time import perf_counter
from typing import Any

import pandas as pd

from .evidence import Evidence, Quantity
from .objectives import ObjectiveConfig
from .policies import Effort, decide
from .results import StrategyResult, Timing
from .schema import ValidatedInput, fov_positions
from .strategies.base import Estimator, EstimatorState, SamplingStrategy


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


def _merge_evidence(strategy: SamplingStrategy, evidences: Iterable[Evidence]) -> Evidence:
    """Combine the estimators' evidence; name collisions raise instead of overwriting."""
    quantities: dict[str, Quantity] = {}
    diagnostics: dict[str, Any] = {}
    for evidence, estimator in zip(evidences, strategy.estimators, strict=True):
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
    return Evidence(quantities, diagnostics)


def _recompute(
    estimator: Estimator, data: ValidatedInput, objective: ObjectiveConfig
) -> tuple[Evidence, float]:
    """Full ``estimate`` on a fresh copy of the estimator and of the input; returns its time."""
    fresh = copy.deepcopy(estimator)
    isolated = _isolated_input(data)
    start = perf_counter()
    evidence = fresh.estimate(isolated, objective=objective)
    return evidence, perf_counter() - start


def _build_result(
    strategy: SamplingStrategy,
    *,
    objective: ObjectiveConfig,
    group: dict[str, object] | None,
    evidence: Evidence,
    effort: Effort,
    data: ValidatedInput,
    compute_seconds: float,
) -> StrategyResult:
    """Apply the policies and fill the framework-owned fields (timing is added by the caller)."""
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


def evaluate_strategy(
    strategy: SamplingStrategy,
    data: ValidatedInput,
    *,
    objective: ObjectiveConfig,
    group: dict[str, object] | None = None,
    setup_seconds: float = 0.0,
) -> StrategyResult:
    """Run one strategy and build its result with framework-owned counts and provenance.

    Each estimator runs on a fresh copy of itself and on its own deep copy of
    the input tables, so neither estimator state nor in-place edits to the
    data can flow to another estimator, analysis group or acquisition prefix.
    ``estimate`` is therefore a pure full recomputation on the data it
    receives. Effort (``n_fovs``, ``n_objects``) is computed from the original
    input and given to the policies. ``setup_seconds`` (for example building
    the visible prefix) is counted as framework overhead.
    """
    wall = perf_counter()
    runs = [_recompute(estimator, data, objective) for estimator in strategy.estimators]
    recompute_seconds = sum(seconds for _, seconds in runs)
    evidence = _merge_evidence(strategy, [evidence for evidence, _ in runs])
    effort = Effort(n_fovs=len(data.fovs), n_objects=len(data.objects))
    result = _build_result(
        strategy,
        objective=objective,
        group=group,
        evidence=evidence,
        effort=effort,
        data=data,
        compute_seconds=recompute_seconds,
    )
    overhead = perf_counter() - wall - recompute_seconds + setup_seconds
    return replace(result, timing=Timing(recompute_seconds, 0.0, 0.0, overhead))


class IncrementalReplay:
    """One strategy replayed FOV by FOV within one analysis group.

    Estimators whose ``start`` returns an ``EstimatorState`` receive only each
    newly revealed FOV; the others are fully recomputed on the visible prefix,
    exactly as in ``evaluate_strategy``. Each estimator is deep-copied once per
    group, and every update receives its own copy of the new rows.
    """

    def __init__(
        self,
        strategy: SamplingStrategy,
        group_data: ValidatedInput,
        *,
        objective: ObjectiveConfig,
        group: dict[str, object],
    ) -> None:
        self.strategy = strategy
        self.objective = objective
        self.group = group
        self.n_fovs = 0
        self.n_objects = 0
        wall = perf_counter()
        template = restrict_input(group_data, group_data.objects.iloc[:0], group_data.fovs.iloc[:0])
        self._estimators = [copy.deepcopy(estimator) for estimator in strategy.estimators]
        self._states: list[EstimatorState | None] = []
        start_seconds = 0.0
        for estimator in self._estimators:
            isolated = _isolated_input(template)
            started = perf_counter()
            state = estimator.start(isolated, objective=objective)
            start_seconds += perf_counter() - started
            if state is not None and not isinstance(state, EstimatorState):
                raise TypeError(
                    f"estimator {estimator.name!r} start() must return an EstimatorState or None"
                )
            self._states.append(state)
        self._pending_initial = start_seconds
        self._pending_overhead = perf_counter() - wall - start_seconds

    @property
    def incremental(self) -> bool:
        """Whether any estimator is updated incrementally."""
        return any(state is not None for state in self._states)

    def step(
        self,
        new: ValidatedInput,
        prefix: Callable[[], tuple[ValidatedInput, float]],
        *,
        record: bool,
        setup_seconds: float,
    ) -> StrategyResult | None:
        """Reveal one FOV; return a result when ``record``, else only warm up the states.

        ``prefix`` builds (once per step, shared across strategies) the visible
        prefix and returns it with the time it took on first use.
        """
        wall = perf_counter()
        self.n_fovs += len(new.fovs)
        self.n_objects += len(new.objects)
        evidences: list[Evidence] = []
        update_seconds = recompute_seconds = 0.0
        data = None
        for estimator, state in zip(self._estimators, self._states, strict=True):
            if state is not None:
                isolated = _isolated_input(new)
                started = perf_counter()
                evidences.append(state.update(isolated))
                update_seconds += perf_counter() - started
            elif record:
                data, prefix_seconds = prefix()
                setup_seconds += prefix_seconds
                evidence, seconds = _recompute(estimator, data, self.objective)
                evidences.append(evidence)
                recompute_seconds += seconds
        if not record:
            # Warm-up before min_fovs: charged to the first recorded result.
            self._pending_initial += update_seconds
            self._pending_overhead += perf_counter() - wall - update_seconds + setup_seconds
            return None

        evidence = _merge_evidence(self.strategy, evidences)
        effort = Effort(n_fovs=self.n_fovs, n_objects=self.n_objects)
        result = _build_result(
            self.strategy,
            objective=self.objective,
            group=self.group,
            evidence=evidence,
            effort=effort,
            data=data if data is not None else new,
            compute_seconds=update_seconds + recompute_seconds,
        )
        overhead = (
            perf_counter() - wall - update_seconds - recompute_seconds
            + setup_seconds + self._pending_overhead
        )
        timing = Timing(recompute_seconds, update_seconds, self._pending_initial, overhead)
        self._pending_initial = self._pending_overhead = 0.0
        return replace(result, timing=timing)


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
