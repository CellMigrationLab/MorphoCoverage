"""Strategy execution utilities."""

from __future__ import annotations

import copy
from collections.abc import Callable, Iterable
from dataclasses import replace
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from ._parallel import check_n_jobs, map_units
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
    for column in visible.select_dtypes("category").columns:
        visible[column] = visible[column].cat.remove_unused_categories()
    return visible


def _isolated_input(data: ValidatedInput) -> ValidatedInput:
    """A deep copy of the input tables for one strategy call."""
    return replace(
        data,
        objects=data.objects.copy(deep=True),
        fovs=data.fovs.copy(deep=True),
    )


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
        observation_unit=data.observation_unit,
        observation_unit_declared=data.observation_unit_declared,
    )


class FovSubsets:
    """Fast subsets of whole FOVs of one input, by FOV row position.

    Each observation's FOV row is looked up once, so a subset of FOVs costs one
    vectorized mask instead of a key match per call; for resampling analyses
    that draw thousands of subsets of the same FOVs. Observations keep their
    table order, as with ``objects_in_fovs``.
    """

    def __init__(self, data: ValidatedInput) -> None:
        self.data = data
        self.positions = fov_positions(data, data.fovs)

    def __len__(self) -> int:
        return len(self.data.fovs)

    def take(self, rows: np.ndarray) -> ValidatedInput:
        """The FOVs at row positions ``rows`` of ``data.fovs``, with their observations."""
        rows = np.asarray(rows, dtype=np.int64)
        mask = np.isin(self.positions, rows)
        return restrict_input(self.data, self.data.objects.loc[mask], self.data.fovs.iloc[rows])


def objects_in_fovs(data: ValidatedInput, fovs: pd.DataFrame) -> pd.DataFrame:
    """Return the observations belonging to the given FOV rows, matched on the full FOV key."""
    return data.objects.loc[fov_positions(data, fovs) >= 0]


def iter_analysis_groups(
    data: ValidatedInput,
) -> Iterable[tuple[dict[str, object], ValidatedInput]]:
    """Yield independent analysis groups, preserving zero-observation FOV groups."""
    if not data.group_cols:
        yield {}, restrict_input(data, data.objects, data.fovs)
        return

    group_cols = list(data.group_cols)
    # Groups in order of first appearance in the FOV table. Each observation's FOV
    # row is looked up once, and rows are split by group with a stable sort, so
    # both tables keep their order within a group: one pass over the
    # observations instead of one per group.
    fov_codes = data.fovs.groupby(group_cols, sort=False, dropna=False).ngroup().to_numpy(np.int64)
    first = np.unique(fov_codes, return_index=True)[1]
    positions = fov_positions(data, data.fovs)
    object_codes = np.where(positions >= 0, fov_codes[np.maximum(positions, 0)], -1)
    fov_rows = _rows_by_code(fov_codes, len(first))
    object_rows = _rows_by_code(object_codes, len(first))
    for code, values in enumerate(data.fovs[group_cols].iloc[first].itertuples(index=False)):
        group = dict(zip(group_cols, values, strict=True))
        yield (
            group,
            restrict_input(
                data, data.objects.iloc[object_rows[code]], data.fovs.iloc[fov_rows[code]]
            ),
        )


def _rows_by_code(codes: np.ndarray, n_codes: int) -> list[np.ndarray]:
    """Row positions of each code ``0 .. n_codes - 1``, in table order; ``-1`` is dropped."""
    order = np.argsort(codes, kind="stable")
    counts = np.bincount(codes[codes >= 0], minlength=n_codes)
    start = int((codes < 0).sum())  # -1 codes sort first
    return np.split(order[start:], np.cumsum(counts)[:-1])


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
        n_observations=effort.n_observations,
        observation_unit=effort.observation_unit,
        observation_unit_declared=data.observation_unit_declared,
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
    receives. Effort (``n_fovs``, ``n_observations``) is computed from the original
    input and given to the policies. ``setup_seconds`` (for example building
    the visible prefix) is counted as framework overhead.
    """
    wall = perf_counter()
    runs = [_recompute(estimator, data, objective) for estimator in strategy.estimators]
    recompute_seconds = sum(seconds for _, seconds in runs)
    evidence = _merge_evidence(strategy, [evidence for evidence, _ in runs])
    effort = Effort(
        n_fovs=len(data.fovs),
        n_observations=len(data.objects),
        observation_unit=data.observation_unit,
    )
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
        self.n_observations = 0
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
        self.n_observations += len(new.objects)
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
        effort = Effort(
            n_fovs=self.n_fovs,
            n_observations=self.n_observations,
            observation_unit=new.observation_unit,
        )
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
            perf_counter()
            - wall
            - update_seconds
            - recompute_seconds
            + setup_seconds
            + self._pending_overhead
        )
        timing = Timing(recompute_seconds, update_seconds, self._pending_initial, overhead)
        self._pending_initial = self._pending_overhead = 0.0
        return replace(result, timing=timing)


def evaluate_strategies(
    strategies: Iterable[SamplingStrategy],
    data: ValidatedInput,
    *,
    objective: ObjectiveConfig,
    n_jobs: int = 1,
) -> list[StrategyResult]:
    """Run all strategies independently within every resolved analysis group.

    Every (group, strategy) pair is independent, so ``n_jobs`` runs the pairs
    in that many worker processes (``-1``: every CPU). Results come back in
    the same order and are identical for every value, apart from timings
    (bit for bit when BLAS is single-threaded; see ``morphocoverage._parallel``).
    """
    check_n_jobs(n_jobs)
    strategy_list = list(strategies)
    jobs = [
        (strategy, group_data, objective, group)
        for group, group_data in iter_analysis_groups(data)
        for strategy in strategy_list
    ]
    return map_units(_evaluate_unit, jobs, n_jobs)


def _evaluate_unit(strategy, group_data, objective, group) -> StrategyResult:
    return evaluate_strategy(strategy, group_data, objective=objective, group=group)
