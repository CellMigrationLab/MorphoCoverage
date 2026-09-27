"""Sequential evaluation on deeply sampled real acquisitions.

Every prefix lists its observations in acquisition order (the FOVs in the
order they were revealed, table order within each FOV), which is the order in
which a live ``OnlineSession`` receives them; estimators that draw objects by
row position therefore see the same rows in a replay and at the microscope.

Shared-private helpers: ``_evaluate_prefixes``, ``_check_order_values``,
``_in_acquisition_order``, ``_Acquisition``, ``_replay_strategies``,
``_strategy_units`` and ``_interleave`` are also used by ``benchmark`` and
``retrospective.replay``, so that every replay reveals FOVs the same way.
They are private to the package (not part of its public API) and may change
with it.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from ._checks import as_count
from ._parallel import check_n_jobs, map_units
from .data_check import validate_input_tables
from .evaluator import (
    IncrementalReplay,
    evaluate_strategy,
    iter_analysis_groups,
    restrict_input,
)
from .metadata import column_names
from .objectives import ObjectiveConfig
from .results import StrategyResult
from .schema import ValidatedInput, fov_positions
from .strategies.base import SamplingStrategy


@dataclass(frozen=True)
class SequentialResult:
    """Strategy output associated with one acquisition prefix.

    ``acquired_fovs`` holds one tuple of ``fov_key`` values per revealed FOV,
    in acquisition order.
    """

    step: int
    fov_key: tuple[str, ...]
    acquired_fovs: tuple[tuple[object, ...], ...]
    result: StrategyResult


def evaluate_acquisition_prefixes(
    objects: pd.DataFrame,
    fovs: pd.DataFrame,
    *,
    strategies: Iterable[SamplingStrategy],
    feature_cols: Sequence[str],
    acquisition_order_col: str,
    objective: ObjectiveConfig,
    fov_col: str | None = None,
    group_cols: Sequence[str] | None = None,
    min_fovs: int = 1,
    incremental: bool = False,
    observation_unit: str | None = None,
    n_jobs: int = 1,
) -> list[SequentialResult]:
    """Evaluate strategies after each prefix of a recorded real acquisition.

    The complete linked tables are validated once, so objects from FOVs absent
    from the FOV table raise instead of never being revealed. Acquisition order
    must be strictly defined within each analysis group: ties raise, because
    the simulated history would otherwise depend on table row order.

    MorphoBricks metadata defaults are resolved under the hood. By default,
    condition and repeat remain separate when available; position_id is
    preferred as the spatial FOV identifier, with image_file only a fallback
    when one file equals one position.

    With ``incremental=True``, estimators that support it (``Estimator.start``)
    receive only each newly revealed FOV, which is how an online microscope
    would run them; the others are recomputed on the prefix as usual. Evidence
    and decisions are the same either way, to floating-point rounding; only
    the timing differs. Each result's ``timing`` separates estimator time from
    framework overhead, including building the prefix.

    Every strategy is replayed independently of the others, so ``n_jobs``
    replays the groups in that many worker processes (``-1``: every CPU),
    each strategy apart when there are too few groups to keep them busy.
    Results come back in the same order and are identical for every value,
    apart from timings (bit for bit when BLAS is single-threaded; see ``morphocoverage._parallel``).

    ``min_fovs`` is the first prefix recorded: an integer of at least 1
    (``2.0`` or ``2.5`` is refused, numpy integers are accepted).
    """
    return _evaluate_prefixes(
        objects,
        fovs,
        strategies=strategies,
        feature_cols=feature_cols,
        acquisition_order_col=acquisition_order_col,
        objective=objective,
        fov_col=fov_col,
        group_cols=group_cols,
        min_fovs=min_fovs,
        incremental=incremental,
        observation_unit=observation_unit,
        n_jobs=n_jobs,
    )


def _evaluate_prefixes(
    objects: pd.DataFrame,
    fovs: pd.DataFrame | None,
    *,
    strategies: Sequence[SamplingStrategy],
    feature_cols: Sequence[str],
    acquisition_order_col: str,
    objective: ObjectiveConfig,
    fov_col: str | None,
    group_cols: Sequence[str] | None,
    min_fovs: int,
    incremental: bool,
    observation_unit: str | None,
    n_jobs: int,
    pool: Any = None,
) -> list[SequentialResult]:
    """``evaluate_acquisition_prefixes``, optionally on running workers (``pool``)."""
    check_n_jobs(n_jobs)
    # Read once, so a generator of names is not consumed by the order check.
    feature_cols = column_names(feature_cols, "feature_cols")
    min_fovs = as_count(min_fovs, "min_fovs")
    _check_order_column(fovs, acquisition_order_col, feature_cols)
    data = validate_input_tables(
        objects,
        feature_cols=feature_cols,
        fov_col=fov_col,
        fovs=fovs,
        group_cols=group_cols,
        observation_unit=observation_unit,
    )
    strategy_list = list(strategies)
    groups = list(iter_analysis_groups(data))
    _check_min_fovs(groups, min_fovs)
    # Ordered here, so a tie in any group raises before work is handed out.
    ordered = [_in_acquisition_order(g, g_data, acquisition_order_col) for g, g_data in groups]
    units = _strategy_units(strategy_list, n_jobs, len(groups))
    jobs = [
        (part, group, group_data, order, objective, min_fovs, incremental)
        for (group, group_data), order in zip(groups, ordered, strict=True)
        for part in units
    ]
    parts = iter(map_units(_replay_unit, jobs, n_jobs, pool))
    all_results: list[SequentialResult] = []
    for _ in groups:
        replayed = [next(parts) for _ in units]
        for step, keys, result in _interleave(replayed):
            all_results.append(
                SequentialResult(step=step, fov_key=data.fov_key, acquired_fovs=keys, result=result)
            )
    return all_results


def _strategy_units(strategies: list[SamplingStrategy], n_jobs: int, n_runs: int) -> list[list]:
    """The strategies replayed together in one unit of work, for ``n_runs`` independent runs.

    Strategies replayed together share each visible prefix and send the data
    to a worker once, so they stay together in this process and whenever the
    runs alone can occupy every worker. With fewer runs than workers (one
    sample compared across several strategies, say), every strategy is its
    own unit: sending the data more often costs less than idle workers.
    """
    if n_runs >= check_n_jobs(n_jobs):
        return [strategies]
    return [[strategy] for strategy in strategies]


def _interleave(parts: list[list[tuple]]) -> list[tuple]:
    """Results of strategies replayed apart, in the order of one joint replay.

    A joint replay yields every strategy at step k, in strategy order, before
    step k + 1. Each part is in step order, so a stable sort of the parts
    concatenated in strategy order by step restores exactly that order.
    """
    return sorted((row for part in parts for row in part), key=lambda row: row[0])


def _replay_strategies(
    strategies: Sequence[SamplingStrategy],
    group: Mapping[str, Any],
    acquisition: _Acquisition,
    *,
    objective: ObjectiveConfig,
    min_fovs: int,
    incremental: bool,
) -> list[tuple[int, StrategyResult]]:
    """``(step, result)`` of every strategy after every prefix of ``acquisition``."""
    replay = _replay_incremental if incremental else _replay_full
    return list(replay(strategies, group, acquisition, objective=objective, min_fovs=min_fovs))


def _replay_unit(
    strategies: Sequence[SamplingStrategy],
    group: Mapping[str, Any],
    group_data: ValidatedInput,
    ordered: pd.DataFrame,
    objective: ObjectiveConfig,
    min_fovs: int,
    incremental: bool,
) -> list[tuple[int, tuple, StrategyResult]]:
    acquisition = _Acquisition(group_data, ordered)
    return [
        (step, acquisition.keys[:step], result)
        for step, result in _replay_strategies(
            strategies,
            group,
            acquisition,
            objective=objective,
            min_fovs=min_fovs,
            incremental=incremental,
        )
    ]


def _check_order_column(
    fovs: pd.DataFrame, order_col: str, feature_cols: Sequence[str] | None
) -> None:
    """The acquisition-order column exists, is complete and is not a phenotype feature."""
    if not isinstance(fovs, pd.DataFrame):
        raise TypeError("fovs must be a pandas DataFrame")
    if order_col not in fovs.columns:
        raise ValueError(f"fovs is missing required columns: {[order_col]}")
    if feature_cols is not None and order_col in feature_cols:
        raise ValueError("acquisition order cannot also be a phenotype feature")
    _check_order_values(fovs, order_col)


def _check_order_values(fovs: pd.DataFrame, order_col: str) -> None:
    """The acquisition order is complete and numeric, a date-time or a duration.

    Text is refused rather than sorted as text, where ``"10"`` comes before
    ``"2"`` and the replayed history would silently differ from the recorded one.
    """
    values = fovs[order_col]
    if values.isna().any():
        raise ValueError("acquisition order contains missing values")
    dtype, types = values.dtype, pd.api.types
    numeric = types.is_numeric_dtype(dtype) and not types.is_bool_dtype(dtype)
    if not (numeric or types.is_datetime64_any_dtype(dtype) or types.is_timedelta64_dtype(dtype)):
        raise ValueError(
            f"acquisition order column {order_col!r} holds {dtype} values, not numbers or "
            "date-times; text would sort '10' before '2'. Convert it first, for example "
            f"fovs[{order_col!r}] = pd.to_numeric(fovs[{order_col!r}]) or pd.to_datetime(...)."
        )


def _check_min_fovs(groups: list[tuple[dict, ValidatedInput]], min_fovs: int) -> None:
    """Raise rather than silently dropping groups too small to reach ``min_fovs``."""
    too_small = [group for group, group_data in groups if len(group_data.fovs) < min_fovs]
    if too_small:
        raise ValueError(
            f"min_fovs={min_fovs} exceeds the number of acquired FOVs in "
            f"{len(too_small)} analysis group(s), e.g. {too_small[:3]}; they would "
            "produce no results. Lower min_fovs or exclude those groups explicitly."
        )


def _in_acquisition_order(group: dict, group_data: ValidatedInput, order_col: str) -> pd.DataFrame:
    """The group's FOV rows in acquisition order; ties raise, since row order would decide."""
    order = group_data.fovs[order_col]
    if order.duplicated().any():
        tied = sorted(order[order.duplicated(keep=False)].unique().tolist())
        raise ValueError(
            f"acquisition order has ties within analysis group {group}: {tied[:5]}. "
            "When pooling, acquisition order must be defined across the pooled FOVs."
        )
    return group_data.fovs.sort_values(order_col, kind="stable")


class _Acquisition:
    """One analysis group in acquisition order, sliced into prefixes and new FOVs.

    Each observation's acquisition rank is computed once and the observations
    are sorted by it, so a prefix and a newly revealed FOV are both slices.
    Observations are in acquisition order: FOV by FOV as revealed, table order
    within each FOV, exactly as a live session joins them.
    """

    def __init__(self, group_data: ValidatedInput, ordered: pd.DataFrame) -> None:
        self.data = group_data
        self.ordered = ordered
        self.rank = fov_positions(group_data, ordered)
        self.n_fovs = len(ordered)
        self.keys = tuple(
            ordered.loc[:, list(group_data.fov_key)].itertuples(index=False, name=None)
        )
        # Observations sorted by FOV rank (table order kept within each FOV), so
        # one newly revealed FOV is a slice rather than a mask over the stream.
        by_rank = np.argsort(self.rank, kind="stable")
        by_rank = by_rank[self.rank[by_rank] >= 0]
        self._by_rank = group_data.objects.iloc[by_rank]
        self._bounds = np.searchsorted(self.rank[by_rank], np.arange(self.n_fovs + 1))

    def prefix(self, step: int) -> ValidatedInput:
        """The first ``step`` FOVs and their observations, in acquisition order."""
        return restrict_input(
            self.data, self._by_rank.iloc[: self._bounds[step]], self.ordered.iloc[:step]
        )

    def new_fov(self, step: int) -> ValidatedInput:
        """Only the FOV revealed at ``step`` and its observations."""
        first, last = self._bounds[step - 1], self._bounds[step]
        return restrict_input(
            self.data, self._by_rank.iloc[first:last], self.ordered.iloc[step - 1 : step]
        )


def _replay_full(
    strategies: Sequence[SamplingStrategy],
    group: Mapping[str, Any],
    acquisition: _Acquisition,
    *,
    objective: ObjectiveConfig,
    min_fovs: int,
) -> Iterator[tuple[int, StrategyResult]]:
    """Full recomputation of every strategy on every prefix (the reference)."""
    for step in range(min_fovs, acquisition.n_fovs + 1):
        started = perf_counter()
        prefix = acquisition.prefix(step)
        setup_seconds = perf_counter() - started
        for strategy in strategies:
            yield (
                step,
                evaluate_strategy(
                    strategy, prefix, objective=objective, group=group, setup_seconds=setup_seconds
                ),
            )


class _LazyPrefix:
    """Builds the visible prefix at most once per step, only if an estimator recomputes."""

    def __init__(self, acquisition: _Acquisition, step: int) -> None:
        self.acquisition = acquisition
        self.step = step
        self.data: ValidatedInput | None = None

    def __call__(self) -> tuple[ValidatedInput, float]:
        """The prefix, and the time spent building it on this call (0 once cached)."""
        if self.data is not None:
            return self.data, 0.0
        started = perf_counter()
        self.data = self.acquisition.prefix(self.step)
        return self.data, perf_counter() - started


def _replay_incremental(
    strategies: Sequence[SamplingStrategy],
    group: Mapping[str, Any],
    acquisition: _Acquisition,
    *,
    objective: ObjectiveConfig,
    min_fovs: int,
) -> Iterator[tuple[int, StrategyResult]]:
    """Feed each newly revealed FOV to incremental estimators; recompute the others."""
    replays = [
        IncrementalReplay(strategy, acquisition.data, objective=objective, group=group)
        for strategy in strategies
    ]
    for step in range(1, acquisition.n_fovs + 1):
        started = perf_counter()
        new = acquisition.new_fov(step)
        setup_seconds = perf_counter() - started
        prefix = _LazyPrefix(acquisition, step)
        for replay in replays:
            result = replay.step(new, prefix, record=step >= min_fovs, setup_seconds=setup_seconds)
            if result is not None:
                yield step, result


__all__ = ["SequentialResult", "evaluate_acquisition_prefixes"]
