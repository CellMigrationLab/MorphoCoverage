"""Sequential evaluation on deeply sampled real acquisitions."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from time import perf_counter

import pandas as pd

from .evaluator import (
    IncrementalReplay,
    evaluate_strategy,
    iter_analysis_groups,
    objects_in_fovs,
    restrict_input,
)
from .objectives import ObjectiveConfig
from .results import StrategyResult
from .schema import ValidatedInput, validate_input_tables
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
    observation_unit: str = "object",
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
    """
    if min_fovs < 1:
        raise ValueError("min_fovs must be at least 1")
    if not isinstance(fovs, pd.DataFrame):
        raise TypeError("fovs must be a pandas DataFrame")
    if acquisition_order_col not in fovs.columns:
        raise ValueError(f"fovs is missing required columns: {[acquisition_order_col]}")
    if acquisition_order_col in feature_cols:
        raise ValueError("acquisition order cannot also be a phenotype feature")
    if fovs[acquisition_order_col].isna().any():
        raise ValueError("acquisition order contains missing values")

    data = validate_input_tables(
        objects,
        feature_cols=feature_cols,
        fov_col=fov_col,
        fovs=fovs,
        group_cols=group_cols,
        observation_unit=observation_unit,
    )
    strategy_list = list(strategies)
    key = list(data.fov_key)

    groups = list(iter_analysis_groups(data))
    too_small = [group for group, group_data in groups if len(group_data.fovs) < min_fovs]
    if too_small:
        raise ValueError(
            f"min_fovs={min_fovs} exceeds the number of acquired FOVs in "
            f"{len(too_small)} analysis group(s), e.g. {too_small[:3]}; they would "
            "produce no results. Lower min_fovs or exclude those groups explicitly."
        )

    all_results: list[SequentialResult] = []
    for group, group_data in groups:
        order = group_data.fovs[acquisition_order_col]
        if order.duplicated().any():
            tied = sorted(order[order.duplicated(keep=False)].unique().tolist())
            raise ValueError(
                f"acquisition order has ties within analysis group {group}: {tied[:5]}. "
                "When pooling, acquisition order must be defined across the pooled FOVs."
            )
        ordered = group_data.fovs.sort_values(acquisition_order_col, kind="stable")
        ordered_keys = list(ordered.loc[:, key].itertuples(index=False, name=None))
        replay = _replay_incremental if incremental else _replay_full
        for step, result in replay(
            strategy_list, group, group_data, ordered, objective=objective, min_fovs=min_fovs
        ):
            all_results.append(
                SequentialResult(
                    step=step,
                    fov_key=data.fov_key,
                    acquired_fovs=tuple(ordered_keys[:step]),
                    result=result,
                )
            )

    return all_results


def _prefix(group_data: ValidatedInput, ordered: pd.DataFrame, step: int) -> ValidatedInput:
    """The first ``step`` FOVs of ``ordered`` and their phenotype observations, as a visible input."""
    fovs = ordered.iloc[:step]
    return restrict_input(group_data, objects_in_fovs(group_data, fovs), fovs)


def _replay_full(strategies, group, group_data, ordered, *, objective, min_fovs):
    """Full recomputation of every strategy on every prefix (the reference)."""
    for step in range(min_fovs, len(ordered) + 1):
        started = perf_counter()
        prefix = _prefix(group_data, ordered, step)
        setup_seconds = perf_counter() - started
        for strategy in strategies:
            yield step, evaluate_strategy(
                strategy, prefix, objective=objective, group=group, setup_seconds=setup_seconds
            )


class _LazyPrefix:
    """Builds the visible prefix at most once per step, only if an estimator recomputes."""

    def __init__(self, group_data: ValidatedInput, ordered: pd.DataFrame, step: int) -> None:
        self.args = (group_data, ordered, step)
        self.data: ValidatedInput | None = None

    def __call__(self) -> tuple[ValidatedInput, float]:
        if self.data is not None:
            return self.data, 0.0
        started = perf_counter()
        self.data = _prefix(*self.args)
        return self.data, perf_counter() - started


def _replay_incremental(strategies, group, group_data, ordered, *, objective, min_fovs):
    """Feed each newly revealed FOV to incremental estimators; recompute the others."""
    replays = [
        IncrementalReplay(strategy, group_data, objective=objective, group=group)
        for strategy in strategies
    ]
    for step in range(1, len(ordered) + 1):
        started = perf_counter()
        new_fovs = ordered.iloc[step - 1 : step]
        new = restrict_input(group_data, objects_in_fovs(group_data, new_fovs), new_fovs)
        setup_seconds = perf_counter() - started
        prefix = _LazyPrefix(group_data, ordered, step)
        for replay in replays:
            result = replay.step(new, prefix, record=step >= min_fovs, setup_seconds=setup_seconds)
            if result is not None:
                yield step, result
