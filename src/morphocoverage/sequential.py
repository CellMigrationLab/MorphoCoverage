"""Sequential evaluation on deeply sampled real acquisitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import pandas as pd

from .evaluator import evaluate_strategies
from .objectives import ObjectiveConfig
from .results import StrategyResult
from .schema import validate_input_tables
from .strategies.base import SamplingStrategy


@dataclass(frozen=True)
class SequentialResult:
    """Strategy output associated with one acquisition prefix."""

    step: int
    acquired_fov_ids: tuple[object, ...]
    result: StrategyResult


def evaluate_acquisition_prefixes(
    objects: pd.DataFrame,
    fovs: pd.DataFrame,
    *,
    strategies: Iterable[SamplingStrategy],
    feature_cols: Sequence[str],
    fov_col: str,
    acquisition_order_col: str,
    objective: ObjectiveConfig,
    group_cols: Sequence[str] | None = None,
    min_fovs: int = 1,
) -> list[SequentialResult]:
    """Evaluate strategies after each prefix of a recorded real acquisition.

    This function preserves the real acquisition order and exposes only FOVs
    available at each step. It does not define how withheld FOVs are scored;
    benchmarking outcomes remain separate from strategy execution.
    """
    if min_fovs < 1:
        raise ValueError("min_fovs must be at least 1")
    if acquisition_order_col not in fovs.columns:
        raise ValueError(
            f"fovs is missing acquisition-order column {acquisition_order_col!r}"
        )

    group_cols = tuple(group_cols or ())
    required = [*group_cols, fov_col, acquisition_order_col]
    missing = [column for column in required if column not in fovs.columns]
    if missing:
        raise ValueError(f"fovs is missing required columns: {missing}")

    if fovs[acquisition_order_col].isna().any():
        raise ValueError("acquisition order contains missing values")

    if group_cols:
        grouped_fovs = fovs.groupby(list(group_cols), sort=False, dropna=False)
    else:
        grouped_fovs = [((), fovs)]

    all_results: list[SequentialResult] = []
    for _, group_fovs in grouped_fovs:
        ordered = group_fovs.sort_values(acquisition_order_col, kind="stable")
        ordered_ids = ordered[fov_col].tolist()

        for step in range(min_fovs, len(ordered_ids) + 1):
            prefix_ids = ordered_ids[:step]
            prefix_fovs = ordered.iloc[:step].copy()

            if group_cols:
                key_frame = prefix_fovs.loc[:, [*group_cols, fov_col]]
                prefix_objects = objects.merge(
                    key_frame.drop_duplicates(),
                    on=[*group_cols, fov_col],
                    how="inner",
                    validate="many_to_one",
                )
            else:
                prefix_objects = objects[objects[fov_col].isin(prefix_ids)].copy()

            data = validate_input_tables(
                prefix_objects,
                feature_cols=feature_cols,
                fov_col=fov_col,
                fovs=prefix_fovs,
                group_cols=group_cols,
            )
            for result in evaluate_strategies(
                strategies,
                data,
                objective=objective,
            ):
                all_results.append(
                    SequentialResult(
                        step=step,
                        acquired_fov_ids=tuple(prefix_ids),
                        result=result,
                    )
                )

    return all_results
