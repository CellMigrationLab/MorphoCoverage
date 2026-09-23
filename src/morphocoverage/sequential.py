"""Sequential evaluation on deeply sampled real acquisitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import pandas as pd

from .evaluator import evaluate_strategies
from .metadata import resolve_fov_col, resolve_group_cols
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
    acquisition_order_col: str,
    objective: ObjectiveConfig,
    fov_col: str | None = None,
    group_cols: Sequence[str] | None = None,
    min_fovs: int = 1,
) -> list[SequentialResult]:
    """Evaluate strategies after each prefix of a recorded real acquisition.

    MorphoBricks metadata defaults are resolved under the hood. By default,
    condition and repeat remain separate when available and image_file is
    preferred as the FOV identifier.
    """
    if min_fovs < 1:
        raise ValueError("min_fovs must be at least 1")

    resolved_fov_col = resolve_fov_col(objects, fovs, fov_col)
    resolved_group_cols = resolve_group_cols(objects, fovs, group_cols)

    required = [*resolved_group_cols, resolved_fov_col, acquisition_order_col]
    missing = [column for column in required if column not in fovs.columns]
    if missing:
        raise ValueError(f"fovs is missing required columns: {missing}")

    if fovs[acquisition_order_col].isna().any():
        raise ValueError("acquisition order contains missing values")

    if resolved_group_cols:
        grouped_fovs = fovs.groupby(
            list(resolved_group_cols),
            sort=False,
            dropna=False,
        )
    else:
        grouped_fovs = [((), fovs)]

    all_results: list[SequentialResult] = []
    for _, group_fovs in grouped_fovs:
        ordered = group_fovs.sort_values(acquisition_order_col, kind="stable")
        ordered_ids = ordered[resolved_fov_col].tolist()

        for step in range(min_fovs, len(ordered_ids) + 1):
            prefix_ids = ordered_ids[:step]
            prefix_fovs = ordered.iloc[:step].copy()

            if resolved_group_cols:
                key_cols = [*resolved_group_cols, resolved_fov_col]
                prefix_objects = objects.merge(
                    prefix_fovs.loc[:, key_cols].drop_duplicates(),
                    on=key_cols,
                    how="inner",
                    validate="many_to_one",
                )
            else:
                prefix_objects = objects[
                    objects[resolved_fov_col].isin(prefix_ids)
                ].copy()

            data = validate_input_tables(
                prefix_objects,
                feature_cols=feature_cols,
                fov_col=resolved_fov_col,
                fovs=prefix_fovs,
                group_cols=resolved_group_cols,
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
