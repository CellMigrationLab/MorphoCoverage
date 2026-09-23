"""Sequential evaluation on deeply sampled real acquisitions."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import pandas as pd

from .evaluator import evaluate_strategy, iter_analysis_groups, objects_in_fovs, restrict_input
from .objectives import ObjectiveConfig
from .results import StrategyResult
from .schema import validate_input_tables
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
) -> list[SequentialResult]:
    """Evaluate strategies after each prefix of a recorded real acquisition.

    The complete linked tables are validated once, so objects from FOVs absent
    from the FOV table raise instead of never being revealed. Acquisition order
    must be strictly defined within each analysis group: ties raise, because
    the simulated history would otherwise depend on table row order.

    MorphoBricks metadata defaults are resolved under the hood. By default,
    condition and repeat remain separate when available and image_file is
    preferred as the FOV identifier.
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
    )
    strategy_list = list(strategies)
    key = list(data.fov_key)

    all_results: list[SequentialResult] = []
    for group, group_data in iter_analysis_groups(data):
        order = group_data.fovs[acquisition_order_col]
        if order.duplicated().any():
            tied = sorted(order[order.duplicated(keep=False)].unique().tolist())
            raise ValueError(
                f"acquisition order has ties within analysis group {group}: {tied[:5]}. "
                "When pooling, acquisition order must be defined across the pooled FOVs."
            )
        ordered = group_data.fovs.sort_values(acquisition_order_col, kind="stable")
        ordered_keys = list(ordered.loc[:, key].itertuples(index=False, name=None))

        for step in range(min_fovs, len(ordered) + 1):
            prefix_fovs = ordered.iloc[:step]
            prefix = restrict_input(
                group_data, objects_in_fovs(group_data, prefix_fovs), prefix_fovs
            )
            for strategy in strategy_list:
                all_results.append(
                    SequentialResult(
                        step=step,
                        fov_key=data.fov_key,
                        acquired_fovs=tuple(ordered_keys[:step]),
                        result=evaluate_strategy(
                            strategy, prefix, objective=objective, group=group
                        ),
                    )
                )

    return all_results
