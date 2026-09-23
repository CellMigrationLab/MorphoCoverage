"""High-level entry points for strategy comparison on feature tables."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import pandas as pd

from .evaluator import evaluate_strategies
from .objectives import ObjectiveConfig
from .results import StrategyResult
from .schema import validate_input_tables
from .strategies.base import SamplingStrategy


def run_strategies(
    objects: pd.DataFrame,
    *,
    strategies: Iterable[SamplingStrategy],
    feature_cols: Sequence[str],
    objective: ObjectiveConfig,
    fov_col: str | None = None,
    fovs: pd.DataFrame | None = None,
    group_cols: Sequence[str] | None = None,
) -> list[StrategyResult]:
    """Validate one feature dataset and run strategies within resolved groups.

    By default, canonical MorphoBricks condition and repeat columns are kept
    separate when available. Pass group_cols=[] only when intentional pooling
    is scientifically appropriate.
    """
    data = validate_input_tables(
        objects,
        feature_cols=feature_cols,
        fov_col=fov_col,
        fovs=fovs,
        group_cols=group_cols,
    )
    return evaluate_strategies(strategies, data, objective=objective)
