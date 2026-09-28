"""High-level entry points for strategy comparison on phenotype feature tables."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import pandas as pd

from .data_check import validate_input_tables
from .evaluator import evaluate_strategies
from .objectives import ObjectiveConfig
from .results import StrategyResult
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
    observation_unit: str | None = None,
    n_jobs: int = 1,
) -> list[StrategyResult]:
    """Validate the observation and FOV tables and run strategies within resolved groups.

    The tables are validated by ``validate_input_tables`` (its arguments).
    Pass ``fovs``, the table of every acquired FOV: without it the FOVs are
    those seen in the observations, so acquired FOVs without observations are
    missing and every count of effort is optimistic (a warning says so). By
    default, the canonical MorphoBricks ``condition``, ``repeat`` and, when
    present, ``technical_repeat`` columns are kept separate. Pass
    ``group_cols=[]`` only when intentional pooling is scientifically
    appropriate. Each strategy returns one ``StrategyResult`` per analysis
    group, groups in order of first appearance in the FOV table and
    strategies in the given order within each. ``n_jobs`` runs every (group,
    strategy) pair in that many worker processes (``-1``: every CPU); results
    are identical for every value, apart from timings (bit for bit when BLAS
    is single-threaded; see ``morphocoverage._parallel``).
    """
    data = validate_input_tables(
        objects,
        feature_cols=feature_cols,
        fov_col=fov_col,
        fovs=fovs,
        group_cols=group_cols,
        observation_unit=observation_unit,
    )
    return evaluate_strategies(strategies, data, objective=objective, n_jobs=n_jobs)
