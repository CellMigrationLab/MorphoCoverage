"""Prospective online acquisition session.

The session exposes the same estimator/policy machinery used by retrospective
replay without requiring a completed acquisition or knowledge of future FOVs.
One call to add_fov reveals exactly one spatial acquisition position.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import pandas as pd

from .evaluator import IncrementalReplay
from .metadata import MORPHOBRICKS_FOV_CANDIDATES, MORPHOBRICKS_GROUP_COLUMNS
from .objectives import ObjectiveConfig
from .results import StrategyResult
from .schema import ValidatedInput, validate_input_tables
from .strategies.base import SamplingStrategy


@dataclass
class _GroupSession:
    """Accumulated visible data and strategy states for one analysis group."""

    data: ValidatedInput
    replays: list[IncrementalReplay]


class OnlineSession:
    """Incrementally assess one newly acquired spatial position at a time.

    The session never receives metadata about future FOVs. Strategies with an
    EstimatorState update only from the newly acquired FOV; strategies without
    incremental support are recomputed on the accumulated visible prefix.
    """

    def __init__(
        self,
        *,
        strategies: Iterable[SamplingStrategy],
        feature_cols: Sequence[str],
        objective: ObjectiveConfig,
        fov_col: str | None = None,
        group_cols: Sequence[str] | None = None,
        observation_unit: str = "object",
    ) -> None:
        self.strategies = tuple(strategies)
        if not self.strategies:
            raise ValueError("OnlineSession needs at least one strategy")
        if any(not isinstance(strategy, SamplingStrategy) for strategy in self.strategies):
            raise TypeError("strategies must contain SamplingStrategy instances")
        if isinstance(feature_cols, str):
            raise TypeError("feature_cols must be a sequence of column names, not a string")
        self.feature_cols = tuple(feature_cols)
        if not self.feature_cols:
            raise ValueError("feature_cols must contain at least one feature")
        if not isinstance(objective, ObjectiveConfig):
            raise TypeError("objective must be an ObjectiveConfig")
        self.objective = objective
        if fov_col is not None and not isinstance(fov_col, str):
            raise TypeError("fov_col must be a column name")
        if isinstance(group_cols, str):
            raise TypeError("group_cols must be a sequence of column names, not a string")
        self._requested_fov_col = fov_col
        self._requested_group_cols = None if group_cols is None else tuple(group_cols)
        if not isinstance(observation_unit, str) or not observation_unit.strip():
            raise ValueError("observation_unit must be a non-empty description")
        self.observation_unit = observation_unit.strip()
        self._resolved_fov_col: str | None = None
        self._resolved_group_cols: tuple[str, ...] | None = None
        self._groups: dict[tuple[object, ...], _GroupSession] = {}

    @property
    def fov_col(self) -> str | None:
        """Resolved spatial-position column after the first update."""
        return self._resolved_fov_col

    @property
    def group_cols(self) -> tuple[str, ...] | None:
        """Resolved analysis-group columns after the first update."""
        return self._resolved_group_cols

    def _metadata_frame(self, fov_metadata: Mapping[str, Any] | pd.DataFrame) -> pd.DataFrame:
        if isinstance(fov_metadata, pd.DataFrame):
            if len(fov_metadata) != 1:
                raise ValueError("add_fov requires exactly one FOV metadata row")
            return fov_metadata.copy()
        if not isinstance(fov_metadata, Mapping):
            raise TypeError("fov_metadata must be a mapping or one-row DataFrame")
        return pd.DataFrame([dict(fov_metadata)])

    def _resolve_columns(self, fov: pd.DataFrame) -> tuple[str, tuple[str, ...]]:
        if self._resolved_fov_col is None:
            if self._requested_fov_col is not None:
                fov_col = self._requested_fov_col
            else:
                fov_col = next(
                    (column for column in MORPHOBRICKS_FOV_CANDIDATES if column in fov.columns),
                    None,
                )
                if fov_col is None:
                    raise ValueError(
                        "could not resolve the spatial FOV identifier from fov_metadata; "
                        "provide fov_col or position_id/image_file/fov_id"
                    )
            if self._requested_group_cols is None:
                group_cols = tuple(
                    column for column in MORPHOBRICKS_GROUP_COLUMNS if column in fov.columns
                )
            else:
                group_cols = self._requested_group_cols
            self._resolved_fov_col = fov_col
            self._resolved_group_cols = group_cols
        return self._resolved_fov_col, self._resolved_group_cols or ()

    @staticmethod
    def _stamp_metadata(
        observations: pd.DataFrame,
        fov: pd.DataFrame,
        columns: Sequence[str],
    ) -> pd.DataFrame:
        stamped = observations.copy()
        row = fov.iloc[0]
        for column in columns:
            if column not in fov.columns:
                raise ValueError(f"fov_metadata is missing required column {column!r}")
            value = row[column]
            if column in stamped.columns:
                if not stamped[column].eq(value).all():
                    raise ValueError(
                        f"observations contain values for {column!r} that conflict "
                        "with fov_metadata"
                    )
            else:
                stamped[column] = value
        return stamped

    def add_fov(
        self,
        observations: pd.DataFrame,
        fov_metadata: Mapping[str, Any] | pd.DataFrame,
    ) -> list[StrategyResult]:
        """Add one new spatial position and return one result per strategy."""
        if not isinstance(observations, pd.DataFrame):
            raise TypeError("observations must be a pandas DataFrame")

        started = perf_counter()
        fov = self._metadata_frame(fov_metadata)
        fov_col, group_cols = self._resolve_columns(fov)
        stamped = self._stamp_metadata(observations, fov, [*group_cols, fov_col])
        new = validate_input_tables(
            stamped,
            fovs=fov,
            feature_cols=self.feature_cols,
            fov_col=fov_col,
            group_cols=group_cols,
            observation_unit=self.observation_unit,
        )
        if len(new.fovs) != 1:
            raise ValueError("add_fov must resolve to exactly one spatial acquisition position")

        group_values = tuple(new.fovs[column].iloc[0] for column in group_cols)
        group = dict(zip(group_cols, group_values, strict=True))
        state = self._groups.get(group_values)

        if state is None:
            replays = [
                IncrementalReplay(strategy, new, objective=self.objective, group=group)
                for strategy in self.strategies
            ]
            state = _GroupSession(new, replays)
            self._groups[group_values] = state
        else:
            key = list(state.data.fov_key)
            existing = set(state.data.fovs.loc[:, key].itertuples(index=False, name=None))
            incoming = tuple(new.fovs.loc[:, key].iloc[0])
            if incoming in existing:
                raise ValueError(f"spatial FOV {incoming!r} has already been added")
            if (
                new.feature_cols != state.data.feature_cols
                or new.fov_key != state.data.fov_key
                or new.observation_unit != state.data.observation_unit
            ):
                raise ValueError("new FOV does not match the established session contract")
            state.data = ValidatedInput(
                objects=pd.concat([state.data.objects, new.objects], ignore_index=True),
                fovs=pd.concat([state.data.fovs, new.fovs], ignore_index=True),
                feature_cols=state.data.feature_cols,
                fov_col=state.data.fov_col,
                group_cols=state.data.group_cols,
                fov_key=state.data.fov_key,
                observation_unit=state.data.observation_unit,
            )

        setup_seconds = perf_counter() - started

        def prefix() -> tuple[ValidatedInput, float]:
            return state.data, 0.0

        results: list[StrategyResult] = []
        for replay in state.replays:
            result = replay.step(new, prefix, record=True, setup_seconds=setup_seconds)
            if result is None:
                raise RuntimeError("online updates must always produce a result")
            results.append(result)
        return results
