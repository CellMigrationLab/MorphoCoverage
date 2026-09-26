"""Prospective online acquisition session.

The session exposes the same estimator/policy machinery used by retrospective
replay without requiring a completed acquisition or knowledge of future FOVs.
One call to add_fov reveals exactly one spatial acquisition position.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace
from time import perf_counter
from typing import Any

import pandas as pd

from .data_check import validate_input_tables
from .evaluator import IncrementalReplay
from .metadata import (
    EMPTY_MEANS_ABSENT,
    MORPHOBRICKS_FOV_CANDIDATES,
    MORPHOBRICKS_GROUP_COLUMNS,
    OPTIONAL_GROUP_COLUMNS,
    column_has_values,
)
from .objectives import ObjectiveConfig
from .results import StrategyResult
from .schema import ValidatedInput, resolve_observation_unit
from .strategies.base import SamplingStrategy


class _GroupSession:
    """Visible FOVs and strategy states of one analysis group.

    New FOVs are kept as parts and joined only when an estimator without
    incremental support asks for the whole visible prefix, so a session of
    incremental estimators never re-copies its accumulated data.
    """

    def __init__(self, first: ValidatedInput, replays: list[IncrementalReplay]) -> None:
        self.replays = replays
        self._parts = [first]
        self._keys = {self._key_of(first)}
        self._visible: ValidatedInput | None = first

    @staticmethod
    def _key_of(new: ValidatedInput) -> tuple:
        """FOV key of a one-FOV input."""
        return tuple(new.fovs.loc[:, list(new.fov_key)].iloc[0])

    def append(self, new: ValidatedInput) -> None:
        """Add one validated FOV; it must be new and follow the group's contract."""
        first = self._parts[0]
        if (new.feature_cols, new.fov_key, new.observation_unit) != (
            first.feature_cols,
            first.fov_key,
            first.observation_unit,
        ):
            raise ValueError("new FOV does not match the established session contract")
        key = self._key_of(new)
        if key in self._keys:
            raise ValueError(f"spatial FOV {key!r} has already been added")
        self._keys.add(key)
        self._parts.append(new)
        self._visible = None

    def visible(self) -> tuple[ValidatedInput, float]:
        """All FOVs added so far, joined on first use after each append, and the join time."""
        if self._visible is not None:
            return self._visible, 0.0
        started = perf_counter()
        first = self._parts[0]
        self._visible = replace(
            first,
            objects=pd.concat([part.objects for part in self._parts], ignore_index=True),
            fovs=pd.concat([part.fovs for part in self._parts], ignore_index=True),
        )
        return self._visible, perf_counter() - started


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
        observation_unit: str | None = None,
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
        self.observation_unit, self._unit_declared = resolve_observation_unit(observation_unit)
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
        """The new FOV's metadata as a one-row table."""
        if isinstance(fov_metadata, pd.DataFrame):
            if len(fov_metadata) != 1:
                raise ValueError("add_fov requires exactly one FOV metadata row")
            return fov_metadata.copy()
        if not isinstance(fov_metadata, Mapping):
            raise TypeError("fov_metadata must be a mapping or one-row DataFrame")
        return pd.DataFrame([dict(fov_metadata)])

    def _resolve_columns(self, fov: pd.DataFrame) -> tuple[str, tuple[str, ...]]:
        """The FOV and group columns: fixed by the first FOV, then only checked."""
        if self._resolved_fov_col is None:
            self._resolved_fov_col = self._requested_fov_col or self._detect_fov_col(fov)
            self._resolved_group_cols = (
                self._default_group_cols(fov)
                if self._requested_group_cols is None
                else self._requested_group_cols
            )
        else:
            self._check_contract_drift(fov)
        return self._resolved_fov_col, self._resolved_group_cols or ()

    @staticmethod
    def _detect_fov_col(fov: pd.DataFrame) -> str:
        """The first MorphoBricks FOV identifier present, in order of preference.

        An empty ``position_id`` (a producer that writes every column) is skipped.
        """
        for column in MORPHOBRICKS_FOV_CANDIDATES:
            if column in fov.columns and (
                column not in EMPTY_MEANS_ABSENT or column_has_values(fov, column)
            ):
                return column
        raise ValueError(
            "could not resolve the spatial FOV identifier from fov_metadata; "
            "provide fov_col or position_id/image_file/fov_id"
        )

    @staticmethod
    def _default_group_cols(fov: pd.DataFrame) -> tuple[str, ...]:
        """The canonical group columns present; an optional one only when it holds a value."""
        return tuple(
            column
            for column in MORPHOBRICKS_GROUP_COLUMNS
            if column in fov.columns
            and (column not in OPTIONAL_GROUP_COLUMNS or column_has_values(fov, column))
        )

    def _check_contract_drift(self, fov: pd.DataFrame) -> None:
        """Later FOVs may not silently change the identity or grouping the first one fixed."""
        if self._requested_fov_col is None:
            preferred = MORPHOBRICKS_FOV_CANDIDATES[
                : MORPHOBRICKS_FOV_CANDIDATES.index(self._resolved_fov_col)
            ]
            # An empty position_id (a producer that writes every column) is not an
            # identifier, exactly as _detect_fov_col skips it.
            higher_priority = [
                column
                for column in preferred
                if column in fov.columns
                and (column not in EMPTY_MEANS_ABSENT or column_has_values(fov, column))
            ]
            if higher_priority:
                raise ValueError(
                    "fov_metadata introduces higher-priority FOV identifier(s) "
                    f"{higher_priority} after the session resolved "
                    f"{self._resolved_fov_col!r}; start a new session or pass "
                    "fov_col explicitly"
                )
        if self._requested_group_cols is None:
            later_groups = self._default_group_cols(fov)
            if later_groups != (self._resolved_group_cols or ()):
                raise ValueError(
                    "fov_metadata changes the automatically resolved grouping contract "
                    f"from {self._resolved_group_cols or ()} to {later_groups}; "
                    "start a new session or pass group_cols explicitly"
                )

    @staticmethod
    def _stamp_metadata(
        observations: pd.DataFrame,
        fov: pd.DataFrame,
        columns: Sequence[str],
    ) -> pd.DataFrame:
        """Copy identity values from the FOV row onto the observations; conflicts raise."""
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

    def _as_observations(self, observations: pd.DataFrame) -> pd.DataFrame:
        """The observation rows; an FOV with none may omit or leave untyped the features."""
        if not isinstance(observations, pd.DataFrame):
            raise TypeError("observations must be a pandas DataFrame")
        if len(observations):
            return observations
        # No values exist to convert, so typing the empty feature columns loses nothing.
        empty = observations.copy()
        for column in self.feature_cols:
            empty[column] = pd.Series(dtype=float)
        return empty

    def add_fov(
        self,
        observations: pd.DataFrame,
        fov_metadata: Mapping[str, Any] | pd.DataFrame,
    ) -> list[StrategyResult]:
        """Add one new spatial position and return one result per strategy.

        ``observations`` holds the new position's phenotype rows (possibly
        none) and ``fov_metadata`` its one metadata row.
        """
        started = perf_counter()
        observations = self._as_observations(observations)
        fov = self._metadata_frame(fov_metadata)
        fov_col, group_cols = self._resolve_columns(fov)
        new = validate_input_tables(
            self._stamp_metadata(observations, fov, [*group_cols, fov_col]),
            fovs=fov,
            feature_cols=self.feature_cols,
            fov_col=fov_col,
            group_cols=group_cols,
            observation_unit=self.observation_unit if self._unit_declared else None,
        )
        if len(new.fovs) != 1:
            raise ValueError("add_fov must resolve to exactly one spatial acquisition position")

        group_values = tuple(new.fovs[column].iloc[0] for column in group_cols)
        group = dict(zip(group_cols, group_values, strict=True))
        session = self._groups.get(group_values)
        if session is None:
            replays = [
                IncrementalReplay(strategy, new, objective=self.objective, group=group)
                for strategy in self.strategies
            ]
            session = self._groups[group_values] = _GroupSession(new, replays)
        else:
            session.append(new)
        setup_seconds = perf_counter() - started
        return [
            replay.step(new, session.visible, record=True, setup_seconds=setup_seconds)
            for replay in session.replays
        ]
