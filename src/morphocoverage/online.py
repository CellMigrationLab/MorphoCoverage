"""Prospective online acquisition session.

The session exposes the same estimator/policy machinery used by retrospective
replay without requiring a completed acquisition or knowledge of future FOVs.
One call to add_fov reveals exactly one spatial acquisition position, and is
atomic: an FOV that is refused (by validation or by any estimator) leaves the
session exactly as it was, so the same FOV can be added again once corrected.
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
from .schema import (
    ValidatedInput,
    _check_within_fov_axes,
    _within_fov_axes,
    resolve_observation_unit,
)
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

    def remove_last(self) -> None:
        """Undo the last ``append`` and rebuild every strategy's state without that FOV.

        An estimator may have updated its state before another one refused the
        FOV, so the states are rebuilt from the FOVs accepted before it. Estimators
        are pure functions of the FOVs they have seen (the incremental contract),
        so the rebuilt states equal the ones before the refused FOV.
        """
        removed = self._parts.pop()
        self._keys.discard(self._key_of(removed))
        self._visible = None
        first = self._parts[0]
        self.replays = [
            IncrementalReplay(r.strategy, first, objective=r.objective, group=r.group)
            for r in self.replays
        ]
        for part in self._parts:
            for replay in self.replays:
                replay.step(part, self.visible, record=False, setup_seconds=0.0)

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
        # With an automatically resolved file-level FOV identifier: the distinct
        # z/c/t rows of every FOV so far, for the validator's plane-as-FOV rule,
        # which needs every FOV at once.
        self._plane_rows: pd.DataFrame | None = None

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
        """The FOV and group columns: fixed by the first accepted FOV, then only checked.

        Nothing is stored here; ``add_fov`` keeps the resolution once the FOV is
        accepted, so a refused first FOV does not fix the session's columns.
        """
        if self._resolved_fov_col is None:
            fov_col = self._requested_fov_col or self._detect_fov_col(fov)
            group_cols = (
                self._default_group_cols(fov)
                if self._requested_group_cols is None
                else self._requested_group_cols
            )
            return fov_col, tuple(group_cols)
        self._check_contract_drift(fov)
        return self._resolved_fov_col, self._resolved_group_cols or ()

    def _with_plane_rows(self, new: ValidatedInput) -> pd.DataFrame | None:
        """The plane-check rows including the new FOV's, after the validator's plane rule.

        ``validate_input_tables`` refuses a table in which an automatically
        resolved ``image_file``/``fov_id`` holds a single ``z``/``c``/``t`` value
        that varies across FOVs: each "FOV" may then be one plane of a position.
        The session validates one FOV at a time, so it keeps each FOV's distinct
        z/c/t rows (two copies when one value fills several rows, which is all
        the rule reads of the row count) and applies the same rule to every FOV
        so far. Returns ``None`` when the rule does not apply.
        """
        if self._requested_fov_col is not None or new.fov_col not in ("image_file", "fov_id"):
            return None
        axes = _within_fov_axes(new.objects, self.feature_cols)
        rows = new.objects.loc[:, [*new.fov_key, *axes]].drop_duplicates()
        if len(rows) == 1 and len(new.objects) > 1:
            rows = pd.concat([rows, rows], ignore_index=True)
        if self._plane_rows is not None:
            rows = pd.concat([self._plane_rows, rows], ignore_index=True)
        _check_within_fov_axes(rows, new.fov_key, self.feature_cols, fov_col_was_chosen=True)
        return rows

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
        none) and ``fov_metadata`` its one metadata row. The call is atomic:
        when validation or any estimator refuses the FOV, the error is raised
        and the session is left as it was before the call.

        When the FOV identifier is resolved automatically to ``image_file`` or
        ``fov_id``, the validator's plane-as-FOV rule is applied to all FOVs
        added so far: FOVs that each hold a single, differing ``z``/``c``/``t``
        value are refused, as ``validate_input_tables`` refuses that table.
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
        plane_rows = self._with_plane_rows(new)

        group_values = tuple(new.fovs[column].iloc[0] for column in group_cols)
        group = dict(zip(group_cols, group_values, strict=True))
        session = self._groups.get(group_values)
        if session is None:
            replays = [
                IncrementalReplay(strategy, new, objective=self.objective, group=group)
                for strategy in self.strategies
            ]
            added = _GroupSession(new, replays)
        else:
            session.append(new)
            added = session
        setup_seconds = perf_counter() - started
        try:
            results = [
                replay.step(new, added.visible, record=True, setup_seconds=setup_seconds)
                for replay in added.replays
            ]
        except Exception:
            if session is not None:
                session.remove_last()
            raise
        # Accepted: only now does the FOV become part of the session.
        self._groups[group_values] = added
        self._resolved_fov_col, self._resolved_group_cols = fov_col, group_cols
        self._plane_rows = plane_rows
        return results
