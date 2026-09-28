"""Prospective online acquisition session.

The session exposes the same estimator/policy machinery used by retrospective
replay without requiring a completed acquisition or knowledge of future FOVs.
One call to add_fov reveals exactly one spatial acquisition position, and is
atomic: an FOV that is refused (by validation or by any estimator) leaves the
session exactly as it was, so the same FOV can be added again once corrected.

The session resolves identity and applies the plane rule as the validator
does, one FOV at a time, using the shared-private helpers
``schema._within_fov_axes`` and ``schema._check_within_fov_axes`` (and
``metadata.drop_empty_optional_columns``): a live acquisition is refused
exactly where ``validate_input_tables`` refuses the same linked tables.

Each accepted FOV also passes the data check's within-field copy rule
(``_data_rules._check_copies``): rows that repeat another row exactly, or a
``label_id`` twice in the field, are warned about (``UserWarning``), as
``check_tables`` warns about them, since an object file read or appended
twice would otherwise count every object twice without a word.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Mapping, Sequence
from time import perf_counter
from typing import Any

import pandas as pd

from ._data_problems import Collector
from ._data_rules import _check_copies
from .data_check import validate_input_tables
from .evaluator import IncrementalReplay, restrict_input
from .metadata import (
    MORPHOBRICKS_FOV_CANDIDATES,
    MORPHOBRICKS_GROUP_COLUMNS,
    drop_empty_optional_columns,
    resolve_fov_col,
    resolve_group_cols,
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
    incremental estimators never re-copies its accumulated data. What
    estimators receive (the new FOV, the joined prefix) passes through
    ``restrict_input``, as in a replay: no unused categories (which could name
    FOVs not yet acquired) and a fresh row index.
    """

    def __init__(self, first: ValidatedInput, replays: list[IncrementalReplay]) -> None:
        self.replays = replays
        self._parts = [first]
        self._keys = {self._key_of(first)}
        self._visible: ValidatedInput | None = _visible_part(first)

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
            new = _visible_part(part)
            for replay in self.replays:
                replay.step(new, self.visible, record=False, setup_seconds=0.0)

    def visible(self) -> tuple[ValidatedInput, float]:
        """All FOVs added so far, joined on first use after each append, and the join time."""
        if self._visible is not None:
            return self._visible, 0.0
        started = perf_counter()
        # The parts are joined as validated (so a categorical column keeps its
        # dtype, as in a replay's prefix), then restricted to the visible rows.
        self._visible = restrict_input(
            self._parts[0],
            pd.concat([part.objects for part in self._parts], ignore_index=True),
            pd.concat([part.fovs for part in self._parts], ignore_index=True),
        )
        return self._visible, perf_counter() - started


def _visible_part(data: ValidatedInput) -> ValidatedInput:
    """One validated FOV as estimators receive it: only its own rows and categories."""
    return restrict_input(data, data.objects, data.fovs)


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
        # z/c/t rows of every FOV's observations so far, and every FOV row's
        # z/c/t values, for the validator's plane-as-FOV rule, which needs every
        # FOV at once.
        self._plane_rows: tuple[pd.DataFrame, pd.DataFrame] | None = None

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

    def _resolve_columns(
        self, observations: pd.DataFrame, fov: pd.DataFrame
    ) -> tuple[str, tuple[str, ...]]:
        """The FOV and group columns: fixed by the first accepted FOV, then only checked.

        Nothing is stored here; ``add_fov`` keeps the resolution once the FOV is
        accepted, so a refused first FOV does not fix the session's columns.
        """
        if self._resolved_fov_col is None:
            return self._resolved_for(observations, fov)
        self._check_contract_drift(observations, fov)
        return self._resolved_fov_col, self._resolved_group_cols or ()

    def _resolved_for(
        self, observations: pd.DataFrame, fov: pd.DataFrame
    ) -> tuple[str, tuple[str, ...]]:
        """The FOV and group columns of one FOV, resolved as the validator resolves them.

        ``metadata.resolve_fov_col`` and ``resolve_group_cols`` read both linked
        tables, the observations and the FOV row: a MorphoBricks column present
        in only one of them is an error, never a reason to fall back (a
        ``position_id`` on the observations but not in ``fov_metadata`` is
        refused, not replaced by ``image_file``). The FOV row's identity values
        are copied onto observations that lack them, as ``add_fov`` does, and an
        optional column empty in both (``position_id``, ``technical_repeat``) is
        absent.
        """
        explicit = [
            *(self._requested_group_cols or ()),
            *([self._requested_fov_col] if self._requested_fov_col is not None else []),
        ]
        from_row = [
            column
            for column in (*MORPHOBRICKS_FOV_CANDIDATES, *MORPHOBRICKS_GROUP_COLUMNS)
            if column in fov.columns and column not in observations.columns
        ]
        linked = observations.assign(**{column: fov[column].iloc[0] for column in from_row})
        linked, fov = drop_empty_optional_columns(linked, fov, explicit)
        fov_col = resolve_fov_col(linked, fov, self._requested_fov_col)
        group_cols = resolve_group_cols(linked, fov, self._requested_group_cols)
        return fov_col, tuple(group_cols)

    def _with_plane_rows(self, new: ValidatedInput) -> tuple[pd.DataFrame, pd.DataFrame] | None:
        """The plane-check tables including the new FOV's, after the validator's plane rule.

        ``validate_input_tables`` refuses a table in which an automatically
        resolved ``image_file``/``fov_id`` holds a single ``z``/``c``/``t`` value
        that varies across FOVs, on the observations or in the FOV table: each
        "FOV" may then be one plane of a position (with one row per FOV it
        warns instead). The session validates one FOV at a time, so it keeps
        each FOV's distinct z/c/t rows of observations (two copies when one
        value fills several rows, which is all the rule reads of the row count)
        and each FOV row's z/c/t values, and applies the same rule to every FOV
        so far. Returns ``None`` when the rule does not apply.
        """
        if self._requested_fov_col is not None or new.fov_col not in ("image_file", "fov_id"):
            return None
        key = list(new.fov_key)
        axes = _within_fov_axes(new.objects, self.feature_cols)
        rows = new.objects.loc[:, [*key, *axes]].drop_duplicates()
        if len(rows) == 1 and len(new.objects) > 1:
            rows = pd.concat([rows, rows], ignore_index=True)
        fovs = new.fovs.loc[:, [*key, *_within_fov_axes(new.fovs, self.feature_cols)]]
        if self._plane_rows is not None:
            previous_rows, previous_fovs = self._plane_rows
            rows = pd.concat([previous_rows, rows], ignore_index=True)
            fovs = pd.concat([previous_fovs, fovs], ignore_index=True)
        _check_within_fov_axes(
            rows, new.fov_key, self.feature_cols, fov_col_was_chosen=True, fovs=fovs
        )
        return rows, fovs

    def _check_contract_drift(self, observations: pd.DataFrame, fov: pd.DataFrame) -> None:
        """Later FOVs may not silently change the identity or grouping the first one fixed.

        Each later FOV is resolved as the first was, over its observations and
        its FOV row, so a column present in only one of them is refused here too.
        """
        fov_col, group_cols = self._resolved_for(observations, fov)
        preferred = MORPHOBRICKS_FOV_CANDIDATES[
            : MORPHOBRICKS_FOV_CANDIDATES.index(self._resolved_fov_col)
        ]
        # A lower-priority identifier means the resolved column is missing,
        # which _stamp_metadata refuses by name.
        if self._requested_fov_col is None and fov_col in preferred:
            raise ValueError(
                "fov_metadata introduces higher-priority FOV identifier(s) "
                f"{[fov_col]} after the session resolved "
                f"{self._resolved_fov_col!r}; start a new session or pass "
                "fov_col explicitly"
            )
        if self._requested_group_cols is None and group_cols != (self._resolved_group_cols or ()):
            raise ValueError(
                "fov_metadata changes the automatically resolved grouping contract "
                f"from {self._resolved_group_cols or ()} to {group_cols}; "
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

        The FOV identifier and group columns are resolved from both the
        observations and ``fov_metadata``, as ``validate_input_tables`` resolves
        them: a MorphoBricks column (``position_id``, ``condition``, ...) on
        the observations but not in ``fov_metadata`` is refused, and identity
        values given only in ``fov_metadata`` are copied onto the observations
        (they all belong to this FOV). When the FOV identifier is resolved
        automatically to ``image_file`` or ``fov_id``, the validator's
        plane-as-FOV rule is applied to all FOVs added so far: FOVs that each
        hold a single, differing ``z``/``c``/``t`` value (on the observations or
        in ``fov_metadata``) are refused, or with one observation per FOV warned
        about, as ``validate_input_tables`` does for those tables. An accepted
        FOV whose rows repeat another row exactly, or repeat a ``label_id``, is
        warned about (``UserWarning``), as ``check_tables`` warns: the rules
        count those rows as they are.
        """
        started = perf_counter()
        observations = self._as_observations(observations)
        fov = self._metadata_frame(fov_metadata)
        fov_col, group_cols = self._resolve_columns(observations, fov)
        stamped = self._stamp_metadata(observations, fov, [*group_cols, fov_col])
        validated = validate_input_tables(
            stamped,
            fovs=fov,
            feature_cols=self.feature_cols,
            fov_col=fov_col,
            group_cols=group_cols,
            observation_unit=self.observation_unit if self._unit_declared else None,
        )
        if len(validated.fovs) != 1:
            raise ValueError("add_fov must resolve to exactly one spatial acquisition position")
        plane_rows = self._with_plane_rows(validated)
        # What estimators receive: this FOV's rows only, as in a replay.
        new = _visible_part(validated)

        group_values = tuple(new.fovs[column].iloc[0] for column in group_cols)
        group = dict(zip(group_cols, group_values, strict=True))
        session = self._groups.get(group_values)
        if session is None:
            replays = [
                IncrementalReplay(strategy, new, objective=self.objective, group=group)
                for strategy in self.strategies
            ]
            added = _GroupSession(validated, replays)
        else:
            session.append(validated)
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
        _warn_copies(stamped, [*group_cols, fov_col], _GroupSession._key_of(validated))
        return results


def _warn_copies(observations: pd.DataFrame, key: Sequence[str], fov: tuple) -> None:
    """Warn, as ``check_tables`` does, about copied rows or a repeated ``label_id`` in one FOV."""
    collector = Collector()
    _check_copies(collector, observations, key)
    for problem in collector.problems:
        warnings.warn(
            f"FOV {fov!r}: {problem.title} Where: {problem.where} Fix: {problem.fix} "
            "The rules count these rows as they are.",
            UserWarning,
            stacklevel=3,
        )


__all__ = ["OnlineSession"]
