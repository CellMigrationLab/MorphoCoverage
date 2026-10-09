"""The invariants of a ``ValidatedInput`` and its trusted construction (private; issue 117).

``ValidatedInput`` is public, so building one by hand must not produce state the framework
treats as validated. Its constructor runs ``check_input`` (every invariant ``validate_input_tables``
establishes, in one pass over the tables). The framework's own subsets and copies of an input that
was already validated take ``trusted_input`` instead, which skips the checks: that path is explicit,
named and private, so replay stays as fast as before.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd

from ._observation_units import normalize_observation_unit

if TYPE_CHECKING:
    from .schema import ValidatedInput


def _fail(message: str) -> None:
    raise ValueError(f"ValidatedInput: {message}; build one with validate_input_tables")


def _check_roles(data: ValidatedInput) -> None:
    """The declared columns: names, no repeats, the FOV column last in the key."""
    for name in ("feature_cols", "group_cols", "fov_key"):
        value = getattr(data, name)
        if not isinstance(value, tuple) or not all(isinstance(c, str) and c for c in value):
            _fail(f"{name} must be a tuple of column names, not {value!r}")
        if len(set(value)) != len(value):
            _fail(f"{name} repeats a column: {value}")
    if not data.feature_cols:
        _fail("feature_cols is empty")
    if not data.fov_key or data.fov_key[-1] != data.fov_col:
        _fail(f"fov_key {data.fov_key} must end with the FOV column {data.fov_col!r}")
    if not set(data.group_cols) <= set(data.fov_key):
        _fail(f"group_cols {data.group_cols} must be part of fov_key {data.fov_key}")
    if set(data.feature_cols) & set(data.fov_key):
        _fail("a column is both a feature and part of the FOV identity")


def _check_unit(data: ValidatedInput) -> None:
    """The observation unit is canonical and its declaration a real boolean."""
    if not isinstance(data.observation_unit, str):
        _fail(f"observation_unit must be text, not {data.observation_unit!r}")
    if normalize_observation_unit(data.observation_unit) != data.observation_unit:
        _fail(f"observation_unit {data.observation_unit!r} is not canonical")
    for name in ("observation_unit_declared", "fovs_from_observations"):
        if not isinstance(getattr(data, name), bool):
            _fail(f"{name} must be True or False")


def check_input(data: ValidatedInput) -> None:
    """Refuse a ``ValidatedInput`` the validator could not have produced: missing or repeated
    columns, a FOV table that is not one row per FOV, observations of FOVs it does not list,
    missing identifiers, non-numeric or infinite features."""
    from .schema import _check_feature_values, _check_identity_values, _validated_fov_table

    for name in ("objects", "fovs"):
        if not isinstance(getattr(data, name), pd.DataFrame):
            _fail(f"{name} must be a DataFrame")
    _check_roles(data)
    _check_unit(data)
    for table, name, columns in (
        (data.objects, "objects", (*data.fov_key, *data.feature_cols)),
        (data.fovs, "fovs", data.fov_key),
    ):
        absent = [c for c in columns if c not in table.columns]
        if absent:
            _fail(f"{name} lacks the column(s) {absent}")
    _check_identity_values(data.objects, data.fov_key)
    _validated_fov_table(data.objects, data.fovs, data.fov_key)  # one row per FOV, all listed
    _check_feature_values(data.objects, data.feature_cols)


def trusted_input(cls: type[ValidatedInput], **fields: Any) -> ValidatedInput:
    """A ``ValidatedInput`` built from fields already known to satisfy the invariants (a subset
    or copy of a validated one), without running the checks. Framework code only."""
    instance = object.__new__(cls)
    for name, value in fields.items():
        object.__setattr__(instance, name, value)
    object.__setattr__(instance, "_feature_cache", {})
    return instance


__all__ = ["check_input", "trusted_input"]
