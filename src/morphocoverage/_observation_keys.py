"""Each observation's identity as one text that no other identity shares, for the feature tables.

Private, split out of ``feature_tables``: the columns that identify one observation
(``observation_identity``) and the text each row is matched by (``observation_keys``).

- The **FOV key** columns (the field identifier and its groups) are matched by their typed
  value (``_identity.identity_key``): FOV ``1`` and FOV ``"1"`` are two fields, as the validator
  reads them, and a feature row of the one is never attached to an observation of the other
  (#420).
- The **within-field** columns (``label_id``, ``patch_id``, ``z``/``z0``/``z1``/``c``/``t``) are
  matched as text, a whole number written as a decimal read as that number (``identity_text``),
  so ``1``, ``1.0`` and ``"1"`` are one label (owner, 2026-10-06: a CSV column with one empty
  cell is read as decimals).
- Each component is escaped before the components are joined, so a value holding the
  separator cannot make two different keys one text (#420).
- ``label_id`` and ``patch_id`` name an observation in every row or in none: one empty in every
  observation is read as absent (#404, ``_within_identity``); one empty in some rows of any
  table is refused, never matched as the text ``"nan"`` or ``"None"`` (#596). A FOV key value
  missing in a feature table is refused likewise.
- ``patch_set``, ``object_set`` and ``parent_id`` (MorphoFeature's patch identity, D84) are
  identity metadata, never features: a feature table may carry the ones the observation table
  holds, and each value must agree with the observation its row matches (#603).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from ._identity import identity_key
from ._within_identity import EMPTY_MEANS_ABSENT_WITHIN, filled_within_columns
from ._wording import count, listing
from .metadata import (
    WITHIN_FOV_METADATA_COLUMNS,
    drop_empty_optional_columns,
    empty_values,
    resolve_fov_col,
    resolve_group_cols,
    resolve_identity_cols,
)

# The columns read after the field key, matched as text (``identity_text``).
WITHIN_COLUMNS = (*EMPTY_MEANS_ABSENT_WITHIN, *WITHIN_FOV_METADATA_COLUMNS)
# MorphoFeature's patch identity beyond the match key (D84): never a feature (#603).
IDENTITY_METADATA = ("patch_set", "object_set", "parent_id")
# The separator of a key's components, and the marker of a component's type.
_SEPARATOR, _TYPE = "\x1f", "\x1e"
# The observations named in a message, at most.
_SHOWN = 3


def observation_identity(
    objects: pd.DataFrame,
    fovs: pd.DataFrame | None,
    *,
    fov_col: str | None = None,
    group_cols: Sequence[str] | None = None,
) -> tuple[str, ...]:
    """The columns that identify one observation (see the module): the FOV key columns the
    observation table holds, then ``label_id``, ``patch_id`` and ``z``/``z0``/``z1``/``c``/``t``
    when it has them; a ``label_id`` or ``patch_id`` empty in every observation is absent."""
    explicit = [c for c in (fov_col, *(group_cols or ())) if c is not None]
    objects, fovs = drop_empty_optional_columns(objects, fovs, explicit)
    field = resolve_fov_col(objects, fovs, fov_col)
    groups = resolve_group_cols(objects, fovs, group_cols)
    key = [*resolve_identity_cols(objects, fovs, groups, field), field]
    within = filled_within_columns(objects)
    within += [c for c in WITHIN_FOV_METADATA_COLUMNS if c in objects.columns]
    return tuple(c for c in dict.fromkeys([*key, *within]) if c in objects.columns)


def identity_text(value: Any) -> str:
    """A value as the identity matches it: a whole number written as a decimal (``1.0``, as
    a CSV column with an empty cell is read) as the whole number, anything else as its text."""
    if isinstance(value, float | np.floating) and np.isfinite(value) and float(value).is_integer():
        return str(int(value))
    return str(value)


def _escaped(text: str) -> str:
    """``text`` with the backslash, the separator and the type marker escaped."""
    return text.replace("\\", "\\\\").replace(_SEPARATOR, "\\x1f").replace(_TYPE, "\\x1e")


def _typed_text(value: Any) -> str:
    """A FOV key value as its typed key in one text: its text, then its type."""
    text, kind = identity_key(value)
    return f"{_escaped(text)}{_TYPE}{kind}"


def observation_keys(table: pd.DataFrame, identity: Sequence[str]) -> pd.Series:
    """Each row's identity as one text (see the module): the FOV key typed, the within-field
    columns as ``identity_text``, every component escaped. Missing values are refused before
    (``missing_identity``)."""
    text = pd.DataFrame(
        {
            c: [
                _escaped(identity_text(v)) if c in WITHIN_COLUMNS else _typed_text(v)
                for v in table[c].tolist()
            ]
            for c in identity
        },
        index=table.index,
    )
    return text.agg(_SEPARATOR.join, axis=1) if len(identity) > 1 else text.iloc[:, 0]


def shown_keys(keys: pd.Series) -> str:
    """The first identities of ``keys``, readable: components joined by ``·``, no types."""
    values = [
        " · ".join(part.split(_TYPE)[0] for part in k.split(_SEPARATOR))
        for k in dict.fromkeys(keys)
    ]
    return listing(values, _SHOWN)


def missing_identity(name: str, table: pd.DataFrame, identity: Sequence[str]) -> list[str]:
    """Why rows of ``table`` cannot be matched: an identity column missing in some rows.

    The within-field columns ``z``/``c``/``t`` are left out: an empty plane coordinate is read
    as the plane rule reads it. A ``label_id`` or ``patch_id`` empty in every observation is
    not part of ``identity`` (absent), so here an empty one is always a partial one (#596)."""
    problems = []
    for column in identity:
        if column in WITHIN_FOV_METADATA_COLUMNS:
            continue
        empty = np.asarray(empty_values(table[column]), dtype=bool)
        if empty.any():
            rows = listing([str(i) for i in table.index[empty][:_SHOWN]], _SHOWN)
            problems.append(
                f"{name!r} has no {column!r} in {count(int(empty.sum()), 'row')} (row {rows}) "
                f"of {len(table)}: {column!r} identifies each observation, so it must be filled "
                "in every row of every table (or, for label_id and patch_id, empty in every "
                "row of the observation table, when it does not apply)"
            )
    return problems


def observed_keys(objects: pd.DataFrame, identity: Sequence[str], advice: str) -> pd.Series:
    """Each observation's identity key; ``ValueError`` when an identity value is missing or two
    observations share one (``advice`` says how to tell them apart)."""
    missing = missing_identity("the observation table", objects, identity)
    if missing:
        raise ValueError("other feature tables cannot be matched: " + "; ".join(missing))
    observed = observation_keys(objects, identity)
    if observed.duplicated().any():
        twice = observed[observed.duplicated(keep=False)]
        raise ValueError(
            f"other feature tables are matched to observations by {list(identity)}, but "
            f"{count(twice.nunique(), 'observation')} of the observation table share theirs "
            f"({shown_keys(twice)}): {advice}"
        )
    return observed


def match_problems(
    name: str, table: pd.DataFrame, identity: Sequence[str], observed: pd.Series
) -> tuple[list[str], pd.Series | None]:
    """Whether ``table`` matches every observation exactly once, and its rows' keys (``None``
    when an identity value is missing, so no key could be built)."""
    problems = missing_identity(name, table, identity)
    if problems:
        return problems, None
    keys = observation_keys(table, identity)
    twice = keys[keys.duplicated(keep=False)]
    if len(twice):
        problems.append(
            f"{name!r} holds {count(twice.nunique(), 'observation')} more than once "
            f"({shown_keys(twice)})"
        )
    absent = observed[~observed.isin(keys)]
    if len(absent):
        problems.append(
            f"{name!r} lacks a row for {count(len(absent), 'observation')} of the observation "
            f"table ({shown_keys(absent)})"
        )
    extra = keys[~keys.isin(observed)]
    if len(extra):
        problems.append(
            f"{name!r} holds {count(len(extra), 'row')} matching no observation "
            f"({shown_keys(extra)})"
        )
    if len(absent) and len(extra):
        problems += _type_problems(name, identity, extra, observed)
    return problems, keys


def _key_types(keys: pd.Series, position: int) -> set[str]:
    """The value types of the FOV key component at ``position`` of ``keys`` (typed texts)."""
    return {k.split(_SEPARATOR)[position].rsplit(_TYPE, 1)[-1] for k in keys}


def _type_problems(
    name: str, identity: Sequence[str], extra: pd.Series, observed: pd.Series
) -> list[str]:
    """Why rows that print like observations match none: a FOV key column written with other
    value types than the observation table's, ``1`` against ``"1"`` (#420)."""
    problems = []
    for position, column in enumerate(identity):
        if column in WITHIN_COLUMNS:
            continue
        theirs, own = _key_types(extra, position), _key_types(observed, position)
        if theirs - own:
            problems.append(
                f"{name!r} writes {column!r} as {listing(sorted(theirs), _SHOWN)} where the "
                f"observation table writes it as {listing(sorted(own), _SHOWN)}: a field is "
                "matched by its typed identifier, so 1 and '1' are two fields. Write it with "
                f"the observation table's type (a CSV read with dtype={{{column!r}: str}} "
                "keeps text such as '001')"
            )
    return problems


def metadata_columns(table: pd.DataFrame, objects: pd.DataFrame) -> list[str]:
    """The identity metadata columns (``IDENTITY_METADATA``) of ``table`` that the observation
    table holds too: checked against it (``metadata_problems``), never features."""
    return [c for c in IDENTITY_METADATA if c in table.columns and c in objects.columns]


def _value_texts(values: pd.Series) -> pd.Series:
    """Each value as its typed text, a missing value as one marker (agreeing when both lack)."""
    empty = np.asarray(empty_values(values), dtype=bool)
    texts = [None if e else _typed_text(v) for v, e in zip(values.tolist(), empty, strict=True)]
    return pd.Series(texts, index=values.index, dtype=object)


def metadata_problems(
    name: str, table: pd.DataFrame, keys: pd.Series, objects: pd.DataFrame, observed: pd.Series
) -> list[str]:
    """Where ``table``'s identity metadata disagrees with the observation each row matches
    (``keys`` and ``observed`` as ``match_problems`` built them, every row matched once)."""
    problems = []
    for column in metadata_columns(table, objects):
        own = _value_texts(objects[column]).set_axis(observed.to_numpy())
        theirs = _value_texts(table[column]).set_axis(keys.to_numpy())
        expected = own.reindex(keys.to_numpy())
        differ = ~((theirs.isna() & expected.isna()) | (theirs == expected))
        if differ.any():
            problems.append(
                f"{name!r} gives {column!r} values that differ from the observation table's for "
                f"{count(int(differ.sum()), 'observation')} "
                f"({shown_keys(pd.Series(keys.to_numpy()[differ.to_numpy()]))}); an identity "
                "column must agree with the observation it names"
            )
    return problems


__all__ = [
    "IDENTITY_METADATA",
    "WITHIN_COLUMNS",
    "identity_text",
    "match_problems",
    "metadata_columns",
    "metadata_problems",
    "missing_identity",
    "observation_identity",
    "observation_keys",
    "observed_keys",
    "shown_keys",
]
