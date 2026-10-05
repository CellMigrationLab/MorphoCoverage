"""Other feature tables for the same observations: validated, then attached as named blocks.

The observation table holds the **explainable** features, each tested one by one (the
discovery, the heatmap, SuperPlots, "Did I image enough?" per feature). Features that are
not read one by one, such as the deep-learning embeddings of a DINO model, come as **other
feature tables**, each with a name ("DINOv2", "DINOv3"), so that several models can be
loaded side by side and compared (owner, 2026-10-05). They are never tested feature by
feature; with the explainable features they make the space of the maps, the populations,
unusual observations, novelty, the copy checks and the declared fingerprint
(``workflows``, ``docs/data_format.md``).

``attach_feature_tables`` validates every table against the observation table and returns
the observation table with every table's columns attached (``FeatureTables``). One row of a
table is one observation, identified as the observation table identifies it
(``observation_identity``): the FOV key columns the observation table holds (the analysis
groups and the field identifier, as ``validate_input_tables`` resolves them), then
``label_id`` and the within-field coordinates ``z``, ``z0``, ``z1``, ``c``, ``t`` when the
observation table has them. Values are matched as text, so ``1`` and ``"1"`` are one label.
A table must:

- hold every identity column and match every observation exactly once: no observation
  missing, no row matching none, no observation twice;
- hold numeric feature columns only (every column but the identity), none named like
  MorphoBricks metadata (``condition``, ``z``, ``label_id``, ...);
- hold no infinite value; missing values are kept and counted (``missing``), never
  imputed or dropped, as in the observation table;
- use column names found neither in the observation table nor in another table;
- have a non-empty name of its own (``"explainable"`` and ``"combined"`` name the
  observation table's features and the weighted combined space).

Every problem is collected and raised at once (``ValueError``), naming the table.

``table_weights`` gives the weight of each column in the weighted combined space: every
column is standardised as the analysis standardises it now, then a table of ``p`` columns
is multiplied by ``sqrt(p_explainable / p)``, so every table carries the same total
variance as the explainable one (``p_explainable``, each of its columns at weight 1). This
is the MFA idea of equal weight per table, with each table scaled by the square root of its
number of columns rather than by its first singular value: the weights do not depend on the
data, so a definition of new fitted on the fields seen so far keeps the same geometry at
every step, and with few observations a first singular value would be noisy. Without
other tables nothing is weighted and every analysis is as before.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ._former_repeat import Reader
from ._wording import count, listing
from .metadata import (
    MORPHOBRICKS_FOV_CANDIDATES,
    MORPHOBRICKS_SHARED_COLUMNS,
    TARGET_COLUMN,
    WITHIN_FOV_METADATA_COLUMNS,
    drop_empty_optional_columns,
    resolve_fov_col,
    resolve_group_cols,
    resolve_identity_cols,
)

# The MorphoBricks column naming an observation within its field.
LABEL_COLUMN = "label_id"
# The names of the observation table's features and of the weighted combined space.
EXPLAINABLE = "explainable"
COMBINED = "combined"
RESERVED_NAMES = (EXPLAINABLE, COMBINED)
# Columns that describe a row rather than measure it: never a feature of another table.
_METADATA = {
    *MORPHOBRICKS_SHARED_COLUMNS,
    *MORPHOBRICKS_FOV_CANDIDATES,
    *WITHIN_FOV_METADATA_COLUMNS,
    TARGET_COLUMN,
    LABEL_COLUMN,
    "repeat",
}
# The observations named in a message, at most.
_SHOWN = 3


@dataclass(frozen=True)
class FeatureTables:
    """What ``attach_feature_tables`` checked and attached (see the module).

    - ``objects``: the observation table, its rows and index unchanged, with every table's
      feature columns attached in its row order;
    - ``columns``: each table's name and its feature columns, in the order given;
    - ``missing``: each table's missing values, and ``incomplete`` the observations holding
      at least one;
    - ``identity``: the columns that identify an observation.
    """

    objects: pd.DataFrame
    columns: dict[str, tuple[str, ...]]
    missing: dict[str, int]
    incomplete: dict[str, int]
    identity: tuple[str, ...]

    def lines(self) -> list[str]:
        """One sentence per table: its columns, how it matched, its missing values."""
        key = ", ".join(self.identity)
        lines = []
        for name, columns in self.columns.items():
            missing = self.missing[name]
            gaps = (
                f"{count(missing, 'missing value')} in {count(self.incomplete[name], 'row')}, "
                "kept as missing"
                if missing
                else "no missing value"
            )
            lines.append(
                f"{name}: {count(len(columns), 'feature column')}, every observation matched "
                f"once by {key}; {gaps}."
            )
        return lines

    def summary(self) -> str:
        """The tables in sentences, one line each."""
        return "\n".join(self.lines())


def observation_identity(
    objects: pd.DataFrame,
    fovs: pd.DataFrame | None,
    *,
    fov_col: str | None = None,
    group_cols: Sequence[str] | None = None,
) -> tuple[str, ...]:
    """The columns that identify one observation (see the module): the FOV key columns the
    observation table holds, then ``label_id`` and ``z``/``z0``/``z1``/``c``/``t`` when it
    has them."""
    explicit = [c for c in (fov_col, *(group_cols or ())) if c is not None]
    objects, fovs = drop_empty_optional_columns(objects, fovs, explicit)
    field = resolve_fov_col(objects, fovs, fov_col)
    groups = resolve_group_cols(objects, fovs, group_cols)
    key = [*resolve_identity_cols(objects, fovs, groups, field), field]
    within = [LABEL_COLUMN, *WITHIN_FOV_METADATA_COLUMNS]
    return tuple(c for c in dict.fromkeys([*key, *within]) if c in objects.columns)


def _keys(table: pd.DataFrame, identity: Sequence[str]) -> pd.Series:
    """Each row's identity as one text, so ``1`` and ``"1"`` match."""
    text = table[list(identity)].astype(str)
    return text.agg("\x1f".join, axis=1) if len(identity) > 1 else text.iloc[:, 0]


def _shown(keys: pd.Series) -> str:
    """The first identities of ``keys``, readable."""
    values = [k.replace("\x1f", " · ") for k in dict.fromkeys(keys)]
    return listing(values, _SHOWN)


def _name_problems(name: object, seen: list[str]) -> list[str]:
    """Why ``name`` cannot name a table: empty, reserved, or another table's (any case)."""
    if not isinstance(name, str) or not name.strip():
        return [f"a table name must be a non-empty text, not {name!r}"]
    folded = name.strip().casefold()
    if folded in RESERVED_NAMES:
        reason = (
            f"{name!r} names the observation table's features or the combined space; "
            "give the table another name"
        )
        return [reason]
    if folded in seen:
        return [f"{name!r} names two tables (names are compared ignoring case)"]
    return []


def _match_problems(
    name: str, table: pd.DataFrame, identity: Sequence[str], observed: pd.Series
) -> list[str]:
    """Whether ``table`` matches every observation exactly once."""
    keys = _keys(table, identity)
    problems = []
    twice = keys[keys.duplicated(keep=False)]
    if len(twice):
        problems.append(
            f"{name!r} holds {count(twice.nunique(), 'observation')} more than once "
            f"({_shown(twice)})"
        )
    absent = observed[~observed.isin(keys)]
    if len(absent):
        problems.append(
            f"{name!r} lacks a row for {count(len(absent), 'observation')} of the observation "
            f"table ({_shown(absent)})"
        )
    extra = keys[~keys.isin(observed)]
    if len(extra):
        problems.append(
            f"{name!r} holds {count(len(extra), 'row')} matching no observation ({_shown(extra)})"
        )
    return problems


def _column_problems(
    name: str,
    table: pd.DataFrame,
    columns: list[Any],
    objects: pd.DataFrame,
    taken: Mapping[str, str],
) -> list[str]:
    """Whether the feature columns of ``table`` are numeric, finite, and its own."""
    problems = []
    if not columns:
        problems.append(f"{name!r} holds no feature column beside the identity columns")
    metadata = [c for c in columns if str(c) in _METADATA]
    if metadata:
        problems.append(
            f"{name!r} holds metadata columns {metadata}, which are never features; keep only "
            "the identity columns and the features"
        )
    clash = [c for c in columns if c in objects.columns]
    if clash:
        problems.append(
            f"{name!r} has columns of the observation table: {listing(clash, _SHOWN)}; rename "
            "them, as each feature must have one name"
        )
    shared: dict[str, list[Any]] = {}
    for column in columns:
        if column in taken:
            shared.setdefault(taken[column], []).append(column)
    for other, both in shared.items():
        problems.append(
            f"{other!r} and {name!r} share columns {listing(both, _SHOWN)}; rename them, as "
            "each feature must have one name"
        )
    numeric = [
        c
        for c in columns
        if pd.api.types.is_numeric_dtype(table[c]) and not pd.api.types.is_bool_dtype(table[c])
    ]
    words = [c for c in columns if c not in numeric]
    if words:
        problems.append(
            f"{name!r} has columns that are not numbers: {listing(words, _SHOWN)} (a feature "
            "holds real numbers; an identity column must be named as in the observation table)"
        )
    values = table[numeric].to_numpy(dtype=float) if numeric else np.empty((0, 0))
    infinite = int(np.isinf(values).sum())
    if infinite:
        problems.append(
            f"{name!r} holds {count(infinite, 'infinite value')}, which no feature may hold "
            "(a missing value is allowed)"
        )
    return problems


def attach_feature_tables(
    objects: pd.DataFrame,
    fovs: pd.DataFrame | None,
    feature_tables: Mapping[str, pd.DataFrame],
    *,
    fov_col: str | None = None,
    group_cols: Sequence[str] | None = None,
) -> FeatureTables:
    """Validate ``feature_tables`` (a name for each table) against the observation table and
    attach their columns to it (see the module). ``fov_col`` and ``group_cols`` are as for
    ``validate_input_tables``, which identify the fields."""
    if not isinstance(objects, pd.DataFrame):
        raise TypeError("objects must be a pandas DataFrame")
    if not isinstance(feature_tables, Mapping) or not feature_tables:
        raise TypeError("feature_tables must map each table's name to its DataFrame")
    reader = Reader()
    objects = reader.table(objects, "objects")
    fovs = reader.table(fovs, "fovs")
    tables = {
        name: reader.table(table, f"feature table {name!r}")
        for name, table in feature_tables.items()
    }
    reader.warn()
    identity = observation_identity(objects, fovs, fov_col=fov_col, group_cols=group_cols)
    observed = _keys(objects, identity)
    if observed.duplicated().any():
        twice = observed[observed.duplicated(keep=False)]
        raise ValueError(
            f"other feature tables are matched to observations by {list(identity)}, but "
            f"{count(twice.nunique(), 'observation')} of the observation table share theirs "
            f"({_shown(twice)}): add a label_id column naming each observation in its field"
        )
    problems: list[str] = []
    seen: list[str] = []
    taken: dict[str, str] = {}
    columns: dict[str, tuple[str, ...]] = {}
    blocks = []
    missing: dict[str, int] = {}
    incomplete: dict[str, int] = {}
    for name, table in tables.items():
        named = _name_problems(name, seen)
        problems += named
        label = str(name)
        if not isinstance(table, pd.DataFrame):
            problems.append(f"the table {label!r} must be a pandas DataFrame")
            continue
        seen.append(label.strip().casefold())
        absent = [c for c in identity if c not in table.columns]
        if absent:
            problems.append(
                f"{label!r} lacks the identity columns {absent}: each row must name its "
                f"observation by {list(identity)}"
            )
            continue
        features = [c for c in table.columns if c not in identity]
        found = _match_problems(label, table, identity, observed)
        found += _column_problems(label, table, features, objects, taken)
        taken.update({c: label for c in features})
        problems += found
        if found or named:
            continue
        values = table[features].set_axis(_keys(table, identity).to_numpy(), axis=0)
        block = values.reindex(observed.to_numpy()).set_axis(objects.index, axis=0)
        blocks.append(block)
        columns[label] = tuple(features)
        missing[label] = int(block.isna().to_numpy().sum())
        incomplete[label] = int(block.isna().any(axis=1).sum())
    if problems:
        raise ValueError("the other feature tables cannot be used: " + "; ".join(problems))
    merged = pd.concat([objects, *blocks], axis=1)
    return FeatureTables(merged, columns, missing, incomplete, identity)


def table_weights(columns: Mapping[str, Sequence[str]], explainable: int) -> dict[str, float]:
    """Each column's weight in the weighted combined space: ``sqrt(explainable / p)`` for a
    table of ``p`` columns (see the module); the explainable columns are not listed (1)."""
    if explainable < 1:
        raise ValueError("the combined space needs at least one explainable column")
    weights: dict[str, float] = {}
    for table in columns.values():
        weight = float(np.sqrt(explainable / len(table)))
        weights.update({column: weight for column in table})
    return weights


__all__ = [
    "COMBINED",
    "EXPLAINABLE",
    "LABEL_COLUMN",
    "FeatureTables",
    "attach_feature_tables",
    "observation_identity",
    "table_weights",
]
