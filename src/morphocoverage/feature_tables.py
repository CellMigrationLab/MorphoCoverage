"""Other feature tables for the same observations: validated, then attached as named blocks.

Feature tables come one per provenance (fiberfeatures, CellProfiler, DINOv2), each with a
name and **tagged** ``"explainable"`` or ``"non-explainable"`` (owner, 2026-10-06), any mix
allowed: ``{"fiberfeatures": (table, "explainable"), "DINOv2": (table, "non-explainable")}``.
A bare DataFrame is non-explainable, as before. The tag is read ignoring case, ``_`` or a
space for the hyphen (``"non_explainable"``); any other tag is refused.

- An **explainable** table's columns are explainable features, as the observation table's own
  numeric columns are (an implicit explainable table): declared in ``Setup.features``, each is
  tested one by one (the discovery, the heatmap, SuperPlots, "Did I image enough?").
- A **non-explainable** table, such as the deep-learning embeddings of a DINO model, is never
  tested feature by feature; with the explainable features it makes the maps, the
  populations, the copy checks and a fingerprint of its own, never novelty, which reads the
  explainable features only (owner, 2026-10-05 and 2026-10-06; ``workflows``,
  ``docs/common/data_format.md``).

A column name found in two tables is kept in the observation table and prefixed with its
table's name in every other table, ``"fiberfeatures: area"`` (``renamed``); a name found once
is kept (owner, 2026-10-06), so two embeddings with the same column names load together.

``attach_feature_tables`` validates every table against the observation table and returns
the observation table with every table's columns attached (``FeatureTables``). One row of a
table is one observation, identified as the observation table identifies it
(``observation_identity``): the FOV key columns the observation table holds (the analysis
groups and the field identifier, as ``validate_input_tables`` resolves them), then
``label_id`` and the within-field coordinates ``z``, ``z0``, ``z1``, ``c``, ``t`` when the
observation table has them. Values are matched as text, a whole number written as a decimal
read as that whole number, so ``1``, ``1.0`` and ``"1"`` are one label (owner, 2026-10-06:
a CSV column with one empty cell is read as decimals). A table must:

- hold every identity column and match every observation exactly once: no observation
  missing, no row matching none, no observation twice;
- hold numeric feature columns only (every column but the identity), none named like
  MorphoBricks metadata (``condition``, ``z``, ``label_id``, ...), and none without a value;
- hold no infinite value; missing values are kept and counted (``missing``), never
  imputed, and each analysis that reads the table leaves out the observations missing one
  of its values, and says how many;
- have a non-empty name of its own (``"explainable"``, ``"combined"`` and ``"observation
  table"`` name the explainable features, the weighted combined space and the observation
  table's own features beside explainable tables).

Every problem is collected and raised at once (``ValueError``), naming the table. A column
with one value everywhere is accepted and listed (``constant``): no analysis can use it.

``mfa_weights`` gives the weight of each column in the weighted combined space (owner,
2026-10-06): every table, robustly standardised, is divided by its own first singular value,
as in multiple factor analysis (MFA), so that no table's first axis outweighs another's,
whatever its number of columns and however correlated they are; the weights are scaled so
that the explainable features keep weight 1. ``table_shares`` gives each table's share of a
principal axis of that space. Without other tables nothing is weighted and every analysis is
as before.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
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
# The names of the explainable features, of the weighted combined space, and of the
# observation table's own declared features beside explainable tables (owner, 2026-10-06).
EXPLAINABLE = "explainable"
COMBINED = "combined"
OBSERVATION_TABLE = "observation table"
RESERVED_NAMES = (EXPLAINABLE, COMBINED, OBSERVATION_TABLE)
# The two tags of a table (owner, 2026-10-06); a bare DataFrame is non-explainable.
EXPLAINABLE_TAG = "explainable"
NON_EXPLAINABLE_TAG = "non-explainable"
TAGS = (EXPLAINABLE_TAG, NON_EXPLAINABLE_TAG)
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
    - ``columns``: each table's name and its feature columns (their final names), in the
      order given;
    - ``missing``: each table's missing values, and ``incomplete`` the observations holding
      at least one, which every analysis reading the table leaves out;
    - ``identity``: the columns that identify an observation;
    - ``constant``: each table's columns with one value everywhere, which no analysis uses;
    - ``tags``: each table's tag, ``"explainable"`` or ``"non-explainable"``;
    - ``renamed``: each table's columns prefixed with its name, original to final name.
    """

    objects: pd.DataFrame
    columns: dict[str, tuple[str, ...]]
    missing: dict[str, int]
    incomplete: dict[str, int]
    identity: tuple[str, ...]
    constant: dict[str, tuple[str, ...]] = field(default_factory=dict)
    tags: dict[str, str] = field(default_factory=dict)
    renamed: dict[str, dict[str, str]] = field(default_factory=dict)

    @property
    def explainable(self) -> dict[str, tuple[str, ...]]:
        """The explainable tables and their columns, in the order given."""
        return {n: c for n, c in self.columns.items() if self.tags.get(n) == EXPLAINABLE_TAG}

    @property
    def other(self) -> dict[str, tuple[str, ...]]:
        """The non-explainable tables and their columns, in the order given."""
        return {n: c for n, c in self.columns.items() if self.tags.get(n) != EXPLAINABLE_TAG}

    def lines(self, tags: bool | None = None) -> list[str]:
        """One sentence per table: its columns, how it matched, the columns renamed, its
        missing values and what they remove, its constant columns. Each table's tag is said
        with ``tags``, by default when an explainable table is among them."""
        key = ", ".join(self.identity)
        tagged = bool(self.explainable) if tags is None else tags
        lines = []
        for name, columns in self.columns.items():
            missing, rows = self.missing[name], self.incomplete[name]
            gaps = (
                f"{count(missing, 'missing value')} in {count(rows, 'row')}, which every "
                f"analysis reading {name} leaves out, counted there"
                if missing
                else "no missing value"
            )
            constant = self.constant.get(name, ())
            if constant:
                gaps += (
                    f"; {count(len(constant), 'column')} with one value everywhere "
                    f"({listing(list(constant), _SHOWN)}), which no analysis can use"
                )
            renamed = self.renamed.get(name, {})
            if renamed:  # said once, here (owner, 2026-10-06)
                also = "is also a column name" if len(renamed) == 1 else "are also column names"
                finals = [f'"{f}"' for f in renamed.values()]
                gaps += (
                    f"; {listing(list(renamed), _SHOWN)} {also} in another table, so named "
                    f"{listing(finals, _SHOWN)} here"
                )
            label = f"{name} ({self.tags.get(name, NON_EXPLAINABLE_TAG)})" if tagged else name
            lines.append(
                f"{label}: {count(len(columns), 'feature column')}, every observation matched "
                f"once by {key}; {gaps}."
            )
        return lines

    def summary(self) -> str:
        """The tables in sentences, one line each."""
        return "\n".join(self.lines())


def read_tag(tag: Any) -> str | None:
    """``tag`` as one of ``TAGS`` (case, ``_`` and spaces aside), or ``None`` when it is
    neither."""
    if not isinstance(tag, str):
        return None
    word = re.sub(r"[\s_]+", "-", tag.strip().casefold())
    return word if word in TAGS else None


def any_explainable(feature_tables: Mapping[str, Any] | None) -> bool:
    """Whether any of ``feature_tables`` is tagged explainable."""
    return any(_tagged(given)[1] == EXPLAINABLE_TAG for given in (feature_tables or {}).values())


def _tagged(given: Any) -> tuple[Any, str | None]:
    """A table and its tag from ``given``: a DataFrame (non-explainable) or ``(table, tag)``;
    the tag ``None`` when it is not one of ``TAGS``."""
    if isinstance(given, tuple) and len(given) == 2:
        return given[0], read_tag(given[1])
    return given, NON_EXPLAINABLE_TAG


def _final_names(
    objects: pd.DataFrame, features: Mapping[str, list[Any]]
) -> dict[str, dict[Any, str]]:
    """Each table's final column names: a name found in the observation table or in another
    table prefixed with the table's name, ``"fiberfeatures: area"``; any other kept."""
    seen: dict[Any, int] = {}
    for columns in features.values():
        for column in dict.fromkeys(columns):
            seen[column] = seen.get(column, 0) + 1
    own = set(objects.columns)
    return {
        name: {c: f"{name}: {c}" if c in own or seen[c] > 1 else c for c in columns}
        for name, columns in features.items()
    }


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


def identity_text(value: Any) -> str:
    """A value as the identity matches it: a whole number written as a decimal (``1.0``, as
    a CSV column with an empty cell is read) as the whole number, anything else as its text."""
    if isinstance(value, float | np.floating) and np.isfinite(value) and float(value).is_integer():
        return str(int(value))
    return str(value)


def _keys(table: pd.DataFrame, identity: Sequence[str]) -> pd.Series:
    """Each row's identity as one text, so ``1``, ``1.0`` and ``"1"`` match."""
    text = pd.DataFrame(
        {c: [identity_text(v) for v in table[c].tolist()] for c in identity}, index=table.index
    )
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
            f"{name!r} names the explainable features, the observation table or the combined "
            "space; "
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
    final: Mapping[Any, str],
    objects: pd.DataFrame,
    taken: Mapping[str, str],
) -> list[str]:
    """Whether the feature columns of ``table`` are numeric, finite, and, by their final
    names (``final``), its own."""
    problems = []
    if not columns:
        problems.append(f"{name!r} holds no feature column beside the identity columns")
    metadata = [c for c in columns if str(c) in _METADATA]
    if metadata:
        problems.append(
            f"{name!r} holds metadata columns {metadata}, which are never features; keep only "
            "the identity columns and the features"
        )
    clash = [final[c] for c in columns if final[c] in objects.columns]
    if clash:
        problems.append(
            f"{name!r} has columns named as columns of the observation table even with its "
            f"name before them: {listing(clash, _SHOWN)}; rename them, as each feature must "
            "have one name"
        )
    shared: dict[str, list[Any]] = {}
    for column in columns:
        if final[column] in taken:
            shared.setdefault(taken[final[column]], []).append(final[column])
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
    empty = [c for c, none in zip(numeric, np.isnan(values).all(axis=0), strict=True) if none]
    if empty and len(values):
        problems.append(
            f"{name!r} has columns without a single value: {listing(empty, _SHOWN)}; leave "
            "them out of the table"
        )
    return problems


def _constant(block: pd.DataFrame) -> tuple[str, ...]:
    """The columns of ``block`` with one value everywhere (missing values aside)."""
    values = block.to_numpy(dtype=float)
    if not len(values):
        return ()
    spread = np.nanmax(values, axis=0) - np.nanmin(values, axis=0)
    return tuple(c for c, s in zip(block.columns, spread, strict=True) if not s > 0)


def attach_feature_tables(
    objects: pd.DataFrame,
    fovs: pd.DataFrame | None,
    feature_tables: Mapping[str, pd.DataFrame | tuple[pd.DataFrame, str]],
    *,
    fov_col: str | None = None,
    group_cols: Sequence[str] | None = None,
) -> FeatureTables:
    """Validate ``feature_tables`` (a name for each table, its DataFrame or ``(DataFrame,
    tag)``) against the observation table and attach their columns to it (see the module).
    ``fov_col`` and ``group_cols`` are as for ``validate_input_tables``, which identify the
    fields."""
    if not isinstance(objects, pd.DataFrame):
        raise TypeError("objects must be a pandas DataFrame")
    if not isinstance(feature_tables, Mapping) or not feature_tables:
        raise TypeError(
            "feature_tables must map each table's name to its DataFrame, or to (DataFrame, "
            "'explainable' or 'non-explainable')"
        )
    problems: list[str] = []
    given: dict[Any, tuple[Any, str]] = {}
    for name, value in feature_tables.items():
        table, tag = _tagged(value)
        if tag is None:
            problems.append(
                f"the table {name!r} is tagged {value[1]!r}: tag it 'explainable' (each column "
                "read and tested one by one) or 'non-explainable' (an embedding such as DINO, "
                "read only as a whole)"
            )
        given[name] = (table, tag or NON_EXPLAINABLE_TAG)
    reader = Reader()
    objects = reader.table(objects, "objects")
    fovs = reader.table(fovs, "fovs")
    tables = {
        name: reader.table(table, f"feature table {name!r}") for name, (table, _) in given.items()
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
    seen: list[str] = []
    taken: dict[str, str] = {}
    usable: dict[str, tuple[pd.DataFrame, list[Any]]] = {}
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
        if not named:
            usable[label] = (table, [c for c in table.columns if c not in identity])
    finals = _final_names(objects, {name: features for name, (_, features) in usable.items()})
    columns: dict[str, tuple[str, ...]] = {}
    blocks = []
    missing: dict[str, int] = {}
    incomplete: dict[str, int] = {}
    constant: dict[str, tuple[str, ...]] = {}
    for label, (table, features) in usable.items():
        final = finals[label]
        found = _match_problems(label, table, identity, observed)
        found += _column_problems(label, table, features, final, objects, taken)
        taken.update({final[c]: label for c in features})
        problems += found
        if found:
            continue
        values = table[features].set_axis(_keys(table, identity).to_numpy(), axis=0)
        block = values.reindex(observed.to_numpy()).set_axis(objects.index, axis=0)
        block = block.set_axis([final[c] for c in features], axis=1)
        blocks.append(block)
        columns[label] = tuple(block.columns)
        missing[label] = int(block.isna().to_numpy().sum())
        incomplete[label] = int(block.isna().any(axis=1).sum())
        constant[label] = _constant(block)
    if problems:
        raise ValueError("the other feature tables cannot be used: " + "; ".join(problems))
    merged = pd.concat([objects, *blocks], axis=1)
    tags = {str(name): tag for name, (_, tag) in given.items()}
    renamed = {name: {str(c): f for c, f in finals[name].items() if f != c} for name in columns}
    renamed = {name: pairs for name, pairs in renamed.items() if pairs}
    return FeatureTables(merged, columns, missing, incomplete, identity, constant, tags, renamed)


def mfa_weights(
    z: np.ndarray, features: Sequence[str], blocks: Mapping[str, Sequence[str]]
) -> np.ndarray:
    """Each column's weight in the weighted combined space (see the module): ``z`` holds the
    robustly standardised ``features``, ``blocks`` each table's columns, the explainable
    features first. A table is weighted by the first block's first singular value over its
    own (both on the columns centred on their means), so the first block keeps weight 1; a
    column of no block, or a table none of whose columns is in ``features``, keeps 1."""
    weights = np.ones(len(features))
    position = {f: i for i, f in enumerate(features)}
    singular: dict[str, float] = {}
    for name, columns in blocks.items():
        index = [position[c] for c in columns if c in position]
        if index:
            part = z[:, index] - z[:, index].mean(axis=0)
            singular[name] = float(np.linalg.svd(part, compute_uv=False)[0]) if len(z) else 0.0
    reference = next((v for v in singular.values() if v > 0), 0.0)
    for name, columns in blocks.items():
        if singular.get(name, 0.0) > 0 and reference > 0:
            index = [position[c] for c in columns if c in position]
            weights[index] = reference / singular[name]
    return weights


def table_shares(
    loadings: np.ndarray, features: Sequence[str], blocks: Mapping[str, Sequence[str]]
) -> dict[str, list[float]]:
    """Each table's share of each principal axis: the sum of its columns' squared loadings
    (``loadings``: one row per feature, one column per axis, each axis of unit length)."""
    position = {f: i for i, f in enumerate(features)}
    shares = {}
    for name, columns in blocks.items():
        index = [position[c] for c in columns if c in position]
        shares[name] = [float(np.square(loadings[index, axis]).sum()) for axis in range(2)]
    return shares


__all__ = [
    "COMBINED",
    "EXPLAINABLE",
    "EXPLAINABLE_TAG",
    "LABEL_COLUMN",
    "NON_EXPLAINABLE_TAG",
    "OBSERVATION_TABLE",
    "TAGS",
    "FeatureTables",
    "any_explainable",
    "attach_feature_tables",
    "identity_text",
    "mfa_weights",
    "observation_identity",
    "read_tag",
    "table_shares",
]
