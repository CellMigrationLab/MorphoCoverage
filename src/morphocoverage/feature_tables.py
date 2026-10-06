"""Other feature tables for the same observations: validated, then attached as named blocks.

The observation table holds the **explainable** features, each tested one by one (the
discovery, the heatmap, SuperPlots, "Did I image enough?" per feature). Features that are
not read one by one, such as the deep-learning embeddings of a DINO model, come as **other
feature tables**, each with a name ("DINOv2", "DINOv3"), so that several models can be
loaded side by side and compared (owner, 2026-10-05). They are never tested feature by
feature; with the explainable features they make the maps, the populations, the copy checks
and a fingerprint of their own, never novelty, which reads the explainable features only
(owner, 2026-10-06; ``workflows``, ``docs/common/data_format.md``).

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
- use column names found neither in the observation table nor in another table;
- have a non-empty name of its own (``"explainable"`` and ``"combined"`` name the
  observation table's features and the weighted combined space).

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
      at least one, which every analysis reading the table leaves out;
    - ``identity``: the columns that identify an observation;
    - ``constant``: each table's columns with one value everywhere, which no analysis uses.
    """

    objects: pd.DataFrame
    columns: dict[str, tuple[str, ...]]
    missing: dict[str, int]
    incomplete: dict[str, int]
    identity: tuple[str, ...]
    constant: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def lines(self) -> list[str]:
        """One sentence per table: its columns, how it matched, its missing values and what
        they remove, its constant columns."""
        key = ", ".join(self.identity)
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
    constant: dict[str, tuple[str, ...]] = {}
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
        constant[label] = _constant(block)
    if problems:
        raise ValueError("the other feature tables cannot be used: " + "; ".join(problems))
    merged = pd.concat([objects, *blocks], axis=1)
    return FeatureTables(merged, columns, missing, incomplete, identity, constant)


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
    "LABEL_COLUMN",
    "FeatureTables",
    "attach_feature_tables",
    "identity_text",
    "mfa_weights",
    "observation_identity",
    "table_shares",
]
