"""Validation for the representation-agnostic MorphoCoverage input contract."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .metadata import (
    MORPHOBRICKS_GROUP_COLUMNS,
    WITHIN_FOV_METADATA_COLUMNS,
    check_canonical_spellings,
    resolve_fov_col,
    resolve_group_cols,
    resolve_identity_cols,
)

# Units whose rows are individual detected objects, as the cell-number
# baselines (JCB, Marcotti) assume.
OBJECT_LIKE_UNITS = frozenset({"object", "cell", "nucleus"})

# Plural and spelling variants mapped to one canonical label, so that "cells"
# and "Cell" do not silently count as a different unit from "cell".
_UNIT_ALIASES = {
    "objects": "object",
    "cells": "cell",
    "nuclei": "nucleus",
    "nucleuses": "nucleus",
    "patches": "patch",
    "slices": "slice",
    "planes": "plane",
    "embeddings": "embedding",
    "fov_summaries": "fov_summary",
}


# Assumed when the caller does not say what a row is. Results record that the
# unit was assumed (``observation_unit_declared=False``), never silently.
DEFAULT_OBSERVATION_UNIT = "object"


def resolve_observation_unit(unit: str | None) -> tuple[str, bool]:
    """The canonical unit and whether the caller declared it (``None`` means assumed)."""
    if unit is None:
        return DEFAULT_OBSERVATION_UNIT, False
    return normalize_observation_unit(unit), True


def normalize_observation_unit(unit: object) -> str:
    """Canonical observation-unit label: lowercase, ``_`` for spaces/hyphens, singular.

    Any non-empty label is accepted, since rows may be of a kind this package
    does not know; known plurals are mapped to their singular.
    """
    if not isinstance(unit, str) or not unit.strip():
        raise ValueError(
            "observation_unit must be a non-empty description such as 'cell' or 'slice'"
        )
    label = re.sub(r"[\s-]+", "_", unit.strip().lower())
    return _UNIT_ALIASES.get(label, label)


@dataclass(frozen=True)
class ValidatedInput:
    """Validated FOV metadata and phenotype observations.

    Rows in ``objects`` are generic phenotype observations: cells, nuclei,
    patches, slices, embeddings, FOV summaries, or another declared unit.
    ``observation_unit`` records that meaning. The attribute remains named
    ``objects`` for backwards compatibility with the early public API.
    """

    objects: pd.DataFrame
    fovs: pd.DataFrame
    feature_cols: tuple[str, ...]
    fov_col: str
    group_cols: tuple[str, ...]
    fov_key: tuple[str, ...]
    observation_unit: str = DEFAULT_OBSERVATION_UNIT
    observation_unit_declared: bool = False
    # The feature matrix and FOV positions, computed once per input: selecting columns from a
    # DataFrame costs about a millisecond, and resampling analyses read the same
    # input thousands of times. Inputs are treated as immutable; the framework
    # hands every estimator call a fresh copy with an empty cache
    # (``evaluator._isolated_input``). Code that edits ``objects`` in place and
    # then reads the features again must call ``clear_cache`` in between.
    _feature_cache: dict = field(default_factory=dict, compare=False, repr=False)

    def clear_cache(self) -> None:
        """Forget the cached feature matrix and FOV positions, after editing tables in place."""
        self._feature_cache.clear()

    @property
    def observations(self) -> pd.DataFrame:
        """Generic alias for the backwards-compatible objects table."""
        return self.objects

    @property
    def n_observations(self) -> int:
        """Number of phenotype-observation rows."""
        return len(self.objects)

    def feature_matrix(self) -> np.ndarray:
        """Return the explicitly selected phenotype features as a new float array.

        NaN is preserved. The framework never silently imputes or drops missing
        phenotype measurements; individual estimators/transforms must state
        how they handle missingness. Positive/negative infinity is rejected at
        validation time.
        """
        cached = self._feature_cache.get("matrix")
        if cached is None:
            cached = self.objects.loc[:, list(self.feature_cols)].to_numpy(
                dtype=float, na_value=np.nan, copy=True
            )
            self._feature_cache["matrix"] = cached
        return cached.copy()

    def feature_values(self, feature: str) -> np.ndarray:
        """Return one explicitly selected feature as a new float array."""
        if feature not in self.feature_cols:
            raise ValueError(
                f"feature {feature!r} is not one of the explicitly selected "
                f"feature columns {list(self.feature_cols)}"
            )
        return self.objects[feature].to_numpy(dtype=float, na_value=np.nan, copy=True)


def fov_positions(data: ValidatedInput, fovs: pd.DataFrame) -> np.ndarray:
    """Position of each observation's FOV within fovs (-1 if absent)."""
    if fovs is data.fovs:
        # The input's own FOV table: looked up once and cached, like the features.
        cached = data._feature_cache.get("positions")
        if cached is None:
            cached = _fov_positions(data, fovs)
            data._feature_cache["positions"] = cached
        return cached.copy()
    return _fov_positions(data, fovs)


def _fov_positions(data: ValidatedInput, fovs: pd.DataFrame) -> np.ndarray:
    key = list(data.fov_key)
    lookup = {fov: i for i, fov in enumerate(zip(*(fovs[c].tolist() for c in key)))}
    observation_keys = zip(*(data.objects[c].tolist() for c in key))
    return np.fromiter(
        (lookup.get(fov, -1) for fov in observation_keys), dtype=np.int64, count=len(data.objects)
    )


def _require_columns(table: pd.DataFrame, columns: Sequence[str], table_name: str) -> None:
    """Raise naming every required column that ``table`` lacks."""
    missing = [column for column in columns if column not in table.columns]
    if missing:
        raise ValueError(f"{table_name} is missing required columns: {missing}")


def _attach_fov_metadata(
    objects: pd.DataFrame,
    fovs: pd.DataFrame | None,
    fov_col: str,
    columns: Sequence[str],
) -> pd.DataFrame:
    """Attach FOV-level metadata to observations when the join is unambiguous.

    This permits lean feature tables containing only the FOV identifier plus
    phenotype features. If an FOV identifier is reused across metadata values,
    callers must carry the relevant identity/group columns on observation rows
    instead of asking the framework to guess.
    """
    if fovs is None:
        return objects
    enriched = objects.copy()
    for column in columns:
        if column in enriched.columns or column not in fovs.columns:
            continue
        if fov_col not in enriched.columns or fov_col not in fovs.columns:
            continue
        mapping = fovs.loc[:, [fov_col, column]].drop_duplicates()
        if mapping.duplicated(fov_col).any():
            raise ValueError(
                f"cannot attach {column!r} from the FOV table because {fov_col!r} "
                "is reused across different metadata values; include the grouping "
                "column in the observation table or use a globally unique FOV identifier"
            )
        enriched = enriched.merge(mapping, on=fov_col, how="left", validate="many_to_one")
    return enriched


def _within_fov_axes(objects: pd.DataFrame, feature_cols: Sequence[str]) -> list[str]:
    """The z/c/t-style columns that describe acquisition within a position."""
    return [
        column
        for column in WITHIN_FOV_METADATA_COLUMNS
        if column in objects.columns and column not in feature_cols
    ]


def _check_within_fov_axes(
    objects: pd.DataFrame,
    fov_key: tuple[str, ...],
    feature_cols: Sequence[str],
    *,
    fov_col_was_chosen: bool,
) -> None:
    """Refuse FOV identities that would silently count planes, channels or frames as FOVs.

    - A position spread over several ``image_file`` values must say, on every
      row, which plane/channel/frame each file is (``z``/``c``/``t``), and
      those values must tell the files apart.
    - When the FOV column was resolved automatically to a file-level
      identifier and each FOV holds a single value of a ``z``/``c``/``t``
      column that varies across FOVs, each "FOV" looks like one plane of a
      position. The caller must supply ``position_id`` or name ``fov_col``.
    """
    key = list(fov_key)
    fov_col = key[-1]
    axes = _within_fov_axes(objects, feature_cols)
    if "image_file" in objects.columns and fov_col != "image_file":
        files = objects.groupby(key, dropna=False)["image_file"].transform("nunique")
        split = objects.loc[files > 1]
        if len(split):
            if not axes:
                raise ValueError(
                    f"a {fov_col!r} maps to multiple image_file values, but no z/c/t "
                    "metadata says which plane, channel or time point each file is"
                )
            if split[axes].isna().all(axis=1).any():
                raise ValueError(
                    "rows from a position spread over several files are missing z/c/t "
                    "metadata; document the within-position coordinate on every row"
                )
            per_plane = split.groupby([*key, *axes], dropna=False)["image_file"].nunique()
            if (per_plane > 1).any():
                example = per_plane[per_plane > 1].index[0]
                raise ValueError(
                    f"z/c/t values {axes} do not tell the files of one position apart "
                    f"(e.g. {example}); each file must be a distinct plane/channel/frame"
                )
    if fov_col_was_chosen and fov_col in ("image_file", "fov_id") and axes:
        rows_per_fov = objects.groupby(key, dropna=False)[fov_col].transform("size")
        if (rows_per_fov > 1).any():
            for axis in axes:
                per_fov = objects.groupby(key, dropna=False)[axis].nunique(dropna=False)
                if (per_fov == 1).all() and objects[axis].nunique(dropna=False) > 1:
                    raise ValueError(
                        f"every {fov_col!r} holds a single {axis!r} value that varies across "
                        f"{fov_col!r} values, so each may be one plane/channel/frame of a "
                        "position rather than a spatial FOV. Supply 'position_id', or pass "
                        f"fov_col={fov_col!r} explicitly if each really is one position."
                    )


def _check_column_roles(
    feature_cols: tuple[str, ...], identity_cols: tuple[str, ...], fov_col: str
) -> None:
    """Each column plays one role: phenotype feature, group/identity, or FOV identifier."""
    if not feature_cols:
        raise ValueError("feature_cols must contain at least one explicitly selected feature")
    if len(set(feature_cols)) != len(feature_cols):
        raise ValueError("feature_cols contains duplicate column names")
    if fov_col in feature_cols:
        raise ValueError("fov_col cannot also be a phenotype feature")
    if fov_col in identity_cols:
        raise ValueError("fov_col cannot also be a group column")
    overlap = set(feature_cols).intersection(identity_cols)
    if overlap:
        raise ValueError(f"group columns cannot also be phenotype features: {sorted(overlap)}")


def _check_observation_values(
    objects: pd.DataFrame, fov_key: tuple[str, ...], feature_cols: tuple[str, ...]
) -> None:
    """Identity columns are complete; features are numeric and never infinite (NaN is allowed)."""
    for column in fov_key:
        if objects[column].isna().any():
            kind = "FOV identifiers" if column == fov_key[-1] else "group values"
            raise ValueError(f"objects[{column!r}] contains missing {kind}")
    # Column by column, the first problem wins, as a per-column loop would find
    # it; but the numeric columns are checked for infinity in one array.
    dtypes = objects.dtypes
    numeric = [pd.api.types.is_numeric_dtype(dtypes[column]) for column in feature_cols]
    first_bad = numeric.index(False) if False in numeric else len(feature_cols)
    checked = list(feature_cols[:first_bad])
    if checked:
        values = objects.loc[:, checked].to_numpy(dtype=float, na_value=np.nan)
        infinite = np.isinf(values).any(axis=0)
        if infinite.any():
            column = checked[int(np.argmax(infinite))]
            raise ValueError(f"feature column {column!r} contains infinite values")
    if first_bad < len(feature_cols):
        raise TypeError(f"feature column {feature_cols[first_bad]!r} must be numeric")


def _validated_fov_table(
    objects: pd.DataFrame, fovs: pd.DataFrame | None, fov_key: tuple[str, ...]
) -> pd.DataFrame:
    """The acquisition record: the supplied FOV table, checked, or the FOVs seen in objects.

    Without an FOV table, acquired FOVs with no observations are unknown; with
    one, every FOV of an observation must be listed exactly once.
    """
    key = list(fov_key)
    if fovs is None:
        return objects.loc[:, key].drop_duplicates().reset_index(drop=True)
    _require_columns(fovs, key, "fovs")
    for column in key:
        if fovs[column].isna().any():
            raise ValueError(f"fovs[{column!r}] contains missing values")
    if fovs.duplicated(key).any():
        raise ValueError(f"fovs must contain exactly one row per acquired FOV (key: {key})")
    # Key tuples as Python sets: several times faster than MultiIndex set
    # operations at these sizes, and identity columns contain no missing values.
    listed = set(zip(*(fovs[c].tolist() for c in key)))
    observed = set(zip(*(objects[c].tolist() for c in key)))
    missing = observed - listed
    if missing:
        try:
            examples = sorted(missing)[:3]
        except TypeError:
            examples = list(missing)[:3]
        raise ValueError(
            "objects contains FOV/group combinations that are absent from the "
            f"supplied fovs table, e.g. {examples}"
        )
    return fovs.copy()


def validate_input_tables(
    objects: pd.DataFrame,
    *,
    feature_cols: Sequence[str],
    fov_col: str | None = None,
    fovs: pd.DataFrame | None = None,
    group_cols: Sequence[str] | None = None,
    observation_unit: str | None = None,
) -> ValidatedInput:
    """Validate linked observation/FOV tables using MorphoBricks conventions.

    ``objects`` is retained as the argument name for backwards compatibility,
    but rows may represent any declared phenotype observation unit. Feature
    columns are always explicit; numeric metadata is never inferred.

    NaN feature values are accepted and preserved so heterogeneous real-world
    feature tables can be loaded. MorphoCoverage never silently imputes or
    removes them; estimators/transforms must define their own missing-data
    behavior. Positive/negative infinity remains invalid.

    ``observation_unit`` says what one row is (``"cell"``, ``"patch"``, ...).
    When omitted, rows are assumed to be objects and the result records
    ``observation_unit_declared=False``.
    """
    if not isinstance(objects, pd.DataFrame):
        raise TypeError("objects must be a pandas DataFrame")
    if fovs is not None and not isinstance(fovs, pd.DataFrame):
        raise TypeError("fovs must be a pandas DataFrame when provided")
    if isinstance(feature_cols, str):
        raise TypeError("feature_cols must be a sequence of column names, not a string")
    observation_unit, unit_declared = resolve_observation_unit(observation_unit)

    explicit_columns = [*(group_cols or ()), *([fov_col] if fov_col is not None else [])]
    check_canonical_spellings(objects, fovs, explicit_columns)
    resolved_fov_col = resolve_fov_col(objects, fovs, fov_col)
    metadata_to_attach = list(MORPHOBRICKS_GROUP_COLUMNS)
    if group_cols is not None:
        metadata_to_attach.extend(
            column for column in group_cols if column not in metadata_to_attach
        )
    objects = _attach_fov_metadata(objects, fovs, resolved_fov_col, metadata_to_attach)
    resolved_group_cols = resolve_group_cols(objects, fovs, group_cols)
    identity_cols = resolve_identity_cols(objects, fovs, resolved_group_cols, resolved_fov_col)
    fov_key = (*identity_cols, resolved_fov_col)
    feature_cols = tuple(feature_cols)

    _check_column_roles(feature_cols, identity_cols, resolved_fov_col)
    _require_columns(objects, [*fov_key, *feature_cols], "objects")
    _check_within_fov_axes(objects, fov_key, feature_cols, fov_col_was_chosen=fov_col is None)
    _check_observation_values(objects, fov_key, feature_cols)
    validated_fovs = _validated_fov_table(objects, fovs, fov_key)

    return ValidatedInput(
        objects=objects.copy(),
        fovs=validated_fovs,
        feature_cols=feature_cols,
        fov_col=resolved_fov_col,
        group_cols=resolved_group_cols,
        fov_key=fov_key,
        observation_unit=observation_unit,
        observation_unit_declared=unit_declared,
    )


def fov_table_from_observations(
    objects: pd.DataFrame,
    fov_key: Sequence[str],
    *,
    carry: Sequence[str] = (),
) -> pd.DataFrame:
    """A FOV table reconstructed from an observation table, when no acquisition record exists.

    Published feature tables (for example CellProfiler exports) often come
    without a table of acquired FOVs. This builds one row per distinct
    ``fov_key`` (for example ``["condition", "repeat", "image_file"]``) and
    copies ``carry`` columns that are constant within each FOV (stage
    coordinates, area, acquisition order). It warns every time, because FOVs
    that were acquired but hold no observations are missing from such a table:
    acquisition effort is undercounted and FOVs with nothing in them never
    count, which makes every sampling answer optimistic. Provide the real FOV
    table whenever it exists.
    """
    import warnings

    key = list(fov_key)
    columns = key + [c for c in carry if c not in key]
    missing = [c for c in columns if c not in objects.columns]
    if missing:
        raise ValueError(f"columns missing from the observation table: {missing}")
    if not key:
        raise ValueError("fov_key must name at least one column")
    grouped = objects.groupby(key, sort=False, dropna=False)
    for column in carry:
        varying = grouped[column].nunique(dropna=False) > 1
        if varying.any():
            raise ValueError(
                f"carry column {column!r} varies within FOV {varying.index[varying][0]!r}; "
                "only FOV-level columns can be carried"
            )
    table = objects.loc[:, columns].drop_duplicates(subset=key).reset_index(drop=True)
    warnings.warn(
        f"FOV table reconstructed from observations ({len(table)} FOVs): acquired FOVs "
        "without observations are missing, so effort is undercounted and sampling "
        "answers are optimistic. Provide the acquisition's FOV table if it exists.",
        UserWarning,
        stacklevel=2,
    )
    return table
