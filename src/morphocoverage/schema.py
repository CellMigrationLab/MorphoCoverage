"""Validation for the representation-agnostic MorphoCoverage input contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from .metadata import resolve_fov_col, resolve_group_cols


@dataclass(frozen=True)
class ValidatedInput:
    """Validated FOV metadata and object-level phenotype features."""

    objects: pd.DataFrame
    fovs: pd.DataFrame
    feature_cols: tuple[str, ...]
    fov_col: str
    group_cols: tuple[str, ...]


def _require_columns(table: pd.DataFrame, columns: Sequence[str], table_name: str) -> None:
    missing = [column for column in columns if column not in table.columns]
    if missing:
        raise ValueError(f"{table_name} is missing required columns: {missing}")


def validate_input_tables(
    objects: pd.DataFrame,
    *,
    feature_cols: Sequence[str],
    fov_col: str | None = None,
    fovs: pd.DataFrame | None = None,
    group_cols: Sequence[str] | None = None,
) -> ValidatedInput:
    """Validate linked object/FOV tables using MorphoBricks metadata conventions.

    Feature columns are always explicit; numeric metadata is never inferred as
    phenotype. When group_cols is None, canonical MorphoBricks condition/repeat
    columns are used automatically when available. Pass group_cols=[] to pool
    intentionally.

    When fov_col is omitted, image_file is preferred, followed by fov_id.
    Supplying a complete FOV table preserves acquired fields containing zero
    detected objects.
    """
    if not isinstance(objects, pd.DataFrame):
        raise TypeError("objects must be a pandas DataFrame")
    if fovs is not None and not isinstance(fovs, pd.DataFrame):
        raise TypeError("fovs must be a pandas DataFrame when provided")

    resolved_fov_col = resolve_fov_col(objects, fovs, fov_col)
    resolved_group_cols = resolve_group_cols(objects, fovs, group_cols)
    feature_cols = tuple(feature_cols)

    if not feature_cols:
        raise ValueError("feature_cols must contain at least one explicitly selected feature")
    if len(set(feature_cols)) != len(feature_cols):
        raise ValueError("feature_cols contains duplicate column names")
    if resolved_fov_col in feature_cols:
        raise ValueError("fov_col cannot also be a phenotype feature")
    overlap = set(feature_cols).intersection(resolved_group_cols)
    if overlap:
        raise ValueError(f"group columns cannot also be phenotype features: {sorted(overlap)}")

    _require_columns(
        objects,
        [resolved_fov_col, *resolved_group_cols, *feature_cols],
        "objects",
    )

    if objects[resolved_fov_col].isna().any():
        raise ValueError(
            f"objects[{resolved_fov_col!r}] contains missing FOV identifiers"
        )
    for column in resolved_group_cols:
        if objects[column].isna().any():
            raise ValueError(f"objects[{column!r}] contains missing group values")

    for column in feature_cols:
        if not pd.api.types.is_numeric_dtype(objects[column]):
            raise TypeError(f"feature column {column!r} must be numeric")
        values = objects[column].to_numpy(dtype=float, copy=False)
        if not np.isfinite(values).all():
            raise ValueError(f"feature column {column!r} contains non-finite values")

    fov_key = [*resolved_group_cols, resolved_fov_col]
    if fovs is None:
        validated_fovs = objects.loc[:, fov_key].drop_duplicates().reset_index(drop=True)
    else:
        _require_columns(fovs, fov_key, "fovs")
        validated_fovs = fovs.copy()
        for column in fov_key:
            if validated_fovs[column].isna().any():
                raise ValueError(f"fovs[{column!r}] contains missing values")
        if validated_fovs.duplicated(fov_key).any():
            raise ValueError(
                "fovs must contain exactly one row per acquired FOV within each analysis group"
            )

        fov_index = pd.MultiIndex.from_frame(validated_fovs.loc[:, fov_key])
        object_fov_index = pd.MultiIndex.from_frame(
            objects.loc[:, fov_key].drop_duplicates()
        )
        missing_fovs = object_fov_index.difference(fov_index)
        if len(missing_fovs):
            raise ValueError(
                "objects contains FOV/group combinations that are absent from the supplied fovs table"
            )

    return ValidatedInput(
        objects=objects.copy(),
        fovs=validated_fovs,
        feature_cols=feature_cols,
        fov_col=resolved_fov_col,
        group_cols=resolved_group_cols,
    )
