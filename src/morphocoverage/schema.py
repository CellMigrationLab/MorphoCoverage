"""Validation for the representation-agnostic MorphoCoverage input contract."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .metadata import (
    check_canonical_spellings,
    resolve_fov_col,
    resolve_group_cols,
    resolve_identity_cols,
    WITHIN_FOV_METADATA_COLUMNS,
)


@dataclass(frozen=True)
class ValidatedInput:
    """Validated FOV metadata and object-level phenotype features.

    ``fov_key`` lists the columns that jointly identify one acquired FOV
    (identity columns such as condition/repeat, then ``fov_col``). Within a
    pooled analysis ``fov_col`` alone need not be unique, so strategies that
    need FOV identity must use ``fov_key``.

    Strategies should read phenotype values only through ``feature_matrix()``.
    Other object columns are carried for traceability and must not be used as
    phenotype.
    """

    objects: pd.DataFrame
    fovs: pd.DataFrame
    feature_cols: tuple[str, ...]
    fov_col: str
    group_cols: tuple[str, ...]
    fov_key: tuple[str, ...]

    def feature_matrix(self) -> np.ndarray:
        """Return the explicitly selected phenotype features as a new float array."""
        return self.objects.loc[:, list(self.feature_cols)].to_numpy(dtype=float, copy=True)

    def feature_values(self, feature: str) -> np.ndarray:
        """Return one explicitly selected feature as a float array.

        Raises ``ValueError`` for any column not in ``feature_cols``, so a
        strategy configured with a metadata column (e.g. a stage coordinate)
        fails instead of treating it as phenotype.
        """
        if feature not in self.feature_cols:
            raise ValueError(
                f"feature {feature!r} is not one of the explicitly selected "
                f"feature columns {list(self.feature_cols)}"
            )
        return self.objects[feature].to_numpy(dtype=float, copy=True)


def fov_positions(data: ValidatedInput, fovs: pd.DataFrame) -> np.ndarray:
    """Position of each object's FOV within ``fovs`` (-1 if its FOV is not listed).

    Matching uses the full ``fov_key``, so pooled FOVs that share a name stay
    distinct.
    """
    key = list(data.fov_key)
    lookup = {fov: i for i, fov in enumerate(zip(*(fovs[c].tolist() for c in key)))}
    object_keys = zip(*(data.objects[c].tolist() for c in key))
    return np.fromiter(
        (lookup.get(fov, -1) for fov in object_keys), dtype=np.int64, count=len(data.objects)
    )


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
    intentionally; condition/repeat still remain part of FOV identity.

    When fov_col is omitted, position_id is preferred, followed by image_file
    and then fov_id. FOV means one independently selected spatial acquisition
    position. Z planes, channels and time points are nested observations within
    that position, not additional FOVs. Lowercase z/c/t columns are reserved
    acquisition metadata and cannot be selected as phenotype features.

    Supplying a complete FOV table preserves acquired positions containing zero
    detected objects.
    """
    if not isinstance(objects, pd.DataFrame):
        raise TypeError("objects must be a pandas DataFrame")
    if fovs is not None and not isinstance(fovs, pd.DataFrame):
        raise TypeError("fovs must be a pandas DataFrame when provided")
    if isinstance(feature_cols, str):
        raise TypeError("feature_cols must be a sequence of column names, not a string")

    explicit_columns = [*(group_cols or ()), *([fov_col] if fov_col is not None else [])]
    check_canonical_spellings(objects, fovs, explicit_columns)
    resolved_fov_col = resolve_fov_col(objects, fovs, fov_col)
    resolved_group_cols = resolve_group_cols(objects, fovs, group_cols)
    identity_cols = resolve_identity_cols(objects, fovs, resolved_group_cols, resolved_fov_col)
    fov_key = (*identity_cols, resolved_fov_col)
    feature_cols = tuple(feature_cols)

    if not feature_cols:
        raise ValueError("feature_cols must contain at least one explicitly selected feature")
    if len(set(feature_cols)) != len(feature_cols):
        raise ValueError("feature_cols contains duplicate column names")
    if resolved_fov_col in feature_cols:
        raise ValueError("fov_col cannot also be a phenotype feature")
    within_fov_features = set(feature_cols).intersection(WITHIN_FOV_METADATA_COLUMNS)
    if within_fov_features:
        raise ValueError(
            "within-FOV acquisition metadata cannot be phenotype features: "
            f"{sorted(within_fov_features)}. Preserve z/c/t as metadata while all "
            "planes/channels/time points from the same position share one FOV identity."
        )
    if resolved_fov_col in identity_cols:
        raise ValueError("fov_col cannot also be a group column")
    overlap = set(feature_cols).intersection(identity_cols)
    if overlap:
        raise ValueError(f"group columns cannot also be phenotype features: {sorted(overlap)}")

    _require_columns(objects, [*fov_key, *feature_cols], "objects")

    # When an explicit spatial position spans several source files (for example
    # one TIFF per z-plane/channel/timepoint), require within-position metadata
    # so those files are not an undocumented pseudo-FOV axis.
    if resolved_fov_col == "position_id" and "image_file" in objects.columns:
        file_counts = objects.groupby(list(fov_key), dropna=False)["image_file"].nunique()
        multi_file_positions = file_counts[file_counts > 1]
        if len(multi_file_positions):
            present_axes = [
                column for column in WITHIN_FOV_METADATA_COLUMNS if column in objects.columns
            ]
            if not present_axes:
                raise ValueError(
                    "a position_id maps to multiple image_file values, but no z/c/t "
                    "metadata is present. When one spatial position is split across "
                    "plane/channel/time files, document the relevant within-FOV axis."
                )
            ambiguous_keys = {
                key if isinstance(key, tuple) else (key,)
                for key in multi_file_positions.index.tolist()
            }
            key_values = list(objects.loc[:, list(fov_key)].itertuples(index=False, name=None))
            ambiguous_mask = np.fromiter(
                (key_value in ambiguous_keys for key_value in key_values),
                dtype=bool,
                count=len(objects),
            )
            if objects.loc[ambiguous_mask, present_axes].isna().all(axis=1).any():
                raise ValueError(
                    "rows from a multi-file spatial position are missing z/c/t metadata; "
                    "document the relevant within-FOV coordinate on every such row."
                )

    if objects[resolved_fov_col].isna().any():
        raise ValueError(
            f"objects[{resolved_fov_col!r}] contains missing FOV identifiers"
        )
    for column in identity_cols:
        if objects[column].isna().any():
            raise ValueError(f"objects[{column!r}] contains missing group values")

    for column in feature_cols:
        if not pd.api.types.is_numeric_dtype(objects[column]):
            raise TypeError(f"feature column {column!r} must be numeric")
        values = objects[column].to_numpy(dtype=float, copy=False)
        if not np.isfinite(values).all():
            raise ValueError(f"feature column {column!r} contains non-finite values")

    key = list(fov_key)
    if fovs is None:
        validated_fovs = objects.loc[:, key].drop_duplicates().reset_index(drop=True)
    else:
        _require_columns(fovs, key, "fovs")
        validated_fovs = fovs.copy()
        for column in key:
            if validated_fovs[column].isna().any():
                raise ValueError(f"fovs[{column!r}] contains missing values")
        if validated_fovs.duplicated(key).any():
            raise ValueError(
                f"fovs must contain exactly one row per acquired FOV (key: {key})"
            )

        fov_index = pd.MultiIndex.from_frame(validated_fovs.loc[:, key])
        object_fov_index = pd.MultiIndex.from_frame(objects.loc[:, key].drop_duplicates())
        missing_fovs = object_fov_index.difference(fov_index)
        if len(missing_fovs):
            raise ValueError(
                "objects contains FOV/group combinations that are absent from the "
                f"supplied fovs table, e.g. {list(missing_fovs[:3])}"
            )

    return ValidatedInput(
        objects=objects.copy(),
        fovs=validated_fovs,
        feature_cols=feature_cols,
        fov_col=resolved_fov_col,
        group_cols=resolved_group_cols,
        fov_key=fov_key,
    )
