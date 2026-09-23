"""Validation for the representation-agnostic MorphoCoverage input contract."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

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
    observation_unit: str = "object"

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
        return self.objects.loc[:, list(self.feature_cols)].to_numpy(
            dtype=float, na_value=np.nan, copy=True
        )

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
    key = list(data.fov_key)
    lookup = {fov: i for i, fov in enumerate(zip(*(fovs[c].tolist() for c in key)))}
    observation_keys = zip(*(data.objects[c].tolist() for c in key))
    return np.fromiter(
        (lookup.get(fov, -1) for fov in observation_keys), dtype=np.int64, count=len(data.objects)
    )


def _require_columns(table: pd.DataFrame, columns: Sequence[str], table_name: str) -> None:
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

def validate_input_tables(
    objects: pd.DataFrame,
    *,
    feature_cols: Sequence[str],
    fov_col: str | None = None,
    fovs: pd.DataFrame | None = None,
    group_cols: Sequence[str] | None = None,
    observation_unit: str = "object",
) -> ValidatedInput:
    """Validate linked observation/FOV tables using MorphoBricks conventions.

    ``objects`` is retained as the argument name for backwards compatibility,
    but rows may represent any declared phenotype observation unit. Feature
    columns are always explicit; numeric metadata is never inferred.

    NaN feature values are accepted and preserved so heterogeneous real-world
    feature tables can be loaded. MorphoCoverage never silently imputes or
    removes them; estimators/transforms must define their own missing-data
    behavior. Positive/negative infinity remains invalid.
    """
    if not isinstance(objects, pd.DataFrame):
        raise TypeError("objects must be a pandas DataFrame")
    if fovs is not None and not isinstance(fovs, pd.DataFrame):
        raise TypeError("fovs must be a pandas DataFrame when provided")
    if isinstance(feature_cols, str):
        raise TypeError("feature_cols must be a sequence of column names, not a string")
    if not isinstance(observation_unit, str) or not observation_unit.strip():
        raise ValueError("observation_unit must be a non-empty description such as 'cell' or 'slice'")
    observation_unit = observation_unit.strip()

    explicit_columns = [*(group_cols or ()), *([fov_col] if fov_col is not None else [])]
    check_canonical_spellings(objects, fovs, explicit_columns)
    resolved_fov_col = resolve_fov_col(objects, fovs, fov_col)
    metadata_to_attach = list(MORPHOBRICKS_GROUP_COLUMNS)
    if group_cols is not None:
        metadata_to_attach.extend(column for column in group_cols if column not in metadata_to_attach)
    objects = _attach_fov_metadata(objects, fovs, resolved_fov_col, metadata_to_attach)
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
    if resolved_fov_col in identity_cols:
        raise ValueError("fov_col cannot also be a group column")
    overlap = set(feature_cols).intersection(identity_cols)
    if overlap:
        raise ValueError(f"group columns cannot also be phenotype features: {sorted(overlap)}")

    _require_columns(objects, [*fov_key, *feature_cols], "objects")

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
        raise ValueError(f"objects[{resolved_fov_col!r}] contains missing FOV identifiers")
    for column in identity_cols:
        if objects[column].isna().any():
            raise ValueError(f"objects[{column!r}] contains missing group values")

    for column in feature_cols:
        if not pd.api.types.is_numeric_dtype(objects[column]):
            raise TypeError(f"feature column {column!r} must be numeric")
        values = objects[column].to_numpy(dtype=float, na_value=np.nan, copy=False)
        if np.isinf(values).any():
            raise ValueError(f"feature column {column!r} contains infinite values")

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
            raise ValueError(f"fovs must contain exactly one row per acquired FOV (key: {key})")

        fov_index = pd.MultiIndex.from_frame(validated_fovs.loc[:, key])
        observation_fov_index = pd.MultiIndex.from_frame(objects.loc[:, key].drop_duplicates())
        missing_fovs = observation_fov_index.difference(fov_index)
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
        observation_unit=observation_unit,
    )
