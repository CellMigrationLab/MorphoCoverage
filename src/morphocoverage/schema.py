"""Validation for the representation-agnostic MorphoCoverage input contract.

Without a FOV table the validator warns with ``MissingFovTableWarning`` (a
``UserWarning``), once per calling line of the user's code.

Shared-private: ``_validate_input_tables`` (the validator itself, which
``data_check.validate_input_tables`` wraps), ``_within_fov_axes`` and
``_check_within_fov_axes`` (the plane rule, which ``online`` applies to every
FOV so far).
"""

from __future__ import annotations

import warnings
from collections.abc import Hashable, Sequence
from dataclasses import dataclass, field
from dataclasses import fields as fields_of
from typing import Any

import numpy as np
import pandas as pd

from ._former_repeat import Reader, caller_stacklevel
from ._identity import canonical_order
from ._input_invariants import check_input, trusted_input
from ._input_traps import warn_invalid_rows
from ._observation_units import (  # noqa: F401  (re-exported: the observation unit)
    _UNIT_ALIASES,
    DEFAULT_OBSERVATION_UNIT,
    DEFAULT_UNIT_WORD,
    OBJECT_LIKE_UNITS,
    normalize_observation_unit,
    observation_words,
    resolve_observation_unit,
    unit_plural,
    unit_words,
)
from ._plane_rules import (  # noqa: F401  (re-exported: the plane rule)
    PLANE_FEATURE_ADVICE,
    PLANE_LEVEL_UNITS,
    _check_planes_as_fields,
    _check_planes_within_fields,
    _check_positions_within_fields,
    _check_within_fov_axes,
    _within_fov_axes,
    plane_features,
)
from ._wording import count
from .metadata import (
    CONDITION_COLUMN,
    MORPHOBRICKS_GROUP_COLUMNS,
    REPEAT_COLUMN,
    TECHNICAL_REPEAT_COLUMN,
    _tables,
    attach_mapping,
    check_canonical_spellings,
    column_names,
    drop_empty_optional_columns,
    empty_values,
    resolve_fov_col,
    resolve_group_cols,
    resolve_identity_cols,
    unambiguous_index,
)


@dataclass(frozen=True)
class ValidatedInput:
    """Validated FOV metadata and phenotype observations.

    Rows in ``objects`` are generic phenotype observations: cells, nuclei,
    patches, slices, embeddings, FOV summaries, or another declared unit.
    ``observation_unit`` records that meaning. The attribute remains named
    ``objects`` for backwards compatibility with the early public API.

    ``fovs_from_observations`` is ``True`` when no FOV table was supplied and
    ``fovs`` lists only the FOVs seen in the observations: acquired FOVs
    without observations are then missing, so effort is undercounted.
    """

    objects: pd.DataFrame
    fovs: pd.DataFrame
    feature_cols: tuple[str, ...]
    fov_col: str
    group_cols: tuple[str, ...]
    fov_key: tuple[str, ...]
    observation_unit: str = DEFAULT_OBSERVATION_UNIT
    observation_unit_declared: bool = False
    fovs_from_observations: bool = False
    # The feature matrix and FOV positions, computed once per input (resampling reads them
    # thousands of times). Inputs are immutable: the framework hands every estimator call a
    # fresh copy with an empty cache; code editing ``objects`` in place calls ``clear_cache``.
    # Never passed in, so ``dataclasses.replace`` gives every copy an empty cache.
    _feature_cache: dict = field(default_factory=dict, init=False, compare=False, repr=False)

    def __post_init__(self) -> None:
        check_input(self)  # a hand-built input cannot masquerade as validated (issue 117)

    def with_tables(self, **changes: Any) -> ValidatedInput:
        """A copy of this input with other tables or feature columns (``objects``, ``fovs``,
        ``feature_cols``), trusted: the framework's subsets of an input that was validated."""
        fields = {f.name: getattr(self, f.name) for f in fields_of(self) if f.init}
        return trusted_input(type(self), **{**fields, **changes})

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


def one_per_field(data: ValidatedInput) -> bool:
    """Whether every field of ``data`` with observations holds exactly one."""
    positions = fov_positions(data, data.fovs)
    counts = np.bincount(positions[positions >= 0])
    return bool(len(counts)) and int(counts.max()) == 1


def canonical_fields(data: ValidatedInput) -> np.ndarray:
    """Positions of ``data.fovs`` in a canonical order: by their typed FOV key.

    Resampling draws index the FOVs in this order, so the draws depend on
    which FOVs were acquired, never on the FOV table's row order (row order
    carries no information: permuting it moved a finished acquisition's
    descriptor spread from 24.3 to 27.9 around a tolerance of 25, audit 4,
    P4-11). The key is the value's text and its type (``_identity``), so a
    categorical column's category order cannot change the order, and integer
    ``1`` and text ``"1"`` are two FOVs, not a tie broken by the row order.
    """
    return canonical_order(data.fovs, list(data.fov_key))


def technical_repeat_columns(fovs: pd.DataFrame) -> list[str] | None:
    """The FOV-table columns naming a field's technical repeat with its condition and
    repeat, or None without a technical-repeat column."""
    if TECHNICAL_REPEAT_COLUMN not in fovs.columns:
        return None
    return [c for c in (CONDITION_COLUMN, REPEAT_COLUMN, TECHNICAL_REPEAT_COLUMN) if c in fovs]


def fov_values(data: ValidatedInput, values: np.ndarray) -> list[np.ndarray]:
    """``values`` split by acquired FOV, one array per FOV row (empty for zero-observation FOVs).

    ``values`` is aligned with ``data.objects``; missing values (NaN) are dropped.
    """
    positions = fov_positions(data, data.fovs)
    keep = ~np.isnan(values)
    positions, values = positions[keep], values[keep]
    order = np.argsort(positions, kind="stable")
    counts = np.bincount(positions, minlength=len(data.fovs))
    return np.split(values[order], np.cumsum(counts)[:-1])


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
        # Through (repeat, field) when a field name restarts in each repeat (attach_mapping).
        key, mapping = attach_mapping(enriched, fovs, column, columns, fov_col)
        listed = pd.MultiIndex.from_frame(fovs[key]) if len(key) > 1 else fovs[fov_col]
        present = pd.MultiIndex.from_frame(enriched[key]) if len(key) > 1 else enriched[fov_col]
        unknown = ~np.asarray(present.isin(listed))
        if unknown.any():
            # Otherwise the join leaves NaN group values and the error names those.
            examples = sorted(map(str, enriched.loc[unknown, fov_col].unique()))[:5]
            raise ValueError(
                f"observations of FOVs absent from the supplied fovs table, e.g. "
                f"{fov_col} {examples}; every observation's FOV must be listed in fovs"
            )
        if mapping.duplicated(key).any():
            raise ValueError(
                f"cannot attach {column!r} from the FOV table because {fov_col!r} "
                "is reused across different metadata values; include the grouping "
                "column in the observation table or use a globally unique FOV identifier"
            )
        # A left merge keeps the observations' order but not their index labels.
        enriched = enriched.merge(mapping, on=key, how="left", validate="many_to_one").set_axis(
            enriched.index
        )
    return enriched


def feature_dtype_problem(values: pd.Series) -> str | None:
    """Why a feature column cannot be read as real numbers (``"complex"``, ``"not numeric"``).

    ``None`` when it can: a numeric column, or a column without any value (a
    header-only CSV is read as text columns, yet holds no text).
    """
    if pd.api.types.is_complex_dtype(values.dtype):
        return "complex"
    if pd.api.types.is_numeric_dtype(values.dtype):
        return None
    if not isinstance(values.dtype, pd.CategoricalDtype) and bool(values.isna().all()):
        return None
    return "not numeric"


def _check_column_roles(
    feature_cols: tuple[str, ...], identity_cols: tuple[str, ...], fov_col: str
) -> None:
    """Each column plays one role: phenotype feature, group/identity, or FOV identifier.

    A within-field coordinate (``z``, ``c``, ``t``, ``z0``, ``z1``) is never a
    feature (``PLANE_FEATURE_ADVICE``).
    """
    if not feature_cols:
        raise ValueError("feature_cols must contain at least one explicitly selected feature")
    if len(set(feature_cols)) != len(feature_cols):
        raise ValueError("feature_cols contains duplicate column names")
    planes = plane_features(feature_cols)
    if planes:
        raise ValueError(
            f"feature_cols names the within-field coordinate {planes[0]!r}: {PLANE_FEATURE_ADVICE}"
        )
    if fov_col in feature_cols:
        raise ValueError("fov_col cannot also be a phenotype feature")
    if fov_col in identity_cols:
        raise ValueError("fov_col cannot also be a group column")
    overlap = set(feature_cols).intersection(identity_cols)
    if overlap:
        raise ValueError(f"group columns cannot also be phenotype features: {sorted(overlap)}")


def unhashable_values(values: pd.Series) -> pd.Series:
    """Which entries cannot identify a field or group (lists, arrays, dicts)."""
    if values.dtype != object:
        return pd.Series(False, index=values.index)
    return values.map(lambda value: not isinstance(value, Hashable)).astype(bool)


def _check_hashable_identity(
    objects: pd.DataFrame, fovs: pd.DataFrame | None, columns: Sequence[str]
) -> None:
    """Refuse identity columns holding lists or arrays: they cannot be matched or grouped."""
    for name, table in _tables(objects, fovs):
        for column in dict.fromkeys(columns):
            if column in table.columns and unhashable_values(table[column]).any():
                raise TypeError(
                    f"{name}[{column!r}] holds values such as lists, which cannot identify a "
                    "field or group; store one text or number per row"
                )


def _check_identity_values(objects: pd.DataFrame, fov_key: tuple[str, ...]) -> None:
    """Identity columns are complete: no missing or blank value."""
    for column in fov_key:
        if empty_values(objects[column]).any():
            kind = "FOV identifiers" if column == fov_key[-1] else "group values"
            raise ValueError(f"objects[{column!r}] contains missing {kind}")


def _check_feature_values(objects: pd.DataFrame, feature_cols: tuple[str, ...]) -> None:
    """Features are real numbers (``feature_dtype_problem``) and never infinite (NaN is
    allowed)."""
    # Column by column, the first problem wins, as a per-column loop would find
    # it; but the numeric columns are checked for infinity in one array.
    problems = [feature_dtype_problem(objects[column]) for column in feature_cols]
    bad = [i for i, problem in enumerate(problems) if problem is not None]
    first_bad = bad[0] if bad else len(feature_cols)
    checked = list(feature_cols[:first_bad])
    if checked:
        values = objects.loc[:, checked].to_numpy(dtype=float, na_value=np.nan)
        infinite = np.isinf(values).any(axis=0)
        if infinite.any():
            column = checked[int(np.argmax(infinite))]
            raise ValueError(f"feature column {column!r} contains infinite values")
    if first_bad < len(feature_cols):
        column = feature_cols[first_bad]
        if problems[first_bad] == "complex":
            raise TypeError(f"feature column {column!r} must be real numbers, not complex")
        raise TypeError(f"feature column {column!r} must be numeric")


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
        if empty_values(fovs[column]).any():
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
        # fallback: examples of mixed types: listed unsorted, the error is still raised
        except TypeError:
            examples = list(missing)[:3]
        raise ValueError(
            "objects contains FOV/group combinations that are absent from the "
            f"supplied fovs table, e.g. {examples}"
        )
    return fovs.copy()


def _validate_input_tables(
    objects: pd.DataFrame,
    *,
    feature_cols: Sequence[str],
    fov_col: str | None = None,
    fovs: pd.DataFrame | None = None,
    group_cols: Sequence[str] | None = None,
    observation_unit: str | None = None,
) -> ValidatedInput:
    """Validate linked observation/FOV tables using MorphoBricks conventions (first error raises).

    The rules are those listed by the public ``validate_input_tables``, which
    runs this and attaches the ``check_tables`` report to the error. Without
    ``fovs`` the FOV table is rebuilt from the observations, with a warning
    (``MissingFovTableWarning``) and ``fovs_from_observations=True``.
    """
    objects, fovs, feature_cols, group_cols, fov_col = _read_arguments(
        objects, fovs, feature_cols, group_cols, fov_col
    )
    observation_unit, unit_declared = resolve_observation_unit(observation_unit)
    objects = unambiguous_index(objects)
    fovs = None if fovs is None else unambiguous_index(fovs)

    explicit_columns = [*(group_cols or ()), *([fov_col] if fov_col is not None else [])]
    objects, fovs = drop_empty_optional_columns(objects, fovs, explicit_columns)
    check_canonical_spellings(objects, fovs, explicit_columns)
    resolved_fov_col = resolve_fov_col(objects, fovs, fov_col)
    objects, resolved_group_cols, fov_key = _field_identity(
        objects, fovs, group_cols, resolved_fov_col
    )
    _check_rows(objects, fovs, feature_cols, fov_key, fov_col is None, observation_unit)
    validated_fovs = _validated_fov_table(objects, fovs, fov_key)
    if fovs is None:
        warnings.warn(
            f"no FOV table was given, so it is rebuilt from the observations "
            f"({count(len(validated_fovs), 'FOV')}): acquired FOVs without observations are "
            "missing, so effort is undercounted and sampling answers are optimistic. Pass "
            "fovs=, the table of every acquired FOV.",
            MissingFovTableWarning,
            stacklevel=caller_stacklevel(),
        )

    return trusted_input(
        ValidatedInput,
        objects=objects.copy(),
        fovs=validated_fovs,
        feature_cols=feature_cols,
        fov_col=resolved_fov_col,
        group_cols=resolved_group_cols,
        fov_key=fov_key,
        observation_unit=observation_unit,
        observation_unit_declared=unit_declared,
        fovs_from_observations=fovs is None,
    )


def _read_arguments(
    objects: pd.DataFrame,
    fovs: pd.DataFrame | None,
    feature_cols: Sequence[str],
    group_cols: Sequence[str] | None,
    fov_col: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame | None, Any, Any, str | None]:
    """The tables checked and read with the former name ``repeat`` as the new one, the column
    names read once; ``TypeError``/``ValueError`` for tables or names that cannot be read."""
    if not isinstance(objects, pd.DataFrame):
        raise TypeError("objects must be a pandas DataFrame")
    if fovs is not None and not isinstance(fovs, pd.DataFrame):
        raise TypeError("fovs must be a pandas DataFrame when provided")
    if feature_cols is None:
        raise TypeError("feature_cols is required: name the phenotype columns explicitly")
    # The former name of the biological repeat, read as the new one (``_former_repeat``).
    former = Reader()
    objects, fovs = former.table(objects, "objects"), former.table(fovs, "fovs")
    group_cols = former.columns(group_cols, "group_cols")
    fov_col = former.column(fov_col, "fov_col")
    former.warn()
    feature_cols = column_names(feature_cols, "feature_cols")
    group_cols = column_names(group_cols, "group_cols")
    for table_name, table in (("objects", objects), ("fovs", fovs)):
        repeated = [] if table is None else table.columns[table.columns.duplicated()].unique()
        if len(repeated):
            raise ValueError(
                f"{table_name} has more than one column named {list(repeated)}; "
                "give every column its own name"
            )
    return objects, fovs, feature_cols, group_cols, fov_col


def _field_identity(
    objects: pd.DataFrame,
    fovs: pd.DataFrame | None,
    group_cols: Sequence[str] | None,
    resolved_fov_col: str,
) -> tuple[pd.DataFrame, Any, tuple[str, ...]]:
    """The observations with the FOV table's group columns attached, the analysis groups and
    the field key (the identity columns, then the field identifier)."""
    metadata_to_attach = list(MORPHOBRICKS_GROUP_COLUMNS)
    if group_cols is not None:
        metadata_to_attach.extend(
            column for column in group_cols if column not in metadata_to_attach
        )
    _check_hashable_identity(objects, fovs, [resolved_fov_col, *metadata_to_attach])
    objects = _attach_fov_metadata(objects, fovs, resolved_fov_col, metadata_to_attach)
    resolved_group_cols = resolve_group_cols(objects, fovs, group_cols)
    identity_cols = resolve_identity_cols(objects, fovs, resolved_group_cols, resolved_fov_col)
    return objects, resolved_group_cols, (*identity_cols, resolved_fov_col)


def _check_rows(
    objects: pd.DataFrame,
    fovs: pd.DataFrame | None,
    feature_cols: Any,
    fov_key: tuple[str, ...],
    fov_col_was_chosen: bool,
    observation_unit: str,
) -> None:
    """Each column one role, every column present, the identity complete, planes neither
    fields nor observations, real feature values; invalid rows warned about."""
    identity_cols, resolved_fov_col = fov_key[:-1], fov_key[-1]
    _check_column_roles(feature_cols, identity_cols, resolved_fov_col)
    _require_columns(objects, [*fov_key, *feature_cols], "objects")
    # Missing identifiers first: the plane rule would otherwise name only their symptom.
    _check_identity_values(objects, fov_key)
    _check_within_fov_axes(
        objects, fov_key, feature_cols, fov_col_was_chosen=fov_col_was_chosen, fovs=fovs
    )
    _check_positions_within_fields(
        objects, fov_key, feature_cols, fov_col_was_chosen=fov_col_was_chosen
    )
    _check_planes_within_fields(objects, fov_key, feature_cols, observation_unit)
    _check_feature_values(objects, feature_cols)
    warn_invalid_rows(objects, [*fov_key, *feature_cols], caller_stacklevel() + 1)


class MissingFovTableWarning(UserWarning):
    """No FOV table was given: acquired FOVs without observations are missing."""


def fov_table_from_observations(
    objects: pd.DataFrame,
    fov_key: Sequence[str],
    *,
    carry: Sequence[str] = (),
) -> pd.DataFrame:
    """A FOV table reconstructed from an observation table, when no acquisition record exists.

    Published feature tables (for example CellProfiler exports) often come
    without a table of acquired FOVs. This builds one row per distinct
    ``fov_key`` (for example ``["condition", "biological_repeat", "image_file"]``)
    and copies ``carry`` columns that are constant within each FOV (stage
    coordinates, area, acquisition order). It warns every time, because FOVs
    that were acquired but hold no observations are missing from such a table:
    acquisition effort is undercounted and FOVs with nothing in them never
    count, which makes every sampling answer optimistic. Provide the real FOV
    table whenever it exists.
    """
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


def __getattr__(name: str) -> object:
    """``validate_input_tables`` lives in ``data_check`` (it attaches that report to its
    errors); it is still importable from here, loaded on first use so that ``schema``
    never imports ``data_check`` while ``data_check`` imports ``schema``."""
    if name == "validate_input_tables":
        from .data_check import validate_input_tables

        return validate_input_tables
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DEFAULT_OBSERVATION_UNIT",
    "OBJECT_LIKE_UNITS",
    "PLANE_FEATURE_ADVICE",
    "MissingFovTableWarning",
    "ValidatedInput",
    "feature_dtype_problem",
    "fov_positions",
    "fov_table_from_observations",
    "normalize_observation_unit",
    "observation_words",
    "plane_features",
    "resolve_observation_unit",
    "unhashable_values",
    "unit_plural",
    "unit_words",
]
