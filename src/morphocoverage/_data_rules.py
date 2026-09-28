"""The rules of ``data_check.check_tables``, one function per kind of problem (private).

Each rule reads the tables, records problems on a ``Collector`` in plain words
(what is wrong, where, how to fix it) and never raises for bad data.

Shared-private: every ``_check_*`` rule, ``_attach``, ``_resolve_fov_col``,
``_resolve_groups``, ``_names`` and ``_tables``, used by ``data_check``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from ._copy_rules import check_copies
from ._data_problems import (  # noqa: F401  (re-exported)
    ERROR,
    EXAMPLES,
    WARNING,
    Collector,
    Problem,
    _examples,
    _names,
    _value,
)
from ._wording import count, percent
from .metadata import (
    EMPTY_MEANS_ABSENT,
    MORPHOBRICKS_FOV_CANDIDATES,
    MORPHOBRICKS_GROUP_COLUMNS,
    REPEAT_COLUMN,
    WITHIN_FOV_METADATA_COLUMNS,
    attach_mapping,
    canonical_spelling,
    empty_values,
    near_miss_advice,
    repeated_names,
)
from .metadata import _tables as _linked_tables
from .schema import (
    PLANE_FEATURE_ADVICE,
    feature_dtype_problem,
    plane_features,
    unhashable_values,
)


def _tables(objects: pd.DataFrame, fovs: pd.DataFrame | None) -> list[tuple[str, pd.DataFrame]]:
    """The linked tables, named as these messages name them (``metadata._tables``)."""
    return _linked_tables(objects, fovs, ("observation table", "FOV table"))


def _check_spellings(
    c: Collector, objects: pd.DataFrame, fovs: pd.DataFrame | None, explicit: Sequence[str]
) -> None:
    """Near-miss spellings of the MorphoBricks columns (``Repeat`` for ``repeat``)."""
    for name, table in _tables(objects, fovs):
        for column in table.columns:
            if column in explicit:
                continue
            meant = canonical_spelling(column)
            if meant is not None:
                kind = (
                    "the within-field coordinate"
                    if meant in WITHIN_FOV_METADATA_COLUMNS
                    else "the MorphoBricks column"
                )
                c.error(
                    f"Column {column!r} looks like {kind} {meant!r} "
                    "but is spelt differently, so it would be ignored.",
                    f"{name}, column {column!r}.",
                    near_miss_advice(column, meant),
                )


def _resolve_fov_col(
    c: Collector, objects: pd.DataFrame, fovs: pd.DataFrame | None, fov_col: str | None
) -> str | None:
    """The field identifier, or ``None`` after recording why none can be used."""
    if fov_col is not None:
        missing = [name for name, t in _tables(objects, fovs) if fov_col not in t.columns]
        if missing:
            c.error(
                f"The field identifier fov_col={fov_col!r} is not a column of the "
                f"{' and the '.join(missing)}.",
                f"columns present: {_names(list(objects.columns))}.",
                "Pass the name of the column that identifies each acquired field.",
            )
            return None
        return fov_col
    for candidate in MORPHOBRICKS_FOV_CANDIDATES:
        in_objects = candidate in objects.columns
        in_fovs = fovs is not None and candidate in fovs.columns
        if not in_objects and not in_fovs:
            continue
        if fovs is not None and in_objects != in_fovs:
            present, absent = (
                ("observation table", "FOV table")
                if in_objects
                else ("FOV table", "observation table")
            )
            c.error(
                f"The field identifier {candidate!r} is in the {present} but not in the {absent}.",
                f"{present}, column {candidate!r}.",
                f"Add {candidate!r} to the {absent} so each observation can be matched to its "
                "field.",
            )
            return None
        return candidate
    c.error(
        "No column identifies the acquired fields.",
        f"observation table columns: {_names(list(objects.columns))}.",
        "Add 'position_id' (one value per stage position), or 'image_file' when one file is "
        "one field, or name your column with fov_col=...",
    )
    return None


def _resolve_groups(
    c: Collector,
    objects: pd.DataFrame,
    fovs: pd.DataFrame | None,
    group_cols: Sequence[str] | None,
    fov_col: str,
) -> tuple[list[str], list[str]]:
    """The analysis groups and the identity columns, after checking both tables hold them
    and that ``group_cols`` names each column once."""
    if group_cols is not None:
        groups = list(dict.fromkeys(group_cols))
        repeated = repeated_names(group_cols)
        if repeated:
            c.error(
                "A group column is named more than once.",
                f"group_cols={list(group_cols)}: {_names(repeated)} repeated.",
                "Name each group column once.",
            )
    else:
        groups = [
            g
            for g in MORPHOBRICKS_GROUP_COLUMNS
            if g in objects.columns or (fovs is not None and g in fovs.columns)
        ]
    identity = list(groups) + [
        g
        for g in MORPHOBRICKS_GROUP_COLUMNS
        if g not in groups
        and g != fov_col
        and (g in objects.columns or (fovs is not None and g in fovs.columns))
    ]
    for column in identity:
        in_objects = column in objects.columns
        in_fovs = fovs is not None and column in fovs.columns
        if not in_objects and not in_fovs:
            c.error(
                f"The group column {column!r} is in neither table.",
                f"group_cols={list(group_cols or [])}.",
                f"Add {column!r} to the FOV table, or leave it out of group_cols.",
            )
        elif in_objects and fovs is not None and not in_fovs:
            # A named group column too: the validator requires every key column in the FOV table.
            c.error(
                f"{column!r} is in the observation table but not in the FOV table.",
                f"observation table, column {column!r}.",
                f"Add {column!r} to the FOV table too, so fields without observations keep their "
                "group.",
            )
    return groups, identity


def _attach(
    c: Collector,
    objects: pd.DataFrame,
    fovs: pd.DataFrame | None,
    fov_col: str,
    identity: Sequence[str],
) -> pd.DataFrame:
    """Copy FOV-table-only group columns onto the observations, when unambiguous.

    The copied columns are marked in ``enriched.attrs["attached"]``, so a missing
    value is reported once, in the FOV table it came from.
    """
    if fovs is None or fov_col not in fovs.columns:
        return objects
    enriched = objects.copy()
    enriched.attrs["attached"] = []
    unknown_reported = False
    for column in identity:
        if column in enriched.columns or column not in fovs.columns:
            continue
        # Through (repeat, field) when a field name restarts in each repeat, as the validator.
        key, mapping = attach_mapping(enriched, fovs, column, identity, fov_col)
        if fov_col in enriched.columns and not unknown_reported:
            present = pd.MultiIndex.from_frame(enriched[key]) if len(key) > 1 else enriched[fov_col]
            listed = pd.MultiIndex.from_frame(fovs[key]) if len(key) > 1 else fovs[fov_col]
            unknown = pd.Series(~np.asarray(present.isin(listed)), index=enriched.index)
            if unknown.any():
                # Their group values cannot be copied, so the field check below would not see
                # them.
                unknown_reported = True
                c.error(
                    "Some observations belong to fields that are not in the FOV table.",
                    f"observation table, {_examples(enriched, unknown, key)}.",
                    f"Add those fields to the FOV table, or check that {fov_col!r} is spelt the "
                    "same way in both tables.",
                )
        reused = mapping[key].duplicated(keep=False)
        if reused.any():
            c.error(
                f"{column!r} is only in the FOV table, and the same {fov_col!r} has several "
                f"values of {column!r}, so observations cannot be matched to a {column!r}.",
                f"FOV table, {_examples(mapping, reused, [*key, column])}.",
                f"Add {column!r} to the observation table, or make {fov_col!r} unique across "
                f"{column!r} values.",
            )
            continue
        attached = [*enriched.attrs["attached"], column]
        enriched = enriched.merge(mapping, on=key, how="left").set_axis(enriched.index)
        enriched.attrs["attached"] = attached
    return enriched


def _check_features(
    c: Collector, objects: pd.DataFrame, feature_cols: Sequence[str], taken: set[str]
) -> None:
    """Feature columns: named, distinct, present, numeric, finite; missing values reported."""
    if not feature_cols:
        c.error(
            "No feature column was named.",
            "feature_cols names no column.",
            "Name the numeric phenotype columns, e.g. feature_cols=['area'] (or Setup.readout).",
        )
        return
    # Sorted as text: column names may mix strings and numbers.
    duplicates = sorted(repeated_names(feature_cols), key=str)
    if duplicates:
        c.error(
            "A feature column is named twice.",
            f"feature_cols: {_names(duplicates)}.",
            "Name each feature once.",
        )
    overlap = sorted(set(feature_cols) & taken, key=str)
    if overlap:
        c.error(
            "A column is used both as a feature and as a group or field identifier.",
            f"columns {_names(overlap)}.",
            "Give each column one role: remove it from feature_cols or from group_cols/fov_col.",
        )
    planes = plane_features(feature_cols)
    if planes:
        c.error(
            "A within-field coordinate is named as a feature.",
            f"feature_cols: {_names(planes)}.",
            PLANE_FEATURE_ADVICE[0].upper() + PLANE_FEATURE_ADVICE[1:] + ".",
        )
    missing = [f for f in dict.fromkeys(feature_cols) if f not in objects.columns]
    if missing:
        close = {f: _close_column(f, objects.columns) for f in missing}
        hints = [f"{found!r} for {f!r}" for f, found in close.items() if found is not None]
        c.error(
            f"{count(len(missing), 'feature column')} named in feature_cols "
            f"{'is' if len(missing) == 1 else 'are'} not in the observation table.",
            f"missing: {_names(missing)}"
            + (f"; the table has {', '.join(hints)}" if hints else "")
            + ".",
            "Check the spelling (spaces and case count), or add the columns to the observation "
            "table.",
        )
    for feature in dict.fromkeys(feature_cols):
        if feature not in objects.columns:
            continue
        values = objects[feature]
        problem = feature_dtype_problem(values)
        if problem == "complex":
            c.error(
                f"Feature {feature!r} holds complex numbers.",
                f"observation table, column {feature!r}.",
                "Keep one real number per feature: split it into its real and imaginary parts "
                "(or its modulus and phase), each a feature of its own.",
            )
            continue
        if problem is not None:
            as_numbers = pd.to_numeric(values, errors="coerce")
            bad = as_numbers.isna() & values.notna()
            c.error(
                f"Feature {feature!r} is not numeric.",
                "observation table, "
                + (
                    _examples(objects, bad, [feature])
                    if bad.any()
                    else "column type (every value is a number, stored as text or objects)"
                )
                + ".",
                "Convert the column to numbers (a decimal point, not a comma; empty cells for "
                "missing values)"
                + (
                    "."
                    if bad.any()
                    else f", e.g. objects[{feature!r}] = pd.to_numeric(objects[{feature!r}])."
                ),
            )
            continue
        numbers = values.to_numpy(dtype=float, na_value=np.nan)
        infinite = pd.Series(np.isinf(numbers), index=objects.index)
        if infinite.any():
            c.error(
                f"Feature {feature!r} contains infinite values.",
                f"observation table, {_examples(objects, infinite, [feature])}.",
                "Replace infinite values by missing values (NaN) or fix their computation, for "
                "example a division by zero.",
            )
        nan = pd.Series(np.isnan(numbers), index=objects.index)
        if nan.any():
            c.warning(
                f"Feature {feature!r} is missing in {percent(nan.mean())} of observations.",
                f"observation table, {_examples(objects, nan, [feature])}.",
                "Missing values are kept and never imputed; check they are expected (for "
                "example, a measurement that failed).",
            )


def _close_column(name: Any, columns: Sequence[Any]) -> Any | None:
    """A column that differs from ``name`` only in case or surrounding whitespace.

    ``None`` for a name that is not text (``7``): only text has case or spaces.
    """
    if not isinstance(name, str):
        return None
    target = name.strip().casefold()
    for column in columns:
        if isinstance(column, str) and column != name and column.strip().casefold() == target:
            return column
    return None


def _loose(value: Any) -> str:
    """A value compared loosely: without surrounding spaces or case, a number by its value."""
    text = str(value).strip().casefold()
    try:
        return repr(float(text))
    except ValueError:
        return text


def _near_values(objects: pd.DataFrame, fovs: pd.DataFrame, key: Sequence[str]) -> str:
    """Why observation keys miss the FOV table, when it is spaces, case or number against text."""
    hints = []
    for column in key:
        seen = pd.Series(objects[column].dropna().unique())
        listed = pd.Series(fovs[column].dropna().unique())
        absent = seen[~seen.isin(listed)]
        if absent.empty:
            continue
        by_text = {_loose(v): v for v in listed}
        pairs = [(v, by_text[_loose(v)]) for v in absent if _loose(v) in by_text][:EXAMPLES]
        if pairs:
            shown = "; ".join(f"{_value(a)} against {_value(b)}" for a, b in pairs)
            why = " or ".join(dict.fromkeys(_why_different(a, b) for a, b in pairs))
            hints.append(f"'{column}' {why} ({shown})")
    return (" In these rows " + "; ".join(hints) + ".") if hints else ""


def _why_different(seen: Any, listed: Any) -> str:
    """Why two values that ``_loose`` finds equal are not the same value."""
    if isinstance(seen, str) != isinstance(listed, str):
        return "is text in one table and a number in the other"
    if str(seen).strip().casefold() == str(listed).strip().casefold():
        return "differs only in spaces or case"
    return "writes the same number differently"  # '1' against '1.0'


def _check_copies(
    c: Collector, objects: pd.DataFrame, key: Sequence[str], feature_cols: Sequence[str] = ()
) -> None:
    """Exact copies, observations repeated in every feature of a field, labels twice in one.

    The rule lives in ``_copy_rules.check_copies``; the field is the one the
    validator uses (``key``), so labels that restart in each field, file or
    plane are fine.
    """
    check_copies(c, objects, key, feature_cols)


def _check_label_variants(
    c: Collector, objects: pd.DataFrame, fovs: pd.DataFrame | None, columns: Sequence[str]
) -> None:
    """Group labels of one table that differ only in spaces, case, or number against text.

    ``1`` and ``'1'`` (two files read differently), ``'drug'``, ``'drug '`` and
    ``'Drug'`` (a hand-typed table) are separate groups to the analysis, and
    print alike in its summaries. They may be real, so this is a warning,
    given once per column: from the FOV table, which lists every field, else
    from the observation table.
    """
    reported: set[str] = set()
    for name, table in reversed(_tables(objects, fovs)):
        for column in dict.fromkeys(columns):
            if column not in table.columns or column in reported:
                continue
            values = table[column].dropna()
            if unhashable_values(values).any():
                continue
            by_loose: dict[str, list] = {}
            for value in values.drop_duplicates().tolist():
                by_loose.setdefault(_loose(value), []).append(value)
            variants = [labels for labels in by_loose.values() if len(labels) > 1]
            if not variants:
                continue
            shown = "; ".join(
                " against ".join(_value(label) for label in labels)
                for labels in variants[:EXAMPLES]
            )
            more = f" and {len(variants) - EXAMPLES} more" if len(variants) > EXAMPLES else ""
            reported.add(column)
            c.warning(
                f"{column!r} holds labels that differ only in spaces, case, or number against "
                "text, so each is analysed as a group of its own.",
                f"{name}, column {column!r}: {shown}{more}.",
                "If they name one group, write them the same way in every row (strip the spaces, "
                "use one case, read the column as text in every file); if they are different "
                "groups, give them names that differ clearly.",
            )


def _check_key_values(
    c: Collector, objects: pd.DataFrame, fovs: pd.DataFrame | None, key: Sequence[str]
) -> None:
    """Every key column is complete in both tables; a copied column is checked at its source."""
    attached = set(objects.attrs.get("attached", []))
    for name, table in _tables(objects, fovs):
        for column in key:
            if column not in table.columns or (table is objects and column in attached):
                continue
            empty = empty_values(table[column])
            if not empty.any():
                continue
            role = "field identifier" if column == key[-1] else "group column"
            fix = f"Fill {column!r} in every row of the {name}."
            if column in EMPTY_MEANS_ABSENT:
                fix += " If it does not apply to this dataset, leave it empty in every row instead."
            c.error(
                f"The {role} {column!r} is empty in some rows, so their field or group is unknown.",
                f"{name}, {_examples(table, empty, [column])}.",
                fix,
            )


def _check_fov_table(
    c: Collector, objects: pd.DataFrame, fovs: pd.DataFrame, key: Sequence[str]
) -> None:
    """One row per field, and every observation's field listed (rows with a complete key).

    Skipped when a key column is missing from the FOV table (already reported):
    over the remaining columns, distinct fields would look like one listed twice.
    """
    if any(k not in fovs.columns for k in key):
        return
    key = [k for k in key if k in objects.columns]
    if not key:
        return
    objects = objects[objects[key].notna().all(axis=1)]
    fovs = fovs[fovs[key].notna().all(axis=1)]
    repeated = fovs.duplicated(key, keep=False)
    if repeated.any():
        c.error(
            "The FOV table lists a field more than once.",
            f"FOV table, {_examples(fovs, repeated, key)}.",
            "Keep exactly one row per acquired field (identified by " + ", ".join(key) + ").",
        )
    listed = set(zip(*(fovs[k].tolist() for k in key)))
    observed = pd.Series(
        [row not in listed for row in zip(*(objects[k].tolist() for k in key))],
        index=objects.index,
    )
    if observed.any():
        c.error(
            "Some observations belong to fields that are not in the FOV table.",
            f"observation table, {_examples(objects, observed, key)}."
            + _near_values(objects[observed], fovs, key),
            "Add those fields to the FOV table, or check that condition, repeat and the field "
            "identifier are spelt the same way in both tables.",
        )
    seen = set(zip(*(objects[k].tolist() for k in key)))
    empty = pd.Series(
        [row not in seen for row in zip(*(fovs[k].tolist() for k in key))], index=fovs.index
    )
    if empty.any():
        c.warning(
            f"{count(int(empty.sum()), 'field')} of the FOV table "
            f"{'holds' if int(empty.sum()) == 1 else 'hold'} no observation.",
            f"FOV table, {_examples(fovs, empty, key)}.",
            "An empty field is a real result and counts as effort; if an image failed "
            "(unreadable, failed segmentation), mark it instead of listing it as empty.",
        )


def _check_control(
    c: Collector, fovs: pd.DataFrame, control: Any, from_observations: bool = False
) -> None:
    """The declared control is a non-empty mapping, exists, and in every repeat.

    ``fovs`` is the FOV table, or the observation table when there is none
    (``from_observations``): the fields seen there are then all that is known.
    """
    table = "observation table" if from_observations else "FOV table"
    if not isinstance(control, Mapping) or not control:
        c.error(
            "The control does not name a column and its value.",
            f"control={control!r}.",
            "Name the control by a column and its value, e.g. control={'condition': 'MOCK'}.",
        )
        return
    missing = [k for k in control if k not in fovs.columns]
    if missing:
        c.error(
            f"The control names a column that is not in the {table}.",
            f"control={dict(control)}, missing: {_names(missing)}.",
            f"Name the control by {table} columns, e.g. control={{'condition': 'MOCK'}}.",
        )
        return
    listed = [k for k, value in control.items() if not pd.api.types.is_scalar(value)]
    if listed:
        c.error(
            "The control gives a column several values, or a value that is not one label.",
            f"control={dict(control)}, column {_names(listed)}.",
            "Name one value per column, e.g. control={'condition': 'MOCK'}.",
        )
        return
    read = [*control, *([REPEAT_COLUMN] if REPEAT_COLUMN in fovs.columns else [])]
    if any(unhashable_values(fovs[column]).any() for column in read):
        return  # lists in an identity column: reported with the field identity
    is_control = np.ones(len(fovs), dtype=bool)
    for column, value in control.items():
        is_control &= (fovs[column] == value).to_numpy()
    if not is_control.any():
        values = fovs[next(iter(control))].drop_duplicates().tolist()
        c.error(
            f"No field matches the control {dict(control)}.",
            f"values present: {_names(values)}.",
            "Check the spelling of the control condition.",
        )
        return
    if REPEAT_COLUMN in fovs.columns:
        with_control = set(fovs.loc[is_control, REPEAT_COLUMN])
        lacking = [r for r in fovs[REPEAT_COLUMN].drop_duplicates() if r not in with_control]
        if lacking:
            c.warning(
                f"{count(len(lacking), 'repeat')} {'has' if len(lacking) == 1 else 'have'} no "
                f"control field.",
                f"repeats without {dict(control)}: {_names(lacking)}.",
                "Normalising to the control and comparing conditions work within a repeat; add "
                "the control to those repeats or leave them out.",
            )
