"""The copy rule of ``data_check.check_tables`` (private): rows written twice in one table.

- rows that repeat another row exactly (identifiers included), leaving out rows
  with no value in any feature (failed segmentations look alike by themselves);
- observations that repeat another observation of the same field in every
  feature while another column differs: a merge with a table that has one row
  per channel or plane (a CellProfiler Image table with a Channel column), or a
  table appended to itself and saved with a fresh running index;
- an object label (``label_id``, CellProfiler's ``ObjectNumber``, ``object_id``
  or ``label``) twice in one field.

Every test runs, none hides another: the label test reads the rows that are
not already reported as copies.

Shared-private: ``check_copies`` is used by ``_data_rules._check_copies``.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence

import numpy as np
import pandas as pd

from ._data_problems import Collector, _examples
from ._wording import count
from .metadata import MORPHOBRICKS_FOV_CANDIDATES, WITHIN_FOV_METADATA_COLUMNS

# Columns that name one segmented object of an image, as the common pipelines write them.
LABEL_COLUMNS = ("label_id", "ObjectNumber", "object_id", "label")
# A feature whose distinct values are at least this share of its values is continuous: two
# objects agreeing in it are not a coincidence.
CONTINUOUS_SHARE = 0.5


def _comparable(objects: pd.DataFrame) -> pd.DataFrame:
    """``objects`` with cells that cannot be compared as they are (lists, arrays) as their text.

    An embedding stored as one list per row is valid (it is not a feature), but
    ``DataFrame.duplicated`` cannot hash it.
    """
    comparable = objects.copy()
    for column in objects.columns[(objects.dtypes == object).to_numpy()]:
        values = objects[column]
        if not values.map(lambda value: isinstance(value, Hashable)).all():
            comparable[column] = values.map(repr)
    return comparable


def _numeric_features(objects: pd.DataFrame, feature_cols: Sequence[str]) -> list[str]:
    """The declared features present and numeric (the feature rule reports the others)."""
    return [
        column
        for column in dict.fromkeys(feature_cols)
        if isinstance(column, Hashable)
        and column in objects.columns
        and pd.api.types.is_numeric_dtype(objects[column])
        and not pd.api.types.is_bool_dtype(objects[column])
    ]


def _continuous(objects: pd.DataFrame, features: Sequence[str]) -> list[str]:
    """The features whose distinct values are at least ``CONTINUOUS_SHARE`` of their values."""
    kept = []
    for column in features:
        values = objects[column].dropna()
        if len(values) and values.nunique() >= CONTINUOUS_SHARE * len(values):
            kept.append(column)
    return kept


def check_copies(
    c: Collector, objects: pd.DataFrame, key: Sequence[str], feature_cols: Sequence[str] = ()
) -> None:
    """Exact copies, observations repeated in every feature within a field, labels twice.

    The field is the one the validator uses (``key``: the group columns and the
    resolved field identifier, even one named with ``fov_col``), with its file
    and its plane, channel or frame (``z``, ``c``, ``t``) when recorded, so
    labels that restart in each field, file or plane are fine.
    """
    features = _numeric_features(objects, feature_cols)
    no_value = (
        objects[features].isna().all(axis=1).to_numpy()
        if features
        else np.zeros(len(objects), dtype=bool)
    )
    try:
        copies = objects.duplicated(keep="first")
    except TypeError:  # fallback: a column of lists: compared by their text
        objects = _comparable(objects)
        copies = objects.duplicated(keep="first")
    copies &= ~no_value
    if copies.any():
        c.warning(
            f"{count(int(copies.sum()), 'observation')} {'is' if copies.sum() == 1 else 'are'} "
            "an exact copy of another row.",
            f"observation table, {_examples(objects, copies, list(objects.columns[:4]))}.",
            "A file read or appended twice counts every cell twice; remove the copies, or keep "
            "them if identical rows are real.",
        )
    if c.has_errors():
        # A misspelt group column already breaks field identity; rows would repeat.
        return
    kept = objects[~copies.to_numpy()]
    image = [*key, *MORPHOBRICKS_FOV_CANDIDATES, *WITHIN_FOV_METADATA_COLUMNS]
    field = [col for col in dict.fromkeys(image) if col in kept.columns]
    if not field:
        return
    repeats = pd.Series(False, index=kept.index)
    # Continuous on the distinct observations, so a table appended to itself stays continuous.
    continuous = (
        _continuous(kept.drop_duplicates([*field, *features]), features) if features else []
    )
    if continuous:
        repeats = kept.duplicated([*field, *features], keep="first") & kept[continuous].notna().any(
            axis=1
        )
    if repeats.any():
        n = int(repeats.sum())
        c.warning(
            f"{count(n, 'observation')} {'repeats' if n == 1 else 'repeat'} another observation "
            "of the same field in every feature, although another column differs.",
            f"observation table, {_examples(kept, repeats, [*field, *features[:2]])}.",
            "A table merged with one that has a row per channel, plane or image (a CellProfiler "
            "Image table with a Channel column), or appended to itself and saved with a fresh "
            "index, repeats every object; merge on one row per field and object, or keep the "
            "rows if these objects really are identical.",
        )
    labels = kept[~repeats.to_numpy()]
    for label in (col for col in LABEL_COLUMNS if col in labels.columns):
        twice = labels.duplicated([*field, label], keep="first") & labels[label].notna()
        if twice.any():
            n = int(twice.sum())
            c.warning(
                f"{count(n, label)} {'appears' if n == 1 else 'appear'} twice in the same field.",
                f"observation table, {_examples(labels, twice, [*field, label])}.",
                "A label names one segmented object of a field; check for a file appended "
                "twice, a field identifier shared by two images, or a plane or frame column "
                "not called z, c or t (name it so: 't' for a 'timepoint' column).",
            )


__all__ = ["LABEL_COLUMNS", "check_copies"]
