"""The columns naming an observation within its field, for the feature tables (#404).

``label_id`` names an object, ``patch_id`` a CellPatcher patch. ``patch_id`` is positional
(``image_0000_cell_00001``: the image's place in its folder) and repeats across the folders of a
combined tree run, so it is read after the field key, never alone. A producer that writes every
column (CellPatcher) leaves ``label_id`` empty for grid and random patches: such a column, empty
in every row, is read as absent, as ``metadata.EMPTY_MEANS_ABSENT`` reads an empty
``position_id`` or ``technical_repeat``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandas as pd

from ._wording import listing
from .metadata import column_has_values

# The MorphoBricks column naming an observation within its field.
LABEL_COLUMN = "label_id"
# CellPatcher's patch within its field: grid and random patches have no label, several share a
# field.
PATCH_COLUMN = "patch_id"
# Within-field identity columns read as absent when empty in every row.
EMPTY_MEANS_ABSENT_WITHIN = (LABEL_COLUMN, PATCH_COLUMN)


def filled_within_columns(table: pd.DataFrame) -> list[str]:
    """``label_id`` and ``patch_id`` when ``table`` has a value of them, in that order."""
    return [c for c in EMPTY_MEANS_ABSENT_WITHIN if column_has_values(table, c)]


def feature_candidates(table: pd.DataFrame, identity: Sequence[str]) -> list[Any]:
    """The columns of ``table`` beside the identity, without a ``label_id`` or ``patch_id`` that
    the identity does not use and that is empty in every row (read as absent)."""
    return [
        c
        for c in table.columns
        if c not in identity
        and not (c in EMPTY_MEANS_ABSENT_WITHIN and not column_has_values(table, c))
    ]


def absent_identity_text(label: str, absent: Sequence[str], identity: Sequence[str]) -> str:
    """Why a table lacking identity columns cannot be matched, naming the within-field ones."""
    text = (
        f"{label!r} lacks the identity columns {list(absent)}: each row must name its "
        f"observation by {list(identity)}"
    )
    within = [c for c in absent if c in EMPTY_MEANS_ABSENT_WITHIN]
    if within:
        text += (
            f" ({listing(within)} has values in the observation table, so every table must "
            "carry it; only a column empty in every observation is read as absent)"
        )
    return text


def duplicate_identity_text() -> str:
    """What to do when observations of a field share their identity."""
    return (
        f"add a column that tells the observations of a field apart ({LABEL_COLUMN} for "
        f"objects, {PATCH_COLUMN} for patches), filled in every row"
    )
