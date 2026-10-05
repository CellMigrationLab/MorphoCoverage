"""The labelled rare state as a readout (private; ``_shared.conditions``, ``figures``).

With ``Setup(rare_label=...)`` the label (booleans or 0/1) is also compared between the
conditions like any feature (owner, 2026-10-02): read as 0/1, each well's or repeat's mean
is the share of its cells in the state, so the paired test across repeats compares the
shares, and the heatmap and ranking show it. The column is named ``"<label> share"``.

Its figures plot one point per field, the field's share (``share_by_field``): a cell is in
the state or not, so its 0 or 1 says nothing a reader can see (owner, 2026-10-02); the tests
keep the cells.

A missing label is unknown, never negative (owner, 2026-10-05): it stays missing in the share
column, so every mean, test and field share leaves the cell out of both its numerator and
its denominator, as the label check, the populations, the plan and the clustering in few
fields do. ``unlabelled_note`` says how many cells that is, once, among the report's notes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from .._wording import count
from ..data_check import validate_input_tables

if TYPE_CHECKING:
    from .setup import Setup


def share_column(setup: Setup) -> str | None:
    """``"mitotic share"`` for ``rare_label="mitotic"``, or ``None`` without a label."""
    return None if setup.rare_label is None else f"{setup.rare_label} share"


def compared_columns(setup: Setup) -> list[str]:
    """The columns the conditions are compared on: the declared ones, then the share."""
    share = share_column(setup)
    return [*setup.columns, *([share] if share else [])]


def with_share(objects: pd.DataFrame, setup: Setup) -> pd.DataFrame:
    """``objects`` with the share column (the label as 0.0/1.0) when a label is declared and
    present; unchanged otherwise."""
    share = share_column(setup)
    if share is None or setup.rare_label not in objects.columns or share in objects.columns:
        return objects
    return objects.assign(**{share: objects[setup.rare_label].astype(float)})


def unlabelled_note(objects: pd.DataFrame, setup: Setup) -> str | None:
    """The report's one note on the observations without a label, or ``None`` when every
    observation carries one (or no label is declared)."""
    label = setup.rare_label
    if label is None or label not in objects.columns:
        return None
    missing = int(objects[label].isna().sum())
    if not missing:
        return None
    return (
        f"{count(missing, setup.unit_word)} of {len(objects)} carry no {label} label: unknown, "
        f"not negative, so they are left out of every share of labelled {setup.unit_words} "
        "(numerator and denominator)."
    )


def share_by_field(prepared: pd.DataFrame, fovs: pd.DataFrame, setup: Setup) -> pd.DataFrame:
    """One row per field holding cells: its FOV key and its share of cells in the state
    (``prepared`` carries the share column, ``with_share``)."""
    share = share_column(setup)
    data = validate_input_tables(prepared, fovs=fovs, feature_cols=[share], **setup.common)
    key = list(data.fov_key)
    return data.objects.groupby(key, sort=False, dropna=False)[share].mean().reset_index()


def plotted_values(
    prepared: pd.DataFrame, fovs: pd.DataFrame, setup: Setup, feature: str
) -> pd.DataFrame:
    """The table a figure of ``feature`` plots: the share per field for the share column,
    ``prepared`` itself for every other feature."""
    if feature != share_column(setup) or feature not in prepared.columns:
        return prepared
    return share_by_field(prepared, fovs, setup)


__all__ = [
    "compared_columns",
    "plotted_values",
    "share_by_field",
    "share_column",
    "unlabelled_note",
    "with_share",
]
