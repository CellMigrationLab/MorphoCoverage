"""The labelled rare state as a readout (private; ``_shared.conditions``, ``figures``).

With ``Setup(rare_label=...)`` the label (booleans or 0/1) is also compared between the
conditions like any feature (owner, 2026-10-02): read as 0/1, each well's or repeat's mean
is the share of its cells in the state, so the paired test across repeats compares the
shares, and the heatmap and ranking show it. The column is named ``"<label> share"``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

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


__all__ = ["compared_columns", "share_column", "with_share"]
