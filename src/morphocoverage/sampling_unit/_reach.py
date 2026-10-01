"""Whether a test can help (private): run a test only when the design lets it reach 0.05."""

from __future__ import annotations

import pandas as pd


def reachable(floors: pd.Series, alpha: float = 0.05) -> pd.Series:
    """Which tests can reach ``alpha`` after Bonferroni over the tests kept: their smallest
    possible p times the number kept is at most ``alpha`` (dropping a test lowers the factor,
    so this is repeated until it holds for every test kept)."""
    keep = floors.notna()
    while keep.any():
        kept = keep & (floors * int(keep.sum()) <= alpha)
        if kept.equals(keep):
            break
        keep = kept
    return keep


__all__ = ["reachable"]
