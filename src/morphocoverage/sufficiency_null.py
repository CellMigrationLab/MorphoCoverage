"""Question 2's guard: a definition of new that calls a homogeneous population new says nothing.

``strategies.novelty_null`` measures a definition's own false-alarm rate at a group's size and
feature count. Question 3 and the fields to check already stop listing new states where that
rate is above 20%; question 2 must do the same (audit of the THBS5 and Silencing reports, A2):
a "chance of a new state in one more field" and the "fields still needed" read from such a
definition are not evidence about the biology. A group in that case keeps its counts but gets
``uninformative`` (the reason) and ``null_rate`` in the ``next_fov`` table.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ._wording import percent
from .evaluator import iter_analysis_groups
from .schema import ValidatedInput, fov_positions
from .strategies.novelty_null import not_informative
from .strategies.support_novelty import SupportNoveltyEstimator


def mark_uninformative(
    frame: pd.DataFrame, data: ValidatedInput, novelty: SupportNoveltyEstimator
) -> pd.DataFrame:
    """``frame`` (one row per group, in ``iter_analysis_groups`` order) with the columns
    ``uninformative`` (the reason a list of new states is not informative at the group's size,
    else ``None``) and ``null_rate`` (the definition's rate on a homogeneous population)."""
    reasons: list[str | None] = []
    rates: list[float | None] = []
    for _, group_data in iter_analysis_groups(data):
        positions = fov_positions(group_data, group_data.fovs)
        sizes = np.bincount(positions[positions >= 0], minlength=len(group_data.fovs))
        reason, rate = not_informative(novelty, sizes, group_data.feature_cols)
        reasons.append(reason)
        rates.append(rate)
    if len(reasons) != len(frame):
        raise ValueError("the next-field table and the analysis groups disagree in length")
    return frame.assign(
        uninformative=pd.Series(reasons, index=frame.index, dtype=object),
        null_rate=pd.Series(rates, index=frame.index, dtype=object),
    )


def is_uninformative(row: Any) -> bool:
    """Whether a ``next_fov`` row (or mapping) carries a reason, tolerating older tables."""
    reason = row.get("uninformative") if hasattr(row, "get") else None
    return isinstance(reason, str) and bool(reason)


def split_uninformative(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``(informative, uninformative)`` rows of a ``next_fov`` table."""
    if "uninformative" not in frame.columns:
        return frame, frame.iloc[:0]
    mask = frame["uninformative"].map(lambda r: isinstance(r, str) and bool(r))
    return frame[~mask], frame[mask]


def uninformative_sentence(silent: pd.DataFrame, groups: int) -> str | None:
    """One line for the groups where the chance of a new state is not given, with the fields
    it was judged at and the rates the definition showed on a homogeneous population;
    ``None`` when no group is in that case."""
    if silent.empty:
        return None
    fields = sorted(
        int(imaged) - int(empty)
        for imaged, empty in zip(
            silent["fovs_imaged"], silent["fovs_without_observations"], strict=True
        )
    )
    rates = [float(r) for r in silent["null_rate"].dropna()]
    where = "" if groups == 1 else f" in {_among(len(silent), groups)}"
    rate = ""
    if rates:
        low, high = percent(min(rates)), percent(max(rates))
        shown = low if low == high else f"{low} to {high}"
        rate = (
            f": this definition of new calls {shown} of the fields of a homogeneous "
            "population of that size new"
        )
    size = f"{fields[0]}" if fields[0] == fields[-1] else f"{fields[0]} to {fields[-1]}"
    return (
        f"One more field: not informative{where} with {size} fields in the declared features"
        f"{rate}, so no chance of a new state and no number of fields still needed is given."
    )


def _among(k: int, n: int) -> str:
    return "the group" if n == 1 else (f"all {n} groups" if k == n else f"{k} of {n} groups")


__all__ = [
    "is_uninformative",
    "mark_uninformative",
    "split_uninformative",
    "uninformative_sentence",
]
