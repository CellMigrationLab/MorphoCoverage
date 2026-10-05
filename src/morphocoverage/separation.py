"""How far the conditions separate in a feature space: the neighbours that share a condition.

A descriptive score, with no p-value (owner, 2026-10-05), to read each map and to compare
feature tables side by side (the explainable features, a DINOv2 and a DINOv3 embedding, and
their weighted combined space): per biological repeat, each observation's ``k`` nearest
neighbours among the other observations of its repeat (``k`` = ``K``, or fewer in a repeat
of fewer observations), the share of them that have its condition, against the share
expected by chance, that condition's share of the other observations of the repeat. The
repeat's values are the means over its observations, and a space is summed up by their
medians over the repeats, so a batch between repeats cannot pass for a separation:

    "DINOv2: 61% of neighbours share the condition, against 13% by chance; median over 3
    repeats."

The features of a space are robustly standardised (``embedding``'s median and MAD, a
feature with one value left out), then multiplied by their weights when the space has
them (``feature_tables.table_weights``); distances are Euclidean. An observation missing a
value of the space is left out (counted); a repeat holding one condition says nothing and
is left out (counted). Above ``MAX_POINTS`` observations a repeat is scored on a seeded
sample of the same share of every condition. A share far above chance says the conditions
sit apart in that space, by biology or by a difference in how the conditions were imaged;
near chance, the space does not tell them apart here, which is not proof that they are
alike.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ._checks import as_count, check_seed
from ._seeds import group_rng
from ._style import table
from ._wording import count, percent
from .embedding import _sample, _standardised, weighted
from .metadata import CONDITION_COLUMN, REPEAT_COLUMN
from .schema import fov_positions, validate_input_tables

K = 10
MAX_POINTS = 2000
_COLUMNS = ["space", "repeat", "observations", "k", "share", "chance"]


@dataclass(frozen=True)
class Separation:
    """``table``: one row per space and biological repeat scored (``observations``, ``k``,
    the mean ``share`` of neighbours sharing the condition and its ``chance``); ``spaces``:
    one row per space, the medians over its repeats (``share``, ``chance``, ``repeats``,
    ``left_out``: the observations missing a value, ``single``: the repeats of one
    condition); ``settings``."""

    table: pd.DataFrame
    spaces: pd.DataFrame
    settings: dict[str, Any]

    def line(self, space: str) -> str:
        """``space`` in one sentence (see the module), or why it was not scored."""
        rows = self.spaces[self.spaces["space"] == space]
        if not len(rows) or not rows.iloc[0]["repeats"]:
            return f"{space}: not scored, as no biological repeat holds two conditions."
        row = rows.iloc[0]
        over = (
            f"median over {count(int(row['repeats']), 'repeat')}"
            if row["repeats"] > 1
            else "in its one repeat"
        )
        return (
            f"{space}: {percent(row['share'])} of neighbours share the condition, against "
            f"{percent(row['chance'])} by chance; {over}."
        )

    def comparison(self, indent: int = 0) -> str:
        """The spaces side by side, the most separating first."""
        rows = [
            [r.space, percent(r.share), percent(r.chance), int(r.repeats)]
            for r in self.spaces.sort_values("share", ascending=False).itertuples()
            if r.repeats
        ]
        return table(
            rows, ["features", "neighbours", "by chance", "repeats"], indent=indent, align="lrrr"
        )


def _neighbour_shares(points: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    """Per observation, the share of its ``k`` nearest other observations with its label."""
    from sklearn.neighbors import NearestNeighbors

    nearest = (
        NearestNeighbors(n_neighbors=k + 1, algorithm="brute")
        .fit(points)
        .kneighbors(points, return_distance=False)
    )
    rows = np.arange(len(points))
    # Each observation's own row out, wherever the search put it (an exact duplicate can come
    # first); without it among the k + 1, the farthest.
    own = nearest == rows[:, None]
    drop = np.where(own.any(axis=1), own.argmax(axis=1), k)
    keep = np.ones_like(nearest, dtype=bool)
    keep[rows, drop] = False
    others = nearest[keep].reshape(len(points), k)
    return (labels[others] == labels[:, None]).mean(axis=1)


def _repeat_rows(
    name: str,
    z: np.ndarray,
    meta: pd.DataFrame,
    k: int,
    max_points: int,
    random_state: int,
) -> tuple[list[dict[str, Any]], int]:
    """The rows of one space, a repeat each, and the repeats left out (one condition)."""
    repeats = meta[REPEAT_COLUMN] if REPEAT_COLUMN in meta else pd.Series("all", index=meta.index)
    conditions = meta[CONDITION_COLUMN].astype(str).to_numpy()
    rows, single = [], 0
    for repeat in pd.unique(repeats):
        members = np.flatnonzero((repeats == repeat).to_numpy())
        labels = conditions[members]
        if len(set(labels)) < 2:
            single += 1
            continue
        rng = group_rng(random_state, {REPEAT_COLUMN: repeat})
        chosen = members[_sample(labels, max_points, rng)]
        labels = conditions[chosen]
        n = len(chosen)
        k_here = min(k, n - 1)
        shares = _neighbour_shares(z[chosen], labels, k_here)
        sizes = pd.Series(labels).value_counts()
        chance = (sizes.reindex(labels).to_numpy() - 1) / (n - 1)
        rows.append(
            {
                "space": name,
                "repeat": repeat,
                "observations": n,
                "k": k_here,
                "share": float(shares.mean()),
                "chance": float(chance.mean()),
            }
        )
    return rows, single


def separation(
    objects: pd.DataFrame,
    fovs: pd.DataFrame,
    *,
    spaces: Mapping[str, Sequence[str]],
    weights: Mapping[str, Mapping[str, float] | None] | None = None,
    k: int = K,
    max_points: int = MAX_POINTS,
    random_state: int = 0,
    fov_col: str | None = None,
    group_cols: Sequence[str] | None = None,
    observation_unit: str | None = None,
) -> Separation:
    """The separation of the conditions in each of ``spaces`` (a name and its features; see
    the module). ``weights`` gives a space's weight per feature (the combined space)."""
    k = as_count(k, "k", 1)
    max_points = as_count(max_points, "max_points", 2)
    random_state = check_seed(random_state)
    rows, summaries = [], []
    for name, features in spaces.items():
        data = validate_input_tables(
            objects,
            fovs=fovs,
            feature_cols=list(dict.fromkeys(features)),
            fov_col=fov_col,
            group_cols=group_cols,
            observation_unit=observation_unit,
        )
        meta = data.fovs.iloc[fov_positions(data, data.fovs)].reset_index(drop=True)
        matrix = data.feature_matrix()
        complete = ~np.isnan(matrix).any(axis=1)
        found, single = [], 0
        if CONDITION_COLUMN in meta and complete.sum() > 1:
            z, varies = _standardised(matrix[complete])
            kept = [f for f, v in zip(data.feature_cols, varies, strict=True) if v]
            z = weighted(z, kept, (weights or {}).get(name))
            if kept:
                found, single = _repeat_rows(
                    name, z, meta.loc[complete].reset_index(drop=True), k, max_points, random_state
                )
        rows += found
        scored = pd.DataFrame(found, columns=_COLUMNS)
        summaries.append(
            {
                "space": name,
                "features": len(data.feature_cols),
                "share": float(scored["share"].median()) if len(scored) else np.nan,
                "chance": float(scored["chance"].median()) if len(scored) else np.nan,
                "repeats": len(scored),
                "left_out": int((~complete).sum()),
                "single": single,
            }
        )
    settings = {"k": k, "max_points": max_points, "random_state": random_state}
    return Separation(pd.DataFrame(rows, columns=_COLUMNS), pd.DataFrame(summaries), settings)


__all__ = ["MAX_POINTS", "K", "Separation", "separation"]
