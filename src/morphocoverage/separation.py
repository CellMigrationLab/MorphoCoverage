"""How far the conditions, and the repeats, separate in a feature space: the neighbours that
share them.

A descriptive score, with no p-value (owner, 2026-10-05), to read each map and to compare
feature tables side by side (the explainable features, a DINOv2 and a DINOv3 embedding, and
their weighted combined space):

- **by condition**: per biological repeat, each observation's ``k`` nearest neighbours among
  the other observations of its repeat, the share of them that have its condition, against
  the share expected by chance: that condition's share of the observations it could have as
  neighbours. The repeat's values are the means over its observations, and a space is summed
  up by their medians over the repeats, so a batch between repeats cannot pass for a
  separation;
- **by repeat** (owner, 2026-10-06): per condition, the same with the repeat: the share of
  each observation's neighbours, among the other observations of its condition, that share
  its biological repeat, against chance; the median over the conditions. When the repeats
  separate more than the conditions in a space, the line says so: a batch effect between
  repeats is then larger there than the difference between conditions.

Neighbours never come from the observation's own field, nor from its own technical repeat
(well) when its condition holds another well in that repeat (owner, 2026-10-06): cells of one
field or well resemble each other, and counted as neighbours they read as a separation of
conditions or repeats that only the field or the well shows. Chance is computed over the
observations that remain. ``k`` is ``K``, or fewer in a group where some observation has
fewer than ``K`` possible neighbours with its own label outside its field (well): the
smallest number any has, so that a perfect separation reads 100% (owner, 2026-10-06). An
observation without a possible neighbour of its own label (its condition imaged in one field
of the repeat) is left out of the score.

"Separate more" compares how far each score goes from chance towards 100%: ``(share -
chance) / (1 - chance)``; the repeats separate more when theirs is larger and at least
``NOTABLE``, so two scores at chance are never read as a batch.

    "DINOv2: 61% of neighbours share the condition, against 13% by chance; median over 3
    repeats. Within a condition, 35% share the repeat, against 33% by chance."

The features of a space are robustly standardised (``embedding``'s median and MAD, a
feature with one value left out), then weighted when the space is the combined one
(``blocks``: each table divided by its first singular value, ``feature_tables.mfa_weights``)
or has fixed ``weights``; distances are Euclidean. An observation missing a value of the
space is left out (counted); a repeat holding one condition says nothing and is left out
(counted). Above ``MAX_POINTS`` observations a group is scored on a seeded sample of the
same share of every label. A share far above chance says the conditions sit apart in that
space, by biology or by a difference in how the conditions were imaged; near chance, the
space does not tell them apart here, which is not proof that they are alike.
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
from .embedding import _sample, _standardised, combined
from .feature_tables import COMBINED
from .metadata import CONDITION_COLUMN, REPEAT_COLUMN, TECHNICAL_REPEAT_COLUMN
from .schema import fov_positions, unit_words, validate_input_tables

K = 10
MAX_POINTS = 2000
# How far towards 100% from chance the repeats' score must go to be said as separating.
NOTABLE = 0.1
_COLUMNS = ["space", "score", "group", "observations", "k", "share", "chance"]


def lift(share: float, chance: float) -> float:
    """How far ``share`` goes from ``chance`` towards 100%: 0 at chance, 1 when perfect."""
    if not np.isfinite(share) or not np.isfinite(chance) or chance >= 1:
        return np.nan
    return float((share - chance) / (1 - chance))


@dataclass(frozen=True)
class Separation:
    """``table``: one row per space, score (``condition``: within a biological repeat, the
    ``group``; ``repeat``: within a condition) and group scored (``observations``, ``k``, the
    mean ``share`` of neighbours sharing the label and its ``chance``); ``spaces``: one row
    per space, the medians over its groups (``share``, ``chance`` and ``repeats`` by
    condition; ``repeat_share``, ``repeat_chance`` and ``conditions`` by repeat; ``left_out``:
    the observations missing a value, ``single``: the repeats of one condition);
    ``settings``."""

    table: pd.DataFrame
    spaces: pd.DataFrame
    settings: dict[str, Any]

    def repeats_first(self, space: str) -> bool:
        """Whether the repeats separate more than the conditions in ``space`` (see the
        module)."""
        rows = self.spaces[self.spaces["space"] == space]
        if not len(rows):
            return False
        row = rows.iloc[0]
        by_repeat = lift(row["repeat_share"], row["repeat_chance"])
        by_condition = lift(row["share"], row["chance"])
        if not np.isfinite(by_repeat) or by_repeat < NOTABLE:
            return False
        return not np.isfinite(by_condition) or by_repeat > by_condition

    def line(self, space: str) -> str:
        """``space`` in one or two sentences (see the module), or why it was not scored."""
        rows = self.spaces[self.spaces["space"] == space]
        if not len(rows) or not rows.iloc[0]["repeats"]:
            return f"{space}: not scored, as no biological repeat holds two conditions."
        row = rows.iloc[0]
        over = (
            f"median over {count(int(row['repeats']), 'repeat')}"
            if row["repeats"] > 1
            else "in its one repeat"
        )
        text = (
            f"{space}: {percent(row['share'])} of neighbours share the condition, against "
            f"{percent(row['chance'])} by chance; {over}."
        )
        if row["conditions"]:
            text += (
                f" Within a condition, {percent(row['repeat_share'])} share the repeat, "
                f"against {percent(row['repeat_chance'])} by chance."
            )
        if self.repeats_first(space):
            text += " Here the repeats separate more than the conditions."
        if row["left_out"]:
            unit = self.settings.get("observation_unit", "observation")
            text += f" {count(int(row['left_out']), unit)} with a missing value left out."
        return text

    def ranked(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """The spaces scored, the most separating first, and the combined space apart: it
        joins the others, so it is not ranked among them."""
        scored = self.spaces[self.spaces["repeats"] > 0]
        apart = scored["space"].astype(str).str.casefold() == COMBINED
        return scored[~apart].sort_values("share", ascending=False), scored[apart]

    def comparison(self, indent: int = 0) -> str:
        """The feature tables side by side, the most separating first, the combined space
        last and apart."""
        ranked, apart = self.ranked()
        rows = [
            [
                r.space,
                percent(r.share),
                percent(r.chance),
                int(r.repeats),
                percent(r.repeat_share) if r.conditions else "–",
                percent(r.repeat_chance) if r.conditions else "–",
            ]
            for frame in (ranked, apart)
            for r in frame.itertuples()
        ]
        columns = ["feature table", "condition", "by chance", "repeats", "repeat", "by chance"]
        return table(rows, columns, indent=indent, align="lrrrrr")


def _blocks(meta: pd.DataFrame, fields: np.ndarray) -> np.ndarray:
    """Each observation's own block, whose observations are never its neighbours: its
    well when its condition holds another well in its repeat, else its field."""
    field_ids = pd.Series(fields.astype(str), index=meta.index)
    if TECHNICAL_REPEAT_COLUMN not in meta or meta[TECHNICAL_REPEAT_COLUMN].isna().all():
        return field_ids.to_numpy()
    group = [c for c in (CONDITION_COLUMN, REPEAT_COLUMN) if c in meta]
    keys = meta[[*group, TECHNICAL_REPEAT_COLUMN]].astype(str).agg("\x1f".join, axis=1)
    if not group:
        return keys.to_numpy()
    wells = keys.groupby([meta[c].astype(str) for c in group]).transform("nunique")
    return np.where(wells.to_numpy() > 1, "well\x1f" + keys, "field\x1f" + field_ids)


def _scores(
    points: np.ndarray, labels: np.ndarray, blocks: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray, int] | None:
    """Per observation that has a possible neighbour of its own label outside its block: the
    share of its ``k`` nearest such candidates with its label, and that share by chance; and
    the ``k`` used (see the module). ``None`` when no observation has one."""
    candidates = blocks[:, None] != blocks[None, :]
    same = labels[:, None] == labels[None, :]
    own = (candidates & same).sum(axis=1)
    pool = candidates.sum(axis=1)
    scored = np.flatnonzero(own > 0)
    if not len(scored):
        return None
    k = int(min(k, own[scored].min()))
    squares = np.square(points).sum(axis=1)
    distance = squares[scored, None] + squares[None, :] - 2 * points[scored] @ points.T
    distance[~candidates[scored]] = np.inf
    nearest = np.argpartition(distance, k - 1, axis=1)[:, :k]
    shares = same[scored[:, None], nearest].mean(axis=1)
    return shares, own[scored] / pool[scored], k


def _group_rows(
    name: str,
    score: str,
    z: np.ndarray,
    meta: pd.DataFrame,
    blocks: np.ndarray,
    settings: tuple[int, int, int],
) -> tuple[list[dict[str, Any]], int]:
    """The rows of one score of one space (``condition``: per repeat; ``repeat``: per
    condition), and the groups of one label left out."""
    k, max_points, random_state = settings
    within, label = (
        (REPEAT_COLUMN, CONDITION_COLUMN)
        if score == "condition"
        else (CONDITION_COLUMN, REPEAT_COLUMN)
    )
    groups = meta[within] if within in meta else pd.Series("all", index=meta.index)
    labels_all = meta[label].astype(str).to_numpy()
    rows, single = [], 0
    for group in pd.unique(groups):
        members = np.flatnonzero((groups == group).to_numpy())
        if len(set(labels_all[members])) < 2:
            single += 1
            continue
        rng = group_rng(random_state, {within: group})
        chosen = members[_sample(labels_all[members], max_points, rng)]
        found = _scores(z[chosen], labels_all[chosen], blocks[chosen], k)
        if found is None:
            continue
        shares, chance, k_here = found
        rows.append(
            {
                "space": name,
                "score": score,
                "group": group,
                "observations": len(shares),
                "k": k_here,
                "share": float(shares.mean()),
                "chance": float(chance.mean()),
            }
        )
    return rows, single


def _median(rows: pd.DataFrame, column: str) -> float:
    return float(rows[column].median()) if len(rows) else np.nan


def separation(
    objects: pd.DataFrame,
    fovs: pd.DataFrame,
    *,
    spaces: Mapping[str, Sequence[str]],
    weights: Mapping[str, Mapping[str, float] | None] | None = None,
    blocks: Mapping[str, Mapping[str, Sequence[str]] | None] | None = None,
    k: int = K,
    max_points: int = MAX_POINTS,
    random_state: int = 0,
    fov_col: str | None = None,
    group_cols: Sequence[str] | None = None,
    observation_unit: str | None = None,
) -> Separation:
    """The separation of the conditions and of the repeats in each of ``spaces`` (a name and
    its features; see the module). ``weights`` gives a space's fixed weight per feature,
    ``blocks`` its tables, weighted as the combined space weighs them."""
    k = as_count(k, "k", 1)
    max_points = as_count(max_points, "max_points", 2)
    random_state = check_seed(random_state)
    rows, summaries, unit = [], [], "observation"
    for name, features in spaces.items():
        data = validate_input_tables(
            objects,
            fovs=fovs,
            feature_cols=list(dict.fromkeys(features)),
            fov_col=fov_col,
            group_cols=group_cols,
            observation_unit=observation_unit,
        )
        unit = unit_words(data.observation_unit, data.observation_unit_declared)[0]
        positions = fov_positions(data, data.fovs)
        meta = data.fovs.iloc[positions].reset_index(drop=True)
        matrix = data.feature_matrix()
        complete = ~np.isnan(matrix).any(axis=1)
        found: list[dict[str, Any]] = []
        single = 0
        if CONDITION_COLUMN in meta and complete.sum() > 1:
            z, varies = _standardised(matrix[complete])
            kept = [f for f, v in zip(data.feature_cols, varies, strict=True) if v]
            z = combined(z, kept, (weights or {}).get(name), (blocks or {}).get(name))
            if kept:
                here = meta.loc[complete].reset_index(drop=True)
                own = _blocks(here, positions[complete])
                settings = (k, max_points, random_state)
                found, single = _group_rows(name, "condition", z, here, own, settings)
                if REPEAT_COLUMN in here:
                    found += _group_rows(name, "repeat", z, here, own, settings)[0]
        rows += found
        scored = pd.DataFrame(found, columns=_COLUMNS)
        by_condition = scored[scored["score"] == "condition"]
        by_repeat = scored[scored["score"] == "repeat"]
        summaries.append(
            {
                "space": name,
                "features": len(data.feature_cols),
                "share": _median(by_condition, "share"),
                "chance": _median(by_condition, "chance"),
                "repeats": len(by_condition),
                "repeat_share": _median(by_repeat, "share"),
                "repeat_chance": _median(by_repeat, "chance"),
                "conditions": len(by_repeat),
                "left_out": int((~complete).sum()),
                "single": single,
            }
        )
    settings = {
        "k": k,
        "max_points": max_points,
        "random_state": random_state,
        "observation_unit": unit,
    }
    return Separation(pd.DataFrame(rows, columns=_COLUMNS), pd.DataFrame(summaries), settings)


__all__ = ["MAX_POINTS", "NOTABLE", "K", "Separation", "lift", "separation"]
