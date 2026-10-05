"""The secondary rows of the contrast table, compared across experiments (owner,
2026-10-05).

With several declared controls the contrast table (``_control.CONTRAST_COLUMNS``) holds,
beside the main verdicts (each condition against its own control), two kinds of secondary
row, each tested on the repeat means, one value per biological repeat (no cell-level test:
the repeat is the replicate):

- ``vs_master_raw``: a control against the master control on the raw values, paired across
  the repeats holding both, each repeat's ``mean(control) - mean(master)``. It assumes the
  experiments were imaged and processed alike. Labelled ``THBS5_SiNC vs VEC_SiNC (raw)``.
- ``across_normalised``: two conditions of different experiments sharing a treatment, each
  normalised to its own control, paired across the repeats holding all four: each repeat's
  ``log(mean(a) / mean(control of a)) - log(mean(b) / mean(control of b))``, the log ratio
  the paired test reads against a control (``statistics.repeat_values``). Labelled
  ``SiCol6a1 effect: THBS5_SiCol6a1/THBS5_SiNC vs VEC_SiCol6a1/VEC_SiNC``.

Where a mean is not positive (a signed feature) the log ratio has no meaning: the effect is
then the raw difference (of differences), and ``values`` says ``raw`` for that feature.
Across the repeats: the paired t-test and, where its rule allows (``MIN_REPEATS`` repeats
and a floor that can reach 0.05 once corrected, ``_reach``), the exact sign-flip test, each
row's ``family`` corrected as one Benjamini-Hochberg family. A row held in fewer than
``MIN_REPEATS`` repeats is not compared (``status``), and the report names it once.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from .._control import ACROSS, MASTER_RAW
from .._seeds import group_rng
from ..statistics import benjamini_hochberg, paired_t
from ._reach import reachable
from ._units import side_units
from .contrast_repeats import MIN_REPEATS, sign_flip_test

RAW, NORMALISED = "raw", "normalised"
ACROSS_COLUMNS = [
    "comparison",
    "kind",
    "condition",
    "reference",
    "condition_control",
    "reference_control",
    "family",
    "repeats",
    "held",
    "values",
    "effects",
    "effect",
    "fold_change",
    "lower",
    "higher",
    "p_paired_t",
    "q_paired_t",
    "p_sign_flip",
    "q_sign_flip",
    "p_sign_flip_floor",
    "p_sign_flip_exact",
    "status",
]
NOT_COMPARED = f"fewer than {MIN_REPEATS} biological repeats hold"


def condition_means(
    all_groups: list,
    described: Sequence[Mapping[str, Any]],
    feature: str,
    technical_col: str | None,
    conditions: Sequence[Any],
) -> dict[tuple[Any, Any], float]:
    """Each ``(condition, repeat)`` mean of the observations of ``conditions`` (their
    analysis groups' fields pooled, as ``condition_contrast`` pools a side)."""
    chosen: dict[tuple[Any, Any], list] = {}
    for d in described:
        if d["condition"] in conditions:
            chosen.setdefault((d["condition"], d["repeat"]), []).append(d["group"])
    means = {}
    for key, members in chosen.items():
        units, _ = side_units(all_groups, feature, members, technical_col)
        fields = [f for u in units for f in u]
        values = np.concatenate(fields) if fields else np.array([])
        means[key] = float(values.mean()) if len(values) else np.nan
    return means


def _effects(kind: str, sides: list[np.ndarray]) -> tuple[str, np.ndarray, float]:
    """``(values, effects, fold change)``: per repeat, on the log scale when every mean is
    positive, else the raw difference (of differences); the fold change on the log scale."""
    means = np.array(sides, dtype=float)
    positive = means.size > 0 and bool((means > 0).all())
    if kind == MASTER_RAW:
        effects = means[0] - means[1]
        fold = float(np.exp(np.log(means[0] / means[1]).mean())) if positive else np.nan
        return RAW, effects, fold
    if positive:
        effects = np.log(means[0] / means[1]) - np.log(means[2] / means[3])
        return NORMALISED, effects, float(np.exp(effects.mean()))
    return RAW, (means[0] - means[1]) - (means[2] - means[3]), np.nan


def _sides(row: Mapping[str, Any]) -> tuple[Any, ...]:
    """The conditions whose repeat means a row compares, in the order of ``_effects``."""
    if row["kind"] == MASTER_RAW:
        return row["condition"], row["reference"]
    return row["condition"], row["condition_control"], row["reference"], row["reference_control"]


def _row(
    entry: Mapping[str, Any],
    means: Mapping[tuple[Any, Any], float],
    repeats: Sequence[Any],
    random_state: int,
) -> dict[str, Any]:
    sides = _sides(entry)
    held = [r for r in repeats if all(np.isfinite(means.get((c, r), np.nan)) for c in sides)]
    if held:
        values, effects, fold = _effects(
            entry["kind"], [[means[(c, r)] for r in held] for c in sides]
        )
    else:
        values, effects, fold = NORMALISED, np.array([]), np.nan
    row: dict[str, Any] = {
        "comparison": entry["label"],
        **{c: entry[c] for c in ("kind", "condition", "reference", "family")},
        "condition_control": entry["condition_control"],
        "reference_control": entry["reference_control"],
        "repeats": len(held),
        "held": list(held),
        "values": values,
        "effects": effects,
        "effect": float(effects.mean()) if len(effects) else np.nan,
        "fold_change": fold,
        "lower": int((effects < 0).sum()),
        "higher": int((effects > 0).sum()),
        "p_paired_t": np.nan,
        "q_paired_t": np.nan,
        "p_sign_flip": np.nan,
        "q_sign_flip": np.nan,
        "p_sign_flip_floor": np.nan,
        "p_sign_flip_exact": False,
        "status": "ok",
    }
    if len(held) < MIN_REPEATS:
        row["status"] = f"{NOT_COMPARED} {'both' if entry['kind'] == MASTER_RAW else 'all four'}"
        return row
    rng = group_rng(random_state, {"across": entry["label"]})
    p, floor, exact = sign_flip_test(effects, rng)
    row.update(p_paired_t=paired_t(effects), p_sign_flip=p, p_sign_flip_floor=floor)
    row["p_sign_flip_exact"] = exact
    return row


def across_rows(
    means: Mapping[tuple[Any, Any], float],
    rows: pd.DataFrame,
    repeats: Sequence[Any],
    random_state: int,
) -> pd.DataFrame:
    """One tested row per secondary row of a contrast table (``vs_master_raw``,
    ``across_normalised``), from each ``(condition, repeat)`` mean, in the order of
    ``repeats``; each ``family`` corrected as one Benjamini-Hochberg family."""
    secondary = rows[rows["kind"].isin([MASTER_RAW, ACROSS])]
    tested = [_row(entry, means, repeats, random_state) for entry in secondary.to_dict("records")]
    table = pd.DataFrame(tested, columns=ACROSS_COLUMNS)
    for family in pd.unique(table["family"]):
        mine = table["family"] == family
        floors = table.loc[mine, "p_sign_flip_floor"].where(table.loc[mine, "p_sign_flip"].notna())
        kept = reachable(floors)
        table.loc[mine & ~table.index.isin(kept[kept].index), "p_sign_flip"] = np.nan
        for column in ("p_paired_t", "p_sign_flip"):
            values = table.loc[mine, column].to_numpy(float)
            table.loc[mine, column.replace("p_", "q_", 1)] = benjamini_hochberg(values)
    return table


def across_table(
    sides: tuple[list, Sequence[Mapping[str, Any]], str, str | None],
    rows: pd.DataFrame | None,
    random_state: int,
) -> pd.DataFrame:
    """``condition_contrast``'s secondary rows tested; ``sides`` is its ``(all_groups,
    described, feature, technical_col)``. Empty without such rows or repeats. ``attrs`` holds
    the contrast table (``contrasts``) and, with ``vs_master_raw`` rows, the master control's
    repeats and the repeats of the data (``master_repeats``)."""
    all_groups, described, feature, technical_col = sides
    table = pd.DataFrame(columns=ACROSS_COLUMNS)
    order = list(dict.fromkeys(d["repeat"] for d in described))
    secondary = (
        rows[rows["kind"].isin([MASTER_RAW, ACROSS])] if rows is not None else pd.DataFrame()
    )
    if len(secondary) and any(r is not None for r in order):
        conditions = {c for entry in secondary.to_dict("records") for c in _sides(entry)}
        means = condition_means(all_groups, described, feature, technical_col, conditions)
        table = across_rows(means, secondary, order, random_state)
    table.attrs["contrasts"] = rows
    masters = secondary.loc[secondary["kind"] == MASTER_RAW, "reference"] if len(secondary) else []
    if len(masters):
        master = masters.iloc[0]
        held = [
            r
            for r in order
            if any(d["condition"] == master and d["repeat"] == r for d in described)
        ]
        table.attrs["master_repeats"] = (master, held, order)
    return table


__all__ = [
    "ACROSS_COLUMNS",
    "NORMALISED",
    "NOT_COMPARED",
    "RAW",
    "across_rows",
    "across_table",
    "condition_means",
]
