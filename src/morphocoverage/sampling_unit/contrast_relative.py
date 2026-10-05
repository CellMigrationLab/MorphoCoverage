"""Two conditions of different experiments, each normalised to its own reference (owner,
2026-10-05).

A condition ``a`` imaged with its reference ``ref_a`` and a condition ``b`` with its own
``ref_b`` are compared on the values normalised to their own reference, biological repeat by
biological repeat, in the repeats that hold all four: the effect of one repeat is
``log(mean(a) / mean(ref_a)) - log(mean(b) / mean(ref_b))``, the same log ratio the paired
test across repeats reads for a condition against its control (``statistics.repeat_values``).
When a mean is not positive the log ratio has no meaning and the effect is the raw difference
of differences, ``(mean(a) - mean(ref_a)) - (mean(b) - mean(ref_b))``, as the paired test
does, and the row says so (``values``). Across the repeats: the paired t-test on these
effects and, where its rule allows (``MIN_REPEATS`` repeats, and a floor that can reach 0.05
once corrected), the exact sign-flip test; Benjamini-Hochberg in the same family as the
comparisons it builds on (``shared_correction``). There is no cell-level test: the repeat is
the replicate. Each row is labelled ``a/ref_a vs b/ref_b`` (``relative_label``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from .._seeds import group_rng
from ..metadata import REPEAT_COLUMN
from ..statistics import paired_t
from ._experiments import corrected, experiments
from ._reach import reachable
from ._units import side_units
from .contrast_repeats import MIN_REPEATS, sign_flip_test

RELATIVE_COLUMNS = [
    "comparison",
    "condition",
    "reference",
    "other",
    "other_reference",
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
    "experiment",
    "status",
]


def relative_label(condition: Any, reference: Any, other: Any, other_reference: Any) -> str:
    """``THBS5_SiCol6a1/THBS5_SiNC vs VEC_SiCol6a1/VEC_SiNC``: each condition over its own
    reference."""
    return f"{condition}/{reference} vs {other}/{other_reference}"


def check_relative(relative: Any, name: str = "relative") -> tuple[tuple[Any, ...], ...]:
    """``relative`` as a tuple of ``(a, ref_a, b, ref_b)``; ``TypeError`` or ``ValueError``
    for anything else, a condition used as its own reference, or one tuple twice."""
    if isinstance(relative, (str, Mapping)) or not isinstance(relative, Sequence):
        raise TypeError(f"{name} must be a sequence of (a, ref_a, b, ref_b) tuples")
    tuples = []
    for entry in relative:
        if isinstance(entry, str) or not isinstance(entry, Sequence) or len(entry) != 4:
            raise TypeError(f"{name} must hold (a, ref_a, b, ref_b) tuples, got {entry!r}")
        if any(not pd.api.types.is_scalar(c) for c in entry):
            raise TypeError(f"{name} must name each condition by one label, got {entry!r}")
        a, ref_a, b, ref_b = entry
        if a == ref_a or b == ref_b or (a, ref_a) == (b, ref_b):
            raise ValueError(f"{name} compares a condition with itself in {tuple(entry)!r}")
        tuples.append(tuple(entry))
    if len(set(tuples)) < len(tuples):
        raise ValueError(f"{name} names a tuple twice")
    return tuple(tuples)


def relative_values(
    a: np.ndarray, ref_a: np.ndarray, b: np.ndarray, ref_b: np.ndarray
) -> tuple[str, np.ndarray]:
    """``("normalised", log-ratio differences)`` when every mean is positive, else
    ``("raw", differences of differences)``: one value per repeat."""
    means = np.array([a, ref_a, b, ref_b], dtype=float)
    if means.size and (means > 0).all():
        return "normalised", np.log(means[0] / means[1]) - np.log(means[2] / means[3])
    return "raw", (means[0] - means[1]) - (means[2] - means[3])


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


def relative_rows(
    means: Mapping[tuple[Any, Any], float],
    relative: Sequence[tuple[Any, ...]],
    repeats: Sequence[Any],
    random_state: int,
) -> pd.DataFrame:
    """One row per tuple of ``relative`` (see the module), from each ``(condition, repeat)``
    mean of the observations, in the order of ``repeats``; the q-values and the family are
    set by ``shared_correction``."""
    rows = []
    for entry in relative:
        held = [r for r in repeats if all(np.isfinite(means.get((c, r), np.nan)) for c in entry)]
        sides = [np.array([means[(c, r)] for r in held]) for c in entry]
        values, effects = relative_values(*sides) if held else ("normalised", np.array([]))
        label = relative_label(*entry)
        row: dict[str, Any] = {
            "comparison": label,
            "condition": entry[0],
            "reference": entry[1],
            "other": entry[2],
            "other_reference": entry[3],
            "repeats": len(held),
            "held": list(held),
            "values": values,
            "effects": effects,
            "effect": float(effects.mean()) if len(effects) else np.nan,
            "fold_change": float(np.exp(effects.mean()))
            if values == "normalised" and len(effects)
            else np.nan,
            "lower": int((effects < 0).sum()),
            "higher": int((effects > 0).sum()),
            "p_paired_t": np.nan,
            "q_paired_t": np.nan,
            "p_sign_flip": np.nan,
            "q_sign_flip": np.nan,
            "p_sign_flip_floor": np.nan,
            "p_sign_flip_exact": False,
            "experiment": 0,
            "status": "ok",
        }
        if len(held) < MIN_REPEATS:
            row["status"] = f"fewer than {MIN_REPEATS} biological repeats hold all four conditions"
        else:
            rng = group_rng(random_state, {"relative": label})
            p, floor, exact = sign_flip_test(effects, rng)
            row.update(
                p_paired_t=paired_t(effects),
                p_sign_flip=p,
                p_sign_flip_floor=floor,
                p_sign_flip_exact=exact,
            )
        rows.append(row)
    return pd.DataFrame(rows, columns=RELATIVE_COLUMNS)


def relative_table(
    across: pd.DataFrame,
    table: pd.DataFrame,
    sides: tuple[list, Sequence[Mapping[str, Any]], str, str | None],
    relative: Sequence[tuple[Any, ...]],
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``condition_contrast``'s comparisons across repeats (``across``) and the ``relative``
    rows, corrected together (``shared_correction``); ``table`` is its per-repeat table and
    ``sides`` its ``(all_groups, described, feature, technical_col)``. Without relative
    tuples or a repeat column, ``across`` and an empty table."""
    empty = pd.DataFrame(columns=RELATIVE_COLUMNS)
    if not relative or table.empty or REPEAT_COLUMN not in table:
        return across, empty
    all_groups, described, feature, technical_col = sides
    if all(d["repeat"] is None for d in described):
        return across, empty
    conditions = {c for entry in relative for c in entry}
    means = condition_means(all_groups, described, feature, technical_col, conditions)
    order = list(dict.fromkeys(d["repeat"] for d in described))
    rows = relative_rows(means, relative, order, random_state)
    paired = table[table["mean"].notna() & table["control_mean"].notna()]
    held = {c: list(b[REPEAT_COLUMN]) for c, b in paired.groupby("condition", sort=False)}
    return shared_correction(across, rows, held)


def shared_correction(
    across: pd.DataFrame, relative: pd.DataFrame, held: Mapping[Any, Sequence[Any]]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The comparisons across repeats (``across``) and the ``relative`` rows corrected as one
    family per sub-experiment (``_experiments.experiments``: linked by a biological repeat
    they share; ``held`` maps each comparison of ``across`` to its repeats): each relative
    row's paired t-test beside the comparisons' normalised one, and its sign-flip test beside
    theirs, kept only where it can reach 0.05 once corrected (``_reach``). Unchanged without
    relative rows."""
    if relative.empty:
        return across, relative
    across, relative = across.copy(), relative.copy()
    family = experiments(
        {**dict(held), **{row.comparison: row.held for row in relative.itertuples()}}
    )
    across["experiment"] = across["condition"].map(family)
    relative["experiment"] = relative["comparison"].map(family)
    p_t = pd.concat([across["p_paired_t_normalised"], relative["p_paired_t"]], ignore_index=True)
    p_flip = pd.concat([across["p_sign_flip"], relative["p_sign_flip"]], ignore_index=True)
    floors = pd.concat(
        [across["p_sign_flip_floor"], relative["p_sign_flip_floor"]], ignore_index=True
    ).where(p_flip.notna())
    families = pd.concat([across["experiment"], relative["experiment"]], ignore_index=True)
    helpful = pd.Series(False, index=p_flip.index)
    for members in families.groupby(families).groups.values():
        helpful.loc[members] = reachable(floors.loc[members])
    p_flip = p_flip.where(helpful)
    q_t, q_flip = corrected(p_t, families), corrected(p_flip, families)
    n = len(across)
    across["p_sign_flip"] = p_flip.to_numpy()[:n]
    relative["p_sign_flip"] = p_flip.to_numpy()[n:]
    across["q_paired_t_normalised"], relative["q_paired_t"] = q_t[:n], q_t[n:]
    across["q_paired_t"] = corrected(across["p_paired_t"], across["experiment"])
    across["q_sign_flip"], relative["q_sign_flip"] = q_flip[:n], q_flip[n:]
    return across, relative


__all__ = [
    "RELATIVE_COLUMNS",
    "check_relative",
    "condition_means",
    "relative_label",
    "relative_rows",
    "relative_table",
    "relative_values",
    "shared_correction",
]
