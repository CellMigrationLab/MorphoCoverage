"""The control means behind the mean-precision rule's reference (``object_rules``): the
control objects pooled per repeat, and the caution when the repeats' control means differ by
more than ``REFERENCE_SHIFT``, compared within each sub-experiment.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .._identity import IdentityKey, display, identity_key
from .._seeds import group_key
from ..metadata import CONDITION_COLUMN
from ..schema import ValidatedInput
from ._experiments import repeat_experiments

# Repeat control means differing by more than this factor are flagged (D3).
REFERENCE_SHIFT = 1.5


def _repeats_of(data: ValidatedInput, repeat_col: str | None) -> list[Any]:
    """The repeats a group spans (``[None]`` without a repeat column)."""
    return data.fovs[repeat_col].drop_duplicates().tolist() if repeat_col else [None]


def _pooled_by_repeat(
    all_groups: list[tuple[dict[str, Any], ValidatedInput]],
    chosen: set[Any],
    values_by_group: dict[Any, np.ndarray],
    repeat_col: str | None,
) -> dict[IdentityKey, np.ndarray]:
    """Every control object of each repeat, the control groups ``chosen`` pooled."""
    by_repeat: dict[IdentityKey, list[np.ndarray]] = {}
    for group, data in all_groups:
        if group_key(group) in chosen:
            for repeat in _repeats_of(data, repeat_col):
                by_repeat.setdefault(identity_key(repeat), []).append(
                    values_by_group[group_key(group)]
                )
    return {r: np.concatenate(v) for r, v in by_repeat.items()}


def _repeat_caution(
    all_groups: list[tuple[dict[str, Any], ValidatedInput]],
    chosen: set[Any],
    pooled: dict[IdentityKey, np.ndarray],
    repeat_col: str | None,
) -> str | None:
    """The caution of the largest shift between the repeats' control means within a
    sub-experiment, or ``None`` when none exceeds ``REFERENCE_SHIFT``."""
    names = display(list(pooled))
    # Control means are compared within a sub-experiment only: two sets of a screen, each
    # with its own control and repeats, are not one control drifting (owner, 2026-10-02).
    block = (
        repeat_experiments([g for g, _ in all_groups], chosen, CONDITION_COLUMN, repeat_col)
        if repeat_col
        else {}
    )
    of_name = {
        identity_key(r): block.get(r) for _, d in all_groups for r in _repeats_of(d, repeat_col)
    }
    # A repeat holding only the control belongs to no sub-experiment: compared with all.
    blocks: dict[int | None, dict[str, np.ndarray]] = {b: {} for b in set(block.values())}
    blocks = blocks or {None: {}}
    for r, values in pooled.items():
        for b in blocks if of_name.get(r) is None else [of_name[r]]:
            blocks[b][names[r]] = values
    shifts = [shift for shift in map(_shift, blocks.values()) if shift is not None]
    return max(shifts)[1] if shifts else None


def _shift(pooled: dict[str, np.ndarray]) -> tuple[float, str] | None:
    """The ratio and a caution when the repeats' control means differ by more than
    ``REFERENCE_SHIFT``; ``None`` below it."""
    levels = {r: float(v.mean()) for r, v in pooled.items() if len(v)}
    if len(levels) < 2 or min(levels.values()) <= 0:
        return None
    low, high = min(levels, key=levels.get), max(levels, key=levels.get)
    ratio = levels[high] / levels[low]
    if ratio <= REFERENCE_SHIFT:
        return None
    return ratio, (
        f"the control means of the repeats differ by {ratio:.1f}x (repeat {low}: "
        f"{levels[low]:.3g}, repeat {high}: {levels[high]:.3g}); each repeat is judged "
        "against its own control, but a shift that large may be a batch effect or a change "
        "of units (a feature exported in other units in one repeat)"
    )
