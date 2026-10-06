"""The fields that sway the result (private; ``retrospective.report``; owner, 2026-10-02).

The inspection's influence reason (``inspect_fovs``) asks whether leaving one field out
changes a summary of its analysis group in which every field weighs one. The result the
report states is the paired t-test on each biological repeat's mean of the observations
(``contrast_repeats``), every observation weighing one and the wells of a repeat pooled.
So in the report a field is influential when leaving it out moves the mean of its
condition in its repeat, as that test computes it, by at least ``threshold`` of it; the
reason then says what this does to the result: the condition's change against the control
(a fold change, or the difference of the means where a value is not positive) and the paired
t-test's p, with the field and without it. A control field (a field of the declared
control's groups) moves every condition compared with it in its repeat: the condition whose
change it moves most is named. With declared comparisons a field moves each comparison its
condition is in, as the condition or as the reference (owner, 2026-10-05).

The influence scales with the group (owner, 2026-10-05): with 3-5 noisy fields a group, leaving
almost any field out moves the mean by 10%, so a field is listed only when its change is also
more than ``INFLUENCE_SPREAD`` robust SDs from its analysis group's median change
(``inspection.stands_out``, which says why this rule), and a group of fewer than
``INFLUENCE_MIN_FIELDS`` fields is not judged, named once.

A relative change needs a positive scale: a readout with negative values (standardised,
centred), or a mean of 0, is not checked, as ``inspect_fovs`` skips it, and the report says
so once.
Without biological repeats kept apart there is no test across repeats to sway, and the
report keeps the field-weighted check (``influence_on_result``).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd

from .._control import comparison_label
from .._style import group_label
from .._wording import change, p_value, percent
from ..data_check import validate_input_tables
from ..evaluator import iter_analysis_groups
from ..metadata import CONDITION_COLUMN, REPEAT_COLUMN
from ..retrospective._text import where
from ..retrospective.inspection import (
    INFLUENCE_MIN_FIELDS,
    INFLUENCE_SPREAD,
    SMALL_GROUP,
    InspectionResult,
    field_rows,
    stands_out,
)
from ..sampling_unit.contrast_repeats import MIN_REPEATS
from ..schema import ValidatedInput, fov_positions
from ..statistics import paired_t, repeat_values

THRESHOLD = 0.10
_NEGATIVE = "has negative values, so a relative change is undefined"
_ZERO = "is 0 here, so a relative change is undefined"


def influence_on_result(contrast: Any) -> bool:
    """Whether the report's influence check is on the result: a contrast whose rows keep
    the biological repeats apart."""
    table = getattr(contrast, "table", None)
    return (
        table is not None
        and len(table) > 0
        and REPEAT_COLUMN in table
        and bool(table[REPEAT_COLUMN].notna().any())
    )


def _where(group: dict[str, Any]) -> dict[str, str]:
    """A group where a check did not run: its key, and its label as the summaries name it."""
    return {"group": where(group), "label": group_label(group)}


def _result(
    means: np.ndarray, controls: np.ndarray, values: str | None = None, log: bool | None = None
) -> tuple:
    """``(change, values, p)`` over the repeats (``statistics.repeat_values``, on the
    readout's scale ``log``); with ``values`` given, on that scale only (NaN when it no
    longer applies)."""
    found, compared, moved = repeat_values(means, controls, log)
    if values is not None and found != values:
        return np.nan, values, np.nan
    p = paired_t(compared) if len(compared) >= MIN_REPEATS else np.nan
    return moved, found, p


def _effect(
    block: pd.DataFrame, repeat: Any, column: str, value: float, log: bool | None = None
) -> tuple[float, str]:
    """How far the condition's change moves with the repeat's mean in ``column`` set to
    ``value`` (the field left out), and the change and p before and after, in words, on the
    scale of the test as acquired."""
    means = block["mean"].to_numpy(float)
    controls = block["control_mean"].to_numpy(float)
    before, values, p_before = _result(means, controls, log=log)
    moved = (means if column == "mean" else controls).copy()
    moved[(block[REPEAT_COLUMN] == repeat).to_numpy()] = value
    after, _, p_after = (
        _result(moved, controls, values, log)
        if column == "mean"
        else _result(means, moved, values, log)
    )
    if np.isnan(after):
        return np.inf, f"{change(before, values)} to one no longer defined as a fold change"
    words = f"{change(before, values)} to {change(after, values)}"
    if np.isfinite(p_before) and np.isfinite(p_after):
        words += f", the paired t-test's p from {p_value(p_before)} to {p_value(p_after)}"
    return abs(after - before), words


def _sums(data: ValidatedInput, readout: str) -> tuple[np.ndarray, np.ndarray]:
    """Per field of ``data.fovs``: the sum and the number of its non-missing readout values."""
    values = data.feature_values(readout)
    positions = fov_positions(data, data.fovs)
    keep = ~np.isnan(values) & (positions >= 0)
    sums = np.bincount(positions[keep], weights=values[keep], minlength=len(data.fovs))
    counts = np.bincount(positions[keep], minlength=len(data.fovs)).astype(float)
    return sums, counts


def _sides(
    group_data: ValidatedInput, condition: Any, in_repeat: pd.DataFrame, setup: Any
) -> tuple[pd.DataFrame, list[str]]:
    """The contrast rows of this repeat a group's fields move, and the side each is on
    (``mean``: the group's condition is compared; ``control_mean``: it is the control).
    With the declared control, a control group moves every condition of its repeat; with
    comparisons, a condition's group moves each comparison it is in, on either side (its
    mean, the same observations, is one on both)."""
    comparisons = getattr(setup, "comparisons", None)
    if comparisons:
        subjects = {comparison_label(c, r): c for c, r in comparisons}
        mine = in_repeat["condition"].map(subjects) == condition
        reference = in_repeat["control_condition"] == str(condition)
        rows = in_repeat[mine | reference]
        return rows, ["mean" if m else "control_mean" for m in mine[mine | reference]]
    if _is_control(group_data, setup.control or {}):
        # Every condition compared with the control in this repeat.
        return in_repeat, ["control_mean"] * len(in_repeat)
    rows = in_repeat[in_repeat["condition"] == condition].head(1)
    return rows, ["mean"] * len(rows)


def _is_control(group_data: ValidatedInput, control: Mapping[str, Any]) -> bool:
    """Whether the group belongs to the declared control (every control column matches)."""
    columns = [c for c in control if c in group_data.fovs.columns]
    if not control or len(columns) < len(control) or group_data.fovs.empty:
        return False
    values = group_data.fovs[columns].drop_duplicates()
    return len(values) == 1 and all(values.iloc[0][c] == v for c, v in control.items())


def _found(
    group: dict[str, Any],
    group_data: ValidatedInput,
    table: pd.DataFrame,
    setup: Any,
    threshold: float,
    skipped: list[dict[str, Any]],
    small: list[dict[str, Any]],
    spread: float | None = INFLUENCE_SPREAD,
    log: bool | None = None,
) -> list[dict[str, Any]]:
    """The influence rows of one analysis group's fields (see the module); a mean of 0 has
    no relative change and goes in ``skipped``, as ``inspect_fovs`` records it, and a group
    too small to judge in ``small``; ``log`` the readout's scale in the test."""
    condition, repeat = group.get(CONDITION_COLUMN), group.get(REPEAT_COLUMN)
    rows, columns = _sides(group_data, condition, table[table[REPEAT_COLUMN] == repeat], setup)
    if rows.empty:
        return []
    readout = setup.readout
    sums, counts = _sums(group_data, readout)
    unit = group_data.observation_unit or "observation"
    size = "observations" if columns[0] == "mean" else "control_observations"
    mean, total = float(rows[columns[0]].iloc[0]), float(rows[size].iloc[0])
    if not mean > 0:
        skipped.append({**_where(group), "summary": f"mean of {readout}", "reason": _ZERO})
        return []
    if spread is not None and np.count_nonzero(counts) < INFLUENCE_MIN_FIELDS:
        small.append({**_where(group), "reason": SMALL_GROUP})
        return []
    positions = [p for p in np.flatnonzero(counts) if total - counts[p] >= 1]
    withouts = [(mean * total - sums[p]) / (total - counts[p]) for p in positions]
    changes = [(without - mean) / mean for without in withouts]
    apart = stands_out(np.asarray(changes), spread)
    found = []
    for position, without, moved, alone in zip(positions, withouts, changes, apart, strict=True):
        if abs(moved) < threshold or not alone:
            continue
        # The condition whose change the field moves most (itself, for a treated field).
        effects = [
            (
                *_effect(
                    table[table["condition"] == r.condition], repeat, columns[i], without, log
                ),
                i,
            )
            for i, r in enumerate(rows.itertuples())
        ]
        _, effect, best = max(effects, key=lambda item: item[0])
        named, column = rows["condition"].iloc[best], columns[best]
        side = f"{named}'s mean" if column == "mean" else "the control's mean"
        whose = (
            "its change against the control"
            if column == "mean"
            else (f"{named}'s change against it")
        )
        if getattr(setup, "comparisons", None):  # the row names the comparison
            side, whose = f"{condition}'s mean", f"the change of {named}"
        found.append(
            {
                "position": int(position),
                "reason": "influence",
                "value": float(moved),
                "detail": (
                    f"removing it {'lowers' if moved < 0 else 'raises'} {side} of "
                    f"{readout} in this repeat (every {unit} weighing one, as in the test "
                    f"across repeats) by {percent(abs(moved))}: {whose} goes from {effect}"
                ),
            }
        )
    return found


def with_result_influence(
    inspection: InspectionResult,
    objects: pd.DataFrame,
    fovs: pd.DataFrame,
    contrast: Any,
    setup: Any,
    threshold: float = THRESHOLD,
    spread: float | None = INFLUENCE_SPREAD,
) -> InspectionResult:
    """``inspection`` (run without summaries) with the influence of each field on the result
    (see the module); unchanged without a contrast across repeats. ``objects`` holds the
    values the contrast compared (``Setup.compared_values``). ``spread=None``: the threshold
    alone, every group judged."""
    if not influence_on_result(contrast):
        return inspection
    table = contrast.table
    table = table[table["mean"].notna() & table["control_mean"].notna()]
    data = validate_input_tables(objects, fovs=fovs, feature_cols=[setup.readout], **setup.common)
    trace = inspection.settings.get("trace", [])
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    small: list[dict[str, Any]] = []
    for group, both in iter_analysis_groups(data):
        group_data = replace(both, feature_cols=(setup.readout,))
        values = group_data.feature_values(setup.readout)
        if np.nanmin(values, initial=0.0) < 0:
            skipped.append(
                {**_where(group), "summary": f"mean of {setup.readout}", "reason": _NEGATIVE}
            )
            continue
        log = contrast.settings.get("log_scale")
        found = _found(group, group_data, table, setup, threshold, skipped, small, spread, log)
        rows += field_rows(group, group_data, found, trace)
    frame = pd.concat([inspection.fovs, pd.DataFrame(rows)], ignore_index=True)
    checked = list(inspection.settings.get("checked", []))
    order = ("new state", "influence", "count", "neighbours")
    checked = [reason for reason in order if reason in {*checked, "influence"}]
    settings = {
        **inspection.settings,
        "checked": checked,
        "influence_on": "result",
        "influence_threshold": float(threshold),
        "influence_spread": None if spread is None else float(spread),
        "skipped_summaries": skipped,
        "skipped_small": small,
    }
    return replace(inspection, fovs=frame, settings=settings)


__all__ = ["THRESHOLD", "influence_on_result", "with_result_influence"]
