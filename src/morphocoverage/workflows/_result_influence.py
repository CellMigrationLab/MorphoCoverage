"""The fields that sway the result (private; ``retrospective.report``; owner, 2026-10-02).

The inspection's influence reason (``inspect_fovs``) asks whether leaving one field out
changes a summary of its analysis group in which every field weighs one. The result the
report states is the paired t-test on each biological repeat's mean of the observations
(``contrast_repeats``), every observation weighing one and the wells of a repeat pooled.
So in the report a field is influential when leaving it out moves the mean of its
condition in its repeat, as that test computes it, by at least ``threshold`` of it; the
reason then says what this does to the result: the condition's change against the control
(a fold change, or the difference of the means where a mean is not positive) and the paired
t-test's p, with the field and without it. A control field moves every condition compared
with it in its repeat: the condition whose change it moves most is named.

A mean at or below zero has no relative change: those repeats are skipped and said so.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd

from .._wording import change, p_value, percent
from ..data_check import validate_input_tables
from ..evaluator import iter_analysis_groups
from ..metadata import CONDITION_COLUMN, REPEAT_COLUMN
from ..retrospective._text import where
from ..retrospective.inspection import InspectionResult, field_rows
from ..sampling_unit.contrast_repeats import MIN_REPEATS, _paired_t
from ..schema import ValidatedInput, fov_positions

THRESHOLD = 0.10


def _result(means: np.ndarray, controls: np.ndarray) -> tuple[float, str, float]:
    """The change against the control and the paired t-test's p over the repeats, on the log
    ratios when every mean is positive (``"normalised"``), else on the differences."""
    if (means > 0).all() and (controls > 0).all():
        ratios = np.log(means / controls)
        p = _paired_t(ratios) if len(ratios) >= MIN_REPEATS else np.nan
        return float(np.exp(ratios.mean())), "normalised", p
    differences = means - controls
    p = _paired_t(differences) if len(differences) >= MIN_REPEATS else np.nan
    return float(differences.mean()), "raw", p


def _effect(block: pd.DataFrame, repeat: Any, column: str, value: float) -> tuple[float, str]:
    """How far the condition's change moves with the repeat's mean in ``column`` set to
    ``value`` (the field left out), and the change and p before and after, in words."""
    means = block["mean"].to_numpy(float)
    controls = block["control_mean"].to_numpy(float)
    before, values, p_before = _result(means, controls)
    moved = (means if column == "mean" else controls).copy()
    moved[(block[REPEAT_COLUMN] == repeat).to_numpy()] = value
    after, _, p_after = _result(moved, controls) if column == "mean" else _result(means, moved)
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


def _found(
    group: dict[str, Any],
    group_data: ValidatedInput,
    table: pd.DataFrame,
    readout: str,
    threshold: float,
    skipped: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """The influence rows of one analysis group's fields (see the module)."""
    condition, repeat = group.get(CONDITION_COLUMN), group.get(REPEAT_COLUMN)
    in_repeat = table[table[REPEAT_COLUMN] == repeat]
    treated = in_repeat[in_repeat["condition"] == condition]
    controls = in_repeat[in_repeat["control_condition"].astype(str) == str(condition)]
    if treated.empty and controls.empty:
        return []
    sides = [("mean", "observations", treated.head(1))] if len(treated) else []
    sides += [("control_mean", "control_observations", controls)] if len(controls) else []
    sums, counts = _sums(group_data, readout)
    unit = group_data.observation_unit or "observation"
    found = []
    for column, size, rows in sides:
        mean, total = float(rows[column].iloc[0]), float(rows[size].iloc[0])
        if not mean > 0:
            skipped.append(
                {
                    "group": where(group),
                    "summary": f"mean of {readout}",
                    "reason": "is not positive here, so a relative change is undefined",
                }
            )
            continue
        for position in np.flatnonzero(counts):
            if total - counts[position] < 1:
                continue
            without = (mean * total - sums[position]) / (total - counts[position])
            moved = (without - mean) / mean
            if abs(moved) < threshold:
                continue
            # The condition whose change the field moves most (itself, for a treated field).
            effects = [
                (*_effect(table[table["condition"] == r.condition], repeat, column, without), i)
                for i, r in enumerate(rows.itertuples())
            ]
            _, effect, best = max(effects, key=lambda item: item[0])
            named = rows["condition"].iloc[best]
            side = f"{named}'s mean" if column == "mean" else "the control's mean"
            whose = (
                "its change against the control"
                if column == "mean"
                else f"{named}'s change against it"
            )
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
) -> InspectionResult:
    """``inspection`` (run without summaries) with the influence of each field on the result
    (see the module); unchanged without a contrast across repeats."""
    table = getattr(contrast, "table", None)
    if table is None or table.empty or REPEAT_COLUMN not in table:
        return inspection
    table = table[table["mean"].notna() & table["control_mean"].notna()]
    data = validate_input_tables(objects, fovs=fovs, feature_cols=[setup.readout], **setup.common)
    trace = inspection.settings.get("trace", [])
    rows, skipped = [], []
    for group, both in iter_analysis_groups(data):
        group_data = replace(both, feature_cols=(setup.readout,))
        found = _found(group, group_data, table, setup.readout, threshold, skipped)
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
        "skipped_summaries": skipped,
    }
    return replace(inspection, fovs=frame, settings=settings)


__all__ = ["THRESHOLD", "with_result_influence"]
