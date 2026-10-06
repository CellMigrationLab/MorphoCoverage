"""The declared control's rates as other answers read them (private): the pooled rate of a
candidate, the reference's false-alarm rate as ``inspect_fovs`` judges a field, and why a
rate does not describe another analysis's judgements.

Split from ``report_profile`` (owner, 2026-10-06, audit 10) to keep it under 600 lines;
``report_profile`` re-exports these names. Shared-private: used by ``report_profile`` and
``_new_states``.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from .controls import ControlResult

# Confidence of the upper bound printed with the false-alarm rate.
_CONFIDENCE = 0.95


def _control_rate(controls: ControlResult | None, kind: str) -> float | None:
    """A pooled rate of the declared control (``positive``, ``reference_lofo``...), if measured."""
    if controls is None or controls.status == "not assessed":
        return None
    rates = controls.pooled.set_index("candidate")["rate"]
    if kind not in rates.index or pd.isna(rates[kind]):
        return None
    return float(rates[kind])


def _false_alarm_rate(controls: ControlResult | None) -> float | None:
    """The declared reference's false-alarm rate, judged as ``inspect_fovs`` judges a field."""
    return _control_rate(controls, "reference_lofo")


def _mismatch(
    controls: ControlResult, definition: str | None, groups: Sequence[str] | None
) -> str | None:
    """Why the control's rates do not describe another analysis's judgements, or None.

    The rates hold for one definition of new, on its features, judged within
    the control's analysis groups; a result that records another definition
    or other groups is not compared with them. What a result does not record
    is not checked.
    """
    settings = getattr(controls, "settings", {}) or {}
    own = settings.get("novelty_definition")
    if own is not None and definition is not None and own != definition:
        return "the control used another definition of new, or other features"
    lofo = settings.get("lofo_group_cols")
    if lofo is not None and groups is not None and list(lofo) != list(groups):
        return (
            f"the control's rates were measured within {_groups(lofo)}, this answer within "
            f"{_groups(groups)}"
        )
    return None


def _groups(columns: Sequence[str]) -> str:
    """``the groups condition, repeat``, or ``whole strata`` when there are none."""
    return f"the groups {', '.join(columns)}" if len(columns) else "whole strata"


__all__ = ["_CONFIDENCE", "_control_rate", "_false_alarm_rate", "_groups", "_mismatch"]
