"""When ``LiveFolder.watch`` stops (private, split out of ``workflows.live``): its arguments
checked, and whether one field's results end its group's acquisition.

Shared-private: used by ``workflows.live``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from .._checks import is_integer, is_number, is_positive_finite
from ..results import Decision, StrategyResult, _target_decisions


def check_watch(
    interval: float,
    timeout: float | None,
    max_fovs: int | None,
    until_stop: bool | str,
    stop_on: str,
) -> None:
    """Refuse every watch control that is not usable, before any field is read (#360): an
    ``interval`` that is not a finite positive number (``inf`` would sleep for ever after the
    backlog, ``nan`` fail only once a field was added), a ``timeout`` that is not ``None`` or a
    finite number of at least 0, a ``max_fovs`` that is not ``None`` or an integer of at least
    0 (0 adds no field), booleans for any of them, and an unknown ``until_stop`` or
    ``stop_on``."""
    if not is_positive_finite(interval):
        raise ValueError(f"interval must be a finite number of seconds above 0, not {interval!r}")
    if timeout is not None and not (is_number(timeout) and math.isfinite(timeout) and timeout >= 0):
        raise ValueError(f"timeout must be None or a finite number of seconds, not {timeout!r}")
    if max_fovs is not None and not (is_integer(max_fovs) and max_fovs >= 0):
        raise ValueError(f"max_fovs must be None or an integer of at least 0, not {max_fovs!r}")
    if until_stop not in (False, True, "all_groups", "latest_group"):
        raise ValueError("until_stop must be False, True, 'all_groups' or 'latest_group'")
    if stop_on not in ("decision", "rule"):
        raise ValueError(f"stop_on must be 'decision' or 'rule', not {stop_on!r}")


def done(results: Sequence[StrategyResult], stop_on: str) -> bool:
    """Whether one field's results end its group's acquisition (see ``LiveFolder.watch``)."""
    if not results:
        return False
    if stop_on == "rule":
        return all(r.rule_decision is Decision.STOP for r in results)
    return _target_decisions(results)[-1].decision is Decision.STOP


__all__ = ["check_watch", "done"]
