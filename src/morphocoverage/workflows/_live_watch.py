"""When ``LiveFolder.watch`` stops (private, split out of ``workflows.live``): its arguments
checked, and whether one field's results end its group's acquisition.

Shared-private: used by ``workflows.live``.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..results import Decision, StrategyResult, _target_decisions


def check_watch(interval: float, until_stop: bool | str, stop_on: str) -> None:
    """Refuse a watch interval that is not positive, and an unknown ``until_stop`` or
    ``stop_on``."""
    if interval <= 0:
        raise ValueError("interval must be positive")
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
