"""The acquisition-order check of a live session (private; ``live_session`` uses it).

``live_session`` returns a ``_CheckedSession``: an ``OnlineSession`` that also
knows the setup's readout, acquisition-order column and stage columns, and
checks the fields of a group added so far (``order_check``) whenever
``LiveFolder`` or ``LiveTables`` builds a ``LiveUpdate``. The check reads only
those fields, so a replay of the same prefix gives the same check.

Shared-private: ``_CheckedSession`` and ``_order_check`` are used by
``workflows.live``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..online import OnlineSession
from ..order_check import OrderCheck, check_acquisition_order
from ..results import StrategyResult


class _CheckedSession(OnlineSession):
    """An ``OnlineSession`` that can check the acquisition order of a group's fields so far."""

    def __init__(
        self,
        *,
        readout: str,
        acquisition_order_col: str | None,
        stage_cols: tuple[str, str] | None,
        **session: Any,
    ) -> None:
        super().__init__(**session)
        self.readout = readout
        self.acquisition_order_col = acquisition_order_col
        self.stage_cols = stage_cols

    def order_check(self, group: Mapping[str, Any]) -> OrderCheck | None:
        """The check of ``group``'s fields added so far; ``None`` with neither column declared.

        A pure function of those fields, so a replay of the same prefix gives
        the same check (``check_acquisition_order`` on the prefix).
        """
        if self.acquisition_order_col is None and self.stage_cols is None:
            return None
        added = self._groups.get(tuple(group.values()))
        if added is None:
            return None
        visible, _ = added.visible()
        return check_acquisition_order(
            visible,
            self.readout,
            order_col=self.acquisition_order_col,
            stage_cols=self.stage_cols,
        )


def _order_check(session: OnlineSession, results: list[StrategyResult]) -> OrderCheck | None:
    """The order check of the group the latest field joined, when the session runs one."""
    if not isinstance(session, _CheckedSession) or not results:
        return None
    return session.order_check(results[0].group)


__all__: list[str] = []
