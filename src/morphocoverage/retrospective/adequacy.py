"""Apparent and realized adequacy: when a rule would have stopped, and when it was really enough.

Questions R2 (``docs/report/questions.md``) and P4 (``docs/optimise/optimise.md``). Four
points along one replayed acquisition, each a number of FOVs, or ``None`` when not reached
within the stream:

- ``first_apparent``: the first evaluated prefix whose decision is stop.
- ``sustained_apparent``: the earliest prefix from which every later evaluated
  decision is stop. A continue, an unsupported or an undefined decision breaks
  it; if the last prefix is not stop, it is not reached.
- ``first_realized``: the first prefix whose outcomes against the held-back
  reservoir preserve every declared criterion.
- ``sustained_realized``: the earliest such prefix from which every later
  evaluable prefix also preserves every criterion. Held-back outcomes can
  cross and uncross (enough at 10 and 11, not at 12, enough again from 13), so
  this is the preferred answer to "how many FOVs did this experiment really
  need?". A prefix is evaluable when at least one criterion is not preserved
  (then it is not enough) or all are preserved; a prefix with an undefined
  criterion and no failure is skipped, neither confirming nor breaking it.
  The last prefix must be evaluable, as the last decision must be stop for
  ``sustained_apparent``: when every prefix after some field is undefined (a
  field with a missing value makes every later prefix undefined), nothing
  says that the later field counts preserved it, so it is not reached.
  ``last_judged`` gives the last evaluable prefix of each run.

The apparent points use only decisions, the realized points only outcomes;
they are never mixed. Realized points depend on the order of FOVs, never on
the strategy. Realized points are reported for all criteria together
(``ALL_CRITERIA``) and for each criterion on its own. The summary names the
groups the replay skipped (``ReplayResult.skipped``); its wording, short and
full, is ``adequacy_text``.

Shared-private: ``_unjudged`` (used by ``spatial``), re-exported from ``adequacy_text``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from .._frozen import owned
from .adequacy_text import (  # noqa: F401  (the point helpers, re-exported for their callers)
    ALL_CRITERIA,
    _median_point,
    _reached,
    _tail_point,
    _unjudged,
    adequacy_summary,
)
from .criteria import Preservation
from .replay import ReplayResult

_STOP = "stop"
_RUN_KEYS = ("split", "order_id")


def _checked(steps: Sequence[int], values: Sequence[Any]) -> list[tuple[int, Any]]:
    """``(step, value)`` pairs, refusing unequal lengths and steps that do not strictly increase."""
    if len(steps) != len(values):
        raise ValueError("steps and values must have the same length")
    pairs = list(zip(steps, values, strict=True))
    ordered = [s for s, _ in pairs]
    if ordered != sorted(set(ordered)):
        raise ValueError("steps must be strictly increasing")
    return pairs


def first_apparent_adequacy(steps: Sequence[int], decisions: Sequence[str | None]) -> int | None:
    """The first step whose decision is ``"stop"``, or ``None``."""
    return next((s for s, d in _checked(steps, decisions) if d == _STOP), None)


def sustained_apparent_adequacy(
    steps: Sequence[int], decisions: Sequence[str | None]
) -> int | None:
    """The earliest step from which every later decision is ``"stop"``, or ``None``."""
    sustained = None
    for step, decision in _checked(steps, decisions):
        if decision == _STOP:
            sustained = step if sustained is None else sustained
        else:
            sustained = None
    return sustained


def combined_status(statuses: Sequence[str]) -> str:
    """All criteria of one prefix together: not preserved beats undefined beats preserved."""
    values = [Preservation(s) for s in statuses]
    if not values:
        raise ValueError("give at least one criterion status")
    if Preservation.NOT_PRESERVED in values:
        return Preservation.NOT_PRESERVED.value
    if Preservation.UNDEFINED in values:
        return Preservation.UNDEFINED.value
    return Preservation.PRESERVED.value


def first_realized_adequacy(steps: Sequence[int], statuses: Sequence[str]) -> int | None:
    """The first step whose (combined) status is preserved, or ``None``."""
    return next(
        (s for s, v in _checked(steps, statuses) if v == Preservation.PRESERVED.value), None
    )


def sustained_realized_adequacy(steps: Sequence[int], statuses: Sequence[str]) -> int | None:
    """The earliest preserved step after which no evaluable step fails, or ``None``.

    Undefined steps in between are skipped: they neither start nor break a
    sustained run. An undefined last step leaves the run unconfirmed to the
    end of the stream, so the point is not reached (``None``).
    """
    sustained, last = None, None
    for step, status in _checked(steps, statuses):
        Preservation(status)
        last = status
        if status == Preservation.PRESERVED.value:
            sustained = step if sustained is None else sustained
        elif status == Preservation.NOT_PRESERVED.value:
            sustained = None
    return None if last == Preservation.UNDEFINED.value else sustained


def last_judged_step(steps: Sequence[int], statuses: Sequence[str]) -> int | None:
    """The last step whose status is not undefined, or ``None`` when every step is."""
    judged = [s for s, v in _checked(steps, statuses) if v != Preservation.UNDEFINED.value]
    return judged[-1] if judged else None


@owned("apparent", "realized", "settings", "skipped")
@dataclass(frozen=True)
class AdequacyResult:
    """Adequacy points per replayed acquisition.

    ``apparent`` has one row per group, split, order and strategy
    (``first_apparent``, ``sustained_apparent``); ``realized`` one row per
    group, split, order and criterion, including ``ALL_CRITERIA``
    (``first_realized``, ``sustained_realized``, ``judged_fovs``, the
    number of prefixes whose outcome was not undefined: 0 when the criterion
    could never be judged, which is not the same as never being preserved,
    and ``last_judged``, the last such prefix). Both carry ``stream_fovs``,
    the most FOVs the replay could reveal. ``skipped`` lists the groups the
    replay skipped, with the reason.
    """

    apparent: pd.DataFrame
    realized: pd.DataFrame
    group_cols: tuple[str, ...]
    settings: dict[str, Any] = field(default_factory=dict)
    skipped: pd.DataFrame = field(default_factory=pd.DataFrame)

    def summary(self, detail: str = "short") -> str:
        """The points in words: ``"short"`` tables across groups, ``"full"`` per group
        (``adequacy_text``)."""
        return adequacy_summary(self, detail)


def _keyed(frame: pd.DataFrame, keys: list[str]) -> pd.core.groupby.DataFrameGroupBy:
    return frame.groupby(keys, sort=False, dropna=False)


def replayed_strategies(replay: ReplayResult) -> list[tuple[str, str]]:
    """``(strategy, config_id)`` of every replayed strategy, decided anywhere or not."""
    pairs = dict.fromkeys(
        zip(replay.decisions["strategy"], replay.decisions["config_id"], strict=True)
    )
    for config_id, kind in replay.settings.get("strategy_kinds", {}).items():
        pairs.setdefault((kind, config_id))
    return list(pairs)


def run_decisions(
    replay: ReplayResult,
) -> list[tuple[dict[str, Any], dict[str, Any], pd.DataFrame]]:
    """``(run, stream, decisions)`` for every replayed acquisition and strategy.

    ``run`` holds the group, split, order and strategy; ``stream`` the row of
    ``replay.streams``; ``decisions`` the strategy's rows in prefix order, empty
    when the stream was shorter than ``min_fovs`` and the rule never decided,
    so such a run counts as one that never stopped instead of vanishing.
    """
    runs = [*replay.group_cols, *_RUN_KEYS]
    decided = {
        key: rows.sort_values("fovs")
        for key, rows in _keyed(replay.decisions, [*runs, "strategy", "config_id"])
    }
    empty = replay.decisions.iloc[:0]
    found = []
    for _, stream in replay.streams.iterrows():
        for strategy, config_id in replayed_strategies(replay):
            run = {**{k: stream[k] for k in runs}, "strategy": strategy, "config_id": config_id}
            found.append((run, stream, decided.get(tuple(run.values()), empty)))
    return found


def adequacy_points(replay: ReplayResult) -> AdequacyResult:
    """The four adequacy points of every replayed acquisition in ``replay``."""
    if not isinstance(replay, ReplayResult):
        raise TypeError("replay must be a ReplayResult")
    groups = list(replay.group_cols)
    runs = [*groups, *_RUN_KEYS]
    streams = {
        tuple(row[k] for k in runs): (int(row["stream_fovs"]), row["order"])
        for _, row in replay.streams.iterrows()
    }
    apparent = []
    for values, stream, rows in run_decisions(replay):
        steps, decisions = rows["fovs"].tolist(), rows["decision"].tolist()
        apparent.append(
            {
                **values,
                "order": stream["order"],
                "first_apparent": first_apparent_adequacy(steps, decisions),
                "sustained_apparent": sustained_apparent_adequacy(steps, decisions),
                "stream_fovs": int(stream["stream_fovs"]),
            }
        )
    realized = []
    for key, rows in _keyed(replay.outcomes, runs):
        values = dict(zip(runs, key, strict=True))
        stream, order = streams[tuple(values[k] for k in runs)]
        combined = rows.groupby("fovs", sort=True)["status"].agg(combined_status)
        series = [(ALL_CRITERIA, combined.index.tolist(), combined.tolist())]
        for criterion, own in rows.groupby("criterion", sort=False):
            own = own.sort_values("fovs")
            series.append((criterion, own["fovs"].tolist(), own["status"].tolist()))
        for criterion, steps, statuses in series:
            realized.append(
                {
                    **values,
                    "order": order,
                    "criterion": criterion,
                    "first_realized": first_realized_adequacy(steps, statuses),
                    "sustained_realized": sustained_realized_adequacy(steps, statuses),
                    "judged_fovs": sum(s != Preservation.UNDEFINED.value for s in statuses),
                    "last_judged": last_judged_step(steps, statuses),
                    "stream_fovs": stream,
                }
            )
    point_columns = ("first_apparent", "sustained_apparent")
    apparent_table = pd.DataFrame(
        apparent,
        columns=[*runs, "order", "strategy", "config_id", *point_columns, "stream_fovs"],
    )
    realized_table = pd.DataFrame(
        realized,
        columns=[*runs, "order", "criterion", "first_realized", "sustained_realized"]
        + ["judged_fovs", "last_judged", "stream_fovs"],
    )
    for table, columns in (
        (apparent_table, point_columns),
        (realized_table, ("first_realized", "sustained_realized", "last_judged")),
    ):
        for column in columns:
            table[column] = table[column].astype("Int64")  # FOV counts, missing as <NA>
    return AdequacyResult(
        apparent_table,
        realized_table,
        tuple(groups),
        dict(replay.settings),
        skipped=replay.skipped.copy(),
    )


__all__ = [
    "ALL_CRITERIA",
    "AdequacyResult",
    "adequacy_points",
    "adequacy_summary",
    "combined_status",
    "first_apparent_adequacy",
    "first_realized_adequacy",
    "last_judged_step",
    "replayed_strategies",
    "run_decisions",
    "sustained_apparent_adequacy",
    "sustained_realized_adequacy",
]
