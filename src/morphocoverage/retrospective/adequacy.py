"""Apparent and realized adequacy: when a rule would have stopped, and when it was really enough.

Questions R2 and P4 of ``docs/questions.md``. Four points along one replayed
acquisition, each a number of FOVs, or ``None`` when not reached within the
stream:

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

The apparent points use only decisions, the realized points only outcomes;
they are never mixed. Realized points depend on the order of FOVs, never on
the strategy. Realized points are reported for all criteria together
(``ALL_CRITERIA``) and for each criterion on its own.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ._text import order_phrase, rows_of, where
from .criteria import Preservation
from .replay import ReplayResult

ALL_CRITERIA = "all criteria"
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

    Undefined steps are skipped: they neither start nor break a sustained run.
    """
    sustained = None
    for step, status in _checked(steps, statuses):
        Preservation(status)
        if status == Preservation.PRESERVED.value:
            sustained = step if sustained is None else sustained
        elif status == Preservation.NOT_PRESERVED.value:
            sustained = None
    return sustained


@dataclass(frozen=True)
class AdequacyResult:
    """Adequacy points per replayed acquisition.

    ``apparent`` has one row per group, split, order and strategy
    (``first_apparent``, ``sustained_apparent``); ``realized`` one row per
    group, split, order and criterion, including ``ALL_CRITERIA``
    (``first_realized``, ``sustained_realized``, and ``judged_fovs``, the
    number of prefixes whose outcome was not undefined: 0 when the criterion
    could never be judged, which is not the same as never being preserved).
    Both carry ``stream_fovs``, the most FOVs the replay could reveal.
    """

    apparent: pd.DataFrame
    realized: pd.DataFrame
    group_cols: tuple[str, ...]
    settings: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """The points in sentences, one block per group."""
        return adequacy_summary(self)


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
        + ["judged_fovs", "stream_fovs"],
    )
    for table, columns in (
        (apparent_table, point_columns),
        (realized_table, ("first_realized", "sustained_realized")),
    ):
        for column in columns:
            table[column] = table[column].astype("Int64")  # FOV counts, missing as <NA>
    return AdequacyResult(apparent_table, realized_table, tuple(groups), dict(replay.settings))


def _median_point(values: pd.Series, streams: pd.Series) -> str:
    """The median over runs as "field 12"; a run that never reached it counts as beyond its own
    stream, so the median is "not reached" when half or more of the runs never reached it.
    """
    filled = values.astype("Float64").to_numpy(dtype=float, na_value=np.inf)
    median = float(np.median(filled))
    if not np.isfinite(median):
        return (
            f"not within the fields available (up to {int(streams.max())}) in half or more of runs"
        )
    return f"field {median:.0f}" if median.is_integer() else f"field {median:.1f}"


def _tail_point(values: pd.Series, streams: pd.Series) -> str:
    """The 90th percentile over runs, the number a protocol has to serve.

    Taken as an observed field count (the smallest one reached by at least
    90% of runs), never interpolated between runs, so "by field k in 90% of
    runs" is true of the runs; a run that never reached it counts as beyond
    its stream.
    """
    filled = values.astype("Float64").to_numpy(dtype=float, na_value=np.inf)
    tail = float(np.quantile(filled, 0.9, method="inverted_cdf"))
    if not np.isfinite(tail):
        limit = int(streams.max())
        return f"not within the fields available (up to {limit}) in more than 10% of runs"
    return f"by field {int(tail)} in 90% of runs"


def _reached(point: str, reached: str, missed: str) -> str:
    """``reached`` followed by the point, or ``missed`` in its place when it was not reached:
    "first preserved at field 3", "not preserved within the fields available (up to 11) ..."."""
    if point.startswith("not"):
        return f"{missed} {point.removeprefix('not ')}"
    return f"{reached} {point}"


def _unjudged(rows: pd.DataFrame) -> pd.Series:
    """Runs in which no prefix could be judged (every outcome undefined)."""
    if "judged_fovs" not in rows.columns:
        return pd.Series(False, index=rows.index)
    return rows["judged_fovs"].fillna(0).astype(int).eq(0)


def _realized_sentence(rows: pd.DataFrame) -> str:
    """When everything wanted was first, and reliably, preserved over the judged runs."""
    unjudged = _unjudged(rows)
    if unjudged.all():
        return (
            "judged on held-back fields, what you wanted could not be judged at any field count "
            "(the outcome was undefined in every run)."
        )
    rows = rows[~unjudged]
    stream = rows["stream_fovs"]
    first = _median_point(rows["first_realized"], stream)
    lasting = _median_point(rows["sustained_realized"], stream)
    tail = _tail_point(rows["sustained_realized"], stream)
    if first.startswith("not"):
        text = f"everything you wanted was not preserved {first.removeprefix('not ')}"
    else:
        text = (
            f"everything you wanted was first preserved at {first} and "
            f"{_reached(lasting, 'reliably from', 'not reliably')}"
        )
    skipped = (
        f"; {int(unjudged.sum())} of {len(unjudged)} runs could not be judged and are left out"
        if unjudged.any()
        else ""
    )
    return f"judged on held-back fields, {text} (medians; {tail}{skipped})."


def adequacy_summary(result: AdequacyResult) -> str:
    """Per group: when rules would have stopped and when held-back fields confirmed it."""
    keys = list(result.group_cols)
    labels = result.settings.get("labels", {})
    names = result.settings.get("strategy_names", {})
    blocks = []
    frames = [result.apparent[keys], result.realized[keys]]
    all_groups = pd.concat(frames).drop_duplicates() if keys else pd.DataFrame([{}])
    for _, group_row in all_groups.iterrows():
        group = {k: group_row[k] for k in keys}
        apparent, realized = rows_of(result.apparent, group), rows_of(result.realized, group)
        lines = [f"{where(group)}: when was it enough? (medians over hold-outs and orders)"]
        for order, rows in realized[realized["criterion"] == ALL_CRITERIA].groupby("order"):
            lines.append(f"  {order_phrase(order, capital=True)}: {_realized_sentence(rows)}")
            if len(labels) > 1:
                for criterion, own in realized[
                    (realized["order"] == order) & (realized["criterion"] != ALL_CRITERIA)
                ].groupby("criterion", sort=False):
                    label = labels.get(criterion, criterion)
                    judged = own[~_unjudged(own)]
                    if judged.empty:
                        lines.append(f"    {label}: could not be judged in any run.")
                        continue
                    point = _median_point(judged["sustained_realized"], judged["stream_fovs"])
                    reached = _reached(point, "reliably from", "not reliably")
                    lines.append(f"    {label}: {reached}.")
            for config_id, own in apparent[apparent["order"] == order].groupby("config_id"):
                name = names.get(config_id, own["strategy"].iloc[0])
                first = _median_point(own["first_apparent"], own["stream_fovs"])
                lasting = _median_point(own["sustained_apparent"], own["stream_fovs"])
                said = _reached(first, "would have first said stop at", "would not have said stop")
                if not first.startswith("not"):
                    said += ", and " + _reached(
                        lasting, "kept saying stop from", "not kept saying stop"
                    )
                lines.append(f"    {name} {said}.")
        lines.append(
            "  'Reliably' means every later field count, up to the last available field, also "
            "preserved it; each hold-out is a different set of held-back fields."
        )
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


__all__ = [
    "ALL_CRITERIA",
    "AdequacyResult",
    "adequacy_points",
    "adequacy_summary",
    "combined_status",
    "first_apparent_adequacy",
    "first_realized_adequacy",
    "sustained_apparent_adequacy",
    "sustained_realized_adequacy",
]
