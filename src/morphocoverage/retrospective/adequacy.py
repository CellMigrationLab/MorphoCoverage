"""Apparent and realized adequacy: when a rule would have stopped, and when it was really enough.

Roadmap Step 3. Four points along one replayed acquisition, each a number of
FOVs, or ``None`` when not reached within the stream:

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

from ._text import rows_of, where
from .criteria import Preservation
from .replay import ReplayResult

ALL_CRITERIA = "all criteria"
_STOP = "stop"
_ORDER_WORDS = {"recorded": "In the recorded order", "permuted": "In random orders"}
_RUN_KEYS = ("split", "order_id")


def _checked(steps: Sequence[int], values: Sequence[Any]) -> list[tuple[int, Any]]:
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
    (``first_realized``, ``sustained_realized``). Both carry ``stream_fovs``,
    the most FOVs the replay could reveal.
    """

    apparent: pd.DataFrame
    realized: pd.DataFrame
    group_cols: tuple[str, ...]
    settings: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """The points in sentences, one block per group."""
        return adequacy_summary(self)


def _keyed(frame: pd.DataFrame, keys: list[str]):
    return frame.groupby(keys, sort=False, dropna=False)


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
    for key, rows in _keyed(replay.decisions, [*runs, "strategy", "config_id"]):
        rows = rows.sort_values("fovs")
        steps, decisions = rows["fovs"].tolist(), rows["decision"].tolist()
        values = dict(zip([*runs, "strategy", "config_id"], key, strict=True))
        stream, order = streams[tuple(values[k] for k in runs)]
        apparent.append(
            {
                **values,
                "order": order,
                "first_apparent": first_apparent_adequacy(steps, decisions),
                "sustained_apparent": sustained_apparent_adequacy(steps, decisions),
                "stream_fovs": stream,
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
        + ["stream_fovs"],
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
    stream, so the median is "not reached" when more than half of the runs never reached it.
    """
    filled = values.astype("Float64").to_numpy(dtype=float, na_value=np.inf)
    median = float(np.median(filled))
    if not np.isfinite(median):
        return f"not within the fields available (up to {int(streams.max())}) in most runs"
    return f"field {median:.0f}" if median.is_integer() else f"field {median:.1f}"


def _at(point: str) -> str:
    return point if point.startswith("not") else f"at {point}"


def _from(point: str) -> str:
    return point if point.startswith("not") else f"from {point}"


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
            stream = rows["stream_fovs"]
            first = _at(_median_point(rows["first_realized"], stream))
            lasting = _from(_median_point(rows["sustained_realized"], stream))
            lines.append(
                f"  {_ORDER_WORDS.get(order, order)}: judged on held-back fields, everything "
                f"you wanted was first preserved {first} and reliably {lasting}."
            )
            if len(labels) > 1:
                for criterion, own in realized[
                    (realized["order"] == order) & (realized["criterion"] != ALL_CRITERIA)
                ].groupby("criterion", sort=False):
                    lines.append(
                        f"    {labels.get(criterion, criterion)}: reliably "
                        f"{_from(_median_point(own['sustained_realized'], own['stream_fovs']))}."
                    )
            for config_id, own in apparent[apparent["order"] == order].groupby("config_id"):
                name = names.get(config_id, own["strategy"].iloc[0])
                first = _at(_median_point(own["first_apparent"], own["stream_fovs"]))
                lasting = _from(_median_point(own["sustained_apparent"], own["stream_fovs"]))
                lines.append(
                    f"    {name} would have first said stop {first}, and kept saying stop "
                    f"{lasting}."
                )
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
