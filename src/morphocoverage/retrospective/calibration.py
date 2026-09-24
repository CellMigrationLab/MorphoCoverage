"""Stopping-rule calibration: when this rule says stop, how often is that premature?

Roadmap Step 2. Sequential, because a stopping rule can depend on the order in
which FOVs arrive: a random set of 20 FOVs does not say whether the rule would
already have stopped at 14. On each replayed acquisition of ``replay`` (a fixed
held-back reservoir per split, several orders of the stream), the rule's first
stop is found and the prefix at that stop is judged against the reservoir:

- **correct**: stopped, and what you wanted was preserved at the stop;
- **premature**: stopped, and it was not preserved;
- **undefined at the stop**: stopped, and the outcome could not be computed;
- **never stopped**: the stream ended first; whether it was preserved at the
  end of the stream is reported separately, so a rule that stops late or never
  is visible as well as one that stops early.

Shares are over replayed acquisitions (splits x orders). Those are not
independent replicates of the sample; they describe this sample's behaviour
under this rule. For a fixed-FOV rule and random orders, the share of correct
stops at ``k`` estimates the same quantity as ``sampling_reliability`` at ``k``,
the one case where the two analyses coincide (when no outcome is undefined:
here undefined stops stay in the denominator, as "could not be judged").

The next-FOV novelty shuffle check (``calibrate_next_fov_novelty``) is a
different calibration, of a probability rather than of a stop, and is
re-exported here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..calibration import calibrate_next_fov_novelty, pool_calibration
from ._text import percent, where
from .adequacy import ALL_CRITERIA, combined_status, first_apparent_adequacy
from .criteria import Preservation
from .replay import ReplayResult

RUN_COLUMNS = (
    "split",
    "order_id",
    "order",
    "strategy",
    "config_id",
    "criterion",
    "stopped",
    "stop_fovs",
    "stream_fovs",
    "status_at_stop",
    "status_at_end",
)
SUMMARY_COLUMNS = (
    "order",
    "strategy",
    "config_id",
    "criterion",
    "runs",
    "stopped",
    "median_stop_fovs",
    "correct",
    "premature",
    "undefined_at_stop",
    "never_stopped",
    "never_stopped_preserved_at_end",
    "stream_fovs",
)


@dataclass(frozen=True)
class StoppingCalibration:
    """Per-run stops and their shares per group, order kind, strategy and criterion.

    ``runs`` has one row per replayed acquisition, strategy and criterion
    (including ``ALL_CRITERIA``, every criterion at once); ``table`` the
    shares over runs (``SUMMARY_COLUMNS`` after the group columns). ``stopped``,
    ``correct``, ``premature``, ``undefined_at_stop`` and ``never_stopped`` are
    shares of all runs; ``never_stopped_preserved_at_end`` is a share of the
    runs that never stopped (``None`` when all stopped).
    """

    runs: pd.DataFrame
    table: pd.DataFrame
    group_cols: tuple[str, ...]
    settings: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """The calibration in sentences, per group, strategy and order kind."""
        return calibration_summary(self)


def _statuses(outcomes: pd.DataFrame) -> dict[str, pd.Series]:
    """Status by prefix length for each criterion and for all criteria together."""
    series = {
        criterion: rows.set_index("fovs")["status"]
        for criterion, rows in outcomes.groupby("criterion", sort=False)
    }
    series[ALL_CRITERIA] = outcomes.groupby("fovs", sort=True)["status"].agg(combined_status)
    return series


def stopping_calibration(replay: ReplayResult) -> StoppingCalibration:
    """How often each strategy's first stop preserved each criterion in held-back FOVs."""
    if not isinstance(replay, ReplayResult):
        raise TypeError("replay must be a ReplayResult")
    groups = list(replay.group_cols)
    runs_key = [*groups, "split", "order_id"]
    outcome_runs = {
        key if isinstance(key, tuple) else (key,): _statuses(rows)
        for key, rows in replay.outcomes.groupby(runs_key, sort=False, dropna=False)
    }
    streams = {tuple(row[k] for k in runs_key): row for _, row in replay.streams.iterrows()}
    rows = []
    for key, decisions in replay.decisions.groupby(
        [*runs_key, "strategy", "config_id"], sort=False, dropna=False
    ):
        values = dict(zip([*runs_key, "strategy", "config_id"], key, strict=True))
        run = tuple(values[k] for k in runs_key)
        stream = streams[run]
        decisions = decisions.sort_values("fovs")
        stop = first_apparent_adequacy(decisions["fovs"].tolist(), decisions["decision"].tolist())
        end = int(stream["stream_fovs"])
        for criterion, status in outcome_runs[run].items():
            rows.append(
                {
                    **values,
                    "order": stream["order"],
                    "criterion": criterion,
                    "stopped": stop is not None,
                    "stop_fovs": stop,
                    "stream_fovs": end,
                    "status_at_stop": None if stop is None else status.loc[stop],
                    "status_at_end": status.loc[end],
                }
            )
    runs = pd.DataFrame(rows, columns=[*groups, *RUN_COLUMNS])
    runs["stop_fovs"] = runs["stop_fovs"].astype("Int64")
    return StoppingCalibration(runs, _table(runs, groups), tuple(groups), dict(replay.settings))


def _share(mask: pd.Series) -> float:
    return float(mask.mean())


def _table(runs: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    rows = []
    keys = [*groups, "order", "strategy", "config_id", "criterion"]
    for key, block in runs.groupby(keys, sort=False, dropna=False):
        stopped = block["stopped"].astype(bool)
        at_stop = block["status_at_stop"]
        never = block[~stopped]
        stops = block.loc[stopped, "stop_fovs"].astype(float)
        rows.append(
            {
                **dict(zip(keys, key, strict=True)),
                "runs": len(block),
                "stopped": _share(stopped),
                "median_stop_fovs": float(np.median(stops)) if len(stops) else None,
                "correct": _share(stopped & (at_stop == Preservation.PRESERVED.value)),
                "premature": _share(stopped & (at_stop == Preservation.NOT_PRESERVED.value)),
                "undefined_at_stop": _share(stopped & (at_stop == Preservation.UNDEFINED.value)),
                "never_stopped": _share(~stopped),
                "never_stopped_preserved_at_end": (
                    _share(never["status_at_end"] == Preservation.PRESERVED.value)
                    if len(never)
                    else None
                ),
                "stream_fovs": int(block["stream_fovs"].max()),
            }
        )
    return pd.DataFrame(rows, columns=[*groups, *SUMMARY_COLUMNS])


_ORDER_WORDS = {"recorded": "in the recorded order", "permuted": "in random orders"}


def calibration_summary(result: StoppingCalibration) -> str:
    """ "If you had let this rule stop the acquisition" in sentences."""
    keys = list(result.group_cols)
    labels = {ALL_CRITERIA: "everything you wanted", **result.settings.get("labels", {})}
    names = result.settings.get("strategy_names", {})
    blocks = []
    for key, rows in result.table.groupby([*keys, "config_id"], sort=False, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        group = dict(zip(keys, key[: len(keys)], strict=True))
        name = names.get(key[-1], rows["strategy"].iloc[0])
        lines = [f"{where(group)}: if you had let {name} stop the acquisition"]
        for order, own in rows.groupby("order", sort=False):
            first = own.iloc[0]
            if first["stopped"] == 0:
                lines.append(
                    f"  {_ORDER_WORDS.get(order, order)}: it never stopped within the "
                    f"{first['stream_fovs']} fields available ({first['runs']} acquisitions)."
                )
            else:
                lines.append(
                    f"  {_ORDER_WORDS.get(order, order)}: it stopped in "
                    f"{percent(first['stopped'])} of {first['runs']} acquisitions, after a "
                    f"median of {first['median_stop_fovs']:.0f} fields."
                )
            several = own["criterion"].nunique() > 2
            for row in own.itertuples(index=False):
                if row.stopped == 0 or (row.criterion == ALL_CRITERIA and not several):
                    continue
                late = ""
                if row.never_stopped > 0 and row.never_stopped_preserved_at_end is not None:
                    late = (
                        f"; never stopped in {percent(row.never_stopped)}, and "
                        f"{percent(row.never_stopped_preserved_at_end)} of those had it by "
                        "the last field"
                    )
                undefined = (
                    f"; {percent(row.undefined_at_stop)} could not be judged"
                    if row.undefined_at_stop > 0
                    else ""
                )
                lines.append(
                    f"    for {labels.get(row.criterion, row.criterion)}: "
                    f"{percent(row.premature)} of acquisitions stopped too early, "
                    f"{percent(row.correct)} stopped with it preserved{undefined}{late}."
                )
        lines.append(
            "  Judged on held-back fields the rule never saw; each acquisition is one hold-out "
            "and one order of the same fields, not an independent experiment."
        )
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


__all__ = [
    "RUN_COLUMNS",
    "SUMMARY_COLUMNS",
    "StoppingCalibration",
    "calibrate_next_fov_novelty",
    "calibration_summary",
    "pool_calibration",
    "stopping_calibration",
]
