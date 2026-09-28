"""Stopping-rule calibration: when this rule says stop, how often is that premature?

Question P6 of ``docs/questions.md``. Sequential, because a stopping rule can
depend on the order in which FOVs arrive: a random set of 20 FOVs does not say
whether the rule would already have stopped at 14. On each replayed acquisition
of ``replay`` (a fixed held-back reservoir per split, several orders of the
stream), the rule's first stop is found and the prefix at that stop is judged
against the reservoir:

- **correct**: stopped, and what you wanted was preserved at the stop;
- **not preserved at the stop**: stopped, and what you wanted was not
  preserved there. This is the headline risk of letting the rule stop, and
  it splits in two:
- **premature**: not preserved at the stop, though the whole stream was not
  shown to fail it (it preserved it by its last field, or could not judge it
  there);
- **not preserved by the stream**: not preserved at the stop, and not
  preserved by every available field together either. The held-back
  reference is itself a finite sample, so its noise alone can put the whole
  stream outside the tolerance: in simulated wells whose more fields did fix
  the stop, a third to a half of the wrong stops landed here. So "more
  fields of this stream would not have helped" cannot be told from
  reference noise, and the premature share is a lower bound on the stops
  that came too early, the total not preserved at the stop an upper bound
  (compare the reliability ceiling);
- **undefined at the stop**: stopped, and the outcome could not be computed;
- **never stopped**: the stream ended first; whether it was preserved at the
  end of the stream is reported separately, and said in the summary, so a
  rule that stops late or never is visible as well as one that stops early.

Shares are over replayed acquisitions (splits x orders). Those are not
independent replicates of the sample; they describe this sample's behaviour
under this rule. For a fixed-FOV rule and random orders, the share of correct
stops at ``k`` estimates the same quantity as ``sampling_reliability`` at ``k``,
the one case where the two analyses coincide (when no outcome is undefined:
here undefined stops stay in the denominator, as "could not be judged").

The next-FOV novelty shuffle check (``calibrate_next_fov_novelty``) is a
different calibration, of a probability rather than of a stop, and is
re-exported here. The summary names the groups the replay skipped
(``ReplayResult.skipped``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .._wording import count, number
from ..calibration import calibrate_next_fov_novelty, pool_calibration
from ..statistics import median_standard_error
from ._text import fields, order_phrase, percent, skipped_lines, where
from .adequacy import ALL_CRITERIA, combined_status, first_apparent_adequacy, run_decisions
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
    "median_stop_fovs_mc_se",
    "correct",
    "not_preserved_at_stop",
    "premature",
    "not_preserved_by_stream",
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
    ``correct``, ``premature``, ``not_preserved_by_stream``,
    ``undefined_at_stop`` and ``never_stopped`` are shares of all runs (the
    four kinds of stop add up to ``stopped``); ``not_preserved_at_stop`` is
    ``premature`` plus ``not_preserved_by_stream``, the headline risk, of which
    ``premature`` is the lower bound of the stops that came too early and
    ``not_preserved_at_stop`` the upper bound; ``median_stop_fovs_mc_se`` is
    the Monte Carlo standard error of the median stop over the runs that
    stopped (``statistics.median_standard_error``); ``never_stopped_preserved_at_end``
    is a share of the runs that never stopped (``None`` when all stopped).
    ``skipped`` lists the groups the replay skipped, with the reason.
    """

    runs: pd.DataFrame
    table: pd.DataFrame
    group_cols: tuple[str, ...]
    settings: dict[str, Any] = field(default_factory=dict)
    skipped: pd.DataFrame = field(default_factory=pd.DataFrame)

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
    rows = []
    # A run too short for the rule to decide (fewer than min_fovs FOVs) never stopped.
    for values, stream, decisions in run_decisions(replay):
        run = tuple(values[k] for k in runs_key)
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
    return StoppingCalibration(
        runs,
        _table(runs, groups),
        tuple(groups),
        dict(replay.settings),
        skipped=replay.skipped.copy(),
    )


def _share(mask: pd.Series) -> float:
    return float(mask.mean())


def _table(runs: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    rows = []
    keys = [*groups, "order", "strategy", "config_id", "criterion"]
    for key, block in runs.groupby(keys, sort=False, dropna=False):
        stopped = block["stopped"].astype(bool)
        at_stop = block["status_at_stop"]
        failed = stopped & (at_stop == Preservation.NOT_PRESERVED.value)
        # Not preserved by every field of the stream either: no later stop would have helped.
        out_of_reach = failed & (block["status_at_end"] == Preservation.NOT_PRESERVED.value)
        never = block[~stopped]
        stops = block.loc[stopped, "stop_fovs"].astype(float)
        rows.append(
            {
                **dict(zip(keys, key, strict=True)),
                "runs": len(block),
                "stopped": _share(stopped),
                "median_stop_fovs": float(np.median(stops)) if len(stops) else None,
                "median_stop_fovs_mc_se": median_standard_error(stops.to_numpy()),
                "correct": _share(stopped & (at_stop == Preservation.PRESERVED.value)),
                "not_preserved_at_stop": _share(failed),
                "premature": _share(failed & ~out_of_reach),
                "not_preserved_by_stream": _share(out_of_reach),
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


def _median_se(value: object) -> str:
    """`` (Monte Carlo SE 1.2)`` beside a median stop, or '' when it has none."""
    if value is None or pd.isna(value):
        return ""
    return f" (Monte Carlo SE {number(float(value))})"


def _failed_stops(row: Any) -> str:
    """The stops that did not preserve the criterion, with their premature split as bounds.

    The total not preserved at the stop is the headline; how many of them
    later fields of this stream fixed (premature) is a lower bound on the
    stops that came too early, since the held-back reference's own noise can
    put the whole stream outside the tolerance too.
    """
    reach = getattr(row, "not_preserved_by_stream", 0)
    reach = 0.0 if reach is None or pd.isna(reach) else float(reach)
    premature = float(row.premature)
    total = getattr(row, "not_preserved_at_stop", None)
    total = premature + reach if total is None or pd.isna(total) else float(total)
    text = f"{percent(total)} of acquisitions stopped where it was not preserved"
    if total <= 0 or reach <= 0:
        return text + ("" if total <= 0 else ", all of them too early")
    return (
        f"{text}; between {percent(premature)} and {percent(total)} stopped too early "
        f"(in {percent(premature)} every available field was not shown to fail it; in the "
        f"other {percent(reach)} not even every available field together preserved it, "
        "which the held-back fields' own noise can also cause)"
    )


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
                short = ""
                if first["stream_fovs"] < result.settings.get("min_fovs", 1):
                    short = (
                        f"; fewer than the {result.settings['min_fovs']} fields it waits for "
                        "before deciding"
                    )
                lines.append(
                    f"  {order_phrase(order)}: it never stopped within the "
                    f"{first['stream_fovs']} fields available "
                    f"({count(first['runs'], 'acquisition')}{short})."
                )
            else:
                lines.append(
                    f"  {order_phrase(order)}: it stopped in "
                    f"{percent(first['stopped'])} of {count(first['runs'], 'acquisition')}, "
                    f"after a median of {fields(first['median_stop_fovs'])}"
                    f"{_median_se(first.get('median_stop_fovs_mc_se'))}."
                )
            several = own["criterion"].nunique() > 2
            for row in own.itertuples(index=False):
                if row.criterion == ALL_CRITERIA and not several:
                    continue
                label = labels.get(row.criterion, row.criterion)
                if row.stopped == 0:
                    if row.never_stopped_preserved_at_end is not None and not pd.isna(
                        row.never_stopped_preserved_at_end
                    ):
                        lines.append(
                            f"    for {label}: "
                            f"{percent(row.never_stopped_preserved_at_end)} of those "
                            "acquisitions had it by the last field."
                        )
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
                    f"    for {label}: {_failed_stops(row)}, "
                    f"{percent(row.correct)} stopped with it preserved{undefined}{late}."
                )
        lines.append(
            "  Judged on held-back fields the rule never saw; each acquisition is one hold-out "
            "and one order of the same fields, not an independent experiment."
        )
        blocks.append("\n".join(lines))
    blocks += skipped_lines(result.skipped, keys)
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
