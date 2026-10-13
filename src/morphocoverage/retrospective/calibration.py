"""Stopping-rule calibration: when this rule says stop, how often is that premature?

Question P6 of ``docs/optimise/optimise.md``. Sequential, because a stopping rule can
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
under this rule. The Monte Carlo standard error of the median stop counts the
splits, not the runs: the orders of one split share its reservoir and stream.
For a fixed-FOV rule and random orders, the share of correct
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

from .._caveats import assumption, caveat_lines
from .._frozen import owned
from .._style import check_detail, group_columns, group_label, low_to_high, span, table, wrap
from .._wording import listing, number
from ..calibration import calibrate_next_fov_novelty, pool_calibration
from ..statistics import median_standard_error
from ._text import fields, order_name, percent, rule_names, skipped_lines
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


@owned("runs", "table", "settings", "skipped")
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
    the Monte Carlo standard error of the median stop with the reservoir split
    as the unit (``statistics.median_standard_error`` of each split's median
    stop over its orders that stopped, so more orders of the same splits do
    not shrink it; ``None`` below two splits); ``never_stopped_preserved_at_end``
    is a share of the runs that never stopped (``None`` when all stopped).
    ``skipped`` lists the groups the replay skipped, with the reason.
    """

    runs: pd.DataFrame
    table: pd.DataFrame
    group_cols: tuple[str, ...]
    settings: dict[str, Any] = field(default_factory=dict)
    skipped: pd.DataFrame = field(default_factory=pd.DataFrame)

    def summary(self, detail: str = "short") -> str:
        """``"short"``: one table per criterion, rules by order kind, shares over the groups;
        ``"full"``: the calibration in sentences, per group, strategy and order kind."""
        if check_detail(detail) == "short":
            return calibration_short(self)
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
        # Orders of one split share its reservoir and stream, so the split is the
        # Monte Carlo unit: one median stop per split (owner, 2026-10-05).
        split_stops = stops.groupby(block.loc[stopped, "split"], sort=True).median()
        rows.append(
            {
                **dict(zip(keys, key, strict=True)),
                "runs": len(block),
                "stopped": _share(stopped),
                "median_stop_fovs": float(np.median(stops)) if len(stops) else None,
                "median_stop_fovs_mc_se": median_standard_error(split_stops.to_numpy()),
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


def _not_preserved_cell(row: Any) -> str:
    """The stops that did not preserve the criterion, with how many of them were too early.

    The total not preserved at the stop is the headline; how many of them
    later fields of this stream fixed (premature) is a lower bound on the
    stops that came too early, since the held-back reference's own noise can
    put the whole stream outside the tolerance too: the cell gives the range.
    """
    reach = getattr(row, "not_preserved_by_stream", 0)
    reach = 0.0 if reach is None or pd.isna(reach) else float(reach)
    premature = float(row.premature)
    total = getattr(row, "not_preserved_at_stop", None)
    total = premature + reach if total is None or pd.isna(total) else float(total)
    if total <= 0:
        return percent(total)
    if reach <= 0:
        return f"{percent(total)}, all too early"
    return f"{percent(total)}, {low_to_high(percent(premature), percent(total))} too early"


def _never_cell(row: Any, min_fovs: int) -> str | None:
    """The acquisitions that never stopped, and how many had the criterion at their end."""
    if not row.never_stopped > 0:
        return None
    text = percent(row.never_stopped)
    at_end = row.never_stopped_preserved_at_end
    if at_end is not None and not pd.isna(at_end):
        text += f", {percent(at_end)} had it by the last field"
    if row.stopped == 0 and row.stream_fovs < min_fovs:
        text += (
            f" (within {fields(row.stream_fovs)}: fewer than the {min_fovs} fields it waits "
            "for before deciding)"
        )
    return text


def _calibration_row(row: Any, label: str | None, min_fovs: int) -> list[Any]:
    """One order kind (and criterion) of one rule and group, as table cells."""
    stopped = row.stopped > 0
    se = getattr(row, "median_stop_fovs_mc_se", None)
    median = None
    if stopped:
        median = number(row.median_stop_fovs)
        median += "" if se is None or pd.isna(se) else f" ({number(float(se))})"
    preserved = None
    if stopped:
        preserved = percent(row.correct)
        if row.undefined_at_stop > 0:
            preserved += f"; {percent(row.undefined_at_stop)} not judged"
    cells = [order_name(row.order)]
    cells += [label] if label is not None else []
    cells += [
        f"{percent(row.stopped)} of {row.runs}",
        median,
        _not_preserved_cell(row) if stopped else None,
        preserved,
        _never_cell(row, min_fovs),
    ]
    return cells


def _calibration_head(
    result: StoppingCalibration, labels: dict[Any, str], full: dict[Any, str]
) -> tuple[str, list[Any], bool]:
    """The groups, the rules (``full``: their names with settings) and the opening sentence
    of the full form, with the criteria it shows and whether there are several (then a
    "for" column names each)."""
    keys, frame = list(result.group_cols), result.table
    head = [f"Analysis groups: {group_columns(keys)}."] if keys else []
    configs = list(dict.fromkeys(frame["config_id"]))
    listed = "; ".join(full.get(c, str(c)) for c in configs)
    head.append(wrap(f"Rules: {listed}.", 0, hang=2))
    criteria = list(dict.fromkeys(frame["criterion"]))
    several = len(criteria) > 2
    if not several:
        criteria = [c for c in criteria if c != ALL_CRITERIA] or criteria
        judged = listing([labels.get(c, c) for c in criteria])
        head.append(
            wrap(
                f"If you had let each rule stop the acquisition, judged on held-back fields "
                f"the rule never saw, for {judged}:",
                0,
                hang=2,
            )
        )
    else:
        head.append(
            "If you had let each rule stop the acquisition, judged on held-back fields the "
            "rule never saw:"
        )
    return "\n".join(head), criteria, several


def _calibration_groups(
    result: StoppingCalibration,
    labels: dict[Any, str],
    names: dict[Any, str],
    min_fovs: int,
    criteria: list[Any],
    several: bool,
) -> list[str]:
    """One block per group: per rule (by its readable name in ``names``), a table with a row
    per order kind (and criterion)."""
    keys, frame = list(result.group_cols), result.table
    header = ["order", *(["for"] if several else []), "stopped", "median (MC SE)"]
    header += ["not preserved", "preserved", "never stopped"]
    parts = frame.groupby(keys, sort=False, dropna=False) if keys else [((), frame)]
    blocks = []
    for key, rows in parts:
        key = key if isinstance(key, tuple) else (key,)
        lines = [f"{group_label(dict(zip(keys, key, strict=True)))}:"]
        for config, mine in rows.groupby("config_id", sort=False):
            lines.append(f"  {names.get(config, mine['strategy'].iloc[0])}:")
            cells = [
                _calibration_row(
                    row, labels.get(row.criterion, row.criterion) if several else None, min_fovs
                )
                for row in mine.itertuples(index=False)
                if row.criterion in criteria
            ]
            lines.append(table(cells, header, indent=4))
        blocks.append("\n".join(lines))
    return blocks


def _calibration_cautions(frame: pd.DataFrame) -> list[str]:
    """What every row of the full form rests on, and the range "too early" is, if any."""
    cautions = [
        assumption(
            "each acquisition is one hold-out and one order of the same fields, not an "
            "independent experiment"
        )
    ]
    if len(frame) and (frame["not_preserved_by_stream"].fillna(0) > 0).any():
        cautions.append(
            assumption(
                "'too early' is a range: its low end counts the stops that later fields of the "
                "stream fixed; in the rest not even every available field together preserved "
                "it, which the held-back fields' own noise can also cause"
            )
        )
    return cautions


def calibration_summary(result: StoppingCalibration) -> str:
    """ "If you had let this rule stop the acquisition": the full form, in tables.

    Per group (``MOCK · 1A · B6``, the columns named once) and rule (by its
    readable name; the strategies with their settings and config ids are
    listed once), one row per order kind (and criterion, when several are
    judged): the acquisitions that stopped and their median stop with its
    Monte Carlo SE, the stops where the criterion was not preserved (with the
    range that came too early), the stops that preserved it, and the
    acquisitions that never stopped, with how many had it by their last
    field. What every row rests on is said once, at the end.
    """
    keys = list(result.group_cols)
    frame = result.table
    labels = {ALL_CRITERIA: "everything you wanted", **result.settings.get("labels", {})}
    names, full = rule_names(result.settings), rule_names(result.settings, detail="full")
    min_fovs = int(result.settings.get("min_fovs", 1) or 1)
    blocks = []
    if len(frame):
        head, criteria, several = _calibration_head(result, labels, full)
        blocks.append(head)
        blocks += _calibration_groups(result, labels, names, min_fovs, criteria, several)
    cautions = _calibration_cautions(frame)
    if len(frame):
        blocks.append("\n".join(caveat_lines(cautions, indent=0, detail="full")))
    blocks += skipped_lines(result.skipped, keys)
    return "\n\n".join(blocks)


# Above this share of stops where the criterion was not preserved, the report's status for
# "Did I image enough?" asks the reader to look (report_status; a provisional preset).
PREMATURE_CAUTION = 0.1


def _share_cell(rows: pd.DataFrame) -> str:
    """The share of stops that did not preserve the criterion, over the groups."""
    stopped = rows[rows["stopped"] > 0]
    if stopped.empty:
        return "never stopped"
    text = span(stopped["not_preserved_at_stop"], "percent")
    never = len(rows) - len(stopped)
    return f"{text}; {never} never stopped" if never else text


def calibration_short(result: StoppingCalibration) -> str:
    """The calibration across groups: per criterion, each rule's share of stops where the
    criterion was not preserved, by order kind (the range over the groups)."""
    lines, cautions = calibration_short_parts(result)
    return "\n".join([*lines, *caveat_lines(cautions, indent=0)])


def calibration_short_parts(
    result: StoppingCalibration, indent: int = 0
) -> tuple[list[str], list[str]]:
    """The short form's lines (under ``indent``) and its caveats, apart (see
    ``calibration_short``)."""
    frame = result.table
    if frame.empty:
        return skipped_lines(result.skipped, list(result.group_cols)), []
    labels = {ALL_CRITERIA: "everything you wanted", **result.settings.get("labels", {})}
    names = rule_names(result.settings)
    criteria = list(dict.fromkeys(frame["criterion"]))
    if len(criteria) <= 2:
        criteria = [c for c in criteria if c != ALL_CRITERIA] or criteria
    groups = len(frame[list(result.group_cols)].drop_duplicates()) if result.group_cols else 1
    kinds = list(dict.fromkeys(frame["order"]))
    lines = []
    for criterion in criteria:
        own = frame[frame["criterion"].eq(criterion)]
        rows = []
        for config, mine in own.groupby("config_id", sort=False):
            cells = [names.get(config, mine["strategy"].iloc[0])]
            cells += [_share_cell(mine[mine["order"].eq(kind)]) for kind in kinds]
            rows.append(cells)
        over = "" if groups == 1 else f", the range over {groups} groups"
        lines.append(
            wrap(
                f"If you had let each rule stop the acquisition: the share of acquisitions that "
                f"stopped where {labels.get(criterion, criterion)} was not preserved in held-back "
                f"fields{over}:",
                indent,
                hang=indent + 2,
            )
        )
        header = ["rule", *(order_name(kind) for kind in kinds)]
        lines.append(table(rows, header, indent=indent + 2))
    cautions = [
        assumption(
            "judged on held-back fields the rule never saw; each acquisition is one hold-out "
            "and one order of the same fields, not an independent experiment"
        )
    ]
    if (frame["not_preserved_by_stream"].fillna(0) > 0).any():
        cautions.append(
            assumption(
                "a stop counted as not preserved includes stops where not even every available "
                "field preserved it, which the held-back fields' own noise can cause: the full "
                "report splits the two"
            )
        )
    lines += [
        " " * indent + line for line in skipped_lines(result.skipped, list(result.group_cols))
    ]
    return lines, cautions


__all__ = [
    "PREMATURE_CAUTION",
    "RUN_COLUMNS",
    "SUMMARY_COLUMNS",
    "StoppingCalibration",
    "calibrate_next_fov_novelty",
    "calibration_short",
    "calibration_short_parts",
    "calibration_summary",
    "pool_calibration",
    "stopping_calibration",
]
