"""Could the same positions have been visited in a better order? Acquisition-order efficiency.

Questions P7 and P8 of ``docs/questions.md``. This asks about the order of the acquired
positions, not about whether they were the right positions (that is
sampling-frame bias, P9, not implemented):

- ``dispersed_order``: greedy farthest-point order on stage coordinates, from
  a starting FOV; ``AcquisitionOrder.dispersed(seed)`` replays it with a start
  chosen by the seed, compared over several seeds like random orders, so a
  result does not hinge on the first field.
- ``stage_travel``: the summed straight-line stage path between successive
  FOVs; with per-FOV and per-move times and a stage speed, an acquisition time.
- ``order_efficiency``: per sample and order kind, the four adequacy points of
  ``adequacy_points`` with the stage travel needed to reach sustained realized
  adequacy. Runs in which the criteria could not be judged at any field count
  are left out of the medians and counted (``unjudged``), as the adequacy
  summary does, rather than counted as never enough; groups the replay
  skipped are named.
- ``redundancy_distance``: from the semivariogram of FOV means, the distance
  below which neighbouring fields are more alike than random pairs. It states
  a limitation of closely spaced fields; it is not an adequacy claim.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .._wording import count, duration
from ..data_check import validate_input_tables
from ..evaluator import iter_analysis_groups
from ..spatial import fov_means, fov_semivariogram
from ._text import fields as fields_text
from ._text import median_beyond, order_phrase, percent, skipped_lines, where
from .adequacy import ALL_CRITERIA, AdequacyResult, _unjudged, adequacy_points
from .replay import ReplayResult
from .stage import dispersed_order, stage_travel


@dataclass(frozen=True)
class EfficiencyResult:
    """Per sample and order kind: adequacy points, stage travel, and acquisition time.

    ``runs`` has one row per replayed acquisition (group, split, order), with
    ``judged`` false for a run in which the criteria could not be judged at
    any field count; ``table`` the medians per group and order kind over the
    judged runs, with ``unjudged`` the runs left out. Travel is in the stage
    units; time in seconds when the timing settings were given. ``skipped``
    lists the groups the replay skipped, with the reason.
    """

    runs: pd.DataFrame
    table: pd.DataFrame
    group_cols: tuple[str, ...]
    settings: dict[str, Any] = field(default_factory=dict)
    skipped: pd.DataFrame = field(default_factory=pd.DataFrame)

    def summary(self) -> str:
        """The order comparison in sentences."""
        return efficiency_summary(self)


def _time(
    fovs: int, travel: float, per_fov: float | None, per_move: float | None, speed: float | None
) -> float | None:
    if per_fov is None and per_move is None and speed is None:
        return None
    seconds = (per_fov or 0.0) * fovs + (per_move or 0.0) * max(fovs - 1, 0)
    if speed is not None:
        seconds += travel / speed
    return float(seconds)


def order_efficiency(
    replay: ReplayResult,
    adequacy: AdequacyResult | None = None,
    *,
    per_fov_seconds: float | None = None,
    per_move_seconds: float | None = None,
    stage_speed: float | None = None,
) -> EfficiencyResult:
    """Fields and stage travel each order kind needed to be reliably enough.

    ``replay`` must come from ``replay_acquisitions(..., stage_cols=...)``.
    ``adequacy`` defaults to ``adequacy_points(replay)``; one computed from
    another replay (other settings, other data, so other hold-outs, orders or
    outcomes) is refused. The realized points are for all criteria together.
    ``stage_speed`` is in stage units per second.

    ``median_fovs`` (and the travel and time medians) count a run that never
    reached adequacy as beyond its own stream, so it is ``None`` when half or
    more of the runs never got there; ``stream_fovs`` is the shortest stream
    and ``longest_stream_fovs`` the longest, which differ with unequal blocks.
    Runs that could not be judged at any field count are left out of these
    (``unjudged`` counts them): they say nothing about how soon an order
    became enough. ``reached`` is a share of the judged runs.
    """
    if not isinstance(replay, ReplayResult):
        raise TypeError("replay must be a ReplayResult")
    if replay.paths.empty:
        raise ValueError("the replay has no stage paths; run replay_acquisitions with stage_cols")
    for value, label in (
        (per_fov_seconds, "per_fov_seconds"),
        (per_move_seconds, "per_move_seconds"),
        (stage_speed, "stage_speed"),
    ):
        if value is not None and not float(value) > 0:
            raise ValueError(f"{label} must be positive")
    adequacy = adequacy_points(replay) if adequacy is None else adequacy
    _check_same_replay(replay, adequacy)
    groups = list(replay.group_cols)
    run_keys = [*groups, "split", "order_id"]
    realized = adequacy.realized[adequacy.realized["criterion"] == ALL_CRITERIA]
    unjudged = _unjudged(realized)
    travel = {
        key: rows.set_index("fovs")["travel"]
        for key, rows in replay.paths.groupby(run_keys, sort=False, dropna=False)
    }
    rows = []
    for index, run in realized.iterrows():
        key = tuple(run[k] for k in run_keys)
        path = travel[key]
        stream = int(run["stream_fovs"])
        needed = run["sustained_realized"]
        reached = not pd.isna(needed)
        at = int(needed) if reached else stream
        rows.append(
            {
                **{k: run[k] for k in run_keys},
                "order": run["order"],
                "first_realized": run["first_realized"],
                "sustained_realized": needed,
                "travel_to_sustained": float(path.loc[at]) if reached else None,
                "travel_total": float(path.loc[stream]),
                "time_to_sustained": (
                    _time(at, float(path.loc[at]), per_fov_seconds, per_move_seconds, stage_speed)
                    if reached
                    else None
                ),
                "stream_fovs": stream,
                "judged": not bool(unjudged.loc[index]),
            }
        )
    runs = pd.DataFrame(rows)
    table_rows = []
    for key, block in runs.groupby([*groups, "order"], sort=False, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        judged = block[block["judged"]]
        needed = judged["sustained_realized"].astype("Float64")
        table_rows.append(
            {
                **dict(zip([*groups, "order"], key, strict=True)),
                "runs": len(block),
                "unjudged": int((~block["judged"]).sum()),
                "reached": float(needed.notna().mean()) if len(judged) else None,
                # Not reached counts as beyond the run's own stream, whatever its
                # length, so the median stays honest with unequal streams too.
                "median_fovs": median_beyond(needed),
                # Over the same runs as median_fovs: a run that never got there
                # travelled and took longer than any that did.
                "median_travel": median_beyond(judged["travel_to_sustained"]),
                "median_travel_total": float(block["travel_total"].median()),
                "median_time": median_beyond(judged["time_to_sustained"]),
                "stream_fovs": int(block["stream_fovs"].min()),
                "longest_stream_fovs": int(block["stream_fovs"].max()),
            }
        )
    settings = {
        **dict(replay.settings),
        "per_fov_seconds": per_fov_seconds,
        "per_move_seconds": per_move_seconds,
        "stage_speed": stage_speed,
    }
    return EfficiencyResult(
        runs, pd.DataFrame(table_rows), tuple(groups), settings, skipped=replay.skipped.copy()
    )


def _check_same_replay(replay: ReplayResult, adequacy: AdequacyResult) -> None:
    """Refuse adequacy points from another replay: its runs are other hold-outs and orders.

    Equal settings are not enough: a replay of other data with the same
    settings has other outcomes, so the realized points must also be those of
    ``adequacy_points(replay)``.
    """
    if not isinstance(adequacy, AdequacyResult):
        raise TypeError("adequacy must be an AdequacyResult (adequacy_points(replay))")
    theirs, ours = dict(adequacy.settings), dict(replay.settings)
    differ = sorted(k for k in {*theirs, *ours} if theirs.get(k) != ours.get(k))
    if differ or tuple(adequacy.group_cols) != tuple(replay.group_cols):
        raise ValueError(
            "adequacy comes from another replay (settings differ: "
            f"{', '.join(differ) or 'group_cols'}): its runs are other hold-outs and orders; "
            "pass adequacy_points(replay) of this replay, or leave adequacy out"
        )
    own = adequacy_points(replay).realized.reset_index(drop=True)
    given = adequacy.realized.reset_index(drop=True)
    if not (list(own.columns) == list(given.columns) and own.equals(given[own.columns])):
        raise ValueError(
            "adequacy comes from another replay (same settings, other realized points: "
            "other data or other outcomes); pass adequacy_points(replay) of this replay, "
            "or leave adequacy out"
        )


def _distance(value: float, unit: str) -> str:
    if unit == "µm" and value >= 1000:
        return f"{value / 1000:.1f} mm"
    return f"{value:.0f} {unit}"


def _available(row: Any) -> str:
    """``the 12 fields``, or ``the 12 to 24 fields`` when the streams differ in length."""
    longest = getattr(row, "longest_stream_fovs", row.stream_fovs)
    longest = row.stream_fovs if pd.isna(longest) else int(longest)
    span = f"{row.stream_fovs}" if longest == row.stream_fovs else f"{row.stream_fovs} to {longest}"
    return f"the {span} fields"


def efficiency_summary(result: EfficiencyResult) -> str:
    """ "Your raster needed 28 fields and 7 mm; spreading them out ..." in sentences."""
    keys = list(result.group_cols)
    unit = result.settings.get("stage_unit", "µm")
    # The realized points are for every criterion together.
    judged = (
        "for this criterion"
        if len(result.settings.get("labels", {})) <= 1
        else ("for all the criteria together")
    )
    blocks = []
    for key, rows in (
        result.table.groupby(keys, sort=False, dropna=False) if keys else [((), result.table)]
    ):
        key = key if isinstance(key, tuple) else (key,)
        group = dict(zip(keys, key, strict=True))
        lines = [f"{where(group)}: could the same positions have been visited in a better order?"]
        for row in rows.itertuples(index=False):
            label = order_phrase(row.order, capital=True)
            unjudged = int(getattr(row, "unjudged", 0) or 0)
            left_out = (
                f"; {unjudged} of {count(row.runs, 'run')} could not be judged and are left out"
                if unjudged
                else ""
            )
            if unjudged == row.runs:
                lines.append(
                    f"  {label}: could not be judged at any field count "
                    f"(the outcome was undefined in every run)."
                )
                continue
            if row.median_fovs is None or pd.isna(row.median_fovs):
                lines.append(
                    f"  {label}: not reliably enough within {_available(row)} in half or more "
                    f"of runs (never in {percent(1 - row.reached)} of runs{left_out})."
                )
                continue
            travel = (
                ""
                if row.median_travel is None or pd.isna(row.median_travel)
                else f" and {_distance(row.median_travel, unit)} of stage travel"
            )
            time = (
                ""
                if row.median_time is None or pd.isna(row.median_time)
                else f" (about {duration(row.median_time)})"
            )
            fields = fields_text(row.median_fovs)
            # The median hides runs that never got there; a protocol must serve those too.
            missed = (
                f" (never within {_available(row)} in {percent(1 - row.reached)} of runs{left_out})"
                if row.reached < 1
                else (f" ({left_out.removeprefix('; ')})" if left_out else "")
            )
            lines.append(f"  {label}: reliably enough after {fields}{travel}{time}{missed}.")
        reached = rows[rows["median_fovs"].notna()]
        if len(reached) == len(rows) and len(rows) > 1:
            spread = float(reached["median_fovs"].max() - reached["median_fovs"].min())
            if spread <= 1:
                lines.append(
                    f"  The orders do not differ by more than one field {judged}: "
                    "a null result, not a preference for any order."
                )
        lines.append(
            "  Medians over hold-outs and orders; stage travel is the straight-line path "
            "between successive fields."
        )
        blocks.append("\n".join(lines))
    blocks += skipped_lines(result.skipped, keys)
    return "\n\n".join(blocks)


def redundancy_distance(
    objects: pd.DataFrame,
    fovs: pd.DataFrame,
    *,
    feature_cols: Sequence[str],
    stage_cols: tuple[str, str],
    bin_edges: Sequence[float],
    share: float = 0.9,
    stage_unit: str = "µm",
    fov_col: str | None = None,
    group_cols: Sequence[str] | None = None,
    observation_unit: str | None = None,
) -> pd.DataFrame:
    """Per group, the distance below which fields are more alike than random pairs.

    From the semivariogram of FOV means (``fov_semivariogram``): the smallest
    mean lag at which the semivariance reaches ``share`` of the variance of
    the FOV means (``None`` with no ``reason`` when no bin reaches it: fields
    are then alike at every distance measured, up to ``largest_lag``, the
    strongest redundancy). Also reports the smallest lag measured, since
    nothing is said below it. A group with fewer than two fields with
    observations, with a missing feature value (its FOV means are undefined),
    with FOV means that do not vary at all (no distance can be told from
    them), or with no pair of fields within the bins, has nothing measured:
    its row holds ``None`` and says why in ``reason``, and the other groups
    are still computed. With several features, standardize them beforehand if
    their scales differ.
    """
    if not 0 < float(share) <= 1:
        raise ValueError("share must be in (0, 1]")
    data = validate_input_tables(
        objects,
        fovs=fovs,
        feature_cols=feature_cols,
        fov_col=fov_col,
        group_cols=group_cols,
        observation_unit=observation_unit,
    )
    for column in stage_cols:
        if column not in data.fovs.columns:
            raise ValueError(f"{column!r} is not a column of the FOV table")
    rows = []
    for group, group_data in iter_analysis_groups(data):
        row = {
            **group,
            "redundancy_distance": None,
            "smallest_lag": None,
            "largest_lag": None,
            "fov_mean_variance": None,
            "share": float(share),
            "unit": stage_unit,
            "reason": None,
        }
        rows.append(row)
        if np.isnan(group_data.feature_matrix()).any():
            row["reason"] = (
                "missing feature values in this group: its FOV means are undefined "
                "(missing values are never imputed)"
            )
            continue
        means, _ = fov_means(group_data)
        if len(means) < 2:
            row["reason"] = (
                f"only {fields_text(len(means))} with observations: the semivariogram needs "
                "pairs of fields"
            )
            continue
        gamma = fov_semivariogram(group_data, *stage_cols, bin_edges)
        if gamma.empty:
            row["reason"] = (
                f"no pair of fields is between {bin_edges[0]:g} and {bin_edges[-1]:g} "
                f"{stage_unit} apart: widen bin_edges"
            )
            continue
        variance = float(np.mean(np.var(means, axis=0, ddof=1)))
        if not np.ptp(means, axis=0).any():
            # Every semivariance is 0 and reaches 90% of a variance of 0 at the
            # first bin: that would read as the shortest redundancy distance.
            row["fov_mean_variance"] = variance
            row["reason"] = (
                "the FOV means do not vary at all, so no distance at which fields become "
                "alike can be measured"
            )
            continue
        reached = gamma[gamma["gamma"] >= float(share) * variance]
        row.update(
            {
                "redundancy_distance": (
                    float(reached["mean_lag"].iloc[0]) if len(reached) else None
                ),
                "smallest_lag": float(gamma["mean_lag"].iloc[0]),
                "largest_lag": float(gamma["mean_lag"].iloc[-1]),
                "fov_mean_variance": variance,
            }
        )
    return pd.DataFrame(rows)


__all__ = [
    "EfficiencyResult",
    "dispersed_order",
    "efficiency_summary",
    "order_efficiency",
    "redundancy_distance",
    "stage_travel",
]
