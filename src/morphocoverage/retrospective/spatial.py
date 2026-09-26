"""Could the same positions have been visited in a better order? Acquisition-order efficiency.

Roadmap Step 8. This asks about the order of the acquired positions, not about
whether they were the right positions (that is sampling-frame bias, roadmap
Step 9, not implemented):

- ``dispersed_order``: greedy farthest-point order on stage coordinates, from
  a starting FOV; ``AcquisitionOrder.dispersed(seed)`` replays it with a start
  chosen by the seed, compared over several seeds like random orders, so a
  result does not hinge on the first field.
- ``stage_travel``: the summed straight-line stage path between successive
  FOVs; with per-FOV and per-move times and a stage speed, an acquisition time.
- ``order_efficiency``: per sample and order kind, the four adequacy points of
  Step 3 with the stage travel needed to reach sustained realized adequacy.
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

from ..evaluator import iter_analysis_groups
from ..schema import validate_input_tables
from ..spatial import fov_means, fov_semivariogram
from .adequacy import ALL_CRITERIA, AdequacyResult, adequacy_points
from .replay import ReplayResult
from .stage import dispersed_order, stage_travel


@dataclass(frozen=True)
class EfficiencyResult:
    """Per sample and order kind: adequacy points, stage travel, and acquisition time.

    ``runs`` has one row per replayed acquisition (group, split, order);
    ``table`` the medians per group and order kind. Travel is in the stage
    units; time in seconds when the timing settings were given.
    """

    runs: pd.DataFrame
    table: pd.DataFrame
    group_cols: tuple[str, ...]
    settings: dict[str, Any] = field(default_factory=dict)

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
    ``adequacy`` defaults to ``adequacy_points(replay)``. The realized points
    are for all criteria together. ``stage_speed`` is in stage units per
    second.
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
    groups = list(replay.group_cols)
    run_keys = [*groups, "split", "order_id"]
    realized = adequacy.realized[adequacy.realized["criterion"] == ALL_CRITERIA]
    travel = {
        key: rows.set_index("fovs")["travel"]
        for key, rows in replay.paths.groupby(run_keys, sort=False, dropna=False)
    }
    rows = []
    for _, run in realized.iterrows():
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
            }
        )
    runs = pd.DataFrame(rows)
    table_rows = []
    for key, block in runs.groupby([*groups, "order"], sort=False, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        needed = block["sustained_realized"].astype("Float64")
        stream = int(block["stream_fovs"].min())
        table_rows.append(
            {
                **dict(zip([*groups, "order"], key, strict=True)),
                "runs": len(block),
                "reached": float(needed.notna().mean()),
                # Not reached counts as beyond the stream, so the median stays honest.
                "median_fovs": float(needed.fillna(stream + 1).median()),
                "median_travel": (
                    float(block["travel_to_sustained"].dropna().median())
                    if block["travel_to_sustained"].notna().any()
                    else None
                ),
                "median_travel_total": float(block["travel_total"].median()),
                "median_time": (
                    float(block["time_to_sustained"].dropna().median())
                    if block["time_to_sustained"].notna().any()
                    else None
                ),
                "stream_fovs": stream,
            }
        )
    settings = {
        **dict(replay.settings),
        "per_fov_seconds": per_fov_seconds,
        "per_move_seconds": per_move_seconds,
        "stage_speed": stage_speed,
    }
    return EfficiencyResult(runs, pd.DataFrame(table_rows), tuple(groups), settings)


_ORDER_WORDS = {
    "recorded": "your recorded order",
    "permuted": "random orders",
    "dispersed": "spreading the same positions out",
}


def _distance(value: float, unit: str) -> str:
    if unit == "µm" and value >= 1000:
        return f"{value / 1000:.1f} mm"
    return f"{value:.0f} {unit}"


def efficiency_summary(result: EfficiencyResult) -> str:
    """ "Your raster needed 28 fields and 7 mm; spreading them out ..." in sentences."""
    keys = list(result.group_cols)
    unit = result.settings.get("stage_unit", "µm")
    blocks = []
    for key, rows in (
        result.table.groupby(keys, sort=False, dropna=False) if keys else [((), result.table)]
    ):
        key = key if isinstance(key, tuple) else (key,)
        where = ", ".join(f"{k}={v}" for k, v in zip(keys, key, strict=True)) or "all FOVs"
        lines = [f"{where}: could the same positions have been visited in a better order?"]
        for row in rows.itertuples(index=False):
            label = _ORDER_WORDS.get(row.order, row.order)
            if row.median_fovs > row.stream_fovs:
                lines.append(
                    f"  {label}: not reliably enough within the {row.stream_fovs} fields "
                    f"(in {100 * (1 - row.reached):.0f}% of runs)."
                )
                continue
            travel = (
                "" if row.median_travel is None else f" and {_distance(row.median_travel, unit)}"
            )
            time = "" if row.median_time is None else f" (about {row.median_time / 60:.0f} min)"
            fields = "1 field" if row.median_fovs == 1 else f"{row.median_fovs:.0f} fields"
            # The median hides runs that never got there; a protocol must serve those too.
            missed = (
                f" (never within the {row.stream_fovs} fields in "
                f"{100 * (1 - row.reached):.0f}% of runs)"
                if row.reached < 1
                else ""
            )
            lines.append(
                f"  {label}: reliably enough after {fields}{travel} of stage travel{time}{missed}."
            )
        reached = rows[rows["median_fovs"] <= rows["stream_fovs"]]
        if len(reached) == len(rows) and len(rows) > 1:
            spread = float(reached["median_fovs"].max() - reached["median_fovs"].min())
            if spread <= 1:
                lines.append(
                    "  The orders do not differ by more than one field for this criterion: "
                    "a null result, not a preference for any order."
                )
        lines.append(
            "  Medians over hold-outs and orders; stage travel is the straight-line path "
            "between successive fields."
        )
        blocks.append("\n".join(lines))
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
    the FOV means (``None`` when no bin reaches it: fields are then alike at
    every distance measured, up to ``largest_lag``, the strongest redundancy).
    Also reports the smallest lag measured, since nothing is said below it. With several
    features, standardize them beforehand if their scales differ.
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
    rows = []
    for group, group_data in iter_analysis_groups(data):
        gamma = fov_semivariogram(group_data, *stage_cols, bin_edges)
        means, _ = fov_means(group_data)
        variance = float(np.mean(np.var(means, axis=0, ddof=1))) if len(means) > 1 else None
        reached = (
            gamma[gamma["gamma"] >= float(share) * variance] if variance is not None else gamma[:0]
        )
        rows.append(
            {
                **group,
                "redundancy_distance": (
                    float(reached["mean_lag"].iloc[0]) if len(reached) else None
                ),
                "smallest_lag": float(gamma["mean_lag"].iloc[0]) if len(gamma) else None,
                "largest_lag": float(gamma["mean_lag"].iloc[-1]) if len(gamma) else None,
                "fov_mean_variance": variance,
                "share": float(share),
                "unit": stage_unit,
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
