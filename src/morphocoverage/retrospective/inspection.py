"""Which fields should I go back and look at? A traceable list with one reason per line.

Roadmap Step 5. ``inspect_fovs`` lists, per analysis group, the FOVs worth
reopening, each with a plain reason and the numbers behind it, the image file
and stage position when the FOV table has them, and its observation count
beside the group median. Each reason is computed independently and reported
only above its declared threshold; there is no combined score, and no FOV is
called bad, an outlier or an artifact. An influential FOV may be biologically
crucial, a technical artifact, or simply large, and only the image can tell.

Reasons:

- **new state** (with ``novelty``): the FOV holds a cluster of observations
  the other FOVs do not show (support novelty, leave one FOV out); the count of
  those observations is given.
- **influence** (with ``summaries``): removing the FOV changes a declared
  summary (the mean or a quantile of a named feature) by at least
  ``influence_threshold`` in relative terms. Summaries weight every FOV
  equally, so duplicating cells within a FOV does not raise its influence
  beyond its effect as one FOV; a dense FOV is reported by the count reason
  instead.
- **count**: the observation count is at least ``count_ratio`` times the group
  median, at most the median divided by it, or zero.
- **neighbours** (with stage coordinates): the FOV's mean of some feature
  differs from the mean of its ``n_neighbours`` nearest FOVs on the stage by at
  least ``neighbour_threshold`` standard deviations of the FOV means.

Missing feature values: the influence and neighbour reasons use the
non-missing values of each feature; the novelty reason follows the estimator,
which reports nothing when values are missing.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from ..evaluator import iter_analysis_groups
from ..schema import ValidatedInput, fov_positions, validate_input_tables
from ..strategies.support_novelty import SupportNoveltyEstimator

REASONS = ("new state", "influence", "count", "neighbours")
_TRACE = ("image_file", "position_id", "mask_file")


@dataclass(frozen=True)
class Summary:
    """A declared summary of one feature: ``"mean"`` or a quantile in (0, 1)."""

    feature: str
    statistic: str | float

    def __post_init__(self) -> None:
        if not isinstance(self.feature, str):
            raise TypeError("feature must be a column name")
        if self.statistic != "mean":
            if isinstance(self.statistic, bool) or not isinstance(self.statistic, int | float):
                raise ValueError("statistic must be 'mean' or a quantile between 0 and 1")
            if not 0 < float(self.statistic) < 1:
                raise ValueError("statistic must be 'mean' or a quantile between 0 and 1")

    @property
    def label(self) -> str:
        """``mean of area`` or ``95th percentile of area``."""
        if self.statistic == "mean":
            return f"mean of {self.feature}"
        percent = 100 * float(self.statistic)
        return f"{percent:g}th percentile of {self.feature}"


@dataclass(frozen=True)
class InspectionResult:
    """One row per FOV and reason (``fovs``), and the settings used."""

    fovs: pd.DataFrame
    group_cols: tuple[str, ...]
    settings: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """The list in sentences, per group."""
        return inspection_summary(self)


def _weighted(values: np.ndarray, weights: np.ndarray, statistic) -> float | None:
    keep = ~np.isnan(values)
    values, weights = values[keep], weights[keep]
    if not len(values) or weights.sum() <= 0:
        return None
    if statistic == "mean":
        return float(np.average(values, weights=weights))
    # Interpolated weighted quantile (weighted Hazen plotting positions): a
    # step-function quantile jumps between observations, so leaving out any
    # FOV could move it by a whole gap and look influential.
    order = np.argsort(values, kind="stable")
    values, weights = values[order], weights[order]
    positions = (np.cumsum(weights) - weights / 2) / weights.sum()
    return float(np.interp(float(statistic), positions, values))


def _influence(data: ValidatedInput, summaries, threshold) -> list[dict]:
    """Relative change of each FOV-weighted summary when one FOV is left out."""
    positions = fov_positions(data, data.fovs)
    counts = np.bincount(positions, minlength=len(data.fovs)).astype(float)
    rows = []
    for summary in summaries:
        values = data.feature_values(summary.feature)
        weights = 1.0 / counts[positions]
        full = _weighted(values, weights, summary.statistic)
        if full is None or full == 0:
            continue
        for fov in np.unique(positions):
            keep = positions != fov
            without = _weighted(values[keep], weights[keep], summary.statistic)
            if without is None:
                continue
            change = (without - full) / abs(full)
            if abs(change) >= threshold:
                direction = "lowers" if change < 0 else "raises"
                rows.append(
                    {
                        "position": int(fov),
                        "reason": "influence",
                        "value": float(change),
                        "detail": (
                            f"removing it {direction} the {summary.label} by "
                            f"{100 * abs(change):.0f}%"
                        ),
                    }
                )
    return rows


def _counts(data: ValidatedInput, ratio: float) -> list[dict]:
    positions = fov_positions(data, data.fovs)
    counts = np.bincount(positions, minlength=len(data.fovs))
    median = float(np.median(counts))
    rows = []
    for fov, count in enumerate(counts):
        if count == 0 and median > 0:
            detail = f"no observations; the median is {median:g}"
        elif median > 0 and (count >= ratio * median or count <= median / ratio):
            detail = f"{count} observations; the median is {median:g}"
        else:
            continue
        rows.append({"position": fov, "reason": "count", "value": float(count), "detail": detail})
    return rows


def _neighbours(data: ValidatedInput, x_col, y_col, k, threshold, unit) -> list[dict]:
    fovs = data.fovs
    xy = fovs[[x_col, y_col]].to_numpy(dtype=float)
    if len(fovs) <= k or np.isnan(xy).any():
        return []
    positions = fov_positions(data, fovs)
    frame = pd.DataFrame(data.feature_matrix(), columns=list(data.feature_cols))
    means = frame.groupby(positions).mean().reindex(range(len(fovs)))
    tree = cKDTree(xy)
    distances, near = tree.query(xy, k=k + 1)
    rows = []
    for feature in data.feature_cols:
        column = means[feature].to_numpy(dtype=float)
        spread = np.nanstd(column, ddof=1)
        if not np.isfinite(spread) or spread == 0:
            continue
        for fov in range(len(fovs)):
            others = column[near[fov, 1:]]
            if np.isnan(column[fov]) or np.isnan(others).all():
                continue
            z = (column[fov] - np.nanmean(others)) / spread
            if abs(z) >= threshold:
                rows.append(
                    {
                        "position": fov,
                        "reason": "neighbours",
                        "value": float(z),
                        "detail": (
                            f"its mean {feature} is {abs(z):.1f} SD "
                            f"{'above' if z > 0 else 'below'} its {k} nearest fields "
                            f"(within {distances[fov, -1]:.0f} {unit})"
                        ),
                    }
                )
    return rows


def _novel(data: ValidatedInput, novelty: SupportNoveltyEstimator) -> list[dict]:
    flags = novelty.novel_observation_flags(data, new_states_only=True)
    if flags is None or not flags.any():
        return []
    positions = fov_positions(data, data.fovs)
    counts = np.bincount(positions[flags], minlength=len(data.fovs))
    return [
        {
            "position": fov,
            "reason": "new state",
            "value": float(count),
            "detail": f"{count} observations of a state no other field shows",
        }
        for fov, count in enumerate(counts)
        if count
    ]


def _check(summaries, novelty, influence_threshold, count_ratio, stage, neighbour_threshold, k):
    summaries = [s if isinstance(s, Summary) else Summary(*s) for s in summaries]
    if novelty is not None and not isinstance(novelty, SupportNoveltyEstimator):
        raise TypeError("novelty must be a SupportNoveltyEstimator")
    if influence_threshold is not None and not 0 < float(influence_threshold):
        raise ValueError("influence_threshold must be positive")
    if count_ratio is not None and not float(count_ratio) > 1:
        raise ValueError("count_ratio must be greater than 1, or None to skip the count reason")
    if not float(neighbour_threshold) > 0:
        raise ValueError("neighbour_threshold must be positive")
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("n_neighbours must be a positive integer")
    if stage is not None and (len(stage) != 2 or not all(isinstance(c, str) for c in stage)):
        raise TypeError("stage_cols must be two column names (x, y)")
    return summaries


def inspect_fovs(
    objects: pd.DataFrame,
    fovs: pd.DataFrame,
    *,
    feature_cols: Sequence[str],
    summaries: Sequence[Summary | tuple] = (),
    novelty: SupportNoveltyEstimator | None = None,
    influence_threshold: float = 0.10,
    count_ratio: float | None = 3.0,
    stage_cols: tuple[str, str] | None = None,
    stage_unit: str = "µm",
    n_neighbours: int = 4,
    neighbour_threshold: float = 3.0,
    fov_col: str | None = None,
    group_cols: Sequence[str] | None = None,
    observation_unit: str | None = None,
) -> InspectionResult:
    """The fields worth reopening, one row per FOV and reason, per analysis group.

    ``summaries`` are the summaries whose influence is checked, as
    ``Summary(feature, "mean")`` or ``Summary(feature, 0.95)`` (or plain
    tuples). ``novelty`` turns on the new-state reason; ``stage_cols`` the
    neighbour reason, on every feature in ``feature_cols``; ``count_ratio=None``
    turns the count reason off. Thresholds are provisional presets: a 10% relative
    change, a count 3 times the median (or a third of it), 3 SD from the 4
    nearest fields.
    """
    summaries = _check(
        summaries,
        novelty,
        influence_threshold,
        count_ratio,
        stage_cols,
        neighbour_threshold,
        n_neighbours,
    )
    missing = [s.feature for s in summaries if s.feature not in feature_cols]
    if missing:
        raise ValueError(f"summary features must be among feature_cols: {missing}")
    data = validate_input_tables(
        objects,
        fovs=fovs,
        feature_cols=feature_cols,
        fov_col=fov_col,
        group_cols=group_cols,
        observation_unit=observation_unit,
    )
    if stage_cols is not None:
        absent = [c for c in stage_cols if c not in data.fovs.columns]
        if absent:
            raise ValueError(f"stage columns missing from the FOV table: {absent}")
    trace = [c for c in (*_TRACE, *(stage_cols or ())) if c in data.fovs.columns]
    rows = []
    for group, group_data in iter_analysis_groups(data):
        found = []
        if novelty is not None:
            found += _novel(group_data, novelty)
        if summaries:
            found += _influence(group_data, summaries, float(influence_threshold))
        if count_ratio is not None:
            found += _counts(group_data, float(count_ratio))
        if stage_cols is not None:
            found += _neighbours(
                group_data,
                *stage_cols,
                n_neighbours,
                float(neighbour_threshold),
                stage_unit,
            )
        if not found:
            continue
        positions = fov_positions(group_data, group_data.fovs)
        counts = np.bincount(positions, minlength=len(group_data.fovs))
        median = float(np.median(counts))
        key_cols = [c for c in group_data.fov_key if c not in group]
        for item in found:
            fov = group_data.fovs.iloc[item["position"]]
            rows.append(
                {
                    **group,
                    **{c: fov[c] for c in key_cols},
                    **{c: fov[c] for c in trace if c not in key_cols},
                    "reason": item["reason"],
                    "detail": item["detail"],
                    "value": item["value"],
                    "observations": int(counts[item["position"]]),
                    "median_observations": median,
                }
            )
    settings = {
        "summaries": [{"feature": s.feature, "statistic": s.statistic} for s in summaries],
        "novelty": None if novelty is None else dict(novelty.parameters),
        "influence_threshold": float(influence_threshold),
        "count_ratio": None if count_ratio is None else float(count_ratio),
        "stage_cols": None if stage_cols is None else list(stage_cols),
        "n_neighbours": n_neighbours,
        "neighbour_threshold": float(neighbour_threshold),
        "fov_key": list(data.fov_key),
        "trace": trace,
    }
    return InspectionResult(pd.DataFrame(rows), tuple(data.group_cols), settings)


def inspection_summary(result: InspectionResult) -> str:
    """ "Fields to reopen" in sentences, per group."""
    keys = list(result.group_cols)
    if result.fovs.empty:
        return "No field stands out under the declared thresholds."
    fov_key = [c for c in result.settings.get("fov_key", []) if c not in keys]
    trace = [c for c in result.settings.get("trace", []) if c not in fov_key]
    blocks = []
    for key, rows in result.fovs.groupby(keys, sort=False) if keys else [((), result.fovs)]:
        key = key if isinstance(key, tuple) else (key,)
        where = ", ".join(f"{k}={v}" for k, v in zip(keys, key, strict=True)) or "all FOVs"
        lines = [f"{where}: fields to reopen before trusting the result"]
        for _, own in rows.groupby(fov_key, sort=False):
            first = own.iloc[0]
            name = " / ".join(str(first[c]) for c in fov_key)
            extra = ", ".join(f"{c} {first[c]}" for c in trace)
            reasons = "; ".join(own["detail"])
            lines.append(f"  {name}{f' ({extra})' if extra else ''}: {reasons}.")
        lines.append(
            "  None of these is called an error: look at the images to see whether it is "
            "biology, technique or simply a large field."
        )
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


__all__ = ["REASONS", "InspectionResult", "Summary", "inspect_fovs", "inspection_summary"]
