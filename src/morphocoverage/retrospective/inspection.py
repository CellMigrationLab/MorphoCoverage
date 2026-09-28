"""Which fields should I go back and look at? A traceable list with one reason per line.

Question R5 of ``docs/questions.md``. ``inspect_fovs`` lists, per analysis
group, the FOVs worth reopening, each with a plain reason and the numbers
behind it, the image file and stage position when the FOV table has them, and
its observation count beside the group median. Each reason is computed
independently and reported only above its declared threshold; there is no
combined score, and no FOV is called bad, an outlier or an artifact. An
influential FOV may be biologically crucial, a technical artifact, or simply
large, and only the image can tell.

Reasons:

- **new state** (with ``novelty``): the FOV holds a cluster of observations
  the other FOVs do not show (support novelty, leave one FOV out); the count of
  those observations is given.
- **influence** (with ``summaries``): removing the FOV changes a declared
  summary (the mean or a quantile of a named feature) by at least
  ``influence_threshold`` in relative terms. Summaries weight every FOV
  equally, so duplicating objects within a FOV does not raise its influence
  beyond its effect as one FOV; a dense FOV is reported by the count reason
  instead.
- **count**: the observation count is at least ``count_ratio`` times the group
  median, at most the median divided by it, or zero.
- **neighbours** (with stage coordinates): the FOV's mean of some feature
  differs from the mean of its ``n_neighbours`` nearest FOVs on the stage by at
  least ``neighbour_threshold`` standard deviations of the FOV means; an
  empty neighbour has no mean, so the reason says how many of the
  ``n_neighbours`` it was compared with. Fields tied in distance at the
  ``n_neighbours``-th place (every regular grid) are taken in the order of
  the FOV key, never of the table's rows, and the reason says so.

Missing feature values: the influence and neighbour reasons use the
non-missing values of each feature, every FOV still weighing one; the novelty
reason follows the estimator, which cannot judge a group with missing values,
and the summary says so.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from .._checks import is_integer, is_number
from .._style import group_label
from .._wording import count, ordinal, percent
from ..data_check import validate_input_tables
from ..evaluator import iter_analysis_groups
from ..objectives import ObjectiveConfig, SamplingObjective
from ..schema import ValidatedInput, fov_positions
from ..strategies.support_novelty import FOV_QUANTITY, SupportNoveltyEstimator
from ..sufficiency_text import novelty_definition
from ._text import unit_plural, where
from .inspection_text import inspection_summary

REASONS = ("new state", "influence", "count", "neighbours")
_TIED = "; fields tied in distance taken by field name"
_TRACE = ("image_file", "position_id", "mask_file")
_POPULATION = ObjectiveConfig(SamplingObjective.POPULATION)


@dataclass(frozen=True)
class Summary:
    """A declared summary of one feature: ``"mean"`` or a quantile in (0, 1)."""

    feature: str
    statistic: str | float

    def __post_init__(self) -> None:
        if not isinstance(self.feature, str):
            raise TypeError("feature must be a column name")
        if self.statistic != "mean":
            if not is_number(self.statistic):
                raise ValueError("statistic must be 'mean' or a quantile between 0 and 1")
            if not 0 < float(self.statistic) < 1:
                raise ValueError("statistic must be 'mean' or a quantile between 0 and 1")

    @property
    def label(self) -> str:
        """``mean of area`` or ``95th percentile of area``."""
        if self.statistic == "mean":
            return f"mean of {self.feature}"
        share = 100 * float(self.statistic)
        # A whole percentile reads 1st, 22nd; a fractional one 2.5th.
        rank = ordinal(int(share)) if share.is_integer() else f"{share:g}th"
        return f"{rank} percentile of {self.feature}"


@dataclass(frozen=True)
class InspectionResult:
    """One row per FOV and reason (``fovs``), and the settings used.

    ``settings["n_fovs"]`` counts every inspected field and
    ``settings["n_fovs_judged"]`` those of the groups where new states were
    looked for (``None`` without ``novelty``); ``settings["novelty_definition"]``
    is the definition of new in words.
    """

    fovs: pd.DataFrame
    group_cols: tuple[str, ...]
    settings: dict[str, Any] = field(default_factory=dict)

    def summary(self, detail: str = "short") -> str:
        """The fields to reopen: a table (``detail="short"``) or every reason per group
        (``"full"``); see ``inspection_text``."""
        return inspection_summary(self, detail)


def _weighted(values: np.ndarray, weights: np.ndarray, statistic: str) -> float | None:
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


def _influence(
    data: ValidatedInput,
    summaries: Sequence[Summary],
    threshold: float,
    group: str,
    skipped: list[dict],
) -> tuple[list[dict], bool]:
    """Relative change of each FOV-weighted summary when one FOV is left out.

    A relative change needs a positive scale: for a feature with negative
    values (standardized, PCA or embedding coordinates, centred near zero) it
    explodes, and for a summary that is zero (a percentile of a count most
    objects do not have) it is undefined, so such summaries are skipped and
    reported in ``skipped`` with the group they were skipped in. Also returns
    whether any summary was checked in this group.
    """
    positions = fov_positions(data, data.fovs)
    rows, checked = [], False
    for summary in summaries:
        values = data.feature_values(summary.feature)
        why = None
        if np.nanmin(values, initial=0.0) < 0:
            why = "has negative values, so a relative change is undefined"
        else:
            # Each FOV weighs 1 over its non-missing values, so missing values
            # never make a FOV count less.
            present = ~np.isnan(values)
            counts = np.bincount(positions[present], minlength=len(data.fovs)).astype(float)
            weights = np.where(present, 1.0 / np.maximum(counts[positions], 1.0), 0.0)
            full = _weighted(values, weights, summary.statistic)
            if full is None:
                why = "has no value here"
            elif full == 0:
                why = "is 0 here, so a relative change is undefined"
        if why is not None:
            skipped.append({"group": group, "summary": summary.label, "reason": why})
            continue
        checked = True
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
                        # Every FOV weighs one in the summary, however many objects it holds.
                        "detail": (
                            f"removing it {direction} the field-weighted {summary.label} by "
                            f"{percent(abs(change))}"
                        ),
                    }
                )
    return rows, checked


def _counts(data: ValidatedInput, ratio: float) -> list[dict]:
    positions = fov_positions(data, data.fovs)
    counts = np.bincount(positions, minlength=len(data.fovs))
    median = float(np.median(counts))
    unit, words = data.observation_unit, unit_plural(data.observation_unit)
    rows = []
    for fov, held in enumerate(counts):
        if held == 0 and median > 0:
            detail = f"no {words}; the median is {median:g}"
        elif median > 0 and (held >= ratio * median or held <= median / ratio):
            detail = f"{count(int(held), unit, words)}; the median is {median:g}"
        elif median == 0 and held > 0:
            # Most fields are empty: the fields with observations are the unusual ones.
            detail = f"{count(int(held), unit, words)}; most fields have none (the median is 0)"
        else:
            continue
        rows.append({"position": fov, "reason": "count", "value": float(held), "detail": detail})
    return rows


def _neighbours_skipped(fovs: pd.DataFrame, x_col: str, y_col: str, k: int) -> str | None:
    """Why stage neighbours cannot be compared in this group, or None."""
    if len(fovs) <= k:
        return f"{len(fovs)} fields, not more than the {k} neighbours compared"
    if fovs[[x_col, y_col]].isna().any().any():
        return "stage coordinates are missing"
    return None


def _neighbour_words(valued: np.ndarray, k: int) -> str:
    """``its 4 nearest fields``, or ``3 of its 4 nearest fields (1 without a value)``."""
    used = int(valued.sum())
    if used == k:
        return f"its {k} nearest fields"
    return f"{used} of its {k} nearest fields ({count(k - used, 'field')} without a value)"


def _nearest(
    fovs: pd.DataFrame, xy: np.ndarray, fov_key: Sequence[str], k: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Each field's ``k`` nearest other fields, ties in distance broken by field identity.

    On a regular grid several fields sit at the same distance; taking them in
    row order made which were compared depend on the order of the FOV table.
    Distances equal to 1e-9 of the largest are ties, broken by the canonical
    order of the FOV key. Returns the distances and positions of the
    neighbours, and whether a tie at the k-th distance was broken.
    """
    n = len(fovs)
    keys = fovs[list(fov_key)].astype(str).reset_index(drop=True)
    rank = np.empty(n, dtype=np.int64)
    rank[keys.sort_values(list(keys.columns), kind="stable").index.to_numpy()] = np.arange(n)
    distance = cdist(xy, xy)
    scale = float(distance.max()) or 1.0
    rounded = np.round(distance / scale, 9)
    near = np.empty((n, k), dtype=np.int64)
    tied = np.zeros(n, dtype=bool)
    for fov in range(n):
        # Never the FOV itself, even when another field shares its stage position.
        others = np.delete(np.arange(n), fov)
        order = others[np.lexsort((rank[others], rounded[fov, others]))]
        near[fov] = order[:k]
        tied[fov] = len(order) > k and rounded[fov, order[k]] == rounded[fov, order[k - 1]]
    return np.take_along_axis(distance, near, axis=1), near, tied


def _neighbours(
    data: ValidatedInput, x_col: str, y_col: str, k: int, threshold: float, unit: str
) -> list[dict]:
    fovs = data.fovs
    xy = fovs[[x_col, y_col]].to_numpy(dtype=float)
    if _neighbours_skipped(fovs, x_col, y_col, k) is not None:
        return []
    positions = fov_positions(data, fovs)
    frame = pd.DataFrame(data.feature_matrix(), columns=list(data.feature_cols))
    means = frame.groupby(positions).mean().reindex(range(len(fovs)))
    distances, near, tied = _nearest(fovs, xy, data.fov_key, k)
    rows = []
    for feature in data.feature_cols:
        column = means[feature].to_numpy(dtype=float)
        for fov in range(len(fovs)):
            others = column[near[fov]]
            if np.isnan(column[fov]) or np.isnan(others).all():
                continue
            # An empty neighbour has no mean: the comparison is with the others only.
            valued = ~np.isnan(others)
            reach = distances[fov][valued].max()
            # Spread of the other FOVs' means: including the FOV itself would cap
            # its own z at sqrt(n) and hide a lone deviant field in a small group.
            spread = np.nanstd(np.delete(column, fov), ddof=1)
            if not np.isfinite(spread) or spread == 0:
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
                            f"{'above' if z > 0 else 'below'} {_neighbour_words(valued, k)} "
                            f"(within {reach:.0f} {unit}{_TIED if tied[fov] else ''})"
                        ),
                    }
                )
    return rows


def _novel(
    data: ValidatedInput, novelty: SupportNoveltyEstimator, group: str, skipped: list[dict]
) -> list[dict]:
    """FOVs holding a new state; when novelty is not estimable, the reason goes in ``skipped``."""
    flags = novelty.novel_observation_flags(data, new_states_only=True)
    if flags is None:
        quantity = novelty.estimate(data, objective=_POPULATION).quantities[FOV_QUANTITY]
        skipped.append({"group": group, "reason": quantity.reason or "novelty is not estimable"})
        return []
    if not flags.any():
        return []
    positions = fov_positions(data, data.fovs)
    counts = np.bincount(positions[flags], minlength=len(data.fovs))
    unit, words = data.observation_unit, unit_plural(data.observation_unit)
    return [
        {
            "position": fov,
            "reason": "new state",
            "value": float(held),
            "detail": f"{count(int(held), unit, words)} of a state no other field shows",
        }
        for fov, held in enumerate(counts)
        if held
    ]


def _check(
    summaries: Sequence[Summary | tuple],
    novelty: SupportNoveltyEstimator | None,
    influence_threshold: float,
    count_ratio: float | None,
    stage: Sequence[str] | None,
    neighbour_threshold: float | None,
    k: int,
) -> list[Summary]:
    summaries = [s if isinstance(s, Summary) else Summary(*s) for s in summaries]
    if novelty is not None and not isinstance(novelty, SupportNoveltyEstimator):
        raise TypeError("novelty must be a SupportNoveltyEstimator")
    if not is_number(influence_threshold) or not 0 < influence_threshold < np.inf:
        raise ValueError("influence_threshold must be a positive number")
    if count_ratio is not None and not (is_number(count_ratio) and 1 < count_ratio < np.inf):
        raise ValueError("count_ratio must be greater than 1, or None to skip the count reason")
    if not is_number(neighbour_threshold) or not 0 < neighbour_threshold < np.inf:
        raise ValueError("neighbour_threshold must be a positive number")
    if not is_integer(k) or k < 1:
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
    novelty_cols: Sequence[str] | None = None,
) -> InspectionResult:
    """The fields worth reopening, one row per FOV and reason, per analysis group.

    ``summaries`` are the summaries whose influence is checked, as
    ``Summary(feature, "mean")`` or ``Summary(feature, 0.95)`` (or plain
    tuples). ``novelty`` turns on the new-state reason; ``stage_cols`` the
    neighbour reason, on every feature in ``feature_cols``; ``count_ratio=None``
    turns the count reason off. Thresholds are provisional presets: a 10% relative
    change, a count 3 times the median (or a third of it), 3 SD from the 4
    nearest fields. ``novelty_cols`` are the columns new states are judged on
    (default ``feature_cols``), so novelty can read the declared
    representation while the other reasons read the readout.
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
    feature_cols = list(feature_cols)
    novelty_cols = feature_cols if novelty_cols is None else list(novelty_cols)
    data = validate_input_tables(
        objects,
        fovs=fovs,
        feature_cols=list(dict.fromkeys([*feature_cols, *novelty_cols])),
        fov_col=fov_col,
        group_cols=group_cols,
        observation_unit=observation_unit,
    )
    if stage_cols is not None:
        absent = [c for c in stage_cols if c not in data.fovs.columns]
        if absent:
            raise ValueError(f"stage columns missing from the FOV table: {absent}")
    trace = [c for c in (*_TRACE, *(stage_cols or ())) if c in data.fovs.columns]
    rows, skipped, neighbours_skipped, novelty_skipped = [], [], [], []
    checked: set[str] = set()
    n_fovs = n_judged = 0
    for group, both in iter_analysis_groups(data):
        n_fovs += len(both.fovs)
        group_data = replace(both, feature_cols=tuple(feature_cols))
        found = []
        if novelty is not None:
            judged = replace(both, feature_cols=tuple(novelty_cols))
            before = len(novelty_skipped)
            found += _novel(judged, novelty, where(group), novelty_skipped)
            if len(novelty_skipped) == before:
                checked.add("new state")
                n_judged += len(both.fovs)
        if summaries:
            influential, any_checked = _influence(
                group_data, summaries, float(influence_threshold), where(group), skipped
            )
            found += influential
            if any_checked:
                checked.add("influence")
        if count_ratio is not None:
            found += _counts(group_data, float(count_ratio))
            checked.add("count")
        if stage_cols is not None:
            why = _neighbours_skipped(group_data.fovs, *stage_cols, n_neighbours)
            if why is not None:
                neighbours_skipped.append({"group": where(group), "reason": why})
            else:
                checked.add("neighbours")
            found += _neighbours(
                group_data,
                *stage_cols,
                n_neighbours,
                float(neighbour_threshold),
                stage_unit,
            )
        for entry in (*skipped, *neighbours_skipped, *novelty_skipped):
            entry.setdefault("label", group_label(group))  # the group as the summaries name it
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
        "novelty_cols": None if novelty is None else novelty_cols,
        "novelty_definition": None
        if novelty is None
        else novelty_definition(novelty, novelty_cols),
        "influence_threshold": float(influence_threshold),
        "count_ratio": None if count_ratio is None else float(count_ratio),
        "stage_cols": None if stage_cols is None else list(stage_cols),
        "n_neighbours": n_neighbours,
        "neighbour_threshold": float(neighbour_threshold),
        "fov_key": list(data.fov_key),
        "trace": trace,
        "skipped_summaries": skipped,
        "skipped_neighbours": neighbours_skipped,
        "skipped_novelty": novelty_skipped,
        # The reasons judged in at least one group, in the order of REASONS.
        "checked": [reason for reason in REASONS if reason in checked],
        "n_fovs": n_fovs,
        # The fields of the groups where new states were looked for (None without novelty).
        "n_fovs_judged": None if novelty is None else n_judged,
    }
    return InspectionResult(pd.DataFrame(rows), tuple(data.group_cols), settings)


__all__ = ["REASONS", "InspectionResult", "Summary", "inspect_fovs", "inspection_summary"]
