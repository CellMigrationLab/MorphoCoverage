"""Which fields should I go back and look at? A traceable list with one reason per line.

Question R5 of ``docs/report/questions.md``. ``inspect_fovs`` lists, per analysis
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
  ``influence_threshold`` in relative terms, and that change stands out from
  the other fields' of its group (``stands_out``; a group of fewer than
  ``INFLUENCE_MIN_FIELDS`` fields is not judged, owner 2026-10-05). Summaries
  weight every FOV equally, so duplicating objects within a FOV does not raise
  its influence beyond its effect as one FOV; a dense FOV is reported by the
  count reason instead.
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
from dataclasses import dataclass, field
from typing import Any, NamedTuple

import numpy as np
import pandas as pd

from .._acquisition import ACQUISITION, acquisition_identity
from .._checks import is_integer, is_number
from .._frozen import owned
from .._rare_label import check_rare_labels
from .._stage_coordinates import stage_names
from .._wording import ordinal, percent
from ..data_check import validate_input_tables
from ..evaluator import iter_analysis_groups
from ..metadata import column_names
from ..schema import ValidatedInput, fov_positions
from ..statistics import robust_sd
from ..strategies.support_novelty import SupportNoveltyEstimator
from ..sufficiency_text import novelty_definition
from ._field_reasons import _counts, _Gathered, _neighbour_found, _new_state_found
from ._text import where
from .inspection_text import inspection_summary

REASONS = ("new state", "influence", "count", "neighbours")
_TRACE = ("image_file", "position_id", "mask_file")
# The influence scales with the group (owner, 2026-10-05; see stands_out): a field's change
# must lie this many robust SDs from its group's median change, in a group of at least
# INFLUENCE_MIN_FIELDS fields.
INFLUENCE_SPREAD = 3.0
INFLUENCE_MIN_FIELDS = 4
SMALL_GROUP = f"fewer than {INFLUENCE_MIN_FIELDS} fields"
# The settings naming the groups where a reason could not be judged.
SKIPPED_KEYS = ("skipped_summaries", "skipped_neighbours", "skipped_novelty")


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


@owned("fovs", "settings")
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


def stands_out(changes: np.ndarray, spread: float | None = INFLUENCE_SPREAD) -> np.ndarray:
    """Whether each field's leave-one-out change stands out from the others' of its group:
    more than ``spread`` robust SDs (``MAD_TO_SD`` times the median absolute deviation, their
    SD where that is 0: ``statistics.robust_sd``'s detector rule) from the group's median
    change (all ``True`` with ``spread=None``, the threshold alone).

    Why not a multiple of the median absolute change: leaving a field out moves a mean by the
    field's difference from it divided by the others' number, so a lone aberrant field moves
    it about ``n - 1`` times as much as a typical field, and inflates every other field's
    change too. "More than 3 times the median change" could then never flag it in a group of
    4, and the ratio grows with the group. The distance from the median change in robust SDs
    is a robust z-score of the field among its group's (the Hampel identifier, at the 3 SD of
    the neighbour reason) whatever the group's size: in simulations of 4-5 log-normal fields
    with a CV of 30-50%, 1-7% of ordinary fields were flagged (24-48% by the 10% threshold
    alone), and a field 4 times the others 66-97% of the time. With fewer than
    ``INFLUENCE_MIN_FIELDS`` fields one of the two others sets the spread, so such groups are
    not judged (the callers say where). With most changes tied, as for a discrete summary,
    the MAD is 0 and would flag every field off the median: 11% to 23% of null fields in
    simulations (audit 12), so the SD scales them there.
    """
    changes = np.asarray(changes, dtype=float)
    if spread is None:
        return np.ones(len(changes), dtype=bool)
    if not len(changes):
        return np.zeros(0, dtype=bool)
    typical = float(np.median(changes))
    scale = robust_sd(changes, "detector")
    return np.abs(changes - typical) > spread * scale


def _influence(
    data: ValidatedInput,
    summaries: Sequence[Summary],
    threshold: float,
    group: str,
    skipped: list[dict],
    small: list[dict],
    spread: float | None = INFLUENCE_SPREAD,
) -> tuple[list[dict], bool]:
    """Relative change of each FOV-weighted summary when one FOV is left out, listed when it
    is at least ``threshold`` and stands out from the group's (``stands_out``).

    A relative change needs a positive scale: for a feature with negative
    values (standardized, PCA or embedding coordinates, centred near zero) it
    explodes, and for a summary that is zero (a percentile of a count most
    objects do not have) it is undefined, so such summaries are skipped and
    reported in ``skipped`` with the group they were skipped in. A group of fewer than
    ``INFLUENCE_MIN_FIELDS`` fields with a value is not judged and goes in ``small`` (unless
    ``spread`` is None). Also returns whether any summary was checked in this group.
    """
    positions = fov_positions(data, data.fovs)
    rows, checked = [], False
    present = np.zeros(len(positions), dtype=bool)
    for summary in summaries:
        present |= ~np.isnan(data.feature_values(summary.feature))
    fields = len(np.unique(positions[present]))
    if spread is not None and summaries and fields < INFLUENCE_MIN_FIELDS:
        small.append({"group": group, "reason": SMALL_GROUP})
        return rows, checked
    for summary in summaries:
        values = data.feature_values(summary.feature)
        why, weights, full = _relative_base(values, positions, len(data.fovs), summary.statistic)
        if why is not None:
            skipped.append({"group": group, "summary": summary.label, "reason": why})
            continue
        checked = True
        rows += _influential(summary, values, positions, weights, full, threshold, spread)
    return rows, checked


def _relative_base(
    values: np.ndarray, positions: np.ndarray, n_fovs: int, statistic: str | float
) -> tuple[str | None, np.ndarray | None, float | None]:
    """Why a relative change of the summary is undefined here (None when it is defined), the
    weights of the observations and the summary of every field."""
    if np.nanmin(values, initial=0.0) < 0:
        return "has negative values, so a relative change is undefined", None, None
    # Each FOV weighs 1 over its non-missing values, so missing values
    # never make a FOV count less.
    present = ~np.isnan(values)
    counts = np.bincount(positions[present], minlength=n_fovs).astype(float)
    weights = np.where(present, 1.0 / np.maximum(counts[positions], 1.0), 0.0)
    full = _weighted(values, weights, statistic)
    if full is None:
        return "has no value here", weights, full
    if full == 0:
        return "is 0 here, so a relative change is undefined", weights, full
    return None, weights, full


def _influential(
    summary: Summary,
    values: np.ndarray,
    positions: np.ndarray,
    weights: np.ndarray,
    full: float,
    threshold: float,
    spread: float | None,
) -> list[dict]:
    """The fields whose leaving out changes ``summary`` by at least ``threshold`` and whose
    change stands out from the group's."""
    left_out, changes = [], []
    for fov in np.unique(positions):
        keep = positions != fov
        without = _weighted(values[keep], weights[keep], summary.statistic)
        if without is not None:
            left_out.append(int(fov))
            changes.append((without - full) / abs(full))
    apart = stands_out(np.asarray(changes), spread)
    rows = []
    for fov, change, alone in zip(left_out, changes, apart, strict=True):
        if abs(change) < threshold or not alone:
            continue
        direction = "lowers" if change < 0 else "raises"
        rows.append(
            {
                "position": fov,
                "reason": "influence",
                "value": float(change),
                # Every FOV weighs one in the summary, however many objects it holds.
                "detail": (
                    f"removing it {direction} the field-weighted {summary.label} by "
                    f"{percent(abs(change))}"
                ),
            }
        )
    return rows


def _check(
    summaries: Sequence[Summary | tuple],
    novelty: SupportNoveltyEstimator | None,
    influence_threshold: float,
    count_ratio: float | None,
    stage: Sequence[str] | None,
    neighbour_threshold: float | None,
    k: int,
    spread: float | None = INFLUENCE_SPREAD,
) -> list[Summary]:
    summaries = [s if isinstance(s, Summary) else Summary(*s) for s in summaries]
    if novelty is not None and not isinstance(novelty, SupportNoveltyEstimator):
        raise TypeError("novelty must be a SupportNoveltyEstimator")
    _check_numbers(influence_threshold, count_ratio, neighbour_threshold, spread, k)
    if stage is not None:
        stage_names(stage)
    return summaries


def _check_numbers(
    influence_threshold: float,
    count_ratio: float | None,
    neighbour_threshold: float | None,
    spread: float | None,
    k: int,
) -> None:
    """Raise when a threshold, the spread or the number of neighbours is not usable."""
    if not is_number(influence_threshold) or not 0 < influence_threshold < np.inf:
        raise ValueError("influence_threshold must be a positive number")
    if count_ratio is not None and not (is_number(count_ratio) and 1 < count_ratio < np.inf):
        raise ValueError("count_ratio must be greater than 1, or None to skip the count reason")
    if not is_number(neighbour_threshold) or not 0 < neighbour_threshold < np.inf:
        raise ValueError("neighbour_threshold must be a positive number")
    if spread is not None and not (is_number(spread) and 0 < spread < np.inf):
        raise ValueError("influence_spread must be a positive number, or None")
    if not is_integer(k) or k < 1:
        raise ValueError("n_neighbours must be a positive integer")


def field_rows(
    group: dict[str, Any], group_data: ValidatedInput, found: list[dict], trace: Sequence[str]
) -> list[dict[str, Any]]:
    """The inspection's rows for ``found`` (each a reason with the field's ``position`` in
    ``group_data.fovs``): the group, the field's key and trace columns, the reason and its
    numbers, and the field's observation count beside the group's median."""
    if not found:
        return []
    positions = fov_positions(group_data, group_data.fovs)
    counts = np.bincount(positions[positions >= 0], minlength=len(group_data.fovs))
    median = float(np.median(counts))
    key_cols = [c for c in group_data.fov_key if c not in group]
    rows = []
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
    return rows


class _Reasons(NamedTuple):
    """The reasons ``inspect_fovs`` was asked for, with their settings as given (summaries
    and columns checked)."""

    feature_cols: list[str]
    summaries: list[Summary]
    novelty: SupportNoveltyEstimator | None
    novelty_cols: list[str]
    label_col: str | None
    influence_threshold: float
    influence_spread: float | None
    count_ratio: float | None
    stage_cols: tuple[str, str] | None
    stage_unit: str
    n_neighbours: int
    neighbour_threshold: float


def _check_columns(
    data: ValidatedInput, label_col: str | None, stage_cols: tuple[str, str] | None
) -> None:
    """Raise when the label column or a stage column is missing or not usable."""
    if label_col is not None:  # one contract for every label column (issue 344)
        if label_col not in data.objects.columns:
            raise ValueError(f"label_col {label_col!r} is not a column of the observations")
        check_rare_labels(data.objects[label_col], f"label_col {label_col!r}")
    if stage_cols is not None:
        absent = [c for c in stage_cols if c not in data.fovs.columns]
        if absent:
            raise ValueError(f"stage columns missing from the FOV table: {absent}")


def inspect_fovs(
    objects: pd.DataFrame,
    fovs: pd.DataFrame,
    *,
    feature_cols: Sequence[str],
    summaries: Sequence[Summary | tuple] = (),
    novelty: SupportNoveltyEstimator | None = None,
    influence_threshold: float = 0.10,
    influence_spread: float | None = INFLUENCE_SPREAD,
    count_ratio: float | None = 3.0,
    stage_cols: tuple[str, str] | None = None,
    stage_unit: str = "µm",
    n_neighbours: int = 4,
    neighbour_threshold: float = 3.0,
    fov_col: str | None = None,
    group_cols: Sequence[str] | None = None,
    observation_unit: str | None = None,
    novelty_cols: Sequence[str] | None = None,
    label_col: str | None = None,
) -> InspectionResult:
    """The fields worth reopening, one row per FOV and reason, per analysis group.

    ``summaries`` are the summaries whose influence is checked, as
    ``Summary(feature, "mean")`` or ``Summary(feature, 0.95)`` (or plain
    tuples). ``novelty`` turns on the new-state reason; ``stage_cols`` the
    neighbour reason, on every feature in ``feature_cols``; ``count_ratio=None``
    turns the count reason off. Thresholds are provisional presets: a 10% relative
    change that is also more than ``influence_spread`` robust SDs from the group's median
    change (``stands_out``; ``None``: the 10% alone, every group judged), a count 3 times
    the median (or a third of it), 3 SD from the 4 nearest fields. ``novelty_cols`` are
    the columns new states are judged on (default ``feature_cols``), so novelty can read
    the declared representation while the other reasons read the readout. ``label_col`` (a column of
    booleans or 0/1, such as the rare phenotype's label) says how many cells of each new
    state carry the label.
    """
    summaries = _check(
        summaries,
        novelty,
        influence_threshold,
        count_ratio,
        stage_cols,
        neighbour_threshold,
        n_neighbours,
        influence_spread,
    )
    feature_cols = list(column_names(feature_cols, "feature_cols"))  # read once
    missing = [s.feature for s in summaries if s.feature not in feature_cols]
    if missing:
        raise ValueError(f"summary features must be among feature_cols: {missing}")
    novelty_cols = (
        feature_cols if novelty_cols is None else list(column_names(novelty_cols, "novelty_cols"))
    )
    data = validate_input_tables(
        objects,
        fovs=fovs,
        feature_cols=list(dict.fromkeys([*feature_cols, *novelty_cols])),
        fov_col=fov_col,
        group_cols=group_cols,
        observation_unit=observation_unit,
    )
    _check_columns(data, label_col, stage_cols)
    trace = [c for c in (*_TRACE, *(stage_cols or ())) if c in data.fovs.columns]
    reasons = _Reasons(
        feature_cols=feature_cols,
        summaries=summaries,
        novelty=novelty,
        novelty_cols=novelty_cols,
        label_col=label_col,
        influence_threshold=influence_threshold,
        influence_spread=influence_spread,
        count_ratio=count_ratio,
        stage_cols=stage_cols,
        stage_unit=stage_unit,
        n_neighbours=n_neighbours,
        neighbour_threshold=neighbour_threshold,
    )
    gathered = _Gathered()
    rows = []
    for group, both in iter_analysis_groups(data):
        gathered.n_fovs += len(both.fovs)
        group_data = both.with_tables(feature_cols=tuple(feature_cols))
        found = _group_found(reasons, gathered, group, both, group_data)
        gathered.label(group)
        rows += field_rows(group, group_data, found, trace)
    settings = _inspection_settings(reasons, data, trace, gathered)
    return InspectionResult(pd.DataFrame(rows), tuple(data.group_cols), settings)


def _group_found(
    reasons: _Reasons,
    gathered: _Gathered,
    group: dict[str, Any],
    both: ValidatedInput,
    group_data: ValidatedInput,
) -> list[dict]:
    """Every reason asked for, in one group: ``both`` holds every column inspected,
    ``group_data`` the ``feature_cols`` alone."""
    found = []
    if reasons.novelty is not None:
        found += _new_state_found(
            both, reasons.novelty, reasons.novelty_cols, reasons.label_col, where(group), gathered
        )
    if reasons.summaries:
        spread = reasons.influence_spread
        influential, any_checked = _influence(
            group_data,
            reasons.summaries,
            float(reasons.influence_threshold),
            where(group),
            gathered.skipped,
            gathered.small,
            None if spread is None else float(spread),
        )
        found += influential
        if any_checked:
            gathered.checked.add("influence")
    if reasons.count_ratio is not None:
        found += _counts(group_data, float(reasons.count_ratio))
        gathered.checked.add("count")
    if reasons.stage_cols is not None:
        found += _neighbour_found(
            group_data,
            reasons.stage_cols,
            reasons.n_neighbours,
            reasons.neighbour_threshold,
            reasons.stage_unit,
            where(group),
            gathered,
        )
    return found


def _inspection_settings(
    reasons: _Reasons, data: ValidatedInput, trace: list[str], gathered: _Gathered
) -> dict[str, Any]:
    """The inspection's settings: the reasons asked for, and what the groups gathered."""
    novelty, spread, ratio = reasons.novelty, reasons.influence_spread, reasons.count_ratio
    return {
        ACQUISITION: acquisition_identity(data),
        "summaries": [{"feature": s.feature, "statistic": s.statistic} for s in reasons.summaries],
        "novelty": None if novelty is None else dict(novelty.parameters),
        "novelty_cols": None if novelty is None else reasons.novelty_cols,
        "novelty_definition": None
        if novelty is None
        else novelty_definition(novelty, reasons.novelty_cols),
        "influence_threshold": float(reasons.influence_threshold),
        "influence_spread": None if spread is None else float(spread),
        "count_ratio": None if ratio is None else float(ratio),
        "stage_cols": None if reasons.stage_cols is None else list(reasons.stage_cols),
        "n_neighbours": reasons.n_neighbours,
        "neighbour_threshold": float(reasons.neighbour_threshold),
        "fov_key": list(data.fov_key),
        "trace": trace,
        "skipped_summaries": gathered.skipped,
        "skipped_neighbours": gathered.neighbours_skipped,
        "skipped_novelty": gathered.novelty_skipped,
        # Groups where the definition of new is not informative at their size, and each
        # judged group's fields and null rate (owner, 2026-10-06, audit 10).
        "uninformative_novelty": gathered.uninformative,
        "novelty_null": gathered.null,
        # The groups too small to judge a field's influence in (owner, 2026-10-05).
        "skipped_small": gathered.small,
        # The reasons judged in at least one group, in the order of REASONS.
        "checked": [reason for reason in REASONS if reason in gathered.checked],
        "n_fovs": gathered.n_fovs,
        # The fields of the groups where new states were looked for (None without novelty).
        "n_fovs_judged": None if novelty is None else gathered.n_judged,
    }


__all__ = [
    "REASONS",
    "SKIPPED_KEYS",
    "InspectionResult",
    "Summary",
    "inspect_fovs",
    "inspection_summary",
]
