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
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import pandas as pd

from .._checks import is_integer, is_number
from .._style import group_label
from .._wording import count, ordinal, percent
from ..data_check import validate_input_tables
from ..evaluator import iter_analysis_groups
from ..metadata import column_names
from ..objectives import ObjectiveConfig, SamplingObjective
from ..schema import ValidatedInput, fov_positions, observation_words
from ..strategies.novelty_null import not_informative
from ..strategies.support_novelty import FOV_QUANTITY, SupportNoveltyEstimator
from ..sufficiency_text import novelty_definition
from ..transforms import MAD_TO_SD
from ._stage_neighbours import neighbour_rows, stage_skipped
from ._text import where
from .inspection_text import inspection_summary

REASONS = ("new state", "influence", "count", "neighbours")
_TRACE = ("image_file", "position_id", "mask_file")
_POPULATION = ObjectiveConfig(SamplingObjective.POPULATION)
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
    more than ``spread`` robust SDs (``MAD_TO_SD`` times the median absolute deviation) from
    the group's median change (all ``True`` with ``spread=None``, the threshold alone).

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
    not judged (the callers say where).
    """
    changes = np.asarray(changes, dtype=float)
    if spread is None:
        return np.ones(len(changes), dtype=bool)
    if not len(changes):
        return np.zeros(0, dtype=bool)
    typical = float(np.median(changes))
    scale = MAD_TO_SD * float(np.median(np.abs(changes - typical)))
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
        left_out, changes = [], []
        for fov in np.unique(positions):
            keep = positions != fov
            without = _weighted(values[keep], weights[keep], summary.statistic)
            if without is not None:
                left_out.append(int(fov))
                changes.append((without - full) / abs(full))
        apart = stands_out(np.asarray(changes), spread)
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
    return rows, checked


def _counts(data: ValidatedInput, ratio: float) -> list[dict]:
    positions = fov_positions(data, data.fovs)
    counts = np.bincount(positions, minlength=len(data.fovs))
    median = float(np.median(counts))
    unit, words = observation_words(data)
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


def _novel(
    data: ValidatedInput,
    novelty: SupportNoveltyEstimator,
    group: str,
    skipped: list[dict],
    label_col: str | None,
    uninformative: list[dict],
    null: list[dict],
) -> list[dict]:
    """FOVs holding a new state, with how many of its observations carry ``label_col``; when
    novelty is not estimable, the reason goes in ``skipped``, and when the definition is not
    informative at this group's size (its own null rate above 20%, ``novelty_null``; owner,
    2026-10-06, audit 10) in ``uninformative``, no field listed. ``null`` gathers each
    judged group's fields and null rate (the count chance alone would give)."""
    sizes = np.bincount(fov_positions(data, data.fovs), minlength=len(data.fovs))
    reason, rate = not_informative(novelty, sizes, data.feature_cols)
    if reason is not None:
        uninformative.append({"group": group, "reason": reason, "rate": rate})
        return []
    flags = novelty.novel_observation_flags(data, new_states_only=True)
    if flags is None:
        quantity = novelty.estimate(data, objective=_POPULATION).quantities[FOV_QUANTITY]
        skipped.append({"group": group, "reason": quantity.reason or "novelty is not estimable"})
        return []
    null.append({"fields": int((sizes > 0).sum()), "rate": rate})
    if not flags.any():
        return []
    positions = fov_positions(data, data.fovs)
    counts = np.bincount(positions[flags], minlength=len(data.fovs))
    labelled = None
    if label_col is not None and label_col in data.objects.columns:
        marked = flags & data.objects[label_col].fillna(False).to_numpy(dtype=bool)
        labelled = np.bincount(positions[marked], minlength=len(data.fovs))
    unit, words = observation_words(data)

    def detail(fov: int, held: int) -> str:
        others = novelty.min_other_fovs  # more than 1 with one observation per field
        shown = "no other field shows" if others == 1 else f"fewer than {others} other fields show"
        text = f"{count(held, unit, words)} of a state {shown}"
        if labelled is not None and labelled[fov]:
            # The new state is the declared label: say so, it is what the user looks for.
            text += f" ({'all' if labelled[fov] == held else labelled[fov]} labelled {label_col})"
        return text

    return [
        {
            "position": fov,
            "reason": "new state",
            "value": float(held),
            "detail": detail(fov, int(held)),
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
    spread: float | None = INFLUENCE_SPREAD,
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
    if spread is not None and not (is_number(spread) and 0 < spread < np.inf):
        raise ValueError("influence_spread must be a positive number, or None")
    if not is_integer(k) or k < 1:
        raise ValueError("n_neighbours must be a positive integer")
    if stage is not None and (len(stage) != 2 or not all(isinstance(c, str) for c in stage)):
        raise TypeError("stage_cols must be two column names (x, y)")
    if stage is not None and stage[0] == stage[1]:
        raise ValueError("stage_cols must be two different columns (x, y), not one axis twice")
    return summaries


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
    rows, skipped, neighbours_skipped, novelty_skipped, small = [], [], [], [], []
    uninformative: list[dict] = []
    null: list[dict] = []
    checked: set[str] = set()
    n_fovs = n_judged = 0
    for group, both in iter_analysis_groups(data):
        n_fovs += len(both.fovs)
        group_data = replace(both, feature_cols=tuple(feature_cols))
        found = []
        if novelty is not None:
            judged = replace(both, feature_cols=tuple(novelty_cols))
            before = len(novelty_skipped) + len(uninformative)
            found += _novel(
                judged, novelty, where(group), novelty_skipped, label_col, uninformative, null
            )
            if len(novelty_skipped) + len(uninformative) == before:
                checked.add("new state")
                n_judged += len(both.fovs)
        if summaries:
            influential, any_checked = _influence(
                group_data,
                summaries,
                float(influence_threshold),
                where(group),
                skipped,
                small,
                None if influence_spread is None else float(influence_spread),
            )
            found += influential
            if any_checked:
                checked.add("influence")
        if count_ratio is not None:
            found += _counts(group_data, float(count_ratio))
            checked.add("count")
        if stage_cols is not None:
            why = stage_skipped(group_data.fovs, *stage_cols, n_neighbours)
            if why is not None:
                neighbours_skipped.append({"group": where(group), "reason": why})
            else:
                checked.add("neighbours")
            found += neighbour_rows(
                group_data,
                *stage_cols,
                n_neighbours,
                float(neighbour_threshold),
                stage_unit,
            )
        for entry in (*skipped, *neighbours_skipped, *novelty_skipped, *uninformative, *small):
            entry.setdefault("label", group_label(group))  # the group as the summaries name it
        rows += field_rows(group, group_data, found, trace)
    settings = {
        "summaries": [{"feature": s.feature, "statistic": s.statistic} for s in summaries],
        "novelty": None if novelty is None else dict(novelty.parameters),
        "novelty_cols": None if novelty is None else novelty_cols,
        "novelty_definition": None
        if novelty is None
        else novelty_definition(novelty, novelty_cols),
        "influence_threshold": float(influence_threshold),
        "influence_spread": None if influence_spread is None else float(influence_spread),
        "count_ratio": None if count_ratio is None else float(count_ratio),
        "stage_cols": None if stage_cols is None else list(stage_cols),
        "n_neighbours": n_neighbours,
        "neighbour_threshold": float(neighbour_threshold),
        "fov_key": list(data.fov_key),
        "trace": trace,
        "skipped_summaries": skipped,
        "skipped_neighbours": neighbours_skipped,
        "skipped_novelty": novelty_skipped,
        # Groups where the definition of new is not informative at their size, and each
        # judged group's fields and null rate (owner, 2026-10-06, audit 10).
        "uninformative_novelty": uninformative,
        "novelty_null": null,
        # The groups too small to judge a field's influence in (owner, 2026-10-05).
        "skipped_small": small,
        # The reasons judged in at least one group, in the order of REASONS.
        "checked": [reason for reason in REASONS if reason in checked],
        "n_fovs": n_fovs,
        # The fields of the groups where new states were looked for (None without novelty).
        "n_fovs_judged": None if novelty is None else n_judged,
    }
    return InspectionResult(pd.DataFrame(rows), tuple(data.group_cols), settings)


__all__ = [
    "REASONS",
    "SKIPPED_KEYS",
    "InspectionResult",
    "Summary",
    "inspect_fovs",
    "inspection_summary",
]
