"""Sampling reliability: if I acquire ``k`` representative FOVs, how often is that enough?

The strategy-independent answer to "how many FOVs should I image next time?"
(P1 of ``docs/optimise/optimise.md``). No
stopping rule is involved; whether a rule stops at the right time is a
separate, sequential question.

Design, per analysis group (one sample):

1. For each outer split ``s``, hold out a fixed **reservoir** of whole FOVs (or
   spatial blocks) with ``ReservoirSplit``, seeded from ``random_state`` and
   ``s`` together (``_seeds.split_seed``), so another ``random_state`` holds
   back other fields in every split; the remaining FOVs are the stream.
2. For each ``k`` in ``fov_counts`` and each of ``draws`` draws, take ``k``
   stream FOVs at random, as a representative acquisition would.
3. Assess every ``OutcomeCriterion`` of those ``k`` FOVs against the same
   reservoir, for every ``k`` in the split. A reference that shrank as ``k``
   grew would make the curve partly a curve of reference quality.
4. Per split, the share of draws that preserved the criterion is this
   sample's conditional reliability at ``k``; the curve reports its mean and
   spread over splits, the Monte Carlo standard error of that mean with the
   hold-out as the unit (the spread over splits divided by the square root of
   their number when every split held back other fields; splits that repeat
   a hold-out are one cluster, so it stops shrinking once the distinct
   hold-outs are used up: more splits of the same references add no
   reference), the metric's distribution, and the information gain of ``k``
   over the previous listed ``k``.

A draw of FOVs without any observations, or with fewer than the metric needs
(one object has no covariance), is acquired effort and counts as not
preserved (``OutcomeCriterion.assess``). A draw is undefined only when the
metric cannot be computed for other reasons, most often because the reservoir
lacks what the criterion needs (a rare state absent from the held-back FOVs);
the summary says how many draws and hold-outs could be judged, since the
answer is then conditional on the hold-outs that could, and says so when
nothing could be judged at all.

Limits, stated with every result:

- ``k`` cannot exceed the stream: with 28 FOVs and a reservoir of 10, nothing
  is said beyond 18. A listed ``k`` beyond every split's stream gets a row that
  says so (``splits`` 0), so no sample disappears from later analyses; with
  unequal blocks, a ``k`` judged on only some splits says on how many.
- Draws and splits estimate one sample's behaviour; they are not independent
  replicates. Statements across samples count samples or repeats (P2).
- Random draws treat FOVs as exchangeable, as random-position sampling does;
  recorded, spatially ordered acquisition is a different question.
- With few blocks the splits repeat hold-outs (3 blocks give at most 3
  reservoirs of one block); the summary says how many of the splits were
  distinct hold-outs, since repeated ones differ only by the random draws,
  and then states shares and spreads over splits rather than hold-outs.
- A group with fewer than two FOVs (or blocks) cannot hold one back and keep a
  stream; it is skipped with the reason (``ReliabilityResult.skipped``), and
  the other groups are still computed. ``replay_acquisitions`` skips by the
  same rule, in the same words.
- Everything is conditional on the criteria: the metric, its tolerance and
  the features it reads.
- The curve is conservative by construction: the held-back reference is
  itself a finite sample, and its own error adds to that of the ``k``
  fields, so "enough in X%" understates how often ``k`` fields preserve the
  sample's true value (with a 30% reservoir, 62% against 74% at 8 fields in
  simulated wells), and a field count read from it errs high.

Shares are rounded to 12 decimals, so a share of exactly 90% is 0.9 and not
0.8999999999999999 from the averaging over splits.

Shared-private: ``_reservoir_split`` and ``_skip_reason`` (used by ``replay``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .._checks import as_count, check_seed, is_integer, is_share
from .._frozen import owned
from .._parallel import check_n_jobs, map_units
from .._seeds import group_key, group_rng, split_seed
from .._wording import count
from ..data_check import validate_input_tables
from ..evaluator import FovSubsets, iter_analysis_groups
from ..reservoir import ReservoirSplit
from ..schema import ValidatedInput
from .criteria import OutcomeCriterion, Preservation
from .reliability_text import reliability_summary

CURVE_COLUMNS = (
    "criterion",
    "fovs",
    "splits",
    "defined_splits",
    "preserved",
    "preserved_split_sd",
    "preserved_mc_se",
    "defined_draws",
    "undefined_draws",
    "metric_median",
    "metric_p10",
    "metric_p90",
    "gain",
    "reservoir_fovs",
    "stream_fovs",
)


@owned("curve", "draws", "settings", "ceiling", "skipped")
@dataclass(frozen=True)
class ReliabilityResult:
    """The reliability curve, every assessed draw, and the settings that produced them.

    ``curve`` has one row per analysis group, criterion and FOV count (columns
    ``CURVE_COLUMNS`` after the group columns). ``draws`` has one row per
    group, split, FOV count, draw and criterion, with the metric value and the
    preservation. ``ceiling`` has one row per group and criterion: the share of
    splits in which the whole stream preserved the criterion, and
    ``distinct_holdouts``, how many of those splits held back a different set
    of fields, with the longest and shortest whole stream (``stream_fovs``,
    ``shortest_stream_fovs``) and the smallest reservoir (``reservoir_fovs``).
    The reservoir is
    itself a finite sample, so this ceiling can be below 1 however well the
    stream was sampled, and the curve cannot be read above it. ``skipped``
    has one row per group that could not be split (fewer than two FOVs or
    blocks), with the ``reason``. ``summary()`` states the curve in words (short
    across groups; ``detail="full"`` per group).
    """

    curve: pd.DataFrame
    draws: pd.DataFrame
    group_cols: tuple[str, ...]
    settings: Mapping[str, Any] = field(default_factory=dict)
    ceiling: pd.DataFrame = field(default_factory=pd.DataFrame)
    skipped: pd.DataFrame = field(default_factory=pd.DataFrame)

    def summary(self, detail: str = "short") -> str:
        """The curve in words: ``"short"`` one table across groups, ``"full"`` per group.

        See ``reliability_text``; every caveat is stated once, at the end.
        """
        return reliability_summary(self, detail)


def _check_settings(
    criteria: Sequence[OutcomeCriterion],
    fov_counts: Sequence[int],
    reservoir_fraction: float,
    n_splits: int,
    draws: int,
    random_state: int,
) -> tuple[list[OutcomeCriterion], list[int]]:
    """Validate the settings; return the criteria list and sorted distinct counts."""
    criteria = list(criteria)
    if not criteria or not all(isinstance(c, OutcomeCriterion) for c in criteria):
        raise TypeError("criteria must be a non-empty sequence of OutcomeCriterion")
    if len({c.name for c in criteria}) != len(criteria):
        raise ValueError("criteria must have distinct names")
    counts = list(fov_counts) if not isinstance(fov_counts, str) else [fov_counts]
    if not counts or not all(is_integer(k) and k >= 1 for k in counts):
        raise ValueError("fov_counts must be positive integers")
    counts = sorted({int(k) for k in counts})  # numpy integers (np.arange) accepted
    if not is_share(reservoir_fraction):
        raise ValueError("reservoir_fraction must be between 0 and 1")
    as_count(n_splits, "n_splits")
    as_count(draws, "draws")
    check_seed(random_state)
    return criteria, counts


def _reservoir_split(
    fraction: float, random_state: int, split: int, block_col: str | None
) -> ReservoirSplit:
    """The ``ReservoirSplit`` of one outer split, as reliability and replay both hold it out.

    Seeded with ``_seeds.split_seed(random_state, split)``, so neighbouring
    ``random_state`` values share no reservoir, and the same ``random_state``
    and split hold back the same fields in ``sampling_reliability`` and in
    ``replay_acquisitions``.
    """
    return ReservoirSplit(fraction, split_seed(random_state, split), block_col)


def _group_draws(
    group: Mapping[str, Any],
    data: ValidatedInput,
    criteria: Sequence[OutcomeCriterion],
    counts: Sequence[int],
    split_settings: tuple[float, str | None],
    splits: Sequence[int],
    draws: int,
    random_state: int,
) -> tuple[list[dict], list[dict]]:
    """Every assessed draw for one analysis group and the given splits, and the whole streams."""
    rows, ceiling = [], []
    for split in splits:
        split_rows, split_ceiling = _split_draws(
            group, data, criteria, counts, split_settings, split, draws, random_state
        )
        rows += split_rows
        ceiling += split_ceiling
    return rows, ceiling


def _split_draws(
    group: Mapping[str, Any],
    data: ValidatedInput,
    criteria: Sequence[OutcomeCriterion],
    counts: Sequence[int],
    split_settings: tuple[float, str | None],
    split: int,
    draws: int,
    random_state: int,
) -> tuple[list[dict], list[dict]]:
    """Every assessed draw for one analysis group and split, and the whole stream."""
    fraction, block_col = split_settings
    rows, ceiling = [], []
    stream, reservoir = _reservoir_split(fraction, random_state, split, block_col).split(
        group, data
    )
    # Which fields were held back: splits with few blocks can repeat a hold-out.
    holdout = group_key(
        {"fovs": sorted(map(str, reservoir.fovs[list(data.fov_key)].itertuples(index=False)))}
    )
    # A fitted outcome (a wide table's components) is fitted on the stream, never on the
    # held-back fields that judge it (owner, 2026-10-06).
    criteria = [criterion.trained(stream) for criterion in criteria]
    for criterion in criteria:
        status, value = criterion.assess(reservoir, stream)
        ceiling.append(
            {
                **group,
                "split": split,
                "criterion": criterion.name,
                "value": value,
                "status": status.value,
                "stream_fovs": len(stream.fovs),
                "reservoir_fovs": len(reservoir.fovs),
                "holdout": holdout,
            }
        )
    # One generator per group and split: the same draws whatever other groups exist.
    rng = group_rng(random_state, group, split)
    n_stream = len(stream.fovs)
    subsets = FovSubsets(stream)
    for k in (k for k in counts if k <= n_stream):
        for draw in range(draws):
            subset = subsets.take(np.sort(rng.choice(n_stream, size=k, replace=False)))
            for criterion in criteria:
                status, value = criterion.assess(reservoir, subset)
                rows.append(
                    {
                        **group,
                        "split": split,
                        "fovs": k,
                        "draw": draw,
                        "criterion": criterion.name,
                        "value": value,
                        "status": status.value,
                        "reservoir_fovs": len(reservoir.fovs),
                        "stream_fovs": n_stream,
                        "holdout": holdout,
                    }
                )
    return rows, ceiling


def _skip_reason(data: ValidatedInput, fraction: float, block_col: str | None) -> str | None:
    """Why a group cannot be split into a reservoir and a stream, or ``None``.

    The one skip rule of ``sampling_reliability`` and ``replay_acquisitions``:
    a group needs two FOVs (or blocks) to hold one back and keep a stream. A
    ``block_col`` missing from the FOV table, or with missing values, raises
    here as it would in the split.
    """
    units = ReservoirSplit(fraction, 0, block_col).n_blocks(data)
    if units >= 2:
        return None
    what = count(units, "field") if block_col is None else f"{count(units, 'block')} of {block_col}"
    return f"only {what}: at least two are needed to hold one back and keep a stream"


# Shares over splits are ratios of draw counts; their mean and SD carry
# rounding noise (nine splits at exactly 90% average to 0.8999999999999999,
# and identical splits have an SD of 1e-16). Rounding to 12 decimals removes
# the noise, so a share of exactly 90% meets a threshold of 90% and reads "±0%".
_DIGITS = 12


def _clustered_se(per_split: pd.Series, holdouts: pd.Series) -> float | None:
    """The Monte Carlo standard error of the mean over splits, one cluster per hold-out.

    Splits that held back the same fields differ only by their draws, so they
    are one cluster (owner, 2026-10-05): the cluster-robust variance
    ``H / (H - 1) * sum(r_h ** 2) / n ** 2``, with ``r_h`` the summed
    deviations of hold-out ``h``'s splits from the mean, ``H`` the distinct
    hold-outs and ``n`` the splits. With every hold-out distinct it is the SD
    over splits divided by ``sqrt(n)``; ``None`` below two distinct hold-outs.
    """
    clusters = holdouts.reindex(per_split.index)
    n_clusters = int(clusters.nunique())
    if n_clusters < 2:
        return None
    sums = (per_split - per_split.mean()).groupby(clusters).sum()
    variance = n_clusters / (n_clusters - 1) * float((sums**2).sum()) / len(per_split) ** 2
    return round(float(np.sqrt(variance)), _DIGITS)


def _curve_row(runs: pd.DataFrame) -> dict[str, Any]:
    """Aggregate the draws of one group, criterion and FOV count."""
    defined = runs[runs["status"] != Preservation.UNDEFINED.value]
    per_split = (
        defined.assign(ok=defined["status"] == Preservation.PRESERVED.value)
        .groupby("split")["ok"]
        .mean()
    )
    values = defined["value"].to_numpy(dtype=float)
    spread = round(float(per_split.std(ddof=1)), _DIGITS) if len(per_split) > 1 else None
    # The Monte Carlo error of the printed mean over splits, clustered on the hold-out:
    # another random_state draws other hold-outs and draws, but no more references.
    holdouts = runs.groupby("split")["holdout"].first()
    mc_se = None if spread is None else _clustered_se(per_split, holdouts)
    return {
        "splits": int(runs["split"].nunique()),
        "defined_splits": len(per_split),
        "preserved": round(float(per_split.mean()), _DIGITS) if len(per_split) else None,
        "preserved_split_sd": spread,
        "preserved_mc_se": mc_se,
        "defined_draws": len(defined),
        "undefined_draws": len(runs) - len(defined),
        "metric_median": float(np.median(values)) if len(values) else None,
        "metric_p10": float(np.quantile(values, 0.1)) if len(values) else None,
        "metric_p90": float(np.quantile(values, 0.9)) if len(values) else None,
        "reservoir_fovs": int(runs["reservoir_fovs"].min()),
        "stream_fovs": int(runs["stream_fovs"].max()),
    }


def _add_gain(
    curve: pd.DataFrame, criteria: Sequence[OutcomeCriterion], keys: list[str]
) -> pd.DataFrame:
    """Improvement of the median metric from the previous listed ``k`` (None for the first)."""
    better_lower = {c.name: c.direction == "lower_is_better" for c in criteria}
    gains = []
    for _, block in curve.groupby([*keys, "criterion"], sort=False, dropna=False):
        block = block.sort_values("fovs")
        medians = block["metric_median"].astype(float).to_numpy()
        change = np.concatenate([[np.nan], np.diff(medians)])
        if better_lower[block["criterion"].iloc[0]]:
            change = -change
        gains.append(pd.Series(change, index=block.index))
    gain = pd.concat(gains) if gains else pd.Series(dtype=float)
    curve["gain"] = gain.reindex(curve.index).astype(object).where(gain.notna(), None)
    return curve


def sampling_reliability(
    objects: pd.DataFrame,
    fovs: pd.DataFrame,
    *,
    feature_cols: Sequence[str],
    criteria: Sequence[OutcomeCriterion],
    fov_counts: Sequence[int],
    reservoir_fraction: float,
    n_splits: int = 10,
    draws: int = 50,
    random_state: int = 0,
    block_col: str | None = None,
    fov_col: str | None = None,
    group_cols: Sequence[str] | None = None,
    observation_unit: str | None = None,
    n_jobs: int = 1,
) -> ReliabilityResult:
    """How often ``k`` random whole FOVs preserve each criterion in held-back FOVs.

    ``fovs`` lists every acquired FOV, including FOVs without observations,
    which can be drawn and count as acquired. ``reservoir_fraction`` is the
    share of each group's FOVs (or of its ``block_col`` blocks) held back per
    split, a design choice with no default: a larger reservoir is a better
    reference and leaves a shorter stream. ``n_splits`` and ``draws`` are
    computational settings; the curve reports how much the answer varies
    across splits. ``n_jobs`` runs the groups (their splits apart when
    groups are fewer than workers) in that many worker processes (``-1``:
    every CPU); results are identical for every value
    (bit for bit when BLAS is single-threaded; see ``morphocoverage._parallel``).
    Other arguments are those of ``validate_input_tables``.
    """
    check_n_jobs(n_jobs)
    criteria, counts = _check_settings(
        criteria, fov_counts, reservoir_fraction, n_splits, draws, random_state
    )
    # numpy integers are accepted and recorded as the Python values they hold.
    n_splits, draws, random_state = int(n_splits), int(draws), int(random_state)
    data = validate_input_tables(
        objects,
        fovs=fovs,
        feature_cols=feature_cols,
        fov_col=fov_col,
        group_cols=group_cols,
        observation_unit=observation_unit,
    )

    split_settings = (float(reservoir_fraction), block_col)
    # Every split of every group has its own reservoir and generator, so each
    # is an independent unit of work. Splits are sent apart only when the
    # groups are fewer than the workers; otherwise a group's splits stay
    # together, which sends its data once.
    groups, skipped = [], []
    for group, group_data in iter_analysis_groups(data):
        reason = _skip_reason(group_data, float(reservoir_fraction), block_col)
        if reason is None:
            groups.append((group, group_data))
        else:
            skipped.append({**group, "fovs": len(group_data.fovs), "reason": reason})
    apart = len(groups) < check_n_jobs(n_jobs)
    blocks = [[split] for split in range(n_splits)] if apart else [list(range(n_splits))]
    jobs = [
        (group, group_data, criteria, counts, split_settings, splits, draws, random_state)
        for group, group_data in groups
        for splits in blocks
    ]
    rows, ceiling_rows = [], []
    for group_rows, group_ceiling in map_units(_group_draws, jobs, n_jobs):
        rows += group_rows
        ceiling_rows += group_ceiling
    keys = list(data.group_cols)
    draw_table = pd.DataFrame(
        rows,
        columns=[*keys, "split", "fovs", "draw", "criterion", "value", "status"]
        + ["reservoir_fovs", "stream_fovs", "holdout"],
    )
    curve_rows = []
    for key, runs in draw_table.groupby([*keys, "criterion", "fovs"], sort=False, dropna=False):
        values = dict(zip([*keys, "criterion", "fovs"], key, strict=True))
        curve_rows.append({**values, **_curve_row(runs)})
    # The hold-out identity only clusters the Monte Carlo error; the ceiling reports it.
    draw_table = draw_table.drop(columns="holdout")
    curve_rows += _beyond_every_stream(ceiling_rows, curve_rows, criteria, counts, keys)
    curve = pd.DataFrame(curve_rows, columns=[*keys, *CURVE_COLUMNS])
    if len(curve):
        curve = _add_gain(curve, criteria, keys)
    settings = {
        "criteria": [c.parameters for c in criteria],
        "fov_counts": counts,
        "reservoir_fraction": float(reservoir_fraction),
        "block_col": block_col,
        "n_splits": n_splits,
        "draws": draws,
        "random_state": random_state,
        "labels": {c.name: c.label for c in criteria},
    }
    ceiling = _ceiling(pd.DataFrame(ceiling_rows), keys)
    skipped_table = pd.DataFrame(skipped, columns=[*keys, "fovs", "reason"])
    return ReliabilityResult(curve, draw_table, tuple(keys), settings, ceiling, skipped_table)


def _beyond_every_stream(
    ceiling_rows: list[dict],
    curve_rows: list[dict],
    criteria: Sequence[OutcomeCriterion],
    counts: Sequence[int],
    keys: list[str],
) -> list[dict]:
    """Rows saying so for listed counts no split of a group could draw."""
    have = {(tuple(r[k] for k in keys), r["criterion"], r["fovs"]) for r in curve_rows}
    streams: dict[tuple, int] = {}
    for row in ceiling_rows:
        group = tuple(row[k] for k in keys)
        streams[group] = max(streams.get(group, 0), int(row["stream_fovs"]))
    rows = []
    for group, stream in streams.items():
        for criterion in criteria:
            for k in counts:
                if (group, criterion.name, k) in have:
                    continue
                rows.append(
                    {
                        **dict(zip(keys, group, strict=True)),
                        "criterion": criterion.name,
                        "fovs": k,
                        "splits": 0,
                        "defined_splits": 0,
                        "preserved": None,
                        "preserved_split_sd": None,
                        "preserved_mc_se": None,
                        "defined_draws": 0,
                        "undefined_draws": 0,
                        "metric_median": None,
                        "metric_p10": None,
                        "metric_p90": None,
                        "reservoir_fovs": None,
                        "stream_fovs": stream,
                    }
                )
    return rows


def _ceiling(table: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Share of splits in which the whole stream preserved each criterion.

    ``stream_fovs`` is the longest stream and ``shortest_stream_fovs`` the
    shortest (they differ with unequal blocks); ``reservoir_fovs`` is the
    smallest reservoir.
    """
    columns = [
        *keys,
        "criterion",
        "splits",
        "defined_splits",
        "whole_stream_preserved",
        "stream_fovs",
        "shortest_stream_fovs",
        "reservoir_fovs",
        "distinct_holdouts",
    ]
    if table.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for key, runs in table.groupby([*keys, "criterion"], sort=False, dropna=False):
        defined = runs[runs["status"] != Preservation.UNDEFINED.value]
        share = (defined["status"] == Preservation.PRESERVED.value).mean() if len(defined) else None
        values = dict(zip([*keys, "criterion"], key, strict=True))
        rows.append(
            {
                **values,
                "splits": len(runs),
                "defined_splits": len(defined),
                "whole_stream_preserved": (None if share is None else round(float(share), _DIGITS)),
                "stream_fovs": int(runs["stream_fovs"].max()),
                "shortest_stream_fovs": int(runs["stream_fovs"].min()),
                "reservoir_fovs": int(runs["reservoir_fovs"].min()),
                "distinct_holdouts": int(runs["holdout"].nunique()),
            }
        )
    return pd.DataFrame(rows, columns=columns)


__all__ = ["CURVE_COLUMNS", "ReliabilityResult", "reliability_summary", "sampling_reliability"]
