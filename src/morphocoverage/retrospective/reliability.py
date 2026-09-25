"""Sampling reliability: if I acquire ``k`` representative FOVs, how often is that enough?

The strategy-independent answer to "how many FOVs should I image next time?"
(roadmap Step 1, ``docs/proposals/microscopist_report_roadmap.md``). No
stopping rule is involved; whether a rule stops at the right time is a
separate, sequential question.

Design, per analysis group (one sample):

1. For each outer split ``s``, hold out a fixed **reservoir** of whole FOVs (or
   spatial blocks) with ``ReservoirSplit``; the remaining FOVs are the stream.
2. For each ``k`` in ``fov_counts`` and each of ``draws`` draws, take ``k``
   stream FOVs at random, as a representative acquisition would.
3. Assess every ``OutcomeCriterion`` of those ``k`` FOVs against the same
   reservoir, for every ``k`` in the split. A reference that shrank as ``k``
   grew would make the curve partly a curve of reference quality.
4. Per split, the share of draws that preserved the criterion is this
   sample's conditional reliability at ``k``; the curve reports its mean and
   spread over splits, the metric's distribution, and the information gain
   of ``k`` over the previous listed ``k``.

A draw of FOVs without any observations is acquired effort and counts as not
preserved (``OutcomeCriterion.assess``). A draw is undefined only when the
metric cannot be computed for other reasons, most often because the reservoir
lacks what the criterion needs (a rare state absent from the held-back FOVs);
the summary says how many draws and hold-outs could be judged, since the
answer is then conditional on the hold-outs that could.

Limits, stated with every result:

- ``k`` cannot exceed the stream: with 28 FOVs and a reservoir of 10, nothing
  is said beyond 18. A listed ``k`` beyond every split's stream gets a row that
  says so (``splits`` 0), so no sample disappears from later analyses; with
  unequal blocks, a ``k`` judged on only some splits says on how many.
- Draws and splits estimate one sample's behaviour; they are not independent
  replicates. Statements across samples count samples or repeats (Step 4).
- Random draws treat FOVs as exchangeable, as random-position sampling does;
  recorded, spatially ordered acquisition is a different question.
- Everything is conditional on the criteria: the metric, its tolerance and
  the features it reads.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..evaluator import FovSubsets, iter_analysis_groups
from ..reservoir import ReservoirSplit
from ..schema import validate_input_tables
from ._parallel import check_n_jobs, map_groups
from ._text import fields, group_identity, percent, where
from .criteria import OutcomeCriterion, Preservation

CURVE_COLUMNS = (
    "criterion",
    "fovs",
    "splits",
    "defined_splits",
    "preserved",
    "preserved_split_sd",
    "defined_draws",
    "undefined_draws",
    "metric_median",
    "metric_p10",
    "metric_p90",
    "gain",
    "reservoir_fovs",
    "stream_fovs",
)


@dataclass(frozen=True)
class ReliabilityResult:
    """The reliability curve, every assessed draw, and the settings that produced them.

    ``curve`` has one row per analysis group, criterion and FOV count (columns
    ``CURVE_COLUMNS`` after the group columns). ``draws`` has one row per
    group, split, FOV count, draw and criterion, with the metric value and the
    preservation. ``ceiling`` has one row per group and criterion: the share of
    splits in which the whole stream preserved the criterion. The reservoir is
    itself a finite sample, so this ceiling can be below 1 however well the
    stream was sampled, and the curve cannot be read above it. ``summary()``
    states the curve in sentences.
    """

    curve: pd.DataFrame
    draws: pd.DataFrame
    group_cols: tuple[str, ...]
    settings: Mapping[str, Any] = field(default_factory=dict)
    ceiling: pd.DataFrame = field(default_factory=pd.DataFrame)

    def summary(self) -> str:
        """The curve in plain sentences, one block per group and criterion."""
        return reliability_summary(self)


def _check_settings(criteria, fov_counts, reservoir_fraction, n_splits, draws, random_state):
    """Validate the settings; return the criteria list and sorted distinct counts."""
    criteria = list(criteria)
    if not criteria or not all(isinstance(c, OutcomeCriterion) for c in criteria):
        raise TypeError("criteria must be a non-empty sequence of OutcomeCriterion")
    if len({c.name for c in criteria}) != len(criteria):
        raise ValueError("criteria must have distinct names")
    counts = sorted(set(fov_counts))
    if not counts or not all(
        isinstance(k, int) and not isinstance(k, bool) and k >= 1 for k in counts
    ):
        raise ValueError("fov_counts must be positive integers")
    if isinstance(reservoir_fraction, bool) or not 0 < float(reservoir_fraction) < 1:
        raise ValueError("reservoir_fraction must be between 0 and 1")
    for value, label in ((n_splits, "n_splits"), (draws, "draws")):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{label} must be a positive integer")
    if isinstance(random_state, bool) or not isinstance(random_state, int):
        raise TypeError("random_state must be an integer seed")
    return criteria, counts


def _draw_rng(random_state: int, group: Mapping[str, Any], split: int) -> np.random.Generator:
    """A generator for one group and split: the same draws whatever other groups exist."""
    return np.random.default_rng([random_state, group_identity(group), split])


def _group_draws(group, data, criteria, counts, split_settings, n_splits, draws, random_state):
    """Every assessed draw for one analysis group, and the whole stream per split."""
    fraction, block_col = split_settings
    rows, ceiling = [], []
    for split in range(n_splits):
        stream, reservoir = ReservoirSplit(fraction, random_state + split, block_col).split(
            group, data
        )
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
                }
            )
        rng = _draw_rng(random_state, group, split)
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
                        }
                    )
    return rows, ceiling


def _curve_row(runs: pd.DataFrame) -> dict[str, Any]:
    """Aggregate the draws of one group, criterion and FOV count."""
    defined = runs[runs["status"] != Preservation.UNDEFINED.value]
    per_split = (
        defined.assign(ok=defined["status"] == Preservation.PRESERVED.value)
        .groupby("split")["ok"]
        .mean()
    )
    values = defined["value"].to_numpy(dtype=float)
    return {
        "splits": int(runs["split"].nunique()),
        "defined_splits": len(per_split),
        "preserved": float(per_split.mean()) if len(per_split) else None,
        "preserved_split_sd": float(per_split.std(ddof=1)) if len(per_split) > 1 else None,
        "defined_draws": len(defined),
        "undefined_draws": len(runs) - len(defined),
        "metric_median": float(np.median(values)) if len(values) else None,
        "metric_p10": float(np.quantile(values, 0.1)) if len(values) else None,
        "metric_p90": float(np.quantile(values, 0.9)) if len(values) else None,
        "reservoir_fovs": int(runs["reservoir_fovs"].min()),
        "stream_fovs": int(runs["stream_fovs"].max()),
    }


def _add_gain(curve: pd.DataFrame, criteria, keys: list[str]) -> pd.DataFrame:
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
    across splits. ``n_jobs`` runs analysis groups in that many worker
    processes (``-1``: every CPU); results are identical for every value.
    Other arguments are those of ``validate_input_tables``.
    """
    check_n_jobs(n_jobs)
    criteria, counts = _check_settings(
        criteria, fov_counts, reservoir_fraction, n_splits, draws, random_state
    )
    data = validate_input_tables(
        objects,
        fovs=fovs,
        feature_cols=feature_cols,
        fov_col=fov_col,
        group_cols=group_cols,
        observation_unit=observation_unit,
    )

    split_settings = (float(reservoir_fraction), block_col)
    jobs = [
        (group, group_data, criteria, counts, split_settings, n_splits, draws, random_state)
        for group, group_data in iter_analysis_groups(data)
    ]
    rows, ceiling_rows = [], []
    for group_rows, group_ceiling in map_groups(_group_draws, jobs, n_jobs):
        rows += group_rows
        ceiling_rows += group_ceiling
    keys = list(data.group_cols)
    draw_table = pd.DataFrame(
        rows,
        columns=[*keys, "split", "fovs", "draw", "criterion", "value", "status"]
        + ["reservoir_fovs", "stream_fovs"],
    )
    curve_rows = []
    for key, runs in draw_table.groupby([*keys, "criterion", "fovs"], sort=False, dropna=False):
        values = dict(zip([*keys, "criterion", "fovs"], key, strict=True))
        curve_rows.append({**values, **_curve_row(runs)})
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
    return ReliabilityResult(curve, draw_table, tuple(keys), settings, ceiling)


def _beyond_every_stream(ceiling_rows, curve_rows, criteria, counts, keys) -> list[dict]:
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
    """Share of splits in which the whole stream preserved each criterion."""
    columns = [
        *keys,
        "criterion",
        "splits",
        "defined_splits",
        "whole_stream_preserved",
        "stream_fovs",
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
                "whole_stream_preserved": None if share is None else float(share),
                "stream_fovs": int(runs["stream_fovs"].max()),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def reliability_summary(result: ReliabilityResult) -> str:
    """ "How many fields do you need?" in sentences, per group and criterion."""
    blocks = []
    keys = list(result.group_cols)
    labels = result.settings.get("labels", {})
    n_splits = int(result.settings.get("n_splits", 0))
    for key, rows in result.curve.groupby([*keys, "criterion"], sort=False, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        group = dict(zip(keys, key[: len(keys)], strict=True))
        criterion = key[-1]
        rows = rows.sort_values("fovs")
        stream = int(rows["stream_fovs"].max())
        lines = [
            f"{where(group)}: how many fields do you need for {labels.get(criterion, criterion)}?"
        ]
        for row in rows.itertuples(index=False):
            if row.splits == 0:
                lines.append(f"  {fields(row.fovs)}: more than the {stream} fields available")
                continue
            if row.preserved is None or pd.isna(row.preserved):
                lines.append(
                    f"  {fields(row.fovs)}: could not be judged ({row.undefined_draws} draws "
                    "undefined)"
                )
                continue
            spread = (
                ""
                if row.preserved_split_sd is None or pd.isna(row.preserved_split_sd)
                else f" (±{percent(row.preserved_split_sd)} across hold-outs)"
            )
            partial = []
            if n_splits and row.defined_splits < n_splits:
                partial.append(f"judged on {row.defined_splits} of {n_splits} hold-outs")
            if row.undefined_draws:
                total = row.undefined_draws + row.defined_draws
                partial.append(f"{percent(row.undefined_draws / total)} of draws not judged")
            note = f" [{'; '.join(partial)}]" if partial else ""
            lines.append(
                f"  {fields(row.fovs)}: enough in {percent(row.preserved)} of random "
                f"acquisitions{spread}{note}"
            )
        match = result.ceiling
        if not match.empty:
            match = match[match["criterion"] == criterion]
            for column, value in group.items():
                match = match[match[column] == value]
        if len(match) and match["whole_stream_preserved"].notna().iloc[0]:
            ceiling = match.iloc[0]
            defined = int(ceiling["defined_splits"])
            judged = (
                f" of the {defined} that could judge it" if defined < int(ceiling["splits"]) else ""
            )
            lines.append(
                f"  All {int(ceiling['stream_fovs'])} available fields together were enough in "
                f"{percent(float(ceiling['whole_stream_preserved']))} of hold-outs{judged}: the "
                "held-back fields are themselves a sample, so the curve cannot be read above "
                "this."
            )
        judged = rows[rows["splits"] > 0]
        smallest = int(judged["reservoir_fovs"].min()) if len(judged) else 0
        lines.append(
            f"  Judged on {smallest} or more held-back fields over "
            f"{n_splits or int(rows['splits'].max())} hold-outs; nothing is said beyond "
            f"{stream} fields, and fields are treated as random draws."
        )
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


__all__ = ["CURVE_COLUMNS", "ReliabilityResult", "reliability_summary", "sampling_reliability"]
