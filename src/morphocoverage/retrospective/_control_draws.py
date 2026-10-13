"""Private: the representation control's checks, one stratum at a time (``controls``).

Per stratum, the leave-one-field-out checks (reference, positive, planted) and the draws
after ``fovs_seen`` reference fields, counted per candidate kind. Every random choice is made
here, in stratum and draw order; only the novelty checks, which use none, go to worker
processes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .._parallel import map_units, worker_pool
from .._seeds import group_rng
from ..evaluator import FovSubsets, restrict_input
from ..schema import ValidatedInput, fov_positions, observation_words
from ..strategies.support_novelty import SupportNoveltyEstimator
from ._text import where

CANDIDATES = (
    "reference",
    "positive",
    "planted",
    "reference_lofo",
    "positive_lofo",
    "planted_lofo",
)


@dataclass(frozen=True)
class _ControlRun:
    """What every stratum of one ``representation_control`` call is measured with.

    ``observed`` flags the FOV rows holding observations; ``positive_mask`` and
    ``base_mask`` the positive and reference fields (the undeclared rest without a
    reference); ``with_reference`` says whether a reference was declared.
    """

    novelty: SupportNoveltyEstimator
    data: ValidatedInput
    subsets: FovSubsets
    observed: np.ndarray
    positive_mask: pd.Series
    base_mask: pd.Series
    lofo_cols: list[str]
    planted_objects: int | None
    with_reference: bool
    fovs_seen: int
    draws: int
    random_state: int
    n_jobs: int


@dataclass(frozen=True)
class _StrataCounts:
    """The table rows of every stratum, the strata skipped and why, the strata where nothing
    could be planted and why, and the positive fields without observations."""

    rows: list[dict]
    skipped: list[dict]
    planted_skipped: list[dict]
    positive_empty: int


def _planted(
    subsets: FovSubsets,
    host_rows: np.ndarray,
    donors: list[pd.DataFrame],
    size: int,
    rng: np.random.Generator,
) -> ValidatedInput | None:
    """Each host FOV with ``size`` objects of a random donor FOV (with enough objects) added."""
    usable = [cells for cells in donors if len(cells) >= size]
    if not usable:
        return None
    data = subsets.data
    hosts_input = subsets.take(host_rows)
    hosts = hosts_input.fovs
    key = list(data.fov_key)
    added = []
    for _, host in hosts.iterrows():
        cells = usable[rng.integers(len(usable))]
        picked = cells.iloc[np.sort(rng.choice(len(cells), size=size, replace=False))].copy()
        for column in key:
            picked[column] = host[column]
        added.append(picked)
    objects = pd.concat([hosts_input.objects, *added], ignore_index=True)
    return restrict_input(data, objects, hosts)


def _judge(
    novelty: SupportNoveltyEstimator, seen: ValidatedInput, candidates: Mapping[str, ValidatedInput]
) -> dict[str, tuple[int, int, int]]:
    """Per candidate kind: FOVs judged new, FOVs judged, FOVs that could not be judged."""
    counts = {}
    for kind, table in candidates.items():
        flags = novelty.novel_fovs_against(seen, table)
        if flags is None:
            # Missing values or too few FOVs: this check could not be made.
            counts[kind] = (0, 0, len(table.fovs))
        else:
            counts[kind] = (int(np.sum(flags)), len(flags), 0)
    return counts


def _leave_one_out(
    novelty: SupportNoveltyEstimator,
    subsets: FovSubsets,
    groups: list[np.ndarray],
    positive: np.ndarray,
    with_reference: bool,
) -> dict[str, tuple[int, int, int]]:
    """Per kind: fields judged new, judged, not judged, within each reference group of ``groups``.

    ``groups`` are the reference fields (FOV rows) of each analysis group of
    the stratum. Reference fields are judged leave one field out within their
    group, as ``inspect_fovs`` judges a field (a field with a new state is
    new; empty fields count, as there); each positive field against the
    reference fields of each group, one check per field and group.
    """
    counts = {kind: [0, 0, 0] for kind in ("reference_lofo", "positive_lofo")}
    positive_input = subsets.take(positive) if len(positive) else None
    for rows in groups:
        reference = subsets.take(rows)
        if with_reference:
            members = novelty.novel_observation_flags(reference, new_states_only=True)
            if members is None:
                counts["reference_lofo"][2] += len(rows)
            else:
                positions = fov_positions(reference, reference.fovs)
                counts["reference_lofo"][0] += len(np.unique(positions[members]))
                counts["reference_lofo"][1] += len(rows)
        if positive_input is not None:
            flags = novelty.novel_fovs_against(reference, positive_input)
            if flags is None:
                counts["positive_lofo"][2] += len(positive)
            else:
                counts["positive_lofo"][0] += int(np.sum(flags))
                counts["positive_lofo"][1] += len(flags)
    return {kind: (new, total, unjudged) for kind, (new, total, unjudged) in counts.items()}


def _planted_leave_one_out(
    novelty: SupportNoveltyEstimator,
    subsets: FovSubsets,
    groups: list[np.ndarray],
    donors: list[pd.DataFrame],
    size: int,
    rng: np.random.Generator,
) -> list[tuple]:
    """The ``planted_lofo`` checks: each reference field of a group, with ``size`` objects
    of a random positive field planted, against the other reference fields of its group."""
    jobs = []
    for rows in groups:
        for host in rows:
            planted = _planted(subsets, np.array([host]), donors, size, rng)
            if planted is not None:
                seen = subsets.take(rows[rows != host])
                jobs.append((novelty, seen, {"planted_lofo": planted}))
    return jobs


def _planting_skipped(donors: list[pd.DataFrame], size: int) -> str | None:
    """Why no objects can be planted in this stratum (no donor field holds enough), or None."""
    largest = max((len(cells) for cells in donors), default=0)
    if largest >= size:
        return None
    return f"no positive-control field holds {size} objects to plant (the largest holds {largest})"


def _groups_of(fovs: pd.DataFrame, rows: np.ndarray, columns: list[str]) -> list[np.ndarray]:
    """The FOV rows ``rows`` split by the values of ``columns`` (one part without columns)."""
    if not len(rows):
        return []
    if not columns:
        return [rows]
    parts = fovs.iloc[rows].groupby(columns, sort=False, dropna=False).indices
    return [rows[np.asarray(index)] for index in parts.values()]


def _stratum_rows(
    stratum: Mapping[str, Any],
    counts: Mapping[str, Sequence[int]],
) -> list[dict]:
    """One table row per candidate kind that was measured or could not be judged."""
    label = where(stratum) if stratum else "all FOVs"
    return [
        {
            **stratum,
            "stratum": label,
            "candidate": kind,
            "judged_new": new,
            "candidates": total,
            "not_judged": not_judged,
            "rate": new / total if total else None,
        }
        for kind, (new, total, not_judged) in counts.items()
        if total or not_judged
    ]


def _observations(data: ValidatedInput) -> str:
    """``cells``, ``objects``: the plural of the observation unit."""
    return observation_words(data)[1]


def _draws(
    run: _ControlRun,
    seeing: np.ndarray,
    base: np.ndarray,
    positive: np.ndarray,
    rng: np.random.Generator,
) -> tuple[list[tuple], str | None]:
    """The novelty checks of one stratum's draws, and why nothing was planted there (or None).

    Each draw sees ``fovs_seen`` reference fields holding observations; the
    other reference fields, empty ones included, are the reference candidates
    and the hosts of planted objects.
    """
    subsets, fovs_seen = run.subsets, run.fovs_seen
    positive_input = subsets.take(positive)
    donors = [run.data.objects.loc[subsets.positions == row] for row in positive]
    plant = run.planted_objects is not None
    gap = None
    if plant:
        gap = _planting_skipped(donors, run.planted_objects)
        plant = gap is None
    jobs = []
    for _ in range(run.draws):
        order = rng.permutation(len(seeing))
        seen_rows = seeing[order[:fovs_seen]]
        seen = subsets.take(seen_rows)
        # The unseen fields holding observations, in the draw's order, then the empty ones.
        rest = np.concatenate([seeing[order[fovs_seen:]], base[~np.isin(base, seeing)]])
        candidates = {"positive": positive_input}
        if run.with_reference:
            candidates["reference"] = subsets.take(rest)
        if plant:
            planted = _planted(subsets, rest, donors, run.planted_objects, rng)
            if planted is not None:
                candidates["planted"] = planted
        jobs.append((run.novelty, seen, candidates))
    return jobs, gap


def _planted_lofo_counts(
    run: _ControlRun,
    stratum: Mapping[str, Any],
    reference_groups: list[np.ndarray],
    positive: np.ndarray,
    pool: Any,
) -> list[dict[str, tuple[int, int, int]]]:
    """The judged ``planted_lofo`` checks of one stratum (none without planting)."""
    if run.planted_objects is None or not len(positive):
        return []
    donors = [run.data.objects.loc[run.subsets.positions == row] for row in positive]
    if _planting_skipped(donors, run.planted_objects) is not None:
        return []
    # Its own generator, so the draws keep their random choices.
    jobs = _planted_leave_one_out(
        run.novelty,
        run.subsets,
        reference_groups,
        donors,
        run.planted_objects,
        group_rng(run.random_state, stratum, 1),
    )
    return map_units(_judge, jobs, run.n_jobs, pool)


def _skip_reason(
    run: _ControlRun, declared: np.ndarray, positive: np.ndarray, seeing: np.ndarray
) -> str | None:
    """Why no draw can be made in this stratum, or None when draws can be made."""
    if len(positive) and len(seeing) > run.fovs_seen:
        return None
    # Not imaged there, or imaged empty: apart (audit 10, A24).
    if len(declared) == 0:
        return "the positive control was not imaged here"
    if len(positive) == 0:
        return f"no positive-control field with {_observations(run.data)}"
    return (
        f"{len(seeing)} reference fields with {_observations(run.data)}, not more "
        f"than the {run.fovs_seen} seen"
    )


def _add_counts(counts: dict[str, list[int]], draws: list[dict[str, tuple[int, int, int]]]) -> None:
    """Add each draw's fields judged new, judged and not judged to ``counts``, per kind."""
    for draw in draws:
        for kind, (new, total, not_judged) in draw.items():
            counts[kind][0] += new
            counts[kind][1] += total
            counts[kind][2] += not_judged


def _measure_stratum(
    run: _ControlRun, stratum: Mapping[str, Any], pool: Any
) -> tuple[list[dict], dict | None, dict | None, int]:
    """One stratum's table rows, why it was skipped and why nothing was planted there (each
    None when not), and its positive fields without observations."""
    # One generator per stratum, so adding a plate never changes another plate's rates.
    rng = group_rng(run.random_state, stratum)
    fovs = run.data.fovs
    inside = pd.Series(True, index=fovs.index)
    for column, value in stratum.items():
        inside &= fovs[column].eq(value)
    base = np.flatnonzero((inside & run.base_mask).to_numpy())
    declared = np.flatnonzero((inside & run.positive_mask).to_numpy())
    # An empty positive field cannot show the phenotype: it is counted apart,
    # never as a miss (P1-2).
    positive = declared[run.observed[declared]]
    label = where(stratum) if stratum else "all FOVs"
    counts = {kind: [0, 0, 0] for kind in CANDIDATES}  # new, judged, not judged
    # Judged as inspect_fovs judges a field, within its analysis group; measured
    # wherever the fields exist, also in a stratum the draws below cannot use.
    reference_groups = _groups_of(fovs, base, run.lofo_cols)
    lofo = _leave_one_out(run.novelty, run.subsets, reference_groups, positive, run.with_reference)
    planted_lofo = _planted_lofo_counts(run, stratum, reference_groups, positive, pool)
    # The seen fields hold observations: an empty field adds nothing to the reference.
    seeing = base[run.observed[base]]
    reason = _skip_reason(run, declared, positive, seeing)
    skipped = gap = None
    if reason is not None:
        skipped = {**stratum, "stratum": label, "reason": reason}
        judged = []
    else:
        jobs, why = _draws(run, seeing, base, positive, rng)
        if why is not None:
            # Said, not silently absent: the planted rate was declared.
            gap = {**stratum, "stratum": label, "reason": why}
        judged = map_units(_judge, jobs, run.n_jobs, pool)
    _add_counts(counts, [*judged, lofo, *planted_lofo])
    return _stratum_rows(stratum, counts), skipped, gap, len(declared) - len(positive)


def _measure_strata(run: _ControlRun, strata: list[dict[str, Any]]) -> _StrataCounts:
    """Every stratum measured in turn, with one pool of worker processes for all."""
    rows, skipped, planted_skipped = [], [], []
    positive_empty = 0
    # Every random choice is made here, in stratum and draw order; only the novelty checks,
    # which use none, go to worker processes. One stratum's draws at a time keeps memory low.
    with worker_pool(run.n_jobs) as pool:
        for stratum in strata:
            own, skip, gap, empty = _measure_stratum(run, stratum, pool)
            positive_empty += empty
            if skip is not None:
                skipped.append(skip)
            if gap is not None:
                planted_skipped.append(gap)
            rows += own
    return _StrataCounts(rows, skipped, planted_skipped, positive_empty)


__all__ = ["CANDIDATES", "_ControlRun", "_StrataCounts", "_measure_strata"]
