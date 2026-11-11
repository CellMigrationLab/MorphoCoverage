"""Unusual (out-of-distribution) observations against a declared reference (questions R4).

**``unusual_objects``** labels out-of-distribution objects: objects farther from a
declared reference population, in standardized feature space, than all but
a share ``quantile`` of reference objects are from reference objects of other
fields. The design follows fiberfeatures' ``fit_pixel_categories``: fit on
declared reference data of the same dataset (here, control groups), with every
reference field contributing the same number of observations (at most
``per_fov``, sampled with ``random_state``), record the provenance, and refit
with other seeds to show whether the labels depend on the sample. It differs
where the purpose differs: fiberfeatures fits a Gaussian mixture (scikit-learn)
on normal scores to name categories; MorphoCoverage has no scikit-learn
dependency, and rare states are exactly what a mixture tends to absorb, so
this scores distance to the reference instead: the mean Euclidean distance to
the ``n_neighbors`` nearest reference objects (``scipy.spatial.cKDTree``) after
``transforms.Standardize`` fitted on the reference sample, which keeps
extreme values extreme where a quantile transform would clip them.

Two choices keep the FOV-clustering test honest. A group is never its own
reference: a control group is scored against the other control groups, so
labels never depend on how the tested group's objects sit in its fields. And
the threshold is calibrated the way groups are scored, leaving out one group
at a time: each reference object is scored against the reference objects of
the **other** reference groups, so a shift between groups (a repeat's batch
effect, say) raises the threshold as it raises the scores of the groups
tested, instead of flagging a control group three times as often as the
share asked for. This holds for every group scored, control or not; a tested
object is scored against all reference groups but one, drawn uniformly, and in
the threshold the scores of every left-out group weigh the same whatever its
observations, so calibration and scoring leave a group out the same way and
see an equally dense reference (owner, 2026-10-05). With few reference
groups the threshold rests on few between-group differences and varies from
dataset to dataset: under a repeat shift of half an SD, a nominal 1% flagged
1.2% of control objects on average with six repeats but 2.5% with three
(median 1.3%). With a single reference group (no other group to leave in),
its fields take the groups' place: reference objects are scored against its
other fields, tested objects against all its fields but one, drawn uniformly,
each field weighing the same (owner, 2026-10-05), and the result says so
(``calibration``): a shift between that group and the tested one then raises
the share flagged. A control group with no other control group to compare with
is not scored.

The leave-out was chosen by simulation (``data/simulations/unusual_calibration.py``,
400 datasets a design, the tested group drawn like the reference groups, a nominal 5%).
Leaving groups out by their size in the threshold but uniformly for tested objects, as
before, flagged 3.5% with reference groups of 3, 10 and 30 fields and 3.9% with 2 to 40;
scoring tested objects against the whole of a single reference group flagged 4.0% with 4
fields and 4.7% with 10. Leaving a group out by size for both, or uniformly for both, gave
4.9-5.1% with several groups; under a repeat shift of half an SD with unequal groups the
uniform scheme flagged 5.1-5.8%, the size-weighted one 5.4-6.1%, so the uniform one is
used. With a single reference group of 10 fields it flags 5.3%, of 4 fields 5.5%: the
remaining excess is ``np.quantile``'s, about ``(1 + q(N - 1)) / (N + 1)`` for ``N``
reference observations.

In a screen of sub-experiments (the
conditions sharing their repeats and control, ``_experiments``), a group is scored against
the reference groups of its own sub-experiment (``within_experiment``, owner 2026-10-02), so
one set's control never stands for another's.
Observations with a missing feature value are not scored, and a group none of
whose observations has every feature is reported as not assessed.

Shared: ``clustering`` re-exports ``UnusualObjects`` and ``unusual_objects``,
whose labels ``fov_clustering`` takes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np
import pandas as pd

from .._acquisition import ACQUISITION, acquisition_identity
from .._caveats import caveat_lines
from .._checks import as_count, check_seed
from .._frozen import owned
from .._repeat_alias import refuse_repeat_alias
from .._repeat_groups import pair_by_repeat_group
from .._seeds import group_key, group_rng
from .._style import check_detail
from ..metadata import CONDITION_COLUMN, REPEAT_COLUMN, REPEAT_GROUP_COLUMN
from ..schema import ValidatedInput, fov_positions, observation_words
from ..transforms import Standardize
from ._common import groups, matching_groups, validated
from ._experiments import repeat_experiments, same_experiment
from ._outliers import reference_sample, score
from ._state_text import SEED_AGREEMENT_WARNING, unusual_full, unusual_parts


@owned("labels", "distances", "groups", "settings")
@dataclass(frozen=True)
class UnusualObjects:
    """Out-of-distribution labels for every observation, against a declared reference.

    - ``labels``: aligned with the input ``objects`` (same index): ``True``
      for unusual, ``False`` for typical, missing where not scored.
    - ``distances``: the scores behind the labels, aligned the same way.
    - ``groups``: per analysis group, the reference groups used, the
      threshold and how it was calibrated (``calibration``: leave-one-group-out,
      or leave-one-field-out with a single reference group), the scored and
      unusual observations, and, when ``seed_checks`` were asked for, the
      agreement (Jaccard index of the unusual objects) of refits with other
      seeds, with how many failed (``seed_failures``) and why (``seed_failure``); with no
      unusual observation, no agreement but the most the refits flag (``refit_unusual``);
      ``reference_group`` marks the control groups, whose share is the background.
    """

    labels: pd.Series
    distances: pd.Series
    groups: pd.DataFrame
    group_cols: tuple[str, ...]
    settings: dict[str, Any]

    def summary(self, detail: str = "short") -> str:
        """The unusual observations in words: short (the groups in one table, each caveat
        once) or ``detail="full"`` (with each group's threshold and its calibration)."""
        full = check_detail(detail) == "full"
        text = unusual_full(self) if full else "\n".join(unusual_parts(self)[0])
        return "\n".join([text, *caveat_lines(unusual_parts(self)[1], indent=0, detail=detail)])


def unusual_objects(
    objects: pd.DataFrame,
    fovs: pd.DataFrame,
    *,
    feature_cols: Sequence[str],
    reference: Mapping[str, Any],
    quantile: float = 0.01,
    n_neighbors: int = 5,
    per_fov: int = 100,
    random_state: int = 0,
    seed_checks: int = 2,
    within_experiment: bool = True,
    fov_col: str | None = None,
    group_cols: Sequence[str] | None = None,
    observation_unit: str | None = None,
    own: Mapping[Any, Any] | None = None,
    weights: Mapping[str, float] | None = None,
) -> UnusualObjects:
    """Label observations that lie outside a declared reference population.

    ``reference`` selects the reference groups by FOV metadata, for example
    ``{"condition": "MOCK"}``; it is required, because "unusual" is only
    defined against something. ``quantile`` is the share of reference objects
    that would be called unusual when scored against other reference FOVs.
    ``per_fov`` caps the observations each reference FOV contributes, so
    every reference FOV weighs the same. Labels are aligned with the rows of
    ``objects`` (its index is kept). ``seed_checks`` refits with the
    next seeds and reports how much the labels agree. ``within_experiment`` scores each
    group against the reference groups of its own sub-experiment only (the biological
    repeats its conditions share, ``_experiments.repeat_experiments``). ``own`` maps each
    condition to its own control (several declared controls, owner 2026-10-05): a group is
    then scored against its own control's groups only, among ``reference``'s. ``weights``
    multiply the standardised features, a fixed weight per feature (the others 1); the
    report gives none, as unusual observations read the explainable features only (owner,
    2026-10-06).
    """
    refuse_repeat_alias(reference=reference)
    # The sub-experiments are found from the repeat groups (#617); conditions by their IDs.
    repeat_col = REPEAT_GROUP_COLUMN if within_experiment else None
    objects, fovs = pair_by_repeat_group(objects, fovs, repeat_col, control=reference)[:2]
    if not 0 < quantile < 1:
        raise ValueError("quantile must be between 0 and 1")
    n_neighbors = as_count(n_neighbors, "n_neighbors", 1)
    per_fov = as_count(per_fov, "per_fov", 1)
    seed_checks = as_count(seed_checks, "seed_checks", 0)
    random_state = check_seed(random_state)
    data = validated(objects, fovs, feature_cols, fov_col, group_cols, observation_unit)
    all_groups = groups(data)
    reference_groups = matching_groups(all_groups, reference, "reference")
    by_key = {group_key(g): d for g, d in all_groups}
    matrix = data.feature_matrix()
    scale = np.array([float(weights.get(c, 1.0)) for c in data.feature_cols]) if weights else None
    context = _Scoring(quantile, n_neighbors, per_fov, random_state, seed_checks, scale)
    positions_all = np.arange(len(data.objects))
    labels = np.full(len(data.objects), np.nan)
    distances = np.full(len(data.objects), np.nan)
    observation_group = _observation_groups(data, all_groups)
    rows = []
    reference_keys = {group_key(g) for g in reference_groups}
    block = _experiment_block(all_groups, reference_groups, within_experiment)
    for g_index, (group, _) in enumerate(all_groups):
        others = _others(group, reference_groups, block, own)
        row: dict[str, Any] = {**group, "reference_groups": [dict(g) for g in others]}
        # A control group, scored against the other control groups: the background share.
        row["reference_group"] = group_key(group) in reference_keys
        mine = positions_all[observation_group == g_index]
        if not others:
            rows.append({**row, "status": _no_reference(group, reference_groups, block)})
            continue
        refs = [by_key[group_key(g)] for g in others]
        results = _fits(group, refs, matrix, mine, data.feature_cols, context)
        columns, scores = _scored(group, others, results, data)
        if scores is not None:
            distances[mine] = scores
            labels[mine] = np.where(np.isnan(scores), np.nan, scores > columns["threshold"])
        rows.append({**row, **columns})
    # the validated table keeps the order of the input rows (metadata joins are left joins)
    label_series = pd.Series(labels, index=objects.index).map(
        lambda v: pd.NA if np.isnan(v) else bool(v)
    )
    return UnusualObjects(
        label_series,
        pd.Series(distances, index=objects.index),
        pd.DataFrame(rows),
        tuple(data.group_cols),
        _settings(data, reference, own, context, within_experiment, block, weights),
    )


class _Scoring(NamedTuple):
    """The checked settings every group of ``unusual_objects`` is scored with."""

    quantile: float
    n_neighbors: int
    per_fov: int
    random_state: int
    seed_checks: int
    scale: np.ndarray | None


def _observation_groups(
    data: ValidatedInput, all_groups: list[tuple[dict[str, Any], ValidatedInput]]
) -> np.ndarray:
    """Each observation's analysis group, by its position in ``all_groups``."""
    # Each observation's field by the validated lookup (one type and spelling per identity
    # value, #420), then each field's group.
    key = list(data.fov_key)
    row_of = {fov: i for i, fov in enumerate(zip(*(data.fovs[c].tolist() for c in key)))}
    observation_fov = fov_positions(data, data.fovs)
    fov_group = np.empty(len(data.fovs), dtype=np.int64)
    for g_index, (_, group_data) in enumerate(all_groups):
        members = [row_of[f] for f in zip(*(group_data.fovs[c].tolist() for c in key))]
        fov_group[members] = g_index
    return fov_group[observation_fov]


def _experiment_block(
    all_groups: list[tuple[dict[str, Any], ValidatedInput]],
    reference_groups: list[dict[str, Any]],
    within_experiment: bool,
) -> dict[Any, Any]:
    """Each repeat's sub-experiment when groups are scored within their own, else empty."""
    if not within_experiment:
        return {}
    return repeat_experiments(
        [g for g, _ in all_groups],
        {group_key(g) for g in reference_groups},
        CONDITION_COLUMN,
        REPEAT_COLUMN,
    )


def _others(
    group: dict[str, Any],
    reference_groups: list[dict[str, Any]],
    block: dict[Any, Any],
    own: Mapping[Any, Any] | None,
) -> list[dict[str, Any]]:
    """The reference groups ``group`` is scored against: never itself, those of its own
    sub-experiment, and with ``own`` its own control's only."""
    target = None if own is None else own.get(group.get(CONDITION_COLUMN))
    return [
        g
        for g in reference_groups
        if group_key(g) != group_key(group)
        and same_experiment(block, g.get(REPEAT_COLUMN), group.get(REPEAT_COLUMN))
        and (own is None or g.get(CONDITION_COLUMN) == target)
    ]


def _no_reference(
    group: dict[str, Any], reference_groups: list[dict[str, Any]], block: dict[Any, Any]
) -> str:
    """Why a group has no reference group to be scored against."""
    elsewhere = any(group_key(g) != group_key(group) for g in reference_groups)
    return (
        "no reference group in the repeats of its sub-experiment"
        if elsewhere and block
        else "no other reference group to score against"
    )


def _fits(
    group: dict[str, Any],
    refs: list[ValidatedInput],
    matrix: np.ndarray,
    mine: np.ndarray,
    feature_cols: Sequence[str],
    context: _Scoring,
) -> list[Any]:
    """The group's observations scored against a reference sample drawn with its seed, then
    with each of the ``seed_checks`` next seeds: a result or the reason it failed, each."""
    results = []
    for seed in range(context.random_state, context.random_state + context.seed_checks + 1):
        rng = group_rng(seed, group)
        sample, fov_ids, group_ids = reference_sample(refs, context.per_fov, rng)
        results.append(
            score(
                sample,
                fov_ids,
                group_ids,
                matrix[mine],
                context.n_neighbors,
                context.quantile,
                rng,
                feature_cols,
                context.scale,
            )
        )
    return results


def _scored(
    group: dict[str, Any], others: list[dict[str, Any]], results: list[Any], data: ValidatedInput
) -> tuple[dict[str, Any], np.ndarray | None]:
    """A group's columns from its fits, and the scores behind its labels (``None`` when the
    first fit failed or scored nothing)."""
    if isinstance(results[0], str):
        return {"status": results[0]}, None
    scores, threshold, state, calibration = results[0]
    if np.isnan(scores).all():
        return {"status": f"not assessed: {_unassessed(group, others, scores, data)}"}, None
    flagged = scores > threshold
    # None unusual: no labels to agree on (a Jaccard index with an empty set read 0%);
    # the refits' own count is kept instead (owner, 2026-10-06, audit 10).
    none = not np.nansum(flagged)
    columns = {
        "status": "ok",
        "threshold": threshold,
        "calibration": calibration,
        "scored": int((~np.isnan(scores)).sum()),
        "unusual": int(np.nansum(flagged)),
        "unusual_share": float(np.nanmean(np.where(np.isnan(scores), np.nan, flagged))),
        **_seed_agreement(flagged, results[1:], none),
        "standardize_state": state,
    }
    return columns, scores


def _unassessed(
    group: dict[str, Any], others: list[dict[str, Any]], scores: np.ndarray, data: ValidatedInput
) -> str:
    """Why no observation of the group was scored."""
    # Values relative to the own control are missing where the repeat has none: that
    # is the reason, not a missing feature (audit 10, B4 and C9).
    repeat = group.get(REPEAT_COLUMN)
    if repeat is not None and all(g.get(REPEAT_COLUMN) != repeat for g in others):
        return "its repeat holds no own control"
    words = observation_words(data)[1]
    return f"none of its {len(scores)} {words} has a value of every feature"


def _seed_agreement(flagged: np.ndarray, refits: list[Any], none: bool) -> dict[str, Any]:
    """How the refits with other seeds agree with the labels (the lowest Jaccard index), or
    with no unusual observation the most a refit flags; the failed refits counted."""
    agreement, failed, counts = [], [], []
    for other in refits:
        if isinstance(other, str):
            # A refit that fails is a sign of instability too: counted, never dropped.
            failed.append(other)
            continue
        other_flagged = other[0] > other[1]
        counts.append(int(np.sum(other_flagged)))
        union = np.sum(flagged | other_flagged)
        agreement.append(float(np.sum(flagged & other_flagged) / union) if union else 1.0)
    return {
        "seed_agreement": min(agreement) if agreement and not none else None,
        "refit_unusual": max(counts) if counts and none else None,
        "seed_failures": len(failed),
        "seed_failure": failed[0] if failed else None,
    }


def _settings(
    data: ValidatedInput,
    reference: Mapping[str, Any],
    own: Mapping[Any, Any] | None,
    context: _Scoring,
    within_experiment: bool,
    block: dict[Any, Any],
    weights: Mapping[str, float] | None,
) -> dict[str, Any]:
    """The settings of ``unusual_objects``: the arguments and how observations are scored."""
    return {
        ACQUISITION: acquisition_identity(data),
        "feature_cols": list(data.feature_cols),
        "reference": dict(reference),
        # Several controls: each condition against its own (owner, 2026-10-05).
        "own": dict(own) if own is not None else None,
        "quantile": context.quantile,
        "n_neighbors": context.n_neighbors,
        "per_fov": context.per_fov,
        "random_state": context.random_state,
        "seed_checks": context.seed_checks,
        "within_experiment": bool(within_experiment and len(set(block.values())) > 1),
        "transform": Standardize().describe(),
        "weights": dict(weights) if weights else None,
        "distance": "mean Euclidean distance to the n_neighbors nearest reference observations",
        "observation_unit": data.observation_unit,
        "observation_unit_declared": data.observation_unit_declared,
    }


__all__ = ["SEED_AGREEMENT_WARNING", "UnusualObjects", "unusual_objects"]
