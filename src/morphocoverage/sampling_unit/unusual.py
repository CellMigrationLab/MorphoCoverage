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
object is scored against all reference groups but one (drawn at random), so
it sees as dense a reference as the calibration did. With few reference
groups the threshold rests on few between-group differences and varies from
dataset to dataset: under a repeat shift of half an SD, a nominal 1% flagged
1.2% of control objects on average with six repeats but 2.5% with three
(median 1.3%). With a
single reference group (no other group to leave in), the threshold comes from
reference objects scored against reference objects of other fields of that
group, and the result says so (``calibration``): a shift between that group
and the tested one then raises the share flagged. A control group with no
other control group to compare with is not scored.
Observations with a missing feature value are not scored, and a group none of
whose observations has every feature is reported as not assessed.

Shared: ``clustering`` re-exports ``UnusualObjects`` and ``unusual_objects``,
whose labels ``fov_clustering`` takes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .._checks import as_count, check_seed
from .._seeds import group_key, group_rng
from .._style import caution_lines, check_detail
from ..schema import unit_plural
from ..transforms import Standardize
from ._common import groups, matching_groups, validated
from ._outliers import reference_sample, score
from ._state_text import SEED_AGREEMENT_WARNING, unusual_full, unusual_parts


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
      seeds.
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
        return "\n".join([text, *caution_lines(unusual_parts(self)[1], indent=0)])


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
    fov_col: str | None = None,
    group_cols: Sequence[str] | None = None,
    observation_unit: str | None = None,
) -> UnusualObjects:
    """Label observations that lie outside a declared reference population.

    ``reference`` selects the reference groups by FOV metadata, for example
    ``{"condition": "MOCK"}``; it is required, because "unusual" is only
    defined against something. ``quantile`` is the share of reference objects
    that would be called unusual when scored against other reference FOVs.
    ``per_fov`` caps the observations each reference FOV contributes, so
    every reference FOV weighs the same. Labels are aligned with the rows of
    ``objects`` (its index is kept). ``seed_checks`` refits with the
    next seeds and reports how much the labels agree.
    """
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
    positions_all = np.arange(len(data.objects))
    labels = np.full(len(data.objects), np.nan)
    distances = np.full(len(data.objects), np.nan)
    # map each group's observations back to rows of the validated table
    key = list(data.fov_key)
    row_of = {fov: i for i, fov in enumerate(zip(*(data.fovs[c].tolist() for c in key)))}
    observation_fov = np.fromiter(
        (row_of[f] for f in zip(*(data.objects[c].tolist() for c in key))),
        dtype=np.int64,
        count=len(data.objects),
    )
    fov_group = np.empty(len(data.fovs), dtype=np.int64)
    for g_index, (_, group_data) in enumerate(all_groups):
        members = [row_of[f] for f in zip(*(group_data.fovs[c].tolist() for c in key))]
        fov_group[members] = g_index
    observation_group = fov_group[observation_fov]
    rows = []
    for g_index, (group, _) in enumerate(all_groups):
        others = [g for g in reference_groups if group_key(g) != group_key(group)]
        row: dict[str, Any] = {**group, "reference_groups": [dict(g) for g in others]}
        mine = positions_all[observation_group == g_index]
        if not others:
            rows.append({**row, "status": "no other reference group to score against"})
            continue
        refs = [by_key[group_key(g)] for g in others]
        results = []
        for seed in range(random_state, random_state + seed_checks + 1):
            rng = group_rng(seed, group)
            sample, fov_ids, group_ids = reference_sample(refs, per_fov, rng)
            results.append(
                score(
                    sample,
                    fov_ids,
                    group_ids,
                    matrix[mine],
                    n_neighbors,
                    quantile,
                    rng,
                    data.feature_cols,
                )
            )
        if isinstance(results[0], str):
            rows.append({**row, "status": results[0]})
            continue
        scores, threshold, state, calibration = results[0]
        if np.isnan(scores).all():
            words = unit_plural(data.observation_unit)
            rows.append(
                {
                    **row,
                    "status": f"not assessed: none of its {len(scores)} {words} has a value of "
                    "every feature",
                }
            )
            continue
        flagged = scores > threshold
        distances[mine] = scores
        labels[mine] = np.where(np.isnan(scores), np.nan, flagged)
        agreement = []
        for other in results[1:]:
            if isinstance(other, str):
                continue
            other_flagged = other[0] > other[1]
            union = np.sum(flagged | other_flagged)
            agreement.append(float(np.sum(flagged & other_flagged) / union) if union else 1.0)
        rows.append(
            {
                **row,
                "status": "ok",
                "threshold": threshold,
                "calibration": calibration,
                "scored": int((~np.isnan(scores)).sum()),
                "unusual": int(np.nansum(flagged)),
                "unusual_share": float(np.nanmean(np.where(np.isnan(scores), np.nan, flagged))),
                "seed_agreement": min(agreement) if agreement else None,
                "standardize_state": state,
            }
        )
    # the validated table keeps the order of the input rows (metadata joins are left joins)
    label_series = pd.Series(labels, index=objects.index).map(
        lambda v: pd.NA if np.isnan(v) else bool(v)
    )
    settings = {
        "feature_cols": list(data.feature_cols),
        "reference": dict(reference),
        "quantile": quantile,
        "n_neighbors": n_neighbors,
        "per_fov": per_fov,
        "random_state": random_state,
        "seed_checks": seed_checks,
        "transform": Standardize().describe(),
        "distance": "mean Euclidean distance to the n_neighbors nearest reference observations",
        "observation_unit": data.observation_unit,
    }
    return UnusualObjects(
        label_series,
        pd.Series(distances, index=objects.index),
        pd.DataFrame(rows),
        tuple(data.group_cols),
        settings,
    )


__all__ = ["SEED_AGREEMENT_WARNING", "UnusualObjects", "unusual_objects"]
