"""Are rare or unusual objects spread over the FOVs, or clustered in a few? (questions R4)

An object-count argument about rare states assumes random objects: the chance
of missing a state carried by a share ``p`` of objects in ``n`` objects is
``(1 - p) ** n``. A microscope acquires fields, and what decides whether a
state is seen is the share of **fields** that contain it. When rare objects
cluster (a mitotic group, a clone, a damaged patch, the well edge), many
objects from a few fields can miss a state that the object count says is
covered.

**``fov_clustering``** takes any per-observation state labels: unusual
objects from ``unusual_objects``, clusters from a mixture model fitted elsewhere
(such as fiberfeatures' pixel categories summarized per patch), a gate on one
feature, or manual annotations. Labels are data here, never inferred from
features. Per analysis group and state it reports:

- the state's share of objects and the share of acquired fields holding it;
- the fields expected to hold it if the same objects were spread at random over
  the same fields (hypergeometric, ``scipy.stats.hypergeom``), and the
  **clustering ratio** expected / observed (1: spread like random objects);
- a permutation p-value: how often labels shuffled across the group's objects
  reach as few fields as observed;
- the fields needed to see the state once with ``detection_probability``,
  from the observed field incidence (``rare_planning.first_detection_fovs``),
  with a Clopper-Pearson bound from the lower bound of that incidence, beside
  the fields a random-object argument would ask for.

Fields without observations count as acquired fields that did not show the
state. This is the retrospective, labelled counterpart of the rare-state
planning of ``strategies.rare_planning``; a state is only as good as its
labels.

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

Two choices keep the FOV-clustering test honest. The threshold comes from
reference objects scored against reference objects of **other** fields, so a
reference field is never its own reference. And a group is never its own
reference: a control group is scored against the other control groups, so
labels never depend on how the tested group's objects sit in its fields. A
control group with no other control group to compare with is not scored.
Observations with a missing feature value are not scored.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial import cKDTree

from ..bounds import upper_bound
from ..retrospective._text import fields as fields_text
from ..retrospective._text import group_identity, unit_plural, where
from ..schema import ValidatedInput, fov_positions
from ..strategies.rare_planning import first_detection_fovs
from ..transforms import Standardize
from ._common import check_count, check_seed, groups, matching_groups, validated

_LABEL_CODE = "_morphocoverage_state_code"
# Seed-check agreement below which the labels are called unstable (as in fiberfeatures).
SEED_AGREEMENT_WARNING = 0.9


@dataclass(frozen=True)
class FOVClustering:
    """How each labelled state is spread over the fields, per analysis group.

    ``table`` has one row per group and state; see ``fov_clustering`` for
    the columns.
    """

    table: pd.DataFrame
    group_cols: tuple[str, ...]
    settings: dict[str, Any]

    def state_name(self, state: Any) -> str:
        """A True/False state by its label column (``unusual``, ``not unusual``), else the state."""
        if isinstance(state, (bool, np.bool_)):
            return self.settings["label_col"] if state else f"not {self.settings['label_col']}"
        return f"state {state!r}"

    def summary(self) -> str:
        """Plain sentences, one per group and state."""
        words = unit_plural(self.settings["observation_unit"])
        chance = self.settings["detection_probability"]
        lines = [f"Are the states spread over the fields? Labels {self.settings['label_col']}."]
        for row in self.table.to_dict("records"):
            head = f"{where({c: row[c] for c in self.group_cols})}, {self.state_name(row['state'])}"
            if not row["labelled"]:
                lines.append(
                    f"{head}: no {words.removesuffix('s')} of its "
                    f"{fields_text(int(row['fovs']))} is labelled, so it was not looked for."
                )
                continue
            if not row["state_objects"]:
                lines.append(
                    f"{head}: not seen among {int(row['labelled'])} labelled {words} in "
                    f"{fields_text(int(row['fovs']))}."
                )
                continue
            upper = row["fovs_needed_upper"]
            bound = "" if pd.isna(upper) else f" (up to {int(upper)})"
            lines.append(
                f"{head}: {int(row['state_objects'])} {words} ({row['prevalence']:.1%}) in "
                f"{int(row['fovs_with_state'])} of {fields_text(int(row['fovs']))}; random "
                f"{words} would fill {row['expected_fovs_if_random']:.1f} (clustering ratio "
                f"{row['clustering_ratio']:.1f}, p = {row['p_clustered']:.2g}). To see it "
                f"with {chance:.0%} chance: {fields_text(int(row['fovs_needed']))}{bound} from "
                f"its field incidence, against {fields_text(int(row['fovs_needed_if_random']))} "
                f"if the {words} were random."
            )
        return "\n".join(lines)


def fov_clustering(
    objects: pd.DataFrame,
    fovs: pd.DataFrame,
    *,
    label_col: str,
    states: Sequence[Any] | None = None,
    detection_probability: float = 0.95,
    confidence: float = 0.95,
    n_permutations: int = 999,
    random_state: int = 0,
    fov_col: str | None = None,
    group_cols: Sequence[str] | None = None,
    observation_unit: str | None = None,
) -> FOVClustering:
    """How each state in ``objects[label_col]`` is spread over the acquired fields.

    ``states`` are the label values to assess (default: every value present).
    Missing labels mark observations that were not labelled; they are left
    out. Columns of the result's ``table``: the group, ``state``, the
    ``labelled`` observations, ``state_objects``, ``prevalence``, ``fovs``
    (acquired), ``fovs_with_state``, ``fov_incidence``,
    ``expected_fovs_if_random``, ``clustering_ratio``, ``p_clustered``,
    ``largest_fov_share`` (the share of the state's objects in its fullest
    field), ``fovs_needed`` and ``fovs_needed_upper`` (from the observed field
    incidence and its lower ``confidence`` bound), and
    ``fovs_needed_if_random`` (from the object prevalence and the mean
    observations per acquired field).
    """
    if not isinstance(label_col, str) or label_col not in objects.columns:
        raise ValueError(f"label_col {label_col!r} is not a column of objects")
    if not 0 < detection_probability < 1 or not 0 < confidence < 1:
        raise ValueError("detection_probability and confidence must be between 0 and 1")
    check_count(n_permutations, "n_permutations", 1)
    check_seed(random_state)
    labels = objects[label_col]
    chosen = list(pd.unique(labels.dropna())) if states is None else list(states)
    codes = pd.Series(np.nan, index=objects.index)
    lookup = {state: i for i, state in enumerate(chosen)}
    known = labels.notna()
    # every labelled observation gets a code: its state's, or -1 for "another state"
    codes[known] = [lookup.get(v, -1) for v in labels[known]]
    table = objects.assign(**{_LABEL_CODE: codes.to_numpy(dtype=float)})
    data = validated(table, fovs, [_LABEL_CODE], fov_col, group_cols, observation_unit)
    rows = []
    for group, group_data in groups(data):
        code = group_data.feature_values(_LABEL_CODE)
        positions = fov_positions(group_data, group_data.fovs)
        keep = ~np.isnan(code)
        code, positions = code[keep].astype(int), positions[keep]
        n_fovs = len(group_data.fovs)
        sizes = np.bincount(positions, minlength=n_fovs)
        rng = np.random.default_rng([random_state, group_identity(group)])
        for index, state in enumerate(chosen):
            is_state = code == index
            rows.append(
                {
                    **group,
                    "state": state,
                    **_state_row(
                        is_state,
                        positions,
                        sizes,
                        rng,
                        n_permutations,
                        detection_probability,
                        confidence,
                    ),
                }
            )
    settings = {
        "label_col": label_col,
        "states": chosen,
        "detection_probability": detection_probability,
        "confidence": confidence,
        "n_permutations": n_permutations,
        "random_state": random_state,
        "observation_unit": data.observation_unit,
    }
    return FOVClustering(pd.DataFrame(rows), tuple(data.group_cols), settings)


def _state_row(
    is_state: np.ndarray,
    positions: np.ndarray,
    sizes: np.ndarray,
    rng: np.random.Generator,
    n_permutations: int,
    detection_probability: float,
    confidence: float,
) -> dict[str, Any]:
    labelled, n_fovs, c = len(is_state), len(sizes), int(is_state.sum())
    per_fov = np.bincount(positions[is_state], minlength=n_fovs)
    observed = int((per_fov > 0).sum())
    row: dict[str, Any] = {
        "labelled": labelled,
        "state_objects": c,
        "prevalence": c / labelled if labelled else None,
        "fovs": n_fovs,
        "fovs_with_state": observed,
        "fov_incidence": observed / n_fovs if n_fovs else None,
    }
    empty = [
        "expected_fovs_if_random",
        "clustering_ratio",
        "p_clustered",
        "largest_fov_share",
        "fovs_needed",
        "fovs_needed_upper",
        "fovs_needed_if_random",
    ]
    row.update(dict.fromkeys(empty))
    if c == 0 or labelled == 0:
        return row
    # P(a FOV of n_j labelled objects holds none of the c state objects) under random objects
    miss = stats.hypergeom(labelled, c, sizes).pmf(0)
    expected = float(np.sum(1 - miss))
    null = np.empty(n_permutations)
    for i in range(n_permutations):
        shuffled = rng.permutation(is_state)
        null[i] = np.count_nonzero(np.bincount(positions[shuffled], minlength=n_fovs))
    prevalence = c / labelled
    per_acquired_fov = labelled / n_fovs
    if prevalence < 1:
        objects_needed = math.ceil(math.log(1 - detection_probability) / math.log(1 - prevalence))
        if_random = max(1, math.ceil(objects_needed / per_acquired_fov))
    else:
        if_random = 1
    incidence = observed / n_fovs
    lower = 1 - upper_bound(n_fovs - observed, n_fovs, confidence)
    row.update(
        {
            "expected_fovs_if_random": expected,
            "clustering_ratio": expected / observed,
            "p_clustered": float((1 + np.sum(null <= observed)) / (1 + n_permutations)),
            "largest_fov_share": float(per_fov.max() / c),
            "fovs_needed": first_detection_fovs(incidence, detection_probability)
            if incidence < 1
            else 1,
            "fovs_needed_upper": first_detection_fovs(lower, detection_probability)
            if 0 < lower < 1
            else (1 if lower >= 1 else None),
            "fovs_needed_if_random": if_random,
        }
    )
    return row


@dataclass(frozen=True)
class UnusualObjects:
    """Out-of-distribution labels for every observation, against a declared reference.

    - ``labels``: aligned with the input ``objects`` (same index): ``True``
      for unusual, ``False`` for typical, missing where not scored.
    - ``distances``: the scores behind the labels, aligned the same way.
    - ``groups``: per analysis group, the reference groups used, the
      threshold, the scored and unusual observations, and, when
      ``seed_checks`` were asked for, the agreement (Jaccard index of the
      unusual objects) of refits with other seeds.
    """

    labels: pd.Series
    distances: pd.Series
    groups: pd.DataFrame
    group_cols: tuple[str, ...]
    settings: dict[str, Any]

    def summary(self) -> str:
        """Plain sentences, one per group, with the seed check."""
        words = unit_plural(self.settings["observation_unit"])
        lines = [
            (
                f"Unusual {words}: farther from the reference {dict(self.settings['reference'])} "
                f"than all but {self.settings['quantile']:.1%} of reference {words} are from "
                f"other reference fields ({len(self.settings['feature_cols'])} standardized "
                f"features, {self.settings['n_neighbors']} nearest neighbours)."
            )
        ]
        for row in self.groups.to_dict("records"):
            head = where({c: row[c] for c in self.group_cols})
            if row["status"] != "ok":
                lines.append(f"{head}: {row['status']}.")
                continue
            agreement = row["seed_agreement"]
            check = ""
            if not pd.isna(agreement):
                unstable = agreement < SEED_AGREEMENT_WARNING
                check = f" Refits with other reference samples agree on {agreement:.0%} of them" + (
                    ", so treat the labels as unstable." if unstable else "."
                )
            lines.append(
                f"{head}: {int(row['unusual'])} of {int(row['scored'])} {words} unusual "
                f"({row['unusual_share']:.1%}).{check}"
            )
        return "\n".join(lines)


def _reference_sample(
    group_datas: Sequence[ValidatedInput], per_fov: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Up to ``per_fov`` complete observations of every reference FOV, and their FOV ids."""
    rows, fov_ids, offset = [], [], 0
    for data in group_datas:
        matrix = data.feature_matrix()
        positions = fov_positions(data, data.fovs)
        complete = ~np.isnan(matrix).any(axis=1)
        for j in range(len(data.fovs)):
            members = np.flatnonzero((positions == j) & complete)
            if len(members) > per_fov:
                members = rng.choice(members, size=per_fov, replace=False)
            rows.append(matrix[np.sort(members)])
            fov_ids.append(np.full(len(members), offset + j))
        offset += len(data.fovs)
    return np.concatenate(rows), np.concatenate(fov_ids)


def _score(
    reference: np.ndarray, fov_ids: np.ndarray, tested: np.ndarray, k: int, quantile: float
) -> tuple[np.ndarray, float, dict[str, Any]] | str:
    """Scores of ``tested`` and the threshold from reference FOVs scored against the others."""
    fitted = Standardize().fit_matrix(reference, tuple(str(i) for i in range(reference.shape[1])))
    if isinstance(fitted, str):
        return fitted
    state, _ = fitted
    ref = Standardize().apply_matrix(state, reference)
    tree = cKDTree(ref)
    largest = int(np.bincount(fov_ids).max())
    if len(ref) - largest < k:
        return f"fewer than {k} reference observations outside some reference FOV"
    # leave-one-FOV-out: the k nearest reference objects from other FOVs
    distance, index = tree.query(ref, k=min(len(ref), k + largest))
    other = fov_ids[index] != fov_ids[:, None]
    reference_scores = np.array([d[o][:k].mean() for d, o in zip(distance, other, strict=True)])
    threshold = float(np.quantile(reference_scores, 1 - quantile))
    scores = np.full(len(tested), np.nan)
    complete = ~np.isnan(tested).any(axis=1)
    if complete.any():
        d, _ = tree.query(Standardize().apply_matrix(state, tested[complete]), k=k)
        scores[complete] = np.asarray(d).reshape(complete.sum(), k).mean(axis=1)
    return scores, threshold, state


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
    check_count(n_neighbors, "n_neighbors", 1)
    check_count(per_fov, "per_fov", 1)
    check_count(seed_checks, "seed_checks", 0)
    check_seed(random_state)
    data = validated(objects, fovs, feature_cols, fov_col, group_cols, observation_unit)
    all_groups = groups(data)
    reference_groups = matching_groups(all_groups, reference, "reference")
    by_identity = {group_identity(g): d for g, d in all_groups}
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
    identities = []
    for g_index, (group, group_data) in enumerate(all_groups):
        identities.append(group_identity(group))
        members = [row_of[f] for f in zip(*(group_data.fovs[c].tolist() for c in key))]
        fov_group[members] = g_index
    observation_group = fov_group[observation_fov]
    rows = []
    for g_index, (group, _) in enumerate(all_groups):
        others = [g for g in reference_groups if group_identity(g) != identities[g_index]]
        row: dict[str, Any] = {**group, "reference_groups": [dict(g) for g in others]}
        mine = positions_all[observation_group == g_index]
        if not others:
            rows.append({**row, "status": "no other reference group to score against"})
            continue
        refs = [by_identity[group_identity(g)] for g in others]
        results = []
        for seed in range(random_state, random_state + seed_checks + 1):
            rng = np.random.default_rng([seed, identities[g_index]])
            sample, fov_ids = _reference_sample(refs, per_fov, rng)
            results.append(_score(sample, fov_ids, matrix[mine], n_neighbors, quantile))
        if isinstance(results[0], str):
            rows.append({**row, "status": results[0]})
            continue
        scores, threshold, state = results[0]
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


__all__ = ["FOVClustering", "UnusualObjects", "fov_clustering", "unusual_objects"]
