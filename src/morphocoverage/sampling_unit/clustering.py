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
  from the observed field incidence (``statistics.first_detection_fovs``),
  with a bound from the low end of that incidence's two-sided Clopper-Pearson
  interval (the convention of ``sufficiency``), beside
  the fields a random-object argument would ask for (the same count, from
  the object prevalence, spread over the observations per field).

Fields without observations count as acquired fields that did not show the
state. Fields whose observations carry no label at all were not looked at:
they are left out of the fields assessed and counted (``fovs_unlabelled``),
so annotating half the fields does not halve the field incidence. This is
the retrospective, labelled counterpart of the rare-state planning of
``strategies.rare_planning``; a state is only as good as its labels.

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
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from .._checks import as_count, check_seed
from .._seeds import group_key, group_rng
from .._wording import count, percent, where
from .._wording import fields as fields_text
from ..bounds import interval
from ..schema import fov_positions, unit_plural
from ..statistics import first_detection_fovs
from ..transforms import Standardize
from ._common import groups, matching_groups, validated
from ._outliers import LEAVE_ONE_FIELD_OUT, reference_sample, score

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
        unit = self.settings["observation_unit"]
        words = unit_plural(unit)
        chance = self.settings["detection_probability"]
        lines = [f"Are the states spread over the fields? Labels {self.settings['label_col']}."]
        for row in self.table.to_dict("records"):
            head = f"{where({c: row[c] for c in self.group_cols})}, {self.state_name(row['state'])}"
            if not row["labelled"]:
                lines.append(
                    f"{head}: none of the {words} of its "
                    f"{fields_text(int(row['fovs_acquired']))} is labelled, so it was not "
                    "looked for."
                )
                continue
            unlabelled = int(row["fovs_unlabelled"])
            left_out = (
                f" ({fields_text(unlabelled)} whose {words} carry no label left out)"
                if unlabelled
                else ""
            )
            if not row["state_objects"]:
                lines.append(
                    f"{head}: not seen among {int(row['labelled'])} labelled {words} in "
                    f"{fields_text(int(row['fovs']))}{left_out}."
                )
                continue
            upper = row["fovs_needed_upper"]
            bound = "" if pd.isna(upper) else f" (up to {int(upper)})"
            seen = count(int(row["state_objects"]), unit)
            needed = fields_text(int(row["fovs_needed"]))
            lines.append(
                f"{head}: {seen} ({percent(row['prevalence'])}) in "
                f"{int(row['fovs_with_state'])} of {fields_text(int(row['fovs']))}{left_out}; "
                f"random {words} would fill {row['expected_fovs_if_random']:.1f} (clustering ratio "
                f"{row['clustering_ratio']:.1f}, p = {row['p_clustered']:.2g}). To see it "
                f"with {percent(chance)} chance: {needed}{bound} from "
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
    (the fields assessed: every acquired field except those whose
    observations carry no label), ``fovs_acquired``, ``fovs_unlabelled``,
    ``fovs_with_state``, ``fov_incidence``,
    ``expected_fovs_if_random``, ``clustering_ratio``, ``p_clustered``,
    ``largest_fov_share`` (the share of the state's objects in its fullest
    field), ``fovs_needed`` and ``fovs_needed_upper`` (from the observed field
    incidence and the low end of its two-sided ``confidence`` Clopper-Pearson
    interval), and
    ``fovs_needed_if_random`` (from the object prevalence and the mean
    observations per acquired field).
    """
    if not isinstance(label_col, str) or label_col not in objects.columns:
        raise ValueError(f"label_col {label_col!r} is not a column of objects")
    if not 0 < detection_probability < 1 or not 0 < confidence < 1:
        raise ValueError("detection_probability and confidence must be between 0 and 1")
    n_permutations = as_count(n_permutations, "n_permutations", 1)
    random_state = check_seed(random_state)
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
        n_acquired = len(group_data.fovs)
        observed_fields = np.bincount(positions, minlength=n_acquired) > 0
        keep = ~np.isnan(code)
        code, positions = code[keep].astype(int), positions[keep]
        sizes = np.bincount(positions, minlength=n_acquired)
        # A field whose observations all lack a label was not looked at; a field
        # without observations was, and did not show the state.
        unlabelled = int(np.sum(observed_fields & (sizes == 0)))
        rng = group_rng(random_state, group)
        for index, state in enumerate(chosen):
            is_state = code == index
            rows.append(
                {
                    **group,
                    "state": state,
                    "fovs_acquired": n_acquired,
                    "fovs_unlabelled": unlabelled,
                    **_state_row(
                        is_state,
                        positions,
                        sizes,
                        n_acquired - unlabelled,
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
    n_fovs: int,
    rng: np.random.Generator,
    n_permutations: int,
    detection_probability: float,
    confidence: float,
) -> dict[str, Any]:
    """One state's row; ``n_fovs`` fields were assessed, ``sizes`` their labelled counts."""
    labelled, c = len(is_state), int(is_state.sum())
    per_fov = np.bincount(positions[is_state], minlength=len(sizes))
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
        null[i] = np.count_nonzero(np.bincount(positions[shuffled], minlength=len(sizes)))
    prevalence = c / labelled
    if prevalence < 1:
        # The objects needed, spread over labelled / n_fovs per field (integer ceiling,
        # so no rounding adds a field).
        objects_needed = first_detection_fovs(prevalence, detection_probability)
        if_random = max(1, -(-objects_needed * n_fovs // labelled))
    else:
        if_random = 1
    incidence = observed / n_fovs
    lower = interval(observed, n_fovs, confidence)[0]
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

    def summary(self) -> str:
        """Plain sentences, one per group, with the seed check."""
        words = unit_plural(self.settings["observation_unit"])
        lines = [
            (
                f"Unusual {words}: farther from the reference {dict(self.settings['reference'])} "
                f"than all but {percent(self.settings['quantile'])} of reference {words} are "
                f"from the other reference groups ({len(self.settings['feature_cols'])} "
                f"standardized features, {self.settings['n_neighbors']} nearest neighbours)."
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
                check = (
                    f" Refits with other reference samples agree on {percent(agreement)} of them"
                    + (", so treat the labels as unstable." if unstable else ".")
                )
            calibration = ""
            if row["calibration"] == LEAVE_ONE_FIELD_OUT:
                calibration = (
                    " With a single reference group the threshold comes from its fields scored "
                    "against its other fields, so a shift between that group and this one "
                    "raises the share."
                )
            lines.append(
                f"{head}: {int(row['unusual'])} of {int(row['scored'])} {words} unusual "
                f"({percent(row['unusual_share'])}).{check}{calibration}"
            )
        return "\n".join(lines)


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
                score(sample, fov_ids, group_ids, matrix[mine], n_neighbors, quantile, rng)
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


__all__ = ["FOVClustering", "UnusualObjects", "fov_clustering", "unusual_objects"]
