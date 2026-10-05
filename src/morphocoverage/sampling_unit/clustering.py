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
- the fields needed to see the state in at least ``min_fov_occurrences``
  fields with ``detection_probability``, from the incidence of the fields
  holding it (``statistics.fovs_to_detect``), with a bound from the low end
  of that incidence's two-sided Clopper-Pearson interval (the convention of
  ``sufficiency``), beside the fields a random-object argument would ask for
  (the same count, from the object prevalence, spread over the observations
  per field).

A field **holds** the state when at least ``min_objects_per_fov`` of its
observations carry it. For labels such as ``unusual_objects``', which flag a
share of ordinary objects by construction, give the novelty rule's ``m`` (the
report does): otherwise the fields holding one stray flagged object set the
field count, not the rare state (a 15-object state in 1 of 30 fields read as
"9 fields" where 89 are needed). With ``reference`` (the control groups), each
row also gives the reference groups' own incidence of fields holding the
state, the floor that ordinary fields reach. With a ``technical_repeat``
column, each row counts the technical repeats (wells, dishes) of the group
and those holding the state, and the summary gives that incidence per
condition and repeat: the field counts assume fields from independent
technical repeats, and a state held by few of them needs fields from more.

Fields without observations count as acquired fields that did not show the
state. Fields whose observations carry no label at all were not looked at:
they are left out of the fields assessed and counted (``fovs_unlabelled``),
so annotating half the fields does not halve the field incidence. This is
the retrospective, labelled counterpart of the rare-state planning of
``strategies.rare_planning``; a state is only as good as its labels.

**``unusual_objects``** (module ``unusual``, re-exported here) labels
out-of-distribution objects against a declared reference population.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from .._caveats import caveat_lines
from .._checks import as_count, check_seed
from .._former_repeat import one_warning, read_former
from .._seeds import group_key, group_rng
from .._style import check_detail
from ..bounds import interval
from ..metadata import CONDITION_COLUMN, REPEAT_COLUMN, TECHNICAL_REPEAT_COLUMN
from ..schema import ValidatedInput, fov_positions
from ..statistics import fovs_to_detect
from ._common import groups, matching_groups, validated
from ._state_text import clustering_parts
from .unusual import UnusualObjects, unusual_objects

_LABEL_CODE = "_morphocoverage_state_code"


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

    def summary(self, detail: str = "short") -> str:
        """The states over the fields in words: short (per state one answer and one table of
        the groups where it was seen) or ``detail="full"`` (every number), caveats once."""
        full = check_detail(detail) == "full"
        lines, cautions = clustering_parts(self, full=full)
        return "\n".join([*lines, *caveat_lines(cautions, indent=0, detail=detail)])


@one_warning
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
    min_objects_per_fov: int = 1,
    min_fov_occurrences: int = 1,
    reference: Mapping[str, Any] | None = None,
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
    field), ``fovs_holding_state`` (fields with at least
    ``min_objects_per_fov`` of its objects) and ``holding_incidence``,
    ``fovs_needed`` and ``fovs_needed_upper`` (the fields to see it in at
    least ``min_fov_occurrences`` fields, from the holding incidence and the
    low end of its two-sided ``confidence`` Clopper-Pearson interval),
    ``fovs_needed_if_random`` (the objects needed to see
    ``min_fov_occurrences * min_objects_per_fov`` of them at the object
    prevalence, spread over the mean observations per field),
    ``reference_fovs`` and ``reference_fovs_holding`` (pooled over the
    groups matching ``reference``, never the group itself; with
    ``reference`` only), and ``technical_repeats`` and
    ``technical_repeats_holding`` (with a ``technical_repeat`` column). The
    clustering ratio and its test count every field with one of the objects.
    """
    (reference,) = read_former(reference=reference)
    if not isinstance(label_col, str) or label_col not in objects.columns:
        raise ValueError(f"label_col {label_col!r} is not a column of objects")
    if not 0 < detection_probability < 1 or not 0 < confidence < 1:
        raise ValueError("detection_probability and confidence must be between 0 and 1")
    n_permutations = as_count(n_permutations, "n_permutations", 1)
    m = as_count(min_objects_per_fov, "min_objects_per_fov", 1)
    k = as_count(min_fov_occurrences, "min_fov_occurrences", 1)
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
    all_groups = groups(data)
    references = [] if reference is None else matching_groups(all_groups, reference, "reference")
    rows = []
    for group, group_data in all_groups:
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
        wells = _technical_repeat_of(group_data)
        for index, state in enumerate(chosen):
            is_state = code == index
            row = _state_row(
                is_state, positions, sizes, n_acquired - unlabelled, rng, n_permutations, m, k
            )
            per_fov = np.bincount(positions[is_state], minlength=n_acquired)
            if wells is not None:
                row["technical_repeats"] = len(np.unique(wells))
                row["technical_repeats_holding"] = len(np.unique(wells[per_fov >= m]))
            rows.append(
                {
                    **group,
                    "state": state,
                    "fovs_acquired": n_acquired,
                    "fovs_unlabelled": unlabelled,
                    **row,
                    **_plan(row, detection_probability, confidence, k, m),
                }
            )
    table = pd.DataFrame(rows)
    if reference is not None:
        _add_reference(table, list(data.group_cols), references)
    settings = {
        "label_col": label_col,
        "states": chosen,
        "detection_probability": detection_probability,
        "confidence": confidence,
        "n_permutations": n_permutations,
        "random_state": random_state,
        "observation_unit": data.observation_unit,
        "observation_unit_declared": data.observation_unit_declared,
        "min_objects_per_fov": m,
        "min_fov_occurrences": k,
        "reference": None if reference is None else dict(reference),
    }
    return FOVClustering(table, tuple(data.group_cols), settings)


def _technical_repeat_of(group_data: ValidatedInput) -> np.ndarray | None:
    """Each field's technical repeat (with its condition and repeat), or None without one."""
    fovs = group_data.fovs
    if TECHNICAL_REPEAT_COLUMN not in fovs.columns:
        return None
    columns = [c for c in (CONDITION_COLUMN, REPEAT_COLUMN, TECHNICAL_REPEAT_COLUMN) if c in fovs]
    return fovs.groupby(columns, sort=False, dropna=False).ngroup().to_numpy()


def _add_reference(table: pd.DataFrame, keys: list[str], references: list[dict]) -> None:
    """The reference groups' fields and fields holding each state, pooled, beside each row.

    A group is never its own reference, as in ``unusual_objects``.
    """
    reference_keys = {group_key(g) for g in references}
    own = [group_key({c: row[c] for c in keys}) for row in table.to_dict("records")]
    is_reference = np.array([key in reference_keys for key in own], dtype=bool)
    fields = table["fovs"].fillna(0).to_numpy(dtype=float)
    holding = table["fovs_holding_state"].fillna(0).to_numpy(dtype=float)
    totals, held = np.zeros(len(table)), np.zeros(len(table))
    for i in range(len(table)):
        others = is_reference & (table["state"] == table["state"].iloc[i]).to_numpy()
        others[i] = False
        totals[i], held[i] = fields[others].sum(), holding[others].sum()
    table["reference_fovs"] = totals.astype(int)
    table["reference_fovs_holding"] = held.astype(int)


def _state_row(
    is_state: np.ndarray,
    positions: np.ndarray,
    sizes: np.ndarray,
    n_fovs: int,
    rng: np.random.Generator,
    n_permutations: int,
    m: int,
    k: int,
) -> dict[str, Any]:
    """One state's counts and clustering; ``n_fovs`` fields were assessed, ``sizes`` their
    labelled counts. A field holds the state with at least ``m`` of its objects."""
    labelled, c = len(is_state), int(is_state.sum())
    per_fov = np.bincount(positions[is_state], minlength=len(sizes))
    observed = int((per_fov > 0).sum())
    holding = int((per_fov >= m).sum())
    row: dict[str, Any] = {
        "labelled": labelled,
        "state_objects": c,
        "prevalence": c / labelled if labelled else None,
        "fovs": n_fovs,
        "fovs_with_state": observed,
        "fov_incidence": observed / n_fovs if n_fovs else None,
        "fovs_holding_state": holding,
        "holding_incidence": holding / n_fovs if n_fovs else None,
    }
    empty = ["expected_fovs_if_random", "clustering_ratio", "p_clustered", "largest_fov_share"]
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
    row.update(
        {
            "expected_fovs_if_random": expected,
            "clustering_ratio": expected / observed,
            "p_clustered": float((1 + np.sum(null <= observed)) / (1 + n_permutations)),
            "largest_fov_share": float(per_fov.max() / c),
        }
    )
    return row


def _plan(
    row: Mapping[str, Any], detection_probability: float, confidence: float, k: int, m: int
) -> dict[str, Any]:
    """The fields to see the state in at least ``k`` fields, from the holding incidence.

    ``fovs_needed`` from the incidence of the fields holding it,
    ``fovs_needed_upper`` from the low end of its two-sided Clopper-Pearson
    interval, and ``fovs_needed_if_random`` from the objects needed to see
    ``k * m`` of them at the object prevalence, spread over the observations
    per field. ``None`` where no field holds it.
    """
    plan: dict[str, Any] = dict.fromkeys(
        ("fovs_needed", "fovs_needed_upper", "fovs_needed_if_random")
    )
    n_fovs, holding, labelled = row["fovs"], row["fovs_holding_state"], row["labelled"]
    if not row["state_objects"] or not labelled or not holding:
        return plan
    prevalence = row["prevalence"]
    if prevalence < 1:
        # The objects needed, spread over labelled / n_fovs per field (integer ceiling,
        # so no rounding adds a field).
        objects_needed = fovs_to_detect(prevalence, detection_probability, k * m)
        plan["fovs_needed_if_random"] = max(1, -(-objects_needed * n_fovs // labelled))
    else:
        plan["fovs_needed_if_random"] = k
    incidence = holding / n_fovs
    lower = interval(holding, n_fovs, confidence)[0]
    plan["fovs_needed"] = (
        fovs_to_detect(incidence, detection_probability, k) if incidence < 1 else k
    )
    if 0 < lower < 1:
        plan["fovs_needed_upper"] = fovs_to_detect(lower, detection_probability, k)
    elif lower >= 1:
        plan["fovs_needed_upper"] = k
    return plan


__all__ = ["FOVClustering", "UnusualObjects", "fov_clustering", "unusual_objects"]
