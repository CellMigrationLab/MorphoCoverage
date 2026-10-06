"""Would an object-count rule hold if its objects come as whole FOVs? (Foundation, P5)

Both published object-count rules this package reproduces assume that the
objects they count are a random sample of the well. Zweifach (2024, JCB)
"further assume[s] that samples are spatially uniform"; Marcotti et al.
(2026, JCS) assume "this small number of cells is representative of the
broader population" and subsample random cells from every imaged field. A
microscope acquires whole FOVs. Each check here runs the paper's own
procedure twice: with objects drawn at random, as the paper does, and with
the same number of objects obtained by acquiring whole FOVs in random order,
as a microscope does. Nothing else changes. Both are retrospective: they use
every acquired FOV of the group.

**``object_rule_check``** (JCB). Values are divided by a reference mean: the
mean of every object of the declared control groups of the same repeat (the
``biological_repeat`` column, when present), as the paper normalizes "by calculating the
average of all the control samples" within an experiment and as
``normalize_to_control`` does, or each group's own mean when no control is
declared. A control mean pooled over repeats would judge a repeat against
another day's level: with a 2x batch shift the rule asked for 12-14 objects
in one repeat and 43-56 in the other where each repeat's own control gives
24-32 in both. A repeat without control objects is reported as not
applicable, and the summary warns when the repeats' control means differ by
more than 1.5x (a batch shift or a change of units). The rule's number of objects is
``ceil((SD / (reference * target)) ** 2)``, with the SD of every object of
the group; with hundreds of objects the paper's point and upper-95 estimates
agree, so the point estimate is used. Whole FOVs are then drawn at random
until at least that many objects, ``n_draws`` times, **with replacement**, as
from a well with many FOVs like the acquired ones (a group of fewer than ``MIN_FIELDS``
fields with objects is not assessable: its draws would mostly repeat its own few fields;
audit 10, owner, 2026-10-06); the SD of their mean over
the draws, in units of the reference mean, is the real averaging error. With
few acquired FOVs this is still optimistic, by about ``(F - 1) / F`` in
variance. The rule's own prediction for the objects obtained is
``SD / sqrt(n)``; random objects meet the target by construction. The same
formula with the FOV as the unit (the SD of FOV means) gives the number of
FOVs the rule would ask for, as a point estimate and from the paper's
upper-95 SD bound, which it advises when the pilot is small. The rule's
error is relative to the mean, which depends on where zero lies unless the
feature is a non-negative measurement: a group whose values, or whose control
values, include negative ones is reported as not applicable, with the reason
(as ``ScalarPrecisionStrategy`` refuses them).

**``descriptor_spread_check``** (JCS Fig. 2A). For each sample size ``n``:
``n_draws`` samples of ``n`` objects, and the max - min over samples of the
mean, median, SD and IQR. FOV samples take distinct FOVs in random order
until ``n`` objects, with a random part of the last FOV, which is what a
microscope stopping at ``n`` objects has. At the declared threshold (the
paper's reading of where descriptors stabilise, dataset-specific: about
250-300 cells in its idr0139 case study, about 200 in idr0028) it reports
the spread of FOV samples relative to random-object samples, and the
smallest ``n`` from which FOV samples stay within the random-object spread.

Neither check says a published rule is wrong: each answers its question under
its stated assumption, and these checks measure whether the assumption holds
for the acquired FOVs (AGENTS.md section 10).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .._checks import as_count, check_seed
from .._control import control_text
from .._former_repeat import one_warning, read_former
from .._seeds import group_key, group_rng
from .._style import check_detail, group_label
from .._wording import column_word, where
from ..metadata import CONDITION_COLUMN, REPEAT_COLUMN
from ..schema import ValidatedInput, fov_values, observation_words
from ..statistics import (
    describe,
    field_sample,
    field_weighted_mean_se,
    jcb_required_objects,
    sd_upper_bound_factor,
    spread_is_zero,
)
from ._common import canonical_fields, groups, matching_groups, validated
from ._experiments import repeat_experiments
from ._unit_text import (
    descriptor_spread_full,
    descriptor_spread_parts,
    join,
    object_rule_full,
    object_rule_parts,
)
from .contrast_sampling import MIN_FIELDS

DESCRIPTORS = ("mean", "median", "SD", "IQR")
RANDOM_OBJECTS = "random objects"
WHOLE_FOVS = "whole FOVs"
_NEGATIVE = (
    "not applicable: the feature takes negative values here or in the control, and an error "
    "relative to the mean depends on where zero lies; the mean-precision rule applies to "
    "non-negative measurements"
)
_ZERO_REFERENCE = "not applicable: the reference mean is 0, so an error relative to it is undefined"
# Repeat control means differing by more than this factor are flagged (D3).
REFERENCE_SHIFT = 1.5
_NO_REFERENCE = (
    "not applicable: no control object in this group's repeat, so there is no reference mean "
    "of the same repeat"
)
_CONSTANT = (
    "the feature does not vary, so the mean of any objects is exact and the rule has "
    "nothing to average"
)


@dataclass(frozen=True)
class ObjectRuleCheck:
    """The mean-precision rule applied to whole FOVs, per analysis group and target.

    ``table`` has, per group and target: the rule's number of objects
    (``rule_objects``), the objects and FOVs a draw really took
    (``objects_obtained``, ``fovs_used``), the averaging error the rule
    predicts for those objects (``predicted_error``) and the ``real_error``
    of their mean, both as fractions of the reference mean, their
    ``error_ratio``, whether the real error ``meets_target``, and the FOVs the
    rule asks for with the FOV as the unit (``fov_rule_fovs`` and
    ``fov_rule_fovs_upper95``). Those count fields with observations, since
    the SD of field means comes from them; ``fov_rule_acquired`` and
    ``fov_rule_acquired_upper95`` scale them by the acquired fields per field
    with observations (``fovs`` over ``fovs_with_observations``), the fields
    a microscope would acquire to get them.
    """

    table: pd.DataFrame
    group_cols: tuple[str, ...]
    settings: dict[str, Any]

    def summary(self, detail: str = "short") -> str:
        """The rule with whole fields in words: short (one row per target, ranges over the
        groups) or ``detail="full"`` (every number of each group and target)."""
        if check_detail(detail) == "full":
            return object_rule_full(self)
        return join(*object_rule_parts(self))


def _reference_means(
    all_groups: list[tuple[dict[str, Any], ValidatedInput]],
    feature: str,
    control: Mapping[str, Any] | None,
    repeat_col: str | None,
) -> tuple[dict[str, float | None], str, dict[str, bool], str | None]:
    """The mean that normalizes each group, its label, per group whether its control has
    negatives, and a caution when the repeats' control means differ by more than 1.5x.

    The mean is each group's own, or the mean of the control objects of the
    repeats the group spans (its own repeat, unless ``group_cols`` pool the
    repeats): as in the JCB supplementary spreadsheet and R code within one
    experiment, the control mean pools every control object of the repeat, so
    control groups weigh by their object counts. ``None`` for a group whose
    repeats hold no control object.
    """
    values_by_group = {}
    for group, data in all_groups:
        values = data.feature_values(feature)
        values_by_group[group_key(group)] = values[~np.isnan(values)]
    if control is None:
        means = {k: float(v.mean()) if len(v) else None for k, v in values_by_group.items()}
        return means, "each group's own mean", dict.fromkeys(means, False), None
    chosen = {group_key(g) for g in matching_groups(all_groups, control, "control")}

    def repeats_of(data: ValidatedInput) -> list[Any]:
        return data.fovs[repeat_col].drop_duplicates().tolist() if repeat_col else [None]

    by_repeat: dict[str, list[np.ndarray]] = {}
    for group, data in all_groups:
        if group_key(group) in chosen:
            for repeat in repeats_of(data):
                by_repeat.setdefault(repr(repeat), []).append(values_by_group[group_key(group)])
    pooled = {r: np.concatenate(v) for r, v in by_repeat.items()}
    if not any(len(v) for v in pooled.values()):
        raise ValueError("the control groups have no observations of the feature")
    means, negative = {}, {}
    for group, data in all_groups:
        parts = [pooled[repr(r)] for r in repeats_of(data) if repr(r) in pooled]
        values = np.concatenate(parts) if parts else np.array([])
        means[group_key(group)] = float(values.mean()) if len(values) else None
        negative[group_key(group)] = bool((values < 0).any())
    within = f" of the same {column_word(repeat_col)}" if repeat_col else ""
    label = f"mean of all objects of the control groups {control_text(control)}{within}"
    names = {repr(r): str(r) for _, d in all_groups for r in repeats_of(d)}
    # Control means are compared within a sub-experiment only: two sets of a screen, each
    # with its own control and repeats, are not one control drifting (owner, 2026-10-02).
    block = (
        repeat_experiments([g for g, _ in all_groups], chosen, CONDITION_COLUMN, repeat_col)
        if repeat_col
        else {}
    )
    of_name = {repr(r): block.get(r) for _, d in all_groups for r in repeats_of(d)}
    # A repeat holding only the control belongs to no sub-experiment: compared with all.
    blocks: dict[int | None, dict[str, np.ndarray]] = {b: {} for b in set(block.values())}
    blocks = blocks or {None: {}}
    for r, values in pooled.items():
        for b in blocks if of_name.get(r) is None else [of_name[r]]:
            blocks[b][names[r]] = values
    shifts = [shift for shift in map(_shift, blocks.values()) if shift is not None]
    caution = max(shifts)[1] if shifts else None
    return means, label, negative, caution


def _shift(pooled: dict[str, np.ndarray]) -> tuple[float, str] | None:
    """The ratio and a caution when the repeats' control means differ by more than
    ``REFERENCE_SHIFT``; ``None`` below it."""
    levels = {r: float(v.mean()) for r, v in pooled.items() if len(v)}
    if len(levels) < 2 or min(levels.values()) <= 0:
        return None
    low, high = min(levels, key=levels.get), max(levels, key=levels.get)
    ratio = levels[high] / levels[low]
    if ratio <= REFERENCE_SHIFT:
        return None
    return ratio, (
        f"the control means of the repeats differ by {ratio:.1f}x (repeat {low}: "
        f"{levels[low]:.3g}, repeat {high}: {levels[high]:.3g}); each repeat is judged "
        "against its own control, but a shift that large may be a batch effect or a change "
        "of units (a feature exported in other units in one repeat)"
    )


def _canonical_fov_values(data: ValidatedInput, feature: str) -> list[np.ndarray]:
    """The feature's values per field, fields by FOV key and values sorted, so no draw
    depends on the tables' row order (``canonical_fields``)."""
    values = fov_values(data, data.feature_values(feature))
    return [np.sort(values[i]) for i in canonical_fields(data)]


def _fovs_until(
    sizes: np.ndarray, sums: np.ndarray, n: int, draws: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean, objects and FOVs of whole FOVs drawn with replacement until ``n`` objects."""
    means, obtained, used = np.empty(draws), np.empty(draws), np.empty(draws)
    # enough FOVs per batch for most draws to reach n in one go
    batch = max(4, 2 * math.ceil(n / max(sizes.mean(), 1e-12)) + 4)
    for d in range(draws):
        count, total, k = 0, 0.0, 0
        while count < n:
            picks = rng.integers(0, len(sizes), size=batch)
            stop = np.searchsorted(count + np.cumsum(sizes[picks]), n)  # first reaching n
            chosen = picks[: stop + 1] if stop < batch else picks
            count += int(sizes[chosen].sum())
            total += float(sums[chosen].sum())
            k += len(chosen)
        means[d], obtained[d], used[d] = total / count, count, k
    return means, obtained, used


@one_warning
def object_rule_check(
    objects: pd.DataFrame,
    fovs: pd.DataFrame,
    *,
    feature: str,
    targets: Sequence[float] = (0.05, 0.025),
    control: Mapping[str, Any] | None = None,
    n_draws: int = 4000,
    random_state: int = 0,
    fov_col: str | None = None,
    group_cols: Sequence[str] | None = None,
    observation_unit: str | None = None,
    repeat_col: str | None = REPEAT_COLUMN,
    min_fields: int = MIN_FIELDS,
) -> ObjectRuleCheck:
    """The JCB number of objects, and the averaging error it really gives from whole FOVs.

    ``targets`` are averaging errors as fractions of the reference mean: the
    paper's 5% for independent samples and 2.5% for paired ones. ``control``
    selects the control groups whose mean normalizes every group of the same
    repeat (``repeat_col``; the default may be absent, and ``None`` pools the
    control objects of every repeat), for example ``{"condition": "MOCK"}``;
    ``None`` uses each group's own mean. ``fovs``
    lists every acquired FOV; FOVs without observations can be drawn and add
    no objects. A group of fewer than ``min_fields`` fields with objects is not assessable
    (audit 10, owner, 2026-10-06; ``min_fields=2``, the rule before, reproduces
    ``docs/preliminary_findings.md`` entry 9 on idr0139's wells of 4 fields). Other arguments
    are those of ``validate_input_tables``.
    """
    control, repeat_col = read_former(control=control, repeat_col=repeat_col)
    targets = [float(t) for t in targets]
    if not targets or any(not 0 < t < 1 for t in targets):
        raise ValueError("targets must be fractions between 0 and 1, such as 0.05")
    n_draws = as_count(n_draws, "n_draws", 2)
    min_fields = as_count(min_fields, "min_fields", 2)
    random_state = check_seed(random_state)
    data = validated(objects, fovs, [feature], fov_col, group_cols, observation_unit)
    words = observation_words(data)[1]
    all_groups = groups(data)
    if repeat_col is not None and repeat_col not in data.fovs.columns:
        if repeat_col != REPEAT_COLUMN:
            raise ValueError(
                f"repeat_col {repeat_col!r} is not a column of the FOV table; name the "
                "biological-repeat column, or pass repeat_col=None to pool every repeat's control"
            )
        repeat_col = None  # the default is absent: one implicit repeat
    references, label, negative_control, shift = _reference_means(
        all_groups, feature, control, repeat_col
    )
    rows = []
    for group, group_data in all_groups:
        per_fov = _canonical_fov_values(group_data, feature)
        sizes = np.array([len(v) for v in per_fov], dtype=float)
        sums = np.array([v.sum() for v in per_fov])
        filled = [v for v in per_fov if len(v)]
        reference = references[group_key(group)]
        base = {
            **group,
            "fovs": len(per_fov),
            "fovs_with_observations": len(filled),
            "observations": int(sizes.sum()),
        }
        negative = negative_control[group_key(group)] or any(bool((v < 0).any()) for v in filled)
        for target in targets:
            if reference is None and control is not None:
                rows.append({**base, "target": target, "status": _NO_REFERENCE})
                continue
            if negative:
                rows.append({**base, "target": target, "status": _NEGATIVE})
                continue
            if len(filled) < min_fields:
                # Drawn with replacement, a few fields repeat themselves: below MIN_FIELDS the
                # check reads its own resampling (audit 10, owner, 2026-10-06; before, 2).
                status = f"fewer than {min_fields} fields with {words}"
                rows.append({**base, "target": target, "status": status})
                continue
            if reference == 0:
                rows.append({**base, "target": target, "status": _ZERO_REFERENCE})
                continue
            relative = np.concatenate(filled) / reference
            sd = float(np.std(relative, ddof=1))
            if spread_is_zero(sd, relative):
                rows.append({**base, "target": target, "status": _CONSTANT})
                continue
            n = max(1, jcb_required_objects(sd, target))
            rng = group_rng(random_state, group, int(target * 1e6))
            means, obtained, used = _fovs_until(sizes, sums / reference, n, n_draws, rng)
            real = float(means.std(ddof=1))
            predicted = float(np.mean(sd / np.sqrt(obtained)))
            # The mean-precision rule with the field as the unit: the SD of one field's mean, from
            # the unweighted mean of field means (statistics.field_weighted_mean_se), not the
            # object-weighted ratio estimator of ScalarPrecisionStrategy(resampling_unit="fov").
            _, fov_se, n_filled = field_weighted_mean_se(np.array([v.mean() for v in filled]))
            fov_sd = fov_se * math.sqrt(n_filled) / reference
            upper = fov_sd * sd_upper_bound_factor(n_filled - 1)
            fov_rule = max(1, jcb_required_objects(fov_sd, target))
            fov_rule_upper = max(1, jcb_required_objects(upper, target))
            # Fields with observations, scaled to the acquired fields that hold that many.
            per_filled = len(per_fov) / n_filled
            rows.append(
                {
                    **base,
                    "target": target,
                    "status": "ok",
                    "object_sd": sd,
                    "rule_objects": n,
                    "rule_exceeds_acquired": bool(n > sizes.sum()),
                    "objects_obtained": float(obtained.mean()),
                    "fovs_used": float(used.mean()),
                    "predicted_error": predicted,
                    "real_error": real,
                    "error_ratio": real / predicted if predicted > 0 else None,
                    "meets_target": bool(real <= target),
                    "fov_rule_fovs": fov_rule,
                    "fov_rule_fovs_upper95": fov_rule_upper,
                    "fov_rule_acquired": math.ceil(round(fov_rule * per_filled, 9)),
                    "fov_rule_acquired_upper95": math.ceil(round(fov_rule_upper * per_filled, 9)),
                }
            )
    settings = {
        "feature": feature,
        "targets": targets,
        "control": None if control is None else dict(control),
        "reference": label,
        "repeat_col": repeat_col,
        "reference_shift": shift,
        "n_draws": n_draws,
        "random_state": random_state,
        "observation_unit": data.observation_unit,
        "observation_unit_declared": data.observation_unit_declared,
    }
    return ObjectRuleCheck(pd.DataFrame(rows), tuple(data.group_cols), settings)


@dataclass(frozen=True)
class DescriptorSpreadCheck:
    """JCS Fig. 2A with random objects and with whole FOVs, per analysis group.

    - ``curves``: per group, sample size ``n`` and ``sampling``
      (``"random objects"`` or ``"whole FOVs"``), the max - min of each
      descriptor over the samples.
    - ``at_threshold``: per group and descriptor, both spreads at the
      threshold, their ``ratio`` (whole FOVs over random objects), and
      ``objects_to_match``, the smallest ``n`` from which FOV samples stay
      within the random-object spread at the threshold (missing when they
      never do within the sizes tried).
    """

    curves: pd.DataFrame
    at_threshold: pd.DataFrame
    group_cols: tuple[str, ...]
    settings: dict[str, Any]

    def summary(self, detail: str = "short") -> str:
        """The descriptor spread in words: short (one row per descriptor over the groups) or
        ``detail="full"`` (each group's ratio and match per descriptor)."""
        if check_detail(detail) == "full":
            return descriptor_spread_full(self)
        return join(*descriptor_spread_parts(self))


def descriptor_spread_check(
    objects: pd.DataFrame,
    fovs: pd.DataFrame,
    *,
    feature: str,
    threshold: int,
    sizes: Sequence[int] | None = None,
    n_draws: int = 100,
    random_state: int = 0,
    fov_col: str | None = None,
    group_cols: Sequence[str] | None = None,
    observation_unit: str | None = None,
) -> DescriptorSpreadCheck:
    """JCS Fig. 2A for every group, with random objects and with whole FOVs.

    ``threshold`` is the sample size at which the descriptors are read as
    stable (the paper's reading is dataset-specific, so it has no default).
    ``sizes`` default to 10, 20, ... up to twice the threshold; sizes beyond a
    group's observations are skipped. ``n_draws`` is 100 in the paper.
    """
    threshold = as_count(threshold, "threshold", 2)
    n_draws = as_count(n_draws, "n_draws", 2)
    random_state = check_seed(random_state)
    sizes = list(range(10, 2 * threshold + 1, 10)) if sizes is None else list(sizes)
    sizes = sorted({*(as_count(n, "every size", 2) for n in sizes), threshold})
    data = validated(objects, fovs, [feature], fov_col, group_cols, observation_unit)
    keys = list(data.group_cols)
    rows, threshold_rows, skipped = [], [], []
    for group, group_data in groups(data):
        per_fov = [v for v in _canonical_fov_values(group_data, feature) if len(v)]
        pooled = np.concatenate(per_fov) if per_fov else np.array([])
        rng = group_rng(random_state, group)
        spreads: dict[tuple[str, int], np.ndarray] = {}
        for n in [s for s in sizes if s <= len(pooled)]:
            for how in (RANDOM_OBJECTS, WHOLE_FOVS):
                if how == RANDOM_OBJECTS:
                    samples = [rng.choice(pooled, size=n, replace=False) for _ in range(n_draws)]
                else:
                    samples = [field_sample(per_fov, n, rng) for _ in range(n_draws)]
                values = np.array([list(describe(s).values()) for s in samples])
                spread = values.max(axis=0) - values.min(axis=0)
                spreads[(how, n)] = spread
                rows.append(
                    {**group, "n": n, "sampling": how, **dict(zip(DESCRIPTORS, spread.tolist()))}
                )
        if (RANDOM_OBJECTS, threshold) not in spreads:
            skipped.append(
                {"group": where(group), "label": group_label(group), "observations": len(pooled)}
            )
            continue
        tried = sorted(n for how, n in spreads if how == WHOLE_FOVS)
        for index, name in enumerate(DESCRIPTORS):
            reference = spreads[(RANDOM_OBJECTS, threshold)][index]
            above = [n for n in tried if spreads[(WHOLE_FOVS, n)][index] > reference]
            later = [n for n in tried if not above or n > max(above)]
            fov_spread = spreads[(WHOLE_FOVS, threshold)][index]
            threshold_rows.append(
                {
                    **group,
                    "descriptor": name,
                    "threshold": threshold,
                    "random_objects_spread": float(reference),
                    "whole_fovs_spread": float(fov_spread),
                    "ratio": float(fov_spread / reference) if reference > 0 else None,
                    "objects_to_match": later[0] if later else None,
                    "largest_size_tried": tried[-1],
                }
            )
    settings = {
        "feature": feature,
        "threshold": threshold,
        "sizes": sizes,
        "n_draws": n_draws,
        "random_state": random_state,
        "observation_unit": data.observation_unit,
        "observation_unit_declared": data.observation_unit_declared,
        "skipped_groups": skipped,
    }
    return DescriptorSpreadCheck(
        pd.DataFrame(rows), pd.DataFrame(threshold_rows), tuple(keys), settings
    )


__all__ = [
    "DESCRIPTORS",
    "RANDOM_OBJECTS",
    "WHOLE_FOVS",
    "DescriptorSpreadCheck",
    "ObjectRuleCheck",
    "descriptor_spread_check",
    "object_rule_check",
]
