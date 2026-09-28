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
mean of every object of the declared control groups, as the paper normalizes
"by calculating the average of all the control samples" and as its
supplementary spreadsheet and R code compute it, or each group's own mean when
no control is declared. The rule's number of objects is
``ceil((SD / (reference * target)) ** 2)``, with the SD of every object of
the group; with hundreds of objects the paper's point and upper-95 estimates
agree, so the point estimate is used. Whole FOVs are then drawn at random
until at least that many objects, ``n_draws`` times, **with replacement**, as
from a well with many FOVs like the acquired ones; the SD of their mean over
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
from .._seeds import group_key, group_rng
from .._wording import fields as fields_text
from .._wording import percent, rate, where
from ..schema import ValidatedInput, fov_values, unit_plural
from ..statistics import (
    describe,
    field_sample,
    field_weighted_mean_se,
    jcb_required_objects,
    sd_upper_bound_factor,
    spread_is_zero,
)
from ._common import groups, matching_groups, validated

DESCRIPTORS = ("mean", "median", "SD", "IQR")
RANDOM_OBJECTS = "random objects"
WHOLE_FOVS = "whole FOVs"
_NEGATIVE = (
    "not applicable: the feature takes negative values here or in the control, and an error "
    "relative to the mean depends on where zero lies; the JCB rule applies to non-negative "
    "measurements"
)
_ZERO_REFERENCE = "not applicable: the reference mean is 0, so an error relative to it is undefined"
_CONSTANT = (
    "the feature does not vary, so the mean of any objects is exact and the rule has "
    "nothing to average"
)


@dataclass(frozen=True)
class ObjectRuleCheck:
    """The JCB rule applied to whole FOVs, per analysis group and target.

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

    def summary(self) -> str:
        """Plain sentences, one per group and target."""
        words = unit_plural(self.settings["observation_unit"])
        lines = [
            (
                f"Would the JCB object-count rule hold with whole fields? Feature "
                f"{self.settings['feature']}; reference mean: {self.settings['reference']}."
            )
        ]
        for row in self.table.to_dict("records"):
            head = f"{where({c: row[c] for c in self.group_cols})}, target {rate(row['target'])}"
            if row["status"] != "ok":
                lines.append(f"{head}: {row['status']}.")
                continue
            verdict = "meets" if row["meets_target"] else "misses"
            ratio = row["error_ratio"]
            times = "" if ratio is None or pd.isna(ratio) else f" ({ratio:.1f} times)"
            lines.append(
                f"{head}: the rule asks for {int(row['rule_objects'])} {words}, which took "
                f"{row['fovs_used']:.1f} fields. SD / sqrt(n) predicts "
                f"{percent(row['predicted_error'])}; the real error of that mean is "
                f"{percent(row['real_error'])}{times}, so it {verdict} the target. "
                + _field_rule(row, words)
            )
        return "\n".join(lines)


def _field_rule(row: dict[str, Any], words: str) -> str:
    """The rule with fields as the unit, in fields with observations and acquired fields."""
    rule, upper = int(row["fov_rule_fovs"]), int(row["fov_rule_fovs_upper95"])
    acquired, filled = int(row["fovs"]), int(row["fovs_with_observations"])
    if filled == acquired:
        return (
            f"With fields as the unit the rule asks for {fields_text(rule)} ({upper} from the "
            f"upper-95 SD) of {acquired} acquired."
        )
    return (
        f"With fields as the unit the rule asks for {rule} fields with {words} ({upper} from "
        f"the upper-95 SD); {filled} of the {acquired} acquired fields hold {words}, so that "
        f"is about {fields_text(int(row['fov_rule_acquired']))} to acquire "
        f"({int(row['fov_rule_acquired_upper95'])} from the upper-95 SD)."
    )


def _reference_means(
    all_groups: list[tuple[dict[str, Any], ValidatedInput]],
    feature: str,
    control: Mapping[str, Any] | None,
) -> tuple[dict[str, float | None], str, bool]:
    """The mean that normalizes each group, its label, and whether the control has negatives.

    The mean is each group's own, or the mean of all control objects: as in
    the JCB supplementary spreadsheet and R code, the control mean pools
    every control object, so control groups weigh by their object counts.
    """
    values_by_group = {}
    for group, data in all_groups:
        values = data.feature_values(feature)
        values_by_group[group_key(group)] = values[~np.isnan(values)]
    if control is None:
        means = {k: float(v.mean()) if len(v) else None for k, v in values_by_group.items()}
        return means, "each group's own mean", False
    chosen = matching_groups(all_groups, control, "control")
    pooled = np.concatenate([values_by_group[group_key(g)] for g in chosen])
    if not len(pooled):
        raise ValueError("the control groups have no observations of the feature")
    reference = float(pooled.mean())
    label = f"mean of all objects of {len(chosen)} control group(s) {dict(control)}"
    return dict.fromkeys(values_by_group, reference), label, bool((pooled < 0).any())


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
) -> ObjectRuleCheck:
    """The JCB number of objects, and the averaging error it really gives from whole FOVs.

    ``targets`` are averaging errors as fractions of the reference mean: the
    paper's 5% for independent samples and 2.5% for paired ones. ``control``
    selects the control groups whose mean normalizes every group, for example
    ``{"condition": "MOCK"}``; ``None`` uses each group's own mean. ``fovs``
    lists every acquired FOV; FOVs without observations can be drawn and add
    no objects. Other arguments are those of ``validate_input_tables``.
    """
    targets = [float(t) for t in targets]
    if not targets or any(not 0 < t < 1 for t in targets):
        raise ValueError("targets must be fractions between 0 and 1, such as 0.05")
    n_draws = as_count(n_draws, "n_draws", 2)
    random_state = check_seed(random_state)
    data = validated(objects, fovs, [feature], fov_col, group_cols, observation_unit)
    words = unit_plural(data.observation_unit)
    all_groups = groups(data)
    references, label, negative_control = _reference_means(all_groups, feature, control)
    rows = []
    for group, group_data in all_groups:
        per_fov = fov_values(group_data, group_data.feature_values(feature))
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
        negative = negative_control or any(bool((v < 0).any()) for v in filled)
        for target in targets:
            if negative:
                rows.append({**base, "target": target, "status": _NEGATIVE})
                continue
            if len(filled) < 2:
                status = f"fewer than 2 fields with {words}"
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
            # The JCB rule with the field as the unit: the SD of one field's mean, from the
            # unweighted mean of field means (statistics.field_weighted_mean_se), not the
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
        "n_draws": n_draws,
        "random_state": random_state,
        "observation_unit": data.observation_unit,
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

    def summary(self) -> str:
        """Plain sentences, one per group."""
        words = unit_plural(self.settings["observation_unit"])
        threshold = self.settings["threshold"]
        lines = [
            (
                f"Do descriptors stabilise as fast with whole fields? Feature "
                f"{self.settings['feature']}, spread over {self.settings['n_draws']} samples "
                f"of {threshold} {words}."
            )
        ]
        keys = list(self.group_cols)
        for key, rows in (
            self.at_threshold.groupby(keys, sort=False) if keys else [((), self.at_threshold)]
        ):
            group = where(dict(zip(keys, key, strict=True)))
            parts = []
            for row in rows.to_dict("records"):
                match = row["objects_to_match"]
                reach = (
                    f"matched from {int(match)} {words}"
                    if not pd.isna(match)
                    else f"not matched within {int(row['largest_size_tried'])}"
                )
                ratio = row["ratio"]
                shown = (
                    f"{ratio:.1f}x"
                    if ratio is not None and not pd.isna(ratio)
                    else "no ratio (random samples never varied)"
                )
                parts.append(f"{row['descriptor']} {shown} ({reach})")
            lines.append(f"{group}: whole fields against random {words}: {'; '.join(parts)}.")
        for skipped in self.settings.get("skipped_groups", []):
            lines.append(
                f"{skipped['group']}: not checked: {skipped['observations']} {words}, fewer than "
                f"the threshold of {threshold}."
            )
        return "\n".join(lines)


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
        per_fov = [v for v in fov_values(group_data, group_data.feature_values(feature)) if len(v)]
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
            skipped.append({"group": where(group), "observations": len(pooled)})
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
