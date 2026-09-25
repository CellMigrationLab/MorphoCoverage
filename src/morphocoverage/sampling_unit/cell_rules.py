"""Would a cell-number rule hold if the cells come as whole fields? (``docs/questions.md``, F3)

Both published cell-number rules this package reproduces assume the cells
they count are a random sample of the well. Zweifach (2024, JCB) "further
assume[s] that samples are spatially uniform"; Marcotti et al. (2026, JCS)
assume "this small number of cells is representative of the broader
population" and subsample random cells from every imaged field. A microscope
acquires whole fields. Each check here runs the paper's own procedure twice:
with cells drawn at random, as the paper does, and with the same number of
cells obtained by acquiring whole fields in random order, as a microscope
does. Nothing else changes. Both are retrospective: they use every acquired
field of the group.

**``cell_rule_check``** (JCB). Values are divided by a reference mean: the
mean of the declared control groups, as the paper normalizes "by calculating
the average of all the control samples", or each group's own mean when no
control is declared. The rule's number of cells is
``ceil((SD / (reference * target)) ** 2)``, with the SD of every cell of the
group; with hundreds of cells the paper's point and upper-95 estimates
agree, so the point estimate is used. Whole fields are then drawn at random
until at least that many cells, ``n_draws`` times, **with replacement**, as
from a well with many fields like the acquired ones; the SD of their mean over
the draws, in units of the reference mean, is the real averaging error. With
few acquired fields this is still optimistic, by about ``(F - 1) / F`` in
variance. The rule's own prediction for the cells obtained is ``SD / sqrt(n)``;
random cells meet the target by construction. The same formula with the
field as the unit (the SD of field means) gives the number of fields the rule
would ask for, as a point estimate and from the paper's upper-95 SD bound,
which it advises when the pilot is small.

**``descriptor_spread_check``** (JCS Fig. 2A). For each sample size ``n``:
``n_draws`` samples of ``n`` cells, and the max - min over samples of the
mean, median, SD and IQR. Field samples take distinct fields in random order
until ``n`` cells, with a random part of the last field, which is what a
microscope stopping at ``n`` cells has. At the declared threshold (the
paper's reading of where descriptors stabilise, dataset-specific: about
250-300 cells in its idr0139 case study, about 200 in idr0028) it reports
the spread of field samples relative to random-cell samples, and the
smallest ``n`` from which field samples stay within the random-cell spread.

Neither check says a published rule is wrong: each answers its question under
its stated assumption, and these checks measure whether the assumption holds
for the acquired fields (AGENTS.md section 10).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..retrospective._text import fields as fields_text
from ..retrospective._text import group_identity, unit_plural, where
from ..strategies.scalar_precision import sd_upper_bound_factor
from ._common import (
    check_count,
    check_seed,
    field_values,
    groups,
    matching_groups,
    validated,
)

DESCRIPTORS = ("mean", "median", "SD", "IQR")


@dataclass(frozen=True)
class CellRuleCheck:
    """The JCB rule applied to whole fields, per analysis group and target.

    ``table`` has, per group and target: the rule's cells (``rule_cells``),
    the cells and fields a draw really took, the averaging error the rule
    predicts for those cells and the ``real_error`` of their mean (both as
    fractions of the reference mean), their ratio, whether the real error
    ``meets_target``, and the fields the rule asks for with the field as the
    unit (``field_rule_fovs`` and ``field_rule_fovs_upper95``).
    """

    table: pd.DataFrame
    group_cols: tuple[str, ...]
    settings: dict[str, Any]

    def summary(self) -> str:
        """Plain sentences, one per group and target."""
        cells = unit_plural(self.settings["observation_unit"])
        lines = [
            (
                f"Would the JCB cell-number rule hold with whole fields? Feature "
                f"{self.settings['feature']}; reference mean: {self.settings['reference']}."
            )
        ]
        for row in self.table.to_dict("records"):
            group = where({c: row[c] for c in self.group_cols})
            if row["status"] != "ok":
                lines.append(f"{group}, target {row['target']:.1%}: {row['status']}.")
                continue
            verdict = "meets" if row["meets_target"] else "misses"
            lines.append(
                f"{group}, target {row['target']:.1%}: the rule asks for {row['rule_cells']} "
                f"{cells}, which took {row['fields_used']:.1f} fields. SD / sqrt(n) predicts "
                f"{row['predicted_error']:.1%}; the real error of that mean is "
                f"{row['real_error']:.1%} ({row['error_ratio']:.1f} times), so it {verdict} "
                f"the target. With fields as the unit the rule asks for "
                f"{fields_text(row['field_rule_fovs'])} ({row['field_rule_fovs_upper95']} from the "
                f"upper-95 SD) of {row['fovs']} acquired."
            )
        return "\n".join(lines)


def _reference_means(all_groups, feature: str, control) -> tuple[dict[int, float], str]:
    means = {}
    for group, data in all_groups:
        values = data.feature_values(feature)
        values = values[~np.isnan(values)]
        means[group_identity(group)] = float(values.mean()) if len(values) else None
    if control is None:
        return means, "each group's own mean"
    chosen = matching_groups(all_groups, control, "control")
    control_means = [means[group_identity(g)] for g in chosen if means[group_identity(g)]]
    if not control_means:
        raise ValueError("the control groups have no observations of the feature")
    reference = float(np.mean(control_means))
    label = f"mean of {len(control_means)} control group(s) {dict(control)}"
    return {k: reference for k in means}, label


def _fields_until(
    sizes: np.ndarray, sums: np.ndarray, n: int, draws: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Means, cell counts and field counts of whole fields drawn with replacement until n cells."""
    means, cells, used = np.empty(draws), np.empty(draws), np.empty(draws)
    # enough fields for every draw to reach n even when the smallest non-empty fields come up
    batch = max(4, 2 * math.ceil(n / max(sizes.mean(), 1e-12)) + 4)
    for d in range(draws):
        count, total, k = 0, 0.0, 0
        while count < n:
            picks = rng.integers(0, len(sizes), size=batch)
            running = count + np.cumsum(sizes[picks])
            stop = np.searchsorted(running, n)  # first index reaching n
            if stop < batch:
                chosen = picks[: stop + 1]
            else:
                chosen = picks
            count += int(sizes[chosen].sum())
            total += float(sums[chosen].sum())
            k += len(chosen)
        means[d], cells[d], used[d] = total / count, count, k
    return means, cells, used


def cell_rule_check(
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
) -> CellRuleCheck:
    """The JCB number of cells, and the averaging error it really gives from whole fields.

    ``targets`` are averaging errors as fractions of the reference mean: the
    paper's 5% for independent samples and 2.5% for paired ones. ``control``
    selects the control groups whose mean normalizes every group, for example
    ``{"condition": "MOCK"}``; ``None`` uses each group's own mean.
    """
    targets = [float(t) for t in targets]
    if not targets or any(not 0 < t < 1 for t in targets):
        raise ValueError("targets must be fractions between 0 and 1, such as 0.05")
    check_count(n_draws, "n_draws", 2)
    check_seed(random_state)
    data = validated(objects, fovs, [feature], fov_col, group_cols, observation_unit)
    all_groups = groups(data)
    references, label = _reference_means(all_groups, feature, control)
    rows = []
    for group, group_data in all_groups:
        fields = field_values(group_data, group_data.feature_values(feature))
        sizes = np.array([len(f) for f in fields], dtype=float)
        sums = np.array([f.sum() for f in fields])
        filled = [f for f in fields if len(f)]
        reference = references[group_identity(group)]
        base = {**group, "fovs": len(fields), "observations": int(sizes.sum())}
        for target in targets:
            if len(filled) < 2 or reference is None or reference == 0:
                rows.append({**base, "target": target, "status": "too few fields or cells"})
                continue
            cells = np.concatenate(filled) / reference
            sd = float(cells.std(ddof=1))
            n = max(1, math.ceil((sd / target) ** 2))
            rng = np.random.default_rng([random_state, group_identity(group), int(target * 1e6)])
            means, obtained, used = _fields_until(sizes, sums / reference, n, n_draws, rng)
            real = float(means.std(ddof=1))
            predicted = float(np.mean(sd / np.sqrt(obtained)))
            field_sd = float(np.std([f.mean() for f in filled], ddof=1)) / reference
            upper = field_sd * sd_upper_bound_factor(len(filled) - 1)
            rows.append(
                {
                    **base,
                    "target": target,
                    "status": "ok",
                    "cell_sd": sd,
                    "rule_cells": n,
                    "rule_exceeds_acquired": bool(n > sizes.sum()),
                    "cells_obtained": float(obtained.mean()),
                    "fields_used": float(used.mean()),
                    "predicted_error": predicted,
                    "real_error": real,
                    "error_ratio": real / predicted if predicted > 0 else None,
                    "meets_target": bool(real <= target),
                    "field_rule_fovs": max(1, math.ceil((field_sd / target) ** 2)),
                    "field_rule_fovs_upper95": max(1, math.ceil((upper / target) ** 2)),
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
    return CellRuleCheck(pd.DataFrame(rows), tuple(data.group_cols), settings)


@dataclass(frozen=True)
class DescriptorSpreadCheck:
    """JCS Fig. 2A with random cells and with whole fields, per analysis group.

    - ``curves``: per group, sample size and sampling (``"random cells"`` or
      ``"whole fields"``), the max - min of each descriptor over the samples.
    - ``at_threshold``: per group and descriptor, both spreads at the
      threshold, their ratio, and ``fields_cells_to_match``, the smallest
      ``n`` from which field samples stay within the random-cell spread at
      the threshold (``None`` when they never do within the sizes tried).
    """

    curves: pd.DataFrame
    at_threshold: pd.DataFrame
    group_cols: tuple[str, ...]
    settings: dict[str, Any]

    def summary(self) -> str:
        """Plain sentences, one per group."""
        cells = unit_plural(self.settings["observation_unit"])
        threshold = self.settings["threshold"]
        lines = [
            (
                f"Do descriptors stabilise as fast with whole fields? Feature "
                f"{self.settings['feature']}, spread over {self.settings['n_draws']} samples "
                f"of {threshold} {cells}."
            )
        ]
        keys = list(self.group_cols)
        for key, rows in (
            self.at_threshold.groupby(keys, sort=False) if keys else [((), self.at_threshold)]
        ):
            group = where(dict(zip(keys, key if isinstance(key, tuple) else (key,), strict=True)))
            parts = []
            for row in rows.to_dict("records"):
                match = row["fields_cells_to_match"]
                reach = (
                    f"matched from {int(match)} {cells}"
                    if not pd.isna(match)
                    else f"not matched within {int(row['largest_size_tried'])}"
                )
                parts.append(f"{row['descriptor']} {row['ratio']:.1f}x ({reach})")
            lines.append(f"{group}: whole fields against random {cells}: " + "; ".join(parts) + ".")
        return "\n".join(lines)


def _descriptors(values: np.ndarray) -> np.ndarray:
    q1, median, q3 = np.percentile(values, [25, 50, 75])
    return np.array([values.mean(), median, values.std(ddof=1), q3 - q1])


def _field_sample(fields: list[np.ndarray], n: int, rng: np.random.Generator) -> np.ndarray:
    taken, count = [], 0
    for i in rng.permutation(len(fields)):
        field = fields[i]
        if count + len(field) >= n:
            taken.append(rng.choice(field, size=n - count, replace=False))
            break
        taken.append(field)
        count += len(field)
    return np.concatenate(taken)


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
    """JCS Fig. 2A for every group, with random cells and with whole fields.

    ``threshold`` is the sample size at which the descriptors are read as
    stable (the paper's reading is dataset-specific, so it has no default).
    ``sizes`` default to 10, 20, ... up to twice the threshold; sizes beyond a
    group's observations are skipped. ``n_draws`` is 100 in the paper.
    """
    check_count(threshold, "threshold", 2)
    check_count(n_draws, "n_draws", 2)
    check_seed(random_state)
    sizes = list(range(10, 2 * threshold + 1, 10)) if sizes is None else list(sizes)
    for n in sizes:
        check_count(n, "every size", 2)
    if threshold not in sizes:
        sizes = sorted({*sizes, threshold})
    data = validated(objects, fovs, [feature], fov_col, group_cols, observation_unit)
    keys = list(data.group_cols)
    rows, threshold_rows = [], []
    for group, group_data in groups(data):
        fields = [f for f in field_values(group_data, group_data.feature_values(feature)) if len(f)]
        cells = np.concatenate(fields) if fields else np.array([])
        rng = np.random.default_rng([random_state, group_identity(group)])
        spreads: dict[tuple[str, int], np.ndarray] = {}
        for n in [s for s in sizes if s <= len(cells)]:
            for how in ("random cells", "whole fields"):
                if how == "random cells":
                    samples = [rng.choice(cells, size=n, replace=False) for _ in range(n_draws)]
                else:
                    samples = [_field_sample(fields, n, rng) for _ in range(n_draws)]
                values = np.array([_descriptors(s) for s in samples])
                spread = values.max(axis=0) - values.min(axis=0)
                spreads[(how, n)] = spread
                rows.append(
                    {**group, "n": n, "sampling": how, **dict(zip(DESCRIPTORS, spread.tolist()))}
                )
        if ("random cells", threshold) not in spreads:
            continue
        tried = sorted(n for how, n in spreads if how == "whole fields")
        for index, name in enumerate(DESCRIPTORS):
            reference = spreads[("random cells", threshold)][index]
            above = [n for n in tried if spreads[("whole fields", n)][index] > reference]
            later = [n for n in tried if not above or n > max(above)]
            field_spread = spreads[("whole fields", threshold)][index]
            threshold_rows.append(
                {
                    **group,
                    "descriptor": name,
                    "threshold": threshold,
                    "random_cells_spread": float(reference),
                    "whole_fields_spread": float(field_spread),
                    "ratio": float(field_spread / reference) if reference > 0 else None,
                    "fields_cells_to_match": later[0] if later else None,
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
    }
    return DescriptorSpreadCheck(
        pd.DataFrame(rows), pd.DataFrame(threshold_rows), tuple(keys), settings
    )


__all__ = [
    "DESCRIPTORS",
    "CellRuleCheck",
    "DescriptorSpreadCheck",
    "cell_rule_check",
    "descriptor_spread_check",
]
