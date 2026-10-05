"""How variable are the FOVs, and what does an object count miss? (the report's Foundation)

For one declared scalar feature, per analysis group:

- **Variance components.** The between-field component ``v_fov`` and the
  within-field component ``v_obs`` of a one-level nested random design, by the
  method of moments (``statistics.nested_variance_components``).
  The intraclass correlation is ``v_fov / (v_fov + v_obs)``, with a negative
  ``v_fov`` estimate set to zero.
- **Design effect.** ``(m v_fov + v_obs) / (v_fov + v_obs)`` with
  ``m = sum n_j^2 / N`` over the fields with observations
  (``statistics.design_effect`` and ``statistics.cluster_size``): how much larger the
  variance of a mean from whole fields is than that of a mean from as many
  objects drawn at random. Its square root, the **error ratio**, is how much an
  object-count rule such as ``SD / sqrt(n)`` understates the real standard
  error. A design effect of 1 means the fields behave like random objects.
- **FOVs that differ more than random objects would.** Each field's mean as a
  z-score, ``(mean_j - mean) / (s_within sqrt(1 / n_j - 1 / N))``, with the
  pooled within-field SD; the field is part of the overall mean, so its
  difference from it varies by ``1 / n_j - 1 / N``, not ``1 / n_j``. Under
  random objects about 5% of fields have ``|z| > 1.96``;
  the share observed is a direct reading of field-to-field variation, and the
  one-way ANOVA p-value (``scipy.stats.f_oneway``) tests it.

This is a description of the acquired fields. It does not say why fields
differ (biology, density, illumination, focus, edge effects) and does not
make neighbouring fields independent: whole-field resampling can itself be
optimistic when neighbours are correlated (AGENTS.md section 3).
Observations with a missing value are left out, counted (``missing_values``)
and stated in the summary. A feature that does not vary beyond rounding (a
constant such as 0.1 leaves components of about 1e-17) is reported as
constant, with no intraclass correlation or design effect. Zero-observation
FOVs are acquired fields: they count in ``fovs`` and in the observations per
acquired field, but carry no mean.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from .._style import check_detail
from ..schema import fov_values
from ..statistics import cluster_size, design_effect, nested_variance_components, spread_is_zero
from ._common import groups, validated
from ._unit_text import join, variability_full, variability_parts

UNUSUAL_Z = 1.96


@dataclass(frozen=True)
class FOVVariability:
    """Field-to-field variation of one feature, per analysis group.

    - ``groups``: one row per group: counts, the feature's mean and object SD,
      the SD of field means, ``v_fov``, ``v_obs``, ``icc``, ``design_effect``,
      ``error_ratio``, ``unusual_fov_share`` (against 5% under random objects)
      and ``anova_p``; ``status`` is ``"ok"`` or says why a value is missing.
    - ``fovs``: one row per acquired FOV: its observations, mean, median,
      SD and z-score, and whether it differs more than random objects would.
    """

    groups: pd.DataFrame
    fovs: pd.DataFrame
    group_cols: tuple[str, ...]
    settings: dict[str, Any]

    def summary(self, detail: str = "short") -> str:
        """The field variability in words: short (the groups in one table, the answer in one
        sentence) or ``detail="full"`` (every number of each group)."""
        if check_detail(detail) == "full":
            return variability_full(self)
        return join(*variability_parts(self))


def _group_row(per_fov: list[np.ndarray]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    sizes = np.array([len(v) for v in per_fov])
    filled = [v for v in per_fov if len(v)]
    observations = int(sizes.sum())
    row: dict[str, Any] = {
        "fovs": len(per_fov),
        "fovs_with_observations": len(filled),
        "observations": observations,
        "observations_per_fov": observations / len(per_fov) if len(per_fov) else 0.0,
    }
    empty = dict.fromkeys(
        ["mean", "object_sd", "fov_mean_sd", "v_fov", "v_obs", "icc", "design_effect"]
        + ["error_ratio", "unusual_fov_share", "anova_p", "cluster_size"]
    )
    row.update(empty)
    fov_rows = [
        {
            "observations": len(v),
            "mean": float(v.mean()) if len(v) else None,
            "median": float(np.median(v)) if len(v) else None,
            "sd": float(v.std(ddof=1)) if len(v) > 1 else None,
            "z": None,
            "unusual": None,
        }
        for v in per_fov
    ]
    if len(filled) < 2:
        row["status"] = "fewer than 2 fields with observations"
        return row, fov_rows
    if observations <= len(filled):
        row["status"] = "every field holds a single observation"
        return row, fov_rows
    values = np.concatenate(filled)
    codes = np.repeat(np.arange(len(filled)), [len(f) for f in filled])
    v_fov, v_obs = nested_variance_components(values, [codes])
    v_fov_kept, v_obs_kept = max(float(v_fov), 0.0), max(float(v_obs), 0.0)
    cluster = cluster_size(sizes)
    total = v_fov_kept + v_obs_kept
    within = np.sqrt(
        sum(((f - f.mean()) ** 2).sum() for f in filled) / (observations - len(filled))
    )
    mean = float(values.mean())
    # Spreads that are rounding noise of zero count as zero: a constant 0.1 is constant.
    # Judged on the object SD, since the variance components of a constant carry the
    # cancellation noise of their sums of squares (about 1e-17 in variance units).
    constant = total <= 0 or spread_is_zero(float(values.std(ddof=1)), values)
    varies_within = not constant and not spread_is_zero(float(within), values)
    if varies_within:
        for fov_row, v in zip(fov_rows, per_fov, strict=True):
            if len(v):
                spread = within * np.sqrt(1 / len(v) - 1 / observations)
                if spread <= 0:
                    continue  # the only field with observations
                z = (float(v.mean()) - mean) / spread
                fov_row["z"] = z
                fov_row["unusual"] = bool(abs(z) > UNUSUAL_Z)
    scored = [r["unusual"] for r in fov_rows if r["unusual"] is not None]
    deff = None if constant else design_effect(v_fov, v_obs, cluster)
    row.update(
        {
            "mean": mean,
            "object_sd": float(values.std(ddof=1)),
            "fov_mean_sd": float(np.std([f.mean() for f in filled], ddof=1)),
            "v_fov": float(v_fov),
            "v_obs": float(v_obs),
            "icc": None if constant else v_fov_kept / total,
            "cluster_size": cluster,
            "design_effect": deff,
            "error_ratio": float(np.sqrt(deff)) if deff is not None else None,
            "unusual_fov_share": float(np.mean(scored)) if scored else None,
            "anova_p": float(stats.f_oneway(*filled).pvalue) if varies_within else None,
            "status": "the feature is constant" if constant else "ok",
        }
    )
    return row, fov_rows


def fov_variability(
    objects: pd.DataFrame,
    fovs: pd.DataFrame,
    *,
    feature: str,
    fov_col: str | None = None,
    group_cols: Sequence[str] | None = None,
    observation_unit: str | None = None,
) -> FOVVariability:
    """How much the FOVs of each analysis group differ in ``feature``.

    ``fovs`` lists every acquired FOV, including FOVs without observations.
    Other arguments are those of ``validate_input_tables``; groups are
    analysed independently and pooled only when ``group_cols=[]`` is asked for.
    """
    data = validated(objects, fovs, [feature], fov_col, group_cols, observation_unit)
    keys = list(data.group_cols)
    group_rows, fov_rows = [], []
    for group, group_data in groups(data):
        values = group_data.feature_values(feature)
        row, rows = _group_row(fov_values(group_data, values))
        group_rows.append({**group, **row, "missing_values": int(np.isnan(values).sum())})
        identity = group_data.fovs[list(data.fov_key)].to_dict("records")
        fov_rows += [{**ident, **r} for ident, r in zip(identity, rows, strict=True)]
    settings = {
        "feature": feature,
        "fov_key": list(data.fov_key),
        "observation_unit": data.observation_unit,
        "observation_unit_declared": data.observation_unit_declared,
        "unusual_z": UNUSUAL_Z,
    }
    return FOVVariability(pd.DataFrame(group_rows), pd.DataFrame(fov_rows), tuple(keys), settings)


__all__ = ["FOVVariability", "fov_variability"]
