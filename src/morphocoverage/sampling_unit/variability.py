"""How variable are the fields of a dataset? (``docs/questions.md``, F1 and F2)

For one declared scalar feature, per analysis group:

- **Variance components.** The between-field component ``v_fov`` and the
  within-field component ``v_obs`` of a one-level nested random design, by the
  method of moments (``retrospective.effort.nested_variance_components``).
  The intraclass correlation is ``v_fov / (v_fov + v_obs)``, with a negative
  ``v_fov`` estimate set to zero.
- **Design effect.** ``(m v_fov + v_obs) / (v_fov + v_obs)`` with
  ``m = sum n_j^2 / N`` over the fields with observations: how much larger the
  variance of a mean from whole fields is than that of a mean from as many
  cells drawn at random. Its square root, the **error ratio**, is how much a
  cell-number rule such as ``SD / sqrt(n)`` understates the real standard
  error. A design effect of 1 means the fields behave like random cells.
- **Fields that differ more than random cells would.** Each field's mean as a
  z-score, ``(mean_j - mean) / (s_within / sqrt(n_j))``, with the pooled
  within-field SD. Under random cells about 5% of fields have ``|z| > 1.96``;
  the share observed is a direct reading of field-to-field variation, and the
  one-way ANOVA p-value (``scipy.stats.f_oneway``) tests it.

This is a description of the acquired fields. It does not say why fields
differ (biology, density, illumination, focus, edge effects) and does not
make neighbouring fields independent: whole-field resampling can itself be
optimistic when neighbours are correlated (AGENTS.md section 3).
Observations with a missing value are left out and counted. Zero-observation
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

from ..retrospective._text import unit_plural, where
from ..retrospective.effort import nested_variance_components
from ._common import field_values, groups, validated

UNUSUAL_Z = 1.96


@dataclass(frozen=True)
class FieldVariability:
    """Field-to-field variation of one feature, per analysis group.

    - ``groups``: one row per group: counts, the feature's mean and cell SD,
      the SD of field means, ``v_fov``, ``v_obs``, ``icc``, ``design_effect``,
      ``error_ratio``, ``unusual_field_share`` (against 5% under random cells)
      and ``anova_p``; ``status`` is ``"ok"`` or says why a value is missing.
    - ``fields``: one row per acquired field: its observations, mean, median,
      SD and z-score, and whether it differs more than random cells would.
    """

    groups: pd.DataFrame
    fields: pd.DataFrame
    group_cols: tuple[str, ...]
    settings: dict[str, Any]

    def summary(self) -> str:
        """Plain sentences, one paragraph per group."""
        cells = unit_plural(self.settings["observation_unit"])
        lines = [f"How variable are the fields? Feature {self.settings['feature']}."]
        for row in self.groups.to_dict("records"):
            group = {c: row[c] for c in self.group_cols}
            head = (
                f"{where(group)}: {row['fovs']} fields, {row['observations']} {cells} "
                f"({row['observations_per_fov']:.0f} per field)."
            )
            if row["status"] != "ok":
                lines.append(f"{head} {row['status']}.")
                continue
            lines.append(
                f"{head} {row['icc']:.0%} of the variance lies between fields. A mean from "
                f"whole fields has {row['design_effect']:.1f} times the variance of one from "
                f"as many random {cells}, so SD / sqrt(n) understates its error "
                f"{row['error_ratio']:.1f}-fold. {row['unusual_field_share']:.0%} of fields "
                f"differ more than random {cells} would let them (about 5% expected)."
            )
        return "\n".join(lines)


def _group_row(fields: list[np.ndarray]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    sizes = np.array([len(f) for f in fields])
    filled = [f for f in fields if len(f)]
    observations = int(sizes.sum())
    row: dict[str, Any] = {
        "fovs": len(fields),
        "fovs_with_observations": len(filled),
        "observations": observations,
        "observations_per_fov": observations / len(fields) if len(fields) else 0.0,
    }
    empty = dict.fromkeys(
        ["mean", "cell_sd", "field_mean_sd", "v_fov", "v_obs", "icc", "design_effect"]
        + ["error_ratio", "unusual_field_share", "anova_p", "cluster_size"]
    )
    row.update(empty)
    field_rows = [
        {
            "observations": len(f),
            "mean": float(f.mean()) if len(f) else None,
            "median": float(np.median(f)) if len(f) else None,
            "sd": float(f.std(ddof=1)) if len(f) > 1 else None,
            "z": None,
            "unusual": None,
        }
        for f in fields
    ]
    if len(filled) < 2:
        row["status"] = "fewer than 2 fields with observations"
        return row, field_rows
    if observations <= len(filled):
        row["status"] = "every field holds a single observation"
        return row, field_rows
    values = np.concatenate(filled)
    codes = np.repeat(np.arange(len(filled)), [len(f) for f in filled])
    v_fov, v_obs = nested_variance_components(values, [codes])
    v_fov_kept, v_obs_kept = max(float(v_fov), 0.0), max(float(v_obs), 0.0)
    cluster = float(np.sum(np.array([len(f) for f in filled], float) ** 2) / observations)
    total = v_fov_kept + v_obs_kept
    within = np.sqrt(
        sum(((f - f.mean()) ** 2).sum() for f in filled) / (observations - len(filled))
    )
    mean = float(values.mean())
    if within > 0:
        for field_row, f in zip(field_rows, fields, strict=True):
            if len(f):
                z = (float(f.mean()) - mean) / (within / np.sqrt(len(f)))
                field_row["z"] = z
                field_row["unusual"] = bool(abs(z) > UNUSUAL_Z)
    scored = [r["unusual"] for r in field_rows if r["unusual"] is not None]
    deff = (cluster * v_fov_kept + v_obs_kept) / total if total > 0 else None
    row.update(
        {
            "mean": mean,
            "cell_sd": float(values.std(ddof=1)),
            "field_mean_sd": float(np.std([f.mean() for f in filled], ddof=1)),
            "v_fov": float(v_fov),
            "v_obs": float(v_obs),
            "icc": v_fov_kept / total if total > 0 else None,
            "cluster_size": cluster,
            "design_effect": deff,
            "error_ratio": float(np.sqrt(deff)) if deff is not None else None,
            "unusual_field_share": float(np.mean(scored)) if scored else None,
            "anova_p": float(stats.f_oneway(*filled).pvalue) if within > 0 else None,
            "status": "ok" if total > 0 else "the feature is constant",
        }
    )
    return row, field_rows


def field_variability(
    objects: pd.DataFrame,
    fovs: pd.DataFrame,
    *,
    feature: str,
    fov_col: str | None = None,
    group_cols: Sequence[str] | None = None,
    observation_unit: str | None = None,
) -> FieldVariability:
    """How much the fields of each analysis group differ in ``feature``.

    ``fovs`` lists every acquired FOV, including FOVs without observations.
    Other arguments are those of ``validate_input_tables``; groups are
    analysed independently and pooled only when ``group_cols=[]`` is asked for.
    """
    data = validated(objects, fovs, [feature], fov_col, group_cols, observation_unit)
    keys = list(data.group_cols)
    group_rows, field_rows = [], []
    for group, group_data in groups(data):
        values = group_data.feature_values(feature)
        fields = field_values(group_data, values)
        row, rows = _group_row(fields)
        group_rows.append({**group, **row, "missing_values": int(np.isnan(values).sum())})
        identity = group_data.fovs[list(data.fov_key)].to_dict("records")
        field_rows += [{**ident, **r} for ident, r in zip(identity, rows, strict=True)]
    settings = {
        "feature": feature,
        "fov_key": list(data.fov_key),
        "observation_unit": data.observation_unit,
        "unusual_z": UNUSUAL_Z,
    }
    return FieldVariability(
        pd.DataFrame(group_rows), pd.DataFrame(field_rows), tuple(keys), settings
    )


__all__ = ["FieldVariability", "field_variability"]
