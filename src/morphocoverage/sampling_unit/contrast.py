"""Do the conditions differ, and how confidently, given that objects came as whole fields?

The lab's Plot&Stats notebook compares conditions with Cohen's d on the
observations and a randomization test that shuffles condition labels over the
observations, with a Bonferroni correction over the comparisons. That treats
every object as an independent draw. The Foundation of the report says a
microscope acquires objects through fields, and objects of one field enter
together, so this check computes each number twice: as Plot&Stats does, on
the objects, and with the field as the unit.

Per repeat, each condition is compared with the declared control condition
(the fields of a condition's analysis groups, for example its wells, are
pooled within the repeat, and the summary says so):

- **Cohen's d on objects** (Plot&Stats' number): the difference of means over
  the pooled standard deviation of all objects;
- **Cohen's d on field means**: the same with one value per field with
  observations, so a field-to-field effect counts once per field;
- **a randomization test over whole fields**: condition labels are shuffled
  over the fields of the repeat, ``n_permutations`` times, and the statistic
  is the absolute Cohen's d on field means; the p-value is
  ``(extreme + 1) / (n + 1)`` as in Plot&Stats, with a Bonferroni factor equal
  to the number of contrasts computed;
- **the JCS effect size** (Marcotti et al. 2026, Eqn 2: the treated mean
  minus the control mean over the control SD) from every object, and from
  ``object_counts`` objects drawn as random objects (the paper's procedure)
  and as whole fields in random order (what a microscope stopping at that
  count has), with the median and interquartile band over ``n_draws`` and the
  share of draws whose sign agrees with the full-data effect, as in
  ``data/crick_idr/jcs_effect_fovs.py``. The statistics themselves live in
  ``morphocoverage.statistics``.

The check describes what these fields support about a difference between
conditions under this feature; it is retrospective, and it is not the user's
biological inference, which needs independent repeats to generalize.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..metadata import MORPHOBRICKS_GROUP_COLUMNS
from ..retrospective._text import group_identity, unit_plural
from ..statistics import cohen_d, field_sample, fields_holding, jcs_effect_size
from ._common import check_count, check_seed, fov_values, groups, validated

DEFAULT_OBJECT_COUNTS = (50, 100, 200, 500)
RANDOM_OBJECTS = "random objects"
WHOLE_FOVS = "whole FOVs"


@dataclass(frozen=True)
class ConditionContrast:
    """Each condition against the control, per repeat, with objects and with fields as the unit.

    - ``table``: one row per repeat and condition: fields and observations on
      each side, ``d_objects``, ``d_fields``, the randomization ``p_value`` over
      whole fields and its Bonferroni-corrected ``p_bonferroni``, the JCS
      effect size from every object (``effect_full``), and ``status``.
    - ``effects``: one row per repeat, condition, sampling and object count:
      the median and interquartile band of the JCS effect size over the draws,
      the share of draws with the sign of ``effect_full``, and for whole
      fields the mean number of fields the objects took.
    """

    table: pd.DataFrame
    effects: pd.DataFrame
    settings: dict[str, Any]

    def summary(self) -> str:
        """Plain sentences, one paragraph per contrast."""
        words = unit_plural(self.settings["observation_unit"])
        control = self.settings["control"]
        repeat_col = self.settings["repeat_col"]
        head = (
            f"Do the conditions differ, and how confidently? Feature {self.settings['feature']}, "
            f"each condition against the control {control}"
            + (f", within each {repeat_col}" if repeat_col else "")
            + ". The fields of a condition's analysis groups are pooled for this comparison. "
            "Objects are compared as Plot&Stats does and, since a microscope acquires them "
            "through fields, with the field as the unit."
        )
        lines = [head]
        if self.table.empty:
            lines.append(self.settings.get("reason", "No condition to compare with the control."))
            return "\n".join(lines)
        n_contrasts = len(self.table)
        for row in self.table.to_dict("records"):
            where = (f"{repeat_col}={row['repeat']}, " if repeat_col else "") + (
                f"{row['condition']} vs {row['control_condition']}"
            )
            counts = (
                f"{int(row['fields'])} fields ({int(row['observations'])} {words}) against "
                f"{int(row['control_fields'])} control fields "
                f"({int(row['control_observations'])} {words})."
            )
            if row["status"] != "ok":
                lines.append(f"{where}: {counts} {row['status']}.")
                continue
            objects_d = f"{row['d_objects']:.2f}" if np.isfinite(row["d_objects"]) else "undefined"
            fields_d = f"{row['d_fields']:.2f}" if np.isfinite(row["d_fields"]) else "undefined"
            lines.append(
                f"{where}: {counts} Cohen's d on {words} {objects_d}; on field means {fields_d}. "
                f"Randomization test over whole fields ({self.settings['n_permutations']} "
                f"shuffles of the condition labels over fields): p = {row['p_value']:.3g}"
                f" (Bonferroni over {n_contrasts} comparison"
                f"{'s' if n_contrasts != 1 else ''}: {row['p_bonferroni']:.3g})."
            )
            effects = self.effects[
                (self.effects["repeat"] == row["repeat"])
                & (self.effects["condition"] == row["condition"])
            ]
            if effects.empty:
                continue
            full = row["effect_full"]
            parts = []
            for n, block in effects.groupby("n_objects", sort=True):
                by = block.set_index("sampling")
                bits = []
                for sampling in (RANDOM_OBJECTS, WHOLE_FOVS):
                    if sampling not in by.index:
                        continue
                    r = by.loc[sampling]
                    bit = (
                        f"{sampling} median {r['median']:.2f} (IQR {r['q25']:.2f} to "
                        f"{r['q75']:.2f}), same sign in {100 * r['same_sign']:.0f}% of draws"
                    )
                    if sampling == WHOLE_FOVS:
                        bit += f", about {r['fields_treated']:.0f} fields"
                    bits.append(bit)
                parts.append(f"from {int(n)} {words}: " + "; ".join(bits))
            lines.append(
                f"  JCS effect size (treated mean minus control mean, over the control SD) from "
                f"every object: {full:.2f}; " + ". ".join(parts) + "."
            )
        return "\n".join(lines)


def _side_fields(
    all_groups, feature: str, chosen: list[dict[str, Any]]
) -> tuple[list[np.ndarray], int]:
    """The fields (arrays of the feature, NaN dropped) of the chosen groups, and the field count."""
    by_identity = {group_identity(g): d for g, d in all_groups}
    fields, acquired = [], 0
    for group in chosen:
        data = by_identity[group_identity(group)]
        fields += fov_values(data, data.feature_values(feature))
        acquired += len(data.fovs)
    return fields, acquired


def condition_contrast(
    objects: pd.DataFrame,
    fovs: pd.DataFrame,
    *,
    feature: str,
    control: Mapping[str, Any],
    condition_col: str = MORPHOBRICKS_GROUP_COLUMNS[0],
    repeat_col: str | None = MORPHOBRICKS_GROUP_COLUMNS[1],
    object_counts: Sequence[int] = DEFAULT_OBJECT_COUNTS,
    n_permutations: int = 2000,
    n_draws: int = 200,
    random_state: int = 0,
    fov_col: str | None = None,
    group_cols: Sequence[str] | None = None,
    observation_unit: str | None = None,
) -> ConditionContrast:
    """Each condition against the control condition, per repeat, with objects and with fields.

    ``control`` names the control condition by FOV-table value, for example
    ``{"condition": "MOCK"}``; its key is the condition column unless
    ``condition_col`` says otherwise. ``repeat_col`` separates the repeats
    (``None`` compares within the whole table). Every analysis group whose
    condition is not the control's is compared with the control groups of
    the same repeat; the groups of one condition (its wells, say) are pooled
    for the comparison, which the summary states. ``object_counts`` are the
    JCS sample sizes; counts above what a side holds are left out.
    """
    if not isinstance(feature, str) or not feature:
        raise TypeError("feature must be one column name")
    if not isinstance(control, Mapping) or not control:
        raise ValueError("control must be a non-empty mapping of FOV-table column to value")
    if condition_col not in control:
        raise ValueError(f"control must name the condition column {condition_col!r}")
    check_count(n_permutations, "n_permutations", 1)
    check_count(n_draws, "n_draws", 2)
    check_seed(random_state)
    counts = sorted({check_count(n, "object_counts", 2) for n in object_counts})
    data = validated(objects, fovs, [feature], fov_col, group_cols, observation_unit)
    if condition_col not in data.fovs.columns:
        raise ValueError(f"condition_col {condition_col!r} is not a column of the FOV table")
    if repeat_col is not None and repeat_col not in data.fovs.columns:
        repeat_col = None  # one implicit repeat
    all_groups = groups(data)

    def value_of(group_data, column):
        values = group_data.fovs[column].drop_duplicates()
        if len(values) != 1:
            raise ValueError(
                f"an analysis group spans several values of {column!r}; add it to group_cols"
            )
        return values.iloc[0]

    described = []
    for group, group_data in all_groups:
        described.append(
            {
                "group": group,
                "condition": value_of(group_data, condition_col),
                "repeat": value_of(group_data, repeat_col) if repeat_col else None,
                "is_control": all(
                    value_of(group_data, column) == value for column, value in control.items()
                ),
            }
        )
    if not any(d["is_control"] for d in described):
        raise ValueError(f"no analysis group matches control {dict(control)}")
    settings = {
        "feature": feature,
        "control": dict(control),
        "condition_col": condition_col,
        "repeat_col": repeat_col,
        "object_counts": counts,
        "n_permutations": n_permutations,
        "n_draws": n_draws,
        "random_state": random_state,
        "observation_unit": data.observation_unit,
    }
    contrasts = sorted(
        {(d["repeat"], d["condition"]) for d in described if not d["is_control"]},
        key=lambda pair: (str(pair[0]), str(pair[1])),
    )
    if not contrasts:
        settings["reason"] = (
            f"Every analysis group is the control {dict(control)}: no other condition to compare."
        )
        return ConditionContrast(pd.DataFrame(), pd.DataFrame(), settings)

    rows, effect_rows = [], []
    for repeat, condition in contrasts:
        controls = [d["group"] for d in described if d["is_control"] and d["repeat"] == repeat]
        treated = [
            d["group"]
            for d in described
            if not d["is_control"] and d["repeat"] == repeat and d["condition"] == condition
        ]
        control_label = "/".join(
            sorted({str(d["condition"]) for d in described if d["group"] in controls})
        )
        base = {
            "repeat": repeat,
            "condition": condition,
            "control_condition": control_label or str(control[condition_col]),
        }
        if not controls:
            rows.append(
                {
                    **base,
                    "fields": 0,
                    "observations": 0,
                    "control_fields": 0,
                    "control_observations": 0,
                    "d_objects": np.nan,
                    "d_fields": np.nan,
                    "p_value": np.nan,
                    "p_bonferroni": np.nan,
                    "effect_full": np.nan,
                    "status": f"no control group in {repeat_col}={repeat}",
                }
            )
            continue
        t_fields, t_acquired = _side_fields(all_groups, feature, treated)
        c_fields, c_acquired = _side_fields(all_groups, feature, controls)
        t_objects = np.concatenate(t_fields) if t_fields else np.array([])
        c_objects = np.concatenate(c_fields) if c_fields else np.array([])
        t_means = np.array([f.mean() for f in t_fields if len(f)])
        c_means = np.array([f.mean() for f in c_fields if len(f)])
        row = {
            **base,
            "fields": t_acquired,
            "observations": len(t_objects),
            "control_fields": c_acquired,
            "control_observations": len(c_objects),
            "d_objects": cohen_d(t_objects, c_objects),
            "d_fields": cohen_d(t_means, c_means),
            "p_value": np.nan,
            "p_bonferroni": np.nan,
            "effect_full": jcs_effect_size(t_objects, c_objects),
            "status": "ok",
        }
        if len(t_means) < 2 or len(c_means) < 2:
            row["status"] = "fewer than 2 fields with observations on one side"
            rows.append(row)
            continue
        rng = np.random.default_rng([random_state, group_identity(base)])
        observed = abs(row["d_fields"])
        pooled_means = np.concatenate([t_means, c_means])
        n_treated = len(t_means)
        extreme = 0
        for _ in range(n_permutations):
            shuffled = rng.permutation(pooled_means)
            statistic = abs(cohen_d(shuffled[:n_treated], shuffled[n_treated:]))
            if np.isfinite(statistic) and statistic >= observed:
                extreme += 1
        row["p_value"] = (extreme + 1) / (n_permutations + 1)
        rows.append(row)
        # The JCS effect size at each sample size, with random objects and with whole fields.
        n_max = min(len(t_objects), len(c_objects))
        full = row["effect_full"]
        for n in [n for n in counts if n <= n_max]:
            for sampling in (RANDOM_OBJECTS, WHOLE_FOVS):
                draws = np.empty(n_draws)
                for i in range(n_draws):
                    if sampling == RANDOM_OBJECTS:
                        t = rng.choice(t_objects, size=n, replace=False)
                        c = rng.choice(c_objects, size=n, replace=False)
                    else:
                        t = field_sample([f for f in t_fields if len(f)], n, rng)
                        c = field_sample([f for f in c_fields if len(f)], n, rng)
                    draws[i] = jcs_effect_size(t, c)
                defined = draws[np.isfinite(draws)]
                effect_rows.append(
                    {
                        **base,
                        "sampling": sampling,
                        "n_objects": n,
                        "median": float(np.median(defined)) if len(defined) else np.nan,
                        "q25": float(np.percentile(defined, 25)) if len(defined) else np.nan,
                        "q75": float(np.percentile(defined, 75)) if len(defined) else np.nan,
                        "same_sign": (
                            float(np.mean(np.sign(defined) == np.sign(full)))
                            if len(defined) and np.isfinite(full) and full != 0
                            else np.nan
                        ),
                        "fields_treated": (
                            fields_holding([f for f in t_fields if len(f)], n, rng)
                            if sampling == WHOLE_FOVS
                            else np.nan
                        ),
                        "draws": len(defined),
                    }
                )
    table = pd.DataFrame(rows)
    tested = table["p_value"].notna()
    table["p_bonferroni"] = np.where(
        tested, np.minimum(1.0, table["p_value"] * int(tested.sum())), np.nan
    )
    return ConditionContrast(table, pd.DataFrame(effect_rows), settings)


__all__ = ["ConditionContrast", "condition_contrast"]
