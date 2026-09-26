"""Do the conditions differ, and how confidently, given that objects came as whole fields?

The lab's Plot&Stats notebook compares conditions with Cohen's d on the
observations and a randomization test that shuffles condition labels over the
observations, with a Bonferroni correction over the comparisons. That treats
every object as an independent draw. The Foundation of the report says a
microscope acquires objects through fields, and objects of one field enter
together, so this check computes the effect size and the test both ways: as
Plot&Stats does, on the objects, and with the field (or the technical repeat)
as the unit.

Per repeat, each condition is compared with the declared control condition
(the fields of a condition's analysis groups, for example its wells, are
pooled within the repeat, and the summary says so):

- **Cohen's d on objects** (Plot&Stats' number): the difference of means over
  the pooled standard deviation of all objects;
- **Cohen's d on field means**: the same with one value per field with
  observations, so a field-to-field effect counts once per field; field
  means that do not vary within either side but differ between them are a
  perfect separation, an infinite d (as ``feature_ranking`` scores it);
- **Cohen's d on technical-repeat means**, when the analysis groups are
  split by the MorphoBricks ``technical_repeat`` column (a well, a dish) and
  each side holds at least two: one value per technical repeat, the unit at
  which a well-to-well shift counts once;
- **a randomization test over whole units**: condition labels are shuffled
  ``n_permutations`` times over the technical repeats of the repeat when
  both sides hold at least two (each keeps its fields together), otherwise,
  without the technical-repeat column, over its whole fields; the statistic
  is the absolute Cohen's d on field means, the p-value
  ``(extreme + 1) / (n + 1)`` as in Plot&Stats, or exact when no more
  distinct assignments exist than shuffles (every one is then enumerated),
  with a Bonferroni factor equal to the number of contrasts computed. With
  few technical repeats only a few distinct assignments exist, and the
  summary says how many. A side holding a single technical repeat gets no
  test: its fields cannot stand in for technical repeats, since a
  well-to-well shift would then pass for a condition effect;
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
from math import comb
from typing import Any

import numpy as np
import pandas as pd

from .._seeds import group_identity
from ..metadata import MORPHOBRICKS_GROUP_COLUMNS, TECHNICAL_REPEAT_COLUMN
from ..schema import unit_plural
from ..statistics import cohen_d, field_sample, fields_holding, jcs_effect_size
from ._common import check_count, check_seed, groups, validated
from ._units import (
    FIELD,
    contrasts_of,
    describe_groups,
    field_means,
    permutation_p,
    relabellings,
    resolve_columns,
    separation_d,
    side_units,
    sides,
    technical_counts,
    unit_means,
    untested_reason,
)
from .object_rules import RANDOM_OBJECTS, WHOLE_FOVS

DEFAULT_OBJECT_COUNTS = (50, 100, 200, 500)


@dataclass(frozen=True)
class ConditionContrast:
    """Each condition against the control, per repeat, with objects and with fields as the unit.

    - ``table``: one row per repeat and condition: fields and observations on
      each side, technical repeats on each side (``technical``,
      ``control_technical``; 0 without the column), ``d_objects``,
      ``d_fields``, ``d_technical``, the randomization ``p_value`` over whole
      units (``permutation_unit``: the technical-repeat column or ``field``,
      with the ``distinct_assignments`` that exist for technical repeats) and
      its Bonferroni-corrected ``p_bonferroni``, whether the p-value is exact
      (``p_exact``: every assignment enumerated), the JCS effect size from
      every object (``effect_full``), and ``status``.
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
        technical_col = self.settings["technical_col"]
        head = (
            f"Do the conditions differ, and how confidently? Feature {self.settings['feature']}, "
            f"each condition against the control {control}"
            + (f", within each {repeat_col}" if repeat_col else "")
            + ". The fields of a condition's analysis groups are pooled for this comparison. "
            "Objects are compared as Plot&Stats does and, since a microscope acquires them "
            "through fields, with the field as the unit"
            + (
                f" and, where each side has at least two, with the technical repeat "
                f"({technical_col}) as the unit."
                if technical_col
                else "."
            )
        )
        lines = [head]
        if self.table.empty:
            lines.append(self.settings.get("reason", "No condition to compare with the control."))
            return "\n".join(lines)
        # The Bonferroni factor counts the contrasts that were tested, as computed.
        n_contrasts = int(self.table["p_value"].notna().sum())
        for row in self.table.to_dict("records"):
            where = (f"{repeat_col}={row['repeat']}, " if repeat_col else "") + (
                f"{row['condition']} vs {row['control_condition']}"
            )
            technical = ""
            if technical_col:
                technical = (
                    f" in {int(row['technical'])} and {int(row['control_technical'])} "
                    f"technical repeats"
                )
            counts = (
                f"{int(row['fields'])} fields ({int(row['observations'])} {words}) against "
                f"{int(row['control_fields'])} control fields "
                f"({int(row['control_observations'])} {words}){technical}."
            )
            if row["status"] != "ok":
                lines.append(f"{where}: {counts} {row['status']}.")
                continue
            technical_d = ""
            if technical_col:
                technical_d = f"; on technical-repeat means {_number(row['d_technical'])}"
            if row["permutation_unit"] == FIELD:
                shuffled = "whole fields"
            else:
                shuffled = (
                    f"whole technical repeats ({row['permutation_unit']}; only "
                    f"{int(row['distinct_assignments'])} distinct assignments exist)"
                )
            how = (
                "every assignment of the condition labels enumerated, so p is exact"
                if row["p_exact"]
                else f"{self.settings['n_permutations']} shuffles of the condition labels"
            )
            lines.append(
                f"{where}: {counts} Cohen's d on {words} {_number(row['d_objects'])}; on field "
                f"means {_number(row['d_fields'])}{technical_d}. Randomization test over "
                f"{shuffled}, {how}: p = {row['p_value']:.3g} (Bonferroni over {n_contrasts} "
                f"comparison{'s' if n_contrasts != 1 else ''}: {row['p_bonferroni']:.3g})."
            )
            if self.effects.empty:
                continue  # no object count fits the data: there are no draws to report
            # Without a repeat column the repeat is None, which == never matches.
            same_repeat = (
                self.effects["repeat"].isna()
                if pd.isna(row["repeat"])
                else self.effects["repeat"] == row["repeat"]
            )
            effects = self.effects[same_repeat & (self.effects["condition"] == row["condition"])]
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


def _number(value: float) -> str:
    """A statistic to two decimals, ``undefined``, or infinite for a perfect separation."""
    if np.isnan(value):
        return "undefined"
    if np.isinf(value):
        return f"{'' if value > 0 else '-'}infinite (no variation within either side)"
    return f"{value:.2f}"


def condition_contrast(
    objects: pd.DataFrame,
    fovs: pd.DataFrame,
    *,
    feature: str,
    control: Mapping[str, Any],
    condition_col: str = MORPHOBRICKS_GROUP_COLUMNS[0],
    repeat_col: str | None = MORPHOBRICKS_GROUP_COLUMNS[1],
    technical_col: str | None = TECHNICAL_REPEAT_COLUMN,
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
    for the comparison, which the summary states. ``technical_col`` names the
    technical-repeat column; when it is one of the analysis-group columns,
    the groups are the technical repeats, compared and shuffled as units
    (see the module). ``object_counts`` are the JCS sample sizes; counts
    above what a side holds are left out.
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
    repeat_col, technical_col = resolve_columns(
        data, condition_col, repeat_col, technical_col, control
    )
    all_groups = groups(data)
    described = describe_groups(all_groups, condition_col, repeat_col, control)
    settings = {
        "feature": feature,
        "control": dict(control),
        "condition_col": condition_col,
        "repeat_col": repeat_col,
        "technical_col": technical_col,
        "object_counts": counts,
        "n_permutations": n_permutations,
        "n_draws": n_draws,
        "random_state": random_state,
        "observation_unit": data.observation_unit,
    }
    contrasts = contrasts_of(described)
    if not contrasts:
        settings["reason"] = (
            f"Every analysis group is the control {dict(control)}: no other condition to compare."
        )
        return ConditionContrast(pd.DataFrame(), pd.DataFrame(), settings)

    rows, effect_rows = [], []
    for repeat, condition in contrasts:
        treated, controls = sides(described, repeat, condition)
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
                    "technical": 0,
                    "control_technical": 0,
                    "d_objects": np.nan,
                    "d_fields": np.nan,
                    "d_technical": np.nan,
                    "permutation_unit": FIELD,
                    "distinct_assignments": np.nan,
                    "p_value": np.nan,
                    "p_bonferroni": np.nan,
                    "p_exact": False,
                    "effect_full": np.nan,
                    "status": f"no control group in {repeat_col}={repeat}",
                }
            )
            continue
        t_units, t_acquired = side_units(all_groups, feature, treated)
        c_units, c_acquired = side_units(all_groups, feature, controls)
        t_fields = [f for u in t_units for f in u]
        c_fields = [f for u in c_units for f in u]
        t_objects = np.concatenate(t_fields) if t_fields else np.array([])
        c_objects = np.concatenate(c_fields) if c_fields else np.array([])
        t_means, c_means = field_means(t_units), field_means(c_units)
        # Groups are technical repeats only when split by that column; units
        # without observations do not count.
        n_technical = technical_counts(t_units, c_units, technical_col)
        by_unit = min(n_technical) >= 2
        row = {
            **base,
            "fields": t_acquired,
            "observations": len(t_objects),
            "control_fields": c_acquired,
            "control_observations": len(c_objects),
            "technical": n_technical[0],
            "control_technical": n_technical[1],
            "d_objects": cohen_d(t_objects, c_objects),
            # The measure feature_ranking scores: a perfect separation is an infinite d.
            "d_fields": separation_d(t_means, c_means),
            "d_technical": (
                separation_d(unit_means(t_units), unit_means(c_units)) if by_unit else np.nan
            ),
            "permutation_unit": technical_col if by_unit else FIELD,
            "distinct_assignments": comb(sum(n_technical), n_technical[0]) if by_unit else np.nan,
            "p_value": np.nan,
            "p_bonferroni": np.nan,
            "p_exact": False,
            "effect_full": jcs_effect_size(t_objects, c_objects),
            "status": "ok",
        }
        reason = untested_reason((len(t_means), len(c_means)), n_technical, technical_col)
        if reason is not None:
            row["status"] = reason
            rows.append(row)
            continue
        if np.isnan(row["d_fields"]):
            # Every field mean equal on both sides (up to rounding): nothing to test.
            row["status"] = "the field means do not vary, so the field-level test is undefined"
            rows.append(row)
            continue
        rng = np.random.default_rng([random_state, group_identity(base)])
        row["p_exact"] = relabellings(t_units, c_units, by_unit) <= n_permutations
        row["p_value"] = permutation_p(
            t_units, c_units, by_unit, abs(row["d_fields"]), n_permutations, rng
        )
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
