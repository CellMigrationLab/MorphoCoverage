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
- **Cohen's d on technical-repeat means**, when the fields have a unit they
  share and each side holds at least two: one value per unit, the level at
  which a well-to-well shift counts once. The unit (``_units.resolve_unit``)
  is the MorphoBricks ``technical_repeat`` column (a well, a dish), or the
  column ``technical_col`` names, whatever ``group_cols`` says (analysis
  groups that pool technical repeats are split by it); without it, the
  analysis group when ``group_cols`` hold a column beyond the condition and
  the repeat (``well``, ``Metadata_Well``) nested within condition x repeat
  with more than one value in some condition. With three or fewer units a
  side the summary says the d is unstable;
- **a randomization test over whole units**: condition labels are shuffled
  ``n_permutations`` times over the units of the repeat when both sides hold
  at least two (each keeps its fields together); the statistic is the
  absolute Cohen's d on field means, the p-value ``(extreme + 1) / (n + 1)``
  as in Plot&Stats, never below the smallest p-value the exact test can
  give, or exact when the distinct assignments are no more than the shuffles
  or 20000 (every one is then enumerated), with a Bonferroni factor equal to
  the number of contrasts tested. With few units only a few distinct
  assignments exist, and the summary says how many. A side holding a single
  unit gets no test: its fields cannot stand in for technical repeats, since
  a well-to-well shift would then pass for a condition effect; its effect
  sizes are still reported. **Without a unit there is no test** (fields of one
  dish or well are not independent draws: with no condition effect, 42-57% of
  field-level tests over one to three unrecorded wells or dishes a side gave
  p <= 0.05), unless the caller declares ``fields_independent=True``: whole
  fields are then shuffled, and the summary says the p-value treats them as
  independent draws;
- **the JCS effect size** (Marcotti et al. 2026, Eqn 2: the treated mean
  minus the control mean over the control SD) from every object, and from
  ``object_counts`` objects drawn as random objects (the paper's procedure)
  and as whole fields in random order (what a microscope stopping at that
  count has, acquired fields without observations included), with the
  median and interquartile band over ``n_draws`` and the share of draws
  whose sign agrees with the full-data effect, as in
  ``data/crick_idr/jcs_effect_fovs.py``. With units on both sides the
  whole-field draws take the fields of units drawn at random with
  replacement (a well, then its fields in random order, until the count),
  so the share reflects well-to-well variation; without, the draws resample
  these fields only, and the summary says their share is not a confidence
  in the effect. The statistics themselves live in ``morphocoverage.statistics``.

Every test and effect size compares means: a difference confined to a
subpopulation or to the shape of the distribution is not tested (the
summary says so). Draws walk the fields in a canonical order, so the row
order of the tables never changes a result. The check describes what these
fields support about a difference between conditions under this feature; it
is retrospective, and it is not the user's biological inference, which needs
independent repeats to generalize.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .._checks import as_count
from .._seeds import group_rng
from .._wording import count, listing, percent
from .._wording import fields as fields_text
from ..metadata import MORPHOBRICKS_GROUP_COLUMNS, TECHNICAL_REPEAT_COLUMN
from ..schema import unit_plural
from ..statistics import cohen_d, field_sample, fields_holding, jcs_effect_size
from ._common import groups, validated
from ._units import (
    FIELD,
    NOT_INDEPENDENT,
    check_arguments,
    contrasts_of,
    control_label,
    describe_groups,
    opposite_signs,
    randomization_test,
    resolve_columns,
    resolve_unit,
    separation_d,
    side_units,
    sides,
    unit_means,
)
from .object_rules import RANDOM_OBJECTS, WHOLE_FOVS

DEFAULT_OBJECT_COUNTS = (50, 100, 200, 500)
_NO_CONTROL = "no control group in"
# Units a side at or below which a d on unit means is printed as unstable.
FEW_UNITS_D = 3
_P_ALPHA = 0.05  # the p-value at which contrasts of opposite signs are flagged


@dataclass(frozen=True)
class ConditionContrast:
    """Each condition against the control, per repeat, with objects and with fields as the unit.

    - ``table``: one row per repeat and condition: fields and observations on
      each side, technical repeats on each side (``technical``,
      ``control_technical``: units of ``settings["unit"]``, the
      technical-repeat column or the nested group column; 0 without a unit),
      ``d_objects``, ``d_fields``, ``d_technical``, the randomization
      ``p_value`` over whole units (``permutation_unit``: the unit or
      ``field``, with the ``distinct_assignments`` that exist for units) and
      its Bonferroni-corrected ``p_bonferroni``, whether the p-value is exact
      (``p_exact``: every assignment enumerated), the smallest p-value the
      test can give (``p_floor``), the JCS effect size from every object
      (``effect_full``), and ``status`` (``"ok"``, or why there is no test;
      the effect sizes are computed either way).
    - ``effects``: one row per repeat, condition, sampling and object count:
      the median and interquartile band of the JCS effect size over the draws,
      the share of draws with the sign of ``effect_full``, and for whole
      fields the mean number of acquired fields the objects took (fields
      without observations included, as a microscope acquires them) and
      ``resampled``, the unit whose technical repeats the draws resampled
      with replacement (``None`` when they resample fields only).
    - ``settings``: the arguments, with ``unit`` (what the test shuffles
      beyond fields, ``None`` without one), ``unit_note`` (why a group column
      was not used as the unit) and ``fields_independent``.
    """

    table: pd.DataFrame
    effects: pd.DataFrame
    settings: dict[str, Any]

    def summary(self) -> str:
        """Plain sentences, one paragraph per contrast, and what the test and draws cannot say."""
        unit = self.settings["observation_unit"]
        words = unit_plural(unit)
        control = self.settings["control"]
        repeat_col = self.settings["repeat_col"]
        shared = self.settings.get("unit")
        head = (
            f"Do the conditions differ, and how confidently? Feature {self.settings['feature']}, "
            f"each condition against the control {control}"
            + (f", within each {repeat_col}" if repeat_col else "")
            + ". The fields of a condition's analysis groups are pooled for this comparison. "
            "Objects are compared as Plot&Stats does and, since a microscope acquires them "
            "through fields, with the field as the unit"
            + (
                f" and, where each side has at least two, with the technical repeat "
                f"({shared}) as the unit."
                if shared
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
            if shared:
                technical = (
                    f" in {int(row['technical'])} and {int(row['control_technical'])} "
                    f"technical repeats"
                )
            counts = (
                f"{fields_text(int(row['fields']))} ({count(int(row['observations']), unit)}) "
                f"against {int(row['control_fields'])} control fields "
                f"({count(int(row['control_observations']), unit)}){technical}."
            )
            if str(row["status"]).startswith(_NO_CONTROL):
                lines.append(f"{where}: {counts} {row['status']}.")
                continue
            technical_d = ""
            if shared and not pd.isna(row["d_technical"]):
                technical_d = f"; on technical-repeat means {_number(row['d_technical'])}"
                fewest = min(int(row["technical"]), int(row["control_technical"]))
                if fewest <= FEW_UNITS_D:
                    technical_d += (
                        f" (from {int(row['technical'])} and {int(row['control_technical'])} "
                        "means, so unstable: a d from so few values can be very large by chance)"
                    )
            sizes = (
                f"Cohen's d on {words} {_number(row['d_objects'])}; on field "
                f"means {_number(row['d_fields'])}{technical_d}."
            )
            lines.append(f"{where}: {counts} {sizes} {self._test(row, n_contrasts)}")
            effects = self._effects_line(row, words)
            if effects:
                lines.append(effects)
        lines += self._cautions(words)
        return "\n".join(lines)

    def _cautions(self, words: str) -> list[str]:
        """What the tests and the draws do not say, and contrasts of opposite signs."""
        lines = []
        note = self.settings.get("unit_note")
        if note:
            lines.append(f"  Note: {note}.")
        if (self.table["status"] == NOT_INDEPENDENT).any():
            lines.append(
                "  No test where the fields' wells or dishes are unknown: record the "
                "technical_repeat column (or name the well column in group_cols) to test over "
                "wells, or pass fields_independent=True (Setup.fields_independent) if the "
                "fields are independent draws of the condition."
            )
        passing = self.table[self.table["p_value"] <= _P_ALPHA]
        lines += opposite_signs(
            passing,
            ["condition"],
            self.settings["repeat_col"],
            lambda key: f"{key[0]} differs from the control at p <= {_P_ALPHA:g}",
        )
        lines.append(
            "  The test and the effect sizes compare means: a difference confined to a "
            f"subpopulation of {words} or to the shape of the distribution is not tested (see "
            "the unusual observations and the rare label)."
        )
        if not self.effects.empty:
            whole = self.effects[self.effects["sampling"] == WHOLE_FOVS]["resampled"]
            resampled = whole.dropna().unique().tolist()
            if resampled:
                partly = (
                    "; where a side has fewer than two, they resample its fields only"
                    if whole.isna().any()
                    else ""
                )
                lines.append(
                    "  The whole-field draws take the fields of technical repeats "
                    f"({listing(resampled)}) drawn with replacement, so their same-sign share "
                    f"includes well-to-well variation{partly}; the random-object draws treat "
                    f"{words} as independent draws, so their same-sign share is not a "
                    "confidence in the effect."
                )
            else:
                lines.append(
                    "  The draws resample these fields and objects, not wells or repeats: a "
                    "same-sign share describes these fields and is not a confidence in the "
                    "effect."
                )
        return lines

    def _test(self, row: dict[str, Any], n_contrasts: int) -> str:
        """The randomization-test sentence of one contrast, or why it has none."""
        if row["status"] != "ok":
            return f"No randomization test: {row['status']}."
        if row["permutation_unit"] == FIELD:
            shuffled = (
                "whole fields, treated as independent draws as declared (fields_independent; "
                "if they came from a few wells or dishes, the p-value tests those wells, not "
                "the condition)"
            )
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
        return (
            f"Randomization test over {shuffled}, {how}: p = {row['p_value']:.3g} (Bonferroni "
            f"over {count(n_contrasts, 'comparison')}: {row['p_bonferroni']:.3g})."
        )

    def _effects_line(self, row: dict[str, Any], words: str) -> str:
        """The JCS effect sizes of one contrast, from every object and from draws."""
        full = row["effect_full"]
        start = (
            "  JCS effect size (treated mean minus control mean, over the control SD) from "
            f"every object: {_number(full)}"
        )
        if self.effects.empty:
            return f"{start}." if np.isfinite(full) else ""
        # Without a repeat column the repeat is None, which == never matches.
        same_repeat = (
            self.effects["repeat"].isna()
            if pd.isna(row["repeat"])
            else self.effects["repeat"] == row["repeat"]
        )
        effects = self.effects[same_repeat & (self.effects["condition"] == row["condition"])]
        if effects.empty:
            return f"{start}." if np.isfinite(full) else ""
        parts = []
        for n, block in effects.groupby("n_objects", sort=True):
            by = block.set_index("sampling")
            bits = []
            for sampling in (RANDOM_OBJECTS, WHOLE_FOVS):
                if sampling not in by.index:
                    continue
                r = by.loc[sampling]
                label = sampling
                if sampling == WHOLE_FOVS and isinstance(r["resampled"], str):
                    label = f"whole FOVs of resampled technical repeats ({r['resampled']})"
                bit = f"{label} median {r['median']:.2f} (IQR {r['q25']:.2f} to {r['q75']:.2f})"
                if not pd.isna(r["same_sign"]):
                    bit += f", same sign in {percent(r['same_sign'])} of draws"
                if sampling == WHOLE_FOVS:
                    bit += f", about {fields_text(r['fields_treated'])}"
                bits.append(bit)
            parts.append(f"from {int(n)} {words}: " + "; ".join(bits))
        return f"{start}; " + ". ".join(parts) + "."


def _number(value: float) -> str:
    """A statistic to two decimals, ``undefined``, or infinite for a perfect separation."""
    if np.isnan(value):
        return "undefined"
    if np.isinf(value):
        return f"{'' if value > 0 else '-'}infinite (no variation within either side)"
    return f"{value:.2f}"


def _unit_sample(
    units: list[list[np.ndarray]], n: int, rng: np.random.Generator
) -> tuple[np.ndarray, int]:
    """``n`` objects from whole technical repeats drawn with replacement, and the fields taken.

    A unit is drawn at random (with replacement, as from a population of wells
    like these), its fields are taken whole in random order, with a random
    part of the last one, until ``n`` objects; fields without observations
    count as acquired.
    """
    filled = [u for u in units if sum(len(f) for f in u)]
    taken, total, fields = [], 0, 0
    while total < n:
        unit = filled[int(rng.integers(len(filled)))]
        for i in rng.permutation(len(unit)):
            field, need = unit[i], n - total
            fields += 1
            if len(field) >= need:
                taken.append(rng.choice(field, size=need, replace=False))
                total = n
                break
            taken.append(field)
            total += len(field)
    return np.concatenate(taken), fields


def _effect_rows(
    base: dict[str, Any],
    t_units: list[list[np.ndarray]],
    c_units: list[list[np.ndarray]],
    resampled: str | None,
    counts: list[int],
    n_draws: int,
    full: float,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    """The JCS effect size at each sample size, with random objects and with whole fields.

    Whole fields are drawn from every acquired field, those without
    observations included, as a microscope stopping at ``n`` objects acquires
    them too; so ``fields_treated`` counts acquired fields. With
    ``resampled`` (the technical-repeat column, when both sides hold at least
    two) the whole-field draws take the fields of technical repeats drawn with
    replacement (``_unit_sample``), so their spread includes the well-to-well
    variation.
    """
    t_fields = [f for u in t_units for f in u]
    c_fields = [f for u in c_units for f in u]
    t_objects, c_objects = np.concatenate(t_fields), np.concatenate(c_fields)
    n_max = min(len(t_objects), len(c_objects))
    rows = []
    for n in [n for n in counts if n <= n_max]:
        for sampling in (RANDOM_OBJECTS, WHOLE_FOVS):
            draws, used = np.empty(n_draws), np.empty(n_draws)
            for i in range(n_draws):
                if sampling == RANDOM_OBJECTS:
                    t = rng.choice(t_objects, size=n, replace=False)
                    c = rng.choice(c_objects, size=n, replace=False)
                elif resampled:
                    t, used[i] = _unit_sample(t_units, n, rng)
                    c, _ = _unit_sample(c_units, n, rng)
                else:
                    t = field_sample(t_fields, n, rng)
                    c = field_sample(c_fields, n, rng)
                draws[i] = jcs_effect_size(t, c)
            defined = draws[np.isfinite(draws)]
            fields_treated = np.nan
            if sampling == WHOLE_FOVS:
                fields_treated = (
                    float(used.mean()) if resampled else fields_holding(t_fields, n, rng)
                )
            rows.append(
                {
                    **base,
                    "sampling": sampling,
                    "resampled": resampled if sampling == WHOLE_FOVS else None,
                    "n_objects": n,
                    "median": float(np.median(defined)) if len(defined) else np.nan,
                    "q25": float(np.percentile(defined, 25)) if len(defined) else np.nan,
                    "q75": float(np.percentile(defined, 75)) if len(defined) else np.nan,
                    "same_sign": (
                        float(np.mean(np.sign(defined) == np.sign(full)))
                        if len(defined) and np.isfinite(full) and full != 0
                        else np.nan
                    ),
                    "fields_treated": fields_treated,
                    "draws": len(defined),
                }
            )
    return rows


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
    fields_independent: bool = False,
) -> ConditionContrast:
    """Each condition against the control condition, per repeat, with objects and with fields.

    ``control`` names the control condition by FOV-table value, for example
    ``{"condition": "MOCK"}``; its key is the condition column unless
    ``condition_col`` says otherwise. ``repeat_col`` separates the repeats
    (``None`` compares within the whole table). Every analysis group whose
    condition is not the control's is compared with the control groups of
    the same repeat; the groups of one condition (its wells, say) are pooled
    for the comparison, which the summary states. ``technical_col`` names the
    technical-repeat column of the FOV table (by default the MorphoBricks
    ``technical_repeat`` column, when present); its technical repeats are
    compared and shuffled as units whatever ``group_cols`` says, an analysis
    group holding several being split by it (see the module). Without it, a
    group column nested within condition x repeat (a ``well`` column in
    ``group_cols``) is the unit; without either, there is no randomization
    test unless ``fields_independent=True`` declares the fields independent
    draws, when whole fields are shuffled. ``object_counts`` are the JCS
    sample sizes; counts above what a side holds are left out.
    """
    if not isinstance(feature, str) or not feature:
        raise TypeError("feature must be one column name")
    n_permutations, random_state = check_arguments(
        control, condition_col, n_permutations, random_state, fields_independent
    )
    n_draws = as_count(n_draws, "n_draws", 2)
    counts = sorted({as_count(n, "object_counts", 2) for n in object_counts})
    data = validated(objects, fovs, [feature], fov_col, group_cols, observation_unit)
    repeat_col, technical_col = resolve_columns(
        data, condition_col, repeat_col, technical_col, control
    )
    all_groups = groups(data)
    described = describe_groups(all_groups, condition_col, repeat_col, control)
    unit, unit_note = resolve_unit(data, described, condition_col, repeat_col, technical_col)
    settings = {
        "feature": feature,
        "control": dict(control),
        "condition_col": condition_col,
        "repeat_col": repeat_col,
        "technical_col": technical_col,
        "unit": unit,
        "unit_note": unit_note,
        "fields_independent": bool(fields_independent),
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
        base = {
            "repeat": repeat,
            "condition": condition,
            "control_condition": control_label(described, controls) or str(control[condition_col]),
        }
        if not controls:
            rows.append({**base, **_NO_TEST, "status": f"{_NO_CONTROL} {repeat_col}={repeat}"})
            continue
        t_units, t_acquired = side_units(all_groups, feature, treated, technical_col)
        c_units, c_acquired = side_units(all_groups, feature, controls, technical_col)
        t_fields = [f for u in t_units for f in u]
        c_fields = [f for u in c_units for f in u]
        t_objects, c_objects = np.concatenate(t_fields), np.concatenate(c_fields)
        rng = group_rng(random_state, base)
        test = randomization_test(
            t_units, c_units, unit, n_permutations, rng, bool(fields_independent)
        )
        by_unit = min(test["technical"], test["control_technical"]) >= 2
        row = {
            **base,
            "fields": t_acquired,
            "observations": len(t_objects),
            "control_fields": c_acquired,
            "control_observations": len(c_objects),
            **test,
            "d_objects": cohen_d(t_objects, c_objects),
            "d_technical": (
                separation_d(unit_means(t_units), unit_means(c_units)) if by_unit else np.nan
            ),
            "p_bonferroni": np.nan,
            "effect_full": jcs_effect_size(t_objects, c_objects),
        }
        rows.append(row)
        # The effect sizes do not depend on the test: computed for every contrast.
        resampled = unit if by_unit else None
        effect_rows += _effect_rows(
            base, t_units, c_units, resampled, counts, n_draws, row["effect_full"], rng
        )
    table = pd.DataFrame(rows)[_COLUMNS]
    tested = table["p_value"].notna()
    table["p_bonferroni"] = np.where(
        tested, np.minimum(1.0, table["p_value"] * int(tested.sum())), np.nan
    )
    return ConditionContrast(table, pd.DataFrame(effect_rows), settings)


_COLUMNS = [
    "repeat",
    "condition",
    "control_condition",
    "fields",
    "observations",
    "control_fields",
    "control_observations",
    "technical",
    "control_technical",
    "d_objects",
    "d_fields",
    "d_technical",
    "permutation_unit",
    "distinct_assignments",
    "p_value",
    "p_bonferroni",
    "p_exact",
    "p_floor",
    "effect_full",
    "status",
]
_NO_TEST = {
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
    "p_floor": np.nan,
    "effect_full": np.nan,
}


__all__ = ["ConditionContrast", "condition_contrast"]
