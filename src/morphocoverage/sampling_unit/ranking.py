"""Which features separate the conditions? A gated ranking to choose what to plot.

With many declared features the report cannot plot them all, so it ranks
them the way fiberfeatures scores a feature set (its ``gated_effect_size``):
every feature is tested for every condition against the declared control,
within each repeat, with the same statistics as question 9
(``condition_contrast``): Cohen's d on field means and a randomization test
over whole technical repeats when each side holds at least two, else,
without the technical-repeat column, over whole fields (exact when every
assignment can be enumerated; a side with a single technical repeat is not
tested, and the ranking does not gate on it). The p-values of all
feature-and-contrast tests are
Benjamini-Hochberg corrected together; a test passes the gate when its
q-value is at most ``alpha``. A feature's score is the largest |d| among its
passing contrasts (|d| above ``EFFECT_CAP`` counts as the cap, so a perfectly
separated feature ranks first rather than as undefined); a feature with no
passing contrast is unranked, and the summary tells a feature that was tested
and failed from one that could not be tested. The report lists the ranking under question 9
and the plots draw the readout and the ``TOP_FEATURES`` best-ranked features.

The gate rejects chance separations; |d| then orders the survivors even when
many p-values sit at their floor. With few technical repeats only a few
distinct label assignments exist and the p-value cannot fall below their
share (twice their share with as many on each side, since the sides swapped
give the same |d|), so with three technical repeats a side no test passes a
5% gate; the summary then says so rather than ranking nothing silently.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import comb
from typing import Any

import numpy as np
import pandas as pd

from .._seeds import group_identity
from .._wording import count
from ..metadata import MORPHOBRICKS_GROUP_COLUMNS, TECHNICAL_REPEAT_COLUMN
from ..statistics import benjamini_hochberg
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
    smallest_p,
    technical_counts,
    untested_reason,
)

EFFECT_CAP = 20.0
GATE_ALPHA = 0.05
TOP_FEATURES = 5
MAX_ROWS_LISTED = 24


@dataclass(frozen=True)
class FeatureRanking:
    """The gated tests and the ranking they give.

    - ``tests``: one row per feature, repeat and condition: the sides'
      acquired fields (``fields``, ``control_fields``, as in
      ``condition_contrast``) and technical repeats, ``d_fields``, the
      ``permutation_unit`` (the technical-repeat column or ``field``) with its
      ``distinct_assignments``, ``p_value`` (``p_exact`` when every assignment
      was enumerated), ``q_value``, ``passes_gate`` and ``status``.
    - ``ranking``: one row per feature, best first: ``score`` (the capped |d|
      of the best passing contrast; NaN when none passes), the contrast it
      comes from (``repeat``, ``condition``, ``q_value``), ``passing`` and
      ``tests`` counts.
    """

    tests: pd.DataFrame
    ranking: pd.DataFrame
    settings: dict[str, Any]

    def top(self, k: int = TOP_FEATURES) -> list[str]:
        """The ``k`` best-ranked features that pass the gate, best first."""
        ranked = self.ranking.dropna(subset=["score"])
        return ranked["feature"].head(k).tolist()

    def summary(self, max_rows: int = MAX_ROWS_LISTED) -> str:
        """The ranking in sentences: how it was made, the ranked features, the unranked ones."""
        control = self.settings["control"]
        alpha = self.settings["alpha"]
        n_tests = int(self.tests["p_value"].notna().sum()) if len(self.tests) else 0
        units = "fields"
        if self.settings["technical_col"]:
            units = f"technical repeats ({self.settings['technical_col']}) or fields"
        head = (
            f"Which features separate the conditions? Each of the "
            f"{self.settings['n_features']} declared features, every condition against the "
            f"control {control}"
            + (f" within each {self.settings['repeat_col']}" if self.settings["repeat_col"] else "")
            + f": Cohen's d on field means and a randomization test over whole {units} as in "
            f"the contrast above, Benjamini-Hochberg corrected over "
            f"the {n_tests} tests; a test passes the gate when q <= {alpha:g} (fiberfeatures' "
            f"gated effect size). Features are ranked by the largest |d| among their passing "
            f"contrasts (|d| above {EFFECT_CAP:g} counts as {EFFECT_CAP:g})."
        )
        lines = [head]
        if self.tests.empty:
            lines.append(self.settings.get("reason", "No condition to compare with the control."))
            return "\n".join(lines)
        ranked = self.ranking.dropna(subset=["score"])
        for rank, row in enumerate(ranked.head(max_rows).to_dict("records"), start=1):
            where = f"{row['condition']} vs {row['control_condition']}"
            if self.settings["repeat_col"]:
                where += f", {self.settings['repeat_col']}={row['repeat']}"
            lines.append(
                f"  {rank}. {row['feature']}: |d| {row['score']:.2f} ({where}; q = "
                f"{row['q_value']:.3g}); passes in {int(row['passing'])} of {int(row['tests'])} "
                f"contrasts"
            )
        if len(ranked) > max_rows:
            lines.append(f"  and {len(ranked) - max_rows} more ranked features")
        unranked = self.ranking[self.ranking["score"].isna()]
        # Tested features that failed the gate are not features that were never tested.
        failing = unranked[unranked["tests"] > 0]["feature"].tolist()
        if failing:
            lines.append(
                f"  No contrast passes the gate for {count(len(failing), 'feature')}: "
                f"{_listed(failing)}."
            )
        untested = unranked[unranked["tests"] == 0]["feature"].tolist()
        if untested:
            reasons = []
            for feature in untested:
                statuses = self.tests.loc[self.tests["feature"] == feature, "status"]
                reasons.append(f"{feature} ({'; '.join(dict.fromkeys(statuses))})")
            lines.append(
                f"  Not ranked because no contrast could be tested "
                f"({count(len(untested), 'feature')}): {_listed(reasons)}."
            )
        tested = self.tests[self.tests["p_value"].notna()]
        by_unit = tested[tested["permutation_unit"] != FIELD]
        if len(by_unit):
            floors = [
                smallest_p(int(t), int(c))
                for t, c in zip(by_unit["technical"], by_unit["control_technical"], strict=True)
            ]
            worst = int(np.argmax(floors))
            floor = floors[worst]
            # Only a floor above the gate makes a test unable to pass it.
            consequence = (
                ", so those technical repeats cannot pass the gate however large the difference"
                if floor > alpha
                else ""
            )
            lines.append(
                f"  {len(by_unit)} of the tests shuffled whole technical repeats; with as few as "
                f"{int(by_unit['distinct_assignments'].iloc[worst])} distinct assignments a "
                f"p-value cannot fall below {floor:.2g}{consequence}."
            )
        if not ranked.empty:
            lines.append(
                f"  Plotted by default: the readout and up to {TOP_FEATURES} best-ranked other "
                "features."
            )
        return "\n".join(lines)


def feature_ranking(
    objects: pd.DataFrame,
    fovs: pd.DataFrame,
    *,
    features: Sequence[str],
    control: Mapping[str, Any],
    condition_col: str = MORPHOBRICKS_GROUP_COLUMNS[0],
    repeat_col: str | None = MORPHOBRICKS_GROUP_COLUMNS[1],
    technical_col: str | None = TECHNICAL_REPEAT_COLUMN,
    n_permutations: int = 2000,
    alpha: float = GATE_ALPHA,
    random_state: int = 0,
    fov_col: str | None = None,
    group_cols: Sequence[str] | None = None,
    observation_unit: str | None = None,
) -> FeatureRanking:
    """Rank ``features`` by how confidently they separate each condition from the control.

    Arguments follow ``condition_contrast``: ``control`` names the control
    condition by FOV-table value, ``repeat_col`` separates the repeats,
    ``technical_col`` names the technical repeats when it is an analysis-group
    column, ``alpha`` is the Benjamini-Hochberg gate. See the module.
    """
    features = list(dict.fromkeys(features))
    if not features:
        raise ValueError("features must name at least one column")
    if not isinstance(control, Mapping) or not control:
        raise ValueError("control must be a non-empty mapping of FOV-table column to value")
    if condition_col not in control:
        raise ValueError(f"control must name the condition column {condition_col!r}")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    check_count(n_permutations, "n_permutations", 1)
    check_seed(random_state)
    data = validated(objects, fovs, features, fov_col, group_cols, observation_unit)
    repeat_col, technical_col = resolve_columns(
        data, condition_col, repeat_col, technical_col, control
    )
    all_groups = groups(data)
    described = describe_groups(all_groups, condition_col, repeat_col, control)
    settings: dict[str, Any] = {
        "features": features,
        "n_features": len(features),
        "control": dict(control),
        "condition_col": condition_col,
        "repeat_col": repeat_col,
        "technical_col": technical_col,
        "n_permutations": n_permutations,
        "alpha": alpha,
        "random_state": random_state,
        "observation_unit": data.observation_unit,
    }
    contrasts = contrasts_of(described)
    if not contrasts:
        settings["reason"] = (
            f"Every analysis group is the control {dict(control)}: no other condition to compare."
        )
        return FeatureRanking(pd.DataFrame(), pd.DataFrame(), settings)
    rows = []
    for repeat, condition in contrasts:
        treated, controls = sides(described, repeat, condition)
        control_label = "/".join(
            sorted({str(d["condition"]) for d in described if d["group"] in controls})
        )
        base = {"repeat": repeat, "condition": condition, "control_condition": control_label}
        for feature in features:
            row: dict[str, Any] = {"feature": feature, **base}
            if not controls:
                rows.append(
                    {**row, **_EMPTY, "status": f"no control group in {repeat_col}={repeat}"}
                )
                continue
            t_units, t_acquired = side_units(all_groups, feature, treated)
            c_units, c_acquired = side_units(all_groups, feature, controls)
            n_technical = technical_counts(t_units, c_units, technical_col)
            by_unit = min(n_technical) >= 2
            t_means, c_means = field_means(t_units), field_means(c_units)
            row.update(
                {
                    # Acquired fields, as condition_contrast counts them.
                    "fields": t_acquired,
                    "control_fields": c_acquired,
                    "technical": n_technical[0],
                    "control_technical": n_technical[1],
                    # A perfect separation is an infinite d, capped when ranked.
                    "d_fields": separation_d(t_means, c_means),
                    "permutation_unit": technical_col if by_unit else FIELD,
                    "distinct_assignments": (
                        comb(sum(n_technical), n_technical[0]) if by_unit else np.nan
                    ),
                    "p_value": np.nan,
                    "p_exact": False,
                    "status": "ok",
                }
            )
            reason = untested_reason((len(t_means), len(c_means)), n_technical, technical_col)
            if reason is not None:
                row["status"] = reason
                rows.append(row)
                continue
            if np.isnan(row["d_fields"]):
                row["status"] = "the field means do not vary, so there is nothing to test"
                rows.append(row)
                continue
            rng = np.random.default_rng(
                [random_state, group_identity({**base, "feature": feature})]
            )
            row["p_exact"] = relabellings(t_units, c_units, by_unit) <= n_permutations
            row["p_value"] = permutation_p(
                t_units, c_units, by_unit, abs(row["d_fields"]), n_permutations, rng
            )
            rows.append(row)
    tests = pd.DataFrame(rows)
    tests["q_value"] = benjamini_hochberg(tests["p_value"].to_numpy(dtype=float))
    tests["passes_gate"] = (tests["q_value"] <= alpha).fillna(False) & tests["d_fields"].notna()
    ranking = _rank(tests, features)
    return FeatureRanking(tests, ranking, settings)


_EMPTY = {
    "fields": 0,
    "control_fields": 0,
    "technical": 0,
    "control_technical": 0,
    "d_fields": np.nan,
    "permutation_unit": FIELD,
    "distinct_assignments": np.nan,
    "p_value": np.nan,
    "p_exact": False,
}


def _listed(names: list[str]) -> str:
    """Names joined by commas, the first ``MAX_ROWS_LISTED`` and how many more."""
    shown = ", ".join(names[:MAX_ROWS_LISTED])
    more = len(names) - MAX_ROWS_LISTED
    return f"{shown} and {more} more" if more > 0 else shown


def _rank(tests: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """One row per feature: the capped |d| of its best passing contrast, best first."""
    capped = np.minimum(tests["d_fields"].abs(), EFFECT_CAP)
    scored = tests.assign(score=capped.where(tests["passes_gate"]))
    rows = []
    for feature in features:
        block = scored[scored["feature"] == feature]
        passing = block.dropna(subset=["score"]).sort_values("score", ascending=False)
        best = passing.iloc[0] if len(passing) else None
        rows.append(
            {
                "feature": feature,
                "score": float(best["score"]) if best is not None else np.nan,
                "repeat": best["repeat"] if best is not None else None,
                "condition": best["condition"] if best is not None else None,
                "control_condition": best["control_condition"] if best is not None else None,
                "q_value": float(best["q_value"]) if best is not None else np.nan,
                "passing": int(block["passes_gate"].sum()),
                "tests": int(block["p_value"].notna().sum()),
            }
        )
    ranking = pd.DataFrame(rows)
    return ranking.sort_values(
        ["score", "feature"], ascending=[False, True], na_position="last", kind="stable"
    ).reset_index(drop=True)


__all__ = ["EFFECT_CAP", "GATE_ALPHA", "TOP_FEATURES", "FeatureRanking", "feature_ranking"]
