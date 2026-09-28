"""Which features separate the conditions? A gated ranking to choose what to plot.

With many declared features the report cannot plot them all, so it ranks
them the way fiberfeatures scores a feature set (its ``gated_effect_size``):
every feature is tested for every condition against the declared control,
within each repeat, with the same statistics as question 9
(``condition_contrast``): Cohen's d on field means and a randomization test
over whole technical repeats (the FOV table's technical-repeat column,
whatever ``group_cols`` says, or else a group column nested within
condition x repeat, such as ``well``) when each side holds at least two
(exact when every assignment can be enumerated; a side with a single
technical repeat is not tested, and the ranking does not gate on it).
Without such a unit there is no test, since fields of one dish or well are
not independent draws (with three unrecorded wells a side and no effect, a
null feature was ranked in 95% of datasets), unless ``fields_independent``
declares them independent: whole fields are then shuffled. The p-values of all
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
5% gate; the summary then says so rather than ranking nothing silently. The
same holds for a test over very few fields (3 a side: a floor of 0.1). A
floor below the gate can still be out of reach after the correction: over
``m`` tests a lone contrast at floor ``p`` gets ``q = m p``, and the summary
gives that q and how many tests must reach the floor for one to pass. It
also flags a feature whose passing contrasts for one condition have opposite
signs in different repeats (a shift between repeats or wells, not a
consistent effect).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .._caveats import assumption, caveat_lines
from .._checks import is_number
from .._seeds import group_rng
from .._style import check_detail, table, wrap
from .._wording import count
from ..metadata import MORPHOBRICKS_GROUP_COLUMNS, TECHNICAL_REPEAT_COLUMN
from ..statistics import benjamini_hochberg
from ._common import groups, validated
from ._units import (
    FIELD,
    NOT_INDEPENDENT,
    UNKNOWN_WELLS_ADVICE,
    check_arguments,
    contrasts_of,
    control_label,
    describe_groups,
    opposite_signs,
    randomization_test,
    resolve_columns,
    resolve_unit,
    side_units,
    sides,
)

_UNITS_OUT = ", so those technical repeats cannot pass the gate however large the difference"
_FIELDS_OUT = (
    ", above the gate of {alpha:g}, so those tests cannot pass it however large the difference"
)
_UNKNOWN_WELLS = "the fields' wells or dishes are unknown"
EFFECT_CAP = 20.0
GATE_ALPHA = 0.05
TOP_FEATURES = 5
MAX_ROWS_LISTED = 24
_PLOTTED = f"Plotted by default: the readout and up to {TOP_FEATURES} best-ranked other features."


@dataclass(frozen=True)
class FeatureRanking:
    """The gated tests and the ranking they give.

    - ``tests``: one row per feature, repeat and condition: the sides'
      acquired fields (``fields``, ``control_fields``, as in
      ``condition_contrast``) and technical repeats, ``d_fields``, the
      ``permutation_unit`` (the technical-repeat column or ``field``) with its
      ``distinct_assignments``, ``p_value`` (``p_exact`` when every assignment
      was enumerated; ``p_floor`` the smallest it can be), ``q_value``,
      ``passes_gate`` and ``status``.
    - ``ranking``: one row per feature, best first: ``score`` (the capped |d|
      of the best passing contrast; NaN when none passes), the contrast it
      comes from (``repeat``, ``condition``, ``q_value``), ``passing`` and
      ``tests`` counts.
    """

    tests: pd.DataFrame
    ranking: pd.DataFrame
    settings: dict[str, Any]

    def _floors(self) -> list[str]:
        """Sentences on the smallest p-value the tests can give, where it limits the gate.

        Tests over whole technical repeats are always described, since their
        few assignments set the floor; tests over whole fields only when the
        floor limits the gate. A floor below the gate can still be out of
        reach once corrected: Benjamini-Hochberg over ``m`` tests gives a lone
        contrast at floor ``p`` the q-value ``m p``, and ``k`` contrasts at the
        floor ``m p / k``, so the sentence gives the smallest q a lone contrast
        can reach and how many tests must reach the floor for one to pass.
        """
        alpha = self.settings["alpha"]
        if self.tests.empty:
            return []
        tested = self.tests[self.tests["p_value"].notna()]
        m = len(tested)
        lines = []
        by_unit = tested[tested["permutation_unit"] != FIELD]
        if len(by_unit):
            worst = by_unit.loc[by_unit["p_floor"].idxmax()]
            floor = float(worst["p_floor"])
            lines.append(
                f"  {len(by_unit)} of the tests shuffled whole technical repeats; with as few as "
                f"{int(worst['distinct_assignments'])} distinct assignments a "
                f"p-value cannot fall below {floor:.2g}{_consequence(floor, m, alpha, _UNITS_OUT)}."
            )
        by_field = tested[(tested["permutation_unit"] == FIELD) & (tested["p_floor"] * m > alpha)]
        if len(by_field):
            floor = float(by_field["p_floor"].max())
            lines.append(
                f"  {len(by_field)} of the tests shuffled so few whole fields that a p-value "
                f"cannot fall below {floor:.2g}"
                f"{_consequence(floor, m, alpha, _FIELDS_OUT.format(alpha=alpha))}."
            )
        return lines

    def top(self, k: int = TOP_FEATURES) -> list[str]:
        """The ``k`` best-ranked features that pass the gate, best first."""
        ranked = self.ranking.dropna(subset=["score"])
        return ranked["feature"].head(k).tolist()

    def _units(self) -> str:
        """What the tests shuffle: technical repeats, or fields (declared independent)."""
        if self.settings.get("unit"):
            return f"technical repeats ({self.settings['unit']})"
        if not self.settings.get("fields_independent"):
            return "fields, where the fields are declared independent draws,"
        return "fields"

    def _head(self) -> str:
        """How the ranking was made, in one sentence (the full form's first line)."""
        control, alpha = self.settings["control"], self.settings["alpha"]
        n_tests = int(self.tests["p_value"].notna().sum()) if len(self.tests) else 0
        repeat_col = self.settings["repeat_col"]
        return (
            f"Which features separate the conditions? Each of the "
            f"{self.settings['n_features']} declared features, every condition against the "
            f"control {control}"
            + (f" within each {repeat_col}" if repeat_col else "")
            + f": Cohen's d on field means and a randomization test over whole {self._units()} "
            "as in the contrast above, "
            + (
                f"Benjamini-Hochberg corrected over the {count(n_tests, 'test')}"
                if n_tests
                else "though no contrast could be tested"
            )
            + f"; a test passes the gate when q <= {alpha:g} (fiberfeatures' "
            f"gated effect size). Features are ranked by the largest |d| among their passing "
            f"contrasts (|d| above {EFFECT_CAP:g} counts as {EFFECT_CAP:g})."
        )

    def _short_head(self) -> str:
        """The short form's first line: how many features pass, and how they were tested."""
        n_tests = int(self.tests["p_value"].notna().sum()) if len(self.tests) else 0
        n = self.settings["n_features"]
        if not n_tests:
            return (
                f"Which features separate the conditions? None is ranked: for the {n} declared "
                "features no contrast could be tested."
            )
        passing = int(self.ranking["score"].notna().sum())
        label = " · ".join(str(v) for v in self.settings["control"].values())
        return (
            f"Which features separate the conditions? {passing} of {count(n, 'feature')} pass "
            f"the gate (q <= {self.settings['alpha']:g}, Benjamini-Hochberg over "
            f"{count(n_tests, 'test')}: Cohen's d on field means, each condition against the "
            f"control ({label}), and a randomization test over whole {self._units()}), ranked "
            f"by |d| (capped at {EFFECT_CAP:g}):"
        )

    def _table(self, max_rows: int, indent: int) -> list[str]:
        """The ranked features, best first, as a table, and how many more are ranked."""
        ranked = self.ranking.dropna(subset=["score"])
        if ranked.empty:
            return []
        rows = []
        for rank, row in enumerate(ranked.head(max_rows).to_dict("records"), start=1):
            where = f"{row['condition']} vs {row['control_condition']}"
            if self.settings["repeat_col"]:
                where += f" · {row['repeat']}"
            passes = f"{int(row['passing'])} of {int(row['tests'])}"
            rows.append([rank, row["feature"], f"{row['score']:.2f}", where, _q(row), passes])
        columns = ["rank", "feature", "|d|", "best contrast", "q", "passes in"]
        lines = [table(rows, columns, indent=indent, align="rlrllr")]
        if len(ranked) > max_rows:
            more = f"and {len(ranked) - max_rows} more ranked features"
            lines.append(" " * indent + more)
        return lines

    def _unranked(self, short: bool) -> list[str]:
        """The features tested that failed the gate, and those that could not be tested."""
        limit = 3 if short else MAX_ROWS_LISTED
        unranked = self.ranking[self.ranking["score"].isna()] if len(self.ranking) else []
        if not len(unranked):
            return []
        lines = []
        # Tested features that failed the gate are not features that were never tested.
        failing = unranked[unranked["tests"] > 0]["feature"].tolist()
        if failing:
            lines.append(
                f"No contrast passes the gate for {count(len(failing), 'feature')}: "
                f"{_listed(failing, limit)}."
            )
        untested = unranked[unranked["tests"] == 0]["feature"].tolist()
        if untested:
            reasons = []
            for feature in untested:
                statuses = self.tests.loc[self.tests["feature"] == feature, "status"]
                # The long independence reason is explained once, with the caveats.
                shown = [_UNKNOWN_WELLS if st == NOT_INDEPENDENT else st for st in statuses]
                reasons.append(f"{feature} ({'; '.join(dict.fromkeys(shown))})")
            lines.append(
                f"Not ranked because no contrast could be tested "
                f"({count(len(untested), 'feature')}): {_listed(reasons, limit)}."
            )
        return lines

    def _cautions(self) -> list[str]:
        """What limits the ranking, each once: p-value floors, opposite signs, units."""
        cautions = [line.strip() for line in self._floors()]
        cautions += opposite_signs(
            self.tests[self.tests["passes_gate"]],
            ["feature", "condition"],
            self.settings["repeat_col"],
            lambda key: f"{key[0]} passes the gate for {key[1]}",
        )
        if (self.tests["status"] == NOT_INDEPENDENT).any():
            cautions.append(UNKNOWN_WELLS_ADVICE)
        elif (
            self.settings.get("fields_independent")
            and ((self.tests["permutation_unit"] == FIELD) & self.tests["p_value"].notna()).any()
        ):
            cautions.append(
                assumption(
                    "Tests over whole fields treat the fields as independent draws, as declared "
                    "(fields_independent); if they came from a few wells or dishes, those tests "
                    "test the wells, not the condition."
                )
            )
        note = self.settings.get("unit_note")
        if note:
            cautions.append(f"Note: {note}.")
        return cautions

    def short_parts(
        self, indent: int = 0, max_rows: int = TOP_FEATURES
    ) -> tuple[list[str], list[str]]:
        """The short form under ``indent``: how many pass, the ``max_rows`` best-ranked
        features in a table, the unranked ones in a line each; and the caveats."""
        lines = [wrap(self._short_head(), indent, hang=indent + 2)]
        if self.tests.empty:
            reason = self.settings.get("reason", "No condition to compare with the control.")
            return [*lines, wrap(reason, indent, hang=indent + 2)], []
        lines += self._table(max_rows, indent + 2)
        lines += [wrap(line, indent, hang=indent + 2) for line in self._unranked(short=True)]
        if self.ranking["score"].notna().any():
            lines.append(" " * indent + _PLOTTED)
        return lines, self._cautions()

    def summary(self, max_rows: int = MAX_ROWS_LISTED, detail: str = "short") -> str:
        """The ranking: the best-ranked features in a table and each caveat once
        (``detail="short"``, at most ``TOP_FEATURES`` rows), or how it was made, up to
        ``max_rows`` ranked features and every unranked one (``"full"``)."""
        if check_detail(detail) == "short":
            lines, cautions = self.short_parts(max_rows=min(max_rows, TOP_FEATURES))
            return "\n".join([*lines, *caveat_lines(cautions, indent=0)])
        lines = [wrap(self._head(), 0, hang=2)]
        if self.tests.empty:
            lines.append(self.settings.get("reason", "No condition to compare with the control."))
            return "\n".join(lines)
        lines += self._table(max_rows, 2)
        lines += [wrap(line, 2, hang=4) for line in self._unranked(short=False)]
        lines += caveat_lines(self._cautions(), indent=2, detail="full")
        if self.ranking["score"].notna().any():
            lines.append(f"  {_PLOTTED}")
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
    fields_independent: bool = False,
) -> FeatureRanking:
    """Rank ``features`` by how confidently they separate each condition from the control.

    Arguments follow ``condition_contrast``: ``control`` names the control
    condition by FOV-table value, ``repeat_col`` separates the repeats,
    ``technical_col`` names the FOV-table column of the technical repeats
    (the units whatever ``group_cols`` says; without it a nested group
    column), ``fields_independent`` declares the fields independent draws
    when no unit exists, ``alpha`` is the Benjamini-Hochberg gate. See the
    module.
    """
    features = list(dict.fromkeys(features))
    if not features:
        raise ValueError("features must name at least one column")
    n_permutations, random_state = check_arguments(
        control, condition_col, n_permutations, random_state, fields_independent
    )
    if not is_number(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    data = validated(objects, fovs, features, fov_col, group_cols, observation_unit)
    repeat_col, technical_col = resolve_columns(
        data, condition_col, repeat_col, technical_col, control
    )
    all_groups = groups(data)
    described = describe_groups(all_groups, condition_col, repeat_col, control)
    unit, unit_note = resolve_unit(data, described, condition_col, repeat_col, technical_col)
    settings: dict[str, Any] = {
        "features": features,
        "n_features": len(features),
        "control": dict(control),
        "condition_col": condition_col,
        "repeat_col": repeat_col,
        "technical_col": technical_col,
        "unit": unit,
        "unit_note": unit_note,
        "fields_independent": bool(fields_independent),
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
        base = {
            "repeat": repeat,
            "condition": condition,
            "control_condition": control_label(described, controls),
        }
        for feature in features:
            row: dict[str, Any] = {"feature": feature, **base}
            if not controls:
                rows.append(
                    {**row, **_EMPTY, "status": f"no control group in {repeat_col}={repeat}"}
                )
                continue
            t_units, t_acquired = side_units(all_groups, feature, treated, technical_col)
            c_units, c_acquired = side_units(all_groups, feature, controls, technical_col)
            rng = group_rng(random_state, {**base, "feature": feature})
            # Acquired fields, as condition_contrast counts them.
            row.update({"fields": t_acquired, "control_fields": c_acquired})
            row.update(
                randomization_test(
                    t_units, c_units, unit, n_permutations, rng, bool(fields_independent)
                )
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
    "p_floor": np.nan,
}


def _consequence(floor: float, m: int, alpha: float, out_of_reach: str) -> str:
    """Why tests at ``floor`` cannot pass the gate (``out_of_reach`` when the floor itself
    lies above it), or pass only in numbers after the correction (empty otherwise)."""
    if floor > alpha:
        return out_of_reach
    lone = floor * m
    if lone <= alpha:
        return ""
    needed = math.ceil(lone / alpha - 1e-9)
    return (
        f"; corrected over the {m} tests, a lone contrast at that floor gets q = "
        f"{min(lone, 1.0):.2g}, above the gate of {alpha:g}, so a lone contrast at the floor "
        f"cannot pass: one at the floor passes only if at least {needed} of the {m} tests "
        "reach it"
    )


def _listed(names: list[str], limit: int = MAX_ROWS_LISTED) -> str:
    """Names joined by commas, the first ``limit`` and how many more."""
    shown = ", ".join(names[:limit])
    more = len(names) - limit
    return f"{shown} and {more} more" if more > 0 else shown


def _q(row: Mapping[str, Any]) -> str:
    """The q-value of a ranked feature's best contrast, to three significant digits."""
    return f"{row['q_value']:.3g}"


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
