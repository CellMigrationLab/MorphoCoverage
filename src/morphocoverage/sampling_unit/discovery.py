"""Which features separate the conditions? Discovery by the cell-level test, the paired test
beside it.

With many features the report cannot plot them all, and with one well per condition and
repeat the gated ranking over whole technical repeats (``feature_ranking``) has nothing to
shuffle. ``feature_discovery`` ranks the features the way the lab's Plot&Stats notebook
looks for them, as the owner decided (2026-10-01): every feature, every condition against
the declared control, pooled over the biological repeats:

- **The gate (discovery)**: the cell-level randomization test (``contrast_cells``) in the
  version the readings trust most, on values normalised to each repeat's control when
  every control mean is positive and balanced over the repeats when they are not (one
  permutation test with labels shuffled within each repeat, owner, 2026-10-05); its
  p-values are Benjamini-Hochberg corrected over every feature and condition of a
  sub-experiment (``_experiments``), and a test passes when ``q <= alpha`` (a candidate).
  A feature's score is the largest |d| among its passing contrasts (capped at
  ``EFFECT_CAP``); the plots draw the best-ranked features.
- **The second gate (kept beside it)**: the paired t-test on the repeat means (on the log
  ratios to the control when every observation of the feature is positive, else on the
  differences: one scale per feature, the same for every condition, owner 2026-10-06), corrected
  the same way, within each sub-experiment, over every feature and condition with three or
  more repeats. A passing
  feature that also passes it is *significant across repeats*.
- The direction in each repeat (lower or higher than the control), so a feature that
  passes the cell-level gate but changes direction between repeats is seen as such.
- Cohen's d within each repeat (its mean and range) and the fewest and most observations
  on a side of one repeat, which the page's table across repeats and figure legends give.

The cell-level gate treats every observation as an independent draw: it finds candidates,
and when the difference varies between repeats it passes more often than ``alpha`` without
an effect (question 9 gives that chance for the readout). The paired test holds its rate
but needs ``MIN_REPEATS`` repeats and, with three, has little power. A feature with one
value everywhere is not tested, nor one whose condition and control each hold one value
(no Cohen's d).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .._caveats import assumption, caveat_lines
from .._checks import is_number
from .._control import check_comparisons, compared_name, control_name
from .._former_repeat import one_warning, read_former
from .._frozen import owned
from .._parallel import check_n_jobs
from .._seeds import group_rng
from .._style import check_detail, span, table, wrap
from .._wording import (
    change,
    column_word,
    contrasts_words,
    count,
    features_counted,
    listing,
    p_value,
)
from ..metadata import MORPHOBRICKS_GROUP_COLUMNS, TECHNICAL_REPEAT_COLUMN, column_names
from ..schema import observation_words
from ._common import groups, validated
from ._discovery_tests import near_identical, resting, resting_words, tested_rows
from ._experiments import corrected, declared, experiments, family_noun, not_compared, screen_wide
from ._units import (
    check_arguments,
    contrasts_of,
    describe_groups,
    resolve_columns,
)
from .contrast_cells import BALANCED_PERMUTATIONS, refined
from .contrast_repeats import MIN_REPEATS

EFFECT_CAP = 20.0
GATE_ALPHA = 0.05
TOP_FEATURES = 5
MAX_ROWS_LISTED = 24
N_PERMUTATIONS = 9999
# Why no condition of a feature was tested when their reasons differ (each condition's
# reason is its status, _discovery_tests).
UNTESTED = "no condition could be tested"
TEST_COLUMNS = [
    "feature",
    "condition",
    "control_condition",
    "observations",
    "control_observations",
    "repeats",
    "lower",
    "higher",
    "d",
    "p_cells",
    "normalised",
    "balanced",
    "permutations",
    "q_cells",
    "passes",
    "p_paired",
    "paired_values",
    "change",
    "q_paired",
    "p_paired_raw",
    "q_paired_raw",
    "confirmed",
    "status",
    "experiment",
    "d_repeat_mean",
    "d_repeat_min",
    "d_repeat_max",
    "repeat_observations_min",
    "repeat_observations_max",
    "repeat_contrasts",
    "ci_low",
    "ci_high",
    "repeat_sd",
    "family_sd",
    "near_identical",
    "q_screen",
]


# The ranking's columns, also those of an empty ranking when nothing was compared.
RANKING_COLUMNS = [
    "feature",
    "score",
    "d",
    "condition",
    "q_cells",
    "q_paired",
    "change",
    "ci_low",
    "ci_high",
    "repeat_contrasts",
    "repeats",
    "paired_values",
    "confirmed",
    "direction",
    "passing",
    "tests",
]


def _rank(tests: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """One row per feature, best first: the score and the passing contrast it comes from."""
    rows = []
    for order, feature in enumerate(features):
        mine = tests[tests["feature"] == feature]
        passing = mine[mine["passes"]]
        row = {
            "feature": feature,
            "score": np.nan,
            "d": np.nan,
            "condition": None,
            "q_cells": np.nan,
            "q_paired": np.nan,
            "change": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "repeat_contrasts": "",
            "repeats": 0,
            "paired_values": "",
            "confirmed": False,
            "direction": "",
            "passing": len(passing),
            "tests": int(mine["p_cells"].notna().sum()),
            "order": order,
        }
        if len(passing):
            best = passing.loc[passing["d"].abs().idxmax()]
            word = "lower" if best["d"] < 0 else "higher"
            same = int(best["lower"] if best["d"] < 0 else best["higher"])
            row.update(
                score=min(abs(float(best["d"])), EFFECT_CAP),
                d=float(best["d"]),
                condition=best["condition"],
                q_cells=float(best["q_cells"]),
                q_paired=float(best["q_paired"]),
                change=float(best["change"]),
                ci_low=float(best["ci_low"]),
                ci_high=float(best["ci_high"]),
                repeat_contrasts=best["repeat_contrasts"],
                repeats=int(best["repeats"]),
                paired_values=best["paired_values"],
                confirmed=bool(passing["confirmed"].any()),
                direction=f"{same} of {int(best['repeats'])} {word}",
            )
        rows.append(row)
    ranking = pd.DataFrame(rows, columns=[*RANKING_COLUMNS, "order"])
    ranking = ranking.sort_values(["score", "order"], ascending=[False, True], na_position="last")
    return ranking.drop(columns="order").reset_index(drop=True)


def _limited(row: Mapping[str, Any]) -> list[str]:
    """Each repeat's change and the 95% t interval of a ranked row on exactly 3 repeats."""
    values = row["paired_values"]
    if row["repeats"] != MIN_REPEATS or not row["repeat_contrasts"]:
        return ["–", "–"]
    low, high = row["ci_low"], row["ci_high"]
    interval = f"{change(low, values)} to {change(high, values)}" if np.isfinite(low) else "–"
    return [contrasts_words(row["repeat_contrasts"], values), interval]


def _screen(row: Any) -> str:
    """`` (screen-wide q 0.054)`` beside a verdict whose control label is shared by several
    families (audit 10, owner, 2026-10-06); empty otherwise."""
    q = getattr(row, "q_screen", np.nan)
    return f" (screen-wide q {p_value(q)})" if pd.notna(q) else ""


def _q(value: float) -> str:
    if not np.isfinite(value):
        return "–"
    return p_value(value)


@owned("tests", "ranking", "settings")
@dataclass(frozen=True)
class FeatureDiscovery:
    """The tests (``tests``: one row per feature and condition, ``TEST_COLUMNS``), the
    ranking (``ranking``: one row per feature, best first; ``score`` is NaN when no contrast
    passes the cell-level gate) and the ``settings``."""

    tests: pd.DataFrame
    ranking: pd.DataFrame
    settings: dict[str, Any]

    def top(self, k: int = TOP_FEATURES) -> list[str]:
        """The ``k`` best-ranked features that pass the cell-level gate, best first."""
        return self.ranking.dropna(subset=["score"])["feature"].head(k).tolist()

    def _confirmed_where(self, features: list[str]) -> str:
        """Each feature significant across repeats with the conditions it is significant for,
        so one siRNA's result is not read as the target's."""
        tests = self.tests
        parts = []
        for feature in features[:5]:
            mine = tests[(tests["feature"] == feature) & tests["confirmed"]]
            conditions = listing(
                [
                    f"{r.condition} {change(r.change, r.paired_values)}{_screen(r)}"
                    for r in mine.itertuples()
                ],
                3,
            )
            others = tests[(tests["feature"] == feature) & tests["passes"] & ~tests["confirmed"]]
            tail = (
                f"; not {listing(list(others['condition'].astype(str)), 2)}" if len(others) else ""
            )
            parts.append(f"{feature}, for {conditions}{tail}")
        more = f"; and {len(features) - 5} more" if len(features) > 5 else ""
        return "; ".join(parts) + more

    def confirmed(self) -> list[str]:
        """The passing features the paired test on the repeat means also passes, best first."""
        ranked = self.ranking.dropna(subset=["score"])
        return ranked.loc[ranked["confirmed"], "feature"].tolist()

    def resting(self) -> pd.DataFrame:
        """The tests significant across repeats that rest on near-identical repeats
        (``_discovery_tests.resting``; audit 10, owner, 2026-10-06)."""
        return resting(self.tests, self.settings["alpha"])

    def _unit(self) -> str:
        """The word for one observation (``nucleus``; ``cell`` when none was declared)."""
        return observation_words(self.settings)[0]

    def _head(self) -> str:
        n_tests = int(self.tests["p_cells"].notna().sum()) if len(self.tests) else 0
        n = self.settings["n_features"]
        alpha = self.settings["alpha"]
        if not n_tests:
            return f"Which features separate the conditions? None of the {n} could be tested."
        passing = int(self.ranking["score"].notna().sum())
        against = (
            "its own control"
            if self.settings.get("comparisons")
            else f"the control, {compared_name(self.settings)}"
        )
        # The correction family, said as counted: the tests of each sub-experiment.
        tested = self.tests[self.tests["p_cells"].notna()]
        sizes = tested.groupby("experiment").size() if "experiment" in tested else []
        noun = family_noun(self.settings, plural=True)
        family = (
            f"{count(int(sizes.iloc[0]), 'test')} in each of {len(sizes)} {noun}"
            if len(sizes) > 1 and sizes.nunique() == 1
            else f"{span(list(sizes))} tests in each of {len(sizes)} {noun}"
            if len(sizes) > 1
            else count(n_tests, "test")
        )
        return (
            f"Which features separate the conditions? {passing} of {self._of(n)} "
            f"{'is a candidate' if passing == 1 else 'are candidates'} in the exploratory "
            f"{self._unit()}-level screen: they differ for at least one condition (each "
            f"condition against {against}: Cohen's d on the observations normalised to each "
            "repeat's control, "
            f"randomization test, q ≤ {alpha:g} after Benjamini-Hochberg within each "
            f"{family_noun(self.settings)}: {family}{self._shuffles()}).{self._unpaired()} "
            "Ranked by the "
            "largest difference "
            "(|d|); change: the condition's mean over the control's, the fold change the "
            "paired t-test compares (the difference of the means for a feature with a value "
            f"that is not positive, the same scale for every condition{self._declared()}); a "
            "candidate is "
            "significant across repeats when the paired t-test on the "
            f"repeat means also gives q ≤ {alpha:g}:"
        )

    def _declared(self) -> str:
        """``; declared in the Setup: velocity, the difference``: the scales the Setup
        declares (plan v2, owner, 2026-10-06), which replace the rule; empty without."""
        scales = self.settings.get("scales") or {}
        if not scales:
            return ""
        named = [f"{feature}, the {scale}" for feature, scale in scales.items()]
        return f"; declared in the Setup: {'; '.join(named[:6])}"

    def _of(self, n: int) -> str:
        """``4 features``, or ``4 (3 features and the rare share)``: never "4 of 3 features
        and the rare share" (audit 10, C20)."""
        counted = self._counted(n)
        return counted if counted == count(n, "feature") else f"{n} ({counted})"

    def _counted(self, features: list[str] | int) -> str:
        """``features`` (names, or a number of all the features), the labelled state's share
        among them counted apart (``_wording.features_counted``)."""
        share = self.settings.get("share")
        if isinstance(features, int) and share not in set(self.ranking["feature"]):
            share = None
        return features_counted(features, share)

    def _shuffles(self) -> str:
        """The shuffles of the p-values rerun so that q can reach alpha (``contrast_cells
        .refined``), said as a fact; empty when none was rerun."""
        tests = self.tests
        if "permutations" not in tests:
            return ""
        first = {self.settings["n_permutations"], BALANCED_PERMUTATIONS}
        rerun = tests.loc[~tests["permutations"].isin(first), "permutations"].dropna()
        if rerun.empty:
            return ""
        alpha = self.settings["alpha"]
        return (
            f"; the p-values at or below {alpha:g} rerun with {span(list(rerun))} shuffles, so "
            f"that q can reach {alpha:g} in a family this large"
        )

    def _unpaired(self) -> str:
        """The conditions never compared: no biological repeat holds them and the control."""
        unpaired = [str(c) for c in self.settings.get("unpaired", [])]
        if not unpaired:
            return ""
        label = (
            "its own control" if self.settings.get("comparisons") else compared_name(self.settings)
        )
        return " " + not_compared(unpaired, label, column_word(self.settings["repeat_col"]))

    def _table(self, max_rows: int, indent: int) -> list[str]:
        ranked = self.ranking.dropna(subset=["score"])
        if ranked.empty:
            return []
        # Exactly 3 repeats: each repeat's change and the 95% t interval (plan v2).
        three = bool((ranked["repeats"] == MIN_REPEATS).any())
        rows = [
            [
                rank,
                row["feature"],
                f"{row['d']:+.2f}",
                f"{row['condition']}",
                change(row["change"], row["paired_values"]),
                *(_limited(row) if three else []),
                _q(row["q_cells"]),
                _q(row["q_paired"]),
                row["direction"],
            ]
            for rank, row in enumerate(ranked.head(max_rows).to_dict("records"), start=1)
        ]
        columns = [
            "rank",
            "feature",
            "d",
            "condition",
            "change",
            *(["change in each repeat", "95% t interval"] if three else []),
            f"q {self._unit()} level",
            "q repeat means",
            "repeats",
        ]
        align = "rlrlr" + ("ll" if three else "") + "rrl"
        lines = [table(rows, columns, indent=indent, align=align)]
        if len(ranked) > max_rows:
            lines.append(" " * indent + f"and {len(ranked) - max_rows} more ranked features")
        confirmed = self.confirmed()
        lines.append(
            " " * indent
            + (
                "Significant across repeats (paired t-test on the repeat means, q ≤ "
                f"{self.settings['alpha']:g}): "
                f"{self._counted(confirmed)}"
                + (f" ({self._confirmed_where(confirmed)})." if confirmed else ".")
            )
        )
        return lines

    def _unranked(self, limit: int) -> list[str]:
        unranked = self.ranking[self.ranking["score"].isna()]
        lines = []
        failing = unranked.loc[unranked["tests"] > 0, "feature"].tolist()
        if failing:
            lines.append(
                f"Not a candidate for any condition ({self._counted(failing)}): "
                f"{listing(failing, limit)}."
            )
        # The features no condition could be tested for, by why (each condition's status).
        why: dict[str, list[str]] = {}
        for feature in unranked.loc[unranked["tests"] == 0, "feature"]:
            reasons = set(self.tests.loc[self.tests["feature"] == feature, "status"])
            why.setdefault(reasons.pop() if len(reasons) == 1 else UNTESTED, []).append(feature)
        for reason, untested in why.items():
            lines.append(
                f"Not tested ({self._counted(untested)}, {reason}): {listing(untested, limit)}."
            )
        return lines

    def _cautions(self) -> list[str]:
        cautions = [
            assumption(
                f"Candidates come from the {self._unit()}-level test, which treats every "
                "observation as an independent draw and so finds differences that may not "
                f"repeat; only the paired t-test on the repeat means ({MIN_REPEATS} or more "
                "repeats) decides significance."
            )
        ]
        if len(self.tests) and self.tests["p_paired"].isna().all():
            # The reason from the repeats held, never from the p being missing (issue 90).
            few = bool((self.tests["repeats"] < MIN_REPEATS).all())
            cautions.append(
                f"No paired test: fewer than {MIN_REPEATS} biological repeats hold a condition "
                "and the control."
                if few
                else "No paired test: the repeat effects do not vary (or a value is missing), "
                "so the test has no variance to use."
            )
        mixed = self.tests[self.tests["passes"] & (self.tests["lower"] > 0)]
        mixed = mixed[mixed["higher"] > 0]
        if len(mixed):
            names = listing([f"{r.feature} ({r.condition})" for r in mixed.itertuples()], 4)
            cautions.append(
                f"Candidates, but lower in some repeats and higher in others: {names}; a shift "
                "between repeats, not a consistent effect."
            )
        return cautions + resting_words(self.resting())

    def short_parts(
        self, indent: int = 0, max_rows: int = TOP_FEATURES, limit: int = 3
    ) -> tuple[list[str], list[str]]:
        """The short form under ``indent`` (``max_rows`` ranked features, ``limit`` names per
        list of unranked ones) and its caveats."""
        lines = [wrap(self._head(), indent, hang=indent + 2)]
        if self.tests.empty:
            if self.settings.get("unpaired"):  # named once, "Not compared" (audit 10, C8)
                return lines, []
            reason = self.settings.get("reason", "No condition to compare with the control.")
            return [*lines, wrap(reason, indent, hang=indent + 2)], []
        lines += self._table(max_rows, indent + 2)
        lines += [wrap(line, indent, hang=indent + 2) for line in self._unranked(limit)]
        if self.top():
            lines.append(
                " "
                * indent
                + f"Plotted by default: up to {TOP_FEATURES} best-ranked features (the readout "
                "when none ranks)."
            )
        return lines, self._cautions()

    def summary(self, max_rows: int = MAX_ROWS_LISTED, detail: str = "short") -> str:
        """The ranked features and each caveat once (``"short"``: at most ``TOP_FEATURES``
        rows; ``"full"``: up to ``max_rows`` and every unranked feature)."""
        short = check_detail(detail) == "short"
        rows = min(max_rows, TOP_FEATURES) if short else max_rows
        lines, cautions = self.short_parts(max_rows=rows, limit=3 if short else MAX_ROWS_LISTED)
        return "\n".join([*lines, *caveat_lines(cautions, indent=0, detail=detail)])


@one_warning
def feature_discovery(
    objects: pd.DataFrame,
    fovs: pd.DataFrame,
    *,
    features: Sequence[str],
    control: Mapping[str, Any] | None = None,
    condition_col: str = MORPHOBRICKS_GROUP_COLUMNS[0],
    repeat_col: str | None = MORPHOBRICKS_GROUP_COLUMNS[1],
    technical_col: str | None = TECHNICAL_REPEAT_COLUMN,
    n_permutations: int = N_PERMUTATIONS,
    alpha: float = GATE_ALPHA,
    random_state: int = 0,
    fov_col: str | None = None,
    group_cols: Sequence[str] | None = None,
    observation_unit: str | None = None,
    comparisons: Sequence[tuple[Any, Any]] | None = None,
    n_jobs: int = 1,
    scales: Mapping[str, str] | None = None,
    families: Mapping[Any, Any] | None = None,
) -> FeatureDiscovery:
    """Rank ``features`` by the cell-level gate, with the paired test beside it (see the
    module). Arguments follow ``condition_contrast``; ``n_jobs`` worker processes test the
    features, with the same results for every value (``_parallel``); ``scales``, each
    feature's declared scale (``Setup.scales``), else the automatic rule; ``families``, each
    condition's declared experiment (``Setup.experiment``), else the sub-experiments."""
    control, repeat_col = read_former(control=control, repeat_col=repeat_col)
    features = list(dict.fromkeys(column_names(features, "features")))
    if not features:
        raise ValueError("features must name at least one column")
    if comparisons is not None:
        comparisons = check_comparisons(comparisons)
    n_permutations, random_state = check_arguments(
        control, condition_col, n_permutations, random_state, False, comparisons
    )
    if not is_number(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    data = validated(objects, fovs, features, fov_col, group_cols, observation_unit)
    repeat_col, technical_col = resolve_columns(
        data, condition_col, repeat_col, technical_col, control
    )
    all_groups = groups(data)
    described = describe_groups(all_groups, condition_col, repeat_col, control, comparisons)
    settings: dict[str, Any] = {
        "features": features,
        "n_features": len(features),
        "control": dict(control or {}),
        "comparisons": list(comparisons or []),
        "condition_col": condition_col,
        "repeat_col": repeat_col,
        "n_permutations": n_permutations,
        "alpha": alpha,
        "random_state": random_state,
        "declared_experiments": bool(families),
        "observation_unit": data.observation_unit,
        "observation_unit_declared": data.observation_unit_declared,
        "scales": dict(scales or {}),
    }
    contrasts = contrasts_of(described)
    workers = check_n_jobs(n_jobs)
    rows, retests, held = tested_rows(
        (all_groups, described, contrasts, technical_col, dict(scales or {})),
        features,
        (n_permutations, random_state, alpha),
        workers,
    )
    # The conditions no biological repeat holds beside the control: never compared, said.
    settings["unpaired"] = [
        c for c in dict.fromkeys(condition for _, condition in contrasts) if c not in held
    ]
    if not rows:
        settings["reason"] = f"No condition to compare with the control, {control_name(control)}."
        empty = pd.DataFrame(columns=RANKING_COLUMNS)  # its columns, for top and the page
        return FeatureDiscovery(pd.DataFrame(columns=TEST_COLUMNS), empty, settings)
    tests = pd.DataFrame(rows).reindex(columns=TEST_COLUMNS)
    tests["experiment"] = tests["condition"].map(declared(experiments(held), families))
    # The tests at p <= alpha rerun with the shuffles their family needs (contrast_cells).
    tests["p_cells"], tests["permutations"] = refined(
        tests["p_cells"],
        tests["experiment"],
        tests["permutations"],
        retests,
        lambda index: group_rng(
            random_state,
            {
                "feature": tests.at[index, "feature"],
                "condition": tests.at[index, "condition"],
                "test": "refined",
            },
        ),
        alpha,
        n_jobs=workers,
    )
    # Corrected over every feature and condition of a sub-experiment (_experiments).
    tests["q_cells"] = corrected(tests["p_cells"], tests["experiment"])
    tests["q_paired"] = corrected(tests["p_paired"], tests["experiment"])
    tests["q_paired_raw"] = corrected(tests["p_paired_raw"], tests["experiment"])
    tests["passes"] = (tests["q_cells"] <= alpha).fillna(False) & tests["d"].notna()
    tests["confirmed"] = tests["passes"] & (tests["q_paired"] <= alpha).fillna(False)
    tests["near_identical"], tests["family_sd"] = near_identical(tests)
    tests["q_screen"] = screen_wide(
        tests["p_paired"], tests["control_condition"], tests["experiment"]
    )
    return FeatureDiscovery(tests, _rank(tests, features), settings)


__all__ = ["TOP_FEATURES", "FeatureDiscovery", "feature_discovery"]
