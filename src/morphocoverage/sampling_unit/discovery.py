"""Which features separate the conditions? Discovery by the cell-level test, the paired test
beside it.

With many features the report cannot plot them all, and with one well per condition and
repeat the gated ranking over whole technical repeats (``feature_ranking``) has nothing to
shuffle. ``feature_discovery`` ranks the features the way the lab's Plot&Stats notebook
looks for them, as the owner decided (2026-10-01): every feature, every condition against
the declared control, pooled over the biological repeats:

- **The gate (discovery)**: the cell-level randomization test (``contrast_cells``) in the
  version the readings trust most, on values normalised to each repeat's control when
  every control mean is positive and balanced over the repeats when they are not; its
  p-values are Benjamini-Hochberg corrected over every feature and condition of a
  sub-experiment (``_experiments``), and a test passes when ``q <= alpha`` (a candidate).
  A feature's score is the largest |d| among its passing contrasts (capped at
  ``EFFECT_CAP``); the plots draw the best-ranked features.
- **The second gate (kept beside it)**: the paired t-test on the repeat means (on the log
  ratios to the control when every mean is positive, else on the differences), corrected
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
value everywhere is not tested.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .._caveats import assumption, caveat_lines
from .._checks import is_number
from .._former_repeat import one_warning, read_former
from .._seeds import group_rng
from .._style import check_detail, span, table, wrap
from .._wording import change, count, listing, p_value
from ..metadata import MORPHOBRICKS_GROUP_COLUMNS, TECHNICAL_REPEAT_COLUMN
from ..statistics import cohen_d, paired_t, repeat_values
from ._common import groups, validated
from ._experiments import corrected, experiments
from ._units import (
    check_arguments,
    contrasts_of,
    control_label,
    describe_groups,
    resolve_columns,
    side_units,
    sides,
)
from .contrast_cells import trusted_test
from .contrast_repeats import MIN_REPEATS

EFFECT_CAP = 20.0
GATE_ALPHA = 0.05
TOP_FEATURES = 5
MAX_ROWS_LISTED = 24
N_PERMUTATIONS = 9999
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
]


def _pooled(
    all_groups: list, described: list, contrasts: list, feature: str, technical_col: str | None
) -> dict[Any, dict[str, Any]]:
    """Per condition, its observations and its own control's per repeat, and every control
    observation of the dataset per repeat (as ``condition_contrast`` pools them)."""
    pooled: dict[Any, dict[str, Any]] = {}
    every: dict[Any, np.ndarray] = {}
    for repeat, condition in contrasts:
        treated, controls = sides(described, repeat, condition)
        if not controls:
            continue
        t_units, _ = side_units(all_groups, feature, treated, technical_col)
        c_units, _ = side_units(all_groups, feature, controls, technical_col)
        t_values = np.concatenate([f for u in t_units for f in u])
        c_values = np.concatenate([f for u in c_units for f in u])
        if not len(t_values) or not len(c_values):
            continue
        every.setdefault(repeat, c_values)
        side = pooled.setdefault(
            condition,
            {"by_t": {}, "by_c": {}, "control_condition": control_label(described, controls)},
        )
        side["by_t"][repeat] = t_values
        side["by_c"][repeat] = c_values
    for side in pooled.values():
        side["every"] = every
    return pooled


def _paired(by_t: dict, by_c: dict) -> tuple[float, str, int, int, float, float]:
    """The paired t-test on the repeat means (``statistics.repeat_values``: log ratios when
    every mean is positive), how many repeats are lower and higher than the control, the
    test on the raw differences (as ``condition_contrast`` gives it beside, so the readout's
    reading shares this family) and the change it tests (a fold change, else the mean
    difference)."""
    means = np.array([np.mean(by_t[r]) for r in by_t])
    controls = np.array([np.mean(by_c[r]) for r in by_t])
    lower, higher = int((means < controls).sum()), int((means > controls).sum())
    values, compared, change = repeat_values(means, controls)
    if len(means) < MIN_REPEATS:
        return np.nan, values, lower, higher, np.nan, change
    return paired_t(compared), values, lower, higher, paired_t(means - controls), change


def _per_repeat(by_t: dict, by_c: dict) -> dict[str, float]:
    """Cohen's d of the observations in each repeat (``condition_contrast``'s ``d_objects``),
    its mean and range, and the fewest and most observations on a side in one repeat, for
    the page's table across repeats and each figure's legend."""
    d = np.array([cohen_d(by_t[r], by_c[r]) for r in by_t], dtype=float)
    d = d[np.isfinite(d)]
    sizes = [len(by_t[r]) for r in by_t] + [len(by_c[r]) for r in by_t]
    return {
        "d_repeat_mean": float(d.mean()) if len(d) else np.nan,
        "d_repeat_min": float(d.min()) if len(d) else np.nan,
        "d_repeat_max": float(d.max()) if len(d) else np.nan,
        "repeat_observations_min": min(sizes),
        "repeat_observations_max": max(sizes),
    }


def _row(
    feature: str,
    condition: Any,
    side: dict[str, Any],
    n_permutations: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    by_t, by_c = side["by_t"], side["by_c"]
    p_paired, paired_values, lower, higher, p_raw, change = _paired(by_t, by_c)
    row: dict[str, Any] = {
        "feature": feature,
        "condition": condition,
        "control_condition": side["control_condition"],
        "repeats": len(by_t),
        "lower": lower,
        "higher": higher,
        "p_paired": p_paired,
        "paired_values": paired_values,
        "change": change,
        "p_paired_raw": p_raw,
        "status": "ok",
        **_per_repeat(by_t, by_c),
    }
    values = np.concatenate([*by_t.values(), *side["every"].values()])
    if np.ptp(values) == 0:
        return {**row, "d": np.nan, "p_cells": np.nan, "status": "one value everywhere"}
    test = trusted_test(by_t, by_c, side["every"], n_permutations, rng)
    return {
        **row,
        "observations": test["observations"],
        "control_observations": test["control_observations"],
        "d": test["d"],
        "p_cells": test["p"],
        "normalised": test["normalised"],
        "balanced": test["balanced"],
    }


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
                paired_values=best["paired_values"],
                confirmed=bool(passing["confirmed"].any()),
                direction=f"{same} of {int(best['repeats'])} {word}",
            )
        rows.append(row)
    ranking = pd.DataFrame(rows)
    ranking = ranking.sort_values(["score", "order"], ascending=[False, True], na_position="last")
    return ranking.drop(columns="order").reset_index(drop=True)


def _q(value: float) -> str:
    if not np.isfinite(value):
        return "–"
    return p_value(value)


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
                [f"{r.condition} {change(r.change, r.paired_values)}" for r in mine.itertuples()],
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

    def _head(self) -> str:
        n_tests = int(self.tests["p_cells"].notna().sum()) if len(self.tests) else 0
        n = self.settings["n_features"]
        alpha = self.settings["alpha"]
        if not n_tests:
            return f"Which features separate the conditions? None of the {n} could be tested."
        passing = int(self.ranking["score"].notna().sum())
        label = " · ".join(str(v) for v in self.settings["control"].values())
        # The correction family, said as counted: the tests of each sub-experiment.
        tested = self.tests[self.tests["p_cells"].notna()]
        sizes = tested.groupby("experiment").size() if "experiment" in tested else []
        family = (
            f"{count(int(sizes.iloc[0]), 'test')} in each of {len(sizes)} sub-experiments"
            if len(sizes) > 1 and sizes.nunique() == 1
            else f"{span(list(sizes))} tests in each of {len(sizes)} sub-experiments"
            if len(sizes) > 1
            else count(n_tests, "test")
        )
        return (
            f"Which features separate the conditions? {passing} of {count(n, 'feature')} "
            f"{'is a candidate' if passing == 1 else 'are candidates'} in the exploratory "
            "cell-level screen: they differ for at least one condition (each condition against "
            "the control, "
            f"{label}: Cohen's d on the observations normalised to each repeat's control, "
            f"randomization test, q ≤ {alpha:g} after Benjamini-Hochberg within each "
            f"sub-experiment: {family}). Ranked by the largest difference "
            "(|d|); change: the condition's mean over the control's, the fold change the "
            "paired t-test compares (the difference of the means where a mean is not "
            "positive); a candidate is significant across repeats when the paired t-test on the "
            f"repeat means also gives q ≤ {alpha:g}:"
        )

    def _table(self, max_rows: int, indent: int) -> list[str]:
        ranked = self.ranking.dropna(subset=["score"])
        if ranked.empty:
            return []
        rows = [
            [
                rank,
                row["feature"],
                f"{row['d']:+.2f}",
                f"{row['condition']}",
                change(row["change"], row["paired_values"]),
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
            "q cell level",
            "q repeat means",
            "repeats",
        ]
        lines = [table(rows, columns, indent=indent, align="rlrlrrrl")]
        if len(ranked) > max_rows:
            lines.append(" " * indent + f"and {len(ranked) - max_rows} more ranked features")
        confirmed = self.confirmed()
        lines.append(
            " " * indent
            + (
                "Significant across repeats (paired t-test on the repeat means, q ≤ "
                f"{self.settings['alpha']:g}): "
                f"{count(len(confirmed), 'feature')}"
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
                f"Not a candidate for any condition ({count(len(failing), 'feature')}): "
                f"{listing(failing, limit)}."
            )
        untested = unranked.loc[unranked["tests"] == 0, "feature"].tolist()
        if untested:
            lines.append(
                f"Not tested ({count(len(untested), 'feature')}, one value everywhere): "
                f"{listing(untested, limit)}."
            )
        return lines

    def _cautions(self) -> list[str]:
        cautions = [
            assumption(
                "Candidates come from the cell-level test, which treats every observation as "
                "an independent draw and so finds differences that may not repeat; only the "
                f"paired t-test on the repeat means ({MIN_REPEATS} or more repeats) decides "
                "significance."
            )
        ]
        if len(self.tests) and self.tests["p_paired"].isna().all():
            cautions.append(
                f"No paired test: fewer than {MIN_REPEATS} biological repeats hold a condition "
                "and the control."
            )
        mixed = self.tests[self.tests["passes"] & (self.tests["lower"] > 0)]
        mixed = mixed[mixed["higher"] > 0]
        if len(mixed):
            names = listing([f"{r.feature} ({r.condition})" for r in mixed.itertuples()], 4)
            cautions.append(
                f"Candidates, but lower in some repeats and higher in others: {names}; a shift "
                "between repeats, not a consistent effect."
            )
        return cautions

    def short_parts(
        self, indent: int = 0, max_rows: int = TOP_FEATURES, limit: int = 3
    ) -> tuple[list[str], list[str]]:
        """The short form under ``indent`` (``max_rows`` ranked features, ``limit`` names per
        list of unranked ones) and its caveats."""
        lines = [wrap(self._head(), indent, hang=indent + 2)]
        if self.tests.empty:
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
    control: Mapping[str, Any],
    condition_col: str = MORPHOBRICKS_GROUP_COLUMNS[0],
    repeat_col: str | None = MORPHOBRICKS_GROUP_COLUMNS[1],
    technical_col: str | None = TECHNICAL_REPEAT_COLUMN,
    n_permutations: int = N_PERMUTATIONS,
    alpha: float = GATE_ALPHA,
    random_state: int = 0,
    fov_col: str | None = None,
    group_cols: Sequence[str] | None = None,
    observation_unit: str | None = None,
) -> FeatureDiscovery:
    """Rank ``features`` by the cell-level gate, with the paired test beside it (see the
    module). Arguments follow ``condition_contrast``."""
    control, repeat_col = read_former(control=control, repeat_col=repeat_col)
    features = list(dict.fromkeys(features))
    if not features:
        raise ValueError("features must name at least one column")
    n_permutations, random_state = check_arguments(
        control, condition_col, n_permutations, random_state, False
    )
    if not is_number(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
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
        "n_permutations": n_permutations,
        "alpha": alpha,
        "random_state": random_state,
        "observation_unit": data.observation_unit,
    }
    contrasts = contrasts_of(described)
    rows = []
    held: dict[Any, set] = {}  # the repeats each condition is compared in
    for feature in features:
        pooled = _pooled(all_groups, described, contrasts, feature, technical_col)
        for condition, side in pooled.items():
            held.setdefault(condition, set()).update(side["by_t"])
            rng = group_rng(random_state, {"feature": feature, "condition": condition})
            rows.append(_row(feature, condition, side, n_permutations, rng))
    if not rows:
        settings["reason"] = f"No condition to compare with the control {dict(control)}."
        return FeatureDiscovery(pd.DataFrame(columns=TEST_COLUMNS), pd.DataFrame(), settings)
    tests = pd.DataFrame(rows).reindex(columns=TEST_COLUMNS)
    tests["experiment"] = tests["condition"].map(experiments(held))
    # Corrected over every feature and condition of a sub-experiment (_experiments).
    tests["q_cells"] = corrected(tests["p_cells"], tests["experiment"])
    tests["q_paired"] = corrected(tests["p_paired"], tests["experiment"])
    tests["q_paired_raw"] = corrected(tests["p_paired_raw"], tests["experiment"])
    tests["passes"] = (tests["q_cells"] <= alpha).fillna(False) & tests["d"].notna()
    tests["confirmed"] = tests["passes"] & (tests["q_paired"] <= alpha).fillna(False)
    return FeatureDiscovery(tests, _rank(tests, features), settings)


__all__ = ["TOP_FEATURES", "FeatureDiscovery", "feature_discovery"]
