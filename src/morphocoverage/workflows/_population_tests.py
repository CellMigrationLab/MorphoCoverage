"""Question 10's enrichment tests (private; ``_populations``): per population and condition,
its share against its own control's in each biological repeat, on the log-odds scale, tested
across the repeats by the paired t-test, Benjamini-Hochberg within each sub-experiment over
every population and condition (owner, 2026-10-02; the module ``_populations`` says why).
"""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np
import pandas as pd

from .._control import comparison_label
from ..metadata import CONDITION_COLUMN, REPEAT_COLUMN
from ..sampling_unit._experiments import corrected, experiments
from ..sampling_unit.contrast_repeats import MIN_REPEATS
from ..statistics import paired_t


def _log_odds_ratio(t: Any, c: Any) -> float:
    """The log odds ratio of the condition's share against its control's in one repeat
    (``t`` and ``c`` the cells in the population, ``sum``, and the cells, ``count``). One
    pseudo-cell is added to each side, split by the population's pooled share of that repeat,
    so a share of 0 has log odds and uneven numbers of cells do not tip the ratio (a fixed
    half cell would make 1 of 5000 read lower than 0 of 300); 0, no change, when the
    population is absent from both sides (owner, 2026-10-02: the repeat still counts)."""
    pooled = (t["sum"] + c["sum"]) / (t["count"] + c["count"])
    if not pooled > 0:
        return 0.0

    def odds(k: float, n: float) -> float:
        return float(np.log((k + pooled) / (n - k + 1 - pooled)))

    return odds(t["sum"], t["count"]) - odds(c["sum"], c["count"])


def one_sided(tests: pd.DataFrame) -> pd.Series:
    """The tests on one side of the control in every repeat, over two or more repeats."""
    every = (tests["higher"] == tests["repeats"]) | (tests["lower"] == tests["repeats"])
    return every & (tests["repeats"] >= 2) & (tests["higher"] != tests["lower"])


def enrichment_tests(marked: pd.DataFrame, names: list[str], control: Any) -> pd.DataFrame:
    """Per population and condition, its share against the control's in each biological
    repeat, on the log-odds scale (a population and its complement alike, a small share as
    steady as a large one), tested across the repeats: the one-sample t-test on the log odds
    ratios, Benjamini-Hochberg within each sub-experiment over every population and
    condition (``_experiments``). ``control`` is the control condition, or with several
    controls the ``(condition, control)`` pairs: each condition against its own, its rows
    labelled ``condition vs control`` (owner, 2026-10-05)."""
    repeat = REPEAT_COLUMN if REPEAT_COLUMN in marked else None
    keys = [CONDITION_COLUMN, *([repeat] if repeat else [])]
    # Without a group to test (the declared label's only), the counts are of an empty column.
    counts = marked.assign(_none=0.0).groupby(keys, dropna=False)[names or ["_none"]]
    counts = counts.agg(["sum", "count"])
    rows, held, unpaired = [], {}, []
    for label, condition, own in _pairs(marked, control):
        repeats = (
            [
                r
                for r in pd.unique(marked[repeat])
                if (condition, r) in counts.index and (own, r) in counts.index
            ]
            if repeat
            else [None]
        )
        if not repeat and not ((condition,) in counts.index or condition in counts.index):
            continue
        if not repeats:
            unpaired.append(label)
            continue
        held[label] = set(repeats)
        sides = _Sides(counts, repeat, condition, own)
        for name in names:
            row = _test_row(sides, repeats, name)
            if row is not None:
                rows.append({"feature": name, "condition": label, **row})
    columns = ["feature", "condition", "repeats", "higher", "lower", "d", "p_paired"]
    tests = pd.DataFrame(rows, columns=[*columns, "share", "control_share"])
    if len(tests):
        tests["experiment"] = tests["condition"].map(experiments(held))
        tests["q_paired"] = corrected(tests["p_paired"], tests["experiment"])
    else:
        tests["q_paired"] = pd.Series(dtype=float)
    # The conditions imaged in no biological repeat of the control: not compared, and said.
    tests.attrs["unpaired"] = unpaired
    return tests


def _pairs(marked: pd.DataFrame, control: Any) -> list[tuple[Any, Any, Any]]:
    """``(label, condition, its control)`` of each comparison: with several controls each
    condition against its own (``condition vs control``), else every other condition against
    the one."""
    if isinstance(control, tuple):
        return [(comparison_label(c, r), c, r) for c, r in control]
    return [(c, c, control) for c in pd.unique(marked[CONDITION_COLUMN]) if c != control]


class _Sides(NamedTuple):
    """The counts of one comparison: the cells in each population and the cells (``sum``,
    ``count``) per condition and repeat (``repeat``: the column, ``None`` without), the
    condition's and its own control's."""

    counts: pd.DataFrame
    repeat: str | None
    condition: Any
    own: Any


def _test_row(sides: _Sides, repeats: list[Any], name: str) -> dict[str, Any] | None:
    """The test of one population (``name``) in one comparison over its ``repeats``: the
    sides in the repeats, the mean log odds ratio, the paired t-test's p and both shares;
    ``None`` when there is nothing to compare."""
    counts, repeat, condition, own = sides
    ratios = []
    totals = {"t": [0.0, 0.0], "c": [0.0, 0.0]}  # cells in it, cells, on each side
    for r in repeats:
        t = counts.loc[(condition, r) if repeat else condition, name]
        c = counts.loc[(own, r) if repeat else own, name]
        ratios.append(_log_odds_ratio(t, c))
        totals["t"] = [totals["t"][0] + t["sum"], totals["t"][1] + t["count"]]
        totals["c"] = [totals["c"][0] + c["sum"], totals["c"][1] + c["count"]]
    # Absent from the condition and from its own control in every repeat: nothing
    # to compare. Absent from both in some repeats: no change there (0), kept in n.
    ratios = np.array([r for r in ratios if np.isfinite(r)])
    if not len(ratios) or not totals["t"][0] + totals["c"][0]:
        return None
    tested = len(ratios) >= MIN_REPEATS
    return {
        "repeats": len(ratios),
        # The sides as the test reads them (the sign of each repeat's ratio).
        "higher": int((ratios > 0).sum()),
        "lower": int((ratios < 0).sum()),
        "d": float(ratios.mean()),
        "p_paired": paired_t(ratios) if tested else np.nan,
        # Over the condition's repeats: its share and its own control's.
        "share": totals["t"][0] / totals["t"][1] if totals["t"][1] else np.nan,
        "control_share": totals["c"][0] / totals["c"][1] if totals["c"][1] else np.nan,
    }


__all__ = ["enrichment_tests", "one_sided"]
