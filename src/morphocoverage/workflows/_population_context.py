"""What question 10's groups are, read against the conditions (private; ``_populations``,
``_html_next``; owner, 2026-10-06, audit 10: A7, A8, C3, C4, A13).

- **Mostly one condition** (``composition``): a group of which one condition holds
  ``MOSTLY`` or more of the cells is that condition's shift, not a subpopulation (a condition
  94% in a group and its control 0% is the condition, seen as a group).
- **Against the compared control** (``against_controls``): a group's defining features are
  its cells' Cohen's d against the cells of the control its cells are compared with (each
  condition's own control; a control's parent, else itself), not against every cell pooled,
  so a group of a condition's cells that reads "low area" against cells pooled over two
  experiments 2.4-fold apart reads "high area" against its control, as question 9 does.
- **Values when experiments differ** (``compared_values``): with several controls and no
  ``normalize``, the groups are found on the values normalised to each condition's own
  control (the other tables centred per repeat), as with ``normalize``, so raw values pooled
  across experiments do not make groups of experiments; said in the method line.
- **A rule on few values** (``few_valued``): a rule that thresholds a feature taking few
  distinct values (``_value_checks.has_few_values``) is said, as it rests on the cells off its
  most common value.
- **Ranking** (``by_evidence``): the most different condition is the one of the smallest q,
  then the largest effect; without any q, the largest effect (before, the order of the rows).
- **Rare** (``RARE_SHARE``): the page suggests declaring a group as the rare phenotype only
  when it holds at most this share of the cells and is not mostly one condition.
"""

from __future__ import annotations

import warnings
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from .._wording import listing, percent
from ..metadata import CONDITION_COLUMN
from ..retrospective._value_checks import has_few_values
from ..sampling_unit.populations import DEFINING, defined_by
from ..statistics import cohen_d

if TYPE_CHECKING:
    from .setup import Setup

# One condition holding this share of a group's cells: its shift, not a subpopulation.
MOSTLY = 0.9
# The page suggests a group as the rare phenotype at most at this share of the cells.
RARE_SHARE = 0.1
# Under the table when the groups are described against their compared controls.
AGAINST_WORDS = (
    "Defined by: each group's {units} against those of the control they are compared with, "
    "a group mostly of a control against every other {unit}."
)
EXPERIMENTS_DIFFER = (
    "on values normalised to each condition's own control, as the experiments have their own "
    "controls"
)


def composition(marked: pd.DataFrame, name: str) -> tuple[Any, float] | None:
    """``(condition, share)`` when one condition holds ``MOSTLY`` or more of the group's
    cells, else ``None``."""
    if CONDITION_COLUMN not in marked or name not in marked:
        return None
    inside = marked.loc[marked[name].eq(1), CONDITION_COLUMN]
    if not len(inside):
        return None
    shares = inside.value_counts(normalize=True)
    return (shares.index[0], float(shares.iloc[0])) if shares.iloc[0] >= MOSTLY else None


def mostly_words(mostly: tuple[Any, float], units: str) -> str:
    """``mostly a1, 94% of its cells: its shift, not a subpopulation``."""
    condition, share = mostly
    return f"mostly {condition}, {percent(share)} of its {units}: its shift, not a subpopulation"


def _controls_of(setup: Setup, conditions: pd.Series) -> set[Any] | None:
    """The controls the conditions of a group's cells are compared with, or ``None``."""
    own = setup.own_references
    if own is not None:
        return {own.get(c, c) for c in pd.unique(conditions)}
    single = (setup.single_control or {}).get(CONDITION_COLUMN)
    return None if single is None else {single}


def _d(inside: np.ndarray, outside: np.ndarray) -> float:
    """Cohen's d of ``inside`` against ``outside``, missing values aside."""
    inside, outside = inside[~np.isnan(inside)], outside[~np.isnan(outside)]
    return float(cohen_d(inside, outside)) if len(inside) > 1 and len(outside) > 1 else np.nan


def against_controls(table: pd.DataFrame, marked: pd.DataFrame, setup: Setup) -> bool:
    """Each described group's defining features against the cells of its compared controls,
    in place in ``table`` (``defining``, ``defined_by``); whether any was redescribed."""
    if CONDITION_COLUMN not in marked:
        return False
    features = [c for c in setup.columns if c in marked]
    changed = False
    for index, row in table.iterrows():
        name = row["population"]
        if not row["defining"] or name not in marked:
            continue
        member = marked[name].eq(1).to_numpy()
        controls = _controls_of(setup, marked.loc[member, CONDITION_COLUMN])
        mostly = composition(marked, name)
        if controls is None or (mostly is not None and mostly[0] in controls):
            continue  # a group of a control's own cells: against every other cell
        reference = marked[CONDITION_COLUMN].isin(controls).to_numpy() & ~member
        if reference.sum() < 2 or member.sum() < 2:
            continue
        values = marked[features].to_numpy(dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            d = np.array(
                [_d(values[member, j], values[reference, j]) for j in range(len(features))]
            )
        order = np.argsort(-np.abs(np.nan_to_num(d)))
        pairs = [(features[i], float(d[i])) for i in order[:DEFINING] if np.isfinite(d[i])]
        table.at[index, "defining"] = pairs
        table.at[index, "defined_by"] = defined_by(pairs)
        changed = True
    return changed


def compared_values(
    objects: pd.DataFrame, fovs: pd.DataFrame, setup: Setup, raw: pd.DataFrame | None
) -> tuple[pd.DataFrame, Setup, bool]:
    """The values the groups are found on, the Setup that reads them, and whether they are
    normalised because the experiments differ (see the module); the values as given when
    they cannot be normalised (a control mean that is not positive)."""
    if setup.normalize is not None or setup.comparisons is None or raw is None:
        return objects, setup, False
    from ._feature_space import analysed_tables  # the assembly imports this module

    normalised = replace(setup, normalize="control_mean")
    try:
        values = analysed_tables(normalised.prepared(raw, fovs), fovs, normalised)
    except ValueError:  # a control mean that is not positive: the values as given
        return objects, setup, False
    return values, normalised, True


def few_valued(row: Any, marked: pd.DataFrame) -> list[str]:
    """The features of a group's rule that take few distinct values."""
    terms = row.get("rule_terms", ())
    named = list(dict.fromkeys(term[0] for term in terms or ()))
    return [f for f in named if f in marked and has_few_values(marked[f].to_numpy(dtype=float))]


def few_words(features: list[str], units: str) -> str:
    """``its rule thresholds count, a feature with few distinct values: ...``."""
    what = "features" if len(features) > 1 else "a feature"
    return (
        f"its rule thresholds {listing(features)}, {what} with few distinct values: the rule "
        f"rests on the {units} off the most common value"
    )


def by_evidence(tests: pd.DataFrame) -> pd.DataFrame:
    """The tests ordered by evidence: the smallest q, then the largest effect (|d|)."""
    return tests.assign(_effect=-tests["d"].abs()).sort_values(
        ["q_paired", "_effect", "p_paired"], na_position="last"
    )


__all__ = [
    "AGAINST_WORDS",
    "EXPERIMENTS_DIFFER",
    "MOSTLY",
    "RARE_SHARE",
    "against_controls",
    "by_evidence",
    "compared_values",
    "composition",
    "few_valued",
    "few_words",
    "mostly_words",
]
