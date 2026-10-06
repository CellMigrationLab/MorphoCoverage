"""The whole-table test: does a condition's centre in a feature table move away from its
control's, the same way in every biological repeat? (owner, 2026-10-06; question 9)

A feature table that is not read one column at a time, such as a DINO embedding, is tested
whole, beside the per-feature tests of the explainable features, for each comparison of a
condition with its own control. In the table's space, scaled per comparison (owner,
2026-10-06, audit 8): each column divided by the robust scale (``embedding.robust_scale``:
the MAD as an SD, the SD where the MAD is under a quarter of it) of the within-group
residuals of the comparison's two sides, every observation minus the centre of its side in
its repeat, so another condition's effect or a repeat's batch never changes the scale; a
column that does not vary within the groups is left out. Groups of one observation have no
residual and are not read for it; with no group of two, "not tested: one observation a side
and repeat". The MFA weighting of the combined space divides the whole table by one number,
which changes none of the statistics below, so it is not applied.

- **Repeat-level verdict**, the one to report. The field is the sampling unit: each field's
  observations are averaged first (a field of one image is that image), and a side's centre
  in a repeat is the mean of its fields. Per biological repeat ``r`` holding both sides,
  ``Δ_r`` = centre(condition) − centre(control). For each ``r``, the direction of the other
  repeats' mean shift, ``u_−r``, normalised, and the projection ``s_r = Δ_r · u_−r``: positive
  when repeat ``r`` moves the way the others do. "Shifted the same way in 4 of 5 repeats"
  counts the positive ``s_r``. The test is the one-sided paired t-test that the mean of
  ``s_r`` is above 0, with its t corrected for the repeats the ``s_r`` share: repeat ``q``
  enters every ``u_−r`` but its own, so under no effect the ``s_r`` are positively
  correlated, by up to ``1/(n − 1)`` in a table of many effective dimensions, and the plain
  t-test rejected 8-20% of null datasets at 5% (``data/simulations/whole_table.py``). The
  t is multiplied by ``sqrt((n − 2) / (2 (n − 1)))``, the factor at that largest
  correlation (``n − 1`` degrees of freedom): calibrated in wide tables, conservative in
  narrow ones. It needs ``MIN_REPEATS`` (3) repeats; with fewer, "not tested: n repeats".
- **Shuffle test**, exploratory, never in the verdict (owner, 2026-10-06): the distance
  between the condition's and the control's centres (each the mean of its units' means over
  the repeats holding both), with a null that permutes the condition and control labels of
  the units within each repeat (``permutations`` draws), ``p = (extreme + 1) / (B + 1)``.
  The unit is the highest one below the repeat the data name: the technical repeat (the
  ``technical_repeat`` column), else the analysis groups beyond the condition and the
  repeat (a ``well`` column in ``group_cols``), else the field (``shuffled``). With fewer
  than ``MIN_ASSIGNMENTS`` distinct assignments (one well a side, say) there is no p. It
  asks whether these units differ, not whether the shift repeats: shuffling fields counts a
  shift particular to one repeat's unnamed wells, so it is then anti-conservative when the
  wells of a repeat differ (a batch), as the cell-level screen is.
- **Effect size**: the length of the mean shift over the repeats, ``|mean_r Δ_r|``, in units
  of the within-group spread, on the same scale: the root of the summed variances of the
  observations around their own group's centre (a side in a repeat), the typical distance of
  an observation from its group's centre. One column of a single feature gives a Cohen's d.

Benjamini-Hochberg: the repeat-level tests of one table form one family per
sub-experiment (the conditions linked by the repeats they were compared in, ``_experiments``),
as the per-feature main rows do, and never join the per-feature families.

Observations missing a value of the table are left out (counted, ``left_out``). The
permutations are vectorised (one matrix product per comparison), so the test costs little
beside the per-feature work.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from math import comb, prod
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from .._seeds import group_rng
from ..embedding import robust_scale
from ..metadata import CONDITION_COLUMN, REPEAT_COLUMN, TECHNICAL_REPEAT_COLUMN, empty_values
from ..schema import ValidatedInput, validate_input_tables
from ._experiments import corrected, experiments

MIN_REPEATS = 3
# The distinct label assignments a shuffle needs for a p of 0.05: the distance is the same
# with every label swapped, so the smallest p is about 2 over their number.
MIN_ASSIGNMENTS = 40
PERMUTATIONS = 999
ALPHA = 0.05
# A comparison whose sides hold one observation in every repeat: no within-group residual.
ONE_EACH = "not tested: one observation a side and repeat"
COLUMNS = (
    "table",
    "condition",
    "control",
    "repeats",
    "same_way",
    "mean_projection",
    "t",
    "p_repeats",
    "q_repeats",
    "experiment",
    "distance",
    "effect",
    "p_shuffle",
    "shuffled",
    "units",
    "permutations",
    "fields",
    "observations",
    "left_out",
    "status",
)


def projections(deltas: np.ndarray) -> np.ndarray:
    """``s_r = Δ_r · u_−r`` for each repeat (rows of ``deltas``), ``u_−r`` the other
    repeats' summed shift normalised (0 when it has no length)."""
    others = deltas.sum(axis=0) - deltas
    norms = np.linalg.norm(others, axis=1)
    safe = np.where(norms > 0, norms, 1.0)
    return np.where(norms > 0, np.einsum("ij,ij->i", deltas, others) / safe, 0.0)


def repeat_test(deltas: np.ndarray) -> tuple[int, float, float, float]:
    """The repeat-level verdict of ``deltas`` (one row per repeat): the repeats shifted the
    same way, the mean projection, the corrected t and its one-sided p (see the module);
    NaN without ``MIN_REPEATS`` repeats or when the projections do not vary."""
    n = len(deltas)
    if n < 2:
        return 0, float("nan"), float("nan"), float("nan")
    s = projections(deltas)
    same = int((s > 0).sum())
    if n < MIN_REPEATS:
        return same, float(s.mean()), float("nan"), float("nan")
    sd = float(s.std(ddof=1))
    if not sd > 0:
        return same, float(s.mean()), float("nan"), float("nan")
    t = float(s.mean() / (sd / np.sqrt(n)) * np.sqrt((n - 2) / (2 * (n - 1))))
    return same, float(s.mean()), t, float(stats.t.sf(t, n - 1))


def assignments(side: np.ndarray, repeat: np.ndarray) -> int:
    """The distinct ways to label the units of each ``repeat`` with as many of each side."""
    return prod(
        comb(int((repeat == r).sum()), int(side[repeat == r].sum())) for r in np.unique(repeat)
    )


def shuffle_test(
    means: np.ndarray, side: np.ndarray, repeat: np.ndarray, permutations: int, rng: Any
) -> tuple[float, float]:
    """The distance between the two sides' centres (``means``: one row per unit, ``side``
    1 for the condition, 0 for the control) and its p with the labels permuted within each
    ``repeat``, every permutation at once (see the module); no p with fewer than
    ``MIN_ASSIGNMENTS`` distinct assignments."""
    labels = np.tile(side.astype(float), (permutations + 1, 1))
    for value in np.unique(repeat):
        mine = np.flatnonzero(repeat == value)
        labels[1:, mine] = rng.permuted(labels[1:, mine], axis=1)
    n_condition, n_control = side.sum(), len(side) - side.sum()
    weights = labels / n_condition - (1 - labels) / n_control
    distances = np.linalg.norm(weights @ means, axis=1)
    if assignments(side, repeat) < MIN_ASSIGNMENTS:
        return float(distances[0]), float("nan")
    extreme = int((distances[1:] >= distances[0] - 1e-12 * max(distances[0], 1.0)).sum())
    return float(distances[0]), (extreme + 1) / (permutations + 1)


def _mean_by(values: np.ndarray, keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The rows of ``values`` averaged by ``keys``, and each row's group code."""
    codes = np.unique(keys, return_inverse=True)[1].reshape(-1)
    sums = np.zeros((codes.max() + 1, values.shape[1]))
    np.add.at(sums, codes, values)
    return sums / np.bincount(codes)[:, None], codes


@dataclass(frozen=True)
class WholeTable:
    """The whole-table tests: ``tests`` one row per table and comparison (``COLUMNS``), the
    ``settings`` (tables and columns, permutations, the comparisons tested)."""

    tests: pd.DataFrame
    settings: dict[str, Any] = field(default_factory=dict)

    def significant(self, alpha: float = ALPHA) -> pd.DataFrame:
        """The rows whose repeat-level q is at most ``alpha``."""
        return self.tests[self.tests["q_repeats"] <= alpha]


def _scaled(values: np.ndarray, groups: np.ndarray) -> np.ndarray | None:
    """``values`` divided by the robust scale of their within-group residuals, the columns
    that vary within the groups only (see the module); ``None`` when no group of ``groups``
    holds two observations."""
    sizes = np.bincount(groups)
    several = sizes[groups] > 1
    if not several.any():
        return None
    centres, _codes = _mean_by(values, groups)
    _centre, scale = robust_scale((values - centres[groups])[several])
    varies = scale > 0
    return values[:, varies] / scale[varies]


def _table_rows(
    name: str,
    values: np.ndarray,
    meta: pd.DataFrame,
    pairs: Sequence[tuple[Any, Any]],
    permutations: int,
    random_state: int,
    shuffled: str = "field",
) -> list[dict[str, Any]]:
    """One row per comparison for one table (``values``: the complete observations, raw,
    scaled here per comparison; ``meta`` their condition, repeat, field and, when the data
    name one, the unit shuffled, ``shuffled`` its name)."""
    if "unit" not in meta:
        meta = meta.assign(unit=meta["field"])
    field_codes = np.unique(meta["field"].to_numpy(), return_inverse=True)[1].reshape(-1)
    columns = [CONDITION_COLUMN, REPEAT_COLUMN, "unit"]
    field_meta = meta.groupby(field_codes, sort=True)[columns].first()
    condition = field_meta[CONDITION_COLUMN].to_numpy()
    repeat = field_meta[REPEAT_COLUMN].to_numpy()
    unit = field_meta["unit"].to_numpy()
    rows = []
    for treated, control in pairs:
        both = sorted(
            set(repeat[condition == treated]) & set(repeat[condition == control]), key=str
        )
        row: dict[str, Any] = {
            "table": name,
            "condition": treated,
            "control": control,
            "repeats": len(both),
        }
        if not both:
            rows.append({**row, "status": "not tested: no repeat holds both"})
            continue
        observed = np.isin(meta[REPEAT_COLUMN].to_numpy(), both) & np.isin(
            meta[CONDITION_COLUMN].to_numpy(), [treated, control]
        )
        codes = meta.loc[observed].groupby([CONDITION_COLUMN, REPEAT_COLUMN]).ngroup().to_numpy()
        z = _scaled(values[observed], codes)
        if z is None:
            rows.append({**row, "status": ONE_EACH})
            continue
        # Every field mean of the table; the fields outside the comparison stay at 0.
        means = np.zeros((len(field_meta), z.shape[1]))
        means[np.unique(field_codes[observed])] = _mean_by(z, field_codes[observed])[0]
        chosen = np.isin(repeat, both) & np.isin(condition, [treated, control])
        side = (condition[chosen] == treated).astype(int)
        deltas = np.array(
            [
                means[chosen][(repeat[chosen] == r) & (side == 1)].mean(axis=0)
                - means[chosen][(repeat[chosen] == r) & (side == 0)].mean(axis=0)
                for r in both
            ]
        )
        same, projection, t, p = repeat_test(deltas)
        rng = group_rng(random_state, {"table": name, "condition": treated, "control": control})
        unit_means, unit_codes = _mean_by(means[chosen], unit[chosen])
        first = np.unique(unit_codes, return_index=True)[1]
        distance, p_shuffle = shuffle_test(
            unit_means, side[first], repeat[chosen][first], permutations, rng
        )
        centres, _codes = _mean_by(z, codes)
        spread = np.square(z - centres[codes]).sum() / max(1, len(z) - len(centres))
        shift = float(np.linalg.norm(deltas.mean(axis=0)))
        effect = shift / float(np.sqrt(spread)) if spread > 0 else float("nan")
        tested = len(both) >= MIN_REPEATS and np.isfinite(p)
        rows.append(
            {
                **row,
                "same_way": same,
                "mean_projection": projection,
                "t": t,
                "p_repeats": p,
                "distance": distance,
                "effect": effect,
                "p_shuffle": p_shuffle,
                "shuffled": shuffled,
                "units": len(unit_means),
                "permutations": permutations,
                "fields": int(chosen.sum()),
                "observations": int(observed.sum()),
                "status": "tested" if tested else f"not tested: {len(both)} repeats",
            }
        )
    return rows


def _units(data: ValidatedInput) -> tuple[str, list[str]]:
    """The unit the shuffle test permutes and the columns naming it with the condition and
    the repeat: the technical repeat, else the analysis groups beyond the condition and the
    repeat, else the field (no columns)."""
    frame = data.objects
    base = [CONDITION_COLUMN, REPEAT_COLUMN]
    if TECHNICAL_REPEAT_COLUMN in frame and not empty_values(frame[TECHNICAL_REPEAT_COLUMN]).all():
        return TECHNICAL_REPEAT_COLUMN, [*base, TECHNICAL_REPEAT_COLUMN]
    extra = [c for c in data.group_cols if c not in base and c in frame]
    if extra:
        return ", ".join(extra), [*base, *extra]
    return "field", []


def whole_table_tests(
    objects: pd.DataFrame,
    fovs: pd.DataFrame,
    *,
    tables: Mapping[str, Sequence[str]],
    pairs: Sequence[tuple[Any, Any]],
    permutations: int = PERMUTATIONS,
    random_state: int = 0,
    fov_col: str | None = None,
    group_cols: Sequence[str] | None = None,
    observation_unit: str | None = None,
) -> WholeTable:
    """The whole-table tests of every table (``tables``: each name and its columns) for each
    ``(condition, control)`` of ``pairs`` (see the module)."""
    rows: list[dict[str, Any]] = []
    left: dict[str, int] = {}
    for name, columns in tables.items():
        data = validate_input_tables(
            objects,
            fovs=fovs,
            feature_cols=list(columns),
            fov_col=fov_col,
            group_cols=group_cols,
            observation_unit=observation_unit,
        )
        frame = data.objects
        if CONDITION_COLUMN not in frame or REPEAT_COLUMN not in frame:
            raise ValueError("the whole-table test needs the condition and repeat columns")
        matrix = data.feature_matrix()
        complete = ~np.isnan(matrix).any(axis=1)
        left[name] = int((~complete).sum())
        key = frame.loc[complete].groupby(list(data.fov_key), dropna=False).ngroup()
        shuffled, above = _units(data)
        unit = frame.loc[complete].groupby(above, dropna=False).ngroup() if above else key
        meta = pd.DataFrame(
            {
                CONDITION_COLUMN: frame.loc[complete, CONDITION_COLUMN].to_numpy(),
                REPEAT_COLUMN: frame.loc[complete, REPEAT_COLUMN].to_numpy(),
                "field": key.to_numpy(),
                "unit": unit.to_numpy(),
            }
        )
        found = _table_rows(
            name, matrix[complete], meta, pairs, permutations, random_state, shuffled
        )
        rows += [{**row, "left_out": left[name]} for row in found]
    tests = pd.DataFrame(rows).reindex(columns=list(COLUMNS))
    if len(tests):
        # The sub-experiments: the conditions linked by the repeats they were compared in.
        held = fovs.groupby(CONDITION_COLUMN)[REPEAT_COLUMN].agg(set).to_dict()
        numbers = experiments({c: held.get(c, set()) & held.get(k, set()) for c, k in pairs})
        tests["experiment"] = tests["condition"].map(numbers)
        family = tests["table"].astype(str) + "\x1f" + tests["experiment"].astype(str)
        tests["q_repeats"] = corrected(tests["p_repeats"], family)
    settings = {
        "tables": {n: len(c) for n, c in tables.items()},
        "pairs": [tuple(p) for p in pairs],
        "permutations": permutations,
        "random_state": random_state,
        "left_out": left,
        "min_repeats": MIN_REPEATS,
        "min_assignments": MIN_ASSIGNMENTS,
    }
    return WholeTable(tests, settings)


__all__ = [
    "MIN_ASSIGNMENTS",
    "MIN_REPEATS",
    "ONE_EACH",
    "PERMUTATIONS",
    "WholeTable",
    "assignments",
    "projections",
    "repeat_test",
    "shuffle_test",
    "whole_table_tests",
]
