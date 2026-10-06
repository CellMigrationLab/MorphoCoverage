"""The whole-table test: is the repeat-level verdict calibrated, and how often does it find a
shift? (owner, 2026-10-06)

``sampling_unit.whole_table`` tests whether a condition's centre in a feature table moves
away from its control's the same way in every biological repeat: the repeat shifts
``Δ_r``, their projections ``s_r`` on the other repeats' mean direction, and a one-sided
paired t-test on the ``s_r``; and, exploratory, the distance between the centres with the
labels of the fields permuted within each repeat. This simulates the null (no condition
effect) and planted shifts over designs:

- 3, 4 and 6 biological repeats;
- one image per field (6 to a side and repeat, as the ECM tables) or 20 cells in each of 5
  fields, the field adding a shift of SD ``FIELD_SD``;
- 50 and 384 columns, correlated as an embedding's are: independent normal columns whose
  scales fall as ``j^-0.7``, mixed by a random rotation;
- no batch, or a batch: every repeat shifted (SD ``REPEAT_SD``, both sides alike) and each
  side of a repeat its own well shift (SD ``WELL_SD``, the condition-by-repeat interaction
  a paired test across repeats allows for and a pooled test does not);
- even counts, or uneven: each side and repeat 2 to 10 fields;
- planted shifts: the condition moved by ``effect`` within-group SDs (the root of the summed
  variances around a group's centre) in one random direction, in every repeat.

Each setting is ``N_REPLICATES`` datasets; ``rate`` is the share with p <= 0.05, ``se`` its
standard error. The plain paired t on the ``s_r`` (``plain``) is given beside the corrected
one the package uses (``repeats``), for the reason of the correction, and the image-level
test (``image``). The tests are the package's own (``whole_table._table_rows``) on the
generated observations, scaled as the package does: on the within-group residuals of the
comparison's two sides (owner, 2026-10-06, audit 8).

Results: ``results/whole_table.csv``. Usage, from the repository root::

    PYTHONPATH=src python data/simulations/whole_table.py [workers]
"""

from __future__ import annotations

import sys
import zlib
from itertools import product
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from morphocoverage.metadata import CONDITION_COLUMN, REPEAT_COLUMN
from morphocoverage.sampling_unit.whole_table import _scaled, _table_rows, projections

N_REPLICATES = 400
PERMUTATIONS = 199
FIELD_SD = 0.5
REPEAT_SD = 1.0
WELL_SD = 0.3
REPEATS = (3, 4, 6)
UNITS = {"image per field": (6, 1), "cells per field": (5, 20)}
DIMENSIONS = (50, 384)
BATCH = (False, True)
UNEVEN = (False, True)
EFFECTS = (0.0, 0.3, 0.6)


def _mixing(dimensions: int) -> np.ndarray:
    """Column scales falling as ``j^-0.7``, mixed by a random rotation (one per dimension)."""
    rng = np.random.default_rng(dimensions)
    rotation = np.linalg.qr(rng.normal(size=(dimensions, dimensions)))[0]
    return (np.arange(1, dimensions + 1) ** -0.7)[:, None] * rotation


MIXING = {d: _mixing(d) for d in DIMENSIONS}


def _dataset(
    repeats: int, unit: str, dimensions: int, batch: bool, uneven: bool, effect: float, seed: int
) -> tuple[np.ndarray, pd.DataFrame]:
    """The observations and their condition, repeat and field for one setting."""
    rng = np.random.default_rng(seed)
    fields, cells = UNITS[unit]
    mixing = MIXING[dimensions]
    # The within-group SD (the root of the summed variances) with fields and cells.
    spread = np.sqrt(np.square(mixing).sum() * (1 + (FIELD_SD**2 if cells > 1 else 0)))
    direction = rng.normal(size=dimensions)
    shift = effect * spread * direction / np.linalg.norm(direction)
    meta, offsets = [], []
    for r in range(repeats):
        repeat_shift = rng.normal(size=dimensions) * REPEAT_SD if batch else np.zeros(dimensions)
        for side in ("ctrl", "cond"):
            well = rng.normal(size=dimensions) * WELL_SD if batch else np.zeros(dimensions)
            n_fields = int(rng.integers(2, 11)) if uneven else fields
            for f in range(n_fields):
                field = rng.normal(size=dimensions) * FIELD_SD if cells > 1 else 0.0
                offsets += [repeat_shift + well + field] * cells
                meta += [(side, f"R{r}", f"{side}{r}_{f}")] * cells
    frame = pd.DataFrame(meta, columns=[CONDITION_COLUMN, REPEAT_COLUMN, "field"])
    # Every source of variation on the embedding's correlated axes, the shift on its own.
    values = (rng.normal(size=(len(frame), dimensions)) + np.array(offsets)) @ mixing
    return values + np.outer(frame[CONDITION_COLUMN].eq("cond"), shift), frame


def _plain_p(z: np.ndarray, meta: pd.DataFrame) -> float:
    """The plain one-sided paired t on the projections, uncorrected (for the comparison)."""
    codes = np.unique(meta["field"], return_inverse=True)[1].reshape(-1)
    means = pd.DataFrame(z).groupby(codes).mean().to_numpy()
    first = meta.groupby(codes)[[CONDITION_COLUMN, REPEAT_COLUMN]].first()
    deltas = []
    for r in sorted(first[REPEAT_COLUMN].unique()):
        here = first[REPEAT_COLUMN].to_numpy() == r
        side = first[CONDITION_COLUMN].to_numpy()
        deltas.append(
            means[here & (side == "cond")].mean(0) - means[here & (side == "ctrl")].mean(0)
        )
    s = projections(np.array(deltas))
    return float(stats.ttest_1samp(s, 0, alternative="greater").pvalue)


def _setting(arguments: tuple) -> dict:
    repeats, unit, dimensions, batch, uneven, effect = arguments
    hits = {"repeat_level": 0, "plain": 0, "image_level": 0}
    for replicate in range(N_REPLICATES):
        seed = zlib.crc32(
            repr((repeats, unit, dimensions, batch, uneven, effect, replicate)).encode()
        )
        values, meta = _dataset(repeats, unit, dimensions, batch, uneven, effect, seed)
        row = _table_rows("t", values, meta, [("cond", "ctrl")], PERMUTATIONS, replicate)[0]
        hits["repeat_level"] += row["p_repeats"] <= 0.05
        hits["image_level"] += row["p_shuffle"] <= 0.05
        groups = meta.groupby([CONDITION_COLUMN, REPEAT_COLUMN]).ngroup().to_numpy()
        hits["plain"] += _plain_p(_scaled(values, groups), meta) <= 0.05
    rows = {}
    for test, hit in hits.items():
        rate = hit / N_REPLICATES
        rows[test] = rate
        rows[f"{test}_se"] = float(np.sqrt(rate * (1 - rate) / N_REPLICATES))
    return {
        "repeats": repeats,
        "unit": unit,
        "dimensions": dimensions,
        "batch": batch,
        "uneven": uneven,
        "effect": effect,
        **rows,
    }


def main(workers: int = 1) -> pd.DataFrame:
    """Every setting; the null for every design, the planted shifts without uneven counts."""
    settings = [
        s
        for s in product(REPEATS, UNITS, DIMENSIONS, BATCH, UNEVEN, EFFECTS)
        if s[-1] == 0.0 or not s[4]
    ]
    with Pool(workers) as pool:
        rows = pool.map(_setting, settings)
    table = pd.DataFrame(rows)
    out = Path(__file__).parent / "results" / "whole_table.csv"
    table.round(4).to_csv(out, index=False)
    return table


if __name__ == "__main__":
    print(main(int(sys.argv[1]) if len(sys.argv) > 1 else 1).to_string())
