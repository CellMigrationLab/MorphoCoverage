"""Unbalanced repeats: does balancing (resampling) or normalising keep the cell-level test honest?

The same experiments as ``test_choice.py`` (one well per condition and repeat, the generator
``y = gain_r * (LEVEL + batch_r + well_rc + (delta + e_r) * treated + noise)``), with 3
biological repeats that do not hold the same number of cells. Scenarios (cells per repeat, the
same on both sides unless said):

- ``equal``: 30, 30, 30 (the reference);
- ``one 2x`` / ``one 4x`` / ``one 8x``: one repeat holds 60, 120 or 240 cells, the others 30;
- ``control +3 repeats``: 30 a side in each of the 3 repeats, and the control also imaged in 3
  more repeats (30 cells each) the condition was not, pooled into the control as provided (the
  screen of ``data/lab``: each set's control from every repeat);
- ``one sparse``: 100, 100 and 8 cells (the third repeat under a quarter of the largest).

Tests, each at alpha = 0.05:

- ``pooled`` / ``pooled_norm``: Plot&Stats as provided, |Cohen's d| on every cell, labels
  shuffled over the cells (``N_PERM``), on raw values and on values normalised to the mean of
  the same repeat's control (each repeat's control too, the extra control repeats included);
- ``balanced`` / ``balanced_norm``: the package's balanced test
  (``morphocoverage.sampling_unit.contrast_cells``): each condition against the control of its
  own repeats, the repeats too small to balance to left out (``sparse``), the same number of
  cells from every repeat kept (the fewest), the median p over ``BALANCED_DRAWS`` draws. For
  speed each draw's p is the cell t-test's, which in ``test_choice.csv`` gives the
  randomization test's rate within 1 point in every setting;
- ``paired_t`` / ``paired_t_ratio``: the paired t-test on the repeat means and on the log ratios
  to the control, one value per repeat whatever its cells (the extra control repeats unpaired,
  so not used).

Also written: the mean d of each cell-level version (``d_*``, the treated minus control
difference in pooled observation SDs; at delta = 1 the effect is 1 cell SD), and ``k_approx``:
the false-positive rate the report gives for the test as provided,
``2 P(Z > 1.96 / sqrt(1 + n_eff s^2 / 2))`` with ``n_eff = sum(n_r^2) / sum(n_r)`` (the cells
per repeat weighted by their own share: ``n`` when the repeats are equal), and ``k_balanced``,
the same with the balanced test's cells per repeat.

Results: ``results/unbalanced.csv`` (one row per scenario, delta and s_rep; the share of the
``N_SIMS`` experiments with p <= 0.05), appended setting by setting to
``results/unbalanced.partial.csv`` as ``test_choice.py`` does. Usage, from the repository
root::

    PYTHONPATH=src python data/simulations/unbalanced.py [workers]
"""

from __future__ import annotations

import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_choice import ALPHA, BATCH_SD, GAIN_SD, LEVEL, N_PERM, N_SIMS, cohen_d

from morphocoverage.sampling_unit.contrast_cells import BALANCED_DRAWS, sparse

REPEATS = 3
SCENARIOS = {
    "equal": ((30, 30, 30), 0),
    "one 2x": ((60, 30, 30), 0),
    "one 4x": ((120, 30, 30), 0),
    "one 8x": ((240, 30, 30), 0),
    "control +3 repeats": ((30, 30, 30), 3),
    "one sparse": ((100, 100, 8), 0),
}
OTHER_CELLS = 30
TESTS = (
    "pooled",
    "pooled_norm",
    "balanced",
    "balanced_norm",
    "paired_t",
    "paired_t_ratio",
)
KEY = ["scenario", "delta", "s_rep"]
RESULTS = Path(__file__).resolve().parent / "results"


def one_experiment(rng, sizes, others, delta, s_rep):
    """Per repeat, the control's and the treated well's cells (lists of arrays), and the cells
    of the extra control repeats; the generator of ``test_choice.one_experiment``."""
    s_well = np.sqrt(s_rep**2 / 2 / 2)
    s_effect = np.sqrt(s_rep**2 / 2)
    control, treated = [], []
    for n in sizes:
        gain = np.exp(rng.normal(0, GAIN_SD))
        batch = LEVEL + rng.normal(0, BATCH_SD)
        wells = rng.normal(0, s_well, 2)
        effect = delta + rng.normal(0, s_effect)
        control.append(gain * (batch + wells[0] + rng.normal(0, 1, n)))
        treated.append(gain * (batch + wells[1] + effect + rng.normal(0, 1, n)))
    extra = []
    for _ in range(others):
        gain = np.exp(rng.normal(0, GAIN_SD))
        batch = LEVEL + rng.normal(0, BATCH_SD)
        extra.append(gain * (batch + rng.normal(0, s_well) + rng.normal(0, 1, OTHER_CELLS)))
    return control, treated, extra


def randomization(rng, treated, control):
    """``(d, p)``: Cohen's d on every cell and the randomization p, labels shuffled over them."""
    pooled = np.concatenate([treated, control])
    n = len(treated)
    d = cohen_d(treated, control)
    perms = np.argsort(rng.random((N_PERM, len(pooled))), axis=1)
    shuffled = pooled[perms]
    null = np.abs(cohen_d(shuffled[:, :n], shuffled[:, n:]))
    return float(d), (np.sum(null >= abs(d) - 1e-12) + 1) / (N_PERM + 1)


def balanced(rng, treated, control):
    """``(median d, median p)`` over ``BALANCED_DRAWS`` draws of the same number of cells from
    every repeat that is not too small to balance to (the package's ``sparse``)."""
    small = set(sparse(dict(enumerate(map(len, treated))))) | set(
        sparse(dict(enumerate(map(len, control))))
    )
    kept = [r for r in range(len(treated)) if r not in small]
    per_t = min(len(treated[r]) for r in kept)
    per_c = min(len(control[r]) for r in kept)
    picks_t = np.concatenate(
        [
            treated[r][np.argsort(rng.random((BALANCED_DRAWS, len(treated[r]))), axis=1)[:, :per_t]]
            for r in kept
        ],
        axis=1,
    )
    picks_c = np.concatenate(
        [
            control[r][np.argsort(rng.random((BALANCED_DRAWS, len(control[r]))), axis=1)[:, :per_c]]
            for r in kept
        ],
        axis=1,
    )
    d = cohen_d(picks_t, picks_c)
    p = stats.ttest_ind(picks_t, picks_c, axis=1).pvalue
    return float(np.median(d)), float(np.median(p))


def tests(rng, control, treated, extra):
    out = {}
    every_control = control + extra
    d, out["pooled"] = randomization(rng, np.concatenate(treated), np.concatenate(every_control))
    out["d_pooled"] = d
    norm_t = [t / c.mean() for t, c in zip(treated, control, strict=True)]
    norm_c = [c / c.mean() for c in control]
    norm_every = norm_c + [x / x.mean() for x in extra]
    d, out["pooled_norm"] = randomization(rng, np.concatenate(norm_t), np.concatenate(norm_every))
    out["d_pooled_norm"] = d
    out["d_balanced"], out["balanced"] = balanced(rng, treated, control)
    out["d_balanced_norm"], out["balanced_norm"] = balanced(rng, norm_t, norm_c)
    means_t = np.array([t.mean() for t in treated])
    means_c = np.array([c.mean() for c in control])
    out["paired_t"] = stats.ttest_1samp(means_t - means_c, 0.0).pvalue
    out["paired_t_ratio"] = stats.ttest_1samp(np.log(means_t / means_c), 0.0).pvalue
    return out


def chance(n_eff, s_rep):
    return float(2 * stats.norm.sf(stats.norm.isf(ALPHA / 2) / np.sqrt(1 + n_eff * s_rep**2 / 2)))


def run(setting):
    scenario, delta, s_rep, seed = setting
    sizes, others = SCENARIOS[scenario]
    rng = np.random.default_rng(seed)
    hits = dict.fromkeys(TESTS, 0)
    ds = {f"d_{name}": [] for name in ("pooled", "pooled_norm", "balanced", "balanced_norm")}
    for _ in range(N_SIMS):
        control, treated, extra = one_experiment(rng, sizes, others, delta, s_rep)
        for name, value in tests(rng, control, treated, extra).items():
            if name in ds:
                ds[name].append(value)
            else:
                hits[name] += value <= ALPHA
    sizes_array = np.array(sizes, float)
    small = set(sparse(dict(enumerate(sizes))))
    kept = [n for r, n in enumerate(sizes) if r not in small]
    return {
        "scenario": scenario,
        "cells": "/".join(map(str, sizes)) + (f" +{others}x{OTHER_CELLS}" if others else ""),
        "delta": delta,
        "s_rep": s_rep,
        **{name: count / N_SIMS for name, count in hits.items()},
        **{name: float(np.mean(values)) for name, values in ds.items()},
        "k_approx": chance((sizes_array**2).sum() / sizes_array.sum(), s_rep),
        "k_balanced": chance(min(kept), s_rep),
    }


def main():
    """Every setting, the false-positive ones (delta 0) first, each saved as it finishes (see
    ``test_choice.main``)."""
    grid = [(name, d, s) for name in SCENARIOS for d in (0.0, 1.0) for s in (0.0, 0.25, 0.5)]
    settings = [(*g, 5000 + i) for i, g in enumerate(grid)]
    partial = RESULTS / "unbalanced.partial.csv"
    done = set()
    if partial.exists():
        done = {(s, float(d), float(r)) for s, d, r in pd.read_csv(partial)[KEY].to_numpy()}
    todo = [x for x in settings if (x[0], float(x[1]), float(x[2])) not in done]
    todo.sort(key=lambda x: (x[1], list(SCENARIOS).index(x[0]), x[2]))
    print(f"{len(settings) - len(todo)} settings done, {len(todo)} to run", flush=True)
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 4) as pool:
        for row in pool.imap_unordered(run, todo):
            frame = pd.DataFrame([row]).round(4)
            frame.to_csv(
                partial, mode="a", header=not partial.exists(), index=False, lineterminator="\n"
            )
            print(", ".join(f"{k}={row[k]}" for k in KEY), flush=True)
    order = {name: i for i, name in enumerate(SCENARIOS)}
    table = pd.read_csv(partial)
    table = table.sort_values(
        ["delta", "scenario", "s_rep"], key=lambda c: c.map(order) if c.name == "scenario" else c
    )
    table.reset_index(drop=True).to_csv(
        RESULTS / "unbalanced.csv", index=False, lineterminator="\n"
    )
    partial.unlink()
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
