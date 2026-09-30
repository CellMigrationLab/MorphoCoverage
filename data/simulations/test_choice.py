"""Which test for "do the conditions differ?": paired t on repeat means, or a randomization test?

Data like the lab's experiments: R biological repeats; in each, one well per condition (control,
treated) with n cells. Cell values (in cell-SD units, on a baseline of ``LEVEL``):

    y = gain_r * (LEVEL + batch_r + well_rc + (delta + e_r) * treated + noise),  noise ~ N(0, 1)

gain_r (lognormal, SD of its log ``GAIN_SD``) and batch_r ~ N(0, 0.5) are shared by both wells of
a repeat (a change of scale such as laser power or camera gain; a staining or plating shift);
well_rc ~ N(0, s_well) and e_r ~ N(0, s_effect) make the difference vary between repeats beyond
cell noise. The SD of that repeat-level variation, s_rep = sqrt(s_effect^2 + 2 s_well^2), is the
"SD between repeats" of the report's plan (0.3-0.5 in the lab datasets).

Tests, each at alpha = 0.05:
- paired_t: paired t-test on the per-repeat means (SuperPlots; from 2 repeats);
- t_cells: an unpaired t-test with every cell an independent measurement;
- rand_pooled: |Cohen's d| on all cells pooled, labels shuffled over all cells (Plot&Stats);
- rand_within: the same statistic, labels shuffled within each repeat (cells still the unit);
- rand_minus_ctrl, rand_over_ctrl: the Plot&Stats test on the values normalised to the same
  repeat's control mean, by subtraction or by division (what normalisation removes: the shift
  and the scale both wells share, not the condition-specific part);
- paired_t_ratio: a paired t-test on log(treated mean / control mean) per repeat;
- signflip_d: exact sign-flip test on the per-repeat Cohen's d (repeats the unit);
- lmm_z: a linear mixed model on every cell, ``y ~ treated`` with a random intercept and a
  random treated effect per repeat (REML, statsmodels), its Wald z-test of the treated effect;
- lmm_t: the same fit, its treated effect tested with a t distribution on ``repeats - 1``
  degrees of freedom (the conservative small-sample reading);
- hboot: a hierarchical bootstrap (Saravanan et al. 2020): repeats resampled with replacement,
  then each well's cells with replacement, ``N_BOOT`` times; p is twice the share of
  resampled mean differences on the far side of 0.

The mixed model needs statsmodels, which the package does not depend on; the simulation only.

Written to ``results/test_choice.csv`` (one row per setting: the share of the ``N_SIMS``
experiments with p <= 0.05; at delta = 0 the false-positive rate, else the power), with
``k_approx``: the false-positive rate of a cell-level test from the variance inflation
``k = 1 + cells * s_rep**2 / 2`` (a cell-level test takes the mean difference's variance as
``2 / (cells * repeats)``; with repeats differing it is ``(2 / cells + s_rep**2) / repeats``).
Usage, from the repository root::

    python data/simulations/test_choice.py [workers]
"""

from __future__ import annotations

import itertools
import sys
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

ALPHA = 0.05
N_SIMS = 1000
N_PERM = 999
BATCH_SD = 0.5
LEVEL = 10.0
GAIN_SD = 0.1
N_BOOT = 1000


def cohen_d(a, b, axis=-1):
    na, nb = a.shape[axis], b.shape[axis]
    va, vb = a.var(axis=axis, ddof=1), b.var(axis=axis, ddof=1)
    pooled = ((na - 1) * va + (nb - 1) * vb) / (na + nb - 2)
    return (a.mean(axis=axis) - b.mean(axis=axis)) / np.sqrt(pooled)


def one_experiment(rng, repeats, cells, delta, s_rep):
    # Split s_rep into a well part and an effect part (half the variance each).
    s_well = np.sqrt(s_rep**2 / 2 / 2)
    s_effect = np.sqrt(s_rep**2 / 2)
    gain = np.exp(rng.normal(0, GAIN_SD, repeats))[:, None]
    batch = LEVEL + rng.normal(0, BATCH_SD, repeats)
    wells = rng.normal(0, s_well, (repeats, 2))
    effect = delta + rng.normal(0, s_effect, repeats)
    control = gain * (batch[:, None] + wells[:, [0]] + rng.normal(0, 1, (repeats, cells)))
    treated = gain * (
        batch[:, None] + wells[:, [1]] + effect[:, None] + rng.normal(0, 1, (repeats, cells))
    )
    return control, treated


def pooled_randomization(rng, control, treated):
    """Plot&Stats: |Cohen's d| on every cell pooled, labels shuffled over all cells."""
    pooled = np.concatenate([control.ravel(), treated.ravel()])
    n = control.size
    observed = abs(cohen_d(treated.ravel(), control.ravel()))
    perms = np.array([rng.permutation(pooled.size) for _ in range(N_PERM)])
    shuffled = pooled[perms]
    null = np.abs(cohen_d(shuffled[:, n:], shuffled[:, :n]))
    return (np.sum(null >= observed - 1e-12) + 1) / (N_PERM + 1)


def tests(rng, control, treated):
    repeats, cells = control.shape
    out = {}
    diffs = treated.mean(1) - control.mean(1)
    if repeats >= 2:
        out["paired_t"] = stats.ttest_1samp(diffs, 0.0).pvalue if np.ptp(diffs) > 0 else 1.0
    out["t_cells"] = stats.ttest_ind(treated.ravel(), control.ravel()).pvalue
    out["rand_pooled"] = pooled_randomization(rng, control, treated)
    reference = control.mean(axis=1, keepdims=True)
    out["rand_minus_ctrl"] = pooled_randomization(rng, control - reference, treated - reference)
    out["rand_over_ctrl"] = pooled_randomization(rng, control / reference, treated / reference)
    if repeats >= 2:
        logs = np.log(treated.mean(1) / control.mean(1))
        out["paired_t_ratio"] = stats.ttest_1samp(logs, 0.0).pvalue if np.ptp(logs) > 0 else 1.0
    observed = abs(cohen_d(treated.ravel(), control.ravel()))
    # Shuffled within each repeat, same pooled statistic.
    both = np.concatenate([control, treated], axis=1)  # (repeats, 2 cells)
    keys = rng.random((N_PERM, repeats, 2 * cells)).argsort(axis=2)
    within = np.take_along_axis(np.broadcast_to(both, keys.shape), keys, axis=2)
    null_w = np.abs(
        cohen_d(within[:, :, cells:].reshape(N_PERM, -1), within[:, :, :cells].reshape(N_PERM, -1))
    )
    out["rand_within"] = (np.sum(null_w >= observed - 1e-12) + 1) / (N_PERM + 1)
    # Sign-flip on per-repeat d, exact.
    d_r = cohen_d(treated, control)
    signs = np.array(list(itertools.product((1.0, -1.0), repeat=repeats)))
    flips = np.abs((signs * d_r).mean(1))
    out["signflip_d"] = float(np.mean(flips >= abs(d_r.mean()) - 1e-12))
    if repeats >= 2:
        out["lmm_z"], out["lmm_t"] = mixed_model(control, treated)
    out["hboot"] = hierarchical_bootstrap(rng, control, treated)
    return out


def mixed_model(control, treated):
    """The treated effect's p-values from ``y ~ treated`` with a random intercept and a random
    treated effect per repeat: Wald z, and t on ``repeats - 1`` degrees of freedom. A fit
    that fails counts as no difference found (p = 1)."""
    repeats, cells = control.shape
    frame = pd.DataFrame(
        {
            "y": np.concatenate([control.ravel(), treated.ravel()]),
            "treated": np.repeat([0.0, 1.0], repeats * cells),
            "repeat": np.tile(np.repeat(np.arange(repeats), cells), 2),
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            fit = smf.mixedlm("y ~ treated", frame, groups="repeat", re_formula="~treated").fit(
                reml=True
            )
            effect, se = float(fit.fe_params["treated"]), float(fit.bse_fe["treated"])
        except (ValueError, np.linalg.LinAlgError):
            return 1.0, 1.0
    if not (np.isfinite(effect) and np.isfinite(se)) or se <= 0:
        return 1.0, 1.0
    z = abs(effect / se)
    return float(2 * stats.norm.sf(z)), float(2 * stats.t.sf(z, repeats - 1))


def hierarchical_bootstrap(rng, control, treated):
    """Twice the share of resampled mean differences on the far side of 0 (repeats, then each
    well's cells, with replacement), never below ``1 / N_BOOT``."""
    repeats, cells = control.shape
    chosen = rng.integers(repeats, size=(N_BOOT, repeats))
    rows = []
    for side in (control, treated):
        picked = side[chosen]  # (N_BOOT, repeats, cells)
        cells_idx = rng.integers(cells, size=picked.shape)
        rows.append(np.take_along_axis(picked, cells_idx, axis=2).mean(axis=(1, 2)))
    diff = rows[1] - rows[0]
    tail = min(np.mean(diff <= 0), np.mean(diff >= 0))
    return max(2 * tail, 1 / N_BOOT)


def run(setting):
    repeats, cells, delta, s_rep, seed = setting
    rng = np.random.default_rng(seed)
    names = (
        "paired_t",
        "t_cells",
        "rand_pooled",
        "rand_within",
        "rand_minus_ctrl",
        "rand_over_ctrl",
        "paired_t_ratio",
        "signflip_d",
        "lmm_z",
        "lmm_t",
        "hboot",
    )
    hits = dict.fromkeys(names, 0)
    for _ in range(N_SIMS):
        control, treated = one_experiment(rng, repeats, cells, delta, s_rep)
        for name, p in tests(rng, control, treated).items():
            hits[name] += p <= ALPHA
    inflation = 1 + cells * s_rep**2 / 2
    approx = 2 * stats.norm.sf(stats.norm.isf(ALPHA / 2) / np.sqrt(inflation))
    return {
        "repeats": repeats,
        "cells": cells,
        "delta": delta,
        "s_rep": s_rep,
        **{
            k: (
                v / N_SIMS
                if repeats >= 2 or k not in ("paired_t", "paired_t_ratio", "lmm_z", "lmm_t")
                else np.nan
            )
            for k, v in hits.items()
        },
        "k_approx": approx if delta == 0 else np.nan,
    }


def main():
    grid = [
        (r, c, d, s)
        for r in (1, 2, 3, 4, 6)
        for c in (30,)
        for d in (0.0, 0.5, 1.0, 1.5, 2.0)
        for s in (0.0, 0.25, 0.5)
    ]
    settings = [(*g, 1000 + i) for i, g in enumerate(grid)]
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 4) as pool:
        rows = pool.map(run, settings)
    table = pd.DataFrame(rows)
    out = Path(__file__).resolve().parent / "results" / "test_choice.csv"
    table.round(4).to_csv(out, index=False, lineterminator="\n")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
