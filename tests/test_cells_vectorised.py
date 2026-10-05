"""The cell-level tests computed at once (2026-10-05): the same d and p as the loops they
replaced, kept below as the reference, and the same feature discovery for every ``n_jobs``.

The balanced test (#89) took 35 minutes of the idr0028 report on one CPU: a copy of every
draw for every labelling. ``sampling_unit._balanced`` computes every labelling's statistic
by matrix products from the same random keys, and ``cell_randomization`` finds each
labelling's treated side by a partition instead of a sort, from the same keys.
"""

import numpy as np
import pandas as pd
import pytest

from morphocoverage.sampling_unit import _balanced, contrast_cells, feature_discovery
from morphocoverage.sampling_unit.contrast_cells import balanced_draws, cell_randomization
from morphocoverage.statistics import cohen_d

# The reference: the loops of commit d080bcc, before the change.


def _old_cell_randomization(treated, control, n_permutations, rng):
    d = cohen_d(treated, control)
    if not np.isfinite(d):
        return float(d), np.nan
    pooled = np.concatenate([treated, control])
    n = len(treated)
    extreme = 0
    for start in range(0, n_permutations, 256):
        block = min(256, n_permutations - start)
        shuffled = pooled[np.argsort(rng.random((block, len(pooled))), axis=1)]
        a, b = shuffled[:, :n], shuffled[:, n:]
        pooled_var = ((n - 1) * a.var(axis=1, ddof=1) + (len(b[0]) - 1) * b.var(axis=1, ddof=1)) / (
            len(pooled) - 2
        )
        null = np.abs(a.mean(axis=1) - b.mean(axis=1)) / np.sqrt(pooled_var)
        extreme += int(np.sum(null >= abs(d) - 1e-12))
    return float(d), (extreme + 1) / (n_permutations + 1)


def _old_picks(sizes, keep, draws, rng):
    if all(n == keep for n in sizes):
        return [np.arange(n)[None, :] for n in sizes]
    return [np.argsort(rng.random((draws, n)), axis=1)[:, :keep] for n in sizes]


def _old_median_d(pools, treated_sizes, picks, normalise, orders):
    treated, control = [], []
    for index, (pool, n_t) in enumerate(zip(pools, treated_sizes, strict=True)):
        labelled = pool[None, :] if orders is None else pool[orders[index]]
        mine, theirs = labelled[:, :n_t], labelled[:, n_t:]
        if normalise:
            mean = theirs.mean(axis=1, keepdims=True)
            mine, theirs = mine / mean, theirs / mean
        treated.append(mine[:, picks[0][index]])
        control.append(theirs[:, picks[1][index]])
    a, b = np.concatenate(treated, axis=2), np.concatenate(control, axis=2)
    n_a, n_b = a.shape[2], b.shape[2]
    pooled = ((n_a - 1) * a.var(axis=2, ddof=1) + (n_b - 1) * b.var(axis=2, ddof=1)) / (
        n_a + n_b - 2
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        d = np.where(pooled > 0, (a.mean(axis=2) - b.mean(axis=2)) / np.sqrt(pooled), np.nan)
    finite = np.isfinite(d).any(axis=1)
    median = np.full(len(d), np.nan)
    median[finite] = np.nanmedian(d[finite], axis=1)
    return median


def _old_balanced_p(pools, treated_sizes, picks, normalise, observed, n_permutations, rng):
    if not np.isfinite(observed):
        return np.nan
    width = sum(len(pool) for pool in pools) * max(len(picks[0][0]), len(picks[1][0]))
    block = max(1, min(n_permutations, 4_000_000 // width))
    extreme = 0
    for start in range(0, n_permutations, block):
        count = min(block, n_permutations - start)
        orders = [np.argsort(rng.random((count, len(pool))), axis=1) for pool in pools]
        null = _old_median_d(pools, treated_sizes, picks, normalise, orders)
        extreme += int(np.sum(~(np.abs(null) < abs(observed) - 1e-12)))
    return (extreme + 1) / (n_permutations + 1)


def _old_balanced_draws(treated, control, rng, normalise, draws, n_permutations):
    per_t, per_c = min(len(t) for t in treated), min(len(c) for c in control)
    if per_t < 2 or per_c < 2:
        return np.nan, np.nan, per_t, per_c, lambda count, generator: np.nan
    pools = [np.concatenate([t, c]) for t, c in zip(treated, control, strict=True)]
    sizes = [len(t) for t in treated]
    picks = (
        _old_picks(sizes, per_t, draws, rng),
        _old_picks([len(c) for c in control], per_c, draws, rng),
    )
    observed = float(_old_median_d(pools, sizes, picks, normalise, None)[0])

    def rerun(count, generator):
        return _old_balanced_p(pools, sizes, picks, normalise, observed, count, generator)

    return observed, rerun(n_permutations, rng), per_t, per_c, rerun


# Treated and control observations per repeat, the level and the effect: unbalanced, a side
# drawn whole, both whole (a stratified permutation test), one repeat, a level far from 0,
# and repeats large enough that the earlier loop drew its keys in several blocks.
DESIGNS = {
    "unbalanced": ((36, 12, 12), (20, 20, 20), 10.0, 0.4),
    "treated whole": ((15, 15), (20, 31), 5.0, 0.8),
    "both whole": ((10, 10, 10), (12, 12, 12), 3.0, 0.0),
    "one repeat": ((40,), (25,), 8.0, 1.5),
    "far from 0": ((30, 18), (22, 26), 1000.0, 0.05),
    "several blocks": ((900, 600), (300, 200), 50.0, 0.1),
}


def _sides(design, seed):
    treated_sizes, control_sizes, level, effect = design
    rng = np.random.default_rng(seed)
    gains = np.exp(rng.normal(0, 0.2, len(treated_sizes)))
    treated = [g * (level + effect + rng.normal(0, 1, n)) for g, n in zip(gains, treated_sizes)]
    control = [g * (level + rng.normal(0, 1, n)) for g, n in zip(gains, control_sizes)]
    return treated, control


@pytest.mark.parametrize("name", DESIGNS)
@pytest.mark.parametrize("normalise", [False, True])
def test_the_balanced_test_gives_the_loops_d_and_p(name, normalise, monkeypatch):
    """Same d within 1e-12, same p, same p once rerun with more labellings, given the same
    generator; with few values a block the labellings come in several blocks."""
    draws = 100 if name == "several blocks" else 20
    monkeypatch.setattr(contrast_cells, "BALANCED_DRAWS", draws)
    monkeypatch.setattr(contrast_cells, "BALANCED_PERMUTATIONS", 199)
    if name == "several blocks":
        monkeypatch.setattr(_balanced, "_VALUES_PER_BLOCK", 50_000)
    treated, control = _sides(DESIGNS[name], 3)
    old = _old_balanced_draws(treated, control, np.random.default_rng(5), normalise, draws, 199)
    new = balanced_draws(treated, control, np.random.default_rng(5), normalise)
    assert new[0] == pytest.approx(old[0], abs=1e-12)
    assert new[1:4] == old[1:4]
    assert new[4](301, np.random.default_rng(6)) == old[4](301, np.random.default_rng(6))


@pytest.mark.parametrize("seed", range(12))
def test_the_cell_randomization_gives_the_loops_p(seed):
    """Same d and p, given the same generator: either side the smaller, several blocks of
    labellings, and values with ties (a statistic equal to the observed one counts)."""
    rng = np.random.default_rng(seed)
    n_t, n_c = (int(n) for n in rng.integers(2, 1500, 2))
    treated = 100 + rng.normal(0.2, 1, n_t)
    control = 100 + rng.normal(0, 1, n_c)
    if seed % 3 == 0:
        treated, control = np.round(treated), np.round(control)
    shuffles = 2999 if seed % 2 else 99
    old = _old_cell_randomization(treated, control, shuffles, np.random.default_rng(seed))
    assert cell_randomization(treated, control, shuffles, np.random.default_rng(seed)) == old


def test_the_discovery_is_the_same_for_every_n_jobs():
    """The features are tested in worker processes, each test seeded from its feature and
    condition, and the reruns too (60 tests: the balanced p-values at or below 0.05 rerun
    with 1200 labellings)."""
    rng = np.random.default_rng(1)
    rows = [
        {"condition": condition, "biological_repeat": f"r{repeat}", "effect": effect}
        for repeat in range(3)
        for condition, effect in (("ctrl", 0.0), ("drug", 1.0))
        for _ in range(120 if repeat == 0 else 30)
    ]
    objects = pd.DataFrame(rows)
    objects["position_id"] = [f"field{i}" for i in range(len(objects))]
    objects["area"] = 10 + objects.pop("effect") + rng.normal(0, 1, len(objects))
    features = ["area", *(f"null{i}" for i in range(59))]
    objects = objects.assign(**{name: rng.normal(5, 1, len(objects)) for name in features[1:]})
    fovs = objects[["condition", "biological_repeat", "position_id"]]
    runs = [
        feature_discovery(
            objects,
            fovs,
            features=features,
            control={"condition": "ctrl"},
            n_permutations=99,
            observation_unit="cell",
            n_jobs=n_jobs,
        ).tests
        for n_jobs in (1, 2)
    ]
    pd.testing.assert_frame_equal(*runs)
    assert runs[0]["balanced"].all() and 1200 in set(runs[0]["permutations"])


def _repeats(values):
    return {repeat: np.asarray(x, float) for repeat, x in values.items()}


@pytest.mark.parametrize(
    "case",
    [
        # Unbalanced: the balanced test decides, the test as provided is skipped.
        "balanced",
        # The balanced repeats hold one value, so the balanced test gives no p: the test as
        # provided, which keeps the sparse repeat r3, decides, from the generator as it was.
        "balanced without a p",
    ],
)
def test_the_trusted_test_is_the_test_it_reads(case):
    """``trusted_test`` skips the keys of the test as provided when the balanced test gives
    the p it reads; both p-values are those of the two tests drawn in turn."""
    rng = np.random.default_rng(4)
    if case == "balanced":
        by_t = _repeats({"r1": 10 + rng.normal(0.5, 1, 60), "r2": 10 + rng.normal(0.5, 1, 20)})
        by_c = _repeats({"r1": 10 + rng.normal(0, 1, 25), "r2": 10 + rng.normal(0, 1, 30)})
        kept = ["r1", "r2"]
    else:
        by_t = _repeats({"r1": [1.0] * 20, "r2": [1.0] * 12, "r3": [2.0, 3.0, 4.0]})
        by_c = _repeats({"r1": [1.0] * 20, "r2": [1.0] * 12, "r3": [1.0, 1.5, 2.0]})
        kept = ["r1", "r2"]
    test = contrast_cells.trusted_test(by_t, by_c, by_c, 499, np.random.default_rng(8))
    d, p, d_bal, p_bal, *_ = contrast_cells._tests(
        by_t, by_c, by_c, kept, True, 499, np.random.default_rng(8)
    )
    assert test["normalised"] and test["balanced"] == (case == "balanced")
    expected = (d_bal, p_bal) if case == "balanced" else (d, p)
    assert (test["d"], test["p"]) == expected and np.isfinite(test["p"])
