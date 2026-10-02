"""Populations are candidates on a UMAP, judged on all the features (owner, 2026-10-02, after
HDBSCAN on the principal components split a Gaussian and a straight line into populations
and missed a separate blob): a distinct population has a gap of low density, a group on the
map has its neighbours inside it and no gap, and a continuum gives neither."""

import numpy as np
import pandas as pd
import pytest

from morphocoverage.sampling_unit import find_populations
from morphocoverage.sampling_unit.populations import _purity, _valley


def _tables(values: np.ndarray, seed: int = 0):
    """``values`` as cells of 2 conditions x 3 repeats, 20 cells a field."""
    n = len(values)
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    frame = pd.DataFrame(values[order], columns=[f"f{i}" for i in range(values.shape[1])])
    frame["condition"] = np.where(np.arange(n) % 2, "a", "b")
    frame["biological_repeat"] = [f"r{i % 3}" for i in range(n)]
    frame["image_file"] = [
        f"{c}_{r}_{i // 120}.tif"
        for i, (c, r) in enumerate(zip(frame["condition"], frame["biological_repeat"], strict=True))
    ]
    fovs = frame[["condition", "biological_repeat", "image_file"]].drop_duplicates()
    return frame, fovs, [c for c in frame.columns if c.startswith("f")]


def _found(values: np.ndarray) -> pd.DataFrame:
    objects, fovs, features = _tables(values)
    return find_populations(objects, fovs, features=features, observation_unit="cell").table


def test_a_gaussian_has_no_population():
    rng = np.random.default_rng(1)
    assert _found(rng.normal(size=(2400, 4))).empty


def test_a_line_has_no_distinct_population():
    rng = np.random.default_rng(1)
    t = rng.normal(size=(2400, 1))
    table = _found(np.hstack([t, 2 * t, -t]) + 0.3 * rng.normal(size=(2400, 3)))
    # Its end may read as a group on the map (said as possibly the end of a continuum).
    assert "distinct" not in set(table["kind"])


def test_a_separate_blob_is_a_distinct_population():
    rng = np.random.default_rng(1)
    values = np.vstack([rng.normal(size=(2160, 4)), rng.normal(size=(240, 4)) + [5, 0, 0, 0]])
    table = _found(values).set_index("population")
    assert list(table["kind"]) == ["main", "distinct"]
    assert table.loc["P2", "observations"] == pytest.approx(240, abs=12)
    assert table.loc["P2", "valley"] <= 0.5 and table.loc["P2", "purity"] >= 0.94


def test_the_valley_and_the_purity_tell_a_gap_from_a_tail():
    rng = np.random.default_rng(2)
    blob = np.vstack([rng.normal(size=(900, 2)), rng.normal(size=(100, 2)) + [6, 0]])
    member = np.arange(1000) >= 900
    assert _valley(blob, member) <= 0.5 and _purity(blob, member) > 0.95
    tail = rng.exponential(size=(1000, 2))
    end = tail[:, 0] > np.quantile(tail[:, 0], 0.9)
    assert _valley(tail, end) > 0.5
