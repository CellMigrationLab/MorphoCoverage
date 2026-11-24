"""Tests of ``plots_embedding``: the PCA and UMAP figures."""

from __future__ import annotations

import math
from itertools import pairwise

from morphocoverage.plots_embedding import _spread_labels


def test_labels_of_arrows_ending_together_are_moved_apart():
    """A7: arrows of duplicated features end at one point; their labels must not overprint."""
    spots = _spread_labels([(1.0, 1.0), (1.0, 1.0), (1.0, 1.01), (-5.0, 3.0)], 20.0, 10.0)
    near = [s for s in spots if s[0] == 1.0]
    ys = sorted(y for _, y in near)
    assert all(b - a >= 0.05 * 10.0 - 1e-9 for a, b in pairwise(ys))
    assert (-5.0, 3.0) in spots  # a label elsewhere stays where it was


def _embedding(loadings: dict[str, tuple[float, float]]):
    import numpy as np
    import pandas as pd

    from morphocoverage.embedding import Embedding

    rng = np.random.default_rng(0)
    points = pd.DataFrame(
        {
            "condition": ["a"] * 30 + ["b"] * 30,
            "pc1": rng.normal(0, 3, 60),
            "pc2": rng.normal(0, 3, 60),
            "umap1": np.nan,
            "umap2": np.nan,
        }
    )
    frame = pd.DataFrame(
        [{"feature": f, "pc1": x, "pc2": y, "top": True} for f, (x, y) in loadings.items()]
    )
    return Embedding(points, frame, (0.5, 0.3), {"features": list(loadings)})


def _labels_and_tips(loadings: dict[str, tuple[float, float]]):
    """Each drawn label's anchor and each arrow's tip, from the PCA figure itself."""
    from matplotlib.text import Annotation

    from morphocoverage.plots_embedding import embedding_figures

    [ax] = embedding_figures(_embedding(loadings))["pca"].axes
    tips = [t.xy for t in ax.texts if isinstance(t, Annotation)]
    labels = {
        t.get_text(): (t.get_position(), t.get_horizontalalignment())
        for t in ax.texts
        if not isinstance(t, Annotation) and t.get_text() in loadings
    }
    return labels, tips


def _along(tips: list, x: float, y: float) -> list:
    """The arrow tips pointing along the loading ``(x, y)``."""
    return [t for t in tips if math.isclose(math.atan2(t[1], t[0]), math.atan2(y, x))]


def test_each_loading_label_stays_beside_its_own_arrow():
    """#557: the labels were placed in the order of their height, the names in the loadings'
    order, so a name could sit at another feature's arrow."""
    loadings = {"geneA": (2.0, 3.0), "geneB": (-4.0, -2.0), "geneC": (1.0, 0.0)}
    labels, tips = _labels_and_tips(loadings)
    assert set(labels) == set(loadings) and len(tips) == 3
    for feature, (lx, ly) in loadings.items():
        [tip] = _along(tips, lx, ly)
        (x, y), ha = labels[feature]
        assert (x, y) == (tip[0] * 1.06, tip[1] * 1.06)  # nothing collides: at its own tip
        assert ha == ("left" if lx >= 0 else "right")
        # The nearest arrow tip to the label is its own.
        assert min(tips, key=lambda t: math.dist(t, (x, y))) == tip


def test_labels_of_arrows_ending_together_keep_their_own_arrow():
    # Coincident loadings in shuffled order: the labels move apart vertically, never across.
    loadings = {"z": (-1.0, 0.5), "dup1": (2.0, 1.0), "dup2": (2.0, 1.0), "w": (0.5, -3.0)}
    labels, tips = _labels_and_tips(loadings)
    for feature in ("z", "w"):
        [tip] = _along(tips, *loadings[feature])
        assert labels[feature][0] == (tip[0] * 1.06, tip[1] * 1.06)
    tip = _along(tips, *loadings["dup1"])[0]
    (x1, y1), _ = labels["dup1"]
    (x2, y2), _ = labels["dup2"]
    assert x1 == x2 == tip[0] * 1.06 and y1 != y2
    assert tip[1] * 1.06 in (y1, y2)  # one stays at the tip, the other moves above it
