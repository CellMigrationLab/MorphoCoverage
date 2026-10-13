"""Tests of ``plots_embedding``: the PCA and UMAP figures."""

from __future__ import annotations

from itertools import pairwise

from morphocoverage.plots_embedding import _spread_labels


def test_labels_of_arrows_ending_together_are_moved_apart():
    """A7: arrows of duplicated features end at one point; their labels must not overprint."""
    spots = _spread_labels([(1.0, 1.0), (1.0, 1.0), (1.0, 1.01), (-5.0, 3.0)], 20.0, 10.0)
    near = [s for s in spots if s[0] == 1.0]
    ys = sorted(y for _, y in near)
    assert all(b - a >= 0.05 * 10.0 - 1e-9 for a, b in pairwise(ys))
    assert (-5.0, 3.0) in spots  # a label elsewhere stays where it was
