"""Tests of ``embedding``: the PCA and its loadings."""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pandas as pd

from morphocoverage.embedding import _standardised, embed
from morphocoverage.workflows import _html_maps


def test_a_mostly_zero_column_does_not_outweigh_the_others():
    # Audit B, finding 6: just under half the values at 0, a tiny MAD, a variance of 830,000.
    rng = np.random.default_rng(0)
    relu = np.abs(rng.normal(size=2000))
    relu[:980] = 0.0
    normal = rng.normal(size=2000)
    z, _ = _standardised(np.column_stack([relu, normal]))
    assert z[:, 0].var() <= 16
    # A column with an ordinary MAD is scaled by it as before.
    mad = 1.4826 * np.median(np.abs(normal - np.median(normal)))
    np.testing.assert_allclose(z[:, 1], (normal - np.median(normal)) / mad)


def _shared_axis_tables(conditions=("ctrl", "drug")):
    """``a`` and ``b`` share one strong axis, ``e`` and ``g`` are noise, ``flat`` is constant."""
    rng = np.random.default_rng(1)
    n = 240
    shared = rng.normal(0, 1, n)
    objects = pd.DataFrame(
        {
            "condition": np.repeat(conditions, n // 2),
            "biological_repeat": "r1",
            "image_file": [f"f{i // 10}.tif" for i in range(n)],
            "a": shared + rng.normal(0, 0.1, n),
            "b": -shared + rng.normal(0, 0.1, n),
            "e": rng.normal(0, 1, n),
            "g": rng.normal(0, 1, n),
            "flat": 1.0,
        }
    )
    return objects, objects[["condition", "biological_repeat", "image_file"]].drop_duplicates()


def _pca_caption(objects, fovs, top=(), held=""):
    embedding = embed(objects, fovs, features=["a", "b", "e", "g", "flat"], top=top, umap=False)
    return _html_maps._set(embedding, "pc", {"ctrl": 0, "drug": 1}, "raw", held=held)[
        "caption"
    ], embedding


def test_75_each_axis_names_its_strongest_loadings_and_the_arrows_their_kind():
    caption, embedding = _pca_caption(*_shared_axis_tables())
    head, pc1, pc2 = caption.split("\n")
    assert "arrows: the 4 largest loadings, as no feature ranks." in head
    assert pc1.startswith("PC1, strongest loadings: ") and pc2.startswith("PC2, strongest")
    assert pc1.endswith("a feature with + is higher to the right.")
    assert pc2.endswith("a feature with + is higher towards the top.")
    named = [part.split()[0] for part in pc1.split(": ")[1].split(";")[0].split(", ")]
    assert named[:2] in (["a", "b"], ["b", "a"]) and len(named) == 3
    assert "flat" not in caption
    # Signs as drawn: a and b on opposite sides of PC1, as in the loadings.
    loadings = embedding.loadings.set_index("feature")
    assert f"a {loadings.at['a', 'pc1']:+.2f}" in pc1 and f"b {loadings.at['b', 'pc1']:+.2f}" in pc1
    ranked, _ = _pca_caption(*_shared_axis_tables(), top=("e",), held=", 3 PCs hold 90% of it")
    assert "of the variance, 3 PCs hold 90% of it; arrows: the loadings of the best-ranked" in (
        ranked
    )
    assert ranked.split("\n")[1:] == [pc1, pc2]  # the ranking does not change the loadings


def test_75_the_condition_labels_do_not_change_the_loadings():
    caption, _ = _pca_caption(*_shared_axis_tables())
    relabelled, _ = _pca_caption(*_shared_axis_tables(("drug", "ctrl")))
    assert caption == relabelled


class _RecordingUMAP:
    """Stands in for ``umap.UMAP``: keeps the keyword arguments it was built with."""

    calls: ClassVar[list[dict]] = []

    def __init__(self, **kwargs):
        type(self).calls.append(kwargs)

    def fit_transform(self, z):
        return np.zeros((len(z), type(self).calls[-1]["n_components"]))


def test_umap_is_built_with_compatibility_layout_explicitly(monkeypatch):
    """umap-learn 0.6 plans to change the default of ``compatibility_layout``; the layout must
    not depend on the installed version, so the argument is always passed, as True (today's
    layout)."""
    import umap

    from morphocoverage.sampling_unit.populations import _embedding

    _RecordingUMAP.calls = []
    monkeypatch.setattr(umap, "UMAP", _RecordingUMAP)
    objects, fovs = _shared_axis_tables()
    embed(objects, fovs, features=["a", "b", "e", "g"], max_points=100)
    _embedding(np.random.default_rng(0).normal(size=(60, 4)), 0)
    assert len(_RecordingUMAP.calls) == 2  # the maps' UMAP and the populations' UMAP
    for kwargs in _RecordingUMAP.calls:
        assert "compatibility_layout" in kwargs
        assert kwargs["compatibility_layout"] is True


def test_the_settings_record_the_umap_layout():
    """The layout changes the coordinates, so the run's settings say which one made them."""
    objects, fovs = _shared_axis_tables()
    embedding = embed(objects, fovs, features=["a", "b", "e", "g"], max_points=100)
    assert embedding.settings["compatibility_layout"] is True
    without_umap = embed(objects, fovs, features=["a", "b", "e", "g"], umap=False)
    assert without_umap.settings["compatibility_layout"] is True
