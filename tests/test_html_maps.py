"""The maps of the HTML page (``workflows._html_maps`` and its parts)."""

from __future__ import annotations

import json
import re
import warnings

import numpy as np
import pandas as pd
import pytest
from _helpers_embedding import _cells
from _helpers_feature_tables import KEY

from morphocoverage.embedding import Embedding
from morphocoverage.workflows import Setup, _html_figures, _html_maps, report
from morphocoverage.workflows._feature_space import attached
from morphocoverage.workflows._html_figures import feature_figures
from morphocoverage.workflows._html_map_text import share_lines
from morphocoverage.workflows._html_maps import _maps, _set, _tables, embeddings


def test_an_embedding_only_combined_space_has_its_centred_map():
    objects, fovs, table = _cells()
    halves = {"D1": table[[*KEY, *(f"e{j}" for j in range(6))]]}
    halves["D2"] = table[[*KEY, *(f"e{j}" for j in range(6, 12))]]
    setup = Setup(control={"condition": "ctrl"}, normalize="control_mean", observation_unit="cell")
    merged, setup, _ = attached(objects, fovs, setup, halves)
    tables, _ = _tables(setup, merged, fovs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        maps = _maps(setup, tables, fovs, (), [])
    assert set(maps["centred"]) == {"Combined", "D1", "D2"}


def test_the_combined_map_says_each_tables_share_of_each_axis():
    shares = {"explainable": [0.42, 0.9], "DINOv2": [0.58, 0.1]}
    assert share_lines(shares, (0.3, 0.2)) == [
        "PC1: explainable 42%, DINOv2 58%.",
        "PC2: explainable 90%, DINOv2 10%.",
    ]


def _signed_tables():
    """Two conditions in three repeats; ``speed`` is negative, so it cannot be normalised."""
    rng = np.random.default_rng(0)
    rows = []
    for repeat in ("r1", "r2", "r3"):
        for condition, shift in (("ctrl", 0.0), ("drug", 2.0)):
            for field in range(4):
                for label in range(5):
                    rows.append(
                        {
                            "condition": condition,
                            "biological_repeat": repeat,
                            "repeat_group": repeat,
                            "image_file": f"{condition}_{repeat}_{field}.tif",
                            "label_id": label + 1,
                            "area": rng.normal(20 + shift, 2),
                            "speed": rng.normal(-5, 1),
                            "aspect": rng.normal(2, 0.2),
                        }
                    )
    objects = pd.DataFrame(rows)
    return objects, objects[["condition", "biological_repeat", "image_file"]].drop_duplicates()


@pytest.fixture(scope="module")
def signed_report():
    objects, fovs = _signed_tables()
    setup = Setup(
        readout="area",
        features=("area", "speed", "aspect"),
        control={"condition": "ctrl"},
        observation_unit="cell",
    )
    return report(objects, fovs, setup), objects, fovs


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_340_a_negative_control_mean_leaves_the_raw_views_and_says_why(signed_report, tmp_path):
    result, objects, fovs = signed_report
    markup, _ = embeddings(result, objects, fovs, tmp_path)
    data = json.loads(re.search(r'id="embedding-data">(.*?)</script>', markup).group(1))
    assert data["values"] == ["raw"]
    assert "not shown: in biological_repeat=r1, the control mean of &#x27;speed&#x27; is" in markup
    _, specs = feature_figures(result, objects, fovs, tmp_path)
    assert all(list(spec["views"]) == ["raw"] for spec in specs)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_340_an_unexpected_error_of_a_view_is_not_hidden(signed_report, tmp_path, monkeypatch):
    result, objects, fovs = signed_report
    monkeypatch.setattr(_html_maps, "control_mean_problem", lambda *a, **k: None)
    monkeypatch.setattr(_html_figures, "control_mean_problem", lambda *a, **k: None)

    def broken(*args, **kwargs):
        raise ValueError("an unrelated error")

    monkeypatch.setattr(_html_maps, "normalize_to_control", broken)
    with pytest.raises(ValueError, match="an unrelated error"):
        embeddings(result, objects, fovs, tmp_path)
    drawn = _html_figures.plots
    monkeypatch.setattr(
        _html_figures,
        "plots",
        lambda o, f, view, **k: broken() if view.normalize else drawn(o, f, view, **k),
    )
    with pytest.raises(ValueError, match="an unrelated error"):
        feature_figures(result, objects, fovs, tmp_path)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_75_the_page_counts_the_pcs_that_hold_90_percent(signed_report, tmp_path):
    result, objects, fovs = signed_report
    markup, _ = embeddings(result, objects, fovs, tmp_path)
    data = json.loads(re.search(r'id="embedding-data">(.*?)</script>', markup).group(1))
    caption = data["sets"]["raw"]["PCA"]["caption"]
    k = result.results["populations"].populations.settings["components"]
    assert f"of the variance, {k} PCs hold 90% of it; arrows: " in caption
    assert caption.count("strongest loadings: ") == 2
    assert 'class="caption note" style="white-space: pre-line"' in markup


def _embedding(points: pd.DataFrame) -> Embedding:
    settings = {
        "features": ["a", "b"],
        "umap_points": len(points),
        "n_neighbors": 5,
        "min_dist": 0.1,
    }
    loadings = pd.DataFrame({"feature": ["a", "b"], "pc1": [0.6, -0.8], "pc2": [0.8, 0.6]})
    return Embedding(points, loadings.assign(top=False), (0.7, 0.3), settings)


def test_a_map_without_points_is_none_and_one_without_conditions_draws_them_alike():
    empty = pd.DataFrame({"pc1": [np.nan], "pc2": [np.nan], "field": ["f1"]})
    assert _set(_embedding(empty), "pc", {}, "raw") is None
    points = pd.DataFrame(
        {
            "pc1": [1.0, -2.0],
            "pc2": [0.5, 1.5],
            "umap1": [0.1, 0.2],
            "umap2": [0.3, 0.4],
            "field": ["f1", "f2"],
            "observation": [0, 1],
        }
    )
    drawn = _set(_embedding(points), "umap", {}, "raw", line="Neighbours: none.")
    assert drawn == {
        "x": [0.1, 0.2],
        "y": [0.3, 0.4],
        "cond": [0, 0],
        "labels": ["f1", "f2"],
        "caption": (
            "UMAP of 2 standardised features (raw values), 2 observations (n_neighbors 5, "
            "min_dist 0.1); its axes have no units.\nNeighbours: none."
        ),
        "xlabel": "UMAP 1",
        "ylabel": "UMAP 2",
        "arrows": [],
        "obs": [0, 1],
        "z": {},
    }
    pca = _set(_embedding(points), "pc", {}, "raw")
    assert pca["cond"] == [0, 0]
    assert pca["arrows"] == [["a", 1.2, 1.6], ["b", -1.6, 1.2]]
    assert pca["caption"] == (
        "PCA of 2 standardised features (raw values): PC1 70% and PC2 30% of the variance; "
        "arrows: the 2 largest loadings, as no feature ranks.\n"
        "PC1, strongest loadings: b -0.80, a +0.60; a feature with + is higher to the right.\n"
        "PC2, strongest loadings: a +0.80, b +0.60; a feature with + is higher towards the top."
    )
