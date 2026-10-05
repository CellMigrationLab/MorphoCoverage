"""Owner decisions of 2026-10-05 on the workflows: live stops, HTML figures.

- #85: ``LiveFolder.watch(until_stop=...)`` follows the objective's overall decision;
  ``stop_on="rule"`` keeps the stop on every rule's own verdict.
- #340: a normalised view, or a rule's label, applies or says why not; any other error
  raises.
- #75: under the PCA, each axis's strongest loadings, and the PCs that hold 90% of the
  variance.
"""

from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd
import pytest

from morphocoverage import FixedFOVStrategy, ObjectiveConfig
from morphocoverage.embedding import embed
from morphocoverage.evidence import Component
from morphocoverage.policies import EffortPolicy
from morphocoverage.strategies.base import SamplingStrategy
from morphocoverage.workflows import (
    LiveFolder,
    Setup,
    _html_figures,
    _html_sections,
    live_session,
    report,
    write_fov,
)
from morphocoverage.workflows._html_figures import embeddings, feature_figures
from morphocoverage.workflows._html_sections import what_checked

SETUP = Setup(readout="x", observation_unit="cell")


def _fields(folder, n: int = 6, condition: str = "MOCK") -> None:
    """``n`` fields of one well, deterministic."""
    for index in range(n):
        values = pd.DataFrame({"x": 100.0 + np.arange(5, dtype=float) + index})
        metadata = {
            "condition": condition,
            "biological_repeat": "R1",
            "technical_repeat": "A01",
            "image_file": f"A01_f{index:03d}.tif",
        }
        write_fov(folder, values, metadata)


def _fingerprint_rule(fovs: int) -> SamplingStrategy:
    """A rule that decides the fingerprint only, after ``fovs`` fields."""
    policy = EffortPolicy("n_fovs", fovs, Component.DISTRIBUTION)
    return SamplingStrategy([], [policy], name=f"fingerprint_{fovs}")


def _watch(folder, rules, objective: str = "population", **kwargs) -> int:
    session = live_session(SETUP, rules=rules, objective=ObjectiveConfig(objective))
    seen = LiveFolder(folder, session).watch(
        interval=0.01, timeout=2, until_stop=True, on_update=None, **kwargs
    )
    return len(seen)


def test_85_one_target_one_rule_stops_as_before(tmp_path):
    _fields(tmp_path)
    assert _watch(tmp_path, (FixedFOVStrategy(2),)) == 2
    assert _watch(tmp_path, (FixedFOVStrategy(2),), stop_on="rule") == 2


def test_85_two_targets_stop_when_both_stop(tmp_path):
    # Each rule alone is unsupported for the phenotype objective; together they decide it.
    _fields(tmp_path)
    rules = (FixedFOVStrategy(2), _fingerprint_rule(3))
    assert _watch(tmp_path, rules, "phenotype") == 3


def test_85_a_target_no_rule_decides_never_stops_the_watch(tmp_path):
    _fields(tmp_path)
    rules = (FixedFOVStrategy(2),)
    assert _watch(tmp_path, rules, "phenotype") == 6  # every field, no stop
    assert _watch(tmp_path, rules, "phenotype", stop_on="rule") == 2


def test_85_one_target_still_continuing_keeps_the_watch_going(tmp_path):
    _fields(tmp_path)
    rules = (FixedFOVStrategy(2), _fingerprint_rule(5))
    assert _watch(tmp_path, rules, "phenotype") == 5
    three = (FixedFOVStrategy(2), FixedFOVStrategy(3), FixedFOVStrategy(4))
    assert _watch(tmp_path, three) == 4


def test_85_a_rule_on_a_target_not_required_does_not_hold_the_stop(tmp_path):
    # The live view's overall ignores a fingerprint rule under the population objective.
    _fields(tmp_path)
    rules = (FixedFOVStrategy(2), _fingerprint_rule(5))
    assert _watch(tmp_path, rules) == 2
    assert _watch(tmp_path, rules, stop_on="rule") == 5


def test_85_the_condition_label_does_not_change_the_stop(tmp_path):
    for condition in ("MOCK", "drug"):
        folder = tmp_path / condition
        _fields(folder, condition=condition)
        assert _watch(folder, (FixedFOVStrategy(2), _fingerprint_rule(3)), "phenotype") == 3


def test_85_stop_on_is_checked(tmp_path):
    _fields(tmp_path, 1)
    with pytest.raises(ValueError, match="stop_on"):
        _watch(tmp_path, (FixedFOVStrategy(2),), stop_on="every_rule")


# --- #340: the HTML views decide up front whether they apply ------------------------------


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
    monkeypatch.setattr(_html_figures, "control_mean_problem", lambda *a, **k: None)

    def broken(*args, **kwargs):
        raise ValueError("an unrelated error")

    monkeypatch.setattr(_html_figures, "normalize_to_control", broken)
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


def test_340_the_rules_line_names_an_other_rule_and_raises_on_a_broken_label(monkeypatch):
    class Counted:
        name = "my own rule"

    setup = Setup(readout="x", observation_unit="cell", rules=(FixedFOVStrategy(4), Counted()))
    assert "Stopping rules checked: fixed 4 fields and my own rule." in what_checked(setup)

    def broken(*args, **kwargs):
        raise ValueError("a broken label")

    monkeypatch.setattr(_html_sections, "rule_label", broken)
    with pytest.raises(ValueError, match="a broken label"):
        what_checked(setup)


# --- #75: the PCA axes in features ------------------------------------------------------


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
    return _html_figures._set(embedding, "pc", {"ctrl": 0, "drug": 1}, "raw", held=held)[
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
