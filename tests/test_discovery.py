"""Which features separate the conditions: the cell-level gate, the paired test beside it
(sampling_unit.discovery), and the PCA and UMAP of all the features (embedding)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from morphocoverage.embedding import embed
from morphocoverage.plots_embedding import embedding_figures
from morphocoverage.sampling_unit import feature_discovery

LAB = Path(__file__).resolve().parents[1] / "data" / "lab"
sys.path.insert(0, str(LAB))
import catalog


def _tables(shift=1.0, n_repeats=3, cells=30, seed=0, unbalanced=False):
    """One well per condition and repeat: 'area' differs in every repeat, 'noise' does not,
    'flat' holds one value."""
    rng = np.random.default_rng(seed)
    rows = []
    for repeat in range(n_repeats):
        for condition, effect in (("ctrl", 0.0), ("drug", shift)):
            n = cells * (4 if unbalanced and repeat == 0 else 1)
            for field in range(n):
                rows.append(
                    {
                        "condition": condition,
                        "biological_repeat": f"r{repeat}",
                        "position_id": f"{condition}-{repeat}-{field}",
                        "area": rng.normal(10 + effect, 1),
                        "noise": rng.normal(5, 1),
                        "aspect": rng.normal(2, 0.3),
                        "flat": 1.0,
                    }
                )
    objects = pd.DataFrame(rows)
    fovs = objects[["condition", "biological_repeat", "position_id"]].drop_duplicates()
    return objects, fovs


def _discovery(objects, fovs, features=("area", "noise", "aspect", "flat")):
    return feature_discovery(
        objects,
        fovs,
        features=list(features),
        control={"condition": "ctrl"},
        n_permutations=999,
        observation_unit="cell",
    )


def test_a_consistent_difference_passes_both_gates_and_noise_neither():
    result = _discovery(*_tables())
    assert result.top() == ["area"]
    assert result.confirmed() == ["area"]
    best = result.ranking.iloc[0]
    assert best["direction"] == "3 of 3 higher" and best["d"] > 0
    row = result.tests.set_index("feature").loc["area"]
    assert row["normalised"] and not row["balanced"]
    assert row["paired_values"] == "normalised"
    text = " ".join(result.summary().split())
    assert "1 of 4 features is a candidate in the exploratory cell-level screen" in text
    assert (
        "Significant across repeats (paired t-test on the repeat means, q ≤ 0.05): 1 feature "
        "(area, for drug ×1.096)." in text
    )
    assert "Not tested (1 feature, one value everywhere): flat." in text
    assert (
        "only the paired t-test on the repeat means (3 or more repeats) decides significance"
        in text
    )


def test_unbalanced_repeats_use_the_balanced_test():
    result = _discovery(*_tables(unbalanced=True))
    row = result.tests.set_index("feature").loc["area"]
    assert row["balanced"]
    # Each draw takes the fewest cells any repeat holds, from every repeat.
    assert row["observations"] == 3 * 30


def test_with_two_repeats_there_is_no_paired_gate():
    result = _discovery(*_tables(n_repeats=2))
    assert result.top() == ["area"]
    assert result.confirmed() == []
    assert result.tests["p_paired"].isna().all()
    assert "No paired test: fewer than 3 biological repeats" in " ".join(result.summary().split())


def test_cdm_shape_ranks_features_with_one_well_per_condition():
    data = catalog.load("cct8_cdm_morphology")
    result = feature_discovery(
        *data.tables,
        features=data.feature_cols,
        control=data.control,
        n_permutations=999,
        observation_unit="cell",
    )
    # The ranking over whole technical repeats ranks none here (one well a side).
    assert len(result.top()) == 5
    assert set(result.confirmed()) >= {"Round", "AR"}


def test_embedding_places_every_complete_observation(tmp_path):
    objects, fovs = _tables()
    objects.loc[0, "noise"] = np.nan
    embedding = embed(
        objects,
        fovs,
        features=["area", "noise", "aspect", "flat"],
        top=["area"],
        max_points=100,
        random_state=1,
    )
    assert embedding.settings["constant_features"] == ["flat"]
    assert embedding.settings["incomplete_observations"] == 1
    assert len(embedding.points) == len(objects) - 1
    assert embedding.points["umap1"].notna().sum() == embedding.settings["umap_points"] == 100
    assert sorted(embedding.loadings.loc[embedding.loadings["top"], "feature"]) == ["area"]
    assert 0 < embedding.explained[1] <= embedding.explained[0] < 1
    figures = embedding_figures(embedding, control={"condition": "ctrl"})
    assert set(figures) == {"pca", "umap"}
    figures["pca"].savefig(tmp_path / "pca.png")
    again = embed(objects, fovs, features=["area", "noise", "aspect"], umap=False, random_state=1)
    assert again.points["umap1"].isna().all()
    assert list(embedding_figures(again)) == ["pca"]


def test_an_embedding_needs_three_features_that_vary():
    objects, fovs = _tables()
    with pytest.raises(ValueError, match="at least 3 features that vary"):
        embed(objects, fovs, features=["area", "noise", "flat"], umap=False)


def test_question_9_is_not_assessed_without_a_comparison_or_a_test_across_repeats():
    """#274: the cell-level screen's candidates alone (two repeats, no contrast) and a
    contrast with no condition to compare are no answer, never ✅."""
    from morphocoverage._style import Status
    from morphocoverage.retrospective.report_status import conditions_status
    from morphocoverage.sampling_unit.contrast import condition_contrast

    objects, fovs = _tables(n_repeats=2)
    status, headline = conditions_status(None, None, _discovery(objects, fovs))
    assert status is Status.NOT_ASSESSED and "no test across repeats: n = 2" in headline
    alone = objects["condition"] == "ctrl"
    contrast = condition_contrast(
        objects[alone],
        fovs[fovs["condition"] == "ctrl"],
        feature="area",
        control={"condition": "ctrl"},
    )
    assert conditions_status(contrast, None) == (
        Status.NOT_ASSESSED,
        "no condition to compare with the control",
    )
    # Three repeats: the test across repeats answers, as before.
    assert conditions_status(None, None, _discovery(*_tables()))[0] is Status.OK
