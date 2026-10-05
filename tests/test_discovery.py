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


def test_a_condition_with_nothing_to_balance_against_a_pooled_control_is_not_tested():
    """'drug' holds 4 cells in one of the control's three repeats: as provided its control
    pools the other two, and its one repeat is too small to balance, so no gate (#292)."""
    objects, _ = _tables()
    later = (objects["condition"] == "drug") & (objects["biological_repeat"] != "r0")
    objects.loc[later, "condition"] = "other"
    objects = objects.drop(objects.index[objects["condition"] == "drug"][4:])
    fovs = objects[["condition", "biological_repeat", "position_id"]]
    tests = _discovery(objects, fovs, features=("area",)).tests.set_index("condition")
    assert np.isnan(tests.loc["drug", "p_cells"]) and np.isfinite(tests.loc["other", "p_cells"])
    assert tests.loc["drug", "status"].startswith("its control would include repeats")


def test_a_perfect_separation_is_not_called_one_value_everywhere():
    """Control 0 and drug 1 in every cell: no Cohen's d, so no test, and the feature was
    listed as holding one value everywhere (#297)."""
    objects, fovs = _tables()
    objects["flag"] = (objects["condition"] == "drug").astype(float)
    text = " ".join(_discovery(objects, fovs, features=("area", "flag", "flat")).summary().split())
    assert "Not tested (1 feature, no spread within the condition and the control): flag." in text
    assert "Not tested (1 feature, one value everywhere): flat." in text


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
