"""Which features separate the conditions: the cell-level gate, the paired test beside it
(sampling_unit.discovery), and the PCA and UMAP of all the features (embedding)."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from _helpers_sampling_unit import CONTROL, _make, _tables
from _helpers_sampling_unit import _tables as _discovery_tables
from _helpers_workflows import SCREEN, _sets

from morphocoverage.embedding import embed
from morphocoverage.plots_embedding import embedding_figures
from morphocoverage.sampling_unit import feature_discovery
from morphocoverage.statistics import benjamini_hochberg
from morphocoverage.workflows import Setup, report

LAB = Path(__file__).resolve().parents[1] / "data" / "lab"
sys.path.insert(0, str(LAB))
import catalog


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


def test_each_point_carries_its_observation_and_label_id():
    """#74: a point is found again by its index in the observation table and its label."""
    objects, fovs = _tables()
    objects = objects.assign(label_id=np.arange(len(objects)) % 7 + 1)
    objects.index = [f"cell {i}" for i in range(len(objects))][::-1]
    objects = objects.sample(frac=1, random_state=0)
    objects.loc[objects.index[3], "noise"] = np.nan
    points = embed(objects, fovs, features=["area", "noise", "aspect"], umap=False).points
    kept = objects.drop(index=objects.index[3])
    assert points["observation"].tolist() == kept.index.tolist()
    assert points["label_id"].tolist() == kept["label_id"].tolist()
    # Without a label_id column, the index alone.
    plain = embed(objects.drop(columns="label_id"), fovs, features=["area", "noise", "aspect"])
    assert "label_id" not in plain.points and "observation" in plain.points


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


def _three_repeats() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Three repeats, ctrl and kd, 4 fields of 30 cells a side: ``steady`` changes by x0.83
    in every repeat almost exactly (each repeat's cells the control's own, scaled), the other
    features by a factor that varies between the repeats."""
    rng = np.random.default_rng(14)
    factors = {"steady": (0.83, 0.83, 0.83), "f1": (0.6, 0.9, 1.0), "f2": (0.7, 1.2, 0.95)}
    factors |= {"f3": (1.3, 0.8, 1.1), "f4": (0.75, 1.0, 0.7)}
    rows = []
    for index, repeat in enumerate(("r1", "r2", "r3")):
        cells = {f: np.exp(rng.normal(0, 0.3, (4, 30))) * 10 for f in factors}
        for condition in ("ctrl", "kd"):
            for field in range(4):
                for label in range(30):
                    row = {
                        "condition": condition,
                        "biological_repeat": repeat,
                        "image_file": f"{repeat}_{condition}_{field}.tif",
                        "label_id": label,
                    }
                    for feature, values in cells.items():
                        scale = factors[feature][index] if condition == "kd" else 1.0
                        row[feature] = values[field, label] * scale
                    rows.append(row)
    objects = pd.DataFrame(rows)
    fovs = objects[["condition", "biological_repeat", "image_file"]].drop_duplicates()
    return objects, fovs


def test_a_verdict_on_three_near_identical_repeats_is_said_and_cautioned() -> None:
    # A10: a glioma verdict rested on 3 repeats whose log ratios agreed to 0.0018 (median
    # 0.18 over the family): p 3e-5 from its own tiny SD.
    objects, fovs = _three_repeats()
    discovery = feature_discovery(
        objects,
        fovs,
        features=["steady", "f1", "f2", "f3", "f4"],
        control={"condition": "ctrl"},
        observation_unit="cell",
        n_permutations=199,
    )
    tests = discovery.tests.set_index("feature")
    assert tests.loc["steady", "near_identical"]
    assert not tests.drop(index="steady")["near_identical"].any()
    assert tests.loc["steady", "repeat_sd"] < 0.2 * tests.loc["steady", "family_sd"]
    resting = discovery.resting()
    assert list(resting["feature"]) == ["steady"]
    text = " ".join(discovery.summary().split())
    assert "Rests on 3 near-identical repeats: steady for kd" in text
    assert "under a fifth of its median" in text


def test_a_screen_against_one_control_gives_the_screen_wide_q_beside_each_verdict() -> None:
    # A11: the filopodia hits were significant within families of 2 tests each; over the
    # 12 tests against siCTRL, q 0.063 and 0.054.
    objects, fovs = _sets(SCREEN, {"siA": 0.4, "siB": 0.3})
    common = {"features": ["area", "length"], "observation_unit": "cell", "n_permutations": 199}
    apart = feature_discovery(objects, fovs, control={"condition": "ctrl"}, **common)
    tests = apart.tests
    assert tests["experiment"].nunique() == 2
    expected = benjamini_hochberg(tests["p_paired"].to_numpy(float))
    assert tests["q_screen"].to_numpy() == pytest.approx(expected)
    assert "(screen-wide q " in " ".join(apart.summary().split())
    # Declared one experiment (a screen): one family, its q the screen's, none beside it.
    one = feature_discovery(
        objects,
        fovs,
        control={"condition": "ctrl"},
        families={"siA": "screen", "siB": "screen"},
        **common,
    )
    assert one.tests["experiment"].nunique() == 1
    assert one.tests["q_paired"].to_numpy() == pytest.approx(expected)
    assert one.tests["q_screen"].isna().all()
    # The readout's reading carries it too.
    setup = Setup(readout="area", features=("area", "length"), control={"condition": "ctrl"})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        text = " ".join(report(objects, fovs, setup).summary("full").split())
    assert "(screen-wide q " in text


def test_declared_experiments_are_the_correction_families() -> None:
    # B11: THBS5 vs VEC (untransfected) was corrected with the siRNA set imaged in the same
    # repeats; declared apart, each experiment is its own family.
    layout = {
        "VEC": ["R1", "R2", "R3"],
        "THBS5": ["R1", "R2", "R3"],
        "VEC_SiNC": ["R1", "R2", "R3", "R4"],
        "VEC_SiA": ["R1", "R2", "R3", "R4"],
        "VEC_SiB": ["R1", "R2", "R3", "R4"],
    }
    objects, fovs = _sets(layout, {"THBS5": 0.3, "VEC_SiA": 0.5})
    fovs = fovs.assign(set=np.where(fovs["condition"].isin(["VEC", "THBS5"]), "untr", "siRNA"))
    control = {"condition": {"VEC": ["THBS5"], "VEC_SiNC": ["VEC_SiA", "VEC_SiB"]}}
    declared = Setup(readout="area", features=("area", "length"), control=control)
    assert declared.families(fovs) is None
    setup = Setup(readout="area", features=("area", "length"), control=control, experiment="set")
    families = setup.families(fovs)
    assert families["THBS5"] == "untr" and families["VEC_SiA"] == "siRNA"
    assert "VEC" not in families or families["VEC"] == "untr"
    common = {"features": ["area", "length"], "observation_unit": "cell", "n_permutations": 199}
    common |= setup.contrast_arguments
    together = feature_discovery(objects, fovs, **common).tests
    apart = feature_discovery(objects, fovs, families=families, **common).tests
    assert together["experiment"].nunique() == 1
    split = apart.groupby("experiment")["condition"].agg(lambda c: tuple(sorted(set(c))))
    assert sorted(split) == [("THBS5 vs VEC",), ("VEC_SiA vs VEC_SiNC", "VEC_SiB vs VEC_SiNC")]
    mine = apart[apart["condition"].astype(str).str.startswith("THBS5")]
    expected = benjamini_hochberg(mine["p_paired"].to_numpy(float))
    assert mine["q_paired"].to_numpy() == pytest.approx(expected)
    words = " ".join(setup.declaration())
    assert "Experiments: declared by the column 'set', each corrected on its own." in words
    with pytest.raises(TypeError, match="experiment must be"):
        Setup(readout="area", control={"condition": "VEC"}, experiment=3)
    with pytest.raises(ValueError, match="not a column of the FOV table"):
        Setup(readout="area", control=control, experiment="plate").families(fovs)


def test_the_discovery_and_its_csv_carry_each_repeats_change_and_the_interval():
    objects, fovs = _discovery_tables(n_repeats=3)
    result = feature_discovery(
        objects,
        fovs,
        features=["area", "noise"],
        control={"condition": "ctrl"},
        n_permutations=999,
        observation_unit="cell",
    )
    tests = result.tests.set_index("feature")
    assert {"repeat_contrasts", "ci_low", "ci_high"} <= set(result.tests.columns)
    area = tests.loc["area"]
    assert area["ci_low"] < area["change"] < area["ci_high"]
    assert area["repeat_contrasts"].startswith("r0: ")
    text = result.summary()
    assert "change in each repeat" in text and "95% t interval" in text


def test_a_large_family_reruns_the_screen(monkeypatch):
    """40 null features beside area: 84 tests in E5's one sub-experiment, so the p-values at
    or below 0.05 are rerun with 1680 shuffles and area's q reaches 0.05."""
    objects, fovs = _make("E5")
    rng = np.random.default_rng(3)
    extra = [f"null{i}" for i in range(40)]
    objects = objects.assign(**{name: rng.lognormal(3, 0.2, len(objects)) for name in extra})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        discovery = feature_discovery(
            objects,
            fovs,
            features=["area", "intensity", *extra],
            control=CONTROL,
            n_permutations=99,
        )
    tests = discovery.tests.set_index(["feature", "condition"])
    area = tests.loc[("area", "cond1")]
    assert area["permutations"] == 1680 and area["q_cells"] <= 0.05
    # Rerun or not: the shuffles are the first pass's or the family's, nothing between.
    assert set(tests["permutations"].dropna()) <= {99, 999, 1680}
    assert (tests.loc[tests["p_cells"] < 0.01, "permutations"] == 1680).all()
    assert "the p-values at or below 0.05 rerun with 1680 shuffles" in " ".join(
        discovery.summary().split()
    )


def _partly_shared(left_out):
    """The tables with one repeat of one side removed: 'ctrl' or 'drug' in repeat r2 (K2)."""
    objects, _ = _tables()
    objects = objects[
        ~((objects["condition"] == left_out) & (objects["biological_repeat"] == "r2"))
    ]
    fovs = objects[["condition", "biological_repeat", "position_id"]]
    return _discovery(objects, fovs, features=("area",))


def test_a_treated_only_repeat_is_named_as_not_compared():
    """'drug' holds r0, r1 and r2 but the control only r0 and r1: r2 was never tested, and the
    discovery says so, as condition_contrast does ('2 of its 3 biological repeats ...')."""
    result = _partly_shared("ctrl")
    assert result.settings["uncompared_repeats"] == [
        {"condition": "drug", "compared": ["r0", "r1"], "not_compared": ["r2"]}
    ]
    assert result.settings["unpaired"] == [] and result.settings["control_only_repeats"] == []
    text = " ".join(result.summary().split())
    assert "drug: 2 of its 3 biological repeats hold ctrl; r2 left out." in text


def test_a_control_only_repeat_is_named_as_in_no_comparison():
    result = _partly_shared("drug")
    assert result.settings["control_only_repeats"] == ["r2"]
    assert result.settings["uncompared_repeats"] == []
    text = " ".join(result.summary().split())
    assert (
        "The ctrl of biological repeat r2, which holds no other condition, is in no comparison."
        in text
    )


def test_a_fully_shared_design_names_no_repeat_as_left_out():
    result = _discovery(*_tables())
    assert result.settings["uncompared_repeats"] == []
    assert result.settings["control_only_repeats"] == []
    assert "left out" not in result.summary() and "no comparison" not in result.summary()
