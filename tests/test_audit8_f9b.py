"""Regression tests of the owner's audit-8 decisions on the tests and their words (F9b,
2026-10-06)."""

import numpy as np
import pandas as pd
from scipy import stats

from morphocoverage.sampling_unit.whole_table import ONE_EACH, whole_table_tests


def _table(shift_b: float = 0.0, batch: float = 0.0, images: int = 5, seed: int = 0):
    """One image per field, conditions ctrl, a (+0.5 on d0) and b (``shift_b`` on d0), 4
    repeats, repeat R0 shifted by ``batch`` on every column."""
    rng = np.random.default_rng(seed)
    fovs = pd.DataFrame(
        [
            {"condition": c, "biological_repeat": f"R{r}", "image_file": f"{c}{r}_{i}.tif"}
            for c in ("ctrl", "a", "b")
            for r in range(4)
            for i in range(images)
        ]
    )
    values = rng.normal(size=(len(fovs), 6))
    values[fovs["condition"].eq("a").to_numpy(), 0] += 0.5
    values[fovs["condition"].eq("b").to_numpy(), 0] += shift_b
    values[fovs["biological_repeat"].eq("R0").to_numpy()] += batch
    names = [f"d{j}" for j in range(6)]
    objects = pd.concat([fovs, pd.DataFrame(values, columns=names)], axis=1)
    tests = whole_table_tests(
        objects,
        fovs,
        tables={"DINO": names},
        pairs=[("a", "ctrl"), ("b", "ctrl")],
        fov_col="image_file",
        observation_unit="image",
        permutations=99,
    ).tests
    return tests.set_index("condition")


def test_a4_another_condition_or_a_batch_never_changes_a_comparison():
    # A4: each comparison is scaled on the within-group residuals of its own two sides.
    plain = _table()
    for other in (_table(shift_b=4.0), _table(batch=3.0)):
        for column in ("t", "p_repeats", "effect", "distance", "p_shuffle"):
            assert np.isclose(other.loc["a", column], plain.loc["a", column])


def test_a4_one_observation_a_side_and_repeat_is_not_tested():
    tests = _table(images=1)
    assert set(tests["status"]) == {ONE_EACH} and tests["p_repeats"].isna().all()


def _signed(offset_b: float, seed: int = 0):
    """ctrl, a and b, 4 repeats, 3 fields of 5 cells; ``area`` near 2 for ctrl and a, near
    ``offset_b`` for b; ``positive`` the same plus 10."""
    rng = np.random.default_rng(seed)
    rows = []
    for condition, centre in (("ctrl", 2.0), ("a", 2.5), ("b", offset_b)):
        for r in range(4):
            for f in range(3):
                for _ in range(5):
                    rows.append(
                        {
                            "condition": condition,
                            "biological_repeat": f"R{r}",
                            "image_file": f"{condition}{r}_{f}.tif",
                            "area": centre + rng.normal(scale=0.3),
                        }
                    )
    objects = pd.DataFrame(rows).assign(positive=lambda t: t["area"] + 10)
    fovs = objects[["condition", "biological_repeat", "image_file"]].drop_duplicates()
    return objects, fovs.reset_index(drop=True)


def test_a7_the_main_test_reads_one_scale_per_feature():
    from morphocoverage.sampling_unit.discovery import feature_discovery

    # A7: b's values below 0 put area on differences for every condition, a included.
    objects, fovs = _signed(offset_b=-1.0)
    tests = feature_discovery(
        objects,
        fovs,
        features=["area", "positive"],
        control={"condition": "ctrl"},
        n_permutations=9,
        fov_col="image_file",
    ).tests.set_index(["feature", "condition"])
    assert set(tests.loc["area", "paired_values"]) == {"raw"}
    assert set(tests.loc["positive", "paired_values"]) == {"normalised"}
    means = objects.groupby(["condition", "biological_repeat"])["area"].mean()
    controls = means["ctrl"].to_numpy()
    expected = stats.ttest_1samp(means["a"].to_numpy() - controls, 0.0).pvalue
    assert np.isclose(tests.loc[("area", "a"), "p_paired"], expected)
    # The contrast of the readout, its plan and its words read the same scale.
    from morphocoverage.sampling_unit.contrast import condition_contrast

    contrast = condition_contrast(
        objects,
        fovs,
        feature="area",
        control={"condition": "ctrl"},
        n_permutations=9,
        n_draws=5,
        fov_col="image_file",
    )
    assert contrast.settings["log_scale"] is False
    assert contrast.repeats["p_paired_t_normalised"].isna().all()
    assert contrast.repeats["fold_change"].isna().all()
    assert set(contrast.plan["planned_on"]) <= {"difference", "d"}


def test_seed11_the_trend_line_needs_six_repeats():
    from types import SimpleNamespace

    from morphocoverage.sampling_unit.reading_text import (
        CONSISTENT,
        NOT_SIGNIFICANT,
        condition_readings,
    )
    from morphocoverage.workflows._whole_table import whole_headline

    def reading(n):
        row = {
            "condition": "A",
            "control_condition": "CTRL",
            "repeats": n,
            "lower": 0,
            "higher": n,
            "d_mean": 1.0,
            "q_paired_t": 0.4,
            "q_paired_t_normalised": 0.3,
        }
        contrast = SimpleNamespace(
            table=pd.DataFrame(),
            settings={"repeat_col": "biological_repeat", "observation_unit": "cell"},
            repeats=pd.DataFrame([row]),
            cells=pd.DataFrame(),
            targets=pd.DataFrame(),
        )
        return condition_readings(contrast).iloc[0]

    assert reading(5)["verdict"] == NOT_SIGNIFICANT
    assert "higher in 5 of 5 biological repeats" in reading(5)["reading"]
    assert reading(6)["verdict"] == CONSISTENT

    def whole(n):
        tests = pd.DataFrame(
            [{"table": "DINO", "condition": "a", "status": "tested", "repeats": n}]
        ).assign(same_way=n, q_repeats=0.4)
        return whole_headline(SimpleNamespace(tests=tests))[1]

    assert whole(5) == "no whole-table shift significant across repeats"
    assert whole(6).endswith("a shifted the same way in every repeat")


def test_seed13_the_label_share_is_counted_apart_from_the_features():
    from morphocoverage._wording import features_counted
    from morphocoverage.sampling_unit._experiments import family_words
    from morphocoverage.sampling_unit.discovery import feature_discovery

    assert features_counted(5, "mitotic share") == "4 features and the mitotic share"
    assert features_counted(["area", "mitotic share"], "mitotic share", ", ") == (
        "1 feature, the mitotic share"
    )
    assert features_counted(["mitotic share"], "mitotic share") == "the mitotic share"
    assert features_counted(["area", "perimeter"], "mitotic share") == "2 features"
    settings = {"family_features": 5, "family_share": "mitotic share"}
    assert family_words(settings).endswith(
        "over its 4 features, the mitotic share and their conditions"
    )
    objects, fovs = _signed(offset_b=3.0)
    objects["mitotic share"] = (objects["area"] > 2.5).astype(float)
    found = feature_discovery(
        objects,
        fovs,
        features=["area", "positive", "mitotic share"],
        control={"condition": "ctrl"},
        n_permutations=9,
        fov_col="image_file",
    )
    found.settings["share"] = "mitotic share"
    text = " ".join(found.summary().split())
    assert "of 2 features and the mitotic share" in text


def test_seed17_an_embedding_only_fingerprint_and_label_need_no_population():
    from morphocoverage.multivariate_outcomes import CovarianceError
    from morphocoverage.objectives import RareTarget, SamplingObjective
    from morphocoverage.retrospective.criteria import OutcomeCriterion
    from morphocoverage.strategies import FOVBootstrapStrategy
    from morphocoverage.transforms import Standardize
    from morphocoverage.workflows import Setup, live_session

    fingerprint = OutcomeCriterion(CovarianceError(("d0", "d1"), True), 0.2, "lower_is_better")
    rare = RareTarget(
        detection_probability=0.8, min_fov_occurrences=2, phenotype="r", object_prevalence=0.1
    )
    setup = Setup(
        control={"condition": "ctrl"},
        feature_tables={"DINO": ("d0", "d1")},
        fingerprint=(fingerprint,),
        rare=rare,
        rare_label="r",
    )
    objective = setup.objective()
    assert objective.objective is SamplingObjective.DISTRIBUTION_RARE
    assert {c.value for c in objective.required_components} == {"distribution", "rare"}
    [line] = [line for line in setup.declaration() if line.startswith("Live decision")]
    assert "(phenotypic fingerprint and rare phenotype)" in line
    assert "population" not in line
    rule = FOVBootstrapStrategy("covariance", 0.5, Standardize(), 20, 0, 3)
    session = live_session(setup, rules=(rule,))
    assert session.objective.objective is SamplingObjective.DISTRIBUTION_RARE


def test_seed19_the_whole_distribution_declares_no_mean_target():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))
    from test_guided import _cells as guided_cells

    from morphocoverage.objectives import SamplingObjective
    from morphocoverage.workflows import Answers, scan_design
    from morphocoverage.workflows._declaration import NO_MEAN

    objects = guided_cells()
    fovs = objects[["condition", "biological_repeat", "image_file"]].drop_duplicates()
    scan = scan_design(objects, fovs)

    def setup(aim):
        answers = Answers(
            loaded_correctly=True,
            control={"condition": "siCTRL"},
            aim=aim,
            features=["area", "aspect"],
            readout="area",
            observation_unit="cell",
            fingerprint_tolerance=0.2,
        )
        return answers.setup(scan)

    whole, both = setup("fingerprint"), setup("both")
    assert whole.tolerances == () and whole.stopping_rules() == ()
    assert whole.objective().objective is SamplingObjective.DISTRIBUTION
    assert whole.novelty is not None and whole.fingerprint  # novelty as before
    said = NO_MEAN.format(readout="area")
    assert [line for line in whole.declaration() if line == said] == [said]
    assert both.tolerances == (0.05, 0.025) and both.stopping_rules()
    assert both.objective().objective is SamplingObjective.PHENOTYPE
    # Without a definition of new nothing judges sampling: the report still runs.
    from dataclasses import replace

    from morphocoverage.workflows import report

    text = report(objects, fovs, replace(whole, novelty=None)).summary()
    assert "Population summary: not declared (tolerances=())" in " ".join(text.split())
