"""Regression tests of the owner's audit-8 decisions on the tests and their words (F9b,
2026-10-06)."""

from dataclasses import dataclass

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


def _fields_design(seed: int = 0):
    """3 repeats; the control one field of 60 cells, the condition three fields of 20 cells
    shifted -3, 0 and +3 cell SDs: no effect beyond the field design."""
    rng = np.random.default_rng(seed)
    rows = []
    for r in range(3):
        for condition, shifts, cells in (("ctrl", [0.0], 60), ("a", [-3.0, 0.0, 3.0], 20)):
            for f, shift in enumerate(shifts):
                rows += [
                    {
                        "condition": condition,
                        "biological_repeat": f"R{r}",
                        "image_file": f"{condition}{r}_{f}.tif",
                        "area": 10 + shift + value,
                    }
                    for value in rng.normal(size=cells)
                ]
    objects = pd.DataFrame(rows)
    fovs = objects[["condition", "biological_repeat", "image_file"]].drop_duplicates()
    return objects, fovs.reset_index(drop=True)


def test_a5_beyond_the_mean_reads_field_balanced_subsamples():
    from morphocoverage.sampling_unit.distribution import distribution_contrast

    objects, fovs = _fields_design()
    kwargs = {"features": ["area"], "control": {"condition": "ctrl"}, "fov_col": "image_file"}
    tests = distribution_contrast(objects, fovs, random_state=3, **kwargs).tests
    spread = tests.set_index("statistic").loc["spread"]
    # Pooled, the condition's SD would be sqrt(7) times the control's (log 0.97).
    assert abs(spread["mean"]) < 0.25 and spread["high"] < 0.4
    again = distribution_contrast(objects, fovs, random_state=3, **kwargs).tests
    np.testing.assert_allclose(again["mean"], tests["mean"])


def test_a5_seed10_beyond_the_mean_reads_the_features_the_fingerprint_reads():
    from morphocoverage.multivariate_outcomes import CovarianceError
    from morphocoverage.retrospective.criteria import OutcomeCriterion
    from morphocoverage.workflows import Setup
    from morphocoverage.workflows._distribution import (
        NOT_RUN,
        distribution_by_feature,
        with_distribution,
    )

    objects, fovs = _signed(offset_b=3.0)
    objects = objects.assign(d0=objects["area"] * 2, d1=objects["positive"] / 3)

    def setup(columns):
        criterion = OutcomeCriterion(CovarianceError(columns, True), 0.2, "lower_is_better")
        return Setup(
            readout="area",
            features=["area", "positive"],
            control={"condition": "ctrl"},
            feature_tables={"DINO": ("d0", "d1")},
            fingerprint=(criterion,),
            fov_col="image_file",
        )

    one = distribution_by_feature(objects, fovs, setup(("positive", "d0")))
    assert one.settings["features"] == ["positive"]
    dino = setup(("d0", "d1"))
    assert distribution_by_feature(objects, fovs, dino) is None

    @dataclass(frozen=True)
    class Report:  # the two texts with_distribution writes into
        short: dict
        sections: dict

    empty = Report({"conditions": "Do they differ?"}, {"conditions": "Do they differ?"})
    said = with_distribution(empty, None, "cells", dino)
    assert " ".join(NOT_RUN.split()) in " ".join(said.short["conditions"].split())


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


def _shared(control_fields: int, seed: int = 0):
    """ctrl in ``control_fields`` fields of 4 cells, a and b in 3 fields of 20, 4 repeats;
    a and b both 3 SDs higher on f0."""
    rng = np.random.default_rng(seed)
    rows = []
    for r in range(4):
        for condition, fields, cells in (("ctrl", control_fields, 4), ("a", 3, 20), ("b", 3, 20)):
            for f in range(fields):
                shift = 0.0 if condition == "ctrl" else 3.0
                for _ in range(cells):
                    rows.append(
                        {
                            "condition": condition,
                            "biological_repeat": f"R{r}",
                            "image_file": f"{condition}{r}_{f}.tif",
                            "f0": 10 + shift + rng.normal(),
                            "f1": 10 + rng.normal(),
                        }
                    )
    objects = pd.DataFrame(rows)
    fovs = objects[["condition", "biological_repeat", "image_file"]].drop_duplicates()
    return objects, fovs.reset_index(drop=True)


def test_a9_conditions_significant_together_against_a_small_control_are_named_once():
    from morphocoverage.retrospective.report_answers import conditions_sections
    from morphocoverage.sampling_unit.contrast import condition_contrast
    from morphocoverage.sampling_unit.discovery import feature_discovery
    from morphocoverage.sampling_unit.reading_text import shared_control_line

    def line(fields):
        objects, fovs = _shared(fields)
        kwargs = {"control": {"condition": "ctrl"}, "n_permutations": 99, "fov_col": "image_file"}
        contrast = condition_contrast(objects, fovs, feature="f0", n_draws=5, **kwargs)
        found = feature_discovery(objects, fovs, features=["f0", "f1"], **kwargs)
        full, short = conditions_sections(contrast, None, found)
        if fields == 1:  # said once in each form, as a caution
            assert "⚠️ Shared small control" in "\n".join(full) and short.count("Shared") == 1
        return shared_control_line(contrast, found), shared_control_line(contrast)

    small, alone = line(1)
    assert small.startswith("Shared small control: on f0, a and b against ctrl, 1 field in a")
    assert alone.startswith("Shared small control: a and b against ctrl, 1 field in a repeat.")
    assert line(3) == (None, None)


def test_item10_one_image_per_field_leaves_the_rare_clustering_out_of_question_3():
    import sys
    import warnings
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))
    from test_audit8_b import FEATURES, MITOTIC, _one_per_field

    from morphocoverage.workflows import Setup, report

    objects, fovs, table = _one_per_field()
    objects["mitotic"] = (np.random.default_rng(2).random(len(objects)) < 0.3).astype(int)
    common = {"control": {"condition": "ctrl"}, "observation_unit": "image"}
    rare = {"rare": MITOTIC, "rare_label": "mitotic"}
    explainable = Setup(readout="f1", features=FEATURES, **common, **rare)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for setup, tables in ((explainable, None), (Setup(**common, **rare), {"T": table})):
            short = " ".join(report(objects, fovs, setup, feature_tables=tables).summary().split())
            assert "3. What limits my sampling? — not applicable: with one image per field" in short
            assert "spread like random images" not in short


def test_beyond_the_mean_trend_only_from_six_repeats():
    # The trend line from TREND_REPEATS repeats, as per feature (owner, 2026-10-06, audit 8).
    from morphocoverage.workflows._distribution import _consistent

    rows = pd.DataFrame(
        {
            "feature": ["a", "b"],
            "repeats": [3, 6],
            "lower": [3, 6],
            "higher": [0, 0],
            "q": [0.4, 0.4],
        }
    )
    assert list(_consistent(rows)["feature"]) == ["b"]
