from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from _helpers_sufficiency import PERMUTED, POPULATION

from morphocoverage import ObjectiveConfig, RareTarget, SamplingObjective, validate_input_tables
from morphocoverage.strategies import SupportNoveltyEstimator
from morphocoverage.sufficiency import sufficiency_report


def _rare(**overrides):
    config = {
        "object_prevalence": 0.01,
        "detection_probability": 0.95,
        "min_fov_occurrences": 3,
        "phenotype": "label column 'is_mitotic'",
        **overrides,
    }
    return RareTarget(**config)


def test_population_objective_needs_no_rare_target():
    config = ObjectiveConfig(objective=SamplingObjective.POPULATION)
    assert config.rare is None
    assert {c.value for c in config.required_components} == {"population"}


def test_rare_discovery_requires_a_rare_target():
    with pytest.raises(ValueError, match="RareTarget is required"):
        ObjectiveConfig(objective=SamplingObjective.RARE_DISCOVERY)


def test_balanced_accepts_explicit_rare_target_and_requires_both_components():
    config = ObjectiveConfig(SamplingObjective.BALANCED, rare=_rare())
    assert config.rare.object_prevalence == 0.01
    assert config.rare.min_fov_occurrences == 3
    assert {c.value for c in config.required_components} == {"population", "rare"}


def test_population_objective_rejects_rare_targets():
    with pytest.raises(ValueError, match="population objective"):
        ObjectiveConfig(SamplingObjective.POPULATION, rare=_rare())
    with pytest.raises(ValueError, match="phenotype objective"):
        ObjectiveConfig(SamplingObjective.PHENOTYPE, rare=_rare())


def test_distribution_phenotype_and_all_name_their_components():
    components = lambda config: {c.value for c in config.required_components}
    assert components(ObjectiveConfig(SamplingObjective.DISTRIBUTION)) == {"distribution"}
    assert components(ObjectiveConfig(SamplingObjective.PHENOTYPE)) == {
        "population",
        "distribution",
    }
    assert components(ObjectiveConfig(SamplingObjective.ALL, rare=_rare())) == {
        "population",
        "distribution",
        "rare",
    }
    with pytest.raises(ValueError, match="RareTarget is required"):
        ObjectiveConfig(SamplingObjective.ALL)
    # Balanced keeps its meaning: the summary and the rare state, not the fingerprint.
    assert components(ObjectiveConfig(SamplingObjective.BALANCED, rare=_rare())) == {
        "population",
        "rare",
    }


def test_rare_target_must_be_a_rare_target():
    with pytest.raises(TypeError, match="RareTarget"):
        ObjectiveConfig(SamplingObjective.RARE_DISCOVERY, rare={"object_prevalence": 0.01})


def test_fov_incidence_alone_is_a_valid_target():
    # A state confined to few FOVs is specified by its FOV incidence.
    target = _rare(object_prevalence=None, fov_incidence=0.05)
    assert (target.object_prevalence, target.fov_incidence) == (None, 0.05)


def test_rare_target_needs_prevalence_or_incidence():
    with pytest.raises(ValueError, match="object_prevalence, fov_incidence"):
        _rare(object_prevalence=None)


@pytest.mark.parametrize("field", ["object_prevalence", "fov_incidence", "detection_probability"])
@pytest.mark.parametrize("value", [0, 1, -0.1, 1.1])
def test_rare_target_probabilities_must_be_between_0_and_1(field, value):
    with pytest.raises(ValueError, match=field):
        _rare(**{field: value})


@pytest.mark.parametrize("value", [2.5, True])
def test_min_fov_occurrences_must_be_integer(value):
    with pytest.raises(TypeError, match="integer"):
        _rare(min_fov_occurrences=value)


def test_min_fov_occurrences_must_be_positive():
    with pytest.raises(ValueError, match="at least 1"):
        _rare(min_fov_occurrences=0)


@pytest.mark.parametrize("phenotype", ["", "   ", None])
def test_rare_target_must_declare_its_phenotype_definition(phenotype):
    with pytest.raises(ValueError, match="phenotype"):
        _rare(phenotype=phenotype)


@pytest.mark.parametrize("missing", ["detection_probability", "min_fov_occurrences", "phenotype"])
def test_rare_target_fields_have_no_defaults(missing):
    config = {
        "object_prevalence": 0.01,
        "detection_probability": 0.95,
        "min_fov_occurrences": 3,
        "phenotype": "x",
    }
    del config[missing]
    with pytest.raises(TypeError):
        RareTarget(**config)


def test_a_rare_target_needs_its_detection_probability():
    from morphocoverage import RareTarget

    with pytest.raises(TypeError, match="detection_probability is required"):
        RareTarget(
            detection_probability=None, min_fov_occurrences=1, phenotype="x", fov_incidence=0.1
        )


def test_the_missing_value_message_names_the_repeat_when_field_names_restart():
    from morphocoverage.objectives import ObjectiveConfig, SamplingObjective
    from morphocoverage.strategies import ScalarPrecisionEstimator

    rows, fovs = [], []
    for repeat in ("R1", "R2"):
        for f in range(4):
            fovs.append({"condition": "c", "biological_repeat": repeat, "position_id": f"f{f}"})
            for k in range(3):
                value = np.nan if (repeat == "R2" and f == 1 and k == 0) else 10.0 + f + k
                rows.append({**fovs[-1], "x": value})
    data = validate_input_tables(
        objects=pd.DataFrame(rows), fovs=pd.DataFrame(fovs), feature_cols=["x"]
    )
    estimator = ScalarPrecisionEstimator("x", "fov", "upper_95")
    quantity = estimator.estimate(
        data, objective=ObjectiveConfig(SamplingObjective.POPULATION)
    ).quantities["averaging_cv"]
    assert "condition=c, biological_repeat=R2, position_id=f1" in quantity.reason


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
    from _helpers_guided import _cells as guided_cells

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


def _homogeneous(n_fovs=40):
    rng = np.random.default_rng(0)
    rows = [
        {"image_file": f"f{j}", "a": a, "b": b}
        for j in range(n_fovs)
        for a, b in rng.normal(10, 0.1, size=(20, 2))
    ]
    return pd.DataFrame(rows), pd.DataFrame({"image_file": [f"f{j}" for j in range(n_fovs)]})


def test_e1_a_rule_that_says_yes_has_stop_points_under_every_objective():
    objects, fovs = _homogeneous()
    for objective in (POPULATION, ObjectiveConfig(SamplingObjective.PHENOTYPE)):
        report = sufficiency_report(
            objects,
            fovs,
            feature_cols=["a", "b"],
            strategies=[],
            objective=objective,
            orders=PERMUTED,
            novelty=SupportNoveltyEstimator(0.5, 1, None, 2, 3),
            novelty_target=0.25,
        )
        row = report.enough.iloc[0]
        assert row["sampled_enough"] == "yes"
        assert row["enough_after_fovs"] is not None and not pd.isna(row["enough_after_fovs"])
        assert "not met at any point" not in row["answer"]
