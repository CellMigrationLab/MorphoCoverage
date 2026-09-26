import pytest

from morphocoverage import ObjectiveConfig, RareTarget, SamplingObjective


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
