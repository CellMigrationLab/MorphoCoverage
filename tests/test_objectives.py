import pytest

from morphocoverage import ObjectiveConfig, SamplingObjective


def test_population_objective_needs_no_rare_target():
    config = ObjectiveConfig(objective=SamplingObjective.POPULATION)
    assert config.rare_prevalence is None


def test_rare_discovery_requires_prevalence():
    with pytest.raises(ValueError, match="rare_prevalence is required"):
        ObjectiveConfig(objective=SamplingObjective.RARE_DISCOVERY)


def test_balanced_accepts_explicit_rare_target():
    config = ObjectiveConfig(
        objective=SamplingObjective.BALANCED,
        rare_prevalence=0.01,
        target_detection_probability=0.95,
        min_fov_occurrences=3,
    )
    assert config.rare_prevalence == 0.01
    assert config.min_fov_occurrences == 3


@pytest.mark.parametrize("value", [0, 1, -0.1, 1.1])
def test_rare_prevalence_must_be_probability(value):
    with pytest.raises(ValueError, match="rare_prevalence"):
        ObjectiveConfig(
            objective=SamplingObjective.RARE_DISCOVERY,
            rare_prevalence=value,
        )
