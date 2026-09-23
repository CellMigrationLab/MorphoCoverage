# Sampling objectives

MorphoCoverage does not define "enough imaging" without an acquisition objective.

## Population representation

Question:

> Have enough representative FOVs been acquired to obtain a stable description of the dominant phenotypic population?

This objective prioritizes accurate representation of common population structure. Extremely rare observations should not force indefinite acquisition unless they materially affect the declared population-level target.

## Rare-phenotype discovery

Question:

> Has enough representative imaging been performed to provide the desired sensitivity to phenotypes at or above a user-defined prevalence?

The user should define the scientific target rather than relying on hidden presets.

The target is a `RareTarget`:

- `object_prevalence` and/or `fov_incidence`: at least one is required;
- `detection_probability`: required;
- `min_fov_occurrences`: required, the number of distinct FOVs a rare state must be seen in before it is considered adequately encountered;
- `phenotype`: required, how the state is defined, either an externally labelled column or the estimator's resolution in phenotype space. Without it, "a 1% state" is not a testable quantity.

Cell prevalence and FOV incidence are distinct. A phenotype present in 1% of all cells but restricted to a few FOVs can require substantially deeper imaging than a 1% phenotype dispersed across nearly every FOV.

A rare phenotype is also not automatically equivalent to a single outlier. Independent recurrence, coherent feature-space structure, annotation, or other evidence may be required to distinguish biology from segmentation or imaging artifacts.

## Balanced sampling

Question:

> Is the common population adequately represented, and has the requested rare-state discovery target also been met?

The two components remain visible separately. Every result has `component_decisions`, for example:

```text
component_decisions: {"population": "stop", "rare": "continue"}
decision:            "continue"
```

rather than collapsing both into one opaque coverage percentage.

## How objectives are answered

Estimators report quantities tagged with the component they are evidence about (`population` or `rare`); decision policies decide one component each. The framework combines them:

- `population` needs a population decision, `rare_discovery` a rare one, and `balanced` both;
- balanced stops only when both components stop;
- a required component with no evidence of that kind gives `unsupported`, never `stop`. All current baselines are population rules, so under `rare_discovery` they return `unsupported`, while still reporting their population decision;
- a quantity that needs more data gives `continue`; one that is undefined for the data gives `None`.

## Current API

The objective is represented independently from the strategy:

```python
from morphocoverage import ObjectiveConfig, RareTarget, SamplingObjective

population = ObjectiveConfig(SamplingObjective.POPULATION)

target = RareTarget(
    object_prevalence=0.01,
    fov_incidence=0.05,
    detection_probability=0.95,
    min_fov_occurrences=3,
    phenotype="label column 'is_mitotic'",
)
rare = ObjectiveConfig(SamplingObjective.RARE_DISCOVERY, rare=target)
balanced = ObjectiveConfig(SamplingObjective.BALANCED, rare=target)
```

An objective describes what the experiment is trying to achieve. It does not imply a particular statistical estimator. Different strategies may be more or less suitable for different objectives, and that suitability must be established by benchmarking on real data.
