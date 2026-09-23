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

Relevant parameters include:

- minimum object-level prevalence of interest;
- target detection probability;
- minimum number of independent FOV occurrences required before a rare state is considered adequately encountered.

Cell prevalence and FOV incidence are distinct. A phenotype present in 1% of all cells but restricted to a few FOVs can require substantially deeper imaging than a 1% phenotype dispersed across nearly every FOV.

A rare phenotype is also not automatically equivalent to a single outlier. Independent recurrence, coherent feature-space structure, annotation, or other evidence may be required to distinguish biology from segmentation or imaging artifacts.

## Balanced sampling

Question:

> Is the common population adequately represented, and has the requested rare-state discovery target also been met?

The two components should remain visible separately. A future report may therefore state:

```text
Population representation: sufficient
Rare-discovery target: insufficient
Overall decision: continue
```

rather than collapsing both into one opaque coverage percentage.

## Current API

The objective is represented independently from the strategy:

```python
from morphocoverage import ObjectiveConfig, SamplingObjective

population = ObjectiveConfig(SamplingObjective.POPULATION)

rare = ObjectiveConfig(
    SamplingObjective.RARE_DISCOVERY,
    rare_prevalence=0.01,
    target_detection_probability=0.95,
    min_fov_occurrences=3,
)

balanced = ObjectiveConfig(
    SamplingObjective.BALANCED,
    rare_prevalence=0.01,
    target_detection_probability=0.95,
)
```

An objective describes what the experiment is trying to achieve. It does not imply a particular statistical estimator. Different strategies may be more or less suitable for different objectives, and that suitability must be established by benchmarking on real data.
