# Sampling objectives

MorphoCoverage does not define "enough imaging" without an acquisition objective.
The objective names the targets a decision must cover, and they are the
targets the microscopist declares (`docs/questions.md`, "Declare the
target"; `workflows.Setup.objective()`): the population summary, the
phenotypic fingerprint (the distribution component) and the rare phenotype.

| Objective | Components required |
|---|---|
| `population` | population summary |
| `distribution` | phenotypic fingerprint |
| `rare_discovery` | rare phenotype |
| `phenotype` | population summary and phenotypic fingerprint |
| `balanced` | population summary and rare phenotype |
| `all` | all three |

## Population representation

Question:

> Have enough representative FOVs been acquired to obtain a stable description of the dominant phenotypic population?

This objective prioritizes accurate representation of common population structure. Extremely rare observations should not force indefinite acquisition unless they materially affect the declared population-level target.

## Phenotypic fingerprint (distribution)

Question:

> Have enough representative FOVs been acquired to describe the distribution
> of phenotypes beyond its summary: its spread, its shape, the states present?

Evidence about it is tagged `distribution`: the FOV-bootstrap uncertainty of
the covariance, the quantiles and the PCA summaries (the centroid stays
population evidence), the support-novelty quantities (a field showing a state
no other field supports), the incidence coverage of labelled states, and the
JCS descriptor rules' range of the spread descriptors (SD, IQR:
`max_scale_range`, `max_scale_spread`). Their range of the location
descriptors (mean, median: `max_location_range`, `max_location_spread`) is
population evidence, so with the paper's four descriptors a JCS rule answers
both targets.

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

## Balanced sampling, and the objectives that combine targets

Question:

> Is the common population adequately represented, and has the requested rare-state discovery target also been met?

`balanced` asks this (population summary and rare phenotype); `phenotype`
asks for the summary and the fingerprint; `all` for all three. The combining
rules below are the same for every objective.

The two components remain visible separately. Every result has `component_decisions`, for example:

```text
component_decisions: {"population": "stop", "rare": "continue"}
decision:            "continue"
```

rather than collapsing both into one opaque coverage percentage.

## How objectives are answered

Estimators report quantities tagged with the component they are evidence about (`population`, `distribution` or `rare`); decision policies decide one component each. The framework combines them:

- each objective needs a decision for every component in the table above;
- an objective with several components stops only when every one stops;
- a required component with no evidence of that kind gives `unsupported`, never `stop`. Among the current-practice baselines, the mean-precision rule and the fixed counts are population rules, so under `distribution` they return `unsupported`; the JCS descriptor rules are population evidence from their location descriptors and distribution evidence from their spread descriptors, so with the default descriptors they decide both. Under `rare_discovery` every baseline returns `unsupported`, while still reporting its own verdict;
- a quantity that needs more data gives `continue`; one that is undefined for the data gives `None`.

Every result therefore carries two decisions. `decision` is the objective's,
as above. `rule_decision` is the rule's own verdict, combined over the
components its policies decide, whatever the objective; it is never
`unsupported`. The live decision, `run_benchmark` and the contract tests read
the first; a self-assessment ("did each criterion say enough, by its own
threshold?"), a replay, a calibration and the frontier read the second, since
they ask what the rule said, not whether the rule alone could settle the
objective. `explain()` prints both when they differ, and
`morphocoverage.results.decision_view` renders several rules on one
acquisition state as one line per target and the overall decision, for
example:

```text
Population summary: stop (2 rules); Phenotypic fingerprint: continue (1 rule);
Overall (phenotype objective): continue
```

A required target that no rule decides is printed as such, and the overall
decision stays `unsupported`: a target is never assumed adequate because no
rule looked at it.

## Current API

The objective is represented independently from the strategy:

```python
from morphocoverage import ObjectiveConfig, RareTarget, SamplingObjective

population = ObjectiveConfig(SamplingObjective.POPULATION)
phenotype = ObjectiveConfig(SamplingObjective.PHENOTYPE)  # summary and fingerprint

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
