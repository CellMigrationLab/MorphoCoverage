# Proposal: a microscopist's report — adequacy, limitation, efficiency, action

**Status:** proposed (2026-09-24). Nothing below is implemented yet unless
stated. Items that change an input or output contract need the project
owner's approval first (AGENTS.md section 15).

## Why

MorphoCoverage already answers one question well: *have I imaged this sample
enough, and would another representative FOV likely add something new?* A
microscopist who has just finished an experiment usually wants more:

> What is limiting this experiment, what did the imaging effort buy me, and
> what should I do differently next time?

Today the answers come as estimator outputs (`fov_bootstrap_centroid_max_se`,
`lofo_novel_fov_upper`, ...). This proposal organizes the retrospective report
around the decisions a microscopist makes, and uses the existing estimators,
replay and benchmark machinery underneath. It does not add a default
estimator or a new mathematical claim.

## Core principle

**Retrospective representability is not the prospective sampling
requirement.** A completed dataset may be represented by 14 well-chosen FOVs,
but an experiment that samples representative FOVs at random may need about
23 to meet the same criterion reliably, because it cannot know which 14 to
take. Both numbers are useful: the first says how redundant this dataset was,
the second what to do next time. A "best subset" is never reported without
the prospective number beside it, and is never presented as a
recommendation. This principle should be added to AGENTS.md section 3 when the
first item below is implemented.

## The report: four layers

Every answer names the criterion, the feature representation and the
thresholds it depends on, and states whether the FOVs are treated as
exchangeable (random orders) or in their recorded order.

### 1. Adequacy — did I image enough?

Separately for each thing the experiment cares about: the common
population (mean, distribution), rare or minority states, and spatial
heterogeneity. For example:

> Common phenotype: probably yes; across random orders, 90% of acquisitions
> met the population criterion by 18–22 FOVs. Rare states: less clear; the
> chance of a new state was still falling at 30 FOVs, and a state in fewer
> than about 10% of FOVs cannot be ruled out.

**Exists:** `sufficiency_report` (criteria met, stop-FOV ranges, next-FOV
chance, rule-of-three statement, flagged cells). **Missing:** grouping of the
criteria into these headings, and a statement per heading.

### 2. Limitation — what is preventing adequate sampling?

A **sampling profile** built only from measured quantities, each shown with
its numbers, for example:

- *FOV-to-FOV heterogeneity dominates:* the error of a well mean from whole
  FOVs is 3× what random cells would give (a design effect of about 10;
  preliminary findings, entry 1);
- *spatially clustered:* the semivariogram rises up to about 300 µm;
- *rare states are clustered:* new states come from 3 of 30 FOVs;
- *sample-to-sample variation dominates:* between-sample variance of the
  readout is 4× the within-sample sampling variance;
- *representation-limited:* a positive control is found only 40% of the time
  with these features.

The profile is descriptive and conditional on the features. It must never be
worded as a biological conclusion.

### 3. Efficiency — what did the imaging effort buy?

> You acquired 48 FOVs. Common-phenotype information saturated around 18–22;
> rare-state information kept improving until about 31. The last ~15 FOVs
> added little under either criterion.

And, where stage coordinates exist, whether the redundancy was spatial:

> Adjacent FOVs within about 250 µm were strongly redundant; most of the
> acquisition consisted of contiguous neighbours.

### 4. Action — what should I change next time?

Conditional statements, each with the evidence that triggered it, never a
bare prescription:

> Within-sample uncertainty is already 4× smaller than between-sample
> variation, so additional biological samples would reduce the dominant
> uncertainty more than increasing from 20 to 40 FOVs per sample.

> Random FOVs met the criterion by about 25, the recorded raster by about
> 30, and spatially spread FOVs by about 19: spread the FOVs out.

## Questions and what each needs

| Question | Layer | Builds on | New work | Input needed |
|---|---|---|---|---|
| Did I image enough, for common and rare phenotypes separately? | adequacy | `sufficiency_report` | group criteria by component; one statement per heading | current tables |
| How many FOVs next time? | action | `run_benchmark` replay over random orders | **reliability curve**: share of random `k`-FOV acquisitions meeting each criterion, for `k = 2 .. N` | current tables |
| What did the extra FOVs buy? | efficiency | stop-FOV distributions | per-criterion saturation points and "FOVs beyond saturation" | current tables |
| Which FOVs should I inspect? | limitation | `new_observations` | **FOV influence**: leave-one-FOV-out change in the summaries and in novelty, with plain reasons (only FOV with a state, drives a tail quantile, unusual density) | current tables |
| More FOVs or more samples? | action | JCB planning, design effect (finding 1) | **variance decomposition** (cells within FOV, FOVs within sample, samples within repeat, repeats) and projected uncertainty for more FOVs versus more samples | sample and repeat identity |
| Am I imaging the same thing? | efficiency | semivariogram, blocks | redundancy distance in µm; diversity added along the recorded order | stage coordinates |
| Was my way of choosing FOVs sensible? | action | acquisition orders | add a spatially dispersed order (farthest-point on stage coordinates) and compare recorded, random and dispersed | stage coordinates |
| Did I preferentially image easy or dense regions? | limitation | none | compare imaged positions with all candidate positions | a table of candidate positions from an overview (**new input contract**) |

## Plain-language front door

The internal objectives (`population`, `rare_discovery`, `balanced`) stay.
Users choose what matters in words, and each choice maps to explicit
criteria:

| The user says | Criteria behind it |
|---|---|
| a reliable average phenotype | FOV-unit scalar precision, FOV-bootstrap centroid (largest component SE) |
| the full phenotype distribution | FOV-bootstrap quantiles or covariance; held-out distribution errors in the benchmark |
| minority or rare states | support novelty (clusters of similar cells), rule-of-three statement, rare-incidence planning when a label exists |
| spatial heterogeneity | semivariogram, block bootstrap, recorded versus random versus dispersed orders |

Presets print their criteria and thresholds, and are labelled provisional
until the benchmark supports them; they are not default estimators.

## Order of work

1. **Reliability curve and saturation points** (questions 2 and 3). Uses the
   existing replay; answers the two most common questions. Bounded by the
   number of FOVs acquired: it cannot show that 40 are needed from 28.
2. **Variance decomposition with projections** (question 5, and the
   limitation layer). Needs sample and repeat identity. With few repeats (3
   plates in idr0028) the between-repeat variance is very uncertain, and the
   report must show that uncertainty rather than a confident "add samples".
3. **FOV influence report** (question 4).
4. **Spatial redundancy in µm and the dispersed order** (questions 6 and 7),
   where stage coordinates exist.
5. **Sampling profile**: a synthesis of 1–4 into the limitation layer, with
   the numbers shown.
6. **Sampling-frame bias** (question 8), only after an input-contract
   decision and upstream support.

## Cautions

- Every number is conditional on the definition of "new" and on the
  features; the positive control (preliminary findings, entry 2) found siLATS1
  far less often than siYAP although it moves the YAP/TAZ ratio further.
- Curves from one dataset guide experiments like it: same sample type,
  staining and features.
- A small chance for the next FOV is not evidence that no rare state remains
  (preliminary findings, entry 3).
- "Optimal subset" results overfit the completed dataset; they describe its
  redundancy only.
- None of this performs the user's biological hypothesis test; condition-level
  analyses enter only as retrospective benchmark outcomes (AGENTS.md section 4).
