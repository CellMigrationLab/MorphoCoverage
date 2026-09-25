# Proposal: a microscopist's report — adequacy, limitation, efficiency, action

**Status:** approved and implemented (2026-09-24) in `morphocoverage.retrospective`,
except question 8b (sampling-frame bias, roadmap Step 9, deferred). The "Exists" and
"Missing" notes below record the state when the proposal was written; the roadmap
(`docs/proposals/microscopist_report_roadmap.md`) says what is still open. Items
that change an input or output contract need the project owner's approval first
(AGENTS.md section 15).

## Why

MorphoCoverage already answers one question well: *have I imaged this sample
enough, and would another representative FOV likely add something new?* A
microscopist who has just finished an experiment usually wants more:

> What is limiting this experiment, what did the imaging effort buy me, and
> what should I do differently next time?

Today the answers come as estimator outputs (`fov_bootstrap_centroid_max_se`,
`lofo_novel_fov_upper`, ...). This proposal organizes the retrospective report
around the decisions a microscopist makes, and uses the existing estimators,
replay, hidden-FOV reservoirs and outcome metrics underneath. It does not add
a default estimator or a new mathematical claim.

## Core principles

1. **Retrospective representability is not the prospective sampling
   requirement.** A completed dataset may be represented by 14 well-chosen
   FOVs, but an experiment that samples representative FOVs at random may
   need about 23 to meet the same criterion reliably, because it cannot know
   which 14 to take. The first number says how redundant this dataset was,
   the second what to do next time. A "best subset" is never reported without
   the prospective number beside it, and never as a recommendation. (Added to
   AGENTS.md section 3.)
2. **What an estimator says is not what the sample achieved.** "The rule said
   enough at 20 FOVs in 92% of subsets" and "20 FOVs actually preserved the
   information in the hidden FOVs in 86% of subsets" are different
   statements. Planning advice is based on the second, judged with
   strategy-independent outcome metrics on FOVs the estimator never saw, and
   reported beside the first.
3. **Every answer is conditional on the representation.** Sampling can be
   adequate for features that cannot see the biology of interest. Whether the
   representation detects that biology is asked first, not buried in a list,
   and only for declared controls: "this representation detected the declared
   positive-control phenotype in 94% of candidate FOVs at a 3% false-alarm
   rate", never "the representation is 94% sensitive". Without a control the
   answer is "not assessed: no positive-control phenotype was declared".
4. **Future protocol recommendations require validation across samples, not
   only across FOVs within a sample.** "20 FOVs represent the unseen FOVs of
   this sample" and "20 FOVs will work for the next sample" are different
   claims. The second is estimated on held-out samples or repeats (the
   benchmark's development / held-out roles, or leave-one-sample or
   leave-one-repeat out), never on the samples used to choose methods or
   thresholds.

## The report: four layers

Every answer names the criterion, the feature representation and the
thresholds it depends on, and states whether FOVs are treated as
exchangeable (random orders) or in their recorded order.

### 1. Adequacy — did I image enough?

Separately for the common population (mean, distribution) and for rare or
minority states:

> Common phenotype: probably yes. Rare states: less clear; the chance of a new
> state was still falling at 30 FOVs, and a state in fewer than about 10% of
> FOVs cannot be ruled out.

Three notions of adequacy are kept apart, because they answer different
questions:

- **First apparent adequacy:** the first evaluated prefix at which the
  criterion says stop, what a live acquisition would have done.
- **Sustained apparent adequacy:** the earliest prefix from which every
  subsequent evaluated decision is stop (an undefined decision breaks it, as a
  continue does). A criterion can say stop at FOV 16, stay satisfied to FOV 23,
  and continue again when a new phenotype arrives at FOV 24; its sustained
  adequacy is then at 25 or later, not 16.
- **First and sustained realized adequacy:** the first prefix whose outcomes
  on hidden FOVs are within the declared tolerances, and the earliest prefix
  from which every later prefix stays within them. Hidden-FOV outcomes can
  also cross and uncross, so the sustained version answers "how many FOVs did
  this experiment really require?".

The apparent points use only the estimator's decisions; the realized points
use only strategy-independent outcome metrics. They are never mixed.
"How many FOVs?" (the share of random `k`-FOV sets that preserve the hidden
FOVs, no stopping rule involved) and "does this stopping rule work?" (how often
its stops are premature along acquisition orders) are separate analyses; see
the roadmap, `docs/proposals/microscopist_report_roadmap.md`.

Spatial heterogeneity is **not** an adequacy heading yet. The semivariogram,
block bootstrap and order comparisons describe spatial correlation, not
whether every spatial domain was sampled; they belong to the limitation layer
until a sampling frame or an explicit spatial coverage model exists.

**Exists:** `sufficiency_report` (first stop across orders, next-FOV chance,
rule-of-three statement, flagged cells). **Missing:** sustained and realized
adequacy; grouping of criteria by component, with a statement per heading.

### 2. Limitation — what is preventing adequate sampling?

A **sampling profile** built only from measured quantities, each shown with
its numbers:

- *representation-limited:* a positive control is recognized in only 41% of
  fields with these features (see the first question below);
- *FOV-to-FOV heterogeneity dominates:* the error of a well mean from whole
  FOVs is 3× what random cells would give (a design effect of about 10;
  preliminary findings, entry 1);
- *spatially correlated:* the semivariogram rises up to about 300 µm;
- *rare states are clustered:* new states come from 3 of 30 FOVs;
- *sample-to-sample variation dominates:* see the effort projection below.

The profile is descriptive and conditional on the features. It is never
worded as a biological conclusion.

### 3. Efficiency — what did the imaging effort buy?

> You acquired 48 FOVs. Under the population outcome, subsets of 18–22 random
> FOVs already preserved the hidden FOVs' distribution in most draws;
> rare-state recovery kept improving until about 31. The last ~15 FOVs added
> little under either criterion.

Stated with sustained and realized adequacy, not first crossings.

### 4. Action — what should I change next time?

Conditional statements, each with the evidence behind it, never a bare
prescription. Two kinds are kept separate:

- **Protocol size,** from the protocol-level reliability curve.
- **Where to spend effort,** from projections of counterfactual designs (more
  FOVs per sample versus more samples), with their uncertainty.

## The questions, in order

**Superseded (2026-09-25):** the canonical list is now `docs/questions.md`, grouped by moment of use (REPORT, OPTIMISE, ONLINE). The table below is kept as the original design record.

| # | Question | Builds on | New work | Input needed |
|---|---|---|---|---|
| 1 | Can this representation detect what I care about? | positive and frequency controls (preliminary findings, entries 2–3) | a positive-control report for declared controls only: detection and false-alarm rates per setting, or "not assessed" | a declared positive control (known phenotype, condition or planted state) |
| 2 | Did I image enough, for common and rare phenotypes? | `sufficiency_report` | first, sustained and realized adequacy per component | current tables |
| 3 | What is limiting my sampling? | design effect, semivariogram, novelty, controls | the sampling profile | current tables (stage coordinates for spatial items) |
| 4 | How many FOVs should I image next time? | replay, reservoirs, outcome metrics | **realized reliability curves**, per sample and across samples (below) | current tables; several samples for the protocol curve |
| 5 | What did the additional imaging buy? | stop-FOV distributions | marginal information and sustained or realized saturation per criterion | current tables |
| 6 | Which FOVs should I inspect? | `new_observations` | a traceable **inspect-these-fields** report (below) | current tables; stage coordinates if available |
| 7 | Would my effort be better spent elsewhere? | JCB planning, design effect | **hierarchical variance decomposition with projected uncertainty under alternative allocations** at equal total FOVs (below) | a declared sampling hierarchy; later a cost model |
| 8a | Could the same FOVs have been acquired in a better order? | acquisition orders | recorded versus random versus spatially dispersed (farthest-point on stage coordinates) orders among the acquired positions, with stage travel | stage coordinates |
| 8b | Were the imaged locations representative of the specimen? | none | compare imaged positions with all candidate positions | a table of candidate positions from an overview (**new input contract**) |

Question 8a asks whether visiting the same acquired locations in another
order would have reached adequacy sooner. It cannot say whether they were
the right locations, because unacquired tissue is invisible; that is 8b, and
it needs the sampling-frame input.

FOV counts alone favour orders that jump across the specimen. Each order is
therefore reported with its physical acquisition cost, starting with stage
travel computed from the coordinates, and acquisition time when move and
exposure times are known:

```text
                  FOVs to adequacy   stage travel   acquisition time
Recorded raster          28              7 mm            11 min
Dispersed                19             38 mm            12 min
(illustrative)
```

## Designs of the main new analyses

### Realized reliability curves (question 4)

Scoring each `k`-FOV subset against the rest of the sample would change the
reference along the curve: at `k = 5` of 30 it has 25 FOVs, at `k = 25` only 5,
so differences along the curve would partly reflect reference quality. The
primary design therefore uses a fixed outer reservoir:

1. split the sample once into a **fixed evaluation reservoir** of whole FOVs
   (or spatial blocks; `ReservoirSplit`) and a candidate stream;
2. for each `k`, draw `k` whole FOVs from the stream at random (many draws);
3. let the prospective estimator see only those `k` and record its decision;
4. score what the `k` FOVs missed with strategy-independent outcome metrics
   against the same reservoir for every `k`, with an explicit tolerance per
   metric;
5. repeat over many outer reservoir splits and report the spread.

The cost is that `k` cannot reach the full acquisition. For small datasets the
changing-rest comparison stays available as a secondary cross-validation mode,
labelled as such.

Report both, per `k`:

```text
20 FOVs
  Rule said "enough":               92%
  Outcome met on hidden FOVs:       86%
  Premature stops (said enough, missed):  8%
```

The outcome tolerance is part of the question and stated with the result.
The curve is bounded by the stream: from 28 FOVs with a reservoir of 10 it
cannot say anything beyond 18.

**Protocol level.** A microscopist writes one FOV number into a protocol for
many samples. Above the per-sample curves, report the share of samples whose
outcome is met at each `k`:

```text
FOVs   samples adequately represented
15     25%
20     55%
25     80%
30     90%
```

The user chooses how conservative to be. Samples that need far more than
others are named, which is where smart microscopy would help.

Two rules keep this honest:

- **Denominators are explicit.** Samples acquired to fewer FOVs than `k` (or
  whose stream is too short after the reservoir) cannot be evaluated at `k`,
  and dropping them silently makes deep `k` look better:

  ```text
  30 FOVs:  11 of 12 eligible samples adequate
            18 samples could not be evaluated at this depth
  ```

- **Aggregate at the independent unit.** Twenty wells from one experiment are
  not twenty independent validations; the curve is also reported per
  biological repeat (or whichever unit the hierarchy declares independent),
  and protocol claims are validated on held-out samples or repeats
  (principle 4).

### Traceable inspection report (question 6)

Per flagged FOV, plain reasons with numbers and a way back to the image, not
a single importance score:

```text
FOV P017   image P017.tif   stage x=..., y=...
  - only FOV containing new state C (8 cells listed)
  - leaving it out lowers the q95 of feature X by 18%
  - differs strongly from its spatial neighbours
  - 247 cells (median 90)
```

The microscopist decides whether it is biology or artifact; the report never
does.

### Effort projection (question 7)

The hierarchy is declared, never inferred from column names: generic
grouping columns have no automatic biological meaning, and a `well` or
`plate` column means different things in different experiments. Conceptually:

```python
SamplingHierarchy(fov="position_id", sample="well", biological_repeat="repeat")
```

Estimate the variance components of a sample-level measurement (cells within
FOV, FOVs within sample, samples within repeat, repeats), then project the
uncertainty of that measurement under counterfactual allocations **at equal
total FOV acquisition**:

```text
Current: 6 samples x 30 FOVs
  6 x 40 FOVs      uncertainty -8%
  8 x 30 FOVs      uncertainty -24%
  10 x 20 FOVs     uncertainty -31%
(illustrative)
```

Each projection carries its own uncertainty. With few samples or repeats (3
plates in idr0028) the between-sample components are poorly estimated, and
the report must say so rather than recommend. This is measurement and
sampling uncertainty; it does not perform the biological hypothesis test.

Equal FOV counts are not equal cost: another biological sample carries
preparation, staining, loading and setup cost that another FOV does not. The
first version reports only "at equal total FOV acquisition". A later version
may take an explicit cost model, `C = n_samples * C_sample + n_FOV * C_FOV`,
supplied by the user, and only then speak of equal budget.

## Plain-language front door

The internal objectives (`population`, `rare_discovery`, `balanced`) stay.
Users choose what matters in words, and each choice maps to explicit criteria
and outcome metrics:

| The user says | Criteria and outcomes behind it |
|---|---|
| a reliable average phenotype | FOV-unit scalar precision, FOV-bootstrap centroid (largest component SE); held-out mean error |
| the full phenotype distribution | FOV-bootstrap quantiles or covariance; held-out distribution errors |
| minority or rare states | support novelty (clusters of similar cells), rule-of-three statement, rare-incidence planning when a label exists; held-out rare-state recovery |
| spatial structure (diagnosis, not adequacy) | semivariogram, block bootstrap, order comparisons |

Presets print their criteria, outcomes and tolerances, and are labelled
provisional until the benchmark supports them; they are not default
estimators.

## Order of work

1. **Realized reliability curves** per sample, against repeated fixed outer
   reservoirs, with the premature-stop rate.
2. **First, sustained and realized adequacy** with the definitions above.
3. **Protocol-level reliability** across held-out samples or repeats, with
   explicit denominators.
4. **Inspect-these-fields report** (FOV influence with traceable reasons).
5. **Positive-control report** for declared controls (question 1); the
   positive and frequency controls are a prototype.
6. **Hierarchical effort projection** with a declared hierarchy, at equal total
   FOV acquisition.
7. **Spatial-order efficiency** (question 8a) with stage travel, where stage
   coordinates exist.
8. **Sampling-frame bias** (question 8b), only after an input-contract
   decision and upstream support.

"Optimal subset" search is not on the list.

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
- Outcome tolerances define "preserved"; they are part of every answer.
- None of this performs the user's biological hypothesis test; condition-level
  analyses enter only as retrospective benchmark outcomes (AGENTS.md section 4).
