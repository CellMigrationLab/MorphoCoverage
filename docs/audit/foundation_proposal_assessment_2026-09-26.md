# Assessment: making "why FOVs matter" the foundation instead of question R0

> **Status (2026-09-26, later the same day).** This is a dated record of the
> assessment as written; the "what exists" statements describe the code
> before the work it recommended. Since then, on the same branch: the
> Foundation section and the softened wording (step 1); the front door in
> `Setup` with the declaration printed first (step 2); the distribution
> component, the objectives `distribution`, `phenotype` and `all`, the rule's
> own verdict (`rule_decision`) and the per-target live view (step 3,
> decisions 1 and 3 taken as recommended); the condition contrast as report
> section 9 (decision 2, taken by the owner); the dataset overview and the
> closing "what this dataset cannot tell me" section (R6); normalisation to
> the control per repeat; PlotsOfData-style plots with field-bootstrap
> intervals; and the shared `statistics` module. Part 2's items 1, 3 (in
> part), 5 and 9 are therefore done; items 2, 4, 6, 7 and 10 remain as
> described.


**Date:** 2026-09-26. **Scope:** the proposal to move "Can I count objects, or
must I count fields?" (R0 of `docs/questions.md`) out of the REPORT question
list and into a FOUNDATION section that precedes REPORT, OPTIMISE and ONLINE;
to open REPORT with "what aspect of the phenotype do I want to preserve?"
(population summary, phenotypic fingerprint, rare phenotype, condition
differences); to make OPTIMISE and ONLINE answer per objective; and to soften
the README's "a microscope samples fields, not random objects".

**Method.** Every claim in the proposal was checked against the code that
would carry it (`sampling_unit`, `retrospective.report`, `workflows`,
`objectives`, `evidence`, `outcomes`, `multivariate_outcomes`), the
documents that state the current framing (`README.md`, `docs/questions.md`,
`docs/objects_or_fields.md`, `docs/project_aim.md`, `AGENTS.md`), the
committed real-data results (`docs/preliminary_findings.md` entries 1, 4, 9,
10; `data/crick_idr/results/`) and the tests that pin the current wording.
The full test suite was run on this branch before writing (see "Validation").

## Verdict in one paragraph

The reframing is right and cheap where it is a change of order and wording,
and it should be adopted: the sampling-unit checks are the premise of every
other answer, the repository already says so in three places, and the current
placement as one REPORT question among seven under-states that. The
"what do I want to preserve?" front door is also right, and it is already the
design record's intent (`docs/proposals/microscopist_report.md`, "Plain-language
front door"), but it is **not** implemented: the workflow `Setup` declares one
scalar readout and its tolerances, the objective model has two components
(population, rare) with no distribution component, and the OPTIMISE workflow
judges the mean only. The proposal's illustrative OPTIMISE table (mean,
fingerprint, 1% state at 5 to 30 FOVs) therefore cannot be produced today:
the fingerprint column needs a distribution criterion wired into the
workflow and deeper data than the committed 28-field wells (the variance
criterion hits a reservoir ceiling on idr0028), and the rare column needs
labels and a reservoir design that does not remove the positive fields (audit
finding F11). Three parts of the proposal need the owner's decision before
they are built, because they change public contracts or the project boundary:
a third objective component, "condition differences" as a user-facing REPORT
objective, and a one-line "fields matter: yes/no" verdict.

## 1. What exists, against what the proposal asks for

### 1.1 The Foundation section is a regrouping of existing analyses

Every quantity the proposal lists for the Foundation is computed today:

| Proposed Foundation item | Exists as | Today's home |
|---|---|---|
| objects per FOV | `fov_variability` (`observations_per_fov`) | REPORT R0, report section 1 |
| between-FOV variation | `fov_variability` (between-field share, design effect, error ratio, fields differing more than random objects) | REPORT R0, section 1 |
| random-object versus whole-FOV sampling error | design effect and error ratio of `fov_variability`; `data/crick_idr/fov_vs_objects.py` (findings entry 1) | R0; findings |
| JCB rule, random objects versus whole FOVs | `object_rule_check` | R0 and P5, section 1 |
| JCS stabilisation, random objects versus whole FOVs | `descriptor_spread_check` | R0 and P5, section 1 |
| rare-state object prevalence versus FOV incidence | `fov_clustering` on labels (from `unusual_objects` or any source) | REPORT R4, section 4 |
| spatial autocorrelation where coordinates exist | `redundancy_distance`, `spatial` semivariogram | OPTIMISE P7 only (`optimise` workflow); the `report` workflow does not run it |

So the Foundation costs no new estimator. It is a new `QUESTIONS` entry in
`retrospective/report.py` placed before the current "1.", a move of the
clustering summary (or a copy of it) from section 4, and running
`redundancy_distance` in the `report` workflow when stage coordinates are
declared. `redundancy_distance` takes the two tables and stage columns; it
needs no replay, so the move is straightforward.

### 1.2 The repository already treats R0 as a premise, in words

Three places say what the proposal wants said, and disagree with the question
list about where it belongs:

- `docs/questions.md`: "All three rest on one premise, tested first (R0): a
  microscope samples fields, not objects."
- `README.md`: "The report opens with the premise every other answer rests
  on."
- `docs/architecture.md` line 162: "`morphocoverage.sampling_unit` answers
  the report's premise (R0 ...)".

A premise that every moment of use rests on is not one of REPORT's questions;
the proposal resolves an inconsistency the documents already contain. The
`sampling_unit` module docstring and the report's module docstring number it
as question 1 of the report, so they would change with it.

### 1.3 The three-level hierarchy is already the package's model

The proposal's diagram (object nested in FOV nested in sample/repeat) is the
README's "Why FOVs matter" hierarchy and the `SamplingHierarchy` contract
(`retrospective/hierarchy.py`), and its sentence "cells within one dish are
not separate treatment replicates" is `docs/project_aim.md` section 1.1. The
questions the proposal attaches to each level (object: what phenotypes and
how frequent; FOV: how did they enter the acquisition, what will another
image add; sample: what supports inference across treatments) match R2/R3,
P10 and the project boundary in AGENTS.md section 4. Nothing in the diagram
contradicts a rule; it can be adopted as written.

### 1.4 The "what do I want to preserve?" front door is designed but not built

`docs/proposals/microscopist_report.md` ("Plain-language front door") maps
"a reliable average phenotype", "the full phenotype distribution", "minority
or rare states" and "spatial structure (diagnosis, not adequacy)" to criteria
and outcomes. `docs/questions.md` R2 separates "the average or common
phenotype, the phenotype distribution, and rare or minority phenotypes", with
the distribution marked partial. The pieces exist below the workflows:

- distribution outcomes: `CovarianceError`, `QuantileError`, `FrechetError`,
  `EnergyError`, `MMDError`, `SlicedWassersteinError`
  (`multivariate_outcomes.py`);
- distribution estimators: `FOVBootstrapStrategy` (covariance, quantiles,
  PCA spectrum or subspace), `SupportNoveltyStrategy`;
- rare outcomes and planning: `RareStateDetected`, `FOVIncidencePlanStrategy`,
  `RareIncidenceStrategy`, `RareTarget`;
- `sampling_reliability` and `stopping_calibration` accept any
  `OutcomeCriterion`, so a fingerprint column is a matter of passing one
  (`data/crick_idr/reliability_idr0028.py` already passes `CovarianceError`).

What is missing is the declaration and the wiring:

- `workflows.Setup` has `readout` and `tolerances` (the mean within a
  fraction), `novelty` (the definition of new) and a `PositiveControl`. It
  has no field for a distribution criterion or a `RareTarget`. The `optimise`
  workflow builds its criteria from `RelativeMeanError` only
  (`workflows/retrospective.py`, `_criteria`), and every workflow runs under
  `ObjectiveConfig(SamplingObjective.POPULATION)`.
- The objective model has two components, `Component.POPULATION` and
  `Component.RARE` (`evidence.py`). Every distribution-type quantity
  (`FOVBootstrapStrategy` covariance, `SupportNoveltyStrategy` novelty, the
  JCS descriptor spread) is tagged `POPULATION`. The proposal's ONLINE view
  ("Mean phenotype: STOP / Phenotypic fingerprint: CONTINUE / Rare-state
  target: CONTINUE") cannot be rendered from `component_decisions` today,
  because mean and fingerprint are the same component. `docs/questions.md`
  O4 ("if the objectives disagree, is that obvious?") is marked partial for
  the related reason that the per-component view is not rendered.

### 1.5 The proposal's OPTIMISE table is aspirational on the committed data

The illustrative table (5 to 30 FOVs; mean 92 to 100%, fingerprint 55 to
98%, 1% state 15 to 73%) is the right shape and the right message, but on the
data in the repository today:

- **Mean column:** available. `sampling_reliability` on idr0028 gives 98 to
  100% at 2 fields for the mean within 5% (findings entry 4). The curve stops
  at 18 fields because 28-field wells leave 18 after a 10-field reservoir.
- **Fingerprint column:** computable but capped. The one distribution
  criterion run so far, the variance within 10%, was met by all 18 stream
  fields together in only 40% of hold-outs (median; 10 to 80% across wells),
  so the curve is flat at the ceiling and the report must say "cannot be
  judged with this reservoir" (findings entry 4; audit F2 and F3). The
  covariance, quantile and MMD criteria have not been run as reliability
  curves on real data. `docs/questions.md` open work item 5 already asks for
  30 or more fields per sample for exactly this.
- **Rare column:** needs independent labels (`RareStateDetected` takes a
  label column; the unusual-object proxy is not a named state, findings entry
  10) and a reservoir design that keeps the few positive fields (audit F11:
  with a fixed reservoir the judged hold-outs describe a thinned incidence).
  The planning line that exists is the binomial incidence plan in the
  sufficiency report ("image about 83 independent representative FOVs at
  the observed incidence"), which is not a reliability curve.
- **"1% state":** the package deliberately never converts object prevalence
  to FOV incidence (`FOVIncidencePlanStrategy` returns `not_applicable` for a
  prevalence-only target, `docs/architecture.md`). A column headed "1% state
  detected" should be headed by FOV incidence, or by a declared `RareTarget`
  that states both, or the proposal's own point (prevalence and incidence are
  different acquisition problems, `docs/sampling_objectives.md`) is lost in
  the table header.

The table should be presented as the target output, with the columns that
today print "cannot be judged" printing exactly that.

## 2. Scientific accuracy of the proposal's argument

The argument is consistent with the committed findings, with three
refinements worth carrying into the Foundation text.

- **The two papers' assumptions** are quoted the way the repository quotes
  them (`sampling_unit/object_rules.py` docstring; findings entry 9): JCB
  "further assume[s] that samples are spatially uniform"; JCS assumes the
  small cell sample is representative and calls that likely reasonable for
  cultured cells. The proposal's "these approaches are not wrong" is AGENTS.md
  section 10 rule 6 and `docs/scientific_contract.md` ("a benchmark, not a
  strawman"). Correct.
- **"250 random cells from the well is not the first 250 cells encountered
  while imaging"** is what entries 1 and 9 measure: the same object count
  gives 2.2 to 2.8 times the predicted averaging error on idr0139 and 1.4 to
  1.5 times on idr0028, and the JCS treatment ranking at 250 objects falls
  from 81% correct with random objects to 55% with whole fields. Correct, and
  the Foundation can cite these numbers as its worked example.
- **Refinement 1: what matters is not only spatial structure.** The proposal
  says the distinction "may barely matter" if cells are uniform and "can
  matter enormously" if phenotype is spatially structured. The data say the
  driver is the design effect, about `1 + (m - 1) * ICC`: the between-field
  share of variance **and** the objects per field. On idr0028 only 3.8% of the
  variance lies between fields, yet a whole-field mean has 2.1 times the
  variance of a random-object mean, and the JCB target is still met because
  a field holds about 30 objects and the readout's object SD is small. On
  idr0139 the between-field share is smaller (1.6%) but a field holds about
  270 objects, so the design effect is 6.4 and the rule misses its target in
  77% of the plate's wells. The Foundation should name both drivers, or a
  reader with homogeneous-looking fields will conclude fields do not matter.
- **Refinement 2: FOVs are not the independent unit either.** The proposal
  lists spatial autocorrelation as a Foundation item, which is right; the
  section should also carry the standing caveat that whole-field resampling
  is itself optimistic when neighbouring fields are alike (AGENTS.md section
  3; every `sampling_unit` docstring), so "count fields" is not read as
  "fields are independent draws".
- **Refinement 3: "not every cell in an FOV is a pseudoreplicate" needs its
  companion sentence.** As a statement about biological meaning it is right
  and it is the tone the repository wants (objects remain the phenotype
  observations). As a statement about resampling it must not be read as
  permission: the non-negotiable rule is that objects of one FOV enter
  together and are never bootstrapped or revealed individually. One sentence
  in the Foundation keeps both.
- **The softened README sentence** ("a microscope acquires phenotype
  observations through spatial fields, not as an arbitrary random sample of
  individual objects") is accurate and better than the current one. The
  same change belongs in `docs/questions.md` ("a microscope samples fields,
  not objects") and in the `sampling_unit` module docstring ("A microscope
  samples FOVs, not objects").

## 3. Where the proposal needs a decision before it is built

Each of these changes a public contract, a scientific choice or the project
boundary, so under AGENTS.md section 15 it is the owner's call, not an
implementation detail.

1. **A third objective component (population summary, distribution, rare).**
   `Component` and `SamplingObjective` are in `morphocoverage.__all__`.
   Adding `Component.DISTRIBUTION` and re-tagging the distribution quantities
   of `FOVBootstrapStrategy`, `SupportNoveltyStrategy` and the descriptor
   spread changes which strategies satisfy a `POPULATION` objective and what
   `balanced` means; every contract test that asserts components would move.
   It is the smallest change that makes the proposed ONLINE view honest, and
   the alternative (keeping "fingerprint" inside POPULATION and rendering the
   view from quantity names) would put a scientific distinction in string
   matching. Recommended: add the component, as an explicit, approved API
   change, with the existing two-component objectives unchanged in meaning
   for population-only strategies.
2. **"Condition differences" as a fourth user-facing objective.** AGENTS.md
   section 4 draws the line: condition-level analyses (effect sizes,
   treatment rankings) enter only as **retrospective benchmark outcomes**,
   never as inputs to a prospective estimator, and the package does not own
   the user's biological inference. The whole-field treatment comparison
   exists as a script (`data/crick_idr/jcs_effect_fovs.py`, findings entry 9)
   and is `docs/questions.md` open work item 2, pending its own contract
   registry. Recommended: offer it in OPTIMISE and BENCHMARK as "does the
   treatment ranking survive whole-field sampling at this effort?", not as a
   REPORT objective and not in ONLINE, where it would need a live
   between-condition test.
3. **A one-line Foundation verdict** ("object count alone is / is not an
   adequate description of acquisition effort"). Any such line needs a
   threshold on the design effect or the error ratio, and the repository's
   rule is no hidden preset. Two defensible forms exist without a new
   threshold: state the paper's own target ("the JCB object count met its
   2.5% target with whole fields in 0 of 12 groups; missed it in 236 of 308"),
   which `object_rule_check` already computes as `meets_target`; or state the
   design effect against the user's declared tolerance. Recommended: the
   Foundation ends with those two measured sentences, and "fields matter" is
   left to the reader, as `docs/objects_or_fields.md` does.

## 4. What the change touches

All wording unless marked. The section key `"unit"` should be kept, since
`MicroscopistReport.summary(["unit"])` and `REPORT_QUESTIONS` use it; only the
title and position change.

| Where | Change |
|---|---|
| `src/morphocoverage/retrospective/report.py` | `QUESTIONS`: a Foundation entry before "1."; module docstring; `_NEEDS["unit"]` |
| `src/morphocoverage/workflows/retrospective.py` | run `redundancy_distance` in `report` when `stage_cols` is set; docstrings that say "R0 to R6" |
| `src/morphocoverage/sampling_unit/__init__.py`, `variability.py`, `object_rules.py` | docstrings: "REPORT, R0" becomes the Foundation; the opening sentence softened as in section 2 |
| `docs/questions.md` | the premise paragraph becomes a "Foundation" section with the seven items of section 1.1; the R0 row leaves the REPORT table; P5 keeps its pointer to the checks; "In one line each" |
| `README.md` | the first row of the question table, the REPORT row of the modes table, the "report opens with the premise" sentence, the "Why FOVs matter" section (add the design-effect driver) |
| `docs/objects_or_fields.md`, `docs/architecture.md` line 162, `data/crick_idr/objects_vs_fields.py` docstring | references to "R0" and "first question of the report" |
| `notebooks/MorphoCoverage_Workflows_Example` cell 15 | the numbered list |
| `tests/test_sampling_unit.py` line 401, `tests/test_workflows.py` lines 79, 107, 109, 242 | assert the new title |
| `docs/preliminary_findings.md` | leave as the dated record; entries 1, 9 and 10 refer to R0 as it was named then |
| **Code, needs decision 1:** `evidence.py`, `objectives.py`, the three strategies, `results.py` (a per-component line in `explain`), contract tests | the distribution component and the ONLINE per-objective view |
| **Code, front door:** `workflows/setup.py`, `workflows/retrospective.py` | declare a distribution criterion and a `RareTarget`; build `optimise` criteria from all three; print the declaration at the top of the report |

The results files under `data/crick_idr/results/objects_vs_fields_*` and the
script names need not change; they are tracked outputs and the smallest
coherent change leaves them.

## 5. Recommended order

1. Wording and order only (the Foundation section, the softened sentences,
   the two refinements of section 2, the test titles): one change, no
   estimator, threshold or `__all__` touched, every existing test still
   meaningful.
2. The front door in `Setup`, with the report echoing what was declared and
   the `optimise` table printing "cannot be judged with this reservoir" where
   that is the truth on the user's data.
3. Decision 1, then the ONLINE per-objective view (closes O4).
4. Decision 2, in OPTIMISE and BENCHMARK only, after the multi-group contract
   registry.
5. Data: the deeper acquisitions that `docs/questions.md` already lists as
   open work are what turn the fingerprint and rare columns from
   "cannot be judged" into numbers.

## Validation

- Every file cited above was read in this session; the mapping of section
  1.1 was checked against the function signatures and the workflow code
  that calls them.
- The full test suite was run on this branch after `pip install -e ".[dev]"`:
  2092 passed, 7 warnings, in about two minutes on every CPU. No code was
  changed for this assessment.

---

# Part 2: what to import from *Points of Significance*

**Date:** 2026-09-26, same branch, after the step 1 wording change (the
Foundation section, the renumbered report). **Scope:** the follow-up proposal
that, with the Foundation separated out, lists ten principles from the
*Points of Significance* series and seven roadmap items, and adds no
estimator. Each item was checked against what the repository already does,
where it says so, and what a change would touch.

## Verdict in one paragraph

The proposal's central conclusion is right for this repository and matches
its own rules: be clear about the population, the estimand, the type of
uncertainty, the sampling hierarchy and the validation target before adding
statistical machinery (AGENTS.md sections 3 and 9; `docs/project_aim.md`
sections 5 and 18). Of the seven roadmap items, four are wording or contract
rules that cost one document each and should be adopted (uncertainty
semantics, benchmark governance, the within-sample / generalization /
inference separation, and the "clusters are hypotheses" rule). Two are small
computations on results the package already produces (base-rate reporting
for flagged fields, a "validated across" table for protocol advice). One is
the same front door as Part 1 (the scientific target contract) and carries
the same decisions. The representation-robustness check is a loop over
representations of an analysis that exists, but its "fingerprint reliability"
column depends on the distribution criterion that is not wired in. The
condition-contrast outcomes are the one item that changes a contract:
outcome metrics are group-local today, and a between-group outcome needs its
own registry, as `docs/questions.md` already lists.

## Item by item

### 1. Scientific target contract

Same as Part 1, section 1.4: designed, not built. Two additions the new
proposal makes are worth taking with it. **Weighting / estimand:** the
package already has both weightings and states them per function
(`ScalarPrecisionStrategy` on the object-weighted mean; `inspect_fovs` on a
FOV-weighted summary; `SupportNoveltyStrategy` reports object-weighted and
FOV-weighted fractions), but nothing declares once which estimand the
microscopist means, and `RelativeMeanError` judges one of them. A `Setup`
field would make that explicit. **Required resolution / prevalence:**
`RareTarget` already requires `phenotype` (the definition of the state) and
prevalence or incidence, so the contract exists for the rare component and
is what the front door should reuse rather than redefine.

### 2. Uncertainty semantics

Mostly present in the sentences, absent as a rule. The next-FOV bound is
printed as "nominal 95% upper bound ... approximate: the leave-one-FOV-out
outcomes share their reference FOVs" (`sufficiency.py`); reliability prints
"± across hold-outs" (`reliability.py`); the rule-of-three line says "if FOVs
are independent, representative draws"; calibration says how often a stop
was premature "in held-back fields". The roadmap's rule 2 requires a unit on
every number. What is missing is the one-line contract the proposal states,
"every interval, bound or percentage says what it covers and at what unit",
in AGENTS.md section 3, and a short table in `docs/scientific_contract.md`
naming the five kinds (parameter confidence, next-field prediction,
population tolerance, empirical calibration, protocol generalization) with
the function that produces each. The **FOV-to-FOV tolerance diagnostic**
("representative fields in this sample typically span X to Y") is one
sentence from quantities `fov_variability` already computes (the per-field
means in `fovs`, the SD of field means in `groups`); as a diagnostic it can
join the Foundation without a threshold. It should not be a stopping
criterion, as the proposal says.

### 3. Base-rate-aware rare-state reporting

The inputs exist and are already adjacent in the report: `representation_control`
gives the positive-control detection rate and the reference false-alarm rate
per field; `_detection_adjusted_bounds` divides the unseen-state bound by the
detection rate; the profile line says how many "new" fields are expected by
chance at the false-alarm rate (audit F12). The positive predictive value at
a declared incidence is one formula on those three numbers, and the
incidence is `RareTarget.fov_incidence`, which the workflows do not yet take.
Recommended, with two cautions to print with it: the false-alarm rate is
measured on declared reference fields with this definition of new, so the
PPV holds for fields flagged by that definition, not for any rare biology;
and it is a per-field statement, so it answers "how many flagged fields are
the target" and not "how many target objects exist".

### 4. Representation robustness

`docs/project_aim.md` names it (scenario 14, "representation disagreement";
success criterion 5), and every analysis takes explicit `feature_cols`, so
the table the proposal sketches is a loop: `representation_control` per
representation for the detection and false-alarm columns, `sampling_reliability`
with a distribution criterion for the "fingerprint reliability" column
(computable via `CovarianceError` and the other multivariate outcomes, but
subject to the reservoir ceiling of Part 1, section 1.5). `transforms.PCA`
exists for the PCA rows. The "sampling conclusion is robust between 10 and
30 PCs" sentence is a comparison across rows and needs no new statistic. The
rule "unsupervised clusters are hypotheses about phenotype structure, not
biological states by default" fits AGENTS.md section 3 beside "do not equate
low expected novelty with proof that no rare biology exists"; the package
has no clustering method of its own (no scikit-learn by design), and
`fov_clustering` takes labels from any source, so the rule governs how such
labels are worded, which `docs/objects_or_fields.md` already does for the
unusual-object proxy.

### 5. Within-sample coverage versus generalization versus inference

Already the repository's structure, stated in three places: the "Retrospective
representability is not the prospective sampling requirement" distinction
and the per-moment "may not use" column in `docs/questions.md`;
`protocol_reliability` counting by independent unit with a
leave-one-unit-out check; the project boundary that condition-level
inference is downstream (AGENTS.md section 4; `docs/project_aim.md` section
8). The proposal's three headings are a clearer statement of the same thing
and belong in `docs/scientific_contract.md` as one short section, with the
hierarchy diagram of Part 1.

### 6. Transportability of protocol recommendations

Partly present: `protocol_reliability` reports per stratum and per independent
unit and validates leave-one-unit-out; the report's standing note says a
field count "is not a protocol until it holds on samples it was not chosen
on"; `docs/questions.md` P2 is marked partial for lack of five independent
units. The "validated across" table is derivable from the declared
`SamplingHierarchy` (fov, sample, technical levels, biological repeat) and
the data: a level is "yes" when the protocol curve spans more than one unit
of it, "not assessed" when the level is not declared. Days, preparations and
microscopes are not columns in the MorphoBricks contract, so they would print
"not assessed" until a user declares them as technical levels, which the
hierarchy already allows. Cheap, and it turns P2's caveat into a visible
scope statement. It also states, as the proposal says, what future benchmark
datasets must vary.

### 7. Benchmark governance

The mechanism exists (`BenchmarkRegistry` roles, `tuned_on`, `check_tuning`
refusing a run tuned on held-out data; `docs/benchmarking.md`). The rule the
proposal adds, "a dataset ceases to be held out for a choice once its
results influence that choice", is not written anywhere and cannot be
enforced by code, since the registry cannot see why a setting was chosen.
It belongs in AGENTS.md section 9 and `docs/benchmarking.md` as a process
rule, with the consequence spelled out: idr0139 remains held out for the
settings frozen from idr0028 before entry 7, and becomes development
evidence for any epsilon, transform, estimator, representation or tolerance
changed because of it. The audit's third pass records that no setting was
chosen on idr0139, so the rule costs nothing today and protects the next
dataset.

### 8. Influence versus abnormality

Already solved as the proposal says: `inspect_fovs` gives one reason per
line with its numbers, no combined score, and never calls a field an
artifact (`docs/questions.md` R5). The heading is "Which fields should I
look at?", which is already the neutral wording; no rename is needed.

### 9. Condition-contrast preservation outcomes

The one item that changes a contract. `OutcomeMetric.evaluate(full, stopped)`
scores a stop against the full acquisition **of the same analysis group**
(`outcomes.py`), and every retrospective analysis is group-local by rule
(AGENTS.md section 3, "never pool groups silently"). A "does the treatment
difference survive `k` fields" outcome compares two groups, so it needs a
multi-group outcome contract with its own registry of scenarios before it
is packaged; `docs/questions.md` lists exactly this as open work item 2, and
the analysis exists as `data/crick_idr/jcs_effect_fovs.py` (entry 9: the
JCS ranking at 250 objects is right 81% of the time with random objects and
55% with whole fields). The proposal's four levels (mean difference,
distribution difference, state-frequency difference, treatment ranking) are
the right target list; today the script covers the mean difference and the
ranking. AGENTS.md section 4 already allows all four as **retrospective
benchmark outcomes**, so the boundary question of Part 1 (decision 2) is
settled in the proposal's favour as long as they stay out of REPORT and
ONLINE.

### 10. Repeated monitoring online

The issue is known and half-worded. `docs/proposals/candidate_estimators_roadmap.md`
section 11.2 states that repeated checking of fixed-sample intervals can
invalidate nominal coverage and defers confidence sequences; `docs/online.md`
says a novelty-bound stop after `k` quiet fields is `1 - 0.05 ** (1 / k)`
whatever the data and that premature stops are the retrospective
calibration's business; the sufficiency report prints "nominal". What is
missing is the two-term rule the proposal gives, "nominal per-look bound"
versus "empirically calibrated stopping risk", applied to every ONLINE
sentence, and the calibration of the same `config_id` attached to the live
result (`docs/questions.md` O6, partial). Agreed that confidence sequences
should not move up the roadmap; the wording rule and O6 are what make the
current semantics honest.

## What to put on the roadmap, and in what form

| Proposed item | Form | Touches | Needs a decision |
|---|---|---|---|
| 1. Scientific target contract | code: `Setup` fields for the estimand, a distribution criterion and a `RareTarget`; the report echoes the declaration | `workflows/setup.py`, `workflows/retrospective.py` | Part 1 decisions 1 and 3 |
| 2. Uncertainty semantics | rule in AGENTS.md section 3; a five-kind table in `docs/scientific_contract.md`; one tolerance sentence in the Foundation | docs; `sampling_unit/variability.py` summary | no |
| 3. Base-rate reporting | one PPV line in the report when a `RareTarget` incidence is declared | `retrospective/report.py`, `Setup` | no, once item 1 gives the incidence |
| 4. Representation robustness | a workflow helper that runs the report's control check over several representations and tabulates | `workflows/` | no; the fingerprint column waits on item 1 and deeper data |
| 5. Three-level separation | one section in `docs/scientific_contract.md` | docs | no |
| 6. Transportability table | a "validated across" block in `protocol_reliability`'s summary from the declared hierarchy | `retrospective/protocol.py` | no |
| 7. Benchmark governance | a process rule in AGENTS.md section 9 and `docs/benchmarking.md` | docs | no |
| 9. Condition-contrast outcomes | a multi-group outcome contract and registry, then metrics for the four levels, then `jcs_effect_fovs.py` packaged against them | `outcomes.py`, `tests/contract_registry.py`, new module | yes: a new contract (AGENTS.md section 15) |
| 10. Online wording | the two-term rule in `docs/online.md` and every ONLINE summary; O6 | `results.py`, `online.py`, docs | no |

Items 2, 5, 7 and 10 are wording and could ride one documentation change.
Items 3 and 6 are small computations on existing results. Item 4 is a
helper. Items 1 and 9 are the real work, and item 1 is the same work as Part
1's step 2.

## The test the proposal ends with

"Every proposed method should have to answer which microscopist question it
improves that the existing simpler tools cannot answer reliably" is already
the repository's rule in other words (AGENTS.md section 3, "permit null
results"; `docs/project_aim.md` section 14, baselines every candidate must
beat or justify exceeding; section 18, falsification criteria). Writing it
as one sentence in AGENTS.md section 20 ("Adding or swapping a strategy"),
next to the contract-test requirement, would make it the first question a
contributor answers.
