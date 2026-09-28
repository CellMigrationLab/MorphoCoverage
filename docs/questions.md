# The questions MorphoCoverage answers, by moment of use

This is the canonical list of the microscopist's questions. Every analysis,
report section and benchmark should name the question it answers here.

The questions are grouped by the three moments of use. Each moment fixes
what information may legitimately answer its questions. A fourth way of using
the package, BENCHMARK, compares methods on these questions
(`docs/benchmarking.md`); it answers no microscopist's question of its own.

| Moment | The microscopist asks | May use | May not use |
|---|---|---|---|
| **REPORT** | "I have finished imaging. What can I trust, and what can't I?" | every acquired FOV of this dataset | a statement about another FOV without a calibration of it; any claim about other samples |
| **OPTIMISE / PLAN** | "What should I do next time?" | completed datasets, with outcomes judged on held-back FOVs | a best or minimal subset as a recommendation; tuning on held-out datasets; subset draws counted as samples |
| **ONLINE** | "I am imaging now. Should I acquire another field?" | only the FOVs acquired so far | any future FOV: its identity, metadata, features, or its effect on scaling, PCA or thresholds |

## Foundation: why count fields, and not only objects?

All three moments rest on one foundation, measured first, before any
question is answered. Published cell-number guidance answers a legitimate
question: given a representative sample of objects, how many are needed for
the quantity of interest? Zweifach (JCB 2024) asks how many cells to average
for a target averaging error and assumes a spatially uniform sample;
Marcotti et al. (JCS 2026) ask when the descriptors of a metric stabilise as
cells are subsampled at random from every imaged field of a well. Neither
paper is wrong under its assumption. A microscope, however, acquires
phenotype observations through spatial fields, not as an arbitrary random
sample of individual objects: every object of a field enters together, and a
microscope that stops early acquires whole fields, not the random cells the
JCS saving argument ("a few hundred cells is enough") draws after the whole
well was imaged.

Objects therefore remain the phenotype observations (what states exist, how
frequent they are), nested in fields (how those observations entered the
acquisition, and what another field would add), nested in samples and
repeats (what supports inference across conditions and experiments). We
count fields not because objects stop being meaningful, but because fields
describe how the object population was sampled.

Whether that matters for a given dataset is measured, not assumed, and it
depends on two things: how much of the variance lies between fields and how
many objects a field holds. The design effect of a mean from whole fields is
about `1 + (m - 1) * ICC` for `m` objects per field and between-field share
`ICC`, so a small between-field share with hundreds of objects per field
still makes an object count badly understate the sampling error, while a
dataset with a few dozen objects per field can behave almost like random
objects. Fields are not assumed independent either: neighbouring fields can
be alike, and whole-field resampling is then itself optimistic. For
resampling and replay the rule stays absolute: the objects of one field
enter together and are never drawn, bootstrapped or revealed individually.

The Foundation reports, per analysis group: objects per field; the
between-field share of variance, the design effect and the fields that
differ more than random objects would (`sampling_unit.fov_variability`); the
JCB rule and the JCS stabilisation, each run with random objects and with
whole fields (`object_rule_check`, `descriptor_spread_check`; also P5);
whether labelled rare or unusual states are spread over the fields or
clustered in a few (`fov_clustering`, reported under R4); and, where stage
coordinates exist, the distance over which neighbouring fields are alike
(`redundancy_distance`, reported under P7; not yet run by the `report`
workflow). Whether the imaged positions were the right ones is a separate
question that needs every candidate position (P9, a gap). Its conclusion is
descriptive, in measured quantities, not a verdict: on this data, a mean from
whole fields varies so many times as much as one from as many random
objects, and the published object count met or missed its own target when
its objects came as whole fields. It is the first section of
`microscopist_report`. The results on the committed data, in plain language,
are in `docs/objects_or_fields.md`.

## Declare the target: what must the acquisition preserve?

Right after the Foundation, and before any question is answered, the
microscopist declares what the acquisition must preserve, because there is
no "enough" without "enough for what". Three targets are declared today
(`morphocoverage.workflows.Setup`): a **population summary** (the mean of the
readout within a tolerance: `readout`, `tolerances`), a **phenotypic
fingerprint** (the distribution of phenotypes, as outcome criteria on the
declared features: `fingerprint`), and a **rare phenotype** (a `RareTarget`
with its definition, prevalence and/or field incidence, detection probability
and the fields it must be seen in, plus the label column that marks it when
one exists: `rare`, `rare_label`). Each becomes an outcome criterion judged
on held-back fields in OPTIMISE and BENCHMARK (`Setup.criteria()`), and every
document opens with the declaration in sentences (`Setup.declaration()`); a
target that is not declared is printed as not declared, so an answer about
the mean is never read as an answer about the distribution or about rare
states. A fourth target, the **difference between conditions**, is reported
whenever a control condition is declared and others exist (R7); as a
preservation outcome for OPTIMISE it waits on the multi-group outcome
contract (open work item 2).
The live decision runs under the objective made of the declared targets
(`Setup.objective()`: `population`, `phenotype`, `balanced` or `all`), and
`decision_view` shows each target's decision and the overall (O4).

Two distinctions run through all three:

- **What a rule says is not what the sample achieved.** A criterion's own
  threshold ("the averaging CV is below 5%") is apparent adequacy; held-back
  FOVs preserving the outcome is realized adequacy.
- **Retrospective representability is not the prospective sampling
  requirement.** A completed dataset may be represented by a few well-chosen
  FOVs, while a new experiment that samples FOVs at random needs more to meet
  the same criterion reliably.

Every answer is written for the microscopist, and three rules keep it so:

- **An answer, not only a table.** Each analysis prints the sentences of its
  answer (`summary()`), with the decision they support, and every sentence
  stays traceable to a table column.
- **Plain words.** The sentences say "fields", "held-back fields", "what you
  wanted to preserve" and "how often it would have worked"; reservoir,
  criterion, prefix, outcome metric and `config_id` appear only in the
  detailed tables. Every number comes with its unit (FOVs, % of acquisitions,
  µm, minutes).
- **Only scientific choices are asked.** What to preserve and how closely
  (for example "the well mean within 5%") are declared in words, with
  provisional presets that the summaries print; computational settings
  (outer splits, draws, seeds) have documented defaults sized for a result in
  minutes.

Status: **done** (implemented and reported), **partial** (implemented, but
part of the question is not answered or not reported), **gap** (not
implemented). **Benchmark** marks the questions where methods compete and
are scored against each other (`docs/benchmarking.md`); the others are
measurements or diagnostics to report, not to rank.

## 1. REPORT: "I have finished imaging. What can I trust, and what can't I?"

| # | Question | Answered by | Status |
|---|---|---|---|
| R1 | **Can my analysis detect the phenotype I care about?** Does the selected representation detect a declared positive control, and at what false-alarm rate? | `representation_control`: detection and false-alarm rates for declared controls, or "not assessed"; the unseen-state bound divided by the detection rate | done, for declared controls only |
| R2 | **Did I image enough?** Separately for the average or common phenotype, the phenotype distribution, and rare or minority phenotypes. | `sufficiency_report` (what each criterion says, by its own threshold), `adequacy_points` (when held-back FOVs confirmed it) | average: done. Distribution: partial (criteria exist; judged only through distribution distances, which are not zero-centred). Rare: partial (states seen in few FOVs and a planning line; realized adequacy needs independent labels). **Benchmark** |
| R3 | **Would another representative FOV likely have shown something new?** What phenotype information might still be missing? | next-FOV novelty chance with its nominal bound, checked by `calibrate_next_fov_novelty`; what the FOVs rule out (about 3 / k) | done. **Benchmark** (calibration of a probability) |
| R4 | **What is limiting my sampling?** Too few FOVs, strong FOV heterogeneity, spatial correlation, rare-state clustering, or a poor representation? | the sampling profile of `microscopist_report`: design effect, variance components, redundancy distance, control detection, fields with new states; rare-state clustering by field (`sampling_unit.fov_clustering`, on labels from `unusual_objects` or any other source): field incidence against random objects, and the fields needed to see the state | done; clustering is only as good as its labels |
| R5 | **Which FOVs should I go back and inspect?** Unique states, strong influence, disagreement with neighbours, empty or unusual fields. | `inspect_fovs`, one reason per line, no combined score, no FOV called an artifact | done |
| R6 | **What can this dataset not tell me?** For example: too shallow to assess a 1% state, no positive control, no stage coordinates, only one biological repeat. | the closing section of `microscopist_report`, "What this dataset cannot tell me": every question not assessed with what it needs, then the notes to read the answers with; ceilings stay beside the curves they cap | done |
| R7 | **Do the conditions differ, and how confidently?** Each condition against the declared control, per repeat: the effect size on objects (as the lab's Plot&Stats notebook computes it), on field means and on technical-repeat means, a randomization test that shuffles condition labels over whole technical repeats (each side at least two; no test with one on a side) or, without a technical-repeat column, whole fields, and the JCS effect size from `n` objects drawn as random objects and as whole fields. With several features, which of them separate the conditions: the same test per feature, Benjamini-Hochberg gated, ranked by \|d\| as fiberfeatures scores a feature set; the plots draw the best-ranked. | `sampling_unit.condition_contrast`, `sampling_unit.feature_ranking`; section 9 of `microscopist_report`; run by the `report` workflow whenever a control condition is declared and others exist | done, descriptive: what these fields support about a difference under this feature; generalizing needs independent repeats. Whether a stop at `k` fields preserves the contrast is an OPTIMISE outcome, pending the multi-group outcome contract (open work item 2) |

## 2. OPTIMISE / PLAN: "What should I do next time?"

This is the experimental-design part of MorphoCoverage. A number here is a
recommendation only when it rests on what representative samples of FOVs
preserved in FOVs they did not include, and only once it holds on samples it
was not chosen on.

| # | Question | Answered by | Status |
|---|---|---|---|
| P1 | **How many FOVs should I acquire per sample next time?** At 10, 15, 20, 25 FOVs, how often does representative sampling preserve the required information in unseen FOVs? | `sampling_reliability` against a fixed held-back reservoir, with its ceiling | done within a sample. **Benchmark** (a method's recommended number against the realized need) |
| P2 | **Does that recommendation generalize across samples and biological repeats?** Not "20 worked in this well", but "20 worked in 75% of held-out samples; 25 in 93%". | `protocol_reliability`, counted by independent unit, with a leave-one-unit-out check; held-out datasets (`data/crick_idr/datasets.py`) | partial: implemented; needs at least 5 independent units and deeper held-out data |
| P3 | **What did the additional imaging buy?** From 5 to 10 FOVs, 10 to 20, 20 to 30. | the `gain` column of the reliability curve; adequacy points | partial: the gains are computed but not printed as increments |
| P4 | **When did additional imaging become mostly redundant?** At the same point for common and rare phenotypes? | sustained realized adequacy per criterion, with its 90th percentile | partial: rare phenotypes need independent labels |
| P5 | **Would a simple sampling rule have been sufficient?** A fixed number of FOVs, a fixed area, a fixed object count, the JCB/Zweifach rule, the JCS descriptor rule? | the packaged baselines, replayed and calibrated; fields saved at matched risk (`retrospective.frontier`, `data/crick_idr/benchmark.py`); whether the JCB and JCS object-count rules hold when their objects come as whole fields (`sampling_unit.object_rule_check`, `descriptor_spread_check`, as in the Foundation) | done for the mean and for planted rare states on idr0028, with transfer to idr0139; the distribution questions need deeper data. **Benchmark** |
| P6 | **How reliable is a proposed automated stopping rule?** When it says STOP: how often is it correct, how often premature, how much acquisition does it save? This is ONLINE 6 seen from the planning side. | `stopping_calibration` on replayed acquisitions; savings and risk at matched effort (`retrospective.frontier`) | done. **Benchmark** |
| P7 | **Am I repeatedly imaging essentially the same thing?** How strong is spatial redundancy, and over what distance? | `redundancy_distance`, FOV semivariogram | done, as a diagnostic, not an adequacy claim |
| P8 | **Could I acquire the same information more efficiently?** Would random or spatially dispersed orders beat the recorded raster, and at what cost in stage travel and time? | `order_efficiency` with recorded, random and dispersed orders | done |
| P9 | **Were my selected locations representative of the specimen?** | needs every candidate position from an overview (a new input contract) | gap |
| P10 | **Should I spend effort on more FOVs or more independent samples?** Is the remaining uncertainty within FOVs, between FOVs, between samples or between repeats, and would the next unit of effort be better spent on a FOV, a sample or a repeat? | `effort_projection` on a declared hierarchy, or on the one `Setup` derives from a `technical_repeat` column (field within technical repeat within repeat), at equal total FOVs; JCB multi-sample planning | partial: implemented; needs at least 5 independent units; no cost model yet. **Benchmark** (projection against JCB planning, by simulation) |

## 3. ONLINE: "I am imaging now. Should I acquire another field?"

Deliberately narrower: the microscope knows only what it has acquired, and
the question is continue or stop for representative acquisition. "Where
should I image next?" is not part of it; that is the later targeted-sampling
extension, which changes the sampling distribution and needs its own design.

| # | Question | Answered by | Status |
|---|---|---|---|
| O1 | **Should I continue or stop after this FOV?** | `OnlineSession.add_fov` | done |
| O2 | **Why?** A short explanation, such as "population representation looks adequate, but the rare-state objective is not yet met". | `StrategyResult.explain()`: the rule, its evidence and the effort in one line | partial: one line per rule; no combined sentence across objectives |
| O3 | **Which objective is driving the decision?** The population summary, the phenotypic fingerprint, the rare phenotype, or several. | `component_decisions` of each result; `rule_decision` for what the rule itself said | done |
| O4 | **If the objectives disagree, is that obvious?** For example: "Population summary: stop / Phenotypic fingerprint: continue / Overall: continue". | `results.decision_view` over the rules of one acquisition state, printed by every live update | done; a target no rule decides is printed as such |
| O5 | **How uncertain is the evidence?** A borderline quantity must not become an unconditional STOP. | upper bounds on some quantities (novelty bound, upper-95 SD) | gap: no margin or "borderline" state in the decision |
| O6 | **How does this stopping rule perform on previously acquired datasets?** When this exact rule said STOP in replayed deep datasets, how often was it adequate? This is PLAN 6 seen from the microscope. | `stopping_calibration` of the same `config_id` | partial: not attached to the live session. **Benchmark** |
| O7 | **Can the decision be made fast enough?** Is the analysis faster than the time before the next acquisition? | `Timing` on every result; measured latencies in `docs/online.md` | done |
| O8 | **Does it behave correctly in awkward real acquisitions?** Empty FOV, few objects, missing feature values, patches rather than cells, z/c/t of one position, unsupported estimators, a new condition or repeat, changed grouping or FOV identity. | input validation and the session contract; contract tests | done |
| O9 | **Is there any future-data leakage?** Scaling, PCA, thresholds and other preprocessing must use only FOVs already acquired. | transforms fitted inside `estimate`; the no-future-leakage contract | done |

## In one line each

- **FOUNDATION: why count fields, and not only objects?** Cell-number rules
  assume a representative sample of objects; a microscope acquires objects
  through spatial fields. Does that difference matter for this dataset?
- **REPORT: can I trust the experiment I already performed?** Can I detect
  the phenotype? Did I image enough? What am I still missing? What limited me?
  Which images should I inspect? What can these data not tell me? Do the
  conditions differ, and how confidently?
- **OPTIMISE: how should I design the next experiment?** How many FOVs? How
  reliable is that number across samples? What did extra imaging buy? Could
  simpler rules work? Is the stopping rule safe? Could spatial sampling be
  better? More FOVs or more biological samples?
- **ONLINE: should the microscope acquire another representative FOV now?**
  Continue or stop? For which objective? Why? How safe is that rule on
  retrospective validation? Is it fast and leakage-free?

## Open work, in order

1. The benchmark behind the remaining **Benchmark** rows: the distribution
   questions of R2, P1 (a method's recommended number against the realized
   need), P10 and O6, which need deeper held-out data. The score and the mean
   and rare-state rows are done (`docs/benchmarking.md`, "The score").
2. A multi-group outcome for the benchmark: whether a treatment comparison
   survives whole-field sampling at a given object count. It exists as
   `data/crick_idr/jcs_effect_fovs.py` (the JCS Fig. 3G-I analysis with whole
   fields, `docs/preliminary_findings.md` entry 9) and needs its own contract
   registry before it is packaged (AGENTS.md section 9).
3. P3 printed as increments.
4. O5, a borderline state in the decision, and O6, the calibration of the
   same rule attached to the live session.
5. Data: deeper held-out acquisitions (30 or more FOVs per sample, ideally 60,
   and five or more independent units) for P2, P10 and the distribution and
   rare rows of R2: the one held-out dataset, idr0139, has 4 FOVs per well,
   so it can refute a development conclusion but not set a number;
   independent labels for rare states; an overview for P9.
6. P9, sampling-frame bias: comparing the imaged positions with every
   candidate position of an overview (with covariates such as density or
   region) needs a new input contract (AGENTS.md section 15) and upstream
   support in CellPatcher, so it is not started until the owner approves it.
7. The owner's reading test: on a real dataset, the project owner reads each
   report answer without the code and says what they would do differently;
   not yet done for any section, nor for the whole report on a dataset it was
   not built on.
8. External audits of the stopping-rule calibration (P6) and the protocol
   reliability (P2), each asked for a dataset on which the answer misleads
   while every test passes (AGENTS.md section 14): planned, and run only as
   part of the whole-report audit of 2026-09-25 (`docs/audit/README.md`).
9. P10's intervals: the effort projection's bootstrap intervals covered the
   true sample and field components in only 80–88% of simulated runs, not
   95% (`docs/preliminary_findings.md`, entry 6), so a stated difference
   between allocations also needs five or more independent units; a better
   interval is open.
10. A cost model for P10 (`C = n_samples * C_sample + n_FOV * C_FOV`, supplied
    by the user): today allocations are compared at equal total FOVs, which
    is not equal cost, since a sample costs preparation that a field does not.
11. Excess FOVs beyond the adequacy points in benchmark records, and their
    aggregation into a regime map (`docs/benchmarking.md`): the adequacy
    points exist, the benchmark does not yet report what was imaged past
    them.
12. A diagnostic of technical drift between repeats, batches or plates: a
    shift pooled across groups is invisible to the novelty definition (the
    shifted fields support each other), and the report says "not assessed".
13. Uncertainty semantics: a rule that every interval, bound or percentage
    says what it covers and at what unit, a table in
    `docs/scientific_contract.md` of the five kinds (parameter confidence,
    next-field prediction, population tolerance, empirical calibration,
    protocol generalization) with the function behind each, and one
    field-to-field tolerance sentence in the Foundation ("representative
    fields here span X to Y"). Most sentences already say it; the rule is not
    written.
14. The ONLINE wording rule: every live sentence names a bound as a nominal
    per-look bound and a stopping risk as empirically calibrated, since
    checking a fixed-sample bound after every field does not keep its nominal
    coverage (with O6).
15. The positive predictive value of a flagged field at a declared field
    incidence (`Setup.rare`), from the control's detection and false-alarm
    rates, printed with its two cautions: it holds for fields flagged by this
    definition of new, and it counts fields, not objects.
16. Representation robustness: a workflow helper that runs the control check,
    and a fingerprint reliability once deeper data allow it, over several
    representations (feature sets, PCA sizes) and tabulates them, with the
    rule that unsupervised clusters are hypotheses about phenotype structure,
    not biological states.
17. A "validated across" block in the protocol summary: for each declared
    level of the hierarchy (field, sample, technical level, repeat), whether
    the protocol curve spans more than one unit of it, or "not assessed".
18. Two process rules for AGENTS.md: a dataset stops being held out for a
    choice once its results influence that choice (section 9 and
    `docs/benchmarking.md`; no setting has yet been chosen on idr0139), and a
    new method names the microscopist's question it improves that the simpler
    tools cannot answer reliably (section 20).
19. Smaller items left open by the 2026-09-25 audit: advice when missing
    values make the novelty question not estimable (drop those observations
    or those features, and say so); a cap on the support-novelty cost, which
    grows about as `k · n²` per decision (5.9 s at 60 fields of 300 objects);
    a warning in `sufficiency_report` when only per-object rules are given,
    and the README example with both units; and a sentence that the
    next-field calibration is noisy when few fields remain.
20. The candidate estimators not yet built (learning curves, kNN and PRDC
    support, abundance-based and diversity estimators, occupancy models,
    spatial ICC, pMoSS, biological-fidelity outcomes, continual-monitoring
    methods) and the benchmark they must pass:
    `docs/proposals/candidate_estimators_roadmap.md`.
21. A calibrated near-duplicate detector (fields whose distributions are
    near-identical but not identical), opt-in and calibrated on imaging
    datasets before any threshold is used; the exact integrity checks are in
    `docs/integrity.md`.
