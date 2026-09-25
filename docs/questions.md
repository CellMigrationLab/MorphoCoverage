# The questions MorphoCoverage answers, by moment of use

This is the canonical list of the microscopist's questions. It supersedes the
numbered list in `docs/proposals/microscopist_report.md`, which gave the
reasons and the first designs. Every analysis, report section and benchmark
should name the question it answers here.

The questions are grouped by the three moments of use, plus a fourth
section, CHECK, that tests the premise all three rest on: that the field, not
the cell, is the unit a microscope samples. Each moment fixes what
information may legitimately answer its questions:

| Moment | The microscopist asks | May use | May not use |
|---|---|---|---|
| **REPORT** | "I have finished imaging. What can I trust, and what can't I?" | every acquired FOV of this dataset | a statement about another FOV without a calibration of it; any claim about other samples |
| **OPTIMISE / PLAN** | "What should I do next time?" | completed datasets, with outcomes judged on held-back FOVs | a best or minimal subset as a recommendation; tuning on held-out datasets; subset draws counted as samples |
| **ONLINE** | "I am imaging now. Should I acquire another field?" | only the FOVs acquired so far | any future FOV: its identity, metadata, features, or its effect on scaling, PCA or thresholds |
| **CHECK** | "Can I count cells, or must I count fields? How variable are my fields?" | every acquired FOV of this dataset; declared control or reference groups | a random-cell sample of the finished well presented as what a microscope would have acquired |

Two distinctions run through all three:

- **What a rule says is not what the sample achieved.** A criterion's own
  threshold ("the averaging CV is below 5%") is apparent adequacy; held-back
  FOVs preserving the outcome is realized adequacy.
- **Retrospective representability is not the prospective sampling
  requirement.** A completed dataset may be represented by a few well-chosen
  FOVs, while a new experiment that samples FOVs at random needs more to meet
  the same criterion reliably.

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
| R4 | **What is limiting my sampling?** Too few FOVs, strong FOV heterogeneity, spatial correlation, rare-state clustering, or a poor representation? | the sampling profile of `microscopist_report`: design effect, variance components, redundancy distance, control detection, fields with new states; rare-state clustering by field in `sampling_unit.field_clustering` (F5) | partial: rare-state clustering is measured (F5) but not yet in the profile |
| R5 | **Which FOVs should I go back and inspect?** Unique states, strong influence, disagreement with neighbours, empty or unusual fields. | `inspect_fovs`, one reason per line, no combined score, no FOV called an artifact | done |
| R6 | **What can this dataset not tell me?** For example: too shallow to assess a 1% state, no positive control, no stage coordinates, only one biological repeat. | "not assessed" lines, ceilings and notes spread across the report | partial: needs one section that lists every limit in one place |

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
| P5 | **Would a simple sampling rule have been sufficient?** A fixed number of FOVs, a fixed area, a fixed cell number, the JCB/Zweifach rule, the JCS descriptor rule? | the packaged baselines, replayed and calibrated; fields saved at matched risk (`retrospective.frontier`, `data/crick_idr/benchmark.py`) | done for the mean and for planted rare states on idr0028, with transfer to idr0139; the distribution questions need deeper data. **Benchmark** |
| P6 | **How reliable is a proposed automated stopping rule?** When it says STOP: how often is it correct, how often premature, how much acquisition does it save? This is ONLINE 6 seen from the planning side. | `stopping_calibration` on replayed acquisitions; savings and risk at matched effort (`retrospective.frontier`) | done. **Benchmark** |
| P7 | **Am I repeatedly imaging essentially the same thing?** How strong is spatial redundancy, and over what distance? | `redundancy_distance`, FOV semivariogram | done, as a diagnostic, not an adequacy claim |
| P8 | **Could I acquire the same information more efficiently?** Would random or spatially dispersed orders beat the recorded raster, and at what cost in stage travel and time? | `order_efficiency` with recorded, random and dispersed orders | done |
| P9 | **Were my selected locations representative of the specimen?** | needs every candidate position from an overview (a new input contract) | gap |
| P10 | **Should I spend effort on more FOVs or more independent samples?** Is the remaining uncertainty within FOVs, between FOVs, between samples or between repeats, and would the next unit of effort be better spent on a FOV, a sample or a repeat? | `effort_projection` on a declared hierarchy, at equal total FOVs; JCB multi-sample planning | partial: implemented; needs at least 5 independent units; no cost model yet. **Benchmark** (projection against JCB planning, by simulation) |

## 3. ONLINE: "I am imaging now. Should I acquire another field?"

Deliberately narrower: the microscope knows only what it has acquired, and
the question is continue or stop for representative acquisition. "Where
should I image next?" is not part of it; that is the later targeted-sampling
extension, which changes the sampling distribution and needs its own design.

| # | Question | Answered by | Status |
|---|---|---|---|
| O1 | **Should I continue or stop after this FOV?** | `OnlineSession.add_fov` | done |
| O2 | **Why?** A short explanation, such as "population representation looks adequate, but the rare-state objective is not yet met". | `StrategyResult.explain()`: the rule, its evidence and the effort in one line | partial: one line per rule; no combined sentence across objectives |
| O3 | **Which objective is driving the decision?** Population representation, rare phenotype discovery, or both. | `component_decisions` of each result | done |
| O4 | **If the objectives disagree, is that obvious?** For example: "Population: STOP / Rare-state target: CONTINUE / Overall: CONTINUE". | the per-component decisions exist | partial: not rendered as one view |
| O5 | **How uncertain is the evidence?** A borderline quantity must not become an unconditional STOP. | upper bounds on some quantities (novelty bound, upper-95 SD) | gap: no margin or "borderline" state in the decision |
| O6 | **How does this stopping rule perform on previously acquired datasets?** When this exact rule said STOP in replayed deep datasets, how often was it adequate? This is PLAN 6 seen from the microscope. | `stopping_calibration` of the same `config_id` | partial: not attached to the live session. **Benchmark** |
| O7 | **Can the decision be made fast enough?** Is the analysis faster than the time before the next acquisition? | `Timing` on every result; measured latencies in `docs/online.md` | done |
| O8 | **Does it behave correctly in awkward real acquisitions?** Empty FOV, few cells, missing feature values, patches rather than cells, z/c/t of one position, unsupported estimators, a new condition or repeat, changed grouping or FOV identity. | input validation and the session contract; contract tests | done |
| O9 | **Is there any future-data leakage?** Scaling, PCA, thresholds and other preprocessing must use only FOVs already acquired. | transforms fitted inside `estimate`; the no-future-leakage contract | done |

## 4. CHECK: "Can I count cells, or must I count fields?"

Every cell-number rule, including the published JCB (Zweifach 2024) and JCS
(Marcotti et al. 2026) rules, assumes the cells it counts are a random sample
of the well. The JCS saving argument ("a few hundred cells is enough") draws
those cells from every imaged field, after the whole well was imaged. A
microscope that stops early acquires whole fields instead, so the argument
holds only if fields behave like random cells. This section measures whether
they do, and what it costs when they do not. It is the premise of sections 1
to 3, and the first row of the regime map: when cell-number guidance is
sufficient and when fields must be counted. All of it is retrospective and
lives in `morphocoverage.sampling_unit`.

| # | Question | Answered by | Status |
|---|---|---|---|
| F1 | **Does a mean from whole fields have the error a cell count predicts?** How much larger is its variance than that of a mean of as many random cells? | `field_variability`: the design effect and the error ratio, from the between- and within-field variance components | done |
| F2 | **How variable are my fields?** How much of the variance lies between fields, and which fields differ more than random cells would? | `field_variability`: the between-field share, a z-score per field against random-cell expectation, the share of fields beyond it (5% expected), a one-way ANOVA | done |
| F3 | **Would a published cell-number rule hold if the cells come as whole fields?** The JCB rule at its own 5% and 2.5% targets; the JCS descriptor stabilisation (Fig. 2A) at the paper's threshold. | `cell_rule_check`, `descriptor_spread_check`: each paper's procedure with random cells and with whole fields; the JCB formula with the field as the unit | done |
| F4 | **Which cells are out of distribution?** Cells unlike a declared reference (control) population. | `unusual_cells`: distance to the reference in standardized feature space, thresholded against reference cells of other fields; a group is never its own reference; a seed check | done; labels depend on the representation |
| F5 | **Are rare or unusual cells spread over the fields, or clustered in a few?** And so how many fields, rather than cells, are needed to see them? | `field_clustering` on any labels (from F4, a mixture model, a gate, annotations): field incidence against random-cell expectation, a permutation test, fields needed from field incidence beside those a cell count asks for | done |
| F6 | **Does a treatment comparison survive whole-field sampling at a given cell number?** | `data/crick_idr/jcs_effect_fields.py` (the JCS Fig. 3G-I analysis with whole fields) | partial: compares groups, so it stays a benchmark script until multi-group outcomes have their own contract (AGENTS.md section 9) |

## In one line each

- **REPORT: can I trust the experiment I already performed?** Can I detect the
  phenotype? Did I image enough? What am I still missing? What limited me?
  Which images should I inspect? What can these data not tell me?
- **OPTIMISE: how should I design the next experiment?** How many FOVs? How
  reliable is that number across samples? What did extra imaging buy? Could
  simpler rules work? Is the stopping rule safe? Could spatial sampling be
  better? More FOVs or more biological samples?
- **ONLINE: should the microscope acquire another representative FOV now?**
  Continue or stop? For which objective? Why? How safe is that rule on
  retrospective validation? Is it fast and leakage-free?
- **CHECK: can I count cells, or must I count fields?** How much do my
  fields differ? Would a cell-number rule hold with whole fields? Are rare
  or unusual cells clustered in a few fields?

## Open work, in order

1. The benchmark behind the remaining **Benchmark** rows: the distribution
   questions of R2, P1 (a method's recommended number against the realized
   need), P10 and O6, which need deeper held-out data. The score and the mean
   and rare-state rows are done (`docs/benchmarking.md`, "The score").
2. R6 as its own report section; P3 printed as increments; O4 as one view.
3. O5, a borderline state in the decision, and O6, the calibration of the
   same rule attached to the live session.
4. Data: deeper held-out acquisitions (30 or more FOVs per sample, five or
   more independent units) for P2, P10 and the distribution and rare rows of
   R2; independent labels for rare states; an overview for P9.
