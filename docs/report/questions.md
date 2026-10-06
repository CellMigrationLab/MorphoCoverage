# The report's questions, in the order of the page

REPORT answers "I have finished imaging. What can I trust, and what can't I?" from every
acquired field of this dataset and the declared control groups. This page lists its
questions, R1 to R8 with the Foundation, then walks through the HTML page section by
section, in the order a reader meets them. How to run the report is in [Running a
report](running_a_report.md); what the codes and the status words mean is in [the
documentation's start page](../README.md#reading-the-question-tables). The text report
numbers its answers: the Foundation, 1 (R1), 2 (R2 and R3), 3 (R4), 6 (R5), 9 (R7), 10 (R8)
and the closing "What this dataset cannot tell me" (R6); its 4, 5, 7 and 8 are planning
questions of [optimise](../optimise/optimise.md).

## The questions

| # | Question | Answered by | Status |
|---|---|---|---|
| R1 | **Can my analysis detect the phenotype I care about?** Does the selected representation detect a declared positive control, and at what false-alarm rate? | `representation_control`: detection and false-alarm rates for declared controls, or "not assessed"; the unseen-state bound divided by the detection rate | done, for declared controls only |
| R2 | **Did I image enough?** With a control and other conditions, first: do the fields acquired resolve the difference from the control in each repeat (the 95% interval of d from the fields resampled as acquired excludes 0), and if not, about how many fields a side would; `sampling_unit.contrast_sampling`, `retrospective.report_resolved` ([Did I image enough?](did_i_image_enough.md)). Then, separately, for the average or common phenotype, the phenotype distribution, and rare or minority phenotypes. | `sufficiency_report` (what each criterion says, by its own threshold), `adequacy_points` (when held-back FOVs confirmed it) | average: done. Distribution: partial (criteria exist; judged only through distribution distances, which are not zero-centred; the report estimates each well's fingerprint error from split halves of its fields, and compares the distributions between conditions beyond the mean in R7). Rare: with a label, the report says per well whether it was seen in `k` fields and the fields to see it from its observed field incidence; without one, partial (states seen in few FOVs and a planning line). **Benchmark** |
| R3 | **Would another representative FOV likely have shown something new?** What phenotype information might still be missing? | next-FOV novelty chance with its nominal bound, checked by `calibrate_next_fov_novelty`; what the FOVs rule out (about 3 / k) | done. **Benchmark** (calibration of a probability) |
| R4 | **What is limiting my sampling?** Too few FOVs, strong FOV heterogeneity, spatial correlation, rare-state clustering, or a poor representation? | the sampling profile of `microscopist_report`: design effect, variance components, redundancy distance, control detection, fields with new states; rare-state clustering by field (`sampling_unit.fov_clustering`, on labels from `unusual_objects`, which the report reads on values relative to each own control in its repeat and calls exploratory, plan v2, or any other source): field incidence against random objects, and the fields needed to see the state | done; clustering is only as good as its labels |
| R5 | **Which FOVs should I go back and inspect?** Unique states, strong influence, disagreement with neighbours, empty or unusual fields, copied fields. | `inspect_fovs`, one reason per line, no combined score, no FOV called an artifact; with `report(..., integrity=True)`, the fields the exact data-integrity checks name (`integrity_checks`: copied, reordered or scaled fields, shared or repeated observations, repeated summaries; [Data integrity](integrity.md)) are listed with them | done |
| R6 | **What can this dataset not tell me?** For example: too shallow to assess a 1% state, no positive control, no stage coordinates, only one biological repeat. | the closing section of `microscopist_report`, "What this dataset cannot tell me": every question not assessed with what it needs, then the notes to read the answers with (among them the fields the integrity checks name, which every answer still counts as independent fields); ceilings stay beside the curves they cap | done |
| R7 | **Do the conditions differ, and how confidently?** Each condition against its control: across the biological repeats first, the repeat as the replicate (SuperPlots): the direction in every repeat, the per-repeat d and from 3 repeats the paired t-test on the repeat means, raw and normalised; beside it, the lab's Plot&Stats cell-level test, labelled with its chance of p ≤ 0.05 without an effect, and balanced when the repeats are not; within each repeat, the tests over whole technical repeats; with a `target` column, the reagents of one target; with several features, which of them separate the conditions; and the plan of the next experiment at the design the user ran. The tests, their assumptions and limitations, and what the report and the page show: [Do the conditions differ?](conditions.md). | `sampling_unit.condition_contrast`, `sampling_unit.feature_discovery`, `sampling_unit.feature_ranking`; section 9 of `microscopist_report`; run by the `report` workflow whenever a control condition is declared and others exist | done, descriptive: what these fields support about a difference under this feature; generalizing needs independent repeats. Whether `k` fields a side preserve each repeat's difference from the control is `retrospective.effect_preservation` (standalone; in OPTIMISE once the multi-group outcome contract is settled, open work item 2) |
| R8 | **Are there groups of cells, and is a condition enriched in one?** Groups of cells that sit together on all the declared features, found on every condition pooled (a rare phenotype is one of them, whatever its condition), then each condition compared with the control on its share of cells in each, across the biological repeats. | `sampling_unit.find_populations` (HDBSCAN on a UMAP of the robustly standardised features, each cluster kept as a group when its cells' neighbours are in it, validated on the images, with the simplest feature rule that picks it out), the enrichment by `feature_discovery` on each population's 0/1 column (the paired test on the repeat shares, Benjamini-Hochberg within each sub-experiment); the PCA and UMAP ring them | done (owner, 2026-10-02) |

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
still makes an object count badly understate the sampling error, while
with a few dozen objects per field the error, though larger than random
objects give (about 1.5 times on idr0028, about 30 objects per field), can
stay small enough for the target. Fields are not assumed independent either: neighbouring fields can
be alike, and whole-field resampling is then itself optimistic. For
resampling and replay the rule stays absolute: the objects of one field
enter together and are never drawn, bootstrapped or revealed individually.

When every field holds one observation (one cell per image, as in manual
single-cell imaging), fields and observations coincide, and the field questions
run on the rows as fields (owner, 2026-10-06, replacing the earlier "not
applicable"): the Foundation gives the spread over fields (a design effect of 1
by construction), R4 the unusual images, R2 and R3 novelty across fields with the
field version of the definition of new (a field's observation is new when no
other field holds one within the radius: a state seen once, exploratory until
module P10 of the validation protocol; `support_novelty.per_field`, which the
estimator applies on such data, so `optimise`, `benchmark` and a live session
judge it alike), R1 the positive control judged by it, and R7 one d
(`workflows._one_per_field`). Only the ideas within a field are said not to apply,
once: unusual cells or a labelled state clustering in a few fields, and the
count rule checked with whole fields (random cells and whole fields are the same
draws).

The Foundation reports, per analysis group: objects per field; the
between-field share of variance, the design effect and the fields that
differ more than random objects would (`sampling_unit.fov_variability`); the
mean-precision rule and the JCS stabilisation, each run with random objects and with
whole fields (`object_rule_check`, `descriptor_spread_check`; also P5);
whether labelled rare or unusual states are spread over the fields or
clustered in a few (`fov_clustering`, reported under R4; a group where none was found
reads "15 cells scored in 3 fields, none unusual", a technical repeat holds a state when one
of its fields holds at least the minimum count of it, and a group whose repeat holds no own
control is "not assessed: its repeat holds no own control", never said to lack values; audit
10, owner, 2026-10-06); and, where stage
coordinates exist, the distance over which neighbouring fields are alike
(`redundancy_distance`, reported under P7; not yet run by the `report`
workflow). Whether the imaged positions were the right ones is a separate
question that needs every candidate position (P9, a gap). Its conclusion is
descriptive, in measured quantities, not a verdict: on this data, a mean from
whole fields varies so many times as much as one from as many random
objects, and the published object count met or missed its own target when
its objects came as whole fields. It is the first section of
`microscopist_report`. The results on the committed data, in plain language,
are in [Objects or fields?](objects_or_fields.md).

## The page, section by section

`microscopist_report` assembles the answers in the order of the questions above, with
a sampling profile of measured quantities; a question without its analysis says "not
assessed" and what it needs. The HTML page orders the same answers by what a reader acts
on first.

### The data

"The data" gives one row per check with its chip (the data checks; the integrity
checks, whose fields are listed under "Fields to check"; and three checks of the values:
**fields more alike than chance**, a feature whose field means agree more closely than
cells drawn at random would let them in at least half of the groups, a lower-tail
analysis of variance at p < 0.001, as values normalised per image, rounded or copied
make them; and **few distinct values**, a feature whose most common value holds 90% of
its observations, or that takes at most 25 values with the most common holding a third,
kept in the analysis but named; and **duplicated or derived features**, declared features
that repeat one measurement, one row per group: copies, exact affine relations such as
1 − x or a change of units, and near copies at |r| ≥ 0.999, so that one measurement counts
several times in each test family, and with several controls also within each
experiment, a relation found there only named with where it holds; nothing is dropped,
keep one of them, owner 2026-10-05); and the **acquisition anomalies**, on the raw values
as quality control, never a biological verdict: the fields 5 SDs or more (an engineering
QC threshold) from the other
fields of their condition and repeat (the larger of their robust SD and SD, in groups of 8
fields or more; `sampling_unit.acquisition`), and with several controls the controls apart
in intensity (owner, 2026-10-06, plan v2)
and a line naming what
was not declared and so not checked; the tables, what was checked (the
declaration in short lines) and the declaration in full are under "Show more".

### Fields to check

"Fields to check" lists each field to open and why: the fields the integrity
checks name, then those that sway the result, have an unusual count or show a
state no other field shows. A field sways the result when leaving it out moves its
condition's mean in its repeat, every cell weighing one as in the paired t-test, by
10% or more, and that change stands out from the other fields' of its group: more than
3 robust SDs (1.4826 times the median absolute deviation) from the group's median
leave-one-out change. With 3-5 fields a group, leaving almost any field out moves the
mean by 10%, so the 10% alone listed noise; a group of fewer than 4 fields is not judged,
and the groups are named once (owner, 2026-10-05). The reason gives the condition's
change against the control and the paired t-test's p with the field and without it (a
control field: the condition it moves most), on the values the test compares (raw with
several controls). Without a control, it is the field-weighted mean of its group, by the
same rule.

`inspect_fovs` lists fields to reopen, one reason per line with its numbers: a new state (support novelty), influence on a declared FOV-weighted summary (mean or an interpolated quantile) that also stands out from the group's other fields (`influence_spread`, 3 robust SDs from the median change; groups of fewer than 4 fields not judged; owner, 2026-10-05), an observation count far from the median, or a difference from the nearest stage neighbours. `novelty_cols` names the features the new-state reason is judged on, so the other reasons can read the readout. There is no combined score and no FOV is called an artifact.

### The whole dataset at once

The PCA and UMAP of every observation on all the features, one map per feature table:
[Feature tables and maps](feature_tables_and_maps.md).

### Are there populations or groups of cells?

Population enrichment is exploratory (plan v2, owner, 2026-10-06): not yet validated by the
benchmark of [the validation protocol](../common/validation.md), and the report and the page
say so once in this section.

Question 10, in every report, finds
groups of cells that sit together on all the declared features, from every
condition pooled. The candidates are the clusters of HDBSCAN (scikit-learn) on a
5-dimensional UMAP of the robustly standardised features (`min_dist` 0), a
cluster holding at least 0.5% of the cells (and 15). UMAP evens out density, so
each candidate is then judged on all the features (owner, 2026-10-02): it is kept
as a **group** (`G1`, `G2`, ...) when 94% or more of its cells' 10 nearest
neighbours are in it too. Whether a group is a separate state or the end of a
continuum (the most elongated cells of a gradient, say) is left to the images:
no test of a gap of low density, along the line between the centres or along a
held-out discriminant, told the two apart reliably at the sizes of an experiment
(owner, 2026-10-02). Each group comes with the simplest rule on the features that
picks out its cells (at most two thresholds from a balanced decision tree, in the
features' own units), fitted on all the cells, with the share of its cells it finds
and the share right of those it picks when the same procedure is refitted with whole
fields held out (5 folds of fields, scikit-learn's `StratifiedGroupKFold`, fewer
with fewer fields; cells held out for a group in one field, and the report says so):
these describe the procedure, not the exact thresholds shown (owner, 2026-10-05);
"What to do next" names the fields
that hold most of it, to validate on the images, and gives the code that labels
it with the rule and declares it as the rare phenotype. A cluster holding most
cells is the main population (`P1`), the rest of the cells. On constructed data
this finds no group in a Gaussian, a separate blob as a group its rule finds,
and the end of a straight line at most as a group; groups overlapping strongly
on every feature are not told apart. Without a group the answer is ➖ "No group":
no candidate group passed the check on these features, which says nothing of a
continuum or of rare states (owner, 2026-10-05). With [other feature
tables](feature_tables_and_maps.md#other-feature-tables-embeddings), the groups are searched
in each space apart, the explainable features, each table and the weighted combined space,
and each distinct group is reported once, naming the spaces that found it (groups of two
spaces merged at a Jaccard index of 0.5 or more); a group the explainable features did not
find has no rule ("found in DINOv2 only; no rule in the explainable features describes it"),
and the enrichment tests run on the merged list (owner, 2026-10-06). With `normalize`, the
groups are searched on the normalised features; a cell whose condition has no own control
in its repeat has no normalised value, and those cells are named once, apart from the cells
with a missing value ("36 cells of b1 and b2 in R4 left out: no own control"; owner,
2026-10-06, audit 8). Each group is listed with its share, the
features that set it apart, its neighbours inside, its rule, the fields and
repeats it is seen in, and the share of its cells in its largest field (a group
living mostly in one field may be that image's artefact; the table, `Populations.table`,
also counts the technical repeats holding it, `largest_field` and `technical_repeats`;
owner, 2026-10-05); then each condition is compared with its own
control on its share of cells in it, per biological repeat, on the log odds of
the shares (a small share is as steady there as a large one), by the paired
test across repeats (Benjamini-Hochberg within each sub-experiment); the main
population is listed but not tested, and a population absent from a condition
and its control in every repeat is not compared (absent from both in some
repeats only, it counts as no change there, a log odds ratio of 0, and the
repeat stays in n). A declared label is matched to the population that holds
it; a group 90% or more of whose cells carry the label is the label's own, and
is not tested a second time: its share is compared in "Do the conditions
differ?", and the card says so. The PCA and UMAP ring each population and group (click its legend
entry to hide it), can be coloured by condition, population or any feature, and,
around points drawn with the lasso, list the conditions and populations they
come from and the features that set them apart (Cohen's d).

### Do the conditions differ?

Question 9, each condition against its control: [Do the conditions
differ?](conditions.md), with the tests, the contrast table of several controls, and what
the page shows.

### Did I image enough?

Question 2, R2 with the next-field question R3: [Did I image enough?](did_i_image_enough.md).

### What limits the sampling? Why count fields?

The sampling profile (R4) and the Foundation above, in the words of the observations
("not only images", "random images"); with one observation per field the headline says so
rather than a ratio of 1, the full form gives each measured quantity with where it holds,
never the name of the function that made it, and question 2 reads "no rule met in any of the
6 groups" and "1 needs more wells or biological repeats" (audit 10, owner, 2026-10-06).
Their results on the committed
data, in plain language, are in [Objects or fields?](objects_or_fields.md).

### Can these features detect what I care about?

`representation_control` judges declared positive-control FOVs (and positive cells planted into reference FOVs) against random reference FOVs holding observations, paired within `pair_cols`: detection and false-alarm rates, "positive only" without a reference, "not assessed" without a control. It also measures both rates leaving one field out within the analysis groups of the inspection (`lofo_group_cols`), which the report's chance clause ("false alarms alone would give about 8, up to 25, of the 336 fields judged, so these are compatible with chance"; audit 10, owner, 2026-10-06) and its detection-adjusted bound read, pooled over the strata of `pair_cols` and said so ("pooled over all plates", owner, 2026-10-05).

With a label and no positive control, question 1 is answered by the label: for each labelled cell, whether
the nearest cell of another field (on the declared features, robustly
standardized as the other explainable spaces are, the SD where the MAD is under a quarter of
it, each explainable table weighted by MFA when there are two or more; owner, 2026-10-06,
audit 8) is labelled too, against the share labelled overall, with
`minimum_detection` as for a positive control. Every labelled cell is looked up
against its true nearest cell of another field, and the answer gives how many were
assessed. A declared label the check cannot assess (fewer than 5 labelled cells, no
unlabelled cell to compare, a missing feature value, or no labelled cell with a cell
of another field) is answered ➖ "declared, not assessable" with the reason, never
as if no label were declared (owner, 2026-10-05).

### What to do next

An answer that calls for a look at the images ends with **What to do next**: a
sentence, and the code to run it on `result`, `objects` and `fovs`, folded under
"Show the code". After "Fields to check", once the images confirm that a field should
be excluded, the report again without it (the code lists every flagged field, the
integrity checks' included: delete the line of each one to keep); after a group, open
the images of its fields, then label it with its rule and declare it as the rare
phenotype (`RareTarget`, `rare_label`, its share of cells the rule's), so the report
says whether it was seen in enough fields.

### Plan the next experiment

The page ends with **Plan the next experiment**, two lists to pick from, each choice
with its code filled in from these data:

- *Want to optimise your imaging?* What the next experiment should get right: the mean
  of a feature in every well (the readout first, then the features significant across
  repeats; with the readout, how the stopping rules did on these fields), a group of
  cells or the declared rare phenotype seen in at least 3 fields of every well (planned
  on the condition where it is rarest), the declared fingerprint: `optimise` on these
  data, with field counts that fit the wells once its reservoir is held back. A
  difference from the control is planned in biological repeats, not fields: for each
  condition not yet significant across repeats, the repeats the paired t-test would need
  for 80% power, with the cells as now and twice as many (with fewer than 3 repeats, how
  many to plan), and no code.
- *Want to focus on a specific phenotype on the microscope?* Stop each well once a group
  or the rare phenotype has been seen in 3 fields, or once enough fields without it show
  it rarer than where it is rarest here (`RareIncidenceStrategy`, with
  `live_session(..., objective=ObjectiveConfig("rare_discovery", rare=target))` so the
  stop does not wait for a mean), the acquisition labelling each field's cells as it
  writes them; or once the mean of a feature is known (the other declared targets
  cleared, so the stop does not wait for rules it does not run). Live decides when to
  stop imaging a well, not where to image.
