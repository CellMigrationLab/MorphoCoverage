# What MorphoCoverage said about the lab datasets

One entry per run, newest last: date, dataset, package version (commit), settings, the headline of
each answer, agreement with the dataset's hypotheses (DATASET.md section 4), and surprises. Results
tables and reports of each run are in `results/<dataset id>/`. This log is the cross-dataset record of
how useful MorphoCoverage is on real lab data; it is written after a run, never before.

## 2026-09-30 — first run on the three CCT8-project datasets

**Run.** `PYTHONPATH=src python data/lab/run_lab.py` at commit 40a429f: `report(..., integrity=True)` with
the card's readout, features and control, `default_novelty(features)`, the default rules (JCB per
observation and per field, 5%), `random_state=0`. 7-50 s per dataset. Reports in `results/<id>/`.
Extra checks by hand (not in the stored reports): `integrity_checks` at smaller `min_objects`, and a
copied cell row planted in the CDM tables.

### Headlines

| | filopodia screen (1 cell/field) | CDM shape (1 cell/field) | glioma tracks (1-12 tracks/field) |
|---|---|---|---|
| Data | ✅ no warning, integrity clean | ✅ no warning, integrity clean | ❌ integrity: 3 findings in 6 fields |
| Foundation | ⚠️ not estimated (1 cell per field) | ⚠️ not estimated (1 cell per field) | ⚠️ design effect median 1.0 (1.0-2.7) |
| 2. Enough? | ❌ no rule met in 54 groups | ❌ no rule met in 9 groups | ❌ no rule met in 18 groups; JCB per track undefined |
| 3. Limits | ⚠️ "spread like random cells" | ⚠️ "spread like random cells" | ⚠️ fields 1.0x as variable as random tracks |
| 6. Inspect | ✅ nothing stands out | ⚠️ 4 large cells sway the mean 11-15% | ⚠️ 11 fields (8 "new state") |
| 9. Contrast | ⚠️ 36 per-repeat d, no test | ⚠️ 6 per-repeat d (Area only), no test, 0 of 10 features ranked | ⚠️ 12 per-repeat d, no test, 0 of 19 ranked |

### Against the hypotheses

- **Filopodia screen.** H1 (the per-cell JCB rule met everywhere): **wrong**, met in 0 of 54; the
  rule asks 10-215 cells with its upper-95 SD bound, the groups hold 13-49. H2 (design effect about
  1): not estimable with one cell per field, and the report says so. H3 (no test within a repeat):
  **confirmed**. H4 (directions per repeat): **visible** in the table (MAP4K4 and NF2 above their
  set's control in 3/3 repeats, the other four genes below in 3/3) but never summarised. H5 (the
  control level differs between sets): **flagged**, "control means differ by 2.0x ... may be a batch
  effect or a change of units".
- **CDM shape.** H1 (per-cell rules fail for Area): **confirmed**. H2 and H5 (AR, the paper's
  strongest effect, d 1.55): **not assessed**, because the rules, the Foundation and the contrast
  read the readout (Area) only and the feature ranking is withheld without a test. H3: not
  estimable. H4 (Area lower in 3/3 repeats): **confirmed** in the table. H6 (repeat 3 weakest): not
  singled out; repeat 3 has the largest effect (d -1.13, -0.96).
- **Glioma tracks.** H1 (integrity finds the 28 duplicated tracks): **partly**, 3 of 6 pairs (16 of
  28 tracks). The three missed pairs hold 4 tracks, below `min_objects=5`; 70 of 133 fields are that
  small. With `min_objects=3` all 6 pairs are found and the two clean datasets stay clean. H2
  (design effect above 1): **mostly not supported**: median 1.0, 2.0-2.7 in 4 of 18 groups, 6% of
  fields differ (5% expected). H3: **confirmed**. H4 (CCT8 slower in 3/3, CCT7 mixed):
  **confirmed** in the table. H5 (two control baselines pooled): the 1.8x control shift is flagged,
  again as a possible batch or unit error. The within-repeat track-duration confound (CCT7_2) is
  invisible, as expected: no check looks at acquisition covariates.

### Where the report was useful

- It found copied data (tracks exported twice) that nobody had flagged before curation.
- It flags the control level moving between experimental sets (2.0x, 1.8x).
- It refuses a p-value with one well per side and says why, instead of the cells-as-replicates test
  the paper used.
- The design-effect table for the tracks shows which wells cluster (the CCT7 controls: 40-45% of the
  variance between fields).
- It names the cells or fields that sway a mean (CDM: 4 large cells, 11-15% each).

### Where it did not fit these designs

1. **"Did I image enough?" answers a question the experiments did not ask.** All three get ❌
   because no group reaches a 5% precision of its own mean, while each experiment was built to show a
   difference (d 0.4-2.4) and does so consistently across repeats. With a control declared, the
   headline should say whether the observed difference is resolved (its sign and size stable under
   field subsampling, or a precision of the difference), and keep the 5% precision of each mean as a
   secondary target.
2. **No answer across biological repeats.** Question 9 lists per-repeat d values and says no test is
   possible, but not what the design does support: the biological repeat is the replicate. Per
   condition it should give the direction across repeats (lower in 3/3), the repeat-level mean d,
   and a test on repeat means when there are at least 3 repeats (an exact sign-flip test on 3
   repeats cannot go below p = 0.25, and should say so).
3. **One observation per field (two of three datasets).** The Foundation, clustering of unusual
   cells, novelty ("3 similar cells in a field") and the d-cells / d-fields columns are
   structurally inapplicable, yet they show as ⚠️ ("spread like random cells", "not estimated in all
   54"). They should read ➖ "not applicable: one cell per field", once.
4. **Track data refused by the per-object JCB rule** ("assumes observation_unit in ('cell',
   'nucleus', 'object'), got 'track'"). A track of one cell is an object for this rule; the unit
   list should not decide it.
5. ~~No experiment level.~~ Owner's decision (2026-09-30): no experiment level; a dataset the user
   provides is one experiment, pooled as provided. Each condition is still compared with the control
   of its own biological repeat.
6. **Integrity blind spots for small fields.** `min_objects=5` misses copies in fields of 2-4
   tracks. With one observation per field a copied cell row is never found at any setting (planted
   test: a cell's 10 continuous measurements pasted into 3 other fields, not detected). A row-level
   exact-duplicate check on continuous features, independent of field size, would close both.
7. **Features beyond the readout are not judged.** The AR effect (the CDM paper's strongest) appears
   nowhere in the short report: the ranking is gated on a test that the design cannot give. When
   untested, a descriptive ranking by effect size and consistency across repeats would still help.
8. **No check of acquisition covariates** (Track Duration differs by condition within CCT7_2).

### Display

- The short reports are long (178-234 lines); the screen's contrast table has 36 rows. One row per
  condition across repeats in the short form, per-repeat rows in the full form.
- Multiplicity: "the JCB count rule missed in 4 of 54" at a nominal 95% level is about what chance
  gives (about 3); say so.
- Integrity locations in the overview and question 6 use `condition=..., biological_repeat=...` and
  a run-on list; show pairs with the group label ("B5_04 and B5_04-1, 5 tracks").
- The headline |d| on field means (up to 3.7 from 5-9 fields, 1.6 on tracks) is inflated and
  unflagged.
- Wording: "all 1 groups", "0 of the 1 groups", "technical repeats" when each is `single`.
- `default_novelty` warns with 10 and 19 features; the 8 "new state" track fields were not checked
  against the images.

## 2026-09-30 — second run, after items 2, 3 and 6 of the first run

**Run.** The same settings at commit b0ecf69, after the owner's decisions: answer question 9 across
biological repeats; do not flag what the design cannot test (3 repeats and one well per condition
are the usual design); leave out the field questions with one cell per field; find copies in small
fields. The glioma card now declares only the metrics that do not grow with track length (the
owner: a difference in tracking duration is flagged, not a deal breaker, once normalised metrics
are used).

| | filopodia screen | CDM shape | glioma tracks |
|---|---|---|---|
| Data | ✅ | ❌ 1 finding: a cell measured twice | ❌ 6 findings: all 6 duplicated pairs |
| Foundation | ➖ not applicable (1 cell per field) | ➖ not applicable | ⚠️ as before |
| 2. Enough? | ❌ no rule met (54 groups) | ❌ (9 groups) | ❌ (18 groups) |
| 3. Limits | ➖ not applicable | ➖ not applicable | ⚠️ as before |
| 9. Contrast | ✅ 12 of 12 in the same direction in every repeat | ✅ 2 of 2 | ✅ 2 of 4 (both CCT8 siRNAs) |

- **Integrity** now finds every copy the curation found (the 28 duplicated tracks) and one the
  curation missed: Fiji rows 7 and 8 of siCCT8#5, repeat 1, in the published CDM spreadsheet are
  the same cell measured twice (recorded in that card). The clean filopodia screen stays clean.
- **Question 9** gives what the three designs support: the direction against each set's own
  control in every repeat (the screen's hypothesis 4 exactly: MAP4K4 and NF2 higher, the other four
  genes lower, 3 of 3). The paired t-test on 3 repeat means is below 0.05 for 8 of the 12 siRNAs
  before correction; after Bonferroni over 12, the smallest is 0.058. The paper's significance comes
  from cells as the unit, with the six siCTRL sets pooled; the repeat-level answer is the
  consistent direction.
- **Still to decide:** question 2 is ❌ for all three (a 5% precision of each group's mean; item 1
  of the first run), and the glioma tracks' Foundation and question 3 keep their ⚠️.

## 2026-09-30 — third run: question 2 by the contrast

Question 2 now asks whether the fields acquired resolve the difference from the control in each
repeat (owner's decision). No dataset is ❌ any more:

- **Filopodia screen**: resolved in 28 of 36 siRNA x repeat contrasts; every CCT8, CSNK1A1 and
  SCRIB contrast is resolved in every repeat. 6 would need about 14-98 cells a side (the weaker
  MINK1, MAP4K4#5 and NF2 contrasts), 2 are too small to settle with more cells of these coverslips.
- **CDM shape**: resolved in 2 of 6 (both siRNAs in repeat 3); the area differences of repeats 1 and
  2 (d -0.38 to -0.48) would need about 42-56 cells a side, against 16-27 imaged: concrete advice
  for the next experiment. One (d -0.09) is too small to settle.
- **Glioma tracks**: resolved in 8 of 12 (every CCT8 contrast); 2 CCT7 contrasts would need about
  9-11 positions a side, 2 are too small to settle.
- The 5% precision of each mean is kept as a secondary line: met in 0 of 54, 0 of 9 and 0 of 18
  groups.

## 2026-09-30 — fourth run: planning the next experiment

Question 9 ends with a plan: the biological repeats a paired test across repeats needs (80% power,
Bonferroni-corrected 0.05), with the fields a side acquired and with twice as many.

- **Filopodia screen** (Bonferroni over 12): the strong hits (CCT8, CSNK1A1, SCRIB) need 5-6
  repeats; MINK1 8-10; NF2 11-16. Twice the cells change little for most: the variation between
  repeats dominates; where the repeats agree (siMINK1#3, SD between repeats 0) twice the cells save
  3 repeats.
- **CDM shape** (over 2): 9-11 repeats with the cells imaged, 6-9 with twice as many.
- **Glioma tracks** (over 4): CCT8 needs 6 repeats; the CCT7 effects are too small to plan for
  (more than 30).
- **Pilots**: cut to 2 repeats, every dataset still gets a plan; cut to 1, the report says the
  variation between repeats needs a second repeat and question 2 gives the fields within a repeat
  (tests/test_repeat_plan.py).

## 2026-09-30 — fifth run: the plan at the user's design

The owner's decisions: no experiment level (a dataset is one experiment, pooled as provided), and
a plan that says what to change within the design the user ran, per condition, not how many more
repeats (a plan that always asks for 5-6 repeats is not useful). The repeats a paired test needs
are in the full report only, per condition at 0.05.

- **Filopodia screen** (3 repeats): the strong hits need about 4-10 cells a side per repeat to
  resolve the difference (26-49 were imaged); MAP4K4 12-21; MINK1#10 about 62 and NF2#7 about 98;
  MINK1#3 and NF2#17 are too small to settle in some repeat. Most hits are limited by the repeats
  differing, not by the cells.
- **CDM shape**: siCCT8#8 needs about 56 cells a side per repeat (sampling-limited: more cells help);
  siCCT8#5 is too small to settle in repeat 2.
- **Glioma tracks**: CCT8 needs about 4 positions a side per repeat; CCT7 is too small to settle.

## 2026-09-30 — sixth run: independent reagents for one target

With the new optional `target` column (the gene each siRNA silences), question 9 says whether the
reagents of one target agree. Filopodia screen: for all six genes both siRNAs are on the same side
of their set's control in 3 of 3 repeats (MAP4K4 and NF2 higher, the other four lower). CDM shape:
both CCT8 siRNAs lower in 3 of 3. Glioma tracks: both CCT8 siRNAs slower in 3 of 3; the CCT7 siRNAs
are on one side in 2 of 3 repeats, lower in one and higher in the other.

## 2026-09-30 — seventh run: the cell-level test, balanced when the repeats are not

Question 9 now also gives the lab's Plot&Stats test (Cohen's d on every cell against every control
cell, randomization over cells), labelled with the chance it gives p ≤ 0.05 without an effect. A
balance check comes first. The data count as unbalanced when one repeat holds more than 25% more
observations than another, or when the control is pooled from repeats the condition was not
imaged in. Unbalanced data get the test twice: as provided, and balanced (own-repeat control, the
same count from each repeat, median over 100 draws). A repeat under a quarter of the largest (since the eighth run, the median repeat) is
named instead of balanced to (since the eighth run, it is left out of the balanced test and the
other repeats are balanced). All three datasets are unbalanced, and none has such a small repeat.

- **Filopodia screen**: each set's control is pooled from 18 repeats against the condition's own 3.
  Balancing turns siMAP4K4#5 from d -0.10 (Bonferroni p 1.0) to +0.80 (0.012), in line with its
  3 of 3 repeats above the control. 9 of 12 conditions are at Bonferroni p ≤ 0.05 as provided,
  and 10 of 12 balanced. The chance of a false positive runs about 5-40%, set by each condition's
  variation between repeats.
- **CDM shape**: one repeat holds up to 2.2 times the cells of another. The d barely moves
  (-0.44 to -0.43, -0.57 to -0.55), but siCCT8#5 loses significance (Bonferroni p 0.028 to 0.12).
- **Glioma tracks**: CCT7 and CCT8 were imaged in separate repeats, so the pooled control mixes
  both. As provided, both CCT7 siRNAs differ (d 0.68 and 0.37). Balanced against their own
  repeats' control, they do not (0.22 and -0.10), which matches their 1-2 of 3 repeats in each
  direction. CCT8 stays (d about -1.0). Here the false-positive chance is about 35%.
- Most of the change comes from using each condition's own-repeat control. Equalising cells per
  repeat changes little in these datasets, since no repeat holds more than about half the cells.

## 2026-10-01 — eighth run: raw and normalised values

Question 9 now gives every test on raw values and on values normalised to the mean of the
control of the same repeat (a fold change): Cohen's d, the cell-level randomization test (as
provided and, when unbalanced, balanced), and the paired t on the repeat averages (on log ratios
when normalised). Their assumptions and limitations are in `docs/condition_tests.md`. The
cell-level test's generators now depend on the values tested, so its raw p-values differ slightly
from the seventh run (CDM siCCT8#5: Bonferroni p 0.040 here, 0.028 there).

- **Filopodia screen**: normalising pairs each set's control with its own repeats, as balancing
  did. siMAP4K4#5 goes from d -0.10 raw to +0.95 normalised, and siNF2#7 and #17 from 0.19-0.23
  to 0.46-0.51. All 12 conditions are at Bonferroni p ≤ 0.05 normalised, against 9 raw. Balanced,
  raw and normalised agree (d within 0.1). The paired t on repeat averages finds none of the 12,
  raw or normalised (smallest Bonferroni p 0.06).
- **CDM shape**: normalising changes little (d -0.44 and -0.57 raw, -0.44 and -0.51 normalised).
  The repeats share no large scale here.
- **Glioma tracks**: normalising removes the pooled-control artefact that balancing removed.
  siCCT7#3 goes from d 0.68 (p 0.002) to 0.25 (p 0.21), and siCCT7#6 from 0.37 to -0.11. CCT8
  stays at about -1.05 (p 0.002). The paired t: Bonferroni p 0.20 for both CCT8 siRNAs normalised
  (0.26-0.30 raw).
- **Reading**: in these datasets, normalising does most of what balancing does, because the
  imbalance is mostly a control from other repeats. Where both run, they agree.
- **After the unbalanced-repeats simulation** (`docs/preliminary_findings.md`, entry 12): the
  false-positive label now weights each repeat by its share of the cells. On these datasets it
  moves by 1-4 points (glioma CCT7 34% to 38%), since no repeat holds more than about half the
  cells. The balanced table now carries its own label at the cells per repeat it keeps. No repeat
  here is small enough to leave out under the median-repeat rule.

## 2026-10-01 — ninth run: the report read as a user would

An audit of the short reports found the right numbers but no conclusion: the reader had to
combine five to eight tables per condition. The changes are wording and layout; no statistic
changed.

- **Question 9 opens with a reading per condition**: confirmed across repeats, same direction in
  every repeat but not confirmed, no consistent direction, or one repeat; then the direction and
  d, the paired test (normalised), the cell-level test in its least biased version with its
  false-positive chance here, and whether the target's other reagent agrees. CDM shape: both CCT8
  siRNAs lower in 3 of 3 (d -0.57, -0.59), not confirmed by the paired test (p 0.42, 0.22).
  Glioma tracks: CCT8 the same (d about -1.1, p 0.2); CCT7 no consistent direction, and its raw
  pooled cell-level p 0.002 is named as coming from the control of the CCT8 repeats. Filopodia
  screen: all 12 in the same direction in every repeat, none confirmed with 3 repeats.
- **At a glance** opens with a key (a marker says how far to trust an answer, not what it found),
  says that questions 4, 5, 7 and 8 come from optimise, and no longer says "0 of 10 features pass
  the gate" when none could be tested.
- **No caution for a usual design**: three repeats, options left at their defaults (fingerprint,
  rare phenotype, order, stage) and a missing representation control are notes, not ⚠️.
- **Said once**: the duplicated fields are named in the overview only; the short notes of question
  9 are a sentence per test; question 2's novelty projections are in the full report.
- The short reports are 192 (CDM, from 215) and 288 (glioma, from 313) lines; the screen grows to
  254 (from 231) with its 12 readings.

## 2026-10-01 — tenth run: the design scan before any analysis

`workflows.scan_design` (`results/<id>/scan.txt`) reads the tables before any declaration beyond
the card's features, control and unit. Nothing in it is estimated or tested.

- **What the dataset is**: the screen and the glioma tracks are read as separate acquisitions
  sharing the control (6 and 2 sets of 3 biological repeats, each with its two siRNAs); the CDM
  shape as one experiment with every condition in every repeat. The targets in the card's
  `target` column are listed.
- **Copies**: the glioma tracks give the 6 duplicated pairs of the curation, the CDM shape the
  cell measured twice; the screen gives none.
- **Balance**: unbalanced repeats (more than 25%) in 3 of 5 (glioma), 3 of 3 (CDM) and 5 of 13
  (screen) conditions, stated without a caution (the report tests both ways); no sparse repeat.
- **Columns that would say more**, tried by dropping columns: without `position_id` and
  `image_file` the CDM shape is read as one image per row and the scan says so; without `target`
  the screen's names give all 6 reagent pairs the card records; without `technical_repeat` the
  scan suggests one if a condition was imaged in more than one well.
