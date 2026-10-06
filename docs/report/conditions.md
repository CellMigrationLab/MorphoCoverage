# Do the conditions differ? The tests of question 9, their assumptions and limitations

Question 9 of the report compares each condition with the declared control. It
gives three things, each on raw and on normalised values:

1. **Cohen's d** on the observations (cells, tracks), per repeat and pooled;
2. **the randomization test** on Cohen's d over the observations, as the lab's
   Plot&Stats notebook runs it (the cell-level test);
3. **the paired t-test on the repeat averages**, the biological repeat as the
   replicate (as in SuperPlots).

The code is `morphocoverage.sampling_unit`: `contrast_repeats` (the
across-repeat table and the paired t), `contrast_cells` (the cell-level test,
the balance check and the normalisation), and their words in `repeats_text`
and `cells_text`. The evidence for every number below is the simulation
`data/simulations/test_choice.py` (results in
`data/simulations/results/test_choice.csv`, summary in
`docs/preliminary_findings.md`, entry 11). The runs on the lab datasets are in
`data/lab/FINDINGS.md`.

## Normalised values

A value is normalised by dividing it by the mean of the control **of its own
repeat**. The control's values are divided by their own repeat's mean too. This
is "normalised to the control", a fold change. It removes a change of scale
that both wells of a repeat share (laser power, camera gain, a staining batch),
and it pairs every condition with its own repeat's control even when the
control is pooled from more repeats. Within one repeat it leaves Cohen's d
unchanged; pooled over repeats it does not. The paired t on normalised values
is run on the log of each repeat's ratio of the condition's mean to the
control's.

## Several controls

One control is declared as `control={"condition": "siCTRL"}`, and every condition is
compared with it within each biological repeat, as always. A dataset of several experiments,
each with its own control, declares the controls, each mapping a treatment key to its
condition (a plain list of conditions is accepted too, with no keys), and optionally the
parents of the controls (any condition, with an optional treatment key) and a master control
(owner, 2026-10-05):

```python
setup = Setup(
    readout="Order",
    control={
        "condition": {
            "VEC_SiNC": {"SiCol6a1": "VEC_SiCol6a1", "SiFBN2": "VEC_SiFBN2"},
            "THBS5_SiNC": {"SiCol6a1": "THBS5_SiCol6a1", "SiFBN2": "THBS5_SiFBN2"},
            "VEC": {"THBS5": "THBS5"},
        }
    },
    control_parents={"VEC_SiNC": ("VEC", "transfection"), "THBS5_SiNC": ("THBS5", "transfection")},
    master_control="VEC_SiNC",
)
```

A condition belongs to one control at most, a treatment key names one condition of a
control, a parent may be any condition (a member of another control included) but the links
may not form a cycle, and every condition named must be in the data; each is refused with a
message otherwise. One control declared this way, as a list, a keyed dict or with itself as
the master, and without parents, is `{"condition": control}`: the output is identical, every
other condition compared with it (owner, 2026-10-05). The declaration gives the **contrast table** (`Setup.contrast_table`), one
row per test, shown in the report and saved as `contrasts.csv`:

- `own_control`, **the main verdicts**: each condition against its own control, and each
  control against its parent, raw and normalised, paired per biological repeat, exactly as
  with one control (the paired test across repeats, the cell-level screen, the per-repeat d,
  "Did I image enough?", the feature discovery and the heatmap use these rows only),
  Benjamini-Hochberg within each sub-experiment as before. A row reads
  `THBS5_SiCol6a1 vs THBS5_SiNC`.
- `across_normalised`: two links with the same treatment key under different controls (or
  parents): the two conditions each normalised to its own control, in every repeat holding all
  four the difference of their log ratios, paired across those repeats
  (`SiCol6a1 effect: THBS5_SiCol6a1/THBS5_SiNC vs VEC_SiCol6a1/VEC_SiNC`); a list of
  conditions gives no such row. One Benjamini-Hochberg family over every feature's rows.
- `vs_master_raw`, only with a master: each other control against it on the raw values, in
  the repeats holding both, unless a parent link joins them already
  (`THBS5_SiNC vs VEC_SiNC (raw)`). The report gives the master's repeats once and says that
  raw comparisons across experiments assume the experiments were imaged and processed alike.
  One Benjamini-Hochberg family over every feature's rows.

For the ECM design above that is 7 + 3 + 1 = 11 rows per feature. The secondary rows are
tested on the repeat means, with no cell-level test, for every explainable feature (owner,
2026-10-05): each family is corrected over every feature and its rows, as the main verdicts
are corrected over the features and conditions of their sub-experiment, and the discovery
and the heatmap still read the main verdicts only. "Compared across experiments" gives the
readout's rows first, each with the biological repeats it used and, for an
`across_normalised` row, which of the two ratios is the larger in how many repeats; then the
other features' rows at q ≤ 0.05 and a count of them all, and "At a glance" adds
"across experiments: 1 of 8 significant" when any is. Every row is saved, with its kind,
label, feature, repeats, change, p and q, as `tests_across_experiments.csv`, with a CSV
button under the block on the page. A row held in fewer than 3 repeats is named once as not
compared, and for a feature with a value that is not positive (a signed feature) the raw
differences are compared instead of log ratios, the scale of its main test, and the table
says so for that feature.
`Setup(contrasts=table)` takes a table of the same columns (`kind`, `condition`,
`reference`, `condition_control`, `reference_control`, `family`, `label`) in place of the
generated one. A condition has one `own_control` row: a table giving it two is refused, and
with declared controls its own control comes from the declaration, so a table giving it
another one is refused too (owner, 2026-10-05). The "Controls:" line and every analysis then
read the same own controls.

Only the statistics read the controls: the figures, the PCA and UMAP, the population
detection and the SuperPlots show every condition under its own name. With `normalize`, the
normalised views show each condition divided by its own control's mean in its repeat, a
control by its parent's (else its own, so 1); a condition held in a repeat without its own
control is left out there, and the report names it once. Levels across experiments are read
on the raw values. The tests read the raw values and each normalises to its own control;
with one control too, so their raw columns hold raw values (owner, 2026-10-05; before, with
`normalize`, the contrast read the normalised values and its "raw" d and paired t were the
normalised ones). Unusual observations and the population enrichment take each condition's
own control as the reference ("each against its own control"); the object rules, clustering
and a positive control take every control pooled. A contrast table alone gives own controls
too, so the declaration says unusual observations are assessed against them.

**Controls apart in intensity** (owner, 2026-10-05). Raw comparisons across experiments, and
maps on raw values, assume the experiments were imaged and processed alike. So the controls
are compared with each other on every declared column whose name contains "intensity" (any
case) and on those listed in `Setup(intensity_features=...)`: when two controls' medians, over
all their observations, differ 2-fold or more (`_acquisition_gap.GAP_FOLD`), the report says
so with one ⚠️ caution among the checks of the values ("Controls apart in intensity: VEC_SiNC
and VEC differ 13-fold in median Intensity (mean); ... 2-fold is an engineering threshold for
quality control, not a test.") and in "Do the conditions differ?",
and names each `across_normalised` or `vs_master_raw` row that joins two such controls under
its table, and each main verdict that does, a control against its parent, under the
readings (owner, 2026-10-05). The 2-fold mark is an engineering threshold for quality
control, not a statistical test: it has no error rate, and a smaller gap is not evidence
that the experiments were imaged alike (plan v2, owner, 2026-10-06). With one control nothing is compared.

A condition never compared, no repeat holding both it and its own control, is named once,
its comparison's own reference in the label: "Not compared: THBS5_SiFBN2 vs THBS5_SiNC (no
biological repeat holds both)." The readings name every control, never one hidden behind
"and 1 more" (owner, 2026-10-05).

### The tests of the secondary rows

The main verdicts (`own_control`) are the three answers below, per row, labelled
`condition vs control` and corrected within each sub-experiment as with one control. The
secondary rows:

- **`across_normalised`**: the same treatment in two experiments, each condition normalised to
  its own control: the paired t-test on each repeat's
  `log(mean(a) / mean(control of a)) - log(mean(b) / mean(control of b))`, in the repeats
  holding all four (raw differences of differences where a mean is not positive). Shifts the
  four conditions of a repeat share cancel, so without a difference of effects it gives
  p <= 0.05 in about 5% of experiments (`tests/test_comparisons.py` simulates it).
- **`vs_master_raw`**: a control against the master control, not normalised, the paired
  t-test on each repeat's log ratio of the means, on the feature's scale as every test (the
  difference of the means for a feature read on differences), in the repeats holding both,
  so the fold change printed beside it is what it tests (audit 10, owner, 2026-10-06; before,
  the difference was tested beside a fold change). In a report with other tables, these
  rows are tested feature by feature only, each kind with its reason: a difference of log
  ratios has no meaning in an embedding; the whole-table test compares each condition with
  its own control, not the controls with the master.

The sign-flip test is given where it can reach 0.05 once corrected (`_reach`). The code is
`sampling_unit.contrast_across`, its words `across_text`, and the table `_control`.

## The three answers

| | What it asks | What it treats as the replicate | Responds with |
| --- | --- | --- | --- |
| Cohen's d, per repeat | How far apart are the two wells, in observation SDs? | – (an effect size, not a test) | 1 repeat |
| Cell-level randomization test | Do these wells differ? | the observation | 1 repeat |
| Paired t on repeat averages | Does the effect repeat? | the biological repeat | 3 repeats |

**Balance check.** Before the cell-level test the report checks whether the
repeats weigh the same. The data count as unbalanced when one repeat holds more
than 25% more observations than another, on the condition's side or the
control's. They also count as unbalanced when the control is pooled from
repeats the condition was not imaged in. Unbalanced data get the cell-level
test twice. The first is as provided. The second is balanced: each condition
against the control of its own repeats, as one permutation test (owner,
2026-10-05; before, the median d and median p over 100 draws). Its d is the
median Cohen's d over 100 draws, each with the same number of observations
drawn from every repeat; across the draws every observation is used, each
repeat weighing the same in each. Its p shuffles the condition and control
labels within each repeat, 999 times, and computes the same median d, on the
same draws, for each shuffle: `p = (extreme + 1) / (B + 1)`. Without an
effect it gives p ≤ 0.05 in 5% of experiments, within Monte Carlo error, raw
and normalised (`tests/test_contrast_cells.py`). Normalised, each shuffle divides a repeat by
the mean of its control as shuffled. This p is corrected like the others.
When the control is pooled from other repeats, the test as provided is not
reported for that condition, and when no balanced test can be formed either
(every repeat of the condition too small), the condition gets no cell-level number.
A test not reported leaves the Benjamini-Hochberg family of its sub-experiment:
the q-values are corrected over the tests reported only (owner, 2026-10-05).

**Permutations.** The randomization p is `(extreme + 1) / (B + 1)` over `B`
shuffles: 2000 for the readout's test as provided, 9999 in the feature screen,
999 for the balanced test. It cannot fall below `1 / (B + 1)`, so with `K`
tests in a correction family Benjamini-Hochberg cannot give a q below
`K / (B + 1)`: 210 tests at the balanced test's 999 shuffles gave q 0.07 for
three clear differences, never 0.05. Every test at p ≤ 0.05 after the first
pass is therefore rerun with `B = max(default, ceil(20 K))` shuffles, `K` the
tests of its sub-experiment, so the smallest q, `K / (B + 1)`, is below 0.05
(owner, 2026-10-02: "More permutations"). Only those: a test above 0.05 cannot
pass at any rank, and a smaller p moving never changes the q of a larger one.
The rerun is seeded from the test's own identity, so the result does not
depend on the order of the tests. The balanced test is rerun with `B` new
shuffles, its observed median d kept. The report states the shuffles rerun beside the screen's
family. The pairwise heatmaps (1999 shuffles) correct over the pairs of one
figure and are not rerun: their smallest q, pairs / 2000, is below 0.05 up to
100 pairs (14 conditions); a figure of more conditions is floored.

**Very small repeats.** Balancing down to a repeat that holds very few
observations would keep only that many of every other repeat in each draw. A
repeat holding under a quarter of the median repeat on its side, or fewer than
5 observations, is therefore named in the report, as worth checking first. It is
kept in the test as provided and left out of the balanced one, which balances
the other repeats. A small control repeat is left out of both sides, so each
condition stays paired with its own repeats' control. When every repeat is that
small, or all but one, nothing is balanced and only the test as provided is given: one
repeat alone is not balanced over the repeats, and the reading reads the test as provided,
never a "balanced" test of one repeat (audit 10, owner, 2026-10-06).

## Assumptions

- **Cohen's d** compares means in units of the pooled observation SD. It says
  nothing about a difference confined to a subpopulation or to the shape of the
  distribution (see the unusual objects and the rare label).
- **The cell-level test** assumes the observations are independent draws. Its p
  is therefore about these wells, not about the treatment: a well's own offset
  (its plating, its focus, the day) is shared by every observation in it.
- **The paired t** assumes the per-repeat differences (raw) or log ratios
  (normalised) are normally distributed and the repeats are independent. It
  pairs each condition with the control of the same repeat.
- **A shared control** enters every comparison with it, so a control mean off by chance in
  a repeat moves them all: the false-discovery rate holds, but the conditions' verdicts are
  not independent. When two or more conditions are significant the same way against a
  control imaged in fewer than 3 fields in a repeat, the section says so in one line
  ("Shared small control: on area, siA#1 and siA#2 against siCTRL, 1 field in a repeat";
  owner, 2026-10-06). Three fields is where the dependence levels off (audit 8, two
  conditions of 3 fields of 20 cells under no effect: their changes correlate at 0.82 with a
  control of one field of 4 cells, 0.66 with one field of 20, 0.56 with two, 0.37-0.39 with
  three or five).
- **Normalising** assumes a positive feature on a ratio scale. It is not
  computed when a control mean is 0 or below (a signed feature, a difference
  already).
- **Balancing** assumes the observations of a repeat are exchangeable, so any
  subset of them represents it.
- **Benjamini-Hochberg** correction is applied within each sub-experiment
  (the conditions that share their biological repeats and their control, as
  when a screen runs target by target), each test separately: one part of a
  screen does not dilute another (owner, 2026-10-01). The tests within one
  repeat keep their Bonferroni correction.

## Limitations

- **The cell-level test is not a 5% test once repeats differ.** When the
  treated-vs-control difference varies between repeats by *s* observation SDs,
  it gives p ≤ 0.05 without an effect in about
  `2·P(Z > 1.96/√(1 + n·s²/2))` of experiments, with *n* observations per well.
  The simulation matches this formula within a few points; the report shows it
  for every condition in the column "p ≤ 0.05 without an effect". At 30 cells a
  well this is 16% for *s* = 0.25 and 37% for *s* = 0.5. With 100 cells a well
  it is 34% and 59%: more observations make it worse, not better.
- **A stricter cutoff does not fix it.** At *s* = 0.5 and 30 cells a well,
  p ≤ 0.01 still gives 22% false positives and p ≤ 0.001 gives 12%. The cutoff
  that would give 5% depends on *s*, which one repeat cannot measure.
- **No statistic computed from one repeat can fix it.** With one well a side,
  the difference measured is the effect plus the wells' offsets, and the
  observations hold no information about those offsets. Only more wells or
  more repeats, or outside knowledge of *s*, can separate them.
- **Normalising does not fix it either.** It removes what both wells of a repeat
  share, not the part of the effect that differs between repeats. On raw
  values the cell-level test is at or below the formula, because the shared
  shifts add noise; normalised, it matches the formula, and it has more power.
- **The paired t has little power with few repeats.** With 3 repeats and 30
  cells a well, it finds d 1.5 in 98%, 90% and 63% of experiments as repeats
  differ by 0, 0.25 and 0.5 SD; d 1.0 in 85%, 67% and 39%; d 0.5 in 41%, 27%
  and 18%. It cannot run with fewer than 3 repeats, and the report then
  describes each condition by its direction in every repeat and its d.
- **The balanced p is calibrated for these wells only.** It is the p of one
  permutation test, whose null is that the observations of the condition and
  the control are exchangeable within each repeat (owner, 2026-10-05).
  Balancing removes the extra weight of the repeats holding more observations,
  not the independence assumption: when the difference varies between repeats
  it gives p ≤ 0.05 without an effect as often as the formula says at the
  observations per repeat it keeps.
- **Leaving out a very small repeat changes what is balanced.** The balanced
  test then describes the other repeats; whether the small repeat agrees is
  in the across-repeat table, where it counts as one repeat like the others.
- **Alternatives were tried and not adopted.** A mixed model with a z test is
  inflated at 2-3 repeats, and with a t test it almost never rejects there. A
  hierarchical bootstrap is inflated with few repeats (22%, 12%, 5% at
  *s* = 0.5 over 2, 3, 4 repeats). The exact sign-flip test cannot reach 0.05
  below 6 repeats; the full report adds it from there.
- **Unbalanced repeats make the test as provided worse.** With one repeat
  holding 2, 4 or 8 times the cells of the others, its false-positive rate at
  *s* = 0.25 rises from 16% to 22%, 31% and 46% (normalised;
  `data/simulations/unbalanced.py`). The label follows it within 1-2 points,
  with the cells per well weighted by each repeat's share
  (`sum(n²) / sum(n)`). The balanced permutation test stays at 15-19%, about
  the 16% its label gives at the 30 cells a repeat it keeps (normalised, 500
  experiments a setting with that script's generator, 2026-10-05; the median p
  over draws it replaced gave 12-14%).
- **A control pooled from other repeats is the worst case on raw values.** The
  raw test as provided gives 46% false positives even when the repeats agree,
  because batch shifts between repeats no longer cancel. Normalising cuts it
  to 9%, and the balanced test, against the control of the condition's own
  repeats, to about 5%, its nominal rate (the median p over draws gave 2%).
  Read the balanced numbers whenever the balance line says the control comes
  from other repeats.
- **The simulations are one design.** They have one well per condition and
  repeat and normal cell noise.

## Within each repeat

When `control` names a condition and the FOV table holds other conditions,
the report also answers question 9, "Do the conditions differ, and how
confidently?": each condition against the control, per repeat, with Cohen's
d and the JCS effect size computed on objects (the numbers of the lab's
Plot&Stats notebook) and with the field as the unit, and a randomization test
that shuffles condition labels over whole units. The unit is the FOV table's `technical_repeat` column, whatever
`group_cols` says; without it, a group column beyond condition and repeat
(a `well` column in `group_cols`) when some condition of a repeat holds more
than one of its values. Over wells the p-value is coarser: with 2 wells a
side it cannot go below 1/3. A test that the design cannot make informative is
not run and not flagged: the randomization test runs only when its smallest p can
reach 0.05 after the Bonferroni correction over the contrasts, 4 units a side for
one contrast (`sampling_unit._reach`), so one well per condition and repeat, the
usual design, gets none. A side with a single unit gets no test, since
its fields cannot stand in for wells, and its effect sizes are still
reported. **Without a unit there is no p-value**, unless
`Setup(fields_independent=True)` declares the fields independent draws;
whole fields are then shuffled and the section says the p-value treats them
so. A d on technical-repeat means is computed only when each side holds 4
or more, since from fewer means a d can be very large by chance. The
whole-field draws of the JCS effect size take whole technical repeats with
replacement when both sides have them; otherwise they resample these fields
only, and their "same sign in X% of draws" is not a confidence in the effect.
Every test compares means, and the section says that a difference confined
to a subpopulation is not tested. It describes what these fields support;
generalizing needs independent repeats.

## Many features

With several declared features, `feature_discovery` asks which of them separate
the conditions (owner decision, 2026-10-01: the cell-level test for discovery,
the paired test kept as the second gate):

- **The discovery gate**: the cell-level test of every feature and condition, in
  the version the readings trust most (normalised, balanced when unbalanced),
  Benjamini-Hochberg over all of them within each sub-experiment, a feature passing at q ≤ 0.05; passing
  features are ranked by |d| and the plots draw the best-ranked.
- **The second gate**: the paired t on the repeat means (the scale declared in
  `Setup(scales=...)`, plan v2, owner, 2026-10-06; undeclared, log ratios when every
  observation of the feature is positive, else differences: the scale is chosen once per
  feature from all its observations, the same for every condition, and the test's
  description says so; owner, 2026-10-06, after audit 8 found that choosing it per
  condition from its means let an offset of the feature change the power, from 36% to
  50% for the same effect), Benjamini-Hochberg over every feature and condition of a
  sub-experiment with 3 or more repeats. A passing feature it also passes is *significant across
  repeats*.

The discovery gate inherits the cell-level test's assumption: with variation
between repeats it passes more features than 5% without an effect, so its list
is candidates; the paired gate holds 5% and, with 3 repeats, confirms only
large, consistent effects. The PCA and UMAP (`embedding_plots`) describe how the
conditions sit over all features at once and test nothing; on raw values a
separation can be a batch between repeats.

The table gives each feature's best contrast, both q-values and its direction in each
repeat; a passing feature that goes up in some repeats and down in others is flagged.

When there are wells to shuffle, the section also ranks the features within
each repeat (`sampling_unit.feature_ranking`), as fiberfeatures scores a
feature set: the same test for every feature and contrast, Benjamini-Hochberg
corrected together, a gate at q <= 0.05, and the passing features ordered by
the largest |d| on field means. The ranking says which features are worth
plotting. With few technical repeats a side the p-values have a floor, and
after the correction a lone contrast at the floor cannot pass: the section
gives the smallest q the tests can reach and how many must reach the floor
for one to pass.

## Beyond the mean, and a labelled share

Beyond the mean is exploratory (plan v2, owner, 2026-10-06): not yet validated by the
benchmark of [the validation protocol](../common/validation.md); the report and the page say
so once where it answers.

The report compares a rare label's share between the
conditions like a feature (`"<label> share"`: the ranking, the heatmap, the
figures, which plot one point per field, its share of labelled cells, while the
tests keep the cells). The share stays in every family the features are corrected in (the
main rows and the rows across experiments), and every count names it apart from them: "0
of 4 features and the mitotic share are candidates", "over its 4 features, the mitotic share
and their conditions" (owner, 2026-10-06). With a fingerprint, the report also compares
beyond the mean the distribution of each explainable feature the fingerprint reads (none
for a fingerprint declared on a DINO table alone, which the section says once): per
biological repeat against its own control, the spread (the SD ratio) and the 10%, 50% and
90% quantiles (their shift beyond the shift of the mean, in control SDs), each on the
feature's scale, the log of its values when its tests read log ratios, and each tested
across the repeats (Benjamini-Hochberg within each sub-experiment, a family of its own). A
condition that only moves its mean, or changes every value by one factor, shifts no
quantile beyond it (audit 10, owner, 2026-10-06; before, the quantile shifts held the mean's
and a pure shift read "differs beyond the mean"). A condition whose distribution differs
answers the question like a feature significant across repeats; nothing significant reads
"not significant", never "no difference"; the same direction in every repeat is a trend
only from 6 repeats, and with 2 repeats "lower (higher) in both repeats", not tested (audit
10, owner, 2026-10-06).

Both sides are read on **field-balanced subsamples** (owner, 2026-10-06): per repeat, the
same number of fields a side, the smaller side's, drawn at random among each side's fields,
and the same number of cells in each field, the smallest field's of either side (at most
100), drawn at random; the statistics are averaged over 50 such draws, seeded as every
draw of the package (one draw when nothing is left out). On the cells pooled, the field
design read as an effect: a control of one field of 60 cells against three fields of 20
has no field-to-field variation in its spread. Null calibration
(`data/simulations/beyond_the_mean.py`, results in
`data/simulations/results/beyond_the_mean.csv`; 400 datasets per setting, 3 and 6 repeats,
5 features, field SD 0.5 or 1 cell SD, equal sides, a control of one field of 60 or of 6
cells, a condition that lost cells, and uneven random counts): each statistic gave
p ≤ 0.05 in 3.6-5.8% of tests, and any q ≤ 0.05 over the 20 tests in 2.8-7.0% of
datasets (measured before the quantile shifts were taken beyond the mean's, on the feature's
scale; not rerun since). The same tests on the pooled cells gave up to 56% for the spread and up to 13%
for the outer quantiles, and up to 31% of datasets with a q ≤ 0.05 (6 repeats, a control of
one field of 60 cells, field SD 1; audit 8 found 56% and 34% on its own design, now 4%
and 3%).

When a condition's cells may split into two groups on a feature where the control's do not (Sarle's
bimodality coefficient above 5/9 for the condition without a heavy tail, excess
kurtosis at most 2, since a skewed single-peaked feature reaches 5/9 too, and
below 5/9 for the control, in at least half of its repeats that can be assessed),
the summary says so and points to its figure: described, not tested, on every observation
pooled. A repeat
whose coefficient is undefined, with fewer than 4 cells on a side or no spread,
leaves the denominator and is named ("1 repeat not assessed: 3 cells"), and a
condition with no repeat assessed is said not assessed (owner, 2026-10-05).

## The whole-table test: other feature tables

A feature table that is not read one column at a time, such as a DINO embedding, is tested
whole (owner, 2026-10-06; `sampling_unit.whole_table`, words in `workflows._whole_table`):
for each condition against its own control, does its centre in the table move away from the
control's, the same way in every biological repeat? It runs for every other feature table
of every report, beside the per-feature tests, and is the only test of this question in a
report without explainable features ([embedding-only
reports](../common/setup.md#embedding-only-reports-no-explainable-features)). The comparisons
are the main rows of the contrast table (each condition against its own control, a control
against its parent); its rows across experiments are tested feature by feature only, as a
difference of log ratios has no meaning in an embedding, and the report says so once.

Each comparison scales the table's columns on its own two sides (owner, 2026-10-06): every
observation of the condition and of its control, in the repeats holding both, minus the
centre of its side in its repeat; each column is divided by the robust scale of these
within-group residuals, as the maps scale theirs (the MAD as an SD, the SD where the MAD is
under a quarter of it). Another condition's effect or a repeat's batch thus never changes
a comparison's scale: on all the observations, a condition shifted 4 SDs on one column cut
another comparison's power from 43% to 9% (audit 8); scaled on its own sides, it stays at 46%. A column that does not vary within
the groups is left out; a comparison whose sides hold one observation in every repeat has
no residual and is "not tested, one observation a side and repeat". The MFA weighting of
the combined space divides a whole table by one number, which changes none of the
statistics below.

- **Across repeats, the verdict.** The field is the sampling unit: each field's
  observations are averaged first (a field of one image is that image), and a side's
  centre in a repeat is the mean of its fields. Per repeat `r`, `Δ_r` = centre(condition) −
  centre(control); `u_−r`, the other repeats' mean shift, normalised; `s_r = Δ_r · u_−r`,
  positive when repeat `r` moves the way the others do ("shifted the same way in 4 of 5
  repeats" counts them). The test is one-sided, that the mean of the `s_r` is above 0, a
  paired t-test across repeats with its t **corrected for the repeats the projections
  share**: repeat `q` enters every `u_−r` but its own, so without an effect the `s_r` are
  positively correlated, by up to `1/(n − 1)` in a table of many effective dimensions, and
  the plain paired t rejected 8-20% of null datasets at 5% in the simulation below. The t is
  multiplied by `sqrt((n − 2) / (2 (n − 1)))`, its value at that largest correlation (`n −
  1` degrees of freedom). This is an asymptotic high-dimensional, low-sample-size (HDLSS)
  correlation correction, not an exact one: the correlation reaches `1/(n − 1)` only as the
  effective dimensions grow, so the test is close to calibrated in wide tables (7.5% at 3
  repeats in the simulation below) and conservative in narrow ones (plan v2, owner,
  2026-10-06). It needs 3
  repeats; with fewer, "not tested across repeats, 2 repeats". Benjamini-Hochberg: the tests
  of one table form one family per sub-experiment, as the per-feature main rows do, never
  joining the per-feature families.
- **Shuffle test, exploratory** (owner, 2026-10-06): the distance between the condition's
  and the control's centres over the repeats holding both, against the labels of the units
  permuted within each repeat (999 permutations, all at once), `p = (extreme + 1) / (B +
  1)`, uncorrected. The unit is the highest one below the repeat the data name: the
  technical repeat (`technical_repeat`), else the analysis groups beyond the condition and
  the repeat (a `well` column in `group_cols`), else the field (the CSV's `shuffled`).
  Fewer than 40 distinct assignments, one well a side say, give no p. It is in
  `tests_whole_table.csv` only, never in the verdict or the answer: it asks whether these
  units differ, not whether the shift repeats, and shuffling fields counts any well effect
  the data do not name.
- **The effect size**: the length of the mean shift over the repeats in within-group SDs,
  on the comparison's scale: the root of the summed variances of the observations around their own group's centre (a
  side in a repeat), the typical distance of an observation from its group's centre. For one
  column it is a Cohen's d.

The text reads, per table and comparison: "DINO (384 columns): - THBS5_SiCol6a1 vs
THBS5_SiNC: shifted the same way in 4 of 5 repeats, p 0.03 (q 0.06) across repeats; the
centres 0.41 within-group SDs apart." A shift
significant across repeats is named in the answer's line, so "At a glance" mentions it,
and turns the marker to ✅. When none is, a comparison shifted the same way in every repeat
is named as a trend from 6 repeats only (owner, 2026-10-06): with fewer, the projections are
all positive without an effect too often (25% of null datasets with 3 repeats, 13-15% with
4, 7-9% with 5, 3-5% with 6; audit 8). The page shows the block under "Do the conditions differ?" and
saves the tests as `tests_whole_table.csv`. Observations missing a value of the table are
left out, counted. The cost is a few matrix products per comparison: under a second for the
ECM tables, little beside the per-feature work.

**Calibration** (`data/simulations/whole_table.py`, results in
`data/simulations/results/whole_table.csv`): 400 datasets per setting, 3, 4 and 6 repeats,
one image per field (6 a side and repeat) or 20 cells in each of 5 fields, 50 and 384
correlated columns, with and without a batch (a shift of each repeat and of each well),
even and uneven counts (2 to 10 fields a side and repeat), each comparison scaled on its own
two sides. Without an effect, the repeat-level test rejected 1.8-7.5% at 5% (mean 4.1%
without a batch, 4.4% with one; the highest, 7.5%, with 3 repeats, within two standard
errors of 5%); the plain paired t on the same projections 8-20%. The shuffle test, on fields
since the simulated wells are not named, rejected 3.0-8.3% without a batch, and with one it
is **anti-conservative**, as the cell-level screen is: 17-29% with one image per field and
38-62% with 20 cells per field, since a well's own shift counts for it; hence wells shuffled
when named, and never a verdict. Power, a shift of 0.3 within-group SDs in every repeat: 60%
with 3 repeats, 81% with 4, 98% with 6 (36%, 59% and 87% with a batch); a shift of 0.6: 96%
with 3 repeats (86% with a batch) and at least 99% from 4 (scaled on every observation of
the table, before audit 8: 2.3-7.5% without an effect, and 58%, 79% and 97% power).
In narrow tables the test is conservative, as its correction is set for the largest
correlation of the projections, reached in wide tables (audit 8, 400 null datasets each,
cells in fields): one column 2.8% at 5% with 3 or 4 repeats and 1.8% with 6, two columns
4.5% with 3 repeats, against 4.5% with 200 columns; and with very uneven fields (100 cells a
field in one repeat, 3 in the others) 0.8%.

## Reading them together

With three or more repeats, the paired t answers whether the effect repeats,
at a real 5%. The cell-level test, labelled with its false-positive chance,
says whether these wells differ, and the per-repeat d says by how much. In a
pilot with one or two repeats, only the cell-level test responds. Its label
says how much a positive p can be trusted, and the plan says what the next
repeat should change.

The plan powers the test that decides: the paired t-test on each repeat's log ratio of
the condition's mean to the control's (the difference of the means where a mean is not
positive), with the spread of those values between the repeats and their sampling error
within one (a bootstrap of the fields as acquired, which also gives each repeat's
interval of d). The table gives the change it plans on (the geometric mean of the repeats' fold changes) beside
the mean d.

The plan is made at the design the user ran (a dataset is one experiment, pooled as
provided; a pilot of 1 or 2 repeats included): per condition, the fields a side each repeat
needs to resolve the difference (projected from its interval: fewer than acquired where it
is resolved), led by the repeats a paired test needs for 80% power at 0.05/m, m the
conditions of the sub-experiment (what Benjamini-Hochberg asks at most), with the fields per
repeat as now and twice as many (`sampling_unit.repeat_plan`). The count starts at 3, the
paired test's minimum, never 2 (audit 10, owner, 2026-10-06). No verdict on what limits it
is printed (owner, 2026-10-02): from 3 repeats it contradicted the readings, and telling the
variation between repeats from the sampling within them is P10's question, which needs 5 or
more units.

**Same direction in every repeat, not significant.** A condition whose paired t is not
significant but whose repeats all went the same way is named as a trend only from 6 repeats
(`TREND_REPEATS`; owner, 2026-10-06, audit 8). Without an effect each repeat's sign is a coin
toss, so the exact chance that all `n` repeats go the same way is `2^(1-n)`: 25% at 3
repeats, 12.5% at 4, 6.25% at 5 and 3.1% at 6, half of that when the direction is stated in
advance (plan v2, owner, 2026-10-06). 6 is the first number of repeats where this chance is
below 5%. With fewer the fact is still shown, the direction in each repeat, never as
evidence.

**Three repeats** (plan v2, owner, 2026-10-06). With exactly 3 biological repeats, the
fewest the paired t-test needs, each verdict gives the change in each repeat, the change
overall and its 95% t interval (2 degrees of freedom; for a fold change the interval of the
mean log ratio, back-transformed): in the readings ("change ×0.65, ×0.45 and ×0.61 in the 3
repeats, ×0.56 overall, 95% t interval ×0.35 to ×0.917"), in the text's table of ranked
features and in the page's table across repeats. The section says once: "With 3 biological
repeats, very limited biological replication: each verdict gives the change in each repeat
and the 95% t interval of the change, which is wide; a change that is not significant is not
evidence of no change." The paired t-test itself is unchanged. `tests_across_repeats.csv`
and `feature_discovery.csv` carry each repeat's change (`repeat_contrasts`, `R1: 0.6463;
...`) and the interval (`ci_low`, `ci_high`) at any number of repeats.

With a `target` column, the report also says whether the independent reagents of one
target (two siRNAs of one gene) are on the same side of the control in every repeat
(`contrast_repeats.reagent_agreement`; described, not tested).

## On the page

"Do the conditions differ?" opens with the features significant across repeats,
condition by condition, each with its direction, how many repeats agree, its q and
its d. With fewer than 3 biological repeats nothing can be called significant
(the paired test needs 3), and it says so: with 2 it gives the candidates of the
exploratory cell-level screen that go the same way in both repeats, with 1 every
candidate, said as exploratory. Then a heatmap of Cohen's d of every feature and
condition against the control (● passes the exploratory cell-level screen, ✓ also
significant across repeats; hover a cell for its adjusted p-values), with the features
no condition makes a candidate named under it, then the best-ranked features (the
readout only when it ranks, or alone when none does), each with its text **for a
figure legend** folded under it: n biological repeats, wells per condition, cells a well
(with several controls, each condition counted in the repeats holding its own reference,
owner 2026-10-05), what the points and bars show, the test and its correction family, and per condition
the change, the mean per-repeat d, the adjusted p and p, every number the one the
Across repeats tab gives for that feature and condition. Then the tests behind the
reading. These open with **the tests, in short**: the paired t-test on the biological
repeat means (it decides), the cell-level screen (exploratory: it treats every cell as
independent), and, when they are on the page, beyond the mean and the reagents; what
Cohen's d is in each table (per repeat or on the cells); and what p and the adjusted p
are, with the family they allow for (the features and conditions, in each
sub-experiment). Then one table per test as tabs, each with p and the adjusted p
(Benjamini-Hochberg, the q-value) in their own columns and its data as CSV: across
repeats, every feature and condition with its direction in the repeats, its change
against the control (the condition's mean over the control's, a fold change; the
difference of the means where a mean is not positive, so a significant change can be
read for its size), the mean and range of the per-repeat d and its verdict (raw values
folded); the summary names the same change beside each significant feature; the
cell-level screen in the version the report trusts, with the balance of the repeats and
how often it gives p ≤ 0.05 with no effect for the readout (its other versions folded,
never one whose control comes from repeats the condition was not imaged in); beyond the
mean; reagents. The card's "Show more" holds the section's assumptions only. The
readings condition by condition, the feature ranking, the tables within single repeats
and the plan for the next experiment are in the text report (`report_full.txt`), not on
this card (owner, 2026-10-02). The page calls the q-value the adjusted p throughout.

With several features, every test of "Do the conditions differ?" is corrected over
every feature and condition of a sub-experiment: the readout's paired t-test and its
cell-level screen take the feature discovery's p and q (`workflows._families`), so the
readout reads the same in its own table, the summary and the heatmap.
