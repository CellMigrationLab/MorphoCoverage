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
against the control of its own repeats, with the same number of observations
drawn from every repeat, as the median d and median p over 100 draws. Across
the draws every observation is used, each repeat weighing the same in each.
When the control is pooled from other repeats and no balanced test can be formed
(every repeat of the condition too small), the condition gets no cell-level number.

**Permutations.** The randomization p is `(extreme + 1) / (B + 1)` over `B`
shuffles: 2000 for the readout's test as provided, 9999 in the feature screen,
999 in each balanced draw. It cannot fall below `1 / (B + 1)`, so with `K`
tests in a correction family Benjamini-Hochberg cannot give a q below
`K / (B + 1)`: 210 tests at the balanced test's 999 shuffles gave q 0.07 for
three clear differences, never 0.05. Every test at p ≤ 0.05 after the first
pass is therefore rerun with `B = max(default, ceil(20 K))` shuffles, `K` the
tests of its sub-experiment, so the smallest q, `K / (B + 1)`, is below 0.05
(owner, 2026-10-02: "More permutations"). Only those: a test above 0.05 cannot
pass at any rank, and a smaller p moving never changes the q of a larger one.
The rerun is seeded from the test's own identity, so the result does not
depend on the order of the tests. The balanced test is rerun on its median
draws only: the draws at the median |d| (two of the 100), whose mean p, at `B`
shuffles, replaces the median p. All draws hold the same numbers of
observations, so the p falls with |d| and those draws carry the median p, up
to the first pass's Monte Carlo error; rerunning all 100 would cost a hundred
times as much. The report states the shuffles rerun beside the screen's
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
small, nothing is balanced and only the test as provided is given.

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
- **The balanced p is a median over draws**, which describes a typical draw. It
  is not the p of one test; when rerun with more shuffles it is the mean p of
  the two median-|d| draws. Balancing removes the extra weight of the repeats
  holding more observations, not the independence assumption.
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
  (`sum(n²) / sum(n)`). The balanced test stays at 12-14%.
- **A control pooled from other repeats is the worst case on raw values.** The
  raw test as provided gives 46% false positives even when the repeats agree,
  because batch shifts between repeats no longer cancel. Normalising cuts it
  to 9%, and the balanced test, against the control of the condition's own
  repeats, to 2%. Read the balanced numbers whenever the balance line says the
  control comes from other repeats.
- **The simulations are one design.** They have one well per condition and
  repeat and normal cell noise.

## Many features

With several declared features, `feature_discovery` asks which of them separate
the conditions (owner decision, 2026-10-01: the cell-level test for discovery,
the paired test kept as the second gate):

- **The discovery gate**: the cell-level test of every feature and condition, in
  the version the readings trust most (normalised, balanced when unbalanced),
  Benjamini-Hochberg over all of them within each sub-experiment, a feature passing at q ≤ 0.05; passing
  features are ranked by |d| and the plots draw the best-ranked.
- **The second gate**: the paired t on the repeat means (log ratios when every
  mean is positive), Benjamini-Hochberg over every feature and condition of a
  sub-experiment with 3 or more repeats. A passing feature it also passes is *significant across
  repeats*.

The discovery gate inherits the cell-level test's assumption: with variation
between repeats it passes more features than 5% without an effect, so its list
is candidates; the paired gate holds 5% and, with 3 repeats, confirms only
large, consistent effects. The PCA and UMAP (`embedding_plots`) describe how the
conditions sit over all features at once and test nothing; on raw values a
separation can be a batch between repeats.

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
