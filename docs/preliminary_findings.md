# Preliminary findings

These are exploratory, retrospective observations made while building the
framework. They are not benchmark results and not publication claims. Each
entry states what was run, what it shows, and what it does not show.

## 1. Whole FOVs versus random cells in the JCS and JCB settings (2026-09-23)

**Question.** Both cell-number rules we reproduce treat the cells of a sample
as if they were drawn at random:

- Zweifach (JCB 2024) assumes spatially uniform samples, with averaging noise
  `SD / sqrt(n)`.
- Marcotti et al. (JCS 2026) subsample random cells from the whole well.

A microscope instead acquires whole fields. Does a well mean estimated from
whole FOVs have the error that random-cell reasoning predicts?

**Method** (`data/crick_idr/fov_vs_cells.py`, complete source plates, fixed seed):

- For every well, draw `k` whole FOVs at random, 400 times, and take the
  relative RMSE of their mean against the mean of all imaged fields of that
  well.
- Compare this with the exact expected error of the same number of cells drawn
  at random from the whole well: simple random sampling with finite-population
  correction.
- Nothing is fitted or modelled on the FOV side.
- Metrics are the papers' own: nuclear / (nuclear + cytoplasmic) mean
  intensity of fascin (idr0139, the JCS case study) or YAP/TAZ (idr0028
  screen B).

| dataset | FOVs `k` | cells | error from `k` FOVs | error from random cells | ratio, median [IQR] | wells with ratio > 1.5 |
|---|---|---|---|---|---|---|
| idr0139, 308 wells, 4 FOVs/well | 1 | 329 | 4.2% | 1.3% | 3.2 [2.0–4.2] | 87% |
| | 2 | 657 | 2.4% | 0.74% | 3.2 [2.0–4.3] | 87% |
| | 3 | 987 | 1.3% | 0.42% | 3.1 [2.0–4.3] | 87% |
| idr0028 B, 1,144 wells, 28 FOVs/well | 1 | 33 | 1.8% | 1.2% | 1.5 [1.3–1.7] | 48% |
| | 4 | 132 | 0.82% | 0.54% | 1.5 [1.4–1.7] | 53% |
| | 8 | 264 | 0.52% | 0.34% | 1.5 [1.4–1.7] | 53% |
| | 14 | 462 | 0.33% | 0.22% | 1.5 [1.3–1.7] | 53% |

Errors are relative RMSE of the well mean, as medians across wells. Every number in this entry is taken from the tracked outputs `data/crick_idr/results/fov_vs_cells_idr0139_fascin.csv` and `fov_vs_cells_idr0028_screenB_yaptaz.csv`, generated from Crick commit `a81d20c` with checksum-verified inputs.

**What this shows**

- **The random-cell assumption understates sampling error in both datasets.**
  The factor is about 3 (roughly 10× in variance) in the JCS paper's own data,
  and about 1.5 (roughly 2.3× in variance) in idr0028.
- **The factor does not shrink as more FOVs are acquired.** That is
  consistent with substantial FOV-level structure rather than a small-sample
  effect.
- **It is metric- and dataset-dependent.** An earlier exploratory run on
  idr0028 plate 1A (negative-control wells, mean nuclear YAP/TAZ intensity)
  gave a variance inflation of about 4.7, against about 2.3 for the ratio. So
  "keep FOVs" is not a constant correction factor; it has to be measured.
- **Consequences for the JCB criterion on idr0139** (single-sample form of
  the criterion, from the Table 1 statistics).
  - The formula asks for 32 cells at a 5% target and 127 at 2.5%, using
    between-cell CV = 0.131 / 0.465 from Table 1. A single FOV (~330 cells)
    satisfies both.
  - If those cells were random draws, only 1% of wells would miss 2.5%, and
    none would miss 5%.
  - With the actual single FOV, **75% of wells miss 2.5% and 35% miss 5%**.
    With 2 FOVs, 46% miss 2.5% and 3% miss 5%.
  - In the JCB paper's own terms, the 5% target (independent samples) is
    roughly reached at 1–2 FOVs. The 2.5% target recommended for paired
    designs is not reached until about 3 FOVs.
- **Consequences for the ~250–300-cell threshold observed in the JCS case
  study.** The paper presents this as an observation for its example, not a
  universal recommendation, and itself calls the stopping point "somewhat
  arbitrary". On idr0139 it is less than one FOV. Their random-cell
  subsampling reproduces the random-cell column above by construction, so it
  cannot reveal the FOV-level error.
- **Their biological conclusions still hold.** The leptomycin B effect (about
  0.10 in `Fascin_Ratio`) is several times larger than FOV-level sampling
  error. This is consistent with the JCS paper's own point that large effects
  are robust to sample size.
- **Regime-map reading, preliminary:**
  - cell-number guidance is adequate for large effects and for a 5% averaging
    target when a few FOVs are imaged anyway;
  - it is misleading when many cells come from one or two FOVs, and for
    tighter targets, because cell count then overstates the independent
    information acquired.

**What this does not show**

- **Not a benchmark.** It says nothing about whether any FOV-aware
  multidimensional strategy stops better than the baselines; that is the
  benchmark question.
- **FOVs are not shown to be independent.** Adjacent fields may be correlated,
  so whole-FOV resampling can itself be optimistic (AGENTS.md §3).
- **The source is unknown.** Between-FOV variation may be technical
  (illumination, focus, local density, edge effects) or biological. It is not
  decomposed here. Either way it enters every acquisition.
- **Finite references.** The reference is the mean of the imaged fields,
  not the whole well. idr0139 has only 4 fields per well, so its `k`-FOV errors
  are measured against a 4-field reference, and the ratio for `k = 3` is
  especially constrained.
- **Limited scope.** One metric per dataset, and one cell line each.
- **One unexplained number.** The JCS Table 1 SEM column (0.013, 0.017, 0.024,
  0.013) matches neither `SD / sqrt(n)` (0.0035–0.0048) nor a 4-FOV standard
  error (0.0034–0.0176). It was not investigated further.

The data used are described in `data/crick_idr/README.md`. The complete plates
are downloaded by the script; only the paper's wells are committed.

## 2. Positive control: does "something new in the next FOV" find a phenotype that is there? (2026-09-24)

**Question.** The sufficiency report estimates the chance that one more FOV
shows something new. Calibration checks that estimate against the data it came
from, but a homogeneous well cannot show whether a real, different phenotype
would be caught. Here the truth is known by construction.

**Method** (`data/crick_idr/positive_control.py`, committed idr0028 subset,
fixed seed). For each plate, the MOCK well is the host. `k` host FOVs are drawn
at random (20 draws for each `k`), and candidate next FOVs are judged against
them with the report's definition of new: 30 standardized intensity features,
fitted on the `k` FOVs only; a cluster of at least `m` unsupported cells, where
another FOV supports a cell only with at least `m` of its own cells within
`epsilon`. Candidates:

- the host's remaining FOVs (nothing new by construction: the false-alarm rate);
- every FOV of a knockdown well on the same plate (siYAP, siLATS1 or siARAP2);
- a remaining host FOV with 3, 5 or 10 cells of a knockdown FOV added.

Rates at `epsilon = sqrt(30)` and `m = 3`, pooled over the three plates and
both knockdown wells of each (tracked in
`data/crick_idr/results/positive_control_idr0028.csv`):

| FOVs seen | host (false alarm) | whole siYAP FOV | whole siLATS1 FOV | whole siARAP2 FOV | host + 3 siYAP cells | host + 10 siLATS1 cells | host + 10 siARAP2 cells |
|---|---|---|---|---|---|---|---|
| 5 | 5.0% | 100% | 65% | 19% | 94% | 16% | 8% |
| 10 | 2.4% | 100% | 53% | 9% | 94% | 10% | 5% |
| 20 | 0.8% | 100% | 39% | 8% | 95% | 4% | 3% |

**What this shows**

- **A strongly different phenotype is found, even as 3 cells.** Three siYAP
  cells added to a host FOV make it count as new in about 95% of cases, at a
  false-alarm rate of 1–5%.
- **Detection depends on the representation, not on the readout the biologist
  reports.** siLATS1 shifts the YAP/TAZ nuclear ratio further than siYAP
  (+1.4 versus −1.1 SD of the MOCK cells), yet whole siLATS1 fields count as new only 39–65% of
  the time and 10 siLATS1 cells barely beat the false-alarm rate. The 30
  intensity features separate siYAP far more than a
  change in localization, presumably because siYAP lowers YAP/TAZ intensity
  in every compartment. A method is only as sensitive as its features.
- **Weak phenotypes are not distinguished.** siARAP2 fields sit close to the
  false-alarm rate, as expected for a 0.3–0.5 SD shift.
- **The definition matters more than the statistics.** At `epsilon = 0.5 sqrt(30)`
  or with `m = 1`, 22–100% of host FOVs count as new: everything looks new, so
  nothing is informative. `m = 3` at `epsilon = sqrt(30)` was the only setting
  tried with both a low false-alarm rate and full detection of siYAP.
- **Seeing more FOVs lowers false alarms and also lowers detection of moderate
  phenotypes** (siLATS1 65% to 39%): with more FOVs, some host cells resemble
  the knockdown cells and support them.

**What this does not show**

- **Semi-synthetic.** Knockdown cells added to a host FOV were imaged in other
  fields, so the rare-cell case approximates a rare state rather than measuring
  one. The labels are conditions, not verified phenotypes.
- **One dataset and one representation.** Only idr0028 intensity features were
  used; the tuned setting (`epsilon`, `m`) was chosen here and is not validated
  on held-out data.
- **Single candidates, not stopping.** This measures whether one next FOV is
  judged new. It does not yet test whether the probability and bound of the
  report are right when a rare phenotype occurs in a given fraction of FOVs.

## 3. Frequency control: is the next-FOV chance right when a rare state is there? (2026-09-24)

**Question.** Entry 2 shows that the definition of new catches a strong
phenotype in a single candidate FOV. Are the report's probability and 95% bound
right when a rare state occurs in a known fraction of FOVs, and what does
stopping on the bound miss?

**Method** (`data/crick_idr/frequency_control.py`, fixed seed). In the MOCK
wells of plates 1A and 2A (28 FOVs each), 5 siYAP cells from the same plate
are added to a fraction `p` of the FOVs: each FOV independently (spread), or
the same expected number nearest a random FOV by stage position (clustered).
Each planted acquisition (50 per plate, `p` and layout) is replayed in a random
order. After `k` FOVs the estimator (`epsilon = sqrt(30)`, clusters of 3,
symmetric support, transform refitted without each held-out FOV) predicts the chance that the next FOV shows something new,
with its nominal 95% bound. The biological truth is the chance that the next FOV shows
the planted state while none of the first `k` does. A stopping replay stops at
the first `k` whose bound is at most 25%. Results are tracked in
`data/crick_idr/results/frequency_control_idr0028.csv` and
`frequency_control_stops_idr0028.csv`.

| `p` | FOVs seen | predicted | true chance (planted state) | bound | bound held | state already seen |
|---|---|---|---|---|---|---|
| 5% | 5 | 0.10–0.12 | 0.03–0.04 | 0.55–0.58 | 100% | 21–28% |
| 5% | 20 | 0.03 | 0.02 | 0.18–0.19 | 97–99% | 64–68% |
| 10% | 5 | 0.12 | 0.05–0.06 | 0.57–0.58 | 100% | 38–46% |
| 10% | 20 | 0.02–0.03 | 0.01–0.02 | 0.16–0.18 | 96–99% | 85–86% |
| 20% | 5 | 0.11–0.13 | 0.07 | 0.57–0.58 | 99–100% | 65–68% |
| 20% | 20 | 0.01 | 0.00 | 0.16 | 99–100% | 98–99% |

Ranges span the spread and clustered layouts. Stopping when the bound first
reaches 25%:

| `p` | stop (median FOVs) | planted state never seen at the stop, among acquisitions that had it | true chance of it in the next FOV at the stop |
|---|---|---|---|
| 5% | 11 | 46–49% | 3.0–3.3% |
| 10% | 11–11.5 | 27–30% | 2.9–3.2% |
| 20% | 11 | 7–8% | 1.4–1.8% |

**What this shows**

- **The bound keeps its promise.** It was at least the true chance of meeting
  the planted state in the next FOV in 96–100% of cases at every frequency and
  prefix length. The point prediction overstates the true chance, by about the
  host false-alarm rate of entry 2.
- **The promise is about the next FOV, not about whether a rare state exists.**
  Stopping when the bound reaches 25% happens at 11 FOVs in every condition,
  because with no new FOV the bound depends only on the number of FOVs. At
  that stop the planted state had not been seen yet in almost half the
  acquisitions that contained it at a 5% frequency, and in about 30% at 10%,
  while the chance of it appearing in the very next FOV was only about 3%, as
  the bound said. A small next-FOV risk is not evidence that no rare state
  remains (AGENTS.md §3).
- **The actionable rule is a count of FOVs.** Seeing no new FOV among `k`
  bounds the per-FOV frequency of an unseen state below about `3 / k` (rule
  of three). Ruling out a state present in 5% of FOVs at 95% needs about 60
  FOVs with nothing new, more than these wells have (28).
- **Clustering made no difference here** because FOVs were replayed in random
  order; a clustered state then behaves like a spread one. In a raster or
  otherwise spatially ordered acquisition it would be found later.

**What this does not show**

- **Semi-synthetic**, as entry 2: planted cells come from other fields, and
  siYAP is the easiest state to detect. A state detected less reliably (entry
  2: siLATS1) would be missed more often.
- **Random orders only.** Recorded, spatially ordered acquisition was not
  replayed.
- **Two wells of 28 FOVs**, one representation and one setting of the
  definition, chosen on the same dataset.

## 4. How many fields do you need? Sampling reliability on idr0028 (2026-09-24)

**Question.** For a stated tolerance, how often does a random set of `k`
fields give the same answer as fields that were not used? This is roadmap
Step 1 (`morphocoverage.retrospective.sampling_reliability`). It is a property
of the acquisition, not of any stopping rule.

**Method** (`data/crick_idr/reliability_idr0028.py`, fixed seed). The data are
the 12 wells of the committed subset, 28 FOVs each, analysed one well at a
time. The phenotype is the YAP/TAZ nuclear ratio of Marcotti et al. (nuclear
mean / (nuclear + cytoplasmic mean intensity)). Each of 10 splits holds back a
random 10 FOVs as the reference. For each `k`, 50 random sets of `k` whole
FOVs are drawn from the other 18 and judged against that same reference. The
three criteria are the roadmap's provisional presets:

- the well mean within 5%;
- the well mean within 2.5%;
- the variance within 10%.

Results are in `data/crick_idr/results/reliability_idr0028.csv` and
`reliability_ceiling_idr0028.csv`. The ceiling is how often all 18 fields
together met the criterion.

| criterion | 2 fields | 4 fields | 8 fields | 18 fields | all 18 fields (ceiling) |
|---|---|---|---|---|---|
| mean within 5% | 99–100% | 100% | 100% | 100% | 100% |
| mean within 2.5% | 81–100% (median 93%) | 93–100% (99%) | 99–100% | 100% | 100% |
| variance within 10% | 14–30% (20%) | 15–41% (26%) | 14–57% (29%) | 10–80% (30%) | 10–80% (30%) |

Each cell gives the range across the 12 wells, with the median in brackets.

**What this shows**

- **For the well mean, two fields are almost always enough.** At a 5%
  tolerance, 2 fields matched the held-back fields in at least 99% of draws in
  every well. At 2.5%, 2 fields were enough in 81–100% of draws (worst:
  ARAP2 2B C17 and YAP 2A N23), 4 fields in at least 93%, and 8 in at least
  99%. The median error of a 2-field mean was 1%. With 28 FOVs per well, this
  mean was heavily oversampled.
- **The variance within 10% cannot be judged with this design.** Even all 18
  fields matched the 10 held-back fields in only 30% of splits (median; 10–80%
  across wells). The held-back fields are too few to pin the variance down to
  10%. The curve is flat because it is capped by that ceiling, not because
  more fields do not help: the median error falls from 0.26 at 2 fields to
  0.17 at 18. The report states the ceiling for this reason. Here it says that
  the question needs more fields than the well holds, not that 18 fields fail.
- **The criterion decides the answer.** The same wells need 2 fields or cannot
  be answered, depending on whether the question is the mean or the spread.
  A protocol must therefore state its criterion; there is no single number of
  fields.

**What this does not show**

- **Development data only.** idr0028 is the only registered dataset and was
  used to build this analysis. No field count here is a protocol.
- **Nothing beyond 18 fields**, and nothing about wells, plates or repeats:
  every statement is within one well. Draws and splits are not independent
  replicates. The ranges across wells describe these 12 wells only.
- **Fields are treated as random draws** from the well. Spatial order and
  neighbour correlation between fields are not modelled (roadmap Step 8).
- **One feature**, one representation, and provisional tolerances.

## 5. Would a stopping rule have stopped at the right time? Replayed idr0028 acquisitions (2026-09-24)

**Question.** Entry 4 shows how often random sets of fields preserve the well
mean. A stopping rule decides from the fields seen so far, in the order they
arrive. When would the rule have stopped, and was it enough at that point,
judged on fields the rule never saw? This is roadmap Steps 2 and 3
(`replay_acquisitions`, `stopping_calibration`, `adequacy_points`).

**Method** (`data/crick_idr/stopping_idr0028.py`, fixed seed; 45 s). The data
are the same 12 wells and the same YAP/TAZ nuclear ratio as entry 4. Each of 5
splits holds back 10 FOVs, the same reservoirs as the first five splits of
entry 4. The other 18 FOVs are replayed in the recorded order and in 10 random
orders. Every rule decides from 2 FOVs on. The rules are:

- the JCB averaging-CV rule at 5%, on cells as published;
- the same rule on whole FOVs;
- fixed rules of 4 and 8 FOVs.

"Enough" is judged against the held-back FOVs: the well mean within 5%, and
within 2.5%. The tables are `data/crick_idr/results/stopping_calibration_idr0028.csv`
and `adequacy_idr0028.csv`.

| rule | stop (FOVs) | premature, mean within 5% (random orders; median, max over wells) | premature, mean within 2.5% (random orders; median, max over wells) | premature, 2.5%, recorded order (max over wells, of 5) |
|---|---|---|---|---|
| JCB CV ≤ 5%, cells | 2 | 0%, 2% | 10%, 20% | 40% |
| JCB CV ≤ 5%, FOVs | 2 | 0%, 2% | 10%, 20% | 40% |
| fixed 4 FOVs | 4 | 0%, 0% | 0%, 10% | 20% |
| fixed 8 FOVs | 8 | 0%, 0% | 0%, 2% | 0% |

Judged on held-back fields, the mean was first within both tolerances at a
median of 1 FOV (all 660 replayed acquisitions: 1–5), and lastingly from a
median of 1 (1–13; 75% by 2 FOVs).

**What this shows**

- **Both JCB-style rules stopped as early as they were allowed (2 FOVs) in
  every well.** With about a hundred nuclei per field, the averaging CV of the
  mean ratio is below 5% after two fields, whether cells or FOVs are the
  resampling unit. For the mean within 5%, which is the target the rule was
  designed for, that stop was almost always right (at most 2% premature). For
  the stricter 2.5%, it was premature in a median 10% of random acquisitions,
  up to 20% in the worst well. A rule at 5% CV answers the 5% question; it is
  not a promise for 2.5%.
- **The mean was never the hard part here.** Held-back fields confirmed it
  after one or two fields in most runs, and never later than 13. This is the
  regime AGENTS.md section 10 asks us to name: for the mean nuclear YAP/TAZ
  ratio of a well in this screen, the cell-number guidance is sufficient, and
  FOV-aware sampling adds nothing for this readout.
- **Fixed rules.** Four fields were enough for 5% in every run, and for 2.5% in
  all but up to 10% of random acquisitions of the worst well. Eight fields
  were almost never premature.

**What this does not show**

- **One easy readout.** The stop rules were not tested on a question they find
  hard (the variance, a quantile, a rare state), since entry 4 shows the
  variance cannot be judged with this reservoir.
- **Development data, 12 wells**, recorded-order results over only 5 hold-outs
  per well. Replayed acquisitions are hold-outs and orders of the same fields,
  not independent experiments.
- **The minimum of 2 FOVs is a setting of this run**, and it caps how early the
  rules could stop.
