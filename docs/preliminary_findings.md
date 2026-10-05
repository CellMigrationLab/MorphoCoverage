# Preliminary findings

These are retrospective observations made while building the framework,
not publication claims. Entries 1 to 7, 9 and 10 are exploratory analyses,
not benchmark results. Entry 8 is the result of the locked benchmark
(`docs/benchmarking.md`, "The score"), preliminary because it rests on one
development dataset, one shallow held-out dataset and few independent units.
Each entry states what was run, what it shows, and what it does not show.

**What the numbers are from.** Every tracked table and number below comes
from the 2026-09-28 run of `data/crick_idr/run_all.py` after the fourth,
adversarial audit (`docs/audit/README.md`), with the benchmark under protocol
version 5 (`data/crick_idr/benchmark_protocol.py`, locked by digest
`5ebaa47d` before its run). The tables that run left unchanged (entries 1
to 3, the treatment comparisons of entry 9, and several tables of entries 4,
6 and 7) keep the commit of their earlier run. `run_all.py` writes its run
times to `data/crick_idr/results/logs/run_all.txt`, which is not tracked, so
the run times quoted below are those of the earlier runs and were not
remeasured.

## 1. Whole FOVs versus random cells in the JCS and JCB settings (2026-09-23)

**Question.** Both cell-number rules we reproduce treat the cells of a sample
as if they were drawn at random:

- Zweifach (JCB 2024) assumes spatially uniform samples, with averaging noise
  `SD / sqrt(n)`.
- Marcotti et al. (JCS 2026) subsample random cells from the whole well.

A microscope instead acquires whole fields. Does a well mean estimated from
whole FOVs have the error that random-cell reasoning predicts?

**Method** (`data/crick_idr/fov_vs_objects.py`, complete source plates, fixed seed):

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
| idr0139, 308 wells, 4 FOVs/well | 1 | 329 | 4.2% | 1.3% | 3.2 [1.9–4.2] | 87% |
| | 2 | 657 | 2.4% | 0.74% | 3.2 [2.0–4.3] | 87% |
| | 3 | 987 | 1.3% | 0.42% | 3.1 [2.0–4.3] | 87% |
| idr0028 B, 1,144 wells, 28 FOVs/well | 1 | 33 | 1.8% | 1.2% | 1.5 [1.3–1.7] | 49% |
| | 4 | 132 | 0.82% | 0.54% | 1.5 [1.4–1.7] | 53% |
| | 8 | 264 | 0.53% | 0.34% | 1.5 [1.3–1.7] | 53% |
| | 14 | 462 | 0.33% | 0.22% | 1.5 [1.4–1.7] | 54% |

Errors are relative RMSE of the well mean, as medians across wells. The spread of the per-well ratios (the IQR, and the share of wells above 1.5) is mostly estimation noise from 4 (or 28) imaged fields per well: simulated wells that all share one ratio give the same spread. It does not show that wells differ, and per-well extremes are not quoted. Every number in this entry is taken from the tracked outputs `data/crick_idr/results/fov_vs_objects_idr0139_fascin.csv` and `fov_vs_objects_idr0028_screenB_yaptaz.csv`, generated from Crick commit `a81d20c` with checksum-verified inputs. The complete plates are stored in the repository (`idr0139_plate`, `idr0028_screenB_plates`), and the tables are regenerated from them. The fourth audit's rerun left both tables unchanged; two IQR ends above now follow the tables' rounding (`ratio_q25` 1.948 at one FOV on idr0139, and 1.353 at 14 FOVs on idr0028), where they read 2.0 and 1.3 before.

**What this shows**

- **The random-cell assumption understates sampling error in both datasets.**
  The factor is about 3 (roughly 10× in variance) in the JCS paper's own data,
  and about 1.5 (roughly 2.3× in variance) in idr0028.
- **The factor is the same at every number of FOVs, by construction.** Both
  errors carry the same finite-population correction, so their ratio is
  about the square root of the design effect whatever `k`, with or without
  FOV-level structure. Its constancy is not evidence; its size is.
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
  - These errors are relative to the imaged fields: the reference is the
    mean of the well's 4 imaged fields, whose finite-population correction
    shrinks the error of 2 or 3 of them, so they understate the error
    against the well's other positions. In the JCB paper's own terms,
    relative to the imaged fields, the 5% target (independent samples) is
    roughly reached at 1–2 FOVs, and the 2.5% target recommended for paired
    designs is not reached at 2 FOVs.
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
are stored there with the readout only (`datasets.load("idr0139_plate")` and
`"idr0028_screenB_plates"`); the paper's wells carry every intensity feature.

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
`epsilon`. The features are standardized by mean and SD (`Standardize()`,
passed explicitly by the script). Since the fourth audit (protocol
version 5) the default definition of new of the workflows (`Setup`)
standardizes robustly, by median and MAD (`RobustStandardize`); this control,
entry 3, the report steps of entry 6 and the benchmark's novelty bound
(entry 8) still pass `Standardize()`, so their rates are those of the mean
and SD definition, not of the new default. Candidates:

- the host's remaining FOVs (nothing new by construction: the false-alarm rate);
- every FOV of a knockdown well on the same plate (siYAP, siLATS1 or siARAP2);
- a remaining host FOV with 3, 5 or 10 cells of a knockdown FOV added.

Rates at `epsilon = sqrt(30)` and `m = 3`, pooled over the three plates and
both knockdown wells of each (tracked in
`data/crick_idr/results/positive_control_idr0028.csv`, which the fourth
audit's rerun left unchanged):

| FOVs seen | host (false alarm) | whole siYAP FOV | whole siLATS1 FOV | whole siARAP2 FOV | host + 3 siYAP cells | host + 10 siLATS1 cells | host + 10 siARAP2 cells |
|---|---|---|---|---|---|---|---|
| 5 | 5% | 100% | 32% | 14% | 94% | 15% | 8% |
| 10 | 3% | 100% | 21% | 8% | 95% | 8% | 5% |
| 20 | 1% | 100% | 13% | 6% | 93% | 3% | 2% |

FOVs are keyed by plate and file (`datasets.fov_matrices`), here and in
entry 3: idr0028 reuses file names across plates, and two donor wells
(siLATS1 J23 on plates 1A and 2A, siARAP2 C13 on plates 2A and 2B) have a
namesake on another plate, so a key by file name alone would merge the nuclei
of two fields from two plates into one "whole FOV".

**What this shows**

- **A strongly different phenotype is found, even as 3 cells.** Three siYAP
  cells added to a host FOV make it count as new in about 95% of cases, at a
  false-alarm rate of 1–5%.
- **Detection depends on the representation, not on the readout the biologist
  reports.** siLATS1 shifts the YAP/TAZ nuclear ratio further than siYAP
  (+1.4 versus −1.1 SD of the MOCK cells), yet whole siLATS1 fields count as new only 13–32% of
  the time and 10 siLATS1 cells barely beat the false-alarm rate. The 30
  intensity features separate siYAP far more than a
  change in localization, presumably because siYAP lowers YAP/TAZ intensity
  in every compartment. A method is only as sensitive as its features.
- **Weak phenotypes are not distinguished.** siARAP2 fields sit close to the
  false-alarm rate, as expected for a 0.3–0.5 SD shift.
- **The definition matters more than the statistics.** At `epsilon = 0.5 sqrt(30)`
  or with `m = 1`, 24–100% of host FOVs count as new: everything looks new, so
  nothing is informative. `m = 3` at `epsilon = sqrt(30)` was the only setting
  tried with both a low false-alarm rate and full detection of siYAP.
- **Seeing more FOVs lowers false alarms and also lowers detection of moderate
  phenotypes** (siLATS1 32% to 13%): with more FOVs, some host cells resemble
  the knockdown cells and support them.

**What this does not show**

- **Semi-synthetic.** Knockdown cells added to a host FOV were imaged in other
  fields, so the rare-cell case approximates a rare state rather than measuring
  one. The labels are conditions, not verified phenotypes.
- **One dataset and one representation.** Only idr0028 intensity features were
  used; the tuned setting (`epsilon`, `m`) was chosen here and is not validated
  on held-out data.
- **Single candidates, not stopping.** This measures whether one next FOV is
  judged new. Whether the probability and bound of the report are right when a
  rare phenotype occurs in a given fraction of FOVs is entry 3.

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
`frequency_control_stops_idr0028.csv`; the fourth audit's rerun left both
unchanged (the script standardizes by mean and SD, entry 2).

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

- **The bound is conservative here.** It was at least the true chance of
  meeting the planted state in the next FOV in 96–100% of cases at every
  frequency and prefix length. That does not show it calibrated: the true
  chances sit far below it, and a bound that ignores the data,
  `1 - 0.05 ** (1 / k)` (what the estimator gives when nothing new is seen),
  holds about as often. Telling the two apart needs settings where the truth
  approaches the bound (a higher `p`, a larger `k`). The point prediction
  overstates the true chance, by about the host false-alarm rate of entry 2.
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
  of three), under two conditions: the state is one this definition of new
  detects in the fields that hold it, and the fields are independent,
  representative draws (a random or dispersed order; a raster over a
  spatial trend breaks it). For a state detected in only part of its fields,
  such as the whole siLATS1 fields of entry 2, the bound is about
  `3 / (k × detection rate)`, the detection-adjusted bound of the report
  (R1). Ruling out a state present in 5% of FOVs at 95% needs about 60
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
fields give the same answer as fields that were not used? This is P1 of
`docs/questions.md` (`morphocoverage.retrospective.sampling_reliability`). It is a property
of the acquisition, not of any stopping rule.

**Method** (`data/crick_idr/reliability_idr0028.py`, fixed seed). The data are
the 12 wells of the committed subset, 28 FOVs each, analysed one well at a
time. The phenotype is the YAP/TAZ nuclear ratio of Marcotti et al. (nuclear
mean / (nuclear + cytoplasmic mean intensity)). Each of 10 splits holds back a
random 10 FOVs as the reference. For each `k`, 50 random sets of `k` whole
FOVs are drawn from the other 18 and judged against that same reference. The
three criteria are provisional presets:

- the well mean within 5%;
- the well mean within 2.5%;
- the variance within 10%.

Results are in `data/crick_idr/results/reliability_idr0028.csv` and
`reliability_ceiling_idr0028.csv`. The ceiling is how often all 18 fields
together met the criterion.

| criterion | 2 fields | 4 fields | 8 fields | all 18 fields (ceiling) |
|---|---|---|---|---|
| mean within 5% | 99–100% | 100% | 100% | 100% |
| mean within 2.5% | 78–99% (median 92%) | 86–100% (99%) | 93–100% | 90–100% (100%) |
| variance within 10% | 16–31% (22%) | 16–42% (29%) | 12–48% (32%) | 10–50% (30%) |

Each cell gives the range across the 12 wells, with the median in brackets.
The numbers moved in the third audit's rerun (`docs/audit/README.md`) because
each split's reservoir is now seeded with `split_seed`, so the held-back fields
differ; the fourth audit's rerun left every share unchanged. The curve now
carries the Monte Carlo standard error of each share (`preserved_mc_se`: the
SD of the share over the 10 splits divided by the square root of 10). For the
mean within 2.5% at 2 fields it is 0.4 to 4.6 points (median over wells 1.6;
the largest in ARAP2 2B C17); for the variance within 10% it has a median of
2.6 points at 2 fields and 15 at 18 fields, where each split contributes a
single set of fields. Differences of a few points between wells or field
counts are within that error.

**What this shows**

- **For the well mean, two fields are almost always enough.** At a 5%
  tolerance, 2 fields matched the held-back fields in at least 99% of draws in
  every well. At 2.5%, 2 fields were enough in 78–99% of draws (worst:
  ARAP2 2B C17 and YAP 2A N23), 4 fields in at least 86%, and 8 in
  at least 93%; in ARAP2 2B C17 even all 18 fields met 2.5% in only 90% of
  splits. The median error of a 2-field mean was 1%. With 28 FOVs per well, this
  mean was heavily oversampled.
- **The variance within 10% cannot be judged with this design.** Even all 18
  fields matched the 10 held-back fields in only 30% of splits (median; 10–50%
  across wells). The held-back fields are too few to pin the variance down to
  10%. The curve is flat because it is capped by that ceiling, not because
  more fields do not help: the median error falls from 0.24 at 2 fields to
  0.14 at 18. The report states the ceiling for this reason. Here it says that
  the question needs more fields than the well holds, not that 18 fields fail.
- **The criterion decides the answer.** The same wells need 2 fields or cannot
  be answered, depending on whether the question is the mean or the spread.
  A protocol must therefore state its criterion; there is no single number of
  fields.

**What this does not show**

- **Development data only.** idr0028 is the development dataset and was
  used to build this analysis. No field count here is a protocol.
- **Nothing beyond 18 fields**, and nothing about wells, plates or repeats:
  every statement is within one well. Draws and splits are not independent
  replicates. The ranges across wells describe these 12 wells only.
- **Fields are treated as random draws** from the well. Spatial order and
  neighbour correlation between fields are not modelled (P8, entry 6).
- **One feature**, one representation, and provisional tolerances.

## 5. Would a stopping rule have stopped at the right time? Replayed idr0028 acquisitions (2026-09-24)

**Question.** Entry 4 shows how often random sets of fields preserve the well
mean. A stopping rule decides from the fields seen so far, in the order they
arrive. When would the rule have stopped, and was it enough at that point,
judged on fields the rule never saw? This is P6, R2 and P4 of
`docs/questions.md` (`replay_acquisitions`, `stopping_calibration`,
`adequacy_points`).

**Method** (`data/crick_idr/stopping_idr0028.py`, fixed seed; 40 s in the third audit's run, not remeasured). The data
are the same 12 wells and the same YAP/TAZ nuclear ratio as entry 4. Each of 5
splits holds back 10 FOVs, the same reservoirs as the first five splits of
entry 4. The other 18 FOVs are replayed in the recorded order and in 10 random
orders. Every rule decides from 2 FOVs on. The rules are:

- the single-group JCB-style averaging-CV rule at 5% on objects (own-group
  normalization; the faithful control-normalized multi-sample procedure is
  `plan_objects_to_average` + `PlannedObjectCountStrategy`);
- the same rule on whole FOVs;
- fixed rules of 4 and 8 FOVs.

"Enough" is judged against the held-back FOVs: the well mean within 5%, and
within 2.5%. The tables are `data/crick_idr/results/stopping_calibration_idr0028.csv`
and `adequacy_idr0028.csv`.

The headline is the share of acquisitions **not preserved at the stop**: the
rule stopped, and the mean was not within the tolerance of the held-back
fields there. It splits into stops that were premature (the whole stream
preserved the mean, or could not judge it) and stops that not even the whole
stream could have fixed. The held-back fields are a sample too, so reference
noise alone can put the whole stream outside the tolerance: the premature
share is a lower bound on the stops that came too early, and the share not
preserved at the stop an upper bound. Here no stop fell in the second part
(`not_preserved_by_stream` is 0 in every well, order and criterion), so the
two bounds coincide and every stop not preserved was premature.

| rule | stop (FOVs, median) | not preserved at the stop, mean within 5% (random orders; median, max over wells) | not preserved, mean within 2.5% (random orders; median, max over wells) | not preserved, 2.5%, recorded order (max over wells, of 5) |
|---|---|---|---|---|
| JCB CV ≤ 5%, cells | 2 | 0%, 2% | 10%, 20% | 80% |
| JCB CV ≤ 5%, FOVs | 3 | 0%, 2% | 3%, 16% | 40% |
| fixed 4 FOVs | 4 | 0%, 0% | 1%, 12% | 20% |
| fixed 8 FOVs | 8 | 0%, 0% | 0%, 4% | 20% |

Judged on held-back fields, the mean was first within both tolerances at a
median of 1 FOV (all 660 replayed acquisitions: 1–6), and lastingly from a
median of 1 (1–12; 88% by 2 FOVs). The random orders are drawn per well,
from the replay's seed. Every number moved in the third audit's rerun
(`docs/audit/README.md`) because each split's reservoir is now seeded with
`split_seed`. In the fourth audit's rerun only the JCB rule on FOVs changed:
its upper-95 bound now takes Kish's effective number of fields as its degrees
of freedom and is not yet enough data below 2 effective fields, so it stopped
at 3 FOVs in 638 of its 660 replayed acquisitions, at 2 in 19 and at 4 in 3
(`adequacy_idr0028.csv`, `first_apparent`), where it stopped at 2 in all
of them before.

**What this shows**

- **The JCB-style rule on cells stopped as early as it was allowed (2 FOVs)
  in every well; the rule on FOVs stopped one field later.** With 20 to 36
  nuclei per field, the averaging CV of the mean ratio on cells is below 5%
  after two fields. On FOVs, two fields of unequal size are fewer than 2
  effective fields, so the rule waits for a third (3 FOVs in 638 of 660
  acquisitions). For the mean within 5%, which is the target the rule was
  designed for, both stops were almost always right here: the cell rule's
  stop was not preserved in at most 2% of the random orders of any well
  (ARAP2 2B C17) and in 1 of 5 recorded-order hold-outs of one well (YAP 2A
  N23); the FOV rule's in at most 2% of the random orders of YAP 2A N23 and
  in no recorded-order hold-out. For the stricter 2.5%, the cell rule's stop
  was not preserved in a median 10% of random acquisitions, up to 20% in the
  worst well (ARAP2 2B C17, and 4 of its 5 recorded-order hold-outs); the FOV
  rule's, one field later, in a median 3%, up to 16% (YAP 2A N23), and in 2
  of 5 recorded-order hold-outs of ARAP2 2B C17 and of YAP 1A M23. A rule at
  5% CV answers the 5% question; it is not a promise for 2.5%.
- **The mean was never the hard part here.** Held-back fields confirmed it
  after one or two fields in most runs, and lastingly by 12 at the latest. This is the
  regime AGENTS.md section 10 asks us to name: for the mean nuclear YAP/TAZ
  ratio of a well in this screen, the cell-number guidance is sufficient, and
  FOV-aware sampling adds nothing for this readout.
- **Fixed rules.** Four fields were enough for 5% in every run, and for 2.5% in
  all but up to 12% of random acquisitions of the worst well (YAP 2A N23);
  in the recorded order they were premature in 1 of 5 hold-outs in each of
  four wells (LATS1 1A J23, YAP 1A M23, YAP 2A N23, ARAP2 2B C17). Eight
  fields were premature in at most 4% of random acquisitions (YAP 2A N23)
  and in 1 of 5 recorded-order hold-outs of ARAP2 2B C17.

**What this does not show**

- **One easy readout.** The stop rules were not tested on a question they find
  hard (the variance, a quantile, a rare state), since entry 4 shows the
  variance cannot be judged with this reservoir.
- **Development data, 12 wells**, recorded-order results over only 5 hold-outs
  per well. Replayed acquisitions are hold-outs and orders of the same fields,
  not independent experiments.
- **The minimum of 2 FOVs is a setting of this run**, and it caps how early the
  rules could stop.

## 6. Protocol, fields to inspect, controls, effort and order on idr0028 (2026-09-24)

**Question.** The other report questions on the same 12 wells (P2, R5, R1,
P10, P7 and P8 of `docs/questions.md`, then the whole report): what field number
would a protocol need, which fields should be reopened, can these features see
the declared siYAP phenotype, would more fields or more wells help, and could
the same positions have been visited in a better order?

**Method** (`data/crick_idr/report_steps_idr0028.py`, fixed seeds; 100 s in the
third audit's run, not remeasured; run after the P1 curves of entry
4). Every table is in `data/crick_idr/results/`,
and the assembled report is `microscopist_report_idr0028.txt`. idr0028 remains a
development dataset.

**What this shows**

- **Protocol (P2).** A well counts as adequately represented when at least
  90% of random acquisitions preserved the criterion, and all conditions are
  pooled explicitly.
  - For the well mean within 5%, 2 fields were enough for 12 of 12 wells.
  - Within 2.5%, 2 fields were enough for 8 of 12 wells (by plate: 4 of 5, 3
    of 4, 1 of 3), 4 fields for 11 of 12 (not ARAP2 2B C17), and 8 fields for
    all 12 (reservoirs reseeded with `split_seed` in the third audit).
  - A number chosen on two plates served every well of the third at 5%, and
    11 of 12 wells at 2.5%.
  - For the variance within 10%, no well was adequately represented at any
    depth (the reservoir ceiling of entry 4).
  - With 3 plates the summary says the uncertainty is too large to fix a
    protocol, and it is.
- **Fields to inspect (R5).**
  - Two fields hold far fewer nuclei than their well (6 against a median of
    27 and 21.5).
  - One siYAP field has a mean YAP/TAZ ratio 5.6 SD below its four stage
    neighbours, one siARAP2 field 3.1 SD above, and one MOCK field (2A K16)
    3.2 SD above. The SD is that of the other fields, not including the field
    itself. The MOCK field is listed for the first time in the fourth
    audit's rerun; among its neighbours, fields tied in distance are taken
    by field name (`inspection_idr0028.csv`).
  - Four fields hold a small new state: 3 nuclei no other field of the well
    shows, in 30 standardized features. Judged as the inspection judges a
    field, against the other fields of its well, none of 84 MOCK reference
    fields was called new (at most 4% at 95% confidence), so the report
    expects 0 to 11.8 of the 336 fields by chance: four fields lie within
    that range, so these data cannot tell them from the chance level (the
    third audit measures this rate leave one field out within the
    inspection's analysis groups, now all three MOCK wells).
  - No field moved the well mean or 95th percentile of the ratio by 10% or
    more when left out, as expected with 28 fields per well.
- **Representation control (R1).** Measured after 10 MOCK fields on the
  same plate:
  - siYAP fields were recognized as different in 100% of 1680 checks;
  - MOCK fields with 5 siYAP nuclei planted were recognized in 100% of 720;
  - the other MOCK fields were called different in 2% of 720 (false alarms);
  - judged as the inspection judges a field, against the other MOCK fields
    of its well, MOCK fields with 5 siYAP nuclei planted were recognized in
    100% of 56 checks (plates 1A and 2A; new in the fourth audit's rerun),
    and none of 84 MOCK fields was called new.

  This matches the tracked positive control (entry 2: 100% for siYAP, about
  3% false alarms at 10 fields) computed by independent code.
- **More fields or more wells (P10).** For the mean ratio of a condition:
  - in MOCK, 54% of the variance of the mean is between fields within a well;
  - in the knockdowns, 79–99% is between wells, which are on different plates;
  - neither split is established: the bootstrap interval of the larger
    component includes zero in every condition (the field component in
    MOCK, the well component in the knockdowns), as the report now says for
    each. In the third audit's run the MOCK field component's interval
    excluded zero; the fourth audit widened these intervals for few units
    (`effort_idr0028.csv`: the point estimates and shares are unchanged);
  - fields vary 1.6–2.6 times as much as random cells would (design effect,
    with the cluster size of unequal fields).

  With 3 wells per condition the answer is "cannot tell from these data", as
  designed. Plate and well are confounded here: in MOCK each plate has one
  well. The analysis therefore refuses a plate level and uses plate-qualified
  well and field identities.
- **Order (P8).** For the well mean within 2.5%, recorded, random and
  dispersed orders all reached it after a median of 1 field. The recorded
  serpentine raster needed 2 fields (805 µm of travel) in MOCK 2A K16 and 4
  fields (1.7 mm) in ARAP2 2B C17, where random and dispersed orders needed 1;
  random orders needed 1 field in every well, and dispersed orders 2 fields
  (2.1 mm) in YAP 2A N23 (reservoirs reseeded with `split_seed` in the third
  audit). The random orders are drawn per well. Outside
  ARAP2 2B C17 the orders differ by at most one field, a permitted null result
  (AGENTS.md section 3): the mean is too easy here to tell orders apart. One well where
  the raster is 3 fields slower is not a pattern across 12.
  - Redundancy between neighbouring fields is not shown. Tested against
    stage positions shuffled within each well (199 shuffles), where there is
    no spatial structure by construction, none of the 12 wells shows it
    beyond chance (`redundancy_idr0028.csv`, `beyond_chance`). In 5 wells
    the point estimate suggested that closer fields are more alike, but the
    shuffles gave as large a distance in 6% to 43% of cases (median 38%).
    The earlier count of 5 of 12 wells, reported without that null, was the
    chance level.
- **Objects or fields (the Foundation).** The report opens with the
  Foundation: per well, fields vary 1.0–3.4 times as much as random nuclei
  would (median 2.1; `fov_variability`), and unusual nuclei cluster by field
  in 4 of the 11 wells where they were seen, where seeing them takes about
  1.9 times the fields a count of nuclei predicts (entry 10; 3 in the third
  audit's run, and 5 before the third audit calibrated the unusual-object
  threshold leave one group out).

**What this does not show**

- **One easy readout, 12 wells, 3 plates, development data.** None of these
  numbers is a protocol. The P10 and P2 answers are limited by 3 wells
  per condition and 3 plates, and they say so.
- **The P10 intervals are approximate.** Simulated nested designs with 8
  samples recover the variance components on average (tested), but the
  bootstrap interval covered the true sample and field components in only
  88% and 80% of 40 runs, not 95% (with 40 runs, a Monte Carlo standard
  error of about 5 to 6 points). That simulation predates the fourth audit's
  widening of the intervals for few units and was not rerun. Resampling only the samples was worse (68%
  for the sample component). A stated difference between allocations
  therefore also needs five or more independent units.
- **The reading test** (`docs/questions.md`, open work) has not been done by
  the project owner on this report.

## 7. Held-out check on idr0139: does "two fields are enough for the mean" transfer? (2026-09-25)

**Question.** Entries 4 and 5 found, on the development wells of idr0028, that
the well mean of the YAP/TAZ ratio was within 5% of held-back fields from 2
fields in every well and that the JCB cell-number rule's stop at 2 fields was
almost never premature at 5%. Those wells were also where every preset was
chosen. idr0139 (10 wells × 4 fields, one plate, the JCS case study) is
registered as **held out** (`data/crick_idr/datasets.py`): no setting was ever
chosen on it. Does the conclusion hold there, for the paper's fascin ratio?

**Method** (`data/crick_idr/heldout_idr0139.py`, fixed seed; tables
`results/heldout_*_idr0139.csv`). Per well, 6 hold-outs each keep 2 fields as
the reference and draw 1 or 2 of the other 2 (10 draws). Criteria: the well
mean within 5% and within 2.5%. The single-group JCB-style rule (object unit, own-group normalization,
upper-95 SD, at 5% and 2.5% CV) and a fixed 2-field rule are replayed from 1 field on, in the
recorded and 5 random orders.

| | 1 field | 2 fields | all 2 stream fields (ceiling) |
|---|---|---|---|
| mean within 5%, share of draws enough (median over wells; range) | 68% (17–100%) | 92% (17–100%) | 92% (17–100%) |
| mean within 2.5% | 53% (5–100%) | 42% (0–100%) | 42% (0–100%) |
| wells adequately represented (≥ 90% of draws), within 5% | 2 of 10 | 5 of 10 | |
| wells adequately represented, within 2.5% | 1 of 10 | 2 of 10 | |

| rule | stop | not preserved at the stop, mean within 5% (median; max over wells) | of which premature | not preserved, 2.5% | of which premature |
|---|---|---|---|---|---|
| JCB CV ≤ 5%, cells | 1 field | 32%; 77% | 18%; 33% | 58%; 87% | 7%; 37% |
| JCB CV ≤ 2.5%, cells | 1 field | 32%; 77% | 12%; 33% | 58%; 87% | 2%; 37% |
| fixed 2 fields | 2 fields | 8%; 83% | 0%; 0% | 58%; 100% | 0%; 0% |

Shares are over the random orders, drawn per well. The third audit
(`docs/audit/README.md`) reseeded each split's reservoir with `split_seed`,
which moved every number here, and the calibration now counts separately the
stops that not even both stream fields could have fixed: only the rest are
premature (so a stop at the last stream field is never premature). The
premature share is therefore a lower bound on the stops that came too early,
and the share not preserved at the stop an upper bound. The fourth audit's
rerun left every share of both tables unchanged. The reliability shares now
carry their Monte Carlo standard error over the 6 hold-outs
(`preserved_mc_se`): for the mean within 5% a median of 10 points at 1
field and 8 at 2 fields (up to 18 and 22), so the shares of single wells
are coarse.

**What this shows**

- **The development conclusion does not transfer as a number.** On idr0028,
  2 fields were enough for the mean within 5% in 12 of 12 wells; on idr0139,
  2 fields are enough in 5 of 10 wells, and 1 field (where the JCB rule stops,
  since one field holds about 330 nuclei) in 2 of 10. The JCB stop misses the
  mean within 5% in a median 32% of replayed acquisitions and up to 77% in the
  worst well; in a median 18% (up to 33%) the second field would have fixed
  it. This is the held-out counterpart of entry 1's observation that with
  the actual single field 35% of wells miss 5%.
- **What transfers is the regime, not the count.** idr0139 has about 3 times
  the FOV-level error that random cells predict (entry 1), idr0028 about 1.5
  times; the cell-number rule is sufficient where fields vary little and
  insufficient where they vary more. A protocol must be set per readout and
  per preparation, not carried over.
- **The report says what it can and cannot judge.** The 2-field reference is
  itself noisy: the ceiling is 92% (median; 17% in the worst well) at 5% and
  42% at 2.5%, and the
  protocol summary names the wells whose reference could not judge the
  criterion rather than listing them as needing more fields. So the
  reliability shares above bound nothing by themselves, and the stops are
  bounded only as stated under the table (premature from below, not
  preserved from above): with 4 fields per well the dataset can refute "one
  or two fields suffice" but cannot say how many do.

**What this does not show**

- **One plate, 10 wells, 4 fields each.** The independent unit is the well
  (one repeat), and nothing beyond 2 fields is judged.
- **The reference error is of the same order as the tolerance**: a 2-field
  reference mean has about the sampling error of a 2-field draw, so part of
  every "not preserved" is the reference's own noise. That is why the ceiling
  is printed.
- **One readout.** The fascin ratio only; the 30 intensity features and the
  novelty definition were not tested here.

## 8. The first locked benchmark: fields saved at matched risk (2026-09-25)

**Question.** Do the richer rules, or the JCB and JCS rules per field rather
than per object, need fewer fields than a fixed count for the same risk of
stopping too early? This is the benchmark of `docs/questions.md` (rows R2,
P5 and P6), scored with `retrospective.frontier` (`docs/benchmarking.md`,
"The score").

**Method.** `data/crick_idr/benchmark.py` under protocol version 5
(`benchmark_protocol.py`, locked by digest `5ebaa47d` on 2026-09-28 before
its run; `benchmark_summary.txt` prints `5ebaa47debfc`). Its settings, grids
and seeds are those of version 1, locked before the first run; the protocol
file records what each version changed. Version 3 (second audit) drew
permuted orders per well and fed the rules each prefix in acquisition order;
version 4 (third audit, `docs/audit/README.md`) seeds each split's reservoir
with `split_seed`, counts a subset with too few objects for its metric as not
preserved, gives the JCB rule insufficient data for a feature with no spread
so far, and withholds a frontier interval when fewer than 90% of its
resamples reach the risk. Version 5 (fourth, adversarial audit) changes the
rules and the replay under the same settings: the JCB per-field bound takes
Kish's effective number of fields as its degrees of freedom and is not yet
enough data below 2 effective fields; the FOV bootstrap's stopping SE is
widened for few fields; the JCS spread, plateau and bootstrap are not yet
enough data on a readout constant so far, and the per-field spread needs 5
fields with objects; resamples draw in a canonical (FOV-key, value) order;
each group is seeded from its labels written canonically; the default
novelty definition standardizes robustly (median and MAD), which the
benchmark's novelty bound does not use, since it passes `Standardize()`
(mean and SD) explicitly; and the frontier's interval is widened for its
units and withheld below 5 units. Replays hold back 35% of each idr0028 well (5 splits, recorded and 10 random
orders; 50% of each idr0139 well, 6 splits, recorded and 5 random orders).
Each rule family is swept over a threshold grid fixed in the protocol; risk
is the share of acquisitions that did not preserve the outcome in held-back
fields when the rule stopped; the unit is the well. Before locking, a
reduced dry run (one split, one order, not scored) showed several grids
ending before the cheap end; they were widened under a stated rule (every
grid reaches from never stopping to stopping at the first allowed field), not
on a result. Tables: `results/benchmark_*`. Since then the JCB per-field
upper-95 bound is a bootstrap of whole fields (owner, 2026-10-05); the JCB
per-field numbers below are from the former chi-square bound and were not rerun.

Fields saved against a fixed count, idr0028 (negative: costs fields):

| rule | mean within 5%, risk ≤ 5% | mean within 2.5%, risk ≤ 5% | mean within 2.5%, risk ≤ 10% |
|---|---|---|---|
| fixed count (fields needed) | 1 | 4 | 2 |
| JCB per object | 0.0 | −4.3 | −0.9 |
| JCB per field | −2.0 | +1.0 | −1.0 |
| JCS spread per object | −0.1 | 0.0 | −2.0 |
| JCS spread per field | −4.0 | −1.0 | −3.0 |
| FOV bootstrap of the mean | −1.3 | +0.7 | −0.3 |

The approximate 95% intervals resample the 12 wells and are widened for
their number (with fewer than 20 units they still cover the truth less often
than stated). For the mean within 5% they are narrow, and the summary marks
no cost "(not resolved)": JCB per field 2.0 to 2.0 fields, JCS spread per
field 4.0 to 4.0, the FOV bootstrap 1.2 to 1.4, JCS spread per object 0.0 to
0.1. At the 2.5%
criterion they are wide. JCB per field at risk ≤ 5% saves 1.0 field, with
−0.2 to 1.0 fields saved, and the FOV bootstrap 0.7, with −2.6 to 0.8; every
cost and saving at 2.5% is marked "(not resolved)" except JCS spread per
field at risk ≤ 10%, whose cost of 3.0 fields (1.8 to 3.0) is resolved, and
JCS spread per object at risk ≤ 5%, which matches the fixed count (−9.2 to
0.8 fields saved). Every adaptivity correlation is 0.32 or less. At 2.5% and
5% risk the fixed count needs 4 fields, and every rule's cost there is counted
from 4.

Settings chosen on idr0028 to meet 5% risk, scored on the held-out idr0139:

| mean within 5% | stop on idr0139 | risk on idr0139 |
|---|---|---|
| fixed 1 field, JCB per object, JCS spread per object | 1 field | 36% |
| JCB per field, JCS spread per field, FOV bootstrap | 2 fields (all available) | 27% |

For the mean within 2.5%, every setting reaches 55% risk on idr0139.

Planted rare state (semi-synthetic: 5 siYAP nuclei in 10% or 20% of the
fields of the MOCK wells of plates 1A and 2A, 48 acquisitions):

| rule | risk ≤ 5% | risk ≤ 10% |
|---|---|---|
| fixed count (fields needed) | 16 | 13 |
| novelty bound | −2.0 | −1.9 |
| rare incidence, reading the planted label | +8.9 (Spearman 0.79) | +6.3 (Spearman 0.79) |

No interval is given at either risk level: the independent unit is the
plate, and with 2 units the frontier withholds its interval, which since
protocol version 5 needs at least 5 units (a percentile interval over 2
units covers the truth only about half the time). The intervals the protocol
version 4 run gave at 10% risk are therefore not repeated.

**What this shows**

- **For the well mean on idr0028, nothing beats a fixed count by a resolved
  margin.** This is the null result the regime map needs: the readout is
  easy, one field meets the mean within 5%, and every rule either matches that
  or costs fields; the two savings, JCB per field (1.0 field) and the FOV
  bootstrap (0.7) at 2.5% and 5% risk, have intervals that include 0.
- **Per cell against per field changes where each rule fails.** At 5% the
  per-object rules match a fixed single field (JCS spread per object costs
  0.1), and the per-field rules cost fields, with intervals that exclude
  0: JCB per
  field 2.0, because it is not yet enough data below 2 effective fields and so
  stops at about 3 fields (2.99 on average); JCS spread per field 4.0, because
  it needs 5 fields with objects; the FOV bootstrap 1.3. In the protocol
  version 4 run these costs were 1.3, 1.0 and 1.0 fields. At 2.5% it
  reverses: the per-object JCB rule (upper-95 SD) costs 4.3 fields, while JCB
  per field saves 1.0, because object counts overstate what the fields add.
  Neither difference is resolved over 12 wells.
- **The 1-field settings do not transfer to the held-out wells; the 2-field
  settings cannot be judged there.** At the setting chosen for 5% risk on
  idr0028, idr0139 gives 27–36% risk. Its 2-field reference cannot confirm
  the mean within 5% in 27% of hold-outs even with every field (entry 7), so
  27% is its floor. Stopping at one field (fixed, JCB per object, JCS spread
  per object) is worse than that floor. The per-field rules stop at two
  fields, every stream field, and sit exactly at the floor, so idr0139
  cannot tell whether they transfer. Entry 7's conclusion holds for one
  field: that count does not transfer.
- **For a rare state, the rule that sees the state saves fields; the novelty
  bound does not.** The label rule stops as soon as the state has appeared,
  so easy acquisitions stop early and its stops track the need. The novelty
  bound behaves like a stricter fixed count: with nothing new it depends on
  the number of fields only, and a new state keeps it acquiring. It answers
  "would the next field show something new", not "has the rare state been
  seen" (entry 3).

**What this does not show**

- **Development data, one easy readout.** The distribution questions of R2 and
  the variance could not be scored (entry 4: the reservoir cannot judge them).
- **The held-out dataset is too shallow** to score a rule by itself: its fixed
  count never reaches 5% risk within its 2 stream fields.
- **The rare-state result is semi-synthetic and has 2 independent units.** The
  label rule assumes the state can be classified at the microscope; without
  labels, no packaged rule beat a fixed count.
- **One grid per family**, fixed in advance; a finer grid could move a chosen
  setting by one step.


## 9. Objects or fields, each paper's own procedure (2026-09-25)

**Question** (the Foundation and P5 of `docs/questions.md`). Entry 1 compared errors.
This entry runs each reference paper's own procedure twice: once with objects
(here nuclei with their cytoplasm) drawn at random from the well, as the
paper does, and once with the same number of objects obtained by imaging
whole fields in random order, as a microscope does. Both papers state the
assumption being tested. Zweifach (JCB 2024) "further assume[s] that samples
are spatially uniform". Marcotti et al. (JCS 2026) assume "this small number
of cells is representative of the broader population", and call that "likely
a reasonable assumption" for cultured cells. The JCS saving argument, that a
few hundred cells are enough and imaging more wastes time and money, draws
those cells from every imaged field, after the whole well was imaged.

**Method** (`data/crick_idr/objects_vs_fields.py`, which runs
`morphocoverage.sampling_unit`; stored datasets; seed 0; one analysis group
per well).

- The stored idr0139 tables reproduce JCS Table 1 exactly: wells J05, E22,
  L08 and O02, with the same counts, means, medians and SDs.
- **JCB** (`object_rule_check`). Values are normalized to the mean of every
  object of the negative-control wells of the same repeat (plate), as the
  paper's supplementary spreadsheet and R code do within one experiment, not
  to the mean of the control wells' means (per repeat since the fourth
  audit; before, the control objects of every plate were pooled).
  For each well,
  `n = ceil((SD / target)^2)` at 5% and 2.5%. Whole fields are then drawn at
  random, with replacement, until at least `n` objects, 4000 times (1000 on
  the complete plates); the SD of that average is the real averaging error.
  The JCB formula with the field as the unit (SD of field means, point and
  upper-95 as the paper's calculator gives) says how many fields the rule
  would ask for.
- **JCS Fig. 2A** (`descriptor_spread_check`). For `n` from 10 to 500
  objects, 100 samples of `n` objects; max − min of the mean, median, SD and
  IQR. Field samples take distinct fields in random order until `n` objects,
  with a random part of the last field. The comparison is made at the paper's
  threshold: 250 for idr0139 (Fig. 2) and 200 for idr0028 (Fig. S2).
- The paper's wells and, for the field variability and the JCB rule, every
  well of the complete plates (`idr0139_plate`, 308 wells;
  `idr0028_screenB_plates`, 1,144 wells). Draws come from the imaged fields
  only: 4 per idr0139 well and 28 per idr0028 well. Field errors are
  therefore, if anything, optimistic.

JCB, medians over wells (`results/objects_vs_fields_jcb.csv`):

| | idr0139, paper's 10 wells | idr0139, all 308 wells | idr0028, paper's 12 wells | idr0028, all 1,144 wells |
|---|---|---|---|---|
| object SD, % of the control mean | 25.8% | 26.4% | 6.6% | 6.6% |
| JCB objects at 5% / 2.5% | 28 / 108 | 28 / 112 | 2 / 8 | 2 / 7 |
| fields that holds | 1 | 1 | 1 | 1 |
| real / predicted error for the objects obtained | 2.2 (0.7–3.9) | 2.8 (0.3–6.6) | 1.4 (0.9–1.8) | 1.5 (0.8–13.9) |
| real error of that average | 3.1% | 4.1% | 1.9% | 1.7% |
| wells missing 5% | 2 of 10 | 97 of 308 (31%) | 0 of 12 | 16 of 1,144 (1%) |
| wells missing 2.5% | 8 of 10 | 233 of 308 (76%) | 0 of 12 | 73 of 1,144 (6%) |
| JCB per field at 2.5%, point / upper-95 | 2 / 30 fields | 4 / 50 fields | 1 / 2 fields | 1 / 1 field |

The field variability behind it (`results/objects_vs_fields_variability.csv`),
medians over wells:

| | idr0139, 10 wells | idr0139, 308 wells | idr0028, 12 wells | idr0028, 1,144 wells |
|---|---|---|---|---|
| share of variance between fields | 1.6% | 2.8% | 3.8% | 3.9% |
| design effect | 6.4 | 10.8 | 2.1 | 2.4 |
| fields differing more than random objects allow (5% expected) | 38% | 50% | 16% | 21% |

With hundreds of objects per field, a small between-field share is enough to
dominate the error of a mean.

The ranges in parentheses, in the JCB table above and the JCS table below,
are across wells. As in entry 1,
they are mostly estimation noise from 4 (or 28) imaged fields per well, not
wells that differ: read the medians.

JCS Fig. 2A at the paper's threshold, whole-field spread divided by
random-object spread, medians over wells (`results/objects_vs_fields_jcs.csv`):

| descriptor | idr0139, 250 objects | idr0028, 200 objects | idr0028: objects fields need for the same spread |
|---|---|---|---|
| mean | 1.5 (0.7–2.5) | 1.4 (0.6–1.8) | 330 |
| median | 1.9 (0.5–3.0) | 1.2 (0.8–1.9) | 295 |
| IQR | 1.1 (0.7–2.0) | 1.0 (0.6–1.6) | 240 |
| SD | 0.8 (0.6–2.1) | 1.2 (0.9–1.7) | 250 |

On idr0028 field samples reached the 200-object spread of every descriptor
within 500 objects in all 12 wells (the mean in 11 of 12 in the protocol
version 4 run). On idr0139, field samples reached the 250-object spread of the
mean within 500 objects in only 5 of 10 wells, of the median in 4, of the IQR
in 8 and of the SD in 9 (`objects_vs_fields_summary.txt`). In the fourth
audit's rerun (the JCB reference mean is now that of the control wells of
the same repeat) the JCB medians of the paper's wells and the variability
table did not change; on the complete idr0028 plates the miss counts moved
from 17 and 77 to 16 and 73 wells, and the largest real / predicted ratio
from 14.4 to 13.9.

**Follow-up: the paper's own treatment comparisons, fields against objects**
(`data/crick_idr/jcs_effect_fovs.py`). The same effect-size analysis as JCS
Fig. 3G–I and S3G–I (their Eqn 2, their wells from the companion notebooks,
1000 samples per size) was run both ways. It compares wells, so it stays a
benchmark script (`docs/questions.md`, open work). Its tables
(`results/jcs_effect_fovs*.csv`) were unchanged by the fourth audit's
rerun. Share of samples that rank
two treatments as the full wells do:

| comparison | objects per well | fields that takes | random objects | whole fields |
|---|---|---|---|---|
| idr0139: SN0212398523 > DMSO | 250 | 1–2 of 4 | 81% | 55% |
| | 500 | 2–3 of 4 | 93% | 66% |
| idr0139: Leptomycin B > SN0212398523 | 250 | 2 of 4 | 100% | 100% |
| idr0028: LATS1 > ARAP2 | 200 | about 8 of 28 | 100% | 100% |
| idr0028: ARAP2 > YAP siRNA | 200 | about 8 of 28 | 100% | 100% |

At 250 objects on idr0139, the DMSO effect-size IQR band is 0.02 to 0.13 from
random objects and −0.09 to 0.33 from whole fields.

**What this shows**

- **Both papers are right under their assumption, and the assumption does not
  hold for fields.** The same number of objects gives a less precise average
  when it comes as whole fields: a median of 2.2 to 2.8 times the JCB
  prediction on idr0139 and 1.4 to 1.5 times on idr0028. The complete plates
  agree with the paper's wells, and with entry 1.
- **Whether that matters depends on the readout and on objects per field.**
  On idr0028 the YAP/TAZ ratio varies little between objects and a field
  holds about 30 objects, so one field meets both JCB targets in 94–99% of
  all 1,144 wells (1 − 73/1,144 at 2.5% and 1 − 16/1,144 at 5%), 200 objects
  means about 8 fields, and every treatment
  comparison survives. On idr0139 a field holds about 270 objects: the
  JCB-sized sample misses the paired-design 2.5% target in 76% of the plate's
  308 wells, and the paper's SN0212398523 > DMSO reading at 250 objects comes
  out right in about half of the whole-field samples, a coin flip.
- **The JCB formula still works when the field is the unit.** With the SD of
  field means it asks for 2 to 4 fields at 2.5% on idr0139. With 4 fields,
  however, that SD is so uncertain that the upper-95 bound the paper
  recommends for small pilots asks for 30 to 50 fields. A 4-field pilot cannot
  say how many fields are needed.
- **The JCS stabilization depends on how the objects arrive.** On idr0139,
  location descriptors vary more between field samples than between
  random-object samples (mean 1.5, median 1.9 times the random-object
  spread), spread descriptors barely do (SD 0.8, IQR 1.1). On idr0028 the
  split is less clear than in the protocol version 4 run: the mean varies 1.4
  times as much, the median and the SD 1.2 times, the IQR 1.0. There, fields
  need about 240–330 objects to be as stable as 200 random objects.
- **The number to plan by is fields, not objects.** "X objects" means very
  different effort at 30 and at 270 objects per field. Large effects, such as
  leptomycin B, survive either way.

On the complete idr0028 plates, MOCK means negative control only: the 92
technical-control and 19 empty wells without a gene symbol keep their control
type, since their nuclei (up to 10,134 per well) have a much lower ratio.

**What this does not show**

- **4 fields per idr0139 well** limit every field estimate. A well has more
  positions than were imaged, so the real field error is likely larger.
- **The complete plates carry the readout only**, so the descriptor spread,
  the unusual objects and the treatment comparisons are on the paper's wells.
- **The JCS max − min statistic is noisy.** It is the paper's statistic and
  is kept for faithfulness, but other random streams of the same analysis
  gave median ratios of 1.6 and 1.3 for the mean on idr0139, where this run
  gives 1.5 (1.4 in the protocol version 4 run). The same holds on idr0028:
  the fourth audit found the protocol version 4 run's 1.2 for the mean the
  lowest of five seeds, and this run, after the fourth audit changed how groups
  are seeded and in which order resamples draw, gives 1.4. Read
  the ratios as "above 1 for location descriptors", not to one decimal.
- **The earlier benchmark (entry 8) is a different test.** It scores a rule
  against a tolerance on the realized error in held-back fields and sweeps
  thresholds. The papers' own targets and thresholds are tested here.

## 10. Unusual objects as a rare-state readout: are they clustered by field? (2026-09-25)

**Question** (R4 of `docs/questions.md`). An object-count argument about rare
states assumes random objects: a state carried by a share `p` of objects is
missed in `n` objects with probability `(1 - p)^n`. A microscope acquires
fields, and what decides whether a state is seen is the share of fields that
hold it. Are rare, out-of-distribution objects in these datasets spread over
the fields, or clustered in a few?

**Method** (`data/crick_idr/objects_vs_fields.py`;
`sampling_unit.unusual_objects` and `fov_clustering`; the paper's wells).

- **Unusual objects**: every intensity feature (30 for idr0028, 60 for
  idr0139), standardized on a reference sample of the negative-control wells
  (up to 100 objects per field, so every reference field weighs the same). An
  object's score is its mean distance to the 5 nearest reference objects. It
  is unusual when that score exceeds what all but 1% of reference objects
  reach against the other control wells (calibrated leaving one control well
  out since the third audit; each tested object is scored against all
  control wells but one, drawn at random, so both see an equally dense
  reference; since 2026-10-05 every control well weighs the same in the
  threshold, which the numbers below predate). Each well is scored against the control wells other than
  itself. The design follows
  fiberfeatures' pixel-category fit (reference sampling, provenance, a seed
  check), without its scikit-learn Gaussian mixture: rare states are what a
  mixture tends to absorb, and MorphoCoverage has no scikit-learn dependency.
- **Clustering**: per well, the fields that hold unusual objects against the
  fields expected if the same objects were spread at random over the same
  fields (hypergeometric), a permutation test (999 shuffles of the labels
  over the well's objects), and the fields needed to see the state once with
  95% chance: from its field incidence, and as a random-object argument would
  say (objects needed from the prevalence, divided by objects per field).
  `objects_vs_fields.py` calls `fov_clustering` with its defaults: a field
  holds the state when it has at least one unusual object
  (`min_objects_per_fov = 1`), and no reference is passed, so the control
  wells' own incidence is not printed. The workflows' report, since the
  fourth audit, requires the novelty rule's `m` unusual objects per field and
  prints the control groups' own share; these tables do not, so a field with
  one stray flagged object counts here as holding the state.

Results (`results/objects_vs_fields_rare.csv`, `_unusual.csv`):

| | idr0139, 10 wells | idr0028, 12 wells |
|---|---|---|
| unusual objects per well | 1.9% (0.2–3.4%) | 1.1% (0–2.5%) |
| agreement of refits with other reference samples | 68% (25–84%) | 86% (67–100%) |
| wells where unusual objects cluster by field (p ≤ 0.05) | 2 of 10 | 4 of 12 |
| in those wells, fields needed / fields an object count asks for | 3.0 (3.0–3.0) | 1.9 (1.6–2.2) |

On idr0028, the clustered wells are LATS1 1A I23 (p = 0.029), MOCK 2A K16
(p = 0.007), YAP siRNA 2A N23 (p = 0.002) and ARAP2 2B C13 (p = 0.001). MOCK
2A K16, which clustered before the leave-one-group-out calibration and not in
the protocol version 4 run, clusters again: 12 unusual objects in 6 of 28
fields, where random objects would fill 9.8. One of the four clustered wells
is thus a negative-control well, so clustering of these objects is not
specific to a knockdown. In YAP siRNA 2A N23, 7 unusual objects sit
in 3 of 28 fields (71% in one field), where random objects would fill 6.2.
Seeing them with 95% chance takes 27 fields from their field incidence (up to
131 at the 95% bound, the low end of the two-sided interval), against 12 if the
objects were random.

**What this shows**

- **Rare objects can cluster by field in real data, and then an object count
  under-asks.** In 4 of 12 idr0028 wells, one of them a MOCK well, unusual
  objects occupy fewer fields than random objects would. There, seeing them
  reliably takes about 1.9 times (1.6 to 2.2 times) the fields that the
  object-count argument predicts. That is the rare-state
  version of entry 9: the mean needs fields counted when fields differ; rare
  states need fields counted when they cluster.
- **It is not universal.** In the other wells unusual objects are spread like
  random objects, and an object-count argument gives about the right number
  of fields.

**What this does not show**

- **"Unusual" is a proxy, not a biological state.** It means far from the
  control wells in the chosen intensity features. It mixes rare biology with
  technical outliers (debris, segmentation errors, bright edges), which is
  why the inspection list of the report (R5) exists. Named states need
  labels.
- **idr0139 labels depend on the reference sample** (refits agree on only
  25–84% of unusual objects there, median 68%, and reseeding the groups alone moves the
  clustered wells between 2 and 3 of 10): its fields hold about 270 objects, of
  which 100 are sampled. The idr0139 clustering results should not be read
  beyond "some wells cluster". idr0028 fields hold fewer than 100 objects, so
  every reference object is used, but the control well left out of each
  tested object's reference is drawn at random, so its labels depend on the
  seed too, less so: refits agree on 67–100% (median 86%). Between the
  protocol version 4 run and this one the idr0028 clustered wells went from 3
  to 4 of 12.
- **Multiple wells, one test each.** With 12 wells at p ≤ 0.05, about 0.6
  would be flagged by chance (12 × 0.05); 4 were.
- **The field counts count one stray object.** With `min_objects_per_fov = 1`
  a field holds the state when one of its objects is flagged, and the flag
  marks about 1% of ordinary objects by construction. The fields needed are
  then set by stray flags as much as by a rare state (YAP 1A M23: 1 unusual
  object, "83 fields"); requiring `m` objects per field, as the workflows'
  report does, would count fewer fields holding the state.
- **Few fields per idr0139 well.** With 4 fields, clustering can hardly be
  detected, and the fields needed carry wide bounds.

## 11. Which test for a condition: repeat means or cells? (2026-09-30)

`data/simulations/test_choice.py` (results in `data/simulations/results/test_choice.csv`)
simulates the lab's usual design: 1-6 biological repeats, one well per condition per repeat,
30 cells a well. A batch shift is shared by both wells of a repeat, and the difference varies
between repeats by `s_rep` cell SDs (0, 0.25, 0.5; the lab datasets estimate 0.3-0.5). There are
1000 experiments per setting and alpha is 0.05.

- **False positives without an effect.** The paired t on repeat means, raw or on log ratios, stays
  at 4-6% everywhere.
- The cell-level tests are inflated once repeats differ. Normalised to each repeat's control,
  they give 15-17% at `s_rep` 0.25 and 32-37% at 0.5, which matches
  `2·P(Z > 1.96/√(1 + n·s²/2))` within a few points; that formula labels the cell-level test in
  question 9. On raw values (Plot&Stats, pooled) the rate is at or below it: 5-17% at 0.25 and
  19-36% at 0.5, falling with more repeats as the shared batch shifts add noise.
- Normalising removes the shift that both wells of a repeat share, but not the pseudo-replication.
- The mixed model with a z reference is inflated at 2-3 repeats; with a t reference it almost
  never rejects at 2-3 repeats and has less power than the paired t. The hierarchical bootstrap is inflated with few repeats (22%, 12%, 5% at
  `s_rep` 0.5 over 2, 3, 4 repeats). The sign-flip test cannot reject below 6 repeats.
- **Power of the paired t** (`s_rep` 0 / 0.25 / 0.5):

  | repeats | d 0.5 | d 1.0 | d 1.5 | d 2.0 |
  | --- | --- | --- | --- | --- |
  | 2 | 17 / 11 / 9% | 31 / 23 / 15% | 41 / 33 / 22% | 52 / 40 / 30% |
  | 3 | 41 / 27 / 18% | 85 / 67 / 39% | 98 / 90 / 63% | 100 / 98 / 83% |
  | 4 | 74 / 47 / 24% | 100 / 93 / 67% | 100 / 100 / 92% | 100 / 100 / 99% |
  | 6 | 96 / 77 / 44% | 100 / 100 / 92% | 100 / 100 / 100% | 100 / 100 / 100% |

- **Reading.** With 3 repeats, the paired t finds the effects the lab usually reports (d ≥ 1.5)
  while keeping false positives at the nominal 5%. The cell-level test responds with 1-2 repeats,
  where the paired t cannot, but its p is not a 5% test once repeats differ. So question 9 gives
  both: the paired t across repeats, and the cell-level test labelled with its chance of a
  positive without an effect.
- **Unequal cells per repeat** are simulated in entry 12.

## 12. Unbalanced repeats: resampling, normalising, and the false-positive label (2026-10-01)

`data/simulations/unbalanced.py` (results in `data/simulations/results/unbalanced.csv`) uses
the generator of entry 11 with 3 biological repeats that hold different numbers of cells. There
are 1000 experiments per setting, with no effect or d 1.0, and repeats differing by `s_rep` 0,
0.25 or 0.5 cell SD. It compares Plot&Stats as provided, the package's balanced test, and the
paired t on repeat means, each raw and normalised to the same repeat's control. Each balanced
draw uses the cell t-test p instead of the randomization p; in entry 11 the two differ by at most
1 point.

False positives without an effect, normalised, at `s_rep` 0.25 (0.5):

| cells per repeat | as provided | balanced | paired t | label (as provided) |
| --- | --- | --- | --- | --- |
| 30 / 30 / 30 | 16% (37%) | 17% (36%) | 5% (6%) | 16% (37%) |
| 60 / 30 / 30 | 22% (45%) | 14% (35%) | 4% (5%) | 21% (45%) |
| 120 / 30 / 30 | 31% (58%) | 12% (32%) | 5% (5%) | 32% (58%) |
| 240 / 30 / 30 | 46% (71%) | 12% (32%) | 5% (5%) | 46% (70%) |
| 30 / 30 / 30, control also from 3 other repeats | 22% (44%) | 16% (36%) | 6% (6%) | 16% (37%) |
| 100 / 100 / 8 | 32% (58%) | 32% (59%) | 5% (5%) | 33% (59%) |

- **One repeat holding more cells** makes the test as provided worse, because that repeat weighs
  more. The label matches this within 1-2 points once the cells per well are weighted by each
  repeat's share (`sum(n²) / sum(n)`); the report now uses it. Balancing brings the rate back to
  the equal-repeats level or below; the median p over draws is a little conservative.
- **A control pooled from repeats the condition was not imaged in** is the worst case on raw
  values. Batch shifts between repeats no longer cancel, so the raw test gives 46% false
  positives even when the repeats agree perfectly (`s_rep` 0), and it shrinks d (0.73 for 1.0).
  Normalising cuts that to 9% (22% at 0.25). The balanced test, against the control of the
  condition's own repeats, gives 2% raw and 4% normalised, as with equal repeats.
- **A very small repeat** (8 cells against 100) is left out of the balanced test. The result
  matches two repeats of 100, as the label at 100 cells says. The paired t keeps it as one repeat
  and stays at 2-5%, at some cost in power (77% against 88% at d 1.0 and `s_rep` 0).
- **The small-repeat rule** now compares a repeat with the median repeat, not the largest. With
  240/30/30 cells, the largest-repeat rule called both ordinary repeats too small and balanced on
  the dominant repeat alone (its false-positive rate then that of one repeat of 240).
- **The paired t** is at 2-6% in every scenario: one value per repeat makes it insensitive to the
  cells each repeat holds.
- **Power** (d 1.0) is 96-100% for every cell-level version except the raw test with the control
  from other repeats (88-90%), and 37-93% for the paired t, as in entry 11.
