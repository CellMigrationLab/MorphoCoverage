# Preliminary findings

These are exploratory, retrospective observations made while building the
framework. They are not benchmark results and not publication claims. Each
entry states what was run, what it shows, and what it does not show.

**What the numbers are from.** Every tracked table and number below comes
from the 2026-09-26 run of `data/crick_idr/run_all.py` (run times in
`data/crick_idr/results/logs/run_all.txt`), with the benchmark under protocol
version 3 (`data/crick_idr/benchmark_protocol.py`).

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
| idr0139, 308 wells, 4 FOVs/well | 1 | 329 | 4.2% | 1.3% | 3.2 [2.0–4.2] | 87% |
| | 2 | 657 | 2.4% | 0.74% | 3.2 [2.0–4.3] | 87% |
| | 3 | 987 | 1.3% | 0.42% | 3.1 [2.0–4.3] | 87% |
| idr0028 B, 1,144 wells, 28 FOVs/well | 1 | 33 | 1.8% | 1.2% | 1.5 [1.3–1.7] | 49% |
| | 4 | 132 | 0.82% | 0.54% | 1.5 [1.4–1.7] | 53% |
| | 8 | 264 | 0.53% | 0.34% | 1.5 [1.3–1.7] | 53% |
| | 14 | 462 | 0.33% | 0.22% | 1.5 [1.3–1.7] | 54% |

Errors are relative RMSE of the well mean, as medians across wells. Every number in this entry is taken from the tracked outputs `data/crick_idr/results/fov_vs_objects_idr0139_fascin.csv` and `fov_vs_objects_idr0028_screenB_yaptaz.csv`, generated from Crick commit `a81d20c` with checksum-verified inputs. The complete plates are stored in the repository (`idr0139_plate`, `idr0028_screenB_plates`), and the tables are regenerated from them.

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
`epsilon`. Candidates:

- the host's remaining FOVs (nothing new by construction: the false-alarm rate);
- every FOV of a knockdown well on the same plate (siYAP, siLATS1 or siARAP2);
- a remaining host FOV with 3, 5 or 10 cells of a knockdown FOV added.

Rates at `epsilon = sqrt(30)` and `m = 3`, pooled over the three plates and
both knockdown wells of each (tracked in
`data/crick_idr/results/positive_control_idr0028.csv`):

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
| mean within 2.5% | 83–100% (median 92%) | 92–100% (98%) | 98–100% | 100% |
| variance within 10% | 15–30% (21%) | 17–42% (26%) | 18–64% (32%) | 10–80% (45%) |

Each cell gives the range across the 12 wells, with the median in brackets.

**What this shows**

- **For the well mean, two fields are almost always enough.** At a 5%
  tolerance, 2 fields matched the held-back fields in at least 99% of draws in
  every well. At 2.5%, 2 fields were enough in 83–100% of draws (worst:
  ARAP2 2B C17, YAP 2A N23 and MOCK 2A K16), 4 fields in at least 92%, and 8 in
  at least 98%. The median error of a 2-field mean was 1%. With 28 FOVs per well, this
  mean was heavily oversampled.
- **The variance within 10% cannot be judged with this design.** Even all 18
  fields matched the 10 held-back fields in only 45% of splits (median; 10–80%
  across wells). The held-back fields are too few to pin the variance down to
  10%. The curve is flat because it is capped by that ceiling, not because
  more fields do not help: the median error falls from 0.26 at 2 fields to
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

**Method** (`data/crick_idr/stopping_idr0028.py`, fixed seed; 50 s in the 2026-09-26 run, `results/logs/run_all.txt`). The data
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

| rule | stop (FOVs) | premature, mean within 5% (random orders; median, max over wells) | premature, mean within 2.5% (random orders; median, max over wells) | premature, 2.5%, recorded order (max over wells, of 5) |
|---|---|---|---|---|
| JCB CV ≤ 5%, cells | 2 | 0%, 0% | 9%, 16% | 40% |
| JCB CV ≤ 5%, FOVs | 2 | 0%, 0% | 9%, 16% | 40% |
| fixed 4 FOVs | 4 | 0%, 0% | 2%, 12% | 60% |
| fixed 8 FOVs | 8 | 0%, 0% | 0%, 2% | 0% |

Judged on held-back fields, the mean was first within both tolerances at a
median of 1 FOV (all 660 replayed acquisitions: 1–11), and lastingly from a
median of 1 (1–11; 89% by 2 FOVs). The random orders are drawn per well,
from the replay's seed.

**What this shows**

- **Both JCB-style rules stopped as early as they were allowed (2 FOVs) in
  every well.** With about a hundred nuclei per field, the averaging CV of the
  mean ratio is below 5% after two fields, whether cells or FOVs are the
  resampling unit. For the mean within 5%, which is the target the rule was
  designed for, that stop was always right here (none of the random orders
  and none of the recorded-order hold-outs was premature). For the stricter
  2.5%, it was premature in a median 9% of random acquisitions, up to 16% in
  the worst well. A rule at 5% CV answers the 5% question; it is
  not a promise for 2.5%.
- **The mean was never the hard part here.** Held-back fields confirmed it
  after one or two fields in most runs, and never later than 11. This is the
  regime AGENTS.md section 10 asks us to name: for the mean nuclear YAP/TAZ
  ratio of a well in this screen, the cell-number guidance is sufficient, and
  FOV-aware sampling adds nothing for this readout.
- **Fixed rules.** Four fields were enough for 5% in every run, and for 2.5% in
  all but up to 12% of random acquisitions of the worst well (LATS1 1A J23);
  in the recorded order of one well (YAP 1A M23) they were premature in 3 of
  5 hold-outs. Eight fields were premature in at most 2%.

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

**Method** (`data/crick_idr/report_steps_idr0028.py`, fixed seeds; 102 s in the
2026-09-26 run, `results/logs/run_all.txt`; run after the P1 curves of entry
4). Every table is in `data/crick_idr/results/`,
and the assembled report is `microscopist_report_idr0028.txt`. idr0028 remains a
development dataset.

**What this shows**

- **Protocol (P2).** A well counts as adequately represented when at least
  90% of random acquisitions preserved the criterion, and all conditions are
  pooled explicitly.
  - For the well mean within 5%, 2 fields were enough for 12 of 12 wells.
  - Within 2.5%, 2 fields were enough for 7 of 12 wells (by plate: 3 of 5, 2
    of 4, 2 of 3), and 4 fields for all 12.
  - A number chosen on two plates served every well of the third.
  - For the variance within 10%, no well was adequately represented at any
    depth (the reservoir ceiling of entry 4).
  - With 3 plates the summary says the uncertainty is too large to fix a
    protocol, and it is.
- **Fields to inspect (R5).**
  - Two fields hold far fewer nuclei than their well (6 against a median of
    27 and 21.5).
  - One siYAP field has a mean YAP/TAZ ratio 5.6 SD below its four stage
    neighbours, and one siARAP2 field 3.1 SD above. The SD is that of the
    other fields, not including the field itself.
  - Four fields hold a small new state: 3 nuclei no other field of the well
    shows, in 30 standardized features. Judged as the inspection judges a
    field, against every other field of the well, none of 56 MOCK reference
    fields was called new, so the report expects about 0 of 336 by chance:
    the four fields are not at the chance level (the rate is measured leave
    one field out, as the inspection judges a field).
  - No field moved the well mean or 95th percentile of the ratio by 10% or
    more when left out, as expected with 28 fields per well.
- **Representation control (R1).** Measured after 10 MOCK fields on the
  same plate:
  - siYAP fields were recognized as different in 100% of 1680 checks;
  - MOCK fields with 5 siYAP nuclei planted were recognized in 100% of 720;
  - the other MOCK fields were called different in 2% of 720 (false alarms).

  This matches the tracked positive control (entry 2: 100% for siYAP, about
  3% false alarms at 10 fields) computed by independent code.
- **More fields or more wells (P10).** For the mean ratio of a condition:
  - in MOCK, 54% of the variance of the mean is between fields within a well;
  - in the knockdowns, 79–99% is between wells, which are on different plates;
  - fields vary 1.6–2.6 times as much as random cells would (design effect,
    with the cluster size of unequal fields).

  With 3 wells per condition the answer is "cannot tell from these data", as
  designed. Plate and well are confounded here: in MOCK each plate has one
  well. The analysis therefore refuses a plate level and uses plate-qualified
  well and field identities.
- **Order (P8).** For the well mean within 2.5%, recorded, random and
  dispersed orders all reached it after a median of 1 field. The recorded
  serpentine raster needed 2 fields (805–1114 µm of travel) in 3 wells and 5
  fields (2.0 mm) in YAP 1A M23, where random and dispersed orders needed 1;
  random orders needed 2 fields (444–558 µm) in 2 other wells, and dispersed
  orders 2 fields in 2 wells. The random orders are drawn per well. Outside
  YAP 1A M23 the orders differ by at most one field, a permitted null result
  (AGENTS.md section 3): the mean is too easy here to tell orders apart. One well where
  the raster is 4 fields slower is not a pattern across 12.
  - In 5 of 12 wells, fields closer than about 520–780 µm are more alike than
    random pairs.
  - The other 7 wells show no such redundancy down to the closest spacing
    measured (336 µm).
- **Objects or fields (the Foundation).** The report opens with the
  Foundation: per well, fields vary 1.0–3.4 times as much as random nuclei
  would (median 2.1; `fov_variability`), and unusual nuclei cluster by field
  in 5 of the 11 wells where they were seen (entry 10).

**What this does not show**

- **One easy readout, 12 wells, 3 plates, development data.** None of these
  numbers is a protocol. The P10 and P2 answers are limited by 3 wells
  per condition and 3 plates, and they say so.
- **The P10 intervals are approximate.** Simulated nested designs with 8
  samples recover the variance components on average (tested), but the
  bootstrap interval covered the true sample and field components in only
  88% and 80% of 40 runs, not 95%. Resampling only the samples was worse (68%
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
| mean within 5%, share of draws enough (median over wells; range) | 65% (8–100%) | 100% (17–100%) | 100% (17–100%) |
| mean within 2.5% | 47% (0–100%) | 50% (0–100%) | 50% (0–100%) |
| wells adequately represented (≥ 90% of draws), within 5% | 2 of 10 | 6 of 10 | |
| wells adequately represented, within 2.5% | 1 of 10 | 2 of 10 | |

| rule | stop | premature, mean within 5% (median; max over wells) | premature, 2.5% |
|---|---|---|---|
| JCB CV ≤ 5% or ≤ 2.5%, cells | 1 field | 32%; 90% | 47%; 100% |
| fixed 2 fields | 2 fields | 0%; 83% | 50%; 100% |

Premature shares are over the random orders, drawn per well.

**What this shows**

- **The development conclusion does not transfer as a number.** On idr0028,
  2 fields were enough for the mean within 5% in 12 of 12 wells; on idr0139,
  2 fields are enough in 6 of 10 wells, and 1 field (where the JCB rule stops,
  since one field holds about 330 nuclei) in 2 of 10. The JCB stop is premature
  at 5% in a median 32% of replayed acquisitions and up to 90% in the worst
  well. This is the held-out counterpart of entry 1's observation that with
  the actual single field 35% of wells miss 5%.
- **What transfers is the regime, not the count.** idr0139 has about 3 times
  the FOV-level error that random cells predict (entry 1), idr0028 about 1.5
  times; the cell-number rule is sufficient where fields vary little and
  insufficient where they vary more. A protocol must be set per readout and
  per preparation, not carried over.
- **The report says what it can and cannot judge.** The 2-field reference is
  itself noisy: the ceiling is 100% (median; 17% in the worst well) at 5% and
  50% at 2.5%, and the
  protocol summary names the wells whose reference could not judge the
  criterion rather than listing them as needing more fields. So the shares
  above are lower bounds on nothing and upper bounds on nothing: with 4
  fields per well the dataset can refute "one or two fields suffice" but
  cannot say how many do.

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

**Method.** `data/crick_idr/benchmark.py` under protocol version 3
(`benchmark_protocol.py`, locked by digest `9cceed31` on 2026-09-26 before
its run). Its settings, grids and seeds are those of version 1, locked
before the first run; the protocol file records what each version changed.
Permuted orders are drawn per well from `random_state`, so one well's order
does not depend on the others; the rules see each prefix in acquisition
order; and the per-field precision rule counts only fields with objects in
its variance factor and degrees of freedom. Replays hold back 35% of each idr0028 well (5 splits, recorded and 10 random
orders; 50% of each idr0139 well, 6 splits, recorded and 5 random orders).
Each rule family is swept over a threshold grid fixed in the protocol; risk
is the share of acquisitions that did not preserve the outcome in held-back
fields when the rule stopped; the unit is the well. Before locking, a
reduced dry run (one split, one order, not scored) showed several grids
ending before the cheap end; they were widened under a stated rule (every
grid reaches from never stopping to stopping at the first allowed field), not
on a result. Tables: `results/benchmark_*`.

Fields saved against a fixed count, idr0028 (negative: costs fields):

| rule | mean within 5%, risk ≤ 5% | mean within 2.5%, risk ≤ 5% | mean within 2.5%, risk ≤ 10% |
|---|---|---|---|
| fixed count (fields needed) | 1 | 3 | 2 |
| JCB per object | 0.0 | −5.4 | −0.9 |
| JCB per field | −1.3 | −0.5 | −0.3 |
| JCS spread per object | 0.0 | −1.1 | −2.1 |
| JCS spread per field | −1.0 | −7.5 | 0.0 |
| FOV bootstrap of the mean | −1.0 | −10.4 | 0.0 |

The 95% intervals over the 12 wells are wide. For example, JCB per field at
2.5% has −0.6 to 0.7 fields saved. At the 2.5% criterion every cost's
interval includes 0, so the summary marks each one "(not resolved)". Every
adaptivity correlation is 0.35 or less. At 2.5% and 5% risk the fixed count
needs 3 fields, and every rule's cost there is counted from 3.

Settings chosen on idr0028 to meet 5% risk, scored on the held-out idr0139:

| mean within 5% | stop on idr0139 | risk on idr0139 |
|---|---|---|
| fixed 1 field, JCB per object, JCS spread per object | 1 field | 35% |
| JCB per field, JCS spread per field, FOV bootstrap | 2 fields (all available) | 22% |

For the mean within 2.5%, every setting reaches 58% risk on idr0139.

Planted rare state (semi-synthetic: 5 siYAP nuclei in 10% or 20% of the
fields of the MOCK wells of plates 1A and 2A, 48 acquisitions):

| rule | risk ≤ 5% | risk ≤ 10% |
|---|---|---|
| fixed count (fields needed) | 15 | 12 |
| novelty bound | −3.0 | −2.9 |
| rare incidence, reading the planted label | +8.5 (Spearman 0.66) | +5.5 (Spearman 0.66) |

**What this shows**

- **For the well mean on idr0028, nothing beats a fixed count.** This is the
  null result the regime map needs: the readout is easy, one field meets the
  mean within 5%, and every rule either matches that or costs fields.
- **Per cell against per field changes where each rule fails.** At 5% the
  per-object rules match a fixed single field, and the per-field rules cost one
  field because they need two fields to estimate field-to-field variance. At
  2.5% it reverses: the per-object JCB rule (upper-95 SD) costs 5.4 fields,
  JCB per field only 0.5, because object counts overstate what the fields add.
  Neither cost is resolved over 12 wells.
- **No development setting transfers to the held-out wells.** At the setting
  chosen for 5% risk on idr0028, idr0139 gives 22–35% risk. Part of that is
  idr0139 itself: its 2-field reference cannot confirm the mean within 5% in
  22% of hold-outs even with every field (entry 7). Stopping at one field
  (fixed, JCB per object, JCS spread per object) is worse than that floor;
  stopping at two is exactly at it. Entry 7's conclusion holds: the count
  does not transfer.
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
  object of the negative-control wells, as the paper's supplementary
  spreadsheet and R code do, not to the mean of the control wells' means.
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
| real / predicted error for the objects obtained | 2.2 (0.7–3.9) | 2.8 (0.3–6.6) | 1.4 (0.9–1.8) | 1.5 (0.8–14.4) |
| real error of that average | 3.1% | 4.1% | 1.9% | 1.7% |
| wells missing 5% | 2 of 10 | 97 of 308 (31%) | 0 of 12 | 17 of 1,144 (1%) |
| wells missing 2.5% | 8 of 10 | 233 of 308 (76%) | 0 of 12 | 77 of 1,144 (7%) |
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

JCS Fig. 2A at the paper's threshold, whole-field spread divided by
random-object spread, medians over wells (`results/objects_vs_fields_jcs.csv`):

| descriptor | idr0139, 250 objects | idr0028, 200 objects | idr0028: objects fields need for the same spread |
|---|---|---|---|
| mean | 1.4 (0.8–3.2) | 1.2 (0.6–2.0) | 320 (11 of 12 wells) |
| median | 1.6 (0.6–2.9) | 1.3 (0.9–2.0) | 345 |
| IQR | 1.2 (0.8–2.3) | 1.0 (0.7–1.3) | 245 |
| SD | 1.0 (0.5–2.1) | 1.1 (0.7–1.5) | 240 |

On idr0139, field samples reached the 250-object spread of the mean within
500 objects in only 6 of 10 wells.

**Follow-up: the paper's own treatment comparisons, fields against objects**
(`data/crick_idr/jcs_effect_fovs.py`). The same effect-size analysis as JCS
Fig. 3G–I and S3G–I (their Eqn 2, their wells from the companion notebooks,
1000 samples per size) was run both ways. It compares wells, so it stays a
benchmark script (`docs/questions.md`, open work). Share of samples that rank
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
  holds about 30 objects, so one field meets both JCB targets in 93–99% of
  all 1,144 wells, 200 objects means about 8 fields, and every treatment
  comparison survives. On idr0139 a field holds about 270 objects: the
  JCB-sized sample misses the paired-design 2.5% target in 76% of the plate's
  308 wells, and the paper's SN0212398523 > DMSO reading at 250 objects comes
  out right in about half of the whole-field samples, a coin flip.
- **The JCB formula still works when the field is the unit.** With the SD of
  field means it asks for 2 to 4 fields at 2.5% on idr0139. With 4 fields,
  however, that SD is so uncertain that the upper-95 bound the paper
  recommends for small pilots asks for 30 to 50 fields. A 4-field pilot cannot
  say how many fields are needed.
- **The JCS stabilization depends on how the objects arrive.** Location
  descriptors (mean, median) vary more between field samples than between
  random-object samples; spread descriptors (SD, IQR) barely do. On idr0028,
  fields need about 240–345 objects to be as stable as 200 random objects.
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
  gives 1.4. Read the ratios as "above 1 for location descriptors",
  not to one decimal.
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
  reach against reference objects of other fields. Each well is scored
  against the control wells other than itself. The design follows
  fiberfeatures' pixel-category fit (reference sampling, provenance, a seed
  check), without its scikit-learn Gaussian mixture: rare states are what a
  mixture tends to absorb, and MorphoCoverage has no scikit-learn dependency.
- **Clustering**: per well, the fields that hold unusual objects against the
  fields expected if the same objects were spread at random over the same
  fields (hypergeometric), a permutation test (999 shuffles of the labels
  over the well's objects), and the fields needed to see the state once with
  95% chance: from its field incidence, and as a random-object argument would
  say (objects needed from the prevalence, divided by objects per field).

Results (`results/objects_vs_fields_rare.csv`, `_unusual.csv`):

| | idr0139, 10 wells | idr0028, 12 wells |
|---|---|---|
| unusual objects per well | 1.5% (0.3–2.6%) | 1.1% (0–2.5%) |
| agreement of refits with other reference samples | 50% (42–73%) | 100% |
| wells where unusual objects cluster by field (p ≤ 0.05) | 3 of 10 | 5 of 12 |
| in those wells, fields needed / fields an object count asks for | 3.0 (2.5–3.0) | 1.9 (1.5–2.2) |

On idr0028, the clustered wells are MOCK 1A B6, LATS1 1A I23, MOCK 2A K16,
YAP siRNA 2A N23 and ARAP2 2B C13. In YAP siRNA 2A N23, 7 unusual objects sit
in 3 of 28 fields (71% in one field), where random objects would fill 6.
Seeing them with 95% chance takes 27 fields from their field incidence (up to
131 at the 95% bound, the low end of the two-sided interval), against 12 if the
objects were random.

**What this shows**

- **Rare objects can cluster by field in real data, and then an object count
  under-asks.** In 5 of 12 idr0028 wells, unusual objects occupy fewer fields
  than random objects would. There, seeing them reliably takes about twice
  the fields that the object-count argument predicts. That is the rare-state
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
  42–73% of unusual objects there, and reseeding the groups alone moves the
  clustered wells between 2 and 3 of 10): its fields hold about 270 objects, of
  which 100 are sampled. The idr0139 clustering results should not be read
  beyond "some wells cluster". idr0028 fields hold fewer than 100 objects, so
  every reference object is used and the labels do not depend on the seed.
- **Multiple wells, one test each.** With 12 wells at p ≤ 0.05, about 0.6
  would be flagged by chance; 5 were.
- **Few fields per idr0139 well.** With 4 fields, clustering can hardly be
  detected, and the fields needed carry wide bounds.
