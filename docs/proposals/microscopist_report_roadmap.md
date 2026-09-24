# Roadmap: implementing the microscopist's report

**Status:** for review (2026-09-24), revised after review the same day.
Implements `docs/proposals/microscopist_report.md`, which gives the reasons.
This roadmap gives the order, the deliverables, the definitions and the
acceptance criteria. Nothing here is implemented yet.

**How to review.** Each milestone lists what it builds, how it is validated,
and what must be decided first. The decisions that need the project owner
(AGENTS.md section 15) are collected in the last section. A milestone is done
only when its acceptance criteria pass, its limits are written down, and its
real-data result is recorded in `docs/preliminary_findings.md`, including a
null or negative result.

## Principles every milestone follows

1. **Retrospective representability is not the prospective sampling
   requirement** (AGENTS.md section 3). No best or minimal subset is reported
   as a recommendation, and optimal-subset search is not on this roadmap.
2. **What an estimator says is not what the sample achieved.** Planning
   numbers come from strategy-independent outcome criteria on FOVs no
   estimator saw.
3. **"How many FOVs?" and "does this stopping rule work?" are different
   questions.** The first is a strategy-independent sampling-reliability
   analysis of random whole-FOV sets; the second is a sequential calibration
   of a stopping rule along acquisition orders. They are separate analyses
   with separate outputs.
4. **Every answer is conditional on the representation**, and
   representation checks apply only to declared controls.
5. **Protocol recommendations are validated across held-out samples or
   repeats**, and their uncertainty counts independent units, never subset
   draws or reservoir splits.
6. **Existing contracts hold:** FOV is the resampling unit; groups are never
   pooled silently; features are explicit; zero-observation FOVs count as
   effort; random procedures take seeds; no future-FOV leakage in anything
   prospective.

## Microscopist first

The reason for every step is a decision a microscopist makes. Four rules make
that concrete, and each step below is written against them:

1. **Every step ships an answer, not only a table.** Each step defines the
   sentences the report prints, the decision they support, and adds them to
   `sufficiency_report(...).summary()` when the step lands. Nothing waits for
   the final assembly.
2. **Plain words in everything the user reads.** The report says "fields",
   "held-back fields", "what you wanted to preserve", "how often it would have
   worked"; it never says reservoir, criterion, prefix, outcome metric or
   `config_id` except in the detailed tables. Every number in a sentence comes
   with its unit (FOVs, % of acquisitions, µm, minutes).
3. **Only scientific choices are asked of the user.** What to preserve (the
   readout, the distribution, rare states) and how closely (for example "the
   well mean within 5%") are asked in words, with provisional presets.
   Computational settings (outer splits, draws, seeds) have documented
   defaults sized for a result in minutes, and the report says how stable the
   answer is across them.
4. **A reading test is part of acceptance.** On a real dataset, the project
   owner reads the step's output without the code and says what they would do
   differently. If they cannot, the step is not done.

The example outputs in each step are illustrative wording, not results.

Steps 1, 3, 4, 5, 6, 7 and 8 answer a microscopist's question directly. Step 2
(stopping-rule calibration) answers the question of someone choosing or
building an automated stopping rule for smart microscopy; its answer is still
worded for the microscopist ("this rule would have stopped too early in 7% of
acquisitions").

## Architecture

Three boundaries, kept apart:

```text
prospective               Estimator -> Evidence -> DecisionPolicy -> Decision
retrospective scoring     OutcomeMetric -> OutcomeCriterion -> preserved / not / undefined
retrospective analyses    reliability, calibration, adequacy, protocol, inspection,
                          controls, effort, spatial efficiency
```

`OutcomeCriterion` is deliberately not a `DecisionPolicy`: a policy acts
prospectively on an estimator's evidence; a criterion acts retrospectively on
an outcome measured against hidden FOVs. The retrospective analyses live in a
subpackage, not in `sufficiency.py`, which becomes one consumer of them:

```text
morphocoverage/
    outcomes.py, multivariate_outcomes.py      (existing metrics)
    retrospective/
        criteria.py         OutcomeCriterion
        hierarchy.py        SamplingHierarchy
        reliability.py      sampling reliability and information gain
        calibration.py      stopping-rule calibration (the existing next-FOV
                            novelty check moves or is re-exported here)
        adequacy.py         apparent and realized adequacy points
        protocol.py         protocol-level reliability
        inspection.py       inspect-these-fields report
        controls.py         representation controls
        effort.py           scalar hierarchical effort projection
        spatial.py          acquisition-order efficiency and stage travel
    sufficiency.py          report assembly
```

## Data this roadmap needs

| Need | Why | Have now |
|---|---|---|
| Samples with many FOVs (≥ 30, ideally ≥ 60) | a reservoir plus a stream long enough for the curves | idr0028: 12 wells × 28 FOVs (curves stop at about 18 FOVs once a reservoir is held out) |
| Many samples, several biological repeats | protocol curves and held-out validation | idr0028: 12 wells on 3 plates; idr0139: 4 FOVs per well |
| Stage coordinates | spatial efficiency and stage travel | idr0028: yes |
| Declared positive and reference controls | question 1 | idr0028: knockdown wells (positive) and MOCK wells (reference) |
| Candidate positions from an overview | sampling-frame bias | none |

Step 0 therefore includes registering at least one deeper dataset, for
example the tiled images the project owner can provide.

## Step 0 — Outcome criteria and the hierarchy contract

**Deliverables**

- `OutcomeCriterion(metric: OutcomeMetric, threshold: float, direction:
  "lower_is_better" | "higher_is_better")`, with `assess(reference, subset) ->
  "preserved" | "not_preserved" | "undefined"` and provenance (the metric's
  `config_id`, the threshold and the direction). An undefined metric value
  (`None`) is never counted as preserved. Every retrospective analysis uses
  this one contract, so threshold logic exists once.
- `SamplingHierarchy(fov: str, sample: str, biological_repeat: str | None,
  technical_levels: tuple[str, ...] = ())`, declaring which columns are the
  FOV, the sample and the independent unit, with validation of nesting (each
  FOV in one sample, each sample in one repeat). Never inferred from column
  names.
- Registration of the deeper dataset(s), with development and held-out roles
  fixed before any curve is computed.

**What the microscopist gets.** The ability to start from what they have: a
CellProfiler-style export. `fov_table_from_observations(...)` builds the FOV
table from an observation table when no acquisition record exists, and says
loudly that FOVs without cells are then missing and effort is undercounted.
The report asks, in words, what matters: "a reliable average of <readout>",
"the full distribution", "rare states", each mapped to criteria with
provisional thresholds.

**Validation:** threshold direction; undefined outcomes; each nesting
violation raises a clear error; both types are JSON-serializable.

## Step 1 — Sampling reliability and information gain

**Question:** *if I acquire `k` representative FOVs, how often does the result
preserve what I care about in unseen FOVs, and what did each extra FOV buy?*
No stopping strategy is involved.

**Design**

1. For each sample and outer split `s = 1..S`, hold out a fixed reservoir of
   whole FOVs or blocks (`ReservoirSplit`); the rest is the stream.
2. For each `k` (at most the stream length) and draw `d = 1..D`, draw `k`
   stream FOVs at random.
3. Assess each `OutcomeCriterion` of the `k` FOVs against the reservoir, and
   keep the metric values.
4. Aggregate within the sample: over draws, then over splits.

**Deliverable:** `sampling_reliability(objects, fovs, *, feature_cols,
criteria, fov_counts, reservoir, n_splits, draws, random_state, ...)`. One row
per sample × criterion × `k`:

| column | meaning |
|---|---|
| `fovs` | `k` |
| `preserved` | share of draws in which the criterion was preserved (the sample's conditional reliability at `k`) |
| `preserved_split_sd` | spread of that share across outer splits |
| `metric_median`, `metric_p10`, `metric_p90` | the outcome value itself |
| `gain` | information gain, `L(k-1) - L(k)` of the median outcome: what the next FOV bought |
| `undefined` | draws whose outcome was undefined |
| `reservoir_fovs`, `stream_fovs` | the stated limit on `k` |

A secondary mode scores against the sample's remaining FOVs, labelled as a
changing-reference cross-validation for small datasets.

**What the microscopist sees.**

```text
How many fields do you need? (what you wanted: the well mean of YAP/TAZ ratio within 5%)
  10 fields   worked in 62% of random acquisitions
  15 fields   worked in 84%
  20 fields   worked in 93%     <- each further field changes the answer by < 0.5%
  (judged on 10 held-back fields; stable across 10 different hold-outs)
```

Decision supported: how many fields to plan for a sample like this one. A
first notebook cell on idr0028 shows it.

**Validation**

- *Contracts on constructed scenarios:* whole FOVs only; zero-observation FOVs
  drawable; group independence; determinism under a seed; every `k` in a split
  uses the same reservoir.
- *Known truth:* homogeneous FOVs, where `preserved` rises with `k` toward 1
  and `gain` shrinks; a clustered rare state, where the rare criterion stays
  low until `k` covers the cluster.
- *Real data:* idr0028 with `RelativeMeanError` of the YAP/TAZ ratio at 5% and a
  distribution criterion.

**Acceptance:** those tests; runtime on idr0028 recorded (12 samples, `S = 10`,
`D = 50`, 6 values of `k`) and under 10 minutes.

## Step 2 — Stopping-rule calibration

**Question:** *when this strategy says stop, how often is that premature?*
Sequential, because stopping rules can depend on order: a random set of 20
FOVs does not reproduce whether a rule would already have stopped at 14.

**Design:** on the stream of each fixed reservoir, replay acquisition orders
(recorded, permuted, later dispersed). At the strategy's first stop, assess
the criteria of the prefix against the reservoir.

**Deliverable:** `stopping_calibration(...)`, per sample, strategy and
criterion:

```text
Strategy stopped:   91% of runs      (median at 17 FOVs)
Correct stops:      84%              (criterion preserved at the stop)
Premature stops:     7%              (stopped, not preserved)
Never stopped:       9%              (and whether the criterion was preserved at the end)
```

Each row reports the FOVs at the stop and at the end of the stream, so a rule
that stops late is visible as well as one that stops early. The existing
next-FOV novelty calibration (`morphocoverage.calibration`) is a special case
and joins this module.

**What the microscopist sees.** "If you had let <rule> stop the acquisition, it
would have stopped after a median of 17 fields; 7% of those stops were too
early for what you wanted to preserve, and in 9% it never stopped." Decision
supported: whether to trust an automated stopping rule on the microscope.

**Validation:** a rule that always stops at the first FOV (every stop
premature unless the sample is homogeneous); a rule that never stops (never
stopped = 100%); a fixed-FOV rule, whose calibration must match Step 1 at the
same `k` for random orders (the one case where the two analyses coincide).

## Step 3 — Apparent and realized adequacy

**Question:** *when would a live acquisition have stopped, when did the
criterion settle, and when were the unseen FOVs actually preserved, and stayed
preserved?*

**Definitions (the contract)**

- `first_apparent_adequacy`: the first evaluated prefix whose decision is stop.
- `sustained_apparent_adequacy`: the earliest prefix from which every
  subsequent evaluated decision is stop. An undefined or unsupported decision
  later breaks it, as a continue does. "Not reached" if the last prefix is not
  stop.
- `first_realized_adequacy`: the first prefix whose outcomes against the fixed
  reservoir preserve every declared criterion.
- `sustained_realized_adequacy`: the earliest prefix from which every
  subsequent evaluable prefix preserves every criterion. This is the preferred
  answer to "how many FOVs did this experiment really require?", since hidden-
  FOV outcomes can also cross and uncross (adequate at 10 and 11, not at 12,
  adequate again at 13).

The apparent points use only decisions; the realized points use only
outcomes. They are never mixed.

**What the microscopist sees.** "A live rule would have stopped at field 16, but
the data changed again at field 24; judged on held-back fields, what you
wanted was reliably preserved from field 25." Decision supported: how many of
the acquired fields were really needed, and whether an early stop would have
been a mistake.

**Validation:** stop at 16, stop to 23, continue at 24, stop from 25 gives first
apparent 16 and sustained apparent 25; an undefined decision at 30 moves
sustained apparent past 30; an outcome series preserved at 10–11, not at 12,
preserved from 13 gives first realized 10 and sustained realized 13; realized
points are identical for different strategies.

## Step 4 — Protocol-level reliability across samples

**Question:** *what FOV number should I put in the protocol for experiments
like this?*

**Deliverable:** `protocol_reliability(per_sample, hierarchy, *,
threshold, validation="held_out" | "leave_one_repeat_out")`. From Step 1, a
sample counts as adequately represented at `k` when its conditional
reliability reaches the declared threshold (for example `preserved >= 0.9`).
Reported:

- the share of samples adequately represented at each `k`, with explicit
  denominators ("11 of 12 eligible samples; 18 could not be evaluated at this
  depth");
- the same aggregated per biological repeat;
- the distribution of sample difficulty: each sample's smallest adequate `k`,
  shown as a distribution (for example easy 10–15, typical 18–24, difficult
  30+) and with the difficult samples named. A fixed protocol must serve the
  difficult tail; this is the case for adaptive acquisition.

**Uncertainty counts independent units.** Subset draws and reservoir splits
estimate each sample's conditional behaviour; they are not replicates.
Intervals on protocol-level shares and on generalization come from resampling
the declared independent unit (samples, or repeats when declared), with `n`
equal to the number of those units. A contract test checks that the interval
does not narrow when draws or splits are increased at fixed samples.

**What the microscopist sees.**

```text
For experiments like this one (12 samples, 3 plates):
  20 fields per sample: enough for 9 of 12 samples (plates: 3 of 3, 3 of 4, 3 of 5)
  25 fields: enough for 11 of 12; one sample (<plate, well>) needs more than the fields acquired
  Uncertainty is large with 3 plates: repeat on more plates before fixing a protocol.
```

Decision supported: the FOV number to write into the protocol, and whether a
fixed number is sensible at all or adaptive acquisition would save time.

**Validation:** that test; a constructed set with one very heterogeneous
sample, which is named; tuning on held-out samples raises
(`BenchmarkRegistry`); denominators always shown.

## Step 5 — Inspect-these-fields report

**Question:** *which FOVs should I go back and look at?*

**Deliverable:** `inspect_fovs(...)`: per FOV, a "reason for inspection" with
numbers and a way back to the image (image file, stage position, its number of
observations beside the sample median). Reasons, each computed independently
and reported only above a declared threshold:

- only FOV containing a new state (support-novelty clusters), with its cells;
- leave-one-FOV-out influence on declared summaries (relative change of the
  mean or chosen quantiles of named features);
- observation count far from the sample median;
- differs strongly from its spatial neighbours (with stage coordinates).

No combined score, and no FOV is called bad, an outlier or an artifact:
influential FOVs may be biologically crucial, technical artifacts, or simply
large.

**What the microscopist sees.** A short list of fields to reopen, each with the
image file, where it is on the stage, and why ("the only field with 8 cells of
a state seen nowhere else"; "removing it lowers the 95th percentile of cell
area by 18%"; "412 cells, the median is 90"). Decision supported: which images
to look at before trusting the result.

**Validation:** constructed scenarios where one FOV holds the only rare state,
one drives a tail quantile and one is dense; each is listed with the right
reason and no other is. Duplicating cells within a FOV does not raise its
influence beyond its FOV-level effect.

## Step 6 — Representation controls

**Question:** *could this representation recognize the phenotype I care about
if it appeared?* Can be developed in parallel with Steps 1–5, since the
positive and frequency controls are a prototype.

**Deliverable:** `RepresentationControl(positive=..., reference=...)` and
`representation_control_report(...)`. A positive control is whole FOVs of a
declared phenotype or cells planted into reference FOVs; a reference is the
declared negative population (for example MOCK). Reports:

- with both: "declared positive phenotype recognized in 94% of candidate FOVs;
  reference FOVs called different in 3%";
- with a positive control only: detection rate, "false-alarm rate: not
  assessed";
- with neither: "representation sensitivity: not assessed".

**What the microscopist sees.** "With these features, fields of the declared
positive control were recognized as different in 94% of cases, and reference
fields in 3%." Or: "not assessed: no control was declared, so a 'nothing new'
answer says nothing about phenotypes these features cannot see." Decision
supported: whether to trust the sampling answers for the phenotype they care
about, or to change the features first.

**Validation:** reproduces the tracked positive-control numbers from the same
seed; a positive control identical to the reference gives detection equal to
the false-alarm rate.

## Step 7 — Scalar hierarchical effort projection

**Question:** *for this measurement, would another FOV or another independent
sample reduce its sampling uncertainty more?* Scalar first; multivariate only
after validation.

**Deliverable:** for a declared scalar sample-level quantity (a mean nuclear
YAP/TAZ ratio, a mean cell area, a phenotype-state fraction) and a declared
`SamplingHierarchy` (observation, FOV, sample, biological repeat):

- variance components by nested method-of-moments (ANOVA) estimators for
  unbalanced designs, with intervals from a hierarchical bootstrap over the
  declared levels (numpy and scipy only);
- projected standard error under counterfactual allocations at equal total FOV
  acquisition (for example 6 × 40, 8 × 30, 10 × 24), each with an interval;
- text only when intervals separate; otherwise "cannot tell from these data".

**What the microscopist sees.** "For the mean YAP/TAZ ratio, most of the
remaining uncertainty is between samples: at the same total number of fields,
8 samples × 30 fields would give a 24% smaller standard error than 6 × 40."
Or: "cannot tell from 3 plates." Decision supported: whether to spend the
next experiment on more fields or on more samples.

**Validation:** simulated hierarchies with known components (including no
between-sample variance, and FOV variance dominating), recovered within their
intervals in about 95% of runs; projections match simulated designs;
unbalanced designs and zero-observation FOVs handled; with 3 repeats the
between-repeat interval is wide and nothing is recommended.

Later: a user cost model, `C = n_samples * C_sample + n_FOV * C_FOV`, to speak
of equal budget.

## Step 8 — Acquisition-order efficiency

**Question:** *could the same acquired locations have been visited in a better
order?* (Not whether they were the right locations; that is Step 9.)

**Deliverables**

- `AcquisitionOrder.dispersed(seed)`: greedy farthest-point order on stage
  coordinates from a starting FOV chosen by the seed. Compared over several
  seeds, like permuted orders, so results do not hinge on the first field.
- Stage travel of an order (summed Euclidean path between successive FOVs),
  and acquisition time when per-move and per-FOV times are given.
- Per sample: recorded, random and dispersed orders, each with the four
  adequacy points of Step 3 and stage travel.
- Spatial redundancy stated in µm from the semivariogram, as a limitation, not
  an adequacy claim.

**What the microscopist sees.** "Your raster needed 28 fields and 7 mm of stage
travel; spreading the same positions out would have needed about 19 fields and
38 mm. Neighbouring fields closer than about 250 µm were largely redundant."
Decision supported: how to lay out fields next time.

**Validation:** a constructed gradient where dispersed orders reach sustained
realized adequacy with fewer FOVs but more travel; a spatially uniform sample
where orders do not differ beyond noise. A null result on idr0028's small grid
is acceptable.

## Step 9 — Sampling-frame bias

**Question:** *were the imaged locations representative of the specimen?*
Needs a table of all candidate positions from an overview, with covariates
such as density or region: a new input contract (AGENTS.md section 15) and
upstream support in CellPatcher. Not started until approved.

## Step 10 — The report as one document

Each step already adds its sentences to `summary()`. This step orders them
by the question list of the proposal, adds the sampling profile (what is
limiting this experiment, drawn from Steps 1, 5, 6, 7 and 8), and finishes the
worked notebook from a published CellProfiler export to the answers, started
in Steps 0 and 1. Every sentence stays traceable to a table column. The
reading test (rule 4) is repeated on the whole report and on a dataset the
roadmap was not built on.

## Cross-cutting

- **Contracts:** new analyses get contract tests on the AGENTS.md section 9
  scenarios where they apply; criteria wrap registered metrics, so
  `METRICS_UNDER_CONTRACT` still covers them; the independent-unit test of
  Step 4 is a contract.
- **Runtime:** each step records its runtime on real data.
- **Provenance:** every table records criteria, strategies' `config_id`s,
  reservoir settings, orders and seeds.
- **Docs:** each step updates `docs/benchmarking.md`, `docs/architecture.md`,
  the README question table and the preliminary findings.

## Decisions needed from the project owner

| # | Decision | Needed by |
|---|---|---|
| 1 | Which deeper dataset(s) to register, and their development / held-out roles | Step 0 |
| 2 | The provisional outcome thresholds offered in presets (for example 5% relative mean error) | Step 1 |
| 3 | The `retrospective` subpackage layout, and which names join the top-level `__all__` (AGENTS.md section 15) | Step 0 |
| 4 | The per-sample reliability threshold that makes a sample "adequately represented" in the protocol curve (for example `preserved >= 0.9`) | Step 4 |
| 5 | Whether the effort projection may use a new dependency (for example statsmodels) or stays with numpy/scipy (proposed) | Step 7 |
| 6 | The input contract for candidate positions, and upstream work in CellPatcher | Step 9 |

## Reviews

One PR per step, reviewed before the next starts: Steps 0 and 1 together, then
2, 3, 4, 5, 6 (possibly in parallel), 7, 8, and 10 as its inputs land. After
Step 2 and after Step 4, an external audit asking specifically for a dataset on
which the reliability curve, the calibration or the protocol recommendation is
misleading while all tests pass (AGENTS.md section 14).
