# Online microscope use

MorphoCoverage can run prospectively without a completed acquisition.

`OnlineSession` receives exactly one newly acquired spatial FOV/position at a
time. It uses the same `SamplingStrategy`, estimators, evidence and decision
policies as retrospective replay.

## Minimal pattern

```python
from morphocoverage import (
    FixedFOVStrategy,
    ObjectiveConfig,
    OnlineSession,
    SamplingObjective,
)

session = OnlineSession(
    strategies=[FixedFOVStrategy(10)],
    feature_cols=["feature_a", "feature_b"],
    objective=ObjectiveConfig(SamplingObjective.POPULATION),
    fov_col="position_id",
    group_cols=["condition", "repeat"],
    observation_unit="cell",
)

results = session.add_fov(
    observations=current_features,
    fov_metadata={
        "condition": "control",
        "repeat": "R1",
        "position_id": "P017",
    },
)
```

The session does not know how many future positions will be acquired and never
receives their identifiers or features.

## Live sources: a folder or tables in memory

`morphocoverage.workflows` feeds a session from where the fields actually
arrive ([`workflows.md`](workflows.md#online-continue-or-stop-while-the-microscope-acquires)):

- `LiveFolder(folder, session).watch()` watches a folder with one object file
  per field and an append-only manifest, `fovs.csv`. `write_fov` writes both
  atomically for the acquisition side.
- `LiveTables(session).update(objects, fovs)` reads the rows appended to two
  in-memory tables since the last call, for features computed live and never
  written to disk.

In both sources, the FOV row (a manifest line, or a row of the FOV table) is
the signal that a field is complete. Its objects are added together, through
`add_fov`. `watch(until_stop=True)` returns once every rule says stop in every
group seen so far (`until_stop="latest_group"`: in the latest field's group),
and fields are added one at a time, so a backlog is not added past the stop.
A manifest that is rewritten or truncated under the watcher raises, a line
repeated exactly is skipped with a warning, a field with rows copied or a
`label_id` twice is added with a warning, and `LiveTables` refuses a key that
names two fields (file names that restart in each well without a well
column in the key).

## What one update means

One call to `add_fov` is one independently selected XY acquisition position.

All observations belonging to that position are supplied together. If features
were computed slice-by-slice, channel-by-channel or frame-by-frame, the rows
may carry z/c/t metadata but must share the same FOV/position identity.
The session resolves the FOV identifier and the group columns as
`validate_input_tables` does, with the same resolvers, over the new field's
observations and its `fov_metadata` row: a MorphoBricks column (`condition`,
`position_id`, ...) in the observations but not in the row is refused, and
identity values given only in the row are copied onto the observations, which
all belong to that field. When the
FOV column was chosen automatically as `image_file` or `fov_id`, the session
keeps each field's z/c/t values (from the observations, or from the row) and
applies the validator's plane-as-field rule to every field so far, so
z-planes sent as separate files raise instead of counting as fields (with one
observation per field, a warning). `add_fov` is atomic: a field refused by
validation or by any estimator leaves the session as it was, and can be added
again once corrected.

A zero-observation position is valid. Pass an empty DataFrame that still has
the declared feature columns.

## Conditions and repeats

The same session can receive FOVs from several analysis groups. With the
MorphoBricks defaults, `condition`, `repeat` and, when present,
`technical_repeat` are resolved automatically as the analysis groups, and
each group has independent estimator state.

No evidence is shared between groups unless the caller deliberately chooses a
different grouping contract.

## Incremental versus recomputed methods

An estimator may implement:

```text
Estimator.start(...)
    -> EstimatorState

EstimatorState.update(new_fov)
    -> Evidence
```

Such methods update only from the newly acquired FOV.

Estimators without incremental support remain usable online. The session
recomputes them on the observations accumulated so far. This keeps one
scientific implementation for offline and online use; incremental code is a
performance optimization, not a second estimator.

Contract tests require incremental replay to agree with full recomputation.

## Reading a decision

Every `StrategyResult` has `explain()`: one line with the decision, each
policy's rule beside its evidence, and the effort, for example
`stop: lofo_novel_fov_upper = 0.24 (stop at <= 0.25, distribution); after 11
FOVs, 1100 cells; treats the fields as a random sample of the well's
positions; a raster over a spatial trend can stop early`. Every stop ends with
what it rests on (see [Acquisition order](#acquisition-order-random-or-dispersed)),
and a rule that treats objects as independent draws (the JCB rule per object)
says so. When the rule decides some of the objective's targets but not all (a
JCB rule under a declared rare target) the line says
`stop (decides the population summary only): ...`; when it decides none of
them (a novelty rule under the population objective) it says
`unsupported (the rule itself: continue): ...`, so the rule's own verdict is
never hidden behind the objective's.

Several rules on one acquisition state are read together with
`morphocoverage.results.decision_view(results)`: one entry per target
(population summary, phenotypic fingerprint, rare phenotype), the decision
combined over the rules that decide it and the criteria that decided it,
then the overall decision for the objective; a required target no rule
decides is printed as such and keeps the overall at `unsupported`. The JCB
criterion is named as what it is: an averaging CV of 5% is one standard error
of 5% of the mean (the paper's criterion), not a bound of ±5% on the mean's
error; whether the mean was within the declared tolerances is judged on
held-back fields by `report` and `optimise`. `LiveUpdate.summary(detail="full")` ends with
that view, one entry per line (`docs/questions.md`, O4); the default short block opens with
the field and its decision (`✅ field 8 · MOCK · 1A · B6 — stop: all 3 rules say enough`, `⏳
continue: ...` with what is not yet met, `⚠️` at a stop the order check flags, `➖` while a
declared target has no rule), then one table of the rules and, at a stop, each caveat once. `live_session(setup)` runs under
`setup.objective()`, whose components are the declared targets. Read the quantity, not only the word: with nothing new among `k`
FOVs the novelty bound is `1 - 0.05 ** (1 / k)` whatever the data, so a stop
on it after 11 FOVs is a statement about the next field, not about rare
states (`docs/preliminary_findings.md`, entry 3). A rule's threshold is its
own; whether its stops were premature is the retrospective calibration
(`morphocoverage.retrospective.stopping_calibration`).

## Acquisition order: random or dispersed

Every stopping rule reads the fields seen so far as a random sample of the
well's positions. A raster that tiles the well row by row breaks that when
the readout follows a spatial trend (a gradient, an edge, an empty region) or
drifts over the acquisition: the first rows look settled, and the default JCB
pair can say stop after one or two rows while the well's mean is far off. In
audit 4 (a 10 x 10 well, a gradient of ±20%), the overall stop was more than
5% off in every raster run, against 16% of runs with the same wells in random
order; a rare state confined to the last rows was declared ruled out in every
raster run before the raster reached it, and in no random-order run.

**Acquire fields in a random or dispersed order** (for example a random
permutation of the planned positions, or a farthest-point order of them,
`morphocoverage.retrospective.stage.dispersed_order`); that makes the rules' premise true instead of hoping
it holds.

`live_session(setup)` also checks the fields of each group so far, and
`LiveUpdate.summary()` prints the result once at a stop (`detail="full"`: beside every stop,
in each rule's line and after the overall decision):

- with `Setup.acquisition_order_col`, a trend of the field means of the
  readout with acquisition order (Spearman);
- with `Setup.stage_cols`, alike means in neighbouring fields (Moran's I,
  neighbours within 1.5 times the median nearest-neighbour distance, a
  permutation p-value with a fixed seed), and beside a rare-phenotype stop the
  stage region the fields have covered (in the report: the share of the
  acquired fields' stage extent covered at the recorded order's stop);
- with neither, the stop says that it treats the fields as a random sample of
  the well's positions, and that a raster over a spatial trend can stop early.

The two tests share a 5% level (Bonferroni), and each needs 4 fields with a
mean. The check **flags, it never vetoes**: a flagged stop stays a stop. A
check that finds nothing does not show the fields to be representative: in a
raster over a gradient it has little power in the first rows, where the
early stops happen. It reads only the fields seen so far, so a replay of the
same prefix gives the same check (`order_check.check_acquisition_order`).
`report` prints the same caution beside each rule's stop ("Did each
criterion say enough?"), computed on the group's acquired fields.

## Timing

Every result contains `Timing`:

- estimator recomputation time;
- incremental update time;
- initial fitting/warm-up time;
- framework overhead;
- total MorphoCoverage decision latency.

These times exclude upstream feature extraction and microscope-control latency.
Measured on 30 standardized features, the leave-one-FOV-out support-novelty
estimator took 0.16 s per decision at 30 FOVs of 100 cells, 0.64 s at 60 FOVs
of 100 cells and 5.9 s at 60 FOVs of 300 cells (it refits the transform once
per visible FOV and compares every pair); the FOV bootstrap took 0.03 s.
Compare the measured latency with the time available before the next
acquisition.
End-to-end deployment should therefore measure:

```text
feature extraction
+ MorphoCoverage decision
+ microscope control
```

against the time available before the next acquisition decision.

## Missing feature values

The input layer permits NaN because real feature tables frequently contain
undefined measurements. MorphoCoverage never silently imputes or removes
them.

Each estimator or transform must define its own missing-data behavior.
Packaged scalar/stabilization baselines currently report their affected
evidence as not applicable; Standardize/PCA refuse to fit and return an
explicit reason.

## Scope

The first online question is only:

> Should another representative FOV be acquired?

Choosing where to image next is a later targeted-acquisition problem. It
changes the sampling distribution and must not be silently mixed into the
representative stopping stream.


## Stable live metadata contract

The first FOV fixes the session's automatically resolved spatial identifier
and grouping columns. This identity/grouping contract is immutable for the session. Later FOVs may not silently introduce a higher-priority
identifier (for example `position_id` after the session began with
`image_file`) or change the automatically resolved condition/repeat grouping.
Start a new session, or specify `fov_col` and `group_cols` explicitly, when
the acquisition contract differs.
