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
`add_fov`.

## What one update means

One call to `add_fov` is one independently selected XY acquisition position.

All observations belonging to that position are supplied together. If features
were computed slice-by-slice, channel-by-channel or frame-by-frame, the rows
may carry z/c/t metadata but must share the same FOV/position identity.

A zero-observation position is valid. Pass an empty DataFrame that still has
the declared feature columns.

## Conditions and repeats

The same session can receive FOVs from several analysis groups. With the
MorphoBricks defaults, `condition × repeat` is resolved automatically and
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
`stop: lofo_novel_fov_upper = 0.24 (stop at <= 0.25, population); after 11
FOVs, 1100 cells`. Read the quantity, not only the word: with nothing new among `k`
FOVs the novelty bound is `1 - 0.05 ** (1 / k)` whatever the data, so a stop
on it after 11 FOVs is a statement about the next field, not about rare
states (`docs/preliminary_findings.md`, entry 3). A rule's threshold is its
own; whether its stops were premature is the retrospective calibration
(`morphocoverage.retrospective.stopping_calibration`).

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
