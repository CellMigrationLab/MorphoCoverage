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
    group_cols=["condition", "biological_repeat"],
    observation_unit="cell",
)

results = session.add_fov(
    observations=current_features,
    fov_metadata={
        "condition": "control",
        "biological_repeat": "R1",
        "position_id": "P017",
    },
)
```

The session does not know how many future positions will be acquired and never
receives their identifiers or features.

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
MorphoBricks defaults, `condition`, `biological_repeat` and, when present,
`technical_repeat` are resolved automatically as the analysis groups, and
each group has independent estimator state.

No evidence is shared between groups unless the caller deliberately chooses a
different grouping contract.

A field whose observations or `fov_metadata` still name the biological repeat
`repeat` (its former name) is read as `biological_repeat`, with one
`DeprecationWarning` per `add_fov` call (per `LiveTables.update` or
`LiveFolder.poll`/`watch` call, whatever the number of fields it adds);
`group_cols` or `fov_col` naming `repeat` are read the same way. A field that
holds both names with different values is refused, and the session is left as
it was (`docs/common/data_format.md`).

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

## Online equivalence

`OnlineSession` is the deployment counterpart of incremental replay. It receives one FOV at a time and knows nothing about future acquisition. Contract tests compare live session outputs with `evaluate_acquisition_prefixes(..., incremental=True)` on the same revealed sequence. Scientific evidence and decisions must match; only timing context may differ.

Methods without an incremental `EstimatorState` remain deployable because the session can recompute them from accumulated visible observations. Benchmarking determines whether that latency is acceptable.
