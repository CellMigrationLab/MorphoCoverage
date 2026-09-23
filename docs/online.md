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

## Timing

Every result contains `Timing`:

- estimator recomputation time;
- incremental update time;
- initial fitting/warm-up time;
- framework overhead;
- total MorphoCoverage decision latency.

These times exclude upstream feature extraction and microscope-control latency.
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
