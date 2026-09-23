# Scientific contract

MorphoCoverage studies whether microscopy acquisition has sampled a declared phenotypic representation adequately while preserving the physical acquisition hierarchy of the experiment.

Read [`project_aim.md`](project_aim.md) before changing any estimator or scientific interpretation. That document defines the research question, current-practice benchmark, assumptions, adversarial tests, and success/failure criteria.

## Problem before solution

MorphoCoverage does **not** assume that ecological coverage, missing mass, clustering, nearest-neighbour novelty, distributional distance, or another specific estimator is the correct solution. Candidate methods remain separate until benchmarked against unseen FOVs and simple baselines.

No candidate estimator should be described as "the MorphoCoverage method" merely because it was implemented first.

## Primary acquisition unit

The primary operational acquisition unit is the **field of view (FOV), defined as one independently selected spatial acquisition position (XY position)**. It is not an individual z-plane, channel, time point, or physical file.

Individual objects/cells, 2D slice measurements, channels, and time points are observations nested inside that spatial position.

Consequences:

- object count and FOV count are reported separately;
- whole FOVs, not individual objects, are the default unit for sequential acquisition and FOV-level resampling;
- FOV identity is required for the primary analysis;
- all z-planes/channels/time points from one spatial position must be revealed together during sequential acquisition;
- a dense FOV, deep z-stack, extra channels, or additional time points cannot automatically substitute for independent acquisition of additional spatial positions.

FOVs are **not assumed statistically independent**. Adjacent or overlapping fields may remain spatially correlated. Stage coordinates, physical imaged area, and acquisition design should be preserved when available.

## Two linked input levels

An observation table alone is not a complete record of acquisition because an acquired FOV with zero phenotype observations would disappear.

MorphoCoverage therefore supports:

### FOV table

One row per spatial acquisition position. At minimum: an explicit position/FOV identifier such as `position_id` or `fov_id`. `image_file` may serve as the identifier only when one file corresponds to exactly one spatial position.

Optional metadata may include condition, repeat, sample/well/batch identifiers, acquisition order, stage coordinates, FOV dimensions, physical imaged area, and sampling design.

### Observation table

At minimum: the same spatial position/FOV identifier plus explicitly selected numeric feature columns. One row may represent a cell, nucleus, patch, slice-level measurement, FOV summary, embedding, or another declared observation unit.

If features are generated slice-by-slice or otherwise within a multidimensional acquisition, preserve the relevant lowercase `z`, `c`, and/or `t` metadata on those rows. These dimensions remain nested within the spatial position and are not phenotype features or independent FOVs.

An optional object/label/patch identifier may be retained for traceability. The API records `observation_unit`; object/cell-specific rules must not silently reinterpret other observation types as cells.

## Representation-agnostic input

MorphoCoverage receives precomputed numeric representations. It does not care whether they came from morphometrics, Nyxus, CellProfiler, DINO, another foundation model, or a custom method.

The caller explicitly supplies feature columns. Metadata columns are never inferred as phenotype features. NaN feature values may be loaded but are preserved rather than silently dropped or imputed; every estimator or transform that uses affected features must declare its missing-data behavior.

## Optional grouping

Columns such as condition, repeat, sample, well, or batch may be supplied as generic grouping metadata. Their biological meaning is defined by the caller.

When grouping is requested, analyses operate independently within each group unless pooling is explicit.

## Sampling adequacy is conditional

Any future sampling-adequacy estimate is conditional on the representation, acquisition design, and method. Results require provenance for feature selection, preprocessing, metric/geometry, and any resolution or decision threshold.

A statement such as "95% covered" is scientifically incomplete unless the quantity has a precise validated definition.

## Retrospective versus prospective analyses

Retrospective accumulation may use a completed dataset to measure information that actually arrived in later FOVs. It provides ground truth for benchmarking.

Prospective estimation must use only information available at the current acquisition step. Preprocessing and tuning may not use future FOVs unless explicitly labelled retrospective/transductive.

## Existing cell-number guidance is a benchmark, not a strawman

MorphoCoverage should reproduce the scoped scalar cell-averaging problem addressed by Zweifach (JCB, 2024) and test when it is sufficient versus when spatial/FOV-aware multidimensional sampling provides additional information.

The project should not claim that the JCB framework is incorrect for the problem it was designed to solve.

## Limit of interpretation

Low expected information yield from additional representative FOVs does not prove that no biologically important rare state exists. MorphoCoverage quantifies sampling adequacy under explicit assumptions; it cannot establish absence of arbitrarily rare unseen events.


## Online deployment contract

Prospective microscope use must exercise the same estimators and decision policies as retrospective replay. `OnlineSession` accepts one newly acquired spatial position at a time:

```text
new spatial FOV
    -> feature observations
    -> OnlineSession.add_fov(...)
    -> estimator evidence
    -> decision policy
    -> continue / stop
```

The session must not require or expose future FOV identifiers, metadata, or phenotype values. Estimators that implement `EstimatorState` update only from the newly acquired FOV. Estimators without incremental support remain valid and are recomputed from the accumulated visible session history. Incremental evidence must match full recomputation to floating-point tolerance.


## FOV-level metadata may be normalized

Observation tables may remain lean: condition/repeat or other grouping metadata
can be stored only in the FOV table when the FOV identifier is globally
unambiguous. MorphoCoverage may attach that metadata to observation rows for
analysis. If the identifier is reused across groups, the mapping is ambiguous
and the framework must raise rather than guess.
