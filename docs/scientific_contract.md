# Scientific contract

MorphoCoverage studies whether microscopy acquisition has sampled a declared phenotypic representation adequately while preserving the physical acquisition hierarchy of the experiment.

Read [`project_aim.md`](project_aim.md) before changing any estimator or scientific interpretation. That document defines the research question, current-practice benchmark, assumptions, adversarial tests, and success/failure criteria.

## Problem before solution

MorphoCoverage does **not** assume that ecological coverage, missing mass, clustering, nearest-neighbour novelty, distributional distance, or another specific estimator is the correct solution. Candidate methods remain separate until benchmarked against unseen FOVs and simple baselines.

No candidate estimator should be described as "the MorphoCoverage method" merely because it was implemented first.

## Primary acquisition unit

The primary operational acquisition unit is the **field of view (FOV)**. Individual objects/cells are observations nested inside FOVs.

Consequences:

- object count and FOV count are reported separately;
- whole FOVs, not individual objects, are the default unit for sequential acquisition and FOV-level resampling;
- FOV identity is required for the primary analysis;
- a dense FOV cannot automatically substitute for independent acquisition of additional FOVs.

FOVs are **not assumed statistically independent**. Adjacent or overlapping fields may remain spatially correlated. Stage coordinates, physical imaged area, and acquisition design should be preserved when available.

## Two linked input levels

An object table alone is not a complete record of acquisition because an acquired FOV with zero detected objects would disappear.

MorphoCoverage therefore supports:

### FOV table

At minimum: `fov_id`.

Optional metadata may include condition, repeat, sample/well/batch identifiers, acquisition order, stage coordinates, FOV dimensions, physical imaged area, and sampling design.

### Object table

At minimum: `fov_id` plus explicitly selected numeric feature columns.

An optional object identifier may be retained for traceability.

## Representation-agnostic input

MorphoCoverage receives precomputed numeric representations. It does not care whether they came from morphometrics, Nyxus, CellProfiler, DINO, another foundation model, or a custom method.

The caller explicitly supplies feature columns. Metadata columns are never inferred as phenotype features.

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
