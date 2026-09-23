# Modular architecture

MorphoCoverage is designed as a strategy-comparison framework rather than a single estimator.

## Core flow

```text
MorphoBricks-compatible feature tables
        |
        v
metadata/FOV resolution
        |
        v
condition x repeat grouping
        |
        v
ValidatedInput
        |
        +--> Strategy A
        +--> Strategy B
        +--> Strategy C
        |
        v
canonical StrategyResult
        |
        v
real-dataset sequential benchmark
```

The shared runner owns validation, MorphoBricks metadata handling, grouping, timing, and provenance. Individual strategies own only their mathematical logic.

## MorphoBricks defaults

The framework follows the shared MorphoBricks metadata conventions:

- `condition`
- `repeat`
- `image_file`
- `mask_file`
- `patch_file`

When the caller does not provide an FOV identifier, `image_file` is preferred and `fov_id` is accepted as a fallback. One image file is one FOV, including a z/t/c stack; objects measured on different planes of the same file belong to the same FOV. A candidate column present in only one of the linked tables is an error, never a reason to fall back to the next candidate.

When the caller does not provide grouping columns, `condition` and `repeat` are used automatically. This keeps biological/sample groups independent without requiring every strategy to understand those columns. A canonical column present in only one linked table raises instead of silently pooling, and non-canonical spellings such as `Repeat` raise unless named explicitly.

### FOV identity versus analysis grouping

Two concepts are kept separate:

- **FOV identity** (`ValidatedInput.fov_key`): the columns that jointly identify one acquired FOV. It always contains the analysis group columns plus any canonical `condition`/`repeat` columns, then the FOV column.
- **Analysis groups** (`ValidatedInput.group_cols`): which FOVs are evaluated together.

Passing `group_cols=[]` is an explicit request to pool FOVs into one analysis. It does not remove `condition`/`repeat` from FOV identity, so `R1/img_001.tif` and `R2/img_001.tif` remain two FOVs. Strategies that need FOV identity must use `fov_key`, not `fov_col` alone.

Metadata columns are never inferred as phenotype features. Strategies read phenotype values through `ValidatedInput.feature_matrix()`.

## Strategy contract

Every strategy implements the same interface:

```python
class SamplingStrategy:
    name: str

    def evaluate(self, data, *, objective):
        ...
```

A strategy receives one already validated analysis group at a time.

A strategy instance is configuration only. The framework calls `evaluate` on a fresh deep copy for every analysis group and every acquisition prefix, so no fitted state (scaler, PCA, reference sample) can flow between groups or prefixes. `evaluate` is therefore a full recomputation on the visible data. Efficient incremental updates will need an explicit, separately tested state interface; they must not rely on hidden instance state.

A strategy must not:

- split conditions or repeats;
- infer feature columns;
- read microscopy images;
- silently inspect withheld FOVs;
- redefine the user's sampling objective.

## Sampling objective

The objective is independent of the strategy.

Current objective types are:

- `population`
- `rare_discovery`
- `balanced`

Rare-discovery objectives may specify:

- minimum phenotype prevalence of interest;
- target detection probability;
- minimum number of independent FOV occurrences.

A strategy may perform well for one objective and poorly for another. That is a benchmarking result, not an API failure.

## Canonical result

Every strategy returns the same result structure containing:

- strategy name;
- objective;
- score;
- optional continue/stop decision;
- optional predicted remaining information;
- number of FOVs;
- number of objects;
- resolved condition/repeat group identity;
- strategy parameters;
- diagnostics;
- measured wall-clock runtime.

The number of FOVs, number of objects, group identity and runtime are framework-owned. The framework fills them after the strategy returns; a strategy-supplied count that contradicts the acquisition record raises. `decision` must be `Decision.CONTINUE`, `Decision.STOP` or `None`.

This common contract is what allows many methods to be compared without changing the data pipeline.

## Sequential real-data evaluation

Deeply sampled real datasets are the primary scientific benchmark.

The sequential runner reveals a real acquisition prefix at a time while preserving:

- acquisition order;
- FOV identity;
- condition/repeat grouping;
- zero-object FOVs.

Each strategy is evaluated using only the prefix available at that step.

Leakage guards enforced by the runner:

- the complete linked tables are validated once, so objects from FOVs missing from the FOV table raise rather than never being revealed;
- acquisition order must have no ties within an analysis group; a pooled analysis therefore needs an order defined across the pooled FOVs;
- the acquisition order column cannot be a phenotype feature;
- visible tables are re-indexed and categorical columns drop unused categories, so neither row labels nor category levels reveal withheld FOVs or other groups.

The runner cannot detect leakage introduced upstream, such as features normalized or projected over the whole dataset before they reach MorphoCoverage. That must be recorded as provenance.

The framework deliberately keeps strategy execution separate from the later benchmark outcome calculation so a method is not rewarded by being judged only with its own internal metric.

## Computational benchmarking

Runtime is measured for every strategy evaluation from the start.

Later benchmarking will extend this with:

- incremental-update latency;
- total runtime across a complete acquisition;
- high-percentile latency;
- peak memory;
- empirical scaling with FOV count, object count, and feature dimension;
- compute-to-acquisition ratio for smart microscopy.

## Current baseline strategies

The initial `FixedFOVStrategy` and `FixedObjectStrategy` are architectural baselines only. They exist to prove that strategies can be swapped through one shared runner and to provide simple comparators later.

They are not scientific defaults.

## Extension rule

New strategies should be added behind the existing interface rather than adding strategy-specific branches to the runner.

Examples of future families include:

- scalar precision / JCB-style strategies;
- population-summary stabilization;
- MMD / energy / Wasserstein distribution distances;
- kNN and density novelty;
- ecological incidence/discovery estimators;
- FOV-distribution methods;
- predictive or conformal approaches.

A new method should not require changes to MorphoBricks metadata handling or group separation unless the data contract itself genuinely changes.

See [`morphobricks_fov_contract.md`](morphobricks_fov_contract.md) for what upstream MorphoBricks tools must emit for FOV identity to be correct.
