# AGENTS.md - MorphoCoverage

> This file is the authoritative engineering context for any AI agent or contributor working in this repository. Read it completely before taking action, and read `docs/project_aim.md` before changing any estimator, scientific claim, validation design, or stopping rule. Prefer repository evidence, tests, and documented scientific contracts over assumptions.

## 1. Project identity

| Field | Value |
| --- | --- |
| Repo | `CellMigrationLab/MorphoCoverage` |
| Purpose | Quantify FOV-aware phenotypic sampling completeness from precomputed object-level feature representations. |
| Ecosystem | MorphoBricks |
| Position | Step 3 of the MorphoBricks pipeline (analysis & applications). It consumes features; it does not extract them. See section 18. |
| Language | Python 3.11+ |
| Tests | `pytest` (all tests, including real-data tests on `data/crick_idr`) |
| Lint | `ruff check src tests` (ruff pinned in `pyproject.toml`) |
| Install / environment | `python -m pip install -e ".[dev]"` |

Key dependencies: `numpy`, `pandas` (runtime); `pytest`, `ruff` (development). There are deliberately no scipy, scikit-learn or Torch dependencies.

MorphoCoverage is deliberately independent of CellPatcher, fiberfeatures, Nyxus, CellProfiler, DINO, Torch, microscopy file formats, segmentation tools, and microscope-control software.

## 2. Scientific question

The central questions are:

> Has this sample been imaged sufficiently for the stated analytical objective?

> Is another representative FOV likely to provide enough useful new information to justify acquiring it?

The exact mathematical solution is intentionally unresolved. See `docs/project_aim.md`.

The microscope samples spatial acquisition positions. In MorphoCoverage, **one FOV means one independently selected XY acquisition position**, regardless of how many z-planes, channels, or time points were acquired there. Objects/cells, slice-level measurements, channels, and time points are observations nested inside that position. Therefore cell/object/plane count alone is not assumed to be an adequate measure of spatial sampling effort.

The hierarchy is:

```text
object/cell/slice/timepoint -> FOV = spatial acquisition position -> optional technical/sample unit -> optional biological repeat -> optional condition
```

The package may support arbitrary grouping columns, but it must not assign biological meaning to them automatically.

## 3. Non-negotiable rules

These rules override implementation convenience.

MorphoBricks ecosystem rules (shared with the other MorphoBricks repositories):

- **Never rename shared CSV columns** (`condition`, `repeat`, `image_file`, `mask_file`, `patch_file`). Other repos depend on them. MorphoCoverage rejects near-miss spellings such as `Repeat` rather than renaming them.
- **Never hard-code machine-specific paths.** Use function arguments, config or environment variables; tests locate repository data relative to the test file.
- **Never break public interfaces** (anything in `morphocoverage.__all__`) without explicit user approval.
- **Keep public functionality usable as an imported dependency and from notebooks.** A CLI is deliberately deferred: the project owner asked to avoid premature CLI work until the scientific core stabilizes (see section 18). Add one only when asked.

Scientific rules:

- **FOV identity is mandatory for the primary MorphoCoverage analysis.** FOV means one independently selected spatial acquisition position, not a z-plane, channel, time point, or file. Do not silently degrade to object-, plane-, or frame-level sampling when position identity is absent.
- **FOV is an operational acquisition unit, not an automatic independence assumption.** Preserve spatial coordinates, area, and design metadata when available; whole-FOV resampling does not by itself eliminate spatial autocorrelation.
- **Do not assume one mathematical solution.** Ecological coverage, missing mass, clustering, nearest-neighbour novelty, distributional convergence, Bayesian methods, information-theoretic methods, and simple empirical rules remain candidate approaches until comparative validation supports a default.
- **Current cell-number guidance must be benchmarked fairly.** Reproduce the scoped scalar precision problem before arguing that FOV-aware sampling adds information; identify regimes where the simpler approach is sufficient.
- **Permit null results.** Do not optimize validation, examples, or documentation to guarantee that MorphoCoverage outperforms fixed cells/FOVs or scalar precision rules.
- **FOV/spatial position is the primary acquisition/resampling unit.** Never bootstrap, permute, or sequentially reveal individual objects, z-planes, channels, or time points as independent FOVs. All observations from one position must enter together.
- **Never infer feature columns from all numeric columns.** Feature columns must be supplied explicitly. Numeric metadata such as IDs, stage positions, acquisition order, pixel size, or timepoints must never silently enter phenotype space.
- **Metadata is not phenotype.** Grouping/acquisition columns such as condition, repeat, batch, well, sample, acquisition identifiers, and within-position `z`, `c`, `t` metadata must not enter the phenotype matrix when they encode acquisition dimensions. Feature columns are explicit, so callers must omit such metadata from `feature_cols`.
- **Never pool groups silently.** If the caller requests analysis by condition, repeat, or other groups, analyses operate independently within those groups. Pooling must be explicit.
- **Do not claim that more cells from one FOV equal more independent imaging coverage.** Object count and FOV count are different quantities and must remain separately visible.
- **Preserve acquired FOVs with zero detected objects.** An object-only table is not a complete acquisition record; support a first-class FOV table.
- **Never report a coverage/novelty result without enough provenance to reproduce the phenotype geometry**, including supplied feature columns, preprocessing, distance metric, and morphological resolution/threshold when applicable.
- **Retrospective and prospective analyses are different.** Retrospective analysis may use future/held-out FOVs to measure realized outcomes. A prospective estimator must use only information available up to the current acquisition step.
- **No future-data leakage.** Preprocessing, tuning, threshold selection, dimensionality reduction, and hyperparameter optimization for prospective evaluation must not use future FOVs unless explicitly labelled retrospective/transductive.
- **Scientific estimator changes require scientific tests.** Every estimator must be challenged on synthetic populations with known behavior, including rare phenotypes, spatial clustering, variable cells/FOV, zero-object FOVs, and technical shifts.
- **Random procedures must be reproducible.** Expose `random_state` or equivalent; tests use fixed seeds.
- **Do not equate low expected novelty with proof that no rare biology exists.** The package estimates sampling adequacy under an explicit representation and sampling scheme; it cannot guarantee absence of arbitrarily rare unseen states.
- **Smart microscopy starts with continue/stop representative acquisition.** Do not conflate this with targeted next-FOV selection. Targeted acquisition changes the sampling distribution and requires separate treatment.

## 4. Scope boundaries

### This repo owns

- validation of object-feature tables and complete FOV metadata;
- feature-space preprocessing required to define phenotype geometry;
- distance/similarity calculations used by candidate methods;
- object- and FOV-level novelty/statistics where scientifically justified;
- accumulation and sequential holdout;
- candidate sampling-completeness estimators;
- whole-FOV resampling and uncertainty;
- stopping criteria based on validated estimators;
- representation-agnostic results and plots.

### This repo does not own

- image or mask IO;
- segmentation;
- patch generation;
- morphometric/texture/intensity feature extraction;
- DINO or other foundation-model inference;
- Torch/GPU model execution;
- feature-extractor benchmarking/tuning;
- biological-condition hypothesis testing;
- microscope hardware control.

Do not add dependencies for those concerns merely to make an example easier.

## 5. Inter-repo contract

MorphoCoverage consumes two linked logical tables.

### FOV table

One row per acquired FOV, including FOVs with zero detected objects.

Required:
- caller-specified spatial-position/FOV identifier column. Prefer `position_id` when storage layout does not map one file to one position; `image_file` is a fallback only when one file equals one position.

Optional:
- condition, repeat, sample, well, batch;
- acquisition order;
- stage coordinates;
- FOV dimensions or physical imaged area;
- sampling-design metadata.

### Object table

One row per detected object.

Required:
- the same FOV identifier;
- explicitly supplied numeric phenotype feature columns.

Optional:
- object/label/patch identifiers for traceability.

MorphoCoverage must be able to consume output derived from CellPatcher or any feature-extraction repo without importing those packages.

Where practical, preserve compatibility with MorphoBricks metadata names such as `condition`, `repeat`, and `image_file`, but the core API accepts caller-specified column names rather than hard-coding them.

### Shared CSV column contract (MorphoBricks)

| Column | MorphoBricks requirement | How MorphoCoverage uses it |
| --- | --- | --- |
| `condition` | string, always | Default analysis group; always part of FOV identity. |
| `repeat` | string, always (`YYYY_MM_DD`, or `single`) | Default analysis group; always part of FOV identity. Any label is accepted; the format is not enforced. |
| `image_file` | string, always | Storage provenance; fallback FOV identifier only when one file = one spatial position. |
| `position_id` | string, when needed | Preferred explicit spatial-position identifier when files/planes/scenes do not map 1:1 to FOVs. |
| `mask_file` | string, if used | Preserved as metadata, never a feature. |
| `patch_file` | string, if used | Preserved as metadata, never a feature. |

A canonical column present in only one of the linked tables raises; so does a non-canonical spelling unless named explicitly. Columns MorphoCoverage adds or recognises beyond the shared contract (`fov_id` as a fallback FOV identifier, `label_id`, acquisition order, stage coordinates, area) are documented in `docs/architecture.md` and `docs/morphobricks_fov_contract.md`. The test data in `data/crick_idr/` uses plate IDs as `repeat`, since no dates exist in the source (see its README).

### Outputs

MorphoCoverage currently produces in-memory result objects (`StrategyResult`, `SequentialResult`, each result carrying its `Evidence`), not files. No other repository consumes its output yet. Define a file format here before any downstream repository depends on it.

## 6. Architecture principles

- Prefer small pure functions with typed inputs/outputs.
- Keep statistical primitives separate from orchestration and plotting.
- Do not create abstractions for algorithms that have not yet been implemented and validated.
- Keep public APIs minimal until scientific behavior stabilizes.
- Raise clear errors on ambiguous or invalid input; never silently guess.
- Avoid hidden state and global configuration.
- Avoid heavyweight dependencies unless they provide demonstrated value.
- Notebooks consume package APIs; they are never homes for production algorithms.
- Make the smallest coherent change needed for the task.

## 7. Repository structure

```text
src/morphocoverage/
    __init__.py            # public API (__all__); changes need approval (section 15)
    metadata.py            # MorphoBricks column resolution: FOV key, groups, spellings
    schema.py              # ValidatedInput and linked-table validation
    objectives.py          # sampling objectives and RareTarget, independent of strategies
    evidence.py            # Evidence and Quantity: estimates with direction, units, component
    policies.py            # decision policies, per-objective combination, redecide
    provenance.py          # config_id hashing
    results.py             # StrategyResult and the Decision vocabulary
    evaluator.py           # group iteration, isolation, timing, framework-built results
    sequential.py          # leakage-safe acquisition-prefix replay
    outcomes.py            # retrospective outcome metrics, strategy-independent
    transforms.py          # feature transforms fitted on the visible prefix only
    benchmark.py           # dataset roles, acquisition orders, one record per run
    framework.py           # run_strategies convenience entry point
    strategies/
        base.py            # Estimator interface; SamplingStrategy = estimators + policies
        baselines.py       # fixed-FOV and fixed-object baselines
        scalar_precision.py  # JCB/Zweifach-style scalar precision baseline
        stabilization.py   # Marcotti et al.-style descriptor-stabilization baselines

tests/
    scenarios.py           # constructed acquisitions for contract tests (section 9)
    test_contracts.py      # contracts every strategy and the framework must pass
    test_<module>.py       # unit tests per module or strategy family
    test_crick_idr.py      # real-data integrity and end-to-end tests

data/crick_idr/            # real feature tables (CC BY 4.0) for tests; README gives provenance
docs/                      # scientific aims, contracts, architecture, benchmarking, findings
docs/proposals/            # design proposals awaiting approval; not implemented
```

A new strategy family goes in its own module under `strategies/` (section 20). Future modules should be added only when the corresponding behavior is implemented and tested. Do not pre-create a `novelty`, `coverage`, or estimator architecture that implicitly selects a scientific solution before comparative validation. Data-preparation scripts live next to their data, not in the package.

## 8. Development sequence

Do not skip directly to a polished `analyze()` API.

1. Strict linked FOV/object input contracts and validation.
2. Define explicit sampling objectives: population representation, rare-phenotype discovery, and balanced sampling.
3. Build one modular strategy interface and one canonical result/provenance contract.
4. Build leakage-safe sequential evaluation for deeply sampled real acquisitions.
5. Reproduce simple/current-practice baselines, including fixed cells/FOVs and the scoped scalar cell-averaging problem.
6. Implement substantially different strategy families behind the same interface.
7. Benchmark candidates on deeply sampled real datasets using strategy-independent scientific and computational criteria.
8. Add targeted counterexamples and regression fixtures when real-data failures reveal them.
9. Select a default estimator or decision rule only if evidence supports one.
10. Validate continue/stop decisions prospectively before considering targeted smart-microscopy acquisition.

## 9. Validation philosophy and required contract tests

Deeply sampled real microscopy datasets are the primary scientific benchmark. Constructed or synthetic data are supporting tools for unit tests, explicit counterexamples, and regression tests after a failure mode is identified; they are not the main evidence that a strategy works.

Constructed data are still required for **contract tests**: acquisitions whose structure is known by construction, used to prove that the framework and every strategy handle that structure correctly. The scenarios below are implemented in `tests/scenarios.py` and exercised by `tests/test_contracts.py`. Each backticked id must have either a fixture or a named framework contract; `test_agents_section_9_matches_contract_tests` fails if this list and the tests drift apart.

Required scenarios:

- homogeneous phenotype distribution: `homogeneous`;
- clearly separated multimodal phenotypes: `multimodal`;
- rare states at controlled prevalence, e.g. 10%, 1%, 0.1%: `rare_prevalence_10pct`, `rare_prevalence_1pct`, `rare_prevalence_0p1pct`;
- identical rare-state prevalence distributed uniformly versus clustered into few FOVs: `rare_uniform`, `rare_clustered`;
- unequal numbers of objects per FOV: `unequal_objects_per_fov`;
- acquired FOVs with zero detected objects: `zero_object_fovs`;
- unequal physical FOV area when area metadata are available: `unequal_fov_area`;
- correlation between neighbouring FOVs: `neighbour_correlation`;
- direct comparison against fixed-cell, fixed-FOV, and scalar-precision baselines: `baseline_comparison`;
- duplicated objects within an existing FOV versus acquisition of new FOVs: `duplicated_objects`;
- optional condition/repeat groups kept independent: `condition_repeat_groups`;
- technical/batch shifts: `batch_shift`;
- deterministic results under a fixed seed: `deterministic_under_seed`;
- no future-FOV leakage in prospective estimators: `no_future_fov_leakage`.

Rules for these tests:

- Every fixture must also have a test proving it has the property it claims (exact prevalence, FOV incidence, separation, correlation, shift), so contracts are never checked against hollow data.
- Every strategy added to the package must be added to `STRATEGIES_UNDER_CONTRACT` in `tests/test_contracts.py`. It then has to pass, on every scenario: counts that match the acquisition record, determinism, and group independence. Framework-level contracts (whole-FOV reveal in acquisition order, no future-FOV leakage, explicit features only, zero-object FOVs counted as effort) apply to all strategies automatically.
- An estimator that supports incremental updates (`Estimator.start`) must give, at every step and on every scenario, the same evidence and decisions under `incremental=True` replay as under full recomputation, to floating-point rounding. The incremental-equivalence contract in `tests/test_contracts.py` runs this for every strategy in `STRATEGIES_UNDER_CONTRACT`.
- Every outcome metric added to the package must be added to `METRICS_UNDER_CONTRACT`. It has to return a finite float or None (never NaN), be deterministic, leave its inputs unchanged, and depend only on the full acquisition and the stop prefix, never on which strategy stopped.
- Every feature transform added to the package must be added to `TRANSFORMS_UNDER_CONTRACT`. Its fitted state and output at step k must not change when FOVs acquired after step k are altered, and its fitted state must be JSON-serializable provenance.
- Contract tests check correct handling, not scientific performance. Whether a strategy stops early enough, or detects the clustered rare state, is a benchmarking result on real data. Such expectations must not be encoded as contract tests that force a strategy to "win".
- `baseline_comparison` covers the fixed-cell, fixed-FOV, JCB-style scalar-precision (`ScalarPrecisionStrategy`) and Marcotti-style descriptor-stabilization (`DescriptorPlateauStrategy`, `DescriptorSpreadStrategy`) baselines. Any new current-practice baseline must be added to it.

A high line-coverage percentage is not a substitute for these behavior tests or for scientific validation on real acquisitions.

## 10. JCB comparison rules

The 2024 JCB "how many cells to average" framework is an important benchmark, not an opponent to caricature.

Any implementation comparing against it must:

1. state the scalar parameter being estimated;
2. reproduce the intended cell-averaging precision problem faithfully;
3. state whether spatial-uniformity assumptions hold;
4. compare scalar precision against held-out-FOV phenotypic outcomes;
5. report regimes where the JCB-style criterion is sufficient;
6. avoid language implying the JCB method is wrong simply because it answers a narrower question.

The strongest intended result is a **regime map**:
- when cell-number guidance is sufficient;
- when fixed-FOV guidance is sufficient;
- when richer FOV-aware sampling materially improves decisions.

## 11. Smart microscopy rules

The first prospective question is:

> Should another representative FOV be acquired?

Only after that is validated should the package support:

> Where should the microscope image next?

If acquisition becomes targeted, do not treat targeted FOVs as an unbiased population sample. Use representative/sentinel FOVs, explicit sampling probabilities, or another validated design.

## 12. Coding standards

- PEP 8 style.
- 4-space indentation.
- Descriptive names.
- Type hints on public functions.
- Docstrings on public modules/functions/classes.
- Comments explain scientific assumptions or non-obvious reasoning.
- Prefer specific exception types and actionable error messages.
- Do not perform unrelated refactors, dependency upgrades, API renames, or formatting sweeps.

## 13. Verification

For substantive code changes, when the environment permits:

```bash
pytest
ruff check src tests
```

Never claim a command passed unless it actually ran successfully.

For estimator changes, ordinary unit tests are insufficient: report which synthetic scientific scenarios were exercised and what behavior was expected.

## 14. Working with AI coding agents

AI agents should work through narrowly scoped branches/PRs.

Recommended workflow:

1. Builder reads this file and implements one issue-sized change.
2. Builder runs tests/lint and documents scientific assumptions.
3. A separate reviewer is asked to find a counterexample or dataset where the implementation gives a misleading result.
4. Any discovered failure becomes a regression test before acceptance.

Do not let two agents independently expand the architecture in parallel without a shared issue/specification.

When reviewing statistical code, explicitly ask:

> Can you construct a dataset on which this implementation gives a misleading answer while all current tests still pass?

## 15. When to ask before acting

Ask the user before:

- changing the meaning of FOV or another sampling unit;
- changing required input/output contracts;
- changing a public API after release/use, or anything in `__init__.py` that changes `__all__`;
- changing an output format or the shared CSV columns, or making a change that requires updates in other MorphoBricks repos;
- adding a required heavy dependency;
- selecting or replacing the default scientific estimator;
- changing behavior that materially affects scientific interpretation;
- adding image-processing/model-extraction responsibilities to this repo;
- performing destructive or difficult-to-reverse Git operations.

Do not ask before:

- adding tests;
- improving error messages;
- adding docstrings/comments;
- fixing clear private implementation bugs;
- running tests/lint;
- making small internal changes explicitly required by the current task.

## 16. Completion report

For substantive work, report briefly:

- what changed;
- what scientific behavior changed, if any;
- which tests/validation actually ran;
- assumptions and known limitations;
- unresolved scientific questions that should block broader claims.

## 17. Sampling objectives, MorphoBricks defaults, and benchmarking

The sampling objective is separate from the mathematical strategy. Supported high-level objectives are population representation, rare-phenotype discovery with an explicit user target, and balanced sampling.

MorphoBricks metadata handling is centralized. When available, condition and repeat are kept as independent analysis groups by default and always remain part of FOV identity, even when pooling; image_file is preferred as the FOV identifier (one image file, including a multidimensional stack, is one FOV); and shared metadata columns such as mask_file and patch_file are preserved as metadata rather than phenotype features. Intentional pooling requires an explicit empty group list or another explicit grouping choice.

Strategies must not reimplement condition/repeat splitting. Each strategy receives one resolved analysis group at a time, and every result records its group identity, parameters, objective, object/FOV counts, and measured runtime.

Scientific and computational benchmarking are separate dimensions. Runtime is first-class because smart microscopy requires a decision before the next acquisition is needed. See docs/sampling_objectives.md, docs/benchmarking.md, and docs/architecture.md.

## 18. MorphoBricks pipeline position

```text
[Step 1] raw_data/ + masks/  ->  CellPatcher                  ->  patches/
[Step 2] patches/            ->  Feature extraction            ->  features.csv
[Step 3] features.csv        ->  Analysis & applications       ->  results
                                 **[MorphoCoverage]**
```

| Step | Repo | Description | Link |
| --- | --- | --- | --- |
| 1 | CellPatcher | Image patching | https://github.com/CellMigrationLab/CellPatcher |
| 2 | fiberfeatures (and other extractors, e.g. DINO, Nyxus, CellProfiler) | Feature extraction | https://github.com/CellMigrationLab/fiberfeatures |
| 3 | MorphoCoverage | FOV-aware sampling adequacy | https://github.com/CellMigrationLab/MorphoCoverage |

Inputs: an object feature table from any step-2 tool, plus a FOV table that lists every acquired image, including those without objects. Current step-1/step-2 outputs do not yet emit such a table; `docs/morphobricks_fov_contract.md` lists what they need to change.

Usage contexts: importable API and notebooks. A CLI is deferred (section 3).

## 19. Glossary

| Term | Definition |
| --- | --- |
| Condition | An experimental group, e.g. `treated`, `control` (MorphoBricks shared column). |
| Repeat | A biological or technical replicate, identified by date in MorphoBricks (shared column). |
| Image file / FOV | One acquired field of view: one image file, possibly a z/t/c stack. The primary acquisition and resampling unit. |
| FOV key | The columns that jointly identify one FOV: group columns, condition/repeat, then the FOV column. |
| Object | One detected object (cell, nucleus) within an FOV; a row of the object table. |
| Feature | An explicitly selected numeric descriptor of an object. |
| Analysis group | FOVs evaluated together; `condition` x `repeat` by default, pooled only on explicit request. |
| Zero-object FOV | An acquired FOV with no detected objects; it counts as acquisition effort. |
| Strategy | A swappable method that assesses the acquisition so far: estimators plus decision policies (section 20). |
| Estimator | Computes named evidence quantities from the visible acquisition; never decides. |
| Decision policy | Turns evidence and effort into continue/stop for one objective component. |
| Component | The part of an objective a quantity or policy addresses: `population` or `rare`. |
| config_id | A stable hash of an estimator's, policy's or strategy's class and parameters. |
| Objective | What "enough imaging" means: population representation, rare-phenotype discovery or balanced sampling. |
| Feature transform | Preprocessing (standardization, PCA, ...) fitted inside `estimate` on the visible prefix only; its fitted state is recorded as provenance. |
| Outcome metric | A retrospective, strategy-independent score of a stop against the full acquisition of the same group. |
| Dataset role | `development` (may be used for tuning) or `held_out` (never tuned on); declared in a `BenchmarkRegistry`. |
| Prefix / replay | The first k acquired FOVs of a real acquisition; replay evaluates strategies on successive prefixes. |
| Design effect | The variance of a mean with FOVs as the sampling unit, divided by its variance under random-cell sampling. |

## 20. Adding or swapping a strategy

Strategies are interchangeable: the framework never branches on a strategy's identity. A strategy is estimators plus decision policies; a new family implements an estimator. To add one:

1. Subclass `Estimator` in a module under `strategies/`. Give it a `name` (a constant, or a property that names the variant), a `parameters` property holding every setting that defines the estimate (JSON-serializable, since it defines `config_id`), and `estimate(data, *, objective) -> Evidence`.
2. Read phenotype values only through `data.feature_values(name)` or `data.feature_matrix()`, and FOV identity through `data.fov_key`. Treat the input as one analysis group; never split conditions or repeats.
3. Report each estimate as a named `Quantity` with its `direction`, `units` and `component` (`population` or `rare`). Report a quantity that cannot be computed yet as `insufficient_data`, and one that is undefined for these data as `not_applicable`, each with a reason; never NaN. Only claim the `rare` component for evidence about the objective's `RareTarget`. Never report effort (`n_fovs`, `n_objects`): the framework owns it.
4. Preprocess features only with a `FeatureTransform` from `morphocoverage.transforms`, declared as a parameter (use `transform.describe()` in `parameters`) and fitted inside `estimate` on the data received. Record `fitted.provenance()` in the diagnostics, and report quantities as `insufficient_data` when the transform cannot be fitted. Never fit on data from outside the call.
5. Do not decide inside the estimator. Thresholds belong to a `DecisionPolicy` (usually `ThresholdPolicy`), so they are explicit, sweepable with `redecide`, and part of `config_id`.
6. Keep `estimate` a pure recomputation: it runs on a fresh copy for every group and prefix, so instance state is discarded. Randomness takes an explicit seed parameter. Incremental support is optional: `start(template, *, objective)` returns an `EstimatorState` whose `update(new)` receives only each newly revealed FOV and must return the same evidence as `estimate` on the whole prefix. Full recomputation stays the reference implementation.
7. Document the source, any published parameters, and each deviation from the published method in the module docstring; expose non-standard choices as explicit parameters without defaults. To keep a published parameterization, add a `SamplingStrategy` subclass that pairs the estimator with its published policy, as the baselines do.
8. Export it from `strategies/__init__.py` and `morphocoverage/__init__.py` (ask first, section 15), and add a strategy using it to `STRATEGIES_UNDER_CONTRACT`. `test_every_package_strategy_is_under_contract` fails otherwise; it also covers exported estimators and policies. If it is a current-practice baseline, also add it to `baseline_comparison`.


### Multidimensional acquisition rule

When feature extraction is performed slice-by-slice or frame-by-frame, retain lowercase `z`, `c`, and/or `t` on object/feature rows as applicable. All such rows from one spatial acquisition position must share the same `position_id` (or another explicitly supplied `fov_col`). Sequential replay and benchmarking must reveal the whole position atomically. Extra z-planes/channels/time points increase information within a FOV; they do not count as additional sampled positions.
