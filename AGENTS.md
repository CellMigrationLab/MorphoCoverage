# AGENTS.md - MorphoCoverage

> This file is the authoritative engineering context for any AI agent or contributor working in this repository. Read it completely before taking action, and read `docs/project_aim.md` before changing any estimator, scientific claim, validation design, or stopping rule. Prefer repository evidence, tests, and documented scientific contracts over assumptions.

## 1. Project identity

| Field | Value |
| --- | --- |
| Repo | `CellMigrationLab/MorphoCoverage` |
| Purpose | Quantify FOV-aware phenotypic sampling completeness from precomputed object-level feature representations. |
| Ecosystem | MorphoBricks |
| Position | Downstream statistical analysis/application. It consumes features; it does not extract them. |
| Language | Python 3.11+ |
| Tests | `pytest` |
| Lint | `ruff check src tests` |
| Install | `python -m pip install -e ".[dev]"` |

MorphoCoverage is deliberately independent of CellPatcher, fiberfeatures, Nyxus, CellProfiler, DINO, Torch, microscopy file formats, segmentation tools, and microscope-control software.

## 2. Scientific question

The central questions are:

> Has this sample been imaged sufficiently for the stated analytical objective?

> Is another representative FOV likely to provide enough useful new information to justify acquiring it?

The exact mathematical solution is intentionally unresolved. See `docs/project_aim.md`.

The microscope acquires FOVs. Objects/cells are clustered observations inside those FOVs. Therefore cell number alone is not assumed to be an adequate measure of sampling effort when phenotype is spatially structured.

The hierarchy is:

```text
object/cell -> FOV -> optional technical/sample unit -> optional biological repeat -> optional condition
```

The package may support arbitrary grouping columns, but it must not assign biological meaning to them automatically.

## 3. Non-negotiable scientific rules

These rules override implementation convenience.

- **FOV identity is mandatory for the primary MorphoCoverage analysis.** Do not silently degrade to object-level sampling when FOV information is absent.
- **FOV is an operational acquisition unit, not an automatic independence assumption.** Preserve spatial coordinates, area, and design metadata when available; whole-FOV resampling does not by itself eliminate spatial autocorrelation.
- **Do not assume one mathematical solution.** Ecological coverage, missing mass, clustering, nearest-neighbour novelty, distributional convergence, Bayesian methods, information-theoretic methods, and simple empirical rules remain candidate approaches until comparative validation supports a default.
- **Current cell-number guidance must be benchmarked fairly.** Reproduce the scoped scalar precision problem before arguing that FOV-aware sampling adds information; identify regimes where the simpler approach is sufficient.
- **Permit null results.** Do not optimize validation, examples, or documentation to guarantee that MorphoCoverage outperforms fixed cells/FOVs or scalar precision rules.
- **FOV is the primary acquisition/resampling unit.** Never bootstrap, permute, or sequentially reveal individual objects when an analysis is defined at FOV level.
- **Never infer feature columns from all numeric columns.** Feature columns must be supplied explicitly. Numeric metadata such as IDs, stage positions, acquisition order, pixel size, or timepoints must never silently enter phenotype space.
- **Metadata is not phenotype.** Grouping columns such as condition, repeat, batch, well, sample, or acquisition identifiers must not enter the feature matrix unless the caller explicitly supplies them as features.
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
- caller-specified FOV identifier column.

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

## 7. Initial repository structure

```text
src/morphocoverage/
    __init__.py
    schema.py

tests/
    test_schema.py

docs/
    scientific_contract.md
    project_aim.md
```

Future modules should be added only when the corresponding behavior is implemented and tested. Do not pre-create a `novelty`, `coverage`, or estimator architecture that implicitly selects a scientific solution before comparative validation.

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
- Contract tests check correct handling, not scientific performance. Whether a strategy stops early enough, or detects the clustered rare state, is a benchmarking result on real data. Such expectations must not be encoded as contract tests that force a strategy to "win".
- `baseline_comparison` covers the fixed-cell, fixed-FOV and JCB-style scalar-precision baselines (`ScalarPrecisionStrategy`). Any new current-practice baseline must be added to it.

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
- changing a public API after release/use;
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
