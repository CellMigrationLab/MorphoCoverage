# AGENTS.md - MorphoCoverage

> This file is the authoritative engineering context for any AI agent or contributor working in this repository. Read it completely before taking action, and read `docs/project_aim.md` before changing any estimator, scientific claim, validation design, or stopping rule. Prefer repository evidence, tests, and documented scientific contracts over assumptions.

## 1. Project identity

| Field | Value |
| --- | --- |
| Repo | `CellMigrationLab/MorphoCoverage` |
| Purpose | Quantify FOV-aware phenotypic sampling completeness from precomputed phenotype feature tables. |
| Ecosystem | MorphoBricks |
| Position | Step 3 of the MorphoBricks pipeline (analysis & applications). It consumes features; it does not extract them. See section 18. |
| Language | Python 3.11+ |
| Tests | `pytest` (all tests, including real-data tests on `data/crick_idr`; runs on every CPU with pytest-xdist, `pytest -n 0` runs serially) |
| Lint | `ruff check src tests data` and `ruff format --check src tests data` (ruff pinned in `pyproject.toml`; lines at most 100 characters) |
| Install / environment | `python -m pip install -e ".[dev]"` |

Key dependencies: `numpy`, `pandas`, `scipy`, `matplotlib` (runtime; imported only by `morphocoverage.plots` and `workflows/figures.py`, so importing `morphocoverage.workflows` loads it while the core, `retrospective` and `sampling_unit` do not; added 2026-09-26 at the project owner's request for the PlotsOfData-style figures); `pytest`, `ruff` (development, with `pytest-xdist` to run the suite on every CPU). Prefer a well-established implementation from these libraries over hand-written numerics. There are deliberately no scikit-learn or Torch dependencies: Torch is held back until an estimator demonstrably needs GPU acceleration, and would then be an optional backend with an explicit `device` that raises when unavailable, never a silent CPU fallback.

MorphoCoverage is deliberately independent of CellPatcher, fiberfeatures, Nyxus, CellProfiler, DINO, Torch, microscopy file formats, segmentation tools, and microscope-control software.

## 2. Scientific question

The central questions are:

> Has this sample been imaged sufficiently for the stated analytical objective?

> Is another representative FOV likely to provide enough useful new information to justify acquiring it?

The exact mathematical solution is intentionally unresolved. See `docs/project_aim.md`. The microscopist's questions, grouped by moment of use (REPORT, OPTIMISE, ONLINE) with the information each may use, are listed in `docs/questions.md`, starting with the Foundation: whether the field, not only the object, must be counted as the sampling unit; new analyses should name the question they answer there.

The microscope samples spatial acquisition positions. In MorphoCoverage, **one FOV means one independently selected XY acquisition position**, regardless of how many z-planes, channels, or time points were acquired there. Feature observations such as cells, nuclei, patches, slice-level measurements, embeddings, channels and time points are nested inside that position. Therefore cell/object/plane count alone is not assumed to be an adequate measure of spatial sampling effort.

The hierarchy is:

```text
observation (cell/nucleus/patch/slice/FOV summary/embedding) -> FOV = spatial acquisition position -> optional technical/sample unit -> optional biological repeat -> optional condition
```

The package may support arbitrary grouping columns, but it must not assign biological meaning to them automatically.

## 3. Non-negotiable rules

These rules override implementation convenience.

MorphoBricks ecosystem rules (shared with the other MorphoBricks repositories):

- **Never rename shared CSV columns** (`condition`, `repeat`, `technical_repeat`, `position_id`, `image_file`, `mask_file`, `patch_file`). Other repos depend on them. MorphoCoverage rejects near-miss spellings such as `Repeat` rather than renaming them.
- **Never hard-code machine-specific paths.** Use function arguments, config or environment variables; tests locate repository data relative to the test file.
- **Never break public interfaces** (anything in `morphocoverage.__all__`) without explicit user approval.
- **Keep public functionality usable as an imported dependency and from notebooks.** A CLI is deliberately deferred: the project owner asked to avoid premature CLI work until the scientific core stabilizes (see section 18). Add one only when asked.

Scientific rules:

- **FOV identity is mandatory for the primary MorphoCoverage analysis.** FOV means one independently selected spatial acquisition position, not a z-plane, channel, time point, or file. Do not silently degrade to object-, plane-, or frame-level sampling when position identity is absent.
- **FOV is an operational acquisition unit, not an automatic independence assumption.** Preserve spatial coordinates, area, and design metadata when available; whole-FOV resampling does not by itself eliminate spatial autocorrelation.
- **Do not assume one mathematical solution.** Ecological coverage, missing mass, clustering, nearest-neighbour novelty, distributional convergence, Bayesian methods, information-theoretic methods, and simple empirical rules remain candidate approaches until comparative validation supports a default.
- **Current cell-number guidance must be benchmarked fairly.** Reproduce the scoped scalar precision problem before arguing that FOV-aware sampling adds information; identify regimes where the simpler approach is sufficient.
- **Retrospective representability is not the prospective sampling requirement.** A completed dataset may be represented by a few well-chosen FOVs, while an experiment that samples representative FOVs needs more to meet the same criterion reliably. Never report a best or minimal subset as a recommendation, and never without the prospective requirement beside it. Planning advice ("how many FOVs next time") must rest on what subsets actually preserved in hidden FOVs, judged by strategy-independent outcome metrics, not on how often a stopping rule says enough.
- **Permit null results.** Do not optimize validation, examples, or documentation to guarantee that MorphoCoverage outperforms fixed cells/FOVs or scalar precision rules.
- **FOV/spatial position is the primary acquisition/resampling unit.** Never bootstrap, permute, or sequentially reveal individual objects, z-planes, channels, or time points as independent FOVs. All observations from one position must enter together.
- **Never infer feature columns from all numeric columns.** Feature columns must be supplied explicitly. Numeric metadata such as IDs, stage positions, acquisition order, pixel size, or timepoints must never silently enter phenotype space.
- **Feature rows are generic observations, not necessarily cells.** Record `observation_unit` (for example `cell`, `nucleus`, `patch`, `slice`, `fov_summary`); labels are normalized to lowercase singular form, so `Cells` and `nuclei` mean `cell` and `nucleus`. When no unit is given, rows are assumed to be objects and every result records `observation_unit_declared=False`. FOV count and observation count are different quantities. Object/cell-specific scientific rules must not silently reinterpret arbitrary observations as cells.
- **Missing phenotype values are allowed at the input boundary.** NaN is preserved; infinity is rejected. The framework never silently imputes or drops missing values. Every estimator/transform that uses affected features must state its missing-data behavior explicitly.
- **Metadata is not phenotype.** Grouping/acquisition columns such as condition, repeat, batch, well, sample, acquisition identifiers, and within-position `z`, `c`, `t` metadata must not enter the phenotype matrix when they encode acquisition dimensions. Feature columns are explicit, so callers must omit such metadata from `feature_cols`; validation refuses `z`, `z0`, `z1`, `c` and `t` there, and a feature measured per plane becomes one feature per plane only on explicit request (`planes_as_features`).
- **Never pool groups silently.** If the caller requests analysis by condition, repeat, or other groups, analyses operate independently within those groups. Pooling must be explicit.
- **Do not claim that more observations from one FOV equal more independent imaging coverage.** Observation count and FOV count are different quantities and must remain separately visible.
- **Preserve acquired FOVs with zero phenotype observations.** An observation-only table is not a complete acquisition record; support a first-class FOV table.
- **Never report a coverage/novelty result without enough provenance to reproduce the phenotype geometry**, including supplied feature columns, preprocessing, distance metric, and morphological resolution/threshold when applicable.
- **Retrospective and prospective analyses are different.** Retrospective analysis may use future/held-out FOVs to measure realized outcomes. A prospective estimator must use only information available up to the current acquisition step.
- **No future-data leakage.** Preprocessing, tuning, threshold selection, dimensionality reduction, and hyperparameter optimization for prospective evaluation must not use future FOVs unless explicitly labelled retrospective/transductive.
- **Scientific estimator changes require scientific tests.** Every estimator must pass constructed contract/counterexample scenarios with known behavior (rare phenotypes, spatial clustering, variable observations/FOV, zero-observation FOVs, technical shifts) and must ultimately be evaluated on deeply sampled real datasets. Constructed tests protect contracts; they are not the primary scientific benchmark.
- **Random procedures must be reproducible.** Expose `random_state` or equivalent; tests use fixed seeds.
- **Do not equate low expected novelty with proof that no rare biology exists.** The package estimates sampling adequacy under an explicit representation and sampling scheme; it cannot guarantee absence of arbitrarily rare unseen states.
- **Smart microscopy starts with continue/stop representative acquisition.** Do not conflate this with targeted next-FOV selection. Targeted acquisition changes the sampling distribution and requires separate treatment.

## 4. Scope boundaries

### This repo owns

- validation of phenotype-observation feature tables and complete FOV metadata;
- feature-space preprocessing required to define phenotype geometry;
- distance/similarity calculations used by candidate methods;
- observation- and FOV-level novelty/statistics where scientifically justified;
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
- owning the user's final biological-condition inference workflow;
- microscope hardware control.

MorphoCoverage does not replace the user's final biological inference workflow. It does describe, in the report, how the declared conditions differ in the acquired fields (`sampling_unit.condition_contrast`, decided by the project owner on 2026-09-26: Cohen's d and the JCS effect size on objects, as the lab's Plot&Stats notebook computes them, and with the field as the unit, plus a randomization test that shuffles condition labels over whole technical repeats when each side has at least two, with no test when a side has one, and over whole fields only without a technical-repeat column), always per repeat and always beside the field-aware version, and worded as what these fields support, never as the inference across independent repeats. It may also use condition-level analyses (effect sizes, phenotype-frequency differences, treatment rankings, classification or retrieval) as **retrospective benchmark outcomes**, to test whether a stopping rule preserves the biological conclusion that the full acquisition supports. A strategy can preserve a centroid or a distribution distance and still erase a treatment-specific minority state that decides the result. Such outcomes are computed on held-out or full data after the stop, are never inputs to a prospective estimator, and do not turn the package into a hypothesis-testing tool. It may also use **multi-group planning comparators** (such as the JCB multi-sample planning path) when these are needed to evaluate sampling adequacy. Core prospective estimators remain group-local unless a method is explicitly implemented as a separate multi-group planning step.

Do not add dependencies for those concerns merely to make an example easier.

## 5. Inter-repo contract

MorphoCoverage consumes two linked logical tables.

### FOV table

One row per acquired FOV, including FOVs with zero phenotype observations.

Required:
- caller-specified spatial-position/FOV identifier column. Prefer `position_id` when storage layout does not map one file to one position; `image_file` is a fallback only when one file equals one position.

Optional:
- condition, repeat, technical_repeat (a well, a dish, a coverslip within a repeat), batch;
- acquisition order;
- stage coordinates;
- FOV dimensions or physical imaged area;
- sampling-design metadata.

### Observation table

One row per phenotype observation. Rows may represent cells, nuclei, patches, slice-level measurements, FOV summaries, embeddings, or another declared unit.

Required:
- the same FOV identifier;
- explicitly supplied numeric phenotype feature columns;
- an `observation_unit` declaration in the API when the rows are not ordinary objects.

Optional:
- object/label/patch identifiers and z/c/t coordinates for traceability.

The backwards-compatible API argument is still named `objects`, but `ValidatedInput.observations` and `n_observations` are the canonical generic semantics.

MorphoCoverage must be able to consume output derived from CellPatcher or any feature-extraction repo without importing those packages.

Where practical, preserve compatibility with MorphoBricks metadata names such as `condition`, `repeat`, and `image_file`, but the core API accepts caller-specified column names rather than hard-coding them.

### Shared CSV column contract (MorphoBricks)

| Column | MorphoBricks requirement | How MorphoCoverage uses it |
| --- | --- | --- |
| `condition` | string, always | Default analysis group; always part of FOV identity. |
| `repeat` | string, always (`YYYY_MM_DD`, or `single`) | Default analysis group; always part of FOV identity. Any label is accepted; the format is not enforced. |
| `technical_repeat` | string, optional | The technical repeat within a biological repeat (a well, a dish, a coverslip); never a biological repeat. When present: a default analysis group and part of FOV identity, counted in the report's overview, the colour of the plots' points and the unit their intervals and the condition contrast and feature ranking resample or shuffle first (the contrast and the ranking take the FOV table's column whatever `group_cols` says), and the sample level of the hierarchy `Setup` derives for "more fields or more technical repeats?". Identified with its `condition` and `repeat`, so well `J23` of two plates is two technical repeats. |
| `image_file` | string, always | Storage provenance; fallback FOV identifier only when one file = one spatial position. |
| `position_id` | string, when needed | Preferred explicit spatial-position identifier when files/planes/scenes do not map 1:1 to FOVs. |
| `mask_file` | string, if used | Preserved as metadata, never a feature. |
| `patch_file` | string, if used | Preserved as metadata, never a feature. |

The user-facing description of both tables, every mandatory and optional column, and the validation rules is [`docs/data_format.md`](docs/data_format.md); keep it in step with this section, with `schema.validate_input_tables` and with `data_check.check_tables` (which must accept exactly what the validator accepts). `tests/test_data_check.py` checks that agreement, and `tests/test_docs.py` runs the page's example and checks that every shared column is listed.

An optional shared column that is empty in every row (`position_id`, `technical_repeat`: `metadata.EMPTY_MEANS_ABSENT`) is read as absent, because producers that write every column, CellPatcher among them, leave it empty when it does not apply; a partly empty one still raises, and one named explicitly in `group_cols`/`fov_col` is kept; when only the observation table's copy is empty, the FOV table's values are used. Group metadata such as `condition`/`repeat` may live only in the FOV table when the supplied FOV identifier (with the other group columns, when field names restart in each repeat) maps unambiguously to one metadata value; the validator then attaches it to observation rows. A MorphoBricks column in the observations but not in the FOV table raises. If an FOV identifier is reused across different metadata values, callers must include the grouping columns on the observation rows or use a globally unique FOV identifier. Non-canonical spellings still raise unless named explicitly. Columns MorphoCoverage adds or recognises beyond the shared contract (`fov_id` as a fallback FOV identifier, `label_id`, acquisition order, stage coordinates, area) are documented in `docs/architecture.md` and `docs/morphobricks_fov_contract.md`. The test data in `data/crick_idr/` uses plate IDs as `repeat`, since no dates exist in the source (see its README).

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
    metadata.py            # MorphoBricks column resolution: FOV key, groups, spellings (the one resolver)
    schema.py              # ValidatedInput and linked-table validation
    _plane_rules.py        # private: the plane rule (z/c/t planes are not fields, never features)
    planes.py              # planes_as_features: one feature per plane, objects matched by label_id
    integrity.py           # exact data-integrity checks: copied, reordered or scaled fields, repeated blocks and summaries
    data_check.py          # check_tables: every data problem at once, where and how to fix it
    _data_rules.py         # private: the rules behind check_tables, one function per kind of problem
    _data_problems.py      # private: the problems a data check reports (Problem, Collector) and their wording
    _input_traps.py        # private: tables that validate but are probably not meant: invalid rows, undeclared groups, unimaged wells
    objectives.py          # sampling objectives and RareTarget, independent of strategies
    evidence.py            # Evidence and Quantity: estimates with direction, units, component
    policies.py            # decision policies, per-objective combination, redecide
    provenance.py          # config_id hashing
    results.py             # StrategyResult and the Decision vocabulary
    evaluator.py           # group iteration, isolation, timing, framework-built results
    sequential.py          # leakage-safe acquisition-prefix replay
    online.py              # live OnlineSession: add one newly acquired spatial FOV at a time
    outcomes.py            # retrospective group-local outcome metrics, strategy-independent
    multivariate_outcomes.py  # distribution-level outcome metrics (centroid ... MMD)
    distances.py           # pure two-sample distance primitives (energy, MMD, Fréchet, SW)
    transforms.py          # feature transforms fitted on the visible prefix only
    benchmark.py           # dataset roles, acquisition orders, one record per run
    reservoir.py           # held-out evaluation reservoirs split by spatial block
    spatial.py             # spatial diagnostics: FOV semivariogram against stage distance
    bounds.py              # Clopper-Pearson bounds and intervals for probabilities from few FOVs
    framework.py           # run_strategies convenience entry point
    _areas.py              # private: the one check of an imaged-area column (benchmark, fixed-area rule)
    _deprecated.py         # private: the former cell names, resolved with a DeprecationWarning
    _parallel.py           # private: worker pools behind n_jobs (map_units, worker_pool)
    _wording.py            # private: the one way to write percentages, rates, counts, fields, durations, groups
    _seeds.py              # private: per-group seeds (group_identity, group_seed, group_rng, split_seed) and group_key
    _checks.py             # private: integer and number checks that accept numpy types
    calibration.py         # shuffle check of the next-FOV new-phenotype prediction
    sufficiency.py         # did we sample enough; would one more FOV show something new
    sufficiency_text.py    # the sentences of the sufficiency report
    sufficiency_tables.py  # its criteria tables: each rule on the full acquisition and its stops
    statistics.py          # shared statistics, written once: effect sizes, tolerances, whole-field sampling and bootstrap, variance components, design effect, JCB counts, detection counts, field-level JCB estimands
    normalize.py           # features divided by the control condition's mean, repeat by repeat
    plots.py               # PlotsOfData-style figures, intervals from resampling whole fields (matplotlib)
    retrospective/         # what a completed acquisition achieved, judged on hidden FOVs
        __init__.py        # the subpackage's public names
        criteria.py        # OutcomeCriterion: outcome metric + declared tolerance
        hierarchy.py       # SamplingHierarchy: declared FOV / sample / repeat levels
        reliability.py     # sampling reliability: how often k random FOVs are enough
        reliability_text.py  # its sentences
        replay.py          # replayed orders against a fixed reservoir: decisions and outcomes
        calibration.py     # stopping-rule calibration: how often a rule's stop was premature
        adequacy.py        # first and sustained apparent and realized adequacy points
        protocol.py        # protocol-level reliability across samples, by independent unit
        protocol_text.py   # its sentences
        frontier.py        # the benchmark score: fields saved at matched risk, transfer
        frontier_text.py   # its sentences
        inspection.py      # inspect-these-fields: one traceable reason per FOV and line
        controls.py        # representation controls: detection and false-alarm rates
        controls_text.py   # their sentences
        effort.py          # nested variance components and projected SE at equal FOVs
        effort_design.py   # its standard errors and allocations: acquired and balanced designs
        effort_text.py     # the effort projection in words (technical repeats, samples, caveats)
        stage.py           # dispersed (farthest-point) orders and stage travel
        spatial.py         # acquisition-order efficiency and redundancy distance
        report.py          # the microscopist's report, question by question
        report_profile.py  # its sampling profile, one helper per source, and the control rates
        overview.py        # what is in the dataset: counts, features, conditions, repeats, technical repeats
        _text.py           # private: retrospective wording (samples, units, order phrases); re-exports _wording
    sampling_unit/         # why count fields? the Foundation and R4 of docs/questions.md
        __init__.py        # the subpackage's public names
        variability.py     # FOV variability: variance components, design effect, unusual FOVs
        object_rules.py    # JCB rule and JCS descriptor spread with random objects and whole FOVs
        contrast.py        # each condition against the control: objects, fields and technical repeats as units
        ranking.py         # which features separate the conditions: BH-gated |d| ranking (as fiberfeatures)
        _units.py          # private: sides, units and the randomization test shared by contrast and ranking
        clustering.py      # unusual (out-of-distribution) objects; state clustering by FOV
        _outliers.py       # private: the distances and leave-one-group-out thresholds behind unusual_objects
        _common.py         # private: validation and group helpers shared by the sampling-unit checks
    workflows/             # a new dataset in a few lines: report, optimise, benchmark, live
        __init__.py        # the subpackage's public names
        setup.py           # Setup: every scientific choice of the workflows, declared once
        _declaration.py    # private: the Setup's declaration in sentences, every preset printed
        retrospective.py   # report, optimise, document: the analyses in question order
        benchmarks.py      # benchmark, Family, default_families: the rule families at matched risk
        _document.py       # private: the assembled report of the workflows, its parts and settings
        figures.py         # plots: the readout and the best-ranked features, from a Setup
        live.py            # live_session; LiveFolder (files) and LiveTables (in memory)
        _shared.py         # private: input checks, orders, rule replays on raw values, outcomes on normalised ones
    strategies/
        __init__.py        # the exported strategies, estimators and policies
        base.py            # Estimator interface; SamplingStrategy = estimators + policies
        baselines.py       # fixed-FOV, fixed-area, generic-observation and object-count baselines
        scalar_precision.py  # JCB/Zweifach-style scalar precision baseline
        _averaging_cv.py   # private: its averaging-CV arithmetic (object and FOV units)
        stabilization.py   # Marcotti et al.-style descriptor-stabilization baselines
        _descriptor_resamples.py  # private: the resample arithmetic of the descriptor spread
        jcb_planning.py    # JCB multi-sample planning from disjoint pilot samples
        fov_bootstrap.py   # FOV-bootstrap uncertainty of multivariate population summaries
        _fov_resamples.py  # private: its resample arithmetic, every summary of every resample at once
        rare_planning.py   # binomial FOV-incidence planning for a RareTarget
        rare_incidence.py  # beta-binomial rare-state incidence risk with recurrence
        support_novelty.py # leave-one-FOV-out epsilon-support novelty: estimator and strategy
        support_geometry.py  # its counting: which observations other FOVs support within epsilon
        incidence_coverage.py  # Chao-Jost incidence coverage and Chao2 on fixed labels

tests/
    scenarios.py           # constructed acquisitions for contract tests (section 9)
    contract_registry.py   # every packaged strategy, metric and transform under contract
    test_contracts.py      # section 9 scenarios: contracts every strategy and the framework pass
    test_contracts_*.py    # outcome-metric, transform and incremental contracts
    test_guards.py         # one guard per recurring bug class, package-wide (section 12)
    test_<module>.py       # unit tests per module or strategy family
    test_audit*.py, test_microscopist_audit.py  # regression tests of the audits' findings (docs/audit/README.md)
    test_human_errors.py   # the mistakes people make with the tables, and the message each gives
    test_crick_idr.py      # real-data integrity and end-to-end tests
    test_parallel.py       # n_jobs > 1 gives the results of n_jobs = 1
    test_docs.py           # the documentation's examples run and print what they say
    test_notebooks.py      # example notebooks compile, import existing names, carry no outputs
    conftest.py            # one BLAS thread per test process

A new strategy family goes in its own module under `strategies/` (section 20). Future modules should be added only when the corresponding behavior is implemented and tested. Do not pre-create a `novelty`, `coverage`, or estimator architecture that implicitly selects a scientific solution before comparative validation. Data-preparation scripts live next to their data, not in the package.

## 8. Development sequence

Do not skip directly to a polished `analyze()` API.

1. Strict linked FOV/observation input contracts and validation.
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

Constructed data are still required for **contract tests**: acquisitions whose structure is known by construction, used to prove that the framework and every strategy handle that structure correctly. The scenarios below are implemented in `tests/scenarios.py` and exercised by `tests/test_contracts.py` (with `test_contracts_*.py` for metrics, transforms and incremental updates; what is under contract is listed in `tests/contract_registry.py`). Each backticked id must have either a fixture or a named framework contract; `test_agents_section_9_matches_contract_tests` fails if this list and the tests drift apart.

Required scenarios:

- homogeneous phenotype distribution: `homogeneous`;
- clearly separated multimodal phenotypes: `multimodal`;
- rare states at controlled prevalence, e.g. 10%, 1%, 0.1%: `rare_prevalence_10pct`, `rare_prevalence_1pct`, `rare_prevalence_0p1pct`;
- identical rare-state prevalence distributed uniformly versus clustered into few FOVs: `rare_uniform`, `rare_clustered`;
- unequal numbers of observations per FOV: `unequal_objects_per_fov`;
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
- Every strategy added to the package must be added to `STRATEGIES_UNDER_CONTRACT` in `tests/contract_registry.py`. It then has to pass, on every scenario: counts that match the acquisition record, determinism, and group independence. Framework-level contracts (whole-FOV reveal in acquisition order, no future-FOV leakage, explicit features only, zero-observation FOVs counted as effort) apply to all strategies automatically.
- An estimator that supports incremental updates (`Estimator.start`) must give, at every step and on every scenario, the same evidence and decisions under `incremental=True` replay as under full recomputation, to floating-point rounding. The incremental-equivalence contract in `tests/test_contracts_incremental.py` runs this for every strategy in `STRATEGIES_UNDER_CONTRACT`.
- Every group-local outcome metric added to the package must be added to `METRICS_UNDER_CONTRACT`. It has to return a finite float or None (never NaN), be deterministic, leave its inputs unchanged, and depend only on the full acquisition and the stop prefix, never on which strategy stopped.
- Future multi-group biological-fidelity outcomes require a **separate contract registry and contract-test module** rather than being silently added to `METRICS_UNDER_CONTRACT`. Those contracts must verify determinism, invariance to strategy identity at matched stop prefixes, preservation of the declared biological-repeat unit, no mutation of inputs, and strict retrospective-only use. Until that registry exists, multi-group outcomes remain proposal/benchmark design work rather than packaged metrics.
- Every feature transform added to the package must be added to `TRANSFORMS_UNDER_CONTRACT`. Its fitted state and output at step k must not change when FOVs acquired after step k are altered, and its fitted state must be JSON-serializable provenance.
- Contract tests check correct handling, not scientific performance. Whether a strategy stops early enough, or detects the clustered rare state, is a benchmarking result on real data. Such expectations must not be encoded as contract tests that force a strategy to "win".
- `baseline_comparison` covers the fixed-cell, fixed-FOV, fixed-area (`FixedAreaStrategy`), JCB-style scalar-precision (`ScalarPrecisionStrategy`, and the multi-sample `PlannedObjectCountStrategy`) and Marcotti-style descriptor-stabilization (`DescriptorPlateauStrategy`, `DescriptorSpreadStrategy`) baselines. Any new current-practice baseline must be added to it.

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

Prospective deployment uses `OnlineSession.add_fov(observations, fov_metadata)`. The session receives exactly one newly acquired spatial position at a time and must never require identifiers, metadata, or phenotype values for future FOVs. The first FOV fixes the automatically resolved FOV identifier and grouping contract for that session; later FOVs may not silently change it. Online and retrospective replay must use the same estimators and policies; incremental estimators must remain equivalent to full recomputation.

Only after representative continue/stop is validated should the package support:

> Where should the microscope image next?

If acquisition becomes targeted, do not treat targeted FOVs as an unbiased population sample. Use representative/sentinel FOVs, explicit sampling probabilities, or another validated design.

## 12. Coding standards

- PEP 8 style.
- 4-space indentation.
- Descriptive names.
- Type hints on every function, private helpers included (decided 2026-09-26; the whole package meets it).
- Docstrings on public modules/functions/classes.
- No module above 600 lines: split a growing module along its natural seam (computation and wording, geometry and decision), re-exporting the public names so callers do not change.
- Statistics shared by several analyses live once, in `morphocoverage.statistics`, and are reused rather than re-derived.
- The bug classes the audits kept finding are guarded package-wide by `tests/test_guards.py`, so write new code through the shared helpers: percentages and declared rates through `_wording` (`percent`, `rate`), count and number checks through `_checks`, FOV and group identity only through `metadata`'s resolvers, group lookups by `_seeds.group_key` (never the 32-bit `group_identity`, which only seeds), and every new interval with a coverage test. The live session must give the replay's result at every step of every contract scenario, and `check_tables` must accept exactly what the validator accepts.
- Comments explain scientific assumptions or non-obvious reasoning.
- Prefer specific exception types and actionable error messages.
- Do not perform unrelated refactors, dependency upgrades, API renames, or formatting sweeps.

## 13. Verification

For substantive code changes, when the environment permits:

```bash
pytest
ruff check src tests data
ruff format --check src tests data
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

The sampling objective is separate from the mathematical strategy. Its components are the targets the microscopist declares: the population summary, the phenotypic fingerprint (the `distribution` component) and the rare phenotype with an explicit user target. The objectives are `population`, `distribution`, `rare_discovery`, `phenotype` (summary and fingerprint), `balanced` (summary and rare) and `all`. A rule with no evidence about a required component is `unsupported` for the objective and never `stop`; its own verdict is `StrategyResult.rule_decision`, which self-assessments, replays and calibrations read (`docs/sampling_objectives.md`).

MorphoBricks metadata handling is centralized. When available, condition, repeat and technical_repeat are kept as independent analysis groups by default and always remain part of FOV identity, even when pooling; `position_id` is preferred as the spatial FOV identifier and `image_file` is only a fallback when one file equals one position; shared metadata columns such as mask_file and patch_file are preserved as metadata rather than phenotype features. Intentional pooling requires an explicit empty group list or another explicit grouping choice.

Strategies must not reimplement condition/repeat splitting. Each core prospective strategy receives one resolved analysis group at a time, and every result records its group identity, parameters, objective, FOV count, observation count/unit, and measured runtime. Methods that require several groups belong in an explicit planning/comparator layer (as with JCB multi-sample planning), not inside a normal group-local estimator. Retrospective benchmark outcomes may compare groups when the purpose is to test whether a stopping rule preserves the biological analysis. This scope expansion is deliberate: it changes the repository contract so that biological-condition comparisons are permitted **only** as retrospective benchmark outcomes or explicit planning comparators, never as hidden inputs to ordinary prospective estimators.

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

Inputs: a phenotype-observation feature table from any step-2 tool, plus a FOV table that lists every acquired spatial position, including positions without observations. Current step-1/step-2 outputs do not yet emit such a table; `docs/morphobricks_fov_contract.md` lists what they need to change.

Usage contexts: importable API and notebooks. A CLI is deferred (section 3).

## 19. Glossary

| Term | Definition |
| --- | --- |
| Condition | An experimental group, e.g. `treated`, `control` (MorphoBricks shared column). |
| Repeat | A biological repeat, identified by date in MorphoBricks (shared column `repeat`). |
| Technical repeat | A well, dish or coverslip within a biological repeat (shared optional column `technical_repeat`): objects of one technical repeat share its preparation, so it is the unit resampled or shuffled before fields, and the sample level of the derived hierarchy. |
| FOV / spatial position | One independently selected XY acquisition position. A file, z-plane, channel or timepoint is not automatically a separate FOV. |
| FOV key | The columns that jointly identify one FOV: group columns, condition/repeat, then the FOV column. |
| Observation | One row of phenotype information nested in an FOV; may be a cell, nucleus, patch, slice, FOV summary, embedding, or another declared unit. |
| Feature | An explicitly selected numeric descriptor of an object. |
| Analysis group | FOVs evaluated together; `condition` x `repeat` x `technical_repeat` (those present) by default, pooled only on explicit request. |
| Zero-observation FOV | An acquired spatial position with no phenotype observations; it still counts as acquisition effort. |
| Strategy | A swappable method that assesses the acquisition so far: estimators plus decision policies (section 20). |
| Estimator | Computes named evidence quantities from the visible acquisition; never decides. |
| Decision policy | Turns evidence and effort into continue/stop for one objective component. |
| Component | The part of an objective a quantity or policy addresses: `population`, `distribution` or `rare`. |
| config_id | A stable hash of an estimator's, policy's or strategy's class and parameters. |
| Objective | What "enough imaging" means: `population`, `distribution`, `rare_discovery`, `phenotype` (summary and fingerprint), `balanced` (summary and rare) or `all` (section 17). |
| Feature transform | Preprocessing (standardization, PCA, ...) fitted inside `estimate` on the visible prefix only; its fitted state is recorded as provenance. |
| Outcome metric | A retrospective, strategy-independent score of a stop against the full acquisition of the same group. |
| Dataset role | `development` (may be used for tuning) or `held_out` (never tuned on); declared in a `BenchmarkRegistry`. |
| Prefix / replay | The first k acquired FOVs of a real acquisition; replay evaluates strategies on successive prefixes. |
| Design effect | The variance of a mean with FOVs as the sampling unit, divided by its variance under random-cell sampling. |

## 20. Adding or swapping a strategy

Strategies are interchangeable: the framework never branches on a strategy's identity. A strategy is estimators plus decision policies; a new family implements an estimator. To add one:

1. Subclass `Estimator` in a module under `strategies/`. Give it a `name` (a constant, or a property that names the variant), a `parameters` property holding every setting that defines the estimate (JSON-serializable, since it defines `config_id`), and `estimate(data, *, objective) -> Evidence`.
2. Read phenotype values only through `data.feature_values(name)` or `data.feature_matrix()`, and FOV identity through `data.fov_key`. Treat the input as one analysis group; never split conditions or repeats.
3. Declare the estimator's valid `observation_unit` semantics. Observation-unit-agnostic methods, including FOV-resampling variants, accept any unit. Cell/object-derived methods must never reinterpret slices, patches, embeddings or FOV summaries as cells: they report their quantities as `not_applicable`, with the reason from `unsupported_observation_unit`, so the decision is undefined and a benchmark or live session keeps running.
4. Report each estimate as a named `Quantity` with its `direction`, `units` and `component` (`population` for a summary of the population, `distribution` for the phenotypic fingerprint: spread, shape, the states present; `rare` for a declared `RareTarget`). Report a quantity that cannot be computed yet as `insufficient_data`, and one that is undefined for these data as `not_applicable`, each with a reason; never NaN. Only claim the `rare` component for evidence about the objective's `RareTarget`. Never report effort (`n_fovs`, `n_observations`; `n_objects` is only a compatibility alias): the framework owns it.
5. Preprocess features only with a `FeatureTransform` from `morphocoverage.transforms`, declared as a parameter (use `transform.describe()` in `parameters`) and fitted inside `estimate` on the data received. Record `fitted.provenance()` in the diagnostics, and report quantities as `insufficient_data` when the transform cannot be fitted. Never fit on data from outside the call.
6. Do not decide inside the estimator. Thresholds belong to a `DecisionPolicy` (usually `ThresholdPolicy`), so they are explicit, sweepable with `redecide`, and part of `config_id`.
7. Keep `estimate` a pure recomputation: it runs on a fresh copy for every group and prefix, so instance state is discarded. Randomness takes an explicit seed parameter. Incremental support is optional: `start(template, *, objective)` returns an `EstimatorState` whose `update(new)` receives only each newly revealed FOV and must return the same evidence as `estimate` on the whole prefix. Full recomputation stays the reference implementation.
8. Document the source, any published parameters, and each deviation from the published method in the module docstring; expose non-standard choices as explicit parameters without defaults. To keep a published parameterization, add a `SamplingStrategy` subclass that pairs the estimator with its published policy, as the baselines do.
9. Export it from `strategies/__init__.py`; adding it to the top-level `morphocoverage.__all__` needs approval (section 15), and candidate estimators currently stay in the subpackage. Add a strategy using it to `STRATEGIES_UNDER_CONTRACT`. `test_every_package_strategy_is_under_contract` fails otherwise; it also covers exported estimators and policies. If it is a current-practice baseline, also add it to `baseline_comparison`.


### Multidimensional acquisition rule

When feature extraction is performed slice-by-slice or frame-by-frame, retain lowercase `z`, `c`, and/or `t` on observation rows as applicable. All such rows from one spatial acquisition position must share the same `position_id` (or another explicitly supplied `fov_col`). Sequential replay and benchmarking must reveal the whole position atomically. Extra z-planes/channels/time points increase information within a FOV; they do not count as additional sampled positions.

Validation enforces this rather than trusting it:
- a position spread over several `image_file` values must carry `z`/`c`/`t` values that tell those files apart;
- when the FOV column was chosen automatically as `image_file` or `fov_id`, and each FOV holds a single value of a `z`/`c`/`t` column that varies across FOVs, validation raises instead of counting planes as FOVs (a warning with one row per FOV; a FOV-table `z` or `t` counts only when a value recurs in several FOVs, since a stage height distinct in every FOV is not a plane); supply `position_id` or name `fov_col` explicitly;
- `z`, `z0`, `z1`, `c` and `t` are never features. To compare a feature between planes, `planes_as_features` turns it into one feature per plane (`area_z0`, `area_z1`, ...), objects matched across planes by `label_id` within the field; the field stays one FOV. The live session applies the same rules to every FOV so far.
