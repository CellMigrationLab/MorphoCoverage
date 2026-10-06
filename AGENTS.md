# AGENTS.md - MorphoCoverage

> This file is the authoritative engineering context for any AI agent or contributor working in this repository. Read it completely before taking action, and read `docs/common/project_aim.md` before changing any estimator, scientific claim, validation design, or stopping rule. Prefer repository evidence, tests, and documented scientific contracts over assumptions.

## 1. Project identity

| Field | Value |
| --- | --- |
| Repo | `CellMigrationLab/MorphoCoverage` |
| Purpose | Quantify FOV-aware phenotypic sampling completeness from precomputed phenotype feature tables. |
| Ecosystem | MorphoBricks |
| Position | Step 3 of the MorphoBricks pipeline (analysis & applications). It consumes features; it does not extract them. See section 18. |
| Language | Python 3.11+ |
| Tests | `pytest` (all tests, including real-data tests on `data/crick_idr`; runs on every CPU with pytest-xdist, `pytest -n 0` runs serially) |
| Lint | `python -m ruff check src tests data` and `python -m ruff format --check src tests data` (ruff pinned in `pyproject.toml`; run it as `python -m ruff`, since a `ruff` on the PATH may be another version and format differently; lines at most 100 characters) |
| Install / environment | `python -m pip install -e ".[dev]"` |

Key dependencies: `numpy`, `pandas`, `scipy`, `matplotlib` (runtime; imported only by `morphocoverage.plots`, `plots_embedding` and `workflows/figures.py`, so importing `morphocoverage.workflows` loads it while the core, `retrospective` and `sampling_unit` do not; added 2026-09-26 at the project owner's request for the PlotsOfData-style figures), `umap-learn` (runtime, required; imported only inside `embedding._umap`, since loading it takes seconds; added 2026-10-01 at the project owner's request for the UMAP beside the PCA, which brings scikit-learn and numba in as its own dependencies), `scikit-learn` (runtime, required, 1.3 or later for HDBSCAN; imported only inside `sampling_unit.populations._cluster` and `sampling_unit._population_rule` (a group's feature rule), with umap-learn inside `populations._embedding`; declared 2026-10-02 at the project owner's request to find the populations of question 10); `pytest`, `ruff` (development, with `pytest-xdist` to run the suite on every CPU). Prefer a well-established implementation from these libraries over hand-written numerics (Benjamini-Hochberg is scipy's `false_discovery_control`; a 2026-10-02 survey kept the rest hand-written for a stated reason: field-nested resampling, incremental moments, exact enumeration, a U-statistic, or speed). The package imports scikit-learn in one place only (HDBSCAN for the populations), and there is deliberately no Torch dependency: Torch is held back until an estimator demonstrably needs GPU acceleration, and would then be an optional backend with an explicit `device` that raises when unavailable, never a silent CPU fallback.

MorphoCoverage is deliberately independent of CellPatcher, fiberfeatures, Nyxus, CellProfiler, DINO, Torch, microscopy file formats, segmentation tools, and microscope-control software.

## 2. Scientific question

The central questions are:

> Has this sample been imaged sufficiently for the stated analytical objective?

> Is another representative FOV likely to provide enough useful new information to justify acquiring it?

The exact mathematical solution is intentionally unresolved. See `docs/common/project_aim.md`. The microscopist's questions, grouped by workflow (REPORT, OPTIMISE, LIVE) with the information each may use, are listed in `docs/report/questions.md` (R1-R8, starting with the Foundation: whether the field, not only the object, must be counted as the sampling unit), `docs/optimise/optimise.md` (P1-P10) and `docs/live/live_session.md` (O1-O9), from `docs/README.md`; new analyses should name the question they answer there.

The microscope samples spatial acquisition positions. In MorphoCoverage, **one FOV means one independently selected XY acquisition position**, regardless of how many z-planes, channels, or time points were acquired there. Feature observations such as cells, nuclei, patches, slice-level measurements, embeddings, channels and time points are nested inside that position. Therefore cell/object/plane count alone is not assumed to be an adequate measure of spatial sampling effort.

The hierarchy is:

```text
observation (cell/nucleus/patch/slice/FOV summary/embedding) -> FOV = spatial acquisition position -> optional technical/sample unit -> optional biological repeat -> optional condition
```

The package may support arbitrary grouping columns, but it must not assign biological meaning to them automatically.

## 3. Non-negotiable rules

These rules override implementation convenience.

MorphoBricks ecosystem rules (shared with the other MorphoBricks repositories):

- **Never rename shared CSV columns** (`position_id`, `image_file`, `z` or `z0`/`z1`, `c`, `t` (all 0-based), `condition`, `biological_repeat`, `technical_repeat`, and `mask_file`, `patch_file` as provenance). Other repos depend on them. The biological repeat's column was renamed from `repeat` to `biological_repeat` by the project owner on 2026-09-30; the former name is still read, with a `DeprecationWarning`, at the input boundary only (`_former_repeat`, section 5). MorphoCoverage rejects near-miss spellings such as `Biological_Repeat`, `bio_repeat` or `tec_repeat` ("did you mean ...?") rather than renaming them.
- **Never hard-code machine-specific paths.** Use function arguments, config or environment variables; tests locate repository data relative to the test file.
- **Never break public interfaces** (anything in `morphocoverage.__all__`) without explicit user approval.
- **Keep public functionality usable as an imported dependency and from notebooks.** A CLI is deliberately deferred: the project owner asked to avoid premature CLI work until the scientific core stabilizes (see section 18). Add one only when asked.

Scientific rules:

- **FOV identity is mandatory for the primary MorphoCoverage analysis.** FOV means one independently selected spatial acquisition position, not a z-plane, channel, time point, or file. Do not silently degrade to object-, plane-, or frame-level sampling when position identity is absent.
- **FOV is an operational acquisition unit, not an automatic independence assumption.** Preserve spatial coordinates, area, and design metadata when available; whole-FOV resampling does not by itself eliminate spatial autocorrelation.
- **Do not assume one mathematical solution.** Ecological coverage, missing mass, clustering, nearest-neighbour novelty, distributional convergence, Bayesian methods, information-theoretic methods, and simple empirical rules remain candidate approaches until comparative validation supports a default.
- **Current cell-number guidance must be benchmarked fairly.** Reproduce the scoped scalar precision problem before arguing that FOV-aware sampling adds information; identify regimes where the simpler approach is sufficient.
- **Retrospective representability is not the prospective sampling requirement.** A completed dataset may be represented by a few well-chosen FOVs, while an experiment that samples representative FOVs needs more to meet the same criterion reliably. Never report a best or minimal subset as a recommendation, and never without the prospective requirement beside it. Planning advice ("how many FOVs next time") must rest on what subsets actually preserved in hidden FOVs, judged by strategy-independent outcome metrics, not on how often a stopping rule says enough.
- **A usual design is not flagged.** Three biological repeats, one technical repeat (well, dish) per condition and repeat, and options left at their defaults (no fingerprint, rare phenotype, acquisition order, stage coordinates or representation control) are described, never warned about. A ⚠️ caution is reserved for a measured property of these data or a missing declaration that kept a check from running; everything that holds for every dataset is a standing assumption (`_caveats`: `assumption(...)`, decided 2026-09-28). A test the design cannot make informative is left out, not flagged: it runs only when its smallest possible p can reach 0.05 after Bonferroni (`sampling_unit._reach`). The tone is that of a colleague, never condescending (owner, 2026-09-30).
- **A dataset is one experiment, pooled as provided.** There is no experiment level above the biological repeat; whatever the user supplies is analysed as one experiment (owner, 2026-09-30). Its sub-experiments (the conditions sharing their repeats and control, `sampling_unit._experiments`) are derived from the design for the correction only, never a level the user declares (owner, 2026-10-01).
- **Markers say what an answer found.** ✅ yes (a condition differs across repeats, enough was imaged, the data are clean), ⚠️ partly, a trend only or something to look at, ❌ check the data first, ➖ no conclusion possible here; the chip's word says which ("Differs", "Trend only", "Not yet", "No conclusion (n = 1)"). The verdict across repeats is "significant across repeats" (the paired t-test, with n and q beside it), never "confirmed"; the report is neutral: with several features, "Do the conditions differ?" is answered by all of them alike (✅ when any is significant across repeats for a condition, named), the readout being what optimise and live target, not the report's answer; question 10 finds groups of cells on every cell pooled (HDBSCAN on a UMAP of all declared features, each cluster kept when its cells' neighbours on all of them are in it; whether it is a separate state is for the user to validate on the images, no gap test being reliable, and each comes with the simplest feature rule that labels it, owner 2026-10-02), then asks whether a condition is enriched in one across repeats, and the PCA and UMAP ring them, colour by any feature and describe a selection by its features (owner, 2026-10-02); with a rare label its share is compared like a feature and the label checks the features (question 1); with a fingerprint, the distributions are compared beyond the mean (spread and quantiles, tested across repeats); both targets are answered in "Did I image enough?" from the fields acquired (owner, 2026-10-02); "Did I image enough?" likewise asks every feature the same questions (a table across the features, then one feature at a time, saying when it is enough as well as when it is not; on the page one sentence and a table, the rest in the tabs), the stopping rules staying on the readout, as optimise and live run them (in the text report; on the page, in the plan of the next experiment); the cell-level test is an exploratory screen of candidates; a test that pools the wrong control gets no number in the reading; the design is said in lab words (wells, not technical repeats, when there is one per condition and repeat) (owner, 2026-10-01, after a biologist's reading of the reports).
- **The HTML page guides the next step** (owner, 2026-10-02). It reads: the data (one row per check), the fields to check and why, the whole dataset (PCA/UMAP), populations, whether the conditions differ (the significant features per condition first, the 1-2 repeat cases said as such, each figure with its legend), whether enough was imaged, and last **Plan the next experiment** (`_html_plan`): the reader picks what the next experiment should get right (a feature's mean, a group or the rare phenotype seen in enough fields, the fingerprint: `optimise`; a difference from the control: the repeats needed) or what the microscope should stop on (`live_session`, on a group alone with `objective`), and gets the code filled in. Each answer comes first, its purpose under a "?" and the rest under "Show more"; an answer that calls for a look at the images (fields to check, a group) ends with "What to do next"; the code runs on `result`, `objects`, `fovs` (tested by executing it), and no card repeats the plan's. The plots are the best-ranked features, the readout only when it ranks or none does; every figure carries its data as CSV (`_html_data`).
- **Pilots get advice, not verdicts.** A dataset of one or two biological repeats gets every answer its design supports, the cell-level test leading question 9, and what to change next time within the design the user ran (fields a side per repeat, technical repeats per condition), never a blanket "image 5-6 repeats". The repeats a paired test would need lead the plan, at 0.05/m with m the conditions of the sub-experiment, the correction the verdict uses at most (owner, 2026-10-01; before, full report only at 0.05).
- **Permit null results.** Do not optimize validation, examples, or documentation to guarantee that MorphoCoverage outperforms fixed cells/FOVs or scalar precision rules.
- **FOV/spatial position is the primary acquisition/resampling unit.** Never bootstrap, permute, or sequentially reveal individual objects, z-planes, channels, or time points as independent FOVs. All observations from one position must enter together.
- **Never infer feature columns from all numeric columns.** Feature columns must be supplied explicitly; an other feature table, passed by name, declares every column but its identity columns (section 5, owner 2026-10-05). Numeric metadata such as IDs, stage positions, acquisition order, pixel size, or timepoints must never silently enter phenotype space.
- **Feature rows are generic observations, not necessarily cells.** Record `observation_unit` (for example `cell`, `nucleus`, `patch`, `slice`, `fov_summary`); labels are normalized to lowercase singular form, so `Cells` and `nuclei` mean `cell` and `nucleus`. When no unit is given, rows are assumed to be objects and every result records `observation_unit_declared=False`; the report's words then call them cells (`schema.unit_words`, owner 2026-10-02: "cells could be the default"), and with a declared unit every word for the observations, the cell-level test's name included ("patch-level"), is the unit's (`Setup.unit_words`/`unit_word`, `observation_words` of a result's settings). FOV count and observation count are different quantities. Object/cell-specific scientific rules must not silently reinterpret arbitrary observations as cells.
- **Missing phenotype values are allowed at the input boundary.** NaN is preserved; infinity is rejected. The framework never silently imputes or drops missing values. Every estimator/transform that uses affected features must state its missing-data behavior explicitly.
- **Metadata is not phenotype.** Grouping/acquisition columns such as condition, biological repeat, batch, well, sample, acquisition identifiers, and within-position `z`, `c`, `t` metadata must not enter the phenotype matrix when they encode acquisition dimensions. Feature columns are explicit, so callers must omit such metadata from `feature_cols`; validation refuses `z`, `z0`, `z1`, `c` and `t` there, and a feature measured per plane becomes one feature per plane only on explicit request (`planes_as_features`).
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

MorphoCoverage does not replace the user's final biological inference workflow, but the report answers "Do the conditions differ, and how confidently?" (question 9, `sampling_unit.condition_contrast`; owner decisions of 2026-09-26, 2026-09-30 and 2026-10-01; the tests, their assumptions and limitations and the simulation evidence are in `docs/report/conditions.md`). Per condition against the declared control, it gives:

- **across biological repeats**, the repeat as the replicate (SuperPlots): the direction in every repeat, the per-repeat Cohen's d, and from 3 repeats the paired t-test on the repeat means, raw and on the log ratios to the control of the same repeat (`contrast_repeats`); the exact sign-flip test only where it can reach 0.05 after Bonferroni (from 6 repeats, full report);
- **the lab's Plot&Stats cell-level test**: Cohen's d and a randomization test over the observations pooled as provided, on raw values and on values divided by the mean of the same repeat's control, labelled with its chance of p <= 0.05 without an effect, `2·P(Z > 1.96/√(1 + n·s²/2))` (`n` the observations per well, each repeat weighted by its share; `s` the SD between repeats), checked against `data/simulations/test_choice.py` and `unbalanced.py` (`contrast_cells`). When the repeats are unbalanced (one holds more than 25% more observations than another on a side, or the control is pooled from repeats the condition was not imaged in) it is also given balanced: against the control of the condition's own repeats, one permutation test whose d is the median over 100 draws of the same number of observations from every repeat and whose p shuffles the condition and control labels within each repeat, the same statistic on the same draws for each shuffle, `(extreme + 1) / (B + 1)` (owner, 2026-10-05; before, the median p over the draws, which is not the p of one test). A repeat under a quarter of the median repeat (or with fewer than 5 observations) is named and left out of the balanced test only;
- **within each repeat**, Cohen's d and the JCS effect size on objects, field means and technical-repeat means, with a randomization test over whole technical repeats only when its smallest p can reach 0.05 after Bonferroni (4 a side for one contrast), over whole fields only without a technical-repeat column;
- with a `target` column, whether independent reagents for one target are on the same side of the control in every repeat (described, not tested);
- **one reading per condition** that puts these together (`reading_text`): significant across repeats (the paired test, normalised, q <= 0.05 after Benjamini-Hochberg within the sub-experiment: the conditions sharing their repeats and control, `_experiments`, owner 2026-10-01), same direction in every repeat but not significant, no consistent direction, or one repeat; then the evidence, and when the raw pooled test disagrees, what it does differently;
- planning the next experiment at the design the user ran (`repeat_plan`; see "Pilots get advice", section 3);
- **with several features, which of them separate the conditions** (`discovery`, owner decision of 2026-10-01): every feature and condition by the cell-level test in the version the readings trust most (normalised, balanced when unbalanced), Benjamini-Hochberg over all of them within each sub-experiment as the discovery gate, ranked by |d|; the paired t on the repeat means, corrected the same way, beside it as the second gate ("significant across repeats"), and the direction in each repeat. The plots draw the best-ranked (`plots(..., ranking=report.discovery)`), and `embedding_plots` the PCA (the best-ranked as arrows) and UMAP of all the features. The ranking over whole technical repeats (`ranking`) is shown only when there are wells to shuffle.

**Several controls** (owner, 2026-10-05): `Setup(control={"condition": {control: {treatment: condition, ...} or [conditions], ...}}, control_parents={control: parent or (parent, treatment)}, master_control=...)` declares several experiments; one control is unchanged, and one control declared in these forms without parents is reduced to it, so its output is identical (owner, 2026-10-05). The declaration generates one **contrast table** (`workflows/_contrasts`; columns `_control.CONTRAST_COLUMNS`), one row per test, shown in question 9 and saved as `contrasts.csv`; `Setup(contrasts=table)` takes one in its place. `own_control` rows (each condition against its own control, each control against its parent) are the main verdicts: the tests above, the cell-level test, "Did I image enough?", the discovery and the heatmap use them only (pairs resolved by `sampling_unit._groups`). A condition has one `own_control` row, from the declaration when there is one (a table giving it two, or another, is refused). The secondary rows are tested on the repeat means for every explainable feature, no cell-level test, each kind one Benjamini-Hochberg family over every feature and its rows (`sampling_unit.contrast_across.across_features`, `workflows._families.shared_across`, words in `across_text`: the readout's rows first, then the other features' significant ones and a count; saved as `tests_across_experiments.csv`; owner, 2026-10-05): `across_normalised` (the same treatment key under different controls, each condition normalised to its own control: the difference of the log ratios per repeat holding all four) and `vs_master_raw` (with a master, each other control against it on raw values, unless a parent link joins them; the master's repeats and the assumption that experiments were imaged and processed alike said once); a row held in fewer than 3 repeats is named once as not compared. Two controls whose medians of an intensity (a declared column named "intensity", or listed in `intensity_features`) differ 2-fold or more are said once as a ⚠️ caution among the checks of the values and in question 9, and the secondary rows joining them are named under their table, the main verdicts joining them (a control against its parent) under the readings (`workflows/_acquisition_gap`, owner 2026-10-05). Figures, PCA/UMAP, population detection and SuperPlots show every condition; normalised views divide each condition by its own control (a control by its parent, else 1), the conditions left out named once, and their pairwise panels compare only conditions normalised to the same control (≠ "different references: not compared", owner 2026-10-05); unusual observations and population enrichment use each condition's own control, the object rules, clustering and the positive control every control pooled.

The cell-level test answers whether these wells differ, not whether the effect repeats: its false-positive chance is always printed beside it, and only the paired test across repeats confirms a difference. With a control, question 2 asks whether the fields acquired resolve the difference from the control in each repeat (`contrast_sampling`: a bootstrap over wells and fields as acquired, empty fields included, and the fields a side a resolution would need, projected from the part of the interval within wells; when the part between wells alone is as wide as the difference, "more fields will not resolve it: more wells or biological repeats", owner 2026-10-05).

It may also use condition-level analyses (effect sizes, phenotype-frequency differences, treatment rankings, classification or retrieval) as **retrospective benchmark outcomes**, to test whether a stopping rule preserves the biological conclusion that the full acquisition supports. A strategy can preserve a centroid or a distribution distance and still erase a treatment-specific minority state that decides the result. Such outcomes are computed on held-out or full data after the stop, are never inputs to a prospective estimator, and do not turn the package into a hypothesis-testing tool. It may also use **multi-group planning comparators** (such as the JCB multi-sample planning path) when these are needed to evaluate sampling adequacy. Core prospective estimators remain group-local unless a method is explicitly implemented as a separate multi-group planning step.

Do not add dependencies for those concerns merely to make an example easier.

## 5. Inter-repo contract

MorphoCoverage consumes two linked logical tables.

### FOV table

One row per acquired FOV, including FOVs with zero phenotype observations.

Required:
- caller-specified spatial-position/FOV identifier column. Prefer `position_id` when storage layout does not map one file to one position; `image_file` is a fallback only when one file equals one position.

Optional:
- condition, biological_repeat, technical_repeat (a well, a dish, a coverslip within a biological repeat), batch;
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

### Other feature tables

Optional, by name, for the same observations (owner, 2026-10-05): features that are not read one by one, such as a deep-learning embedding ("DINOv2", "DINOv3"), so several models can be compared side by side. The observation table holds the explainable features, tested one by one as before; the other tables are never tested feature by feature: each is **tested whole** instead, beside the per-feature tests (owner, 2026-10-06; `sampling_unit.whole_table`, `workflows._whole_table`, `docs/report/conditions.md`): per own-control comparison of the contrast table (the rows across experiments are tested feature by feature only, said once) and table, in its robustly standardised space, field means first, the repeat shifts `Δ_r` projected on the other repeats' mean direction and a one-sided paired t that the projections are positive, its t multiplied by `sqrt((n − 2) / (2 (n − 1)))` for the repeats the projections share (the plain t rejected 8-22% of nulls; corrected, 2.3-7.5%, `data/simulations/whole_table.py`), from 3 repeats, Benjamini-Hochberg per table and sub-experiment apart from the per-feature families; beside it an exploratory field-level permutation of the labels within repeats (anti-conservative under a well batch, said) and the shift in within-group SDs; named in the conditions' headline when significant, saved as `tests_whole_table.csv`. They join the maps, the populations, the copy checks and their own fingerprints, never novelty or unusual observations, which read the explainable features only (owner, 2026-10-06, after audit B: the former radius times the square root of the number of tables called 99.6% of null cells new). A wide table's fingerprint reads its first k principal components, fitted on all observations robustly standardised, k at most a fifth of the smallest judged group's observations and at most its columns, said in the line; not assessable below k = 2; explainable features at most that wide keep today's form (owner, 2026-10-06; `workflows._fingerprint_space`, `multivariate_outcomes.ComponentsError`). **Embedding-only reports** (owner, 2026-10-06): `Setup(readout=None)` with no features reads the tables alone in `report`, `html_report`, `scan_design` and `guide` (`workflows._embedding_report`): the whole-table test is the only test of question 9, the targets are the fingerprint of each table and a labelled phenotype as the user chooses (no averages), the mean-precision rules not applicable (said once, in the declaration), each section needing explainable features "not applicable: no explainable features" once with ➖, unusual observations on the tables' MFA-weighted space (the report says which); `optimise` judges those targets (no rule to replay) and `live_session` needs explicit rules. The maps: one per table and a weighted combined map, first, beside a view of each other table centred per repeat (each column minus a median of its repeat; maps, and with normalize the tables' analyses, never tests; centred on the median of all of a repeat's observations when every repeat holds the same conditions, else on its control, only repeats whose control holds at least 5 observations, `CENTRED_MIN_CONTROL`, the others left out and named once with one line on the noise a small control adds; which centring, said in one line; with `normalize` declared the tables' populations, unusual observations and separation read the centred values, never a test, owner 2026-10-06). The weighted combined space (maps and population detection) is MFA (owner, 2026-10-06): each table robustly standardised (median and MAD, the SD where the MAD is under a quarter of the SD, so a column of mostly zeros cannot carry thousands of times the others' variance) and divided by its own first singular value, the explainable features keeping weight 1; under the combined PCA each table's share of each axis ("PC1: explainable 42%, DINOv2 58%"). The populations are searched in each space apart (the explainable features, each table, the combined space; HDBSCAN in each, the other spaces on at most 5,000 observations); each distinct group is reported once, naming the spaces that found it, groups of two spaces merged at a Jaccard index of 0.5 or more; a group the explainable features did not find has no rule ("found in DINOv2 only; no rule in the explainable features describes it"); the enrichment tests run on the merged list, one Benjamini-Hochberg family. The copy checks run on each table apart, the union reported with the table each copy was found in. A declared fingerprint runs per table (the guided session declares one criterion per table, the same tolerance; "The fingerprint of DINOv2 (...)"). A descriptive separation score per table (`separation`, no p-value; owner, 2026-10-05 and 2026-10-06): by condition, per biological repeat, the share of each observation's k nearest neighbours with its condition against chance, the median over repeats; by repeat, per condition, the share with its repeat against chance, the median over conditions, the line saying when the repeats separate more than the conditions; neighbours never from the observation's own field, nor its own well when its condition has another well in the repeat, chance computed over what remains; k = min(10, the fewest such neighbours of its own label any observation of the group has), so a perfect separation reads 100%. It is a line under each map and a comparison table on the page (header "feature table", the combined space apart, not ranked). They are passed as `feature_tables={name: table}` to `report`, `html_report`, `scan_design` and `guide` and checked at loading by `attach_feature_tables` (`feature_tables.py`): each row identified as the observation table identifies an observation (the FOV key columns it holds, then `label_id` and `z`/`z0`/`z1`/`c`/`t` when present; matched as text, a whole number written as a decimal, `1.0`, read as `1`), every observation matched exactly once, every other column numeric (the table is the explicit declaration of its features; metadata names are refused), no infinite value, no column without a value, missing values counted and kept (each analysis reading the table leaves the incomplete observations out and says how many), constant columns listed, no column shared with the observation table or another table, a non-empty name of its own (`explainable` and `combined` are reserved); every problem raised at once. `Setup.feature_tables` records each table's columns. They are never divided by a control.

MorphoCoverage must be able to consume output derived from CellPatcher or any feature-extraction repo without importing those packages.

Where practical, preserve compatibility with MorphoBricks metadata names such as `condition`, `biological_repeat`, and `image_file`, but the core API accepts caller-specified column names rather than hard-coding them.

### Shared CSV column contract (MorphoBricks)

| Column | MorphoBricks requirement | How MorphoCoverage uses it |
| --- | --- | --- |
| `condition` | string, always | Default analysis group; always part of FOV identity. |
| `biological_repeat` | string, always (`YYYY_MM_DD`, or `single`) | Default analysis group; always part of FOV identity. Any label is accepted; the format is not enforced. Formerly `repeat`, which is read as `biological_repeat` with a `DeprecationWarning` (below). |
| `technical_repeat` | string, optional | The technical repeat within a biological repeat (a well, a dish, a coverslip); never a biological repeat. When present: a default analysis group and part of FOV identity, counted in the report's overview, the shape of the plots' points and the unit their intervals and the condition contrast and feature ranking resample or shuffle first (the contrast and the ranking take the FOV table's column whatever `group_cols` says), and the sample level of the hierarchy `Setup` derives for "more fields or more technical repeats?". Identified with its `condition` and `biological_repeat`, so well `J23` of two plates is two technical repeats. |
| `image_file` | string, always | Storage provenance; fallback FOV identifier only when one file = one spatial position. |
| `position_id` | string, when needed | Preferred explicit spatial-position identifier when files/planes/scenes do not map 1:1 to FOVs. |
| `z` or `z0`/`z1`, `c`, `t` | integer, 0-based, when a position has several planes, channels or time points | Within-FOV coordinates: never FOVs, never features (section 20, multidimensional acquisition rule). |
| `mask_file` | string, if used | Preserved as metadata, never a feature. |
| `patch_file` | string, if used | Preserved as metadata, never a feature. |

**A missing hierarchy level is written as the value `single`** (CellPatcher's convention: the level exists with one member), for example `biological_repeat="single"` for a condition without repeats; it is an ordinary label, one member of that level. An empty cell keeps meaning "not recorded".

**The former name `repeat`.** Tables that still use `repeat` are accepted with one `DeprecationWarning` per call ("'repeat' is the former name of 'biological_repeat'; rename the column ... it will be removed in a later version"), at the input boundary only: `_former_repeat` is called where tables and column arguments enter (the validator, `check_tables`, the workflows, `OnlineSession`, `LiveTables`, `LiveFolder` manifests, and the public functions that read a column argument themselves). A `repeat` column in a table without `biological_repeat` is read as `biological_repeat`; the caller's tables are not modified and results use `biological_repeat`. A table with both columns is refused unless they are identical (then warned about, and `biological_repeat` used). `group_cols`, `fov_col`, `control` and `Setup`/`SamplingHierarchy` fields naming `repeat` are translated the same way. The biological repeat seeds random draws under its former name (`_seeds`), so the renaming changed no result. Near misses of either repeat (`bio_repeat`, `biological repeat`, `Metadata_biological_repeat`, `tec_repeat`, `tech_rep`, `Repeat`, `repeats`, ...) are refused by `metadata.canonical_spelling` with "did you mean ...?", never pooled.

The user-facing description of both tables, every mandatory and optional column, and the validation rules is [`docs/common/data_format.md`](docs/common/data_format.md); keep it in step with this section, with `schema.validate_input_tables` and with `data_check.check_tables` (which must accept exactly what the validator accepts). `tests/test_data_check.py` checks that agreement, and `tests/test_docs.py` runs the page's example and checks that every shared column is listed.

An optional shared column that is empty in every row (`position_id`, `technical_repeat`: `metadata.EMPTY_MEANS_ABSENT`) is read as absent, because producers that write every column, CellPatcher among them, leave it empty when it does not apply; a partly empty one still raises, and one named explicitly in `group_cols`/`fov_col` is kept; when only the observation table's copy is empty, the FOV table's values are used. Group metadata such as `condition`/`biological_repeat` may live only in the FOV table when the supplied FOV identifier (with the other group columns, when field names restart in each repeat) maps unambiguously to one metadata value; the validator then attaches it to observation rows. A MorphoBricks column in the observations but not in the FOV table raises. If an FOV identifier is reused across different metadata values, callers must include the grouping columns on the observation rows or use a globally unique FOV identifier. Non-canonical spellings still raise unless named explicitly. Columns MorphoCoverage adds or recognises beyond the shared contract (`fov_id` as a fallback FOV identifier, `label_id`, acquisition order, stage coordinates, area) are documented in `docs/common/architecture.md` and `docs/common/morphobricks_fov_contract.md`. The test data in `data/crick_idr/` uses plate IDs as `biological_repeat`, since no dates exist in the source (see its README); its stored files keep the former name `repeat`, which `datasets.load` renames on load.

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
    _control.py            # private: the control resolved in one place: control rows, pooled controls, labels, the contrast table's columns
    _spelling.py           # private: typing errors of a name (edits), for metadata's near-miss spellings
    schema.py              # ValidatedInput and linked-table validation
    _plane_rules.py        # private: the plane rule (z/c/t planes are not fields, never features)
    planes.py              # planes_as_features: one feature per plane, objects matched by label_id
    integrity.py           # exact data-integrity checks: copied, reordered or scaled fields, repeated blocks and summaries
    _integrity_match.py    # private: its matching at a stated precision, by sorting, with no loop over pairs of fields
    _integrity_text.py     # private: the integrity checks in words (IntegrityResult.summary), short and full
    _integrity_tables.py   # private: the integrity checks on each feature table apart, their union naming the table
    data_check.py          # check_tables: every data problem at once, where and how to fix it
    _data_rules.py         # private: the rules behind check_tables, one function per kind of problem
    _copy_rules.py         # private: its copy rule: repeated rows, observations repeated within a field, labels twice in a field
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
    order_check.py         # acquisition-order check beside every stop: readout trend with order, Moran's I on the stage (flags, never vetoes)
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
    _former_repeat.py      # private: the former column name `repeat`, read as `biological_repeat` with a DeprecationWarning
    _parallel.py           # private: worker pools behind n_jobs (map_units, worker_pool)
    _wording.py            # private: the one way to write percentages, rates, counts, fields, durations, groups
    _style.py              # private: the one text style: markers and Status, rule and group labels, tables, wrapping
    _caveats.py            # private: the two kinds of caveat (cautions about these data, standing assumptions) and how a section states them
    _seeds.py              # private: per-group seeds (group_identity, group_seed, group_rng, split_seed) and group_key
    _checks.py             # private: integer and number checks that accept numpy types
    calibration.py         # shuffle check of the next-FOV new-phenotype prediction
    sufficiency.py         # did we sample enough; would one more FOV show something new
    sufficiency_text.py    # the sentences of the sufficiency report (its answer columns)
    sufficiency_short.py   # its short summary: the rules in one table across groups, each caveat once
    sufficiency_full.py    # its full summary: every number, group by group, each caveat once
    sufficiency_tables.py  # its criteria tables: each rule on the full acquisition and its stops
    statistics.py          # shared statistics, written once: effect sizes, tolerances, whole-field sampling and bootstrap, variance components, design effect, JCB counts, detection counts, field-level JCB estimands
    _repeat_statistics.py  # private, re-exported by statistics: the paired test across repeats, the values it compares (log ratios or differences), Sarle's bimodality and kurtosis
    normalize.py           # features divided by the control condition's mean, biological repeat by biological repeat
    plots.py               # PlotsOfData-style figures, intervals from resampling whole fields (matplotlib)
    glossary.py            # the report's glossary: every test, rule and term in plain words, with its papers
    embedding.py           # PCA and UMAP of the declared features (embed): coordinates and loadings, no figure
    feature_tables.py      # other feature tables (a DINO embedding, ...): validated against the observation table, attached as named blocks; their MFA weights in the combined space
    separation.py          # how far the conditions, and the repeats, separate in a feature space: the share of each observation's neighbours, outside its field, with its condition (repeat) against chance (descriptive)
    plots_embedding.py     # the PCA biplot (the top features as arrows) and the UMAP, coloured by condition
    plots_pairwise.py      # every pair of conditions: signed d and the paired test, the panel under a plot
    _plots_text.py         # private: a plot's title (what is plotted, what the interval resamples), legend
    _palette.py            # private: the colour per condition, technical-repeat shapes, axis style
    _figure_layout.py      # private: a static figure's lettered panels and its legend at the bottom
    retrospective/         # what a completed acquisition achieved, judged on hidden FOVs
        __init__.py        # the subpackage's public names
        criteria.py        # OutcomeCriterion: outcome metric + declared tolerance
        hierarchy.py       # SamplingHierarchy: declared FOV / sample / biological-repeat levels
        reliability.py     # sampling reliability: how often k random FOVs are enough
        reliability_text.py  # its sentences
        effect_preservation.py  # k fields a side against held-back fields: is each repeat's difference from the control kept?
        replay.py          # replayed orders against a fixed reservoir: decisions and outcomes
        calibration.py     # stopping-rule calibration: how often a rule's stop was premature
        adequacy.py        # first and sustained apparent and realized adequacy points
        adequacy_text.py   # its words: "what did the extra fields buy?", short and full
        protocol.py        # protocol-level reliability across samples, by independent unit
        protocol_text.py   # its sentences
        frontier.py        # the benchmark score: fields saved at matched risk, transfer
        frontier_bootstrap.py  # private: its resampling of independent units, interval withheld below 5 units
        frontier_text.py   # its sentences
        inspection.py      # inspect-these-fields: one traceable reason per FOV and line
        inspection_text.py # its words: the fields in a table (short), every reason per group (full)
        controls.py        # representation controls: detection and false-alarm rates
        controls_text.py   # their sentences
        effort.py          # nested variance components and projected SE at equal FOVs
        effort_design.py   # its standard errors and allocations: acquired and balanced designs
        effort_text.py     # the effort projection in words (technical repeats, samples, caveats)
        stage.py           # dispersed (farthest-point) orders and stage travel
        spatial.py         # acquisition-order efficiency and redundancy distance
        order_text.py      # its words: a table per order kind (short), per group and order (full)
        report.py          # the microscopist's report, question by question
        report_profile.py  # its sampling profile, one helper per source, and the control rates
        report_status.py   # the status marker and headline of every answer, question by question
        report_needs.py    # what each question not assessed needs, short and in a sentence
        report_status_plan.py  # the status of the planning answers 4, 5 and 7 (re-exported by report_status)
        report_status_conditions.py  # the status and headline of question 9 (the readings, reagents, ranking)
        report_plan.py     # questions 4, 5 and 7 in their short form
        report_resolved.py # question 2 with a control: do the fields acquired resolve the difference?
        report_enough.py   # question 2, short and full: sufficiency, detection-adjusted bound, calibration
        report_answers.py  # questions 6, 8 and 9 in short, caveats pooled; the closing "cannot tell me"
        report_sampling.py  # the Foundation and questions 1 and 3, short and full, and the counts their headlines read
        overview.py        # what is in the dataset: counts, features, conditions, repeats, technical repeats
        _value_checks.py   # private: checks of the values: fields more alike than chance, features with few distinct values
        _feature_relations.py # private: declared features that repeat one measurement: copies, exact affine relations, near copies, on all rows and within each of several controls' experiments (owner, 2026-10-05)
        _text.py           # private: retrospective wording (samples, units, order phrases); re-exports _wording
    sampling_unit/         # why count fields? the Foundation and R4 of docs/report/questions.md
        __init__.py        # the subpackage's public names
        variability.py     # FOV variability: variance components, design effect, unusual FOVs
        object_rules.py    # mean-precision rule and JCS descriptor spread with random objects and whole FOVs
        contrast.py        # each condition against the control: per repeat, across repeats, cell-level; question 9
        contrast_text.py   # its words: the readings first, then a table per contrast and per draw size, each caveat once
        contrast_repeats.py  # across biological repeats: direction, per-repeat d, paired t raw and normalised, sign-flip
        contrast_across.py # the contrast table's secondary rows: each normalised to its own control, controls against the master raw
        across_text.py     # its words: "Compared across experiments", the two tables
        repeats_text.py    # its words: the across-repeat table and its notes
        contrast_cells.py  # the Plot&Stats cell-level test, raw and normalised, its balance check and balanced version
        _balanced.py       # private: the balanced test's arithmetic, every labelling at once by matrix products, from the same random keys
        cells_text.py      # its words: the balance line, the tables, the false-positive chance, the notes
        reading_text.py    # one verdict per condition, read from every test (confirmed, same direction, mixed, one repeat)
        reagents_text.py   # whether independent reagents for one target agree (with a `target` column)
        repeat_plan.py     # planning the next experiment at the design the user ran: fields a side per repeat
        repeat_plan_text.py  # its words
        contrast_sampling.py  # question 2 with a control: do the fields acquired resolve the difference per repeat?
        legend_text.py     # what to write in a figure legend for each plotted feature (n, unit, test, correction family, change, d, adjusted p)
        _experiments.py    # private: sub-experiments (conditions linked by shared repeats) and Benjamini-Hochberg within each
        _reach.py          # private: run a test only when the design lets its smallest p reach 0.05 after Bonferroni
        ranking.py         # which features separate the conditions: BH-gated |d| ranking over whole technical repeats
        discovery.py       # the same question by the cell-level gate (BH), the paired t on repeat means beside it
        _discovery_tests.py  # private: its tests per feature and condition, the features in worker processes (n_jobs)
        distribution.py    # beyond the mean: spread and 10/50/90% quantiles per repeat, tested across repeats
        whole_table.py     # the whole-table test: a table's centre shifted the same way in every repeat (leave-one-repeat-out projections, paired t corrected), a field-level permutation beside it
        populations.py     # question 10: groups of every cell pooled (HDBSCAN on a UMAP, kept when their neighbours on all features are inside)
        _population_rule.py  # private: the simplest feature rule that picks out a group (a depth-2 balanced tree, fitted on all cells), its procedure checked on held-out fields
        _units.py          # private: sides, units and the randomization test shared by contrast and ranking
        _groups.py         # private: the analysis groups and each comparison's two sides (one control, or (condition, control) pairs)
        unusual.py         # unusual (out-of-distribution) objects against a declared reference (unusual_objects)
        clustering.py      # state clustering by FOV: are rare or unusual objects spread over the fields or clustered
        _outliers.py       # private: the distances and leave-one-group-out thresholds behind unusual_objects
        _common.py         # private: validation and group helpers shared by the sampling-unit checks
        _unit_text.py      # private: the Foundation's checks in words, short and full (variability, mean-precision rule, JCS spread)
        _state_text.py     # private: unusual objects and state clustering in words, short and full
    workflows/             # a new dataset in a few lines: report, optimise, benchmark, live
        __init__.py        # the subpackage's public names
        setup.py           # Setup: every scientific choice of the workflows, declared once
        _setup_parts.py    # private: the positive control, the default definition of new and the presets' checks, re-exported by setup
        _feature_space.py  # private: the Setup's other feature tables: the weighted combined space, the centred view, the tables attached, the space of unusual observations
    _fingerprint_space.py  # private: a wide table's fingerprint on its first k principal components (k at most a fifth of the smallest group), or not assessable
    _whole_table.py    # private: the whole-table test in the report: the comparisons, the words in question 9, the page's block and CSV
    _embedding_report.py  # private: the report of an embedding-only Setup (no readout): the tables alone, what needs explainable features said once
    _results.py        # private: Report and Plan, the results of report and optimise, with their summaries (re-exported by retrospective)
        _contrasts.py      # private: one control or several: the contrast table, own controls, the pooled and figure controls
        _acquisition_gap.py  # private: several controls apart in intensity, said where the data and the comparisons are
        scan.py            # scan_design, DesignScan: the design and how it was loaded, before any analysis
        _scan_rules.py     # private: the scan's findings, the columns that would say more, the questions
        _scan_text.py      # private: the scan in sentences (DesignScan.summary)
        html_report.py     # html_report, zip_results: the report as one HTML page carrying every file; the folder zipped
        _html_render.py    # private: the report's text as HTML, glossary term links, status chips
        _html_sections.py  # private: the page's order, titles, plain answers, heatmap, no numbers
        _html_figures.py   # private: the feature figures as PNG/PDF, palette
        _html_maps.py      # private: the PCA/UMAP data and markup, a map per feature table and the combined one, with the neighbours line under each
        _html_map_text.py  # private: the words around the maps: the comparison of the tables, the neighbours and centred notes, each table's share of an axis
        _html_plotly.py    # private: plotly.js (assets/) and the feature figures' data for it
        _html_plotly_script.py  # private: the script drawing the Plotly figures
        _html_style.py     # private: the page's style sheet (light/dark, drawer, lightbox)
        _html_script.py    # private: the page's script (sort, tabs, downloads, lightbox, drawer)
        _html_tabs.py      # private: tabs (a list to search above 12), the tests behind the reading by kind
        _html_enough.py    # private: "Did I image enough?" on the page: the summary, then one feature at a time
        _html_cards.py     # private: the short cards: "?" help, the data's check rows, fields to check, significant features per condition
        _html_next.py      # private: "What to do next" after the answers that call for a look at the images, with its code
        _html_plan.py      # private: "Plan the next experiment": what to optimise or watch for live, with the code filled in
        _html_data.py      # private: each figure's and table's data as CSV, and the download links
        _html_tests.py     # private: "The tests behind the reading": the tests explained, then one table per test (p, adjusted p)
        _families.py       # private: one correction family: the readout's tests take the feature discovery's p and q
        guided.py          # guide, Answers: load, scan, ask, save the answers (continue or start over), report
        _guided_ask.py     # private: the guided session's questions and how each answer is read
        _guided_controls.py  # private: its control gate: one control, or several with their conditions, parents and master
        _guided_tables.py  # private: its other feature tables: asked for at load with a path and a name, checked, saved with their checksums
        __main__.py        # python -m morphocoverage.workflows: the guided session from a terminal
        _declaration.py    # private: the Setup's declaration in sentences, every preset printed
        retrospective.py   # report, optimise, document: the analyses in question order
        benchmarks.py      # benchmark, Family, default_families: the rule families at matched risk
        _document.py       # private: the assembled report of the workflows, its parts and settings
        _enough_features.py  # private: "Did I image enough?" for every feature alike (light contrasts, precision)
        _enough_targets.py   # private: the rare and fingerprint targets answered in "Did I image enough?"
        _distribution.py   # private: "Beyond the mean" in "Do the conditions differ?" (with a fingerprint)
        _rare_share.py     # private: the labelled rare state's share, compared like a feature
        _label_control.py  # private: the rare label as a positive control of the features (question 1)
        _populations.py    # private: question 10, the populations and each condition's enrichment across repeats
        _population_spaces.py # private: question 10's groups found in each feature space and merged by Jaccard index
        _population_tests.py  # private: question 10's enrichment tests across repeats
        _population_rule_words.py  # private: a group's feature rule in words (fitted on all cells; its procedure checked on held-out fields)
        _result_influence.py  # private: the fields whose removal moves the result (the repeat means the paired test compares)
        figures.py         # plots: the readout and the best-ranked features, from a Setup
        live.py            # live_session; LiveFolder (files) and LiveTables (in memory)
        _manifest.py       # private: the live folder's files (write_fov, the manifest reader), re-exported by live
        _live_order.py     # private: the live session that runs the order check of each group's fields so far
        _live_text.py      # private: the text of one live update (LiveUpdate.summary)
        _shared.py         # private: input checks, orders, rule replays on raw values, outcomes on normalised ones
    strategies/
        __init__.py        # the exported strategies, estimators and policies
        base.py            # Estimator interface; SamplingStrategy = estimators + policies
        baselines.py       # fixed-FOV, fixed-area, generic-observation and object-count baselines
        scalar_precision.py  # JCB/Zweifach-style scalar precision baseline
        _scalar_precision_state.py  # private: its incremental path, per-FOV sufficient statistics for live sessions
        _averaging_cv.py   # private: its averaging-CV arithmetic (object and FOV units)
        stabilization.py   # Marcotti et al.-style descriptor-stabilization baselines
        _descriptor_resamples.py  # private: the resample arithmetic of the descriptor spread
        jcb_planning.py    # JCB multi-sample planning from disjoint pilot samples
        fov_bootstrap.py   # FOV-bootstrap uncertainty of multivariate population summaries
        _fov_bootstrap_checks.py  # private: the checks of its settings and data
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
    test_morphobricks_tables.py  # a fiberfeatures table as written: read as it is, geometry not offered, typos refused
    test_parallel.py       # n_jobs > 1 gives the results of n_jobs = 1
    test_docs.py           # the documentation's examples run and print what they say
    test_notebooks.py      # example notebooks compile, import existing names, carry no outputs
    conftest.py            # one BLAS thread per test process

data/
    crick_idr/             # Crick/IDR feature tables (idr0028 development, idr0139 held out), their analyses and results
    lab/                   # the lab's datasets as MorphoBricks tables: card.toml, DATASET.md, convert.py, NOTICE per
                           # dataset; run_lab.py runs the report, FINDINGS.md logs each run (rules in data/lab/README.md:
                           # only pinned public or owner-released sources, never a manuscript)
    morphobricks/          # small tables as the other MorphoBricks tools write them (fiberfeatures_patches.csv), with provenance
    simulations/           # simulations behind the report's tests and labels (test_choice.py, unbalanced.py,
                           # unusual_calibration.py) and results
    synthetic/             # a constructed end-to-end dataset with known answers (make_synthetic.py, expected.json)

docs/                      # README.md: start here, the three workflows and a reading order
    report/                # REPORT: running_a_report.md, questions.md (R1-R8 in page order), did_i_image_enough.md,
                           # conditions.md (question 9's tests, several controls), figures.md, feature_tables_and_maps.md,
                           # integrity.md, objects_or_fields.md
    optimise/              # OPTIMISE: optimise.md (P1-P10, held-back outcomes), benchmarking.md, sampling_objectives.md
    live/                  # LIVE: live_session.md (O1-O9), stopping_rules.md, online_acquisition.md
    common/                # setup.md, data_format.md, glossary.md (generated from glossary.py), project_aim.md,
                           # scientific_contract.md, architecture.md, morphobricks_fov_contract.md, open_work.md
    audit/, proposals/, preliminary_findings.md  # records, kept as written

`tests/test_contracts.py` checks that every module under `src/morphocoverage/` is listed above, as it checks section 9's scenarios. A new strategy family goes in its own module under `strategies/` (section 20). Future modules should be added only when the corresponding behavior is implemented and tested. Do not pre-create a `novelty`, `coverage`, or estimator architecture that implicitly selects a scientific solution before comparative validation. Data-preparation scripts live next to their data, not in the package.

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
- Future multi-group biological-fidelity outcomes require a **separate contract registry and contract-test module** rather than being silently added to `METRICS_UNDER_CONTRACT`. Those contracts must verify determinism, invariance to strategy identity at matched stop prefixes, preservation of the declared biological-repeat unit, no mutation of inputs, and strict retrospective-only use. Until that registry exists, multi-group outcomes remain proposal/benchmark design work rather than packaged metrics. The one exception is `retrospective.effect_preservation` (owner, 2026-10-05): a standalone retrospective analysis of one feature's per-repeat difference from the control at k fields a side, judged on held-back fields, not an outcome metric and not used by the optimise workflow or the report.
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
python -m ruff check src tests data
python -m ruff format --check src tests data
```

The whole suite runs locally before every push. On GitHub a pull request runs only lint, formatting, the install, the import and the guard tests, and the whole suite runs on demand from the Actions tab (`.github/workflows/tests.yml`, owner 2026-10-05).

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

The sampling objective is separate from the mathematical strategy. Its components are the targets the microscopist declares: the population summary, the phenotypic fingerprint (the `distribution` component) and the rare phenotype with an explicit user target. The objectives are `population`, `distribution`, `rare_discovery`, `phenotype` (summary and fingerprint), `balanced` (summary and rare) and `all`. A rule with no evidence about a required component is `unsupported` for the objective and never `stop`; its own verdict is `StrategyResult.rule_decision`, which self-assessments, replays and calibrations read (`docs/optimise/sampling_objectives.md`).

MorphoBricks metadata handling is centralized. When available, condition, biological_repeat and technical_repeat are kept as independent analysis groups by default and always remain part of FOV identity, even when pooling; `position_id` is preferred as the spatial FOV identifier and `image_file` is only a fallback when one file equals one position; shared metadata columns such as mask_file and patch_file are preserved as metadata rather than phenotype features. Intentional pooling requires an explicit empty group list or another explicit grouping choice.

Strategies must not reimplement condition/repeat splitting. Each core prospective strategy receives one resolved analysis group at a time, and every result records its group identity, parameters, objective, FOV count, observation count/unit, and measured runtime. Methods that require several groups belong in an explicit planning/comparator layer (as with JCB multi-sample planning), not inside a normal group-local estimator. Retrospective benchmark outcomes may compare groups when the purpose is to test whether a stopping rule preserves the biological analysis. This scope expansion is deliberate: it changes the repository contract so that biological-condition comparisons are permitted **only** as retrospective benchmark outcomes or explicit planning comparators, never as hidden inputs to ordinary prospective estimators.

Scientific and computational benchmarking are separate dimensions. Runtime is first-class because smart microscopy requires a decision before the next acquisition is needed. See docs/optimise/sampling_objectives.md, docs/optimise/benchmarking.md, and docs/common/architecture.md.

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

Inputs: a phenotype-observation feature table from any step-2 tool, plus a FOV table that lists every acquired spatial position, including positions without observations. Current step-1/step-2 outputs do not yet emit such a table; `docs/common/morphobricks_fov_contract.md` lists what they need to change.

Usage contexts: importable API and notebooks. A CLI is deferred (section 3).

## 19. Glossary

| Term | Definition |
| --- | --- |
| Condition | An experimental group, e.g. `treated`, `control` (MorphoBricks shared column). |
| Biological repeat | A biological repeat, identified by date in MorphoBricks (shared column `biological_repeat`, formerly `repeat`); `single` when the experiment has one. |
| Technical repeat | A well, dish or coverslip within a biological repeat (shared optional column `technical_repeat`): objects of one technical repeat share its preparation, so it is the unit resampled or shuffled before fields, and the sample level of the derived hierarchy. |
| FOV / spatial position | One independently selected XY acquisition position. A file, z-plane, channel or timepoint is not automatically a separate FOV. |
| FOV key | The columns that jointly identify one FOV: group columns, condition/biological_repeat/technical_repeat, then the FOV column. |
| Observation | One row of phenotype information nested in an FOV; may be a cell, nucleus, patch, slice, FOV summary, embedding, or another declared unit. |
| Feature | An explicitly selected numeric descriptor of an object. |
| Analysis group | FOVs evaluated together; `condition` x `biological_repeat` x `technical_repeat` (those present) by default, pooled only on explicit request. |
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
| Cell-level test | The lab's Plot&Stats test: Cohen's d and a randomization test over the observations of all repeats pooled. It asks whether these wells differ; its chance of p <= 0.05 without an effect is printed beside it (section 4). |
| Paired test across repeats | The paired t-test on the repeat means (SuperPlots), raw and on log ratios to the control: the biological repeat is the replicate; the only test that confirms a difference in question 9. |
| Normalised | Divided by the mean of the control of the same biological repeat (a fold change); removes a scale both wells of a repeat share, not the part of an effect that varies between repeats. |
| Balanced | The cell-level test against the control of the condition's own repeats, with the same number of observations drawn from every repeat: d the median over 100 draws, p one permutation test shuffling labels within each repeat (owner, 2026-10-05); given when the repeats are unbalanced. |

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
