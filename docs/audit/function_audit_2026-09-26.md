# Function audit: every function, one at a time, then the data flow

**Date:** 2026-09-26. **Scope:** `CellMigrationLab/MorphoCoverage` at commit
`0728583` (after the data-format and validator work was merged). **Method:**
each module was read function by function, looking for places where the code,
its docstring and the sentence it prints disagree, or where an input a user can
reasonably give yields a wrong number, a crash or a silent omission. Eight
passes covered the package (A to H below), and a ninth traced the data from
the tables through `Setup.prepared`, the workflows and the report (DF). Every
finding below was **reproduced by running it** before it was fixed. Each fix
is pinned by a regression test that fails on the old code.

Severity: **high**, a wrong answer a microscopist would act on; **medium**, a
wrong number or crash on reasonable input, or a sentence that misstates what
was computed; **low**, wording, docstrings, or an edge case that raises
clearly.

## Summary

| Group | Area | Findings | High |
| --- | --- | --- | --- |
| A | schema, metadata, statistics, spatial | 6 | 0 |
| B | decisions, online, benchmark, evidence | 7 | 0 |
| C | scalar, bootstrap and area strategies | 9 | 1 |
| D | novelty, rare, transforms, outcomes | 9 | 0 |
| E | sufficiency, replay, calibration, adequacy | 10 | 1 |
| F | effort, protocol, inspection, controls, frontier | 16 | 1 |
| G | report and sampling-unit checks | 14 | 0 |
| H | workflows, plots, live | 16 (2 duplicates) | 3 |
| DF | data flow | 4 | 0 |
| U | human errors, from the synthetic sweep | 8 | 0 |

Two findings were outside the plan. One was found while fixing (H, last
row). The other is a size rule: `data_check.py` (A6) and
`retrospective/frontier.py` had outgrown the 600-line rule, and both were split.

The largest problems:

- **C1:** a per-field precision rule stopped after one field.
- **E1:** the sufficiency report printed "yes, met" and "not met at any
  point" for the same rule.
- **F1:** the effort projection understated the field term when fields hold
  unequal numbers of objects.
- **H1:** pooled plots merged same-named wells across repeats, so their
  intervals were too narrow.
- **H2, H3:** one rule was judged on three different feature sets depending
  on the workflow.

## A. Schema, metadata, statistics, spatial

Tests: `tests/test_morphobricks.py`, `tests/test_spatial.py`.

| ID | Sev. | Where | Finding | Fix |
| --- | --- | --- | --- | --- |
| A1 | medium | `metadata.drop_empty_optional_columns` | An optional column empty in every row but present only in the FOV table raised `KeyError`, in the validator and in `check_tables`. | Dropped from each table only where it exists. |
| A2 | medium | `schema._check_observation_values`, `_validated_fov_table` | Blank text (`""`, `" "`) in an identifier or group column was accepted by the validator as a value, creating extra groups silently; `check_tables` refused it, so the two disagreed. | `metadata.empty_values` (vectorised; blank text in text columns counts as missing) is used by both. |
| A3 | medium | `spatial.fov_means`, `fov_semivariogram` | Public helpers paired fields across analysis groups. | Raise "several analysis groups" unless given one group. |
| A4 | low | `schema.unit_plural` | "patchs". | "patches". |
| A5 | low | `statistics` bootstrap docstrings | Did not say that resamples without an object are left out, which makes the interval conditional. | Docstrings say so. |
| A6 | low | `data_check.py` | 606 lines, over the 600-line rule. | Rules moved to `_data_rules.py`. |

## B. Decisions, online, benchmark, evidence

Tests: `tests/test_online.py`, `tests/test_benchmark.py`, `tests/test_objectives.py`,
`tests/test_architecture_regressions.py`.

| ID | Sev. | Where | Finding | Fix |
| --- | --- | --- | --- | --- |
| B1 | medium | `online._check_contract_drift` | An empty `position_id` (CellPatcher writes it empty) was taken as a higher-priority identifier, so a live session stopped after the first field. | Only columns that hold values count. |
| B2 | medium | `benchmark._area` | Missing field areas were summed as 0. | Raises, naming the fields. |
| B3 | medium | `RareTarget` | `detection_probability=None` was accepted. | Raises `TypeError`. |
| B4 | medium | `evidence.RESERVED_QUANTITY_NAMES` | `n_observations` was not reserved, so an estimator could report framework effort as its own quantity. | Reserved. |
| B5 | low | `EffortPolicy` docstring | Named the wrong effort. | Corrected. |
| B6 | low | `redecide` docstring | Said timings were kept. | Corrected. |
| B7 | low | `calibrate_next_fov_novelty` docstring | Did not count draws whose actual value is undefined as unavailable. | Corrected. |

## C. Strategies

Tests: `tests/test_audit_strategies.py`.

| ID | Sev. | Where | Finding | Fix |
| --- | --- | --- | --- | --- |
| C1 | **high** | `ScalarPrecisionStrategy`, FOV unit | One field with objects and one or more empty fields gave CV 0, so the rule said stop. | Fewer than 2 fields with objects is insufficient data. |
| C2 | medium | `plan_objects_to_average` | A pilot of samples with one object each gave CV 0 and a requirement of 0. | Refused: planning needs a sample with 2 or more objects. |
| C3 | medium | `ScalarPrecisionStrategy` pilot | A missing value in `order_col` sorted differently in the incremental and full paths. | Refused. |
| C4 | medium | `ImagedAreaEstimator` | Infinite areas were summed. | Missing, infinite, zero or negative areas are not applicable. |
| C5 | medium | `_precision_from` | A non-negative feature that was all zeros so far was "not applicable" instead of "not enough data"; negatives were caught only later. | Negatives first (not applicable), then zero mean so far (insufficient). |
| C6 | low | `FOVBootstrapEstimator` | A PCA summary that can never be defined (one feature) said continue until `min_fovs`, then undefined. | Not applicable from the first field. |
| C7–C9 | low | docstrings | The pilot was said to be "held fixed" (it is re-selected per prefix); the quantity names were wrong; "objects()" should read "cells". | Corrected. |

## D. Novelty, rare states, transforms, outcomes

Tests: `tests/test_audit_strategies.py`.

| ID | Sev. | Where | Finding | Fix |
| --- | --- | --- | --- | --- |
| D1 | medium | `Standardize.fit_matrix`, `multivariate_outcomes._samples` | A constant feature whose SD was rounding noise (about 1e-17) was accepted, so distances measured noise. | `transforms.constant_columns`, with a relative tolerance, is used by both. |
| D2 | medium | `PCA.fit_matrix` | Asking for more components than the data's rank returned arbitrary components. | Refused: "span fewer than n dimensions". |
| D3 | medium | `multivariate_outcomes` docstring | Wrong bias figures for the energy distance and MMD. | Replaced with the computed two-dimensional values and their formulas. |
| D4 | medium | `rare_incidence` docstring, `docs/architecture.md` | "Stops one field before the plan" holds only for one occurrence. | Qualified, with an example. |
| D5 | low | `first_detection_fovs` | Its closed form rounded differently from `fovs_to_detect(p, beta, 1)`. | It now returns `fovs_to_detect`. |
| D6–D9 | low | docstrings | `novel_fovs_against` (fold fits not checked), the support-novelty cost note, "both quantities" (there are more), the direction in the `IncidenceCoverageStrategy.policies` docstring. | Corrected. |

## E. Sufficiency, replay, calibration, adequacy

Tests: `tests/test_audit_sufficiency.py`.

| ID | Sev. | Where | Finding | Fix |
| --- | --- | --- | --- | --- |
| E1 | **high** | `sufficiency_tables.criteria_tables` | The verdict used the rule's own decision, but the stop points came from the objective's combined decision. Under a phenotype objective a rule could read "yes, met … not met at any point". | `run_benchmark(stop_on="rule")`; the criteria tables use it. |
| E2 | medium | `stopping_calibration`, `adequacy_points` | Groups shorter than `min_fovs` never decided, so they vanished from both tables. | Counted as never stopped; the summary names the reason. |
| E3 | medium | `sufficiency._seen_in_few` | Several rare states were pooled into one incidence, understating the fields one state needs (two states in 2 of 41 fields each: about 30 fields against about 60). | The pooled line is labelled, and the need of a state seen in the fewest recurrent fields is added when several could exist. |
| E4 | medium | `adequacy._tail_point` | `np.quantile` with an unreached run returned NaN, and an interpolated tail rounded below what 90% of runs reached. | Observed counts (`inverted_cdf`). |
| E5 | medium | `adequacy._median_point` | With exactly half the runs unreached it said "in most runs". | "in half or more of runs". |
| E6 | medium | `sufficiency_text` | "about 3 / k" was printed at any confidence. | `-ln(1-confidence)` / k. |
| E7 | medium | `sufficiency_text` | "A state seen in 2 fields is not new" holds only when `min_other_fovs` is 1. | The span follows the settings. |
| E8 | low | `_next_fov_tables` | Fields without observations were reported as 0 when novelty was not estimable. | Counted from the positions. |
| E9 | low | calibration and sufficiency | The median stop printed with `.0f` (2.5 became "2"), and stop quantiles were interpolated. | One decimal when needed; observed quantiles. |
| E10 | low | `replay._check`, `adequacy._checked` | Type hint and missing docstrings. | Added. |

## F. Planning, inspection, controls, frontier

Tests: `tests/test_audit_planning.py`, `tests/test_retrospective_effort.py`.

| ID | Sev. | Where | Finding | Fix |
| --- | --- | --- | --- | --- |
| F1 | **high** | `effort._se`, `_contributions` | The field term of the projected SE was `v_fov / fields`. With unequal fields the truth is `v_fov · Σn²/N²` (the design effect's cluster size). With the true components of a check (10 fields of 2 cells and 2 of 100 per well), the old term gave an SE of 0.13 against 0.27 analytically and 0.26 simulated. | `v_fov · cluster_size / N`, consistent with `design_effect`. |
| F2 | medium | `effort_projection` | A stratum with one repeat or one sample raised a bare `LinAlgError`, which crashed `optimise` with a derived hierarchy. | One repeat drops the repeat level and says so. A stratum that cannot be estimated goes to `EffortResult.skipped` with the reason, and the others are still estimated. |
| F3 | medium | `effort_summary` | "The intervals overlap" was printed when no alternative had the acquired total. | Each reason is stated. |
| F4 | medium | `control_summary` | Checks that could not be judged were not mentioned, and printed "nan%". | Counted; "not judged". |
| F5 | medium | `inspect_fovs` | One missing value turned novelty off silently: "No field stands out". | The summary says new states were not looked for, and why. |
| F6 | medium | `order_efficiency` | The median fields came from all runs, but the median travel and time only from the runs that reached the target. | Every median over all runs; an unreached run counts as beyond. |
| F7 | medium | `protocol_summary` | With one unit it said "the others reached no number". | "With a single plate, no number can be chosen on other plates …". |
| F8 | medium | `frontier` | Families from a replay with other seeds or splits were scored against these reservoirs whenever the stream sizes matched. | Each family records how its reservoirs were drawn; a mismatch is refused. |
| F9 | low | effort module docstring | The formula left out the yield and cluster size. | Corrected. |
| F10 | low | `_check_separable` | Its advice ("declare the hierarchy without 'well'") was impossible to follow. | "It needs more than one field per well". |
| F11 | low | inspection influence | A field with missing values weighed less than one. | Each field weighs one over its non-missing values. |
| F12 | low | protocol median | `int()` truncated 7.5 to 7. | 7.5. |
| F13 | low | `Summary.label` | "1th", "22th". | Proper ordinals. |
| F14 | low | `frontier_summary` | "Intervals resample the 1 units". | "No interval: there is a single unit". |
| F15 | low | `frontier_summary` | `:g` failed on a non-numeric setting. | The setting is printed as given. |
| F16 | low | `protocol_summary` | "technical_repeats", "samples". | "technical repeats", named by the hierarchy. |

`retrospective/frontier.py` was 603 lines: its sentences moved to `frontier_text.py`.

## G. Report and sampling-unit checks

Tests: `tests/test_audit_report.py`.

| ID | Sev. | Where | Finding | Fix |
| --- | --- | --- | --- | --- |
| G1 | medium | `FOVVariability.summary`, report profile | Crashed when no object varies within a field (the share is undefined). | Said in words; the profile leaves the clause out. |
| G2 | medium | `fov_variability` z-scores | The field z-score used `s/√n_j`, but the field is part of the mean, so its variance is `s²(1/n_j − 1/N)`. With 3 random fields about 1.6% were called unusual, not the 5% the text says. | `√(1/n_j − 1/N)`: 4.6%, 4.8% and 5.1% with 3, 5 and 12 fields. |
| G3 | medium | `ConditionContrast.summary` | The Bonferroni count in the text was the number of rows, not the number of tested contrasts. | Tested count. |
| G4 | medium | `ConditionContrast.summary` | The JCS line was dropped without a repeat column (None never equals None). | Matched with `isna`. |
| G5 | medium | `condition_contrast` | An undefined field-level d got the smallest possible p-value, with status "ok". | No test; the status says why. |
| G6 | medium | `feature_ranking` | A perfect separation (no variance) had an undefined d, and was reported as "fewer than 2 fields"; the docstring promised it would rank first. | `separation_d` returns ±∞, capped when ranked; equal means say "do not vary". |
| G7 | medium | ranking summary | "Cannot pass the gate" was printed even when the p-value floor was below alpha. | Only when the floor exceeds alpha. |
| G8 | medium | `FOVClustering.summary` | A group with 0 labelled objects was "not seen". | "no object … is labelled, so it was not looked for". |
| G9 | medium | report | Redundancy without an order replay marked question 8 as assessed. | Question 8 needs `order_efficiency`; the sampling-frame caveat moves to the profile. |
| G10 | medium | report docstring and needs | They said adequacy answers question 2; it is under question 5. | Corrected. |
| G11 | low | descriptor spread summary | "nanx". | "no ratio (random samples never varied)". |
| G12 | low | `descriptor_spread_check` | Groups smaller than the threshold were skipped silently. | Named in the summary. |
| G13 | low | contrast module docstring | "Each number twice" was not true of every number. | Reworded. |
| G14 | low | `_NEEDS["profile"]` | "Run any of the analyses below". | Names the analyses. |

## H. Workflows, plots, live

Tests: `tests/test_audit_workflows.py`.

| ID | Sev. | Where | Finding | Fix |
| --- | --- | --- | --- | --- |
| H1 | **high** | `plots_of_data`, pooled repeats | With more repeats than are drawn side by side, wells with the same name in different repeats were merged into one bootstrap unit. Over 9 repeats there were 2 units instead of 18, and the interval was about 35% too narrow. | Units are keyed by (repeat, technical repeat); colours are ranked within the repeat, as the legend says. |
| H2 | **high** | workflows | One rule saw three feature sets: the readout plus features (report, sufficiency), the readout plus criterion columns (optimise and benchmark replays), `Setup.columns` (live). | Every rule reads `Setup.columns`. |
| H3 | **high** | `default_families` | "FOV bootstrap of the mean" read every judged column, so declaring a fingerprint changed the locked family; the docstring says the readout. | New optional `features` on `FOVBootstrapEstimator`/`Strategy` (ids unchanged when unset); the family reads the readout. |
| H4 | medium | `report` inspection | With novelty, the influence, empty-field and stage-neighbour reasons were dropped. | One call: novelty on the features (`inspect_fovs(novelty_cols=...)`), the other reasons on the readout. |
| H5 | medium | plot legend and palette | The note "not coloured" was wrong (the first ranks were coloured); `max_technical` above 8 was capped silently; the palette's grey looked like the overflow grey. | The note says what is coloured; above 8 raises; the grey was replaced. |
| H6 | medium | `LiveFolder.poll` | The default reader parsed "001" as 1, so the manifest conflicted and the session stuck. | Metadata columns are read as text, as the manifest is. |
| H7 | – | | Duplicate of A1. | |
| H8 | – | | Duplicate of DF1. | |
| H9 | medium | `benchmark` | Frontiers were keyed by label: two criteria printing as "5.0%" lost one. | The second is named by its criterion. |
| H10 | low | `workflows.plots` | The top-k count included the readout, which is always plotted. | The readout never takes a top place. |
| H11 | low | `write_fov` | An existing empty manifest got no header. | Header written; an incomplete one is refused. |
| H12 | low | `Setup.declaration` | Without features it claimed the readout is used for unusual objects and novelty; neither runs. | Says they are not assessed. |
| H13 | low | `Setup.declaration` | "Of the same repeat" was printed without a repeat column. | Says the whole table then. |
| H14 | low | `LiveTables.update` | `iterrows` turned integer metadata into floats. | Records keep each column's type. |
| H15 | low | docstrings | R1–R6 (there is R7), "coloured by repeat", the resampling wording, the `LiveTables` key, "for each tolerance". | Corrected. |
| H16 | low | `LiveUpdate.summary` | Always said "objects". | The session's unit. |
| – | medium | `ConditionContrast.summary` | Found while fixing: when no object count fits the data, the effects table has no columns and the summary raised `KeyError`. | Skipped when empty. |

## DF. Data flow

The trace followed one dataset from its two tables to the report. These
steps were checked and found correct:

- normalisation keeps its rows aligned;
- no report or plan result keys collide;
- a duplicated observation index is handled;
- the live path runs on raw values and is scale-invariant;
- a re-sorted live table is detected;
- `n_jobs=1` and `n_jobs=-1` give identical records.

Tests: `tests/test_audit_workflows.py`.

| ID | Sev. | Where | Finding | Fix |
| --- | --- | --- | --- | --- |
| DF1 | medium | `Setup.resolved_hierarchy` | `optimise` crashed when `group_cols` left out a `technical_repeat` column that exists: the derived hierarchy did not match the reliability groups. | No hierarchy is derived then; declare one to ask for it. |
| DF2 | low | plots | Normalised values were labelled with the feature name only. | The axis label adds "relative to control". |
| DF3 | medium | question 7 | Control normalisation per repeat removes between-repeat variance (idr0028 ARAP2: 50% becomes 0%), and the answer did not say the values were normalised. | The effort summary says so and what it removes. |
| DF4 | low | `workflows.plots` docstring | "Interval from resampling whole fields". | Technical repeats, then their whole fields. |

## End to end, and human error

After the fixes, `data/synthetic` builds an acquisition whose answers are
known. `tests/test_end_to_end.py` checks every workflow against that truth.
`tests/test_human_errors.py` makes the mistakes people make with such tables.
That sweep found these gaps, now closed:

| ID | Sev. | Mistake | Before | Now |
| --- | --- | --- | --- | --- |
| U1 | medium | `technical repeat`, `condition ` or a byte-order mark before the first header | Not recognised as a near miss. The column was ignored: without a declared control, conditions were silently merged. | Refused by the validator and `check_tables` (`metadata.canonical_spelling`), naming the column meant. |
| U2 | medium | A cell table appended twice | Ran, every count doubled. | Warned: exact copies of rows, or a `label_id` twice in one field. |
| U3 | medium | `check_tables(observation_unit=5)` or any other refusal it did not know | Passed on with a text about z-planes. | Its own message, with the matching place and fix. |
| U4 | low | `group_cols="condition"` in `check_tables` | Nine errors, one per letter. | One error: "a single string". |
| U5 | low | `' drug'` / `'Drug'` against `'drug'`, or `5` against `'05'` | "Not in the FOV table" with no reason. | The error names the column and says spaces or case, or number against text. |
| U6 | low | `'area '` for `'area'` | "Not in the observation table". | "The table has 'area ' for 'area'". |
| U7 | low | A misspelt control in the workflows | "No analysis group matches control". | Lists the values present. |
| U8 | low | No FOV table, or an empty observation table | `check_tables` passed silently, then the workflows raised a bare `TypeError`. | A warning, and a workflow error that says what to pass. |

## What changed for users

These are behaviour changes, not only wording:

- `effort_projection` no longer raises for a stratum it cannot estimate. It
  lists that stratum in `EffortResult.skipped`, with the reason, and the
  summary prints it.
- `run_benchmark` takes `stop_on="decision"` (default) or `"rule"`.
- `inspect_fovs` takes `novelty_cols`.
- `plots_of_data` takes `value_note`, and refuses `max_technical > 8`.
- `FOVBootstrapEstimator` and `FOVBootstrapStrategy` take an optional
  `features`. The configuration ids of existing rules are unchanged.
- `frontier` refuses families whose replay drew other reservoirs or orders.
- The `report`, `optimise` and `benchmark` workflows give the rules
  `Setup.columns`.

The committed idr0028 and idr0139 results were regenerated with the scripts
of `data/crick_idr/run_all.py`. The tracked tables predated the earlier
`well` to `technical_repeat` rename, which changes each group's seed, so
every table changed, not only those these fixes touch. `benchmark.py` was not
rerun. Its locked protocol digest (`benchmark_protocol.lock`, version 2)
already differed from `benchmark_protocol.py` at `0728583`, so the script
refuses to run. Locking a new version is a protocol decision left to the
maintainers, and the committed benchmark tables are unchanged.
