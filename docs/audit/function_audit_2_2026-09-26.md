# Second function audit: functions, data flow, fallbacks, duplication, placement, math

**Date:** 2026-09-26. **Scope:** `CellMigrationLab/MorphoCoverage` at commit
`07c20f5` (after the first audit, `function_audit_2026-09-26.md`, and the
regenerated results were merged). **Method:** ten passes. Nine read the
package module by module, function by function (A to I); for each function
they checked the code against its docstring, against the sentence it prints,
and against the mathematics it claims. A tenth pass (J) read the package as a
whole, looking for:

- silent fallbacks;
- duplicated functions;
- import cycles;
- functions that live in the wrong module.

Every finding was **reproduced by a script** before it was fixed. Where a fix
needed a scientific or API decision, the owner chose (the "Decision" notes
below). Each fix is pinned by a regression test that fails on the old code:
`tests/test_audit2.py` (shared foundations) and
`tests/test_audit2_<group>.py`.

Severity: **high**, a wrong answer a microscopist would act on; **medium**, a
wrong number or a crash on reasonable input, a silent fallback, or a sentence
that misstates what was computed; **low**, wording, docstrings, duplication,
or an edge case.

## Summary

| Group | Area | Findings | High |
| --- | --- | --- | --- |
| A | schema, metadata, data_check, objectives, provenance | 21 | 0 |
| B | online, sequential, benchmark, calibration, reservoir, bounds | 18 | 1 |
| C | scalar precision, JCB planning, stabilization, fixed area | 17 | 1 |
| D | distances, incidence coverage, bootstrap, transforms | 7 | 0 |
| E | normalize, outcomes, sufficiency, spatial, statistics | 13 | 1 |
| F | retrospective criteria, reliability, protocol, adequacy, stage | 16 | 1 |
| G | retrospective effort, report, controls, inspection, frontier | 19 | 2 |
| H | sampling-unit checks, plots | 19 | 4 |
| I | workflows and data flow | 12 | 1 |
| J | structure: duplication, cycles, placement, fallbacks | 21 | 0 |

J repeats several findings of the other groups from the structural side;
they are listed once, where they were fixed, and cross-referenced.

The high-severity findings:

- **B-1:** a live session counted 12 z-plane files as 12 fields and stopped.
  The plane-as-field rule never ran live, because the session validated one
  field at a time.
- **C-1:** the per-field precision rule counted empty fields in `k/(k-1)`
  and in the degrees of freedom. The nominal 95% upper bound covered 49% of
  the time (37% with 90% of fields empty).
- **E-1:** normalising to a negative control mean flipped the sign of every
  effect (d +2.42 became −2.42).
- **F-1:** an empty subset was scored "not preserved" even when the reference
  itself left the metric undefined. Reliability read 0% where it could not be
  judged.
- **G-1:** the effort projection assumed equal sample and repeat sizes; with
  unequal ones the SE was 0.42 against 0.69 analytically and 0.69 simulated.
- **G-13:** the report's "expected by chance" used the false-alarm rate after
  `fovs_seen` fields. The inspection judges each field against all the others
  (leave one field out). It said "24 of 60 expected at 41%"; the right rate
  is 3%.
- **H-1, H-2:** the permutation p-value never counted infinite (perfectly
  separating) shuffles, and dropped float ties. A 2-versus-2 contrast got
  p = 1/3 in truth but passed the ranking gate.
- **H-3:** with one technical repeat on a side, fields were shuffled as if
  they were wells: 60% false positives under a well effect.
- **H-4:** a constant feature (0.1 everywhere) passed the "> 0" checks through
  rounding noise: ICC 0.49, Cohen's d 1.79, ranked first with an infinite d.
- **I-1:** with normalisation, `report`, `optimise` and `benchmark` replayed
  the rules on normalised values, while the live session runs on raw ones.
  Rules that are not scale-invariant disagreed on 149 of 180 steps.

## Shared foundations (fixed first)

Several groups found the same statistic, bound or wording written two or
three times, sometimes with different answers. These were moved once, before
the per-group fixes. Tests: `tests/test_audit2.py`.

- `statistics.py` now holds every shared statistic:
  - `nested_variance_components`, `sd_upper_bound_factor`, `fovs_to_detect`,
    `first_detection_fovs`, `srs_relative_rmse`, `cluster_size`,
    `design_effect`, and `jcb_required_objects` (C-8: 0.07/0.01 squares to
    49.000000000000014, which is 49, not 50);
  - the two field-level JCB estimands (C-14 below);
  - the rounding tolerances `RELATIVE_TOLERANCE`, `mean_is_zero` and
    `spread_is_zero`.

  The old import places re-export them.
- `bounds.py` gains `lower_bound` and `interval`, the two-sided
  Clopper-Pearson interval. `sufficiency` and `clustering` use the same
  convention: the low end of the two-sided interval (J-9: they gave 501
  against 342 fields for the same count).
- `extra_trials_for_bound(keep_rate=True)` returns a count after which the
  bound stays met (B-10: the old answer could fail again one trial later).
- `bounds` accepts numpy integers (B-14).
- `statistics.bootstrap_nested` resamples wells with replacement and keeps
  each drawn well's fields as they are (E-5; Ren et al. 2010). The old
  two-stage resample counted the field variance twice: 98% coverage with
  many wells, and 71–78% with two or three. The docstring says two or three
  wells give no reliable interval with any scheme.
- Wording and seeds:
  - `_wording.py` holds the one way to write a percentage, a count, "N
    fields", a duration and a group (J-16). An inline percentage printed 0%
    for 0.3%, "about 0 min" appeared (F-14), and there were four "fields"
    helpers.
  - `_seeds.py` holds the one per-group seed (J-14/B-16).
  - `_checks.py` holds integer and number checks that accept numpy types.
- Import cycles:
  - `sampling_unit` no longer imports `retrospective` (J-11).
  - `validate_input_tables` lives in `data_check`, which attaches its report
    to the errors. `schema` re-exports it lazily, so `schema` never imports
    `data_check` (J-19).
- `object_rules` uses `field_sample`, `stabilization.describe` and
  `jcb_required_objects` instead of its own copies (E-11, H-13, C-16).
- `field_sample` and `fields_holding` refuse to take more objects than the
  fields hold (E-8).

## A. Schema, metadata, data check, objectives, provenance

Tests: `tests/test_audit2_a.py`.

| ID | Sev. | Where | Finding | Fix |
| --- | --- | --- | --- | --- |
| A-1 | medium | `metadata.empty_values` | An object column holding dates, integers or booleans crashed the validator and `check_tables` (`.str` on non-text). | Such columns are tested entry by entry. |
| A-2 | medium | categorical group columns | A blank category (`' '`) was accepted as a group. | It counts as a missing group value, in both rule sets. |
| A-3 | medium | `group_cols`, `feature_cols` | A generator was consumed by the first read and the groups were pooled silently; a `pd.Index` or an array crashed. | `metadata.column_names` reads them once into a tuple; a bare string is refused. |
| A-4 | medium | `check_tables` | Integer column names crashed the near-miss check; `feature_cols=None` raised a bare `TypeError`. | Handled; a clear error in both. |
| A-5 | medium | `check_tables` | A column of lists crashed the duplicate-row check, which the validator accepted. | Unhashable cells are compared by their text. |
| A-6 | medium | both rule sets | A column name used twice gave a false "must be numeric" or an `AttributeError`. | Refused before any other rule. |
| A-7 | medium | `label_id` rule | Ignored an explicit `fov_col`, so labels restarting per field were flagged. | Uses the resolved field and any file or z/c/t column. |
| A-8 | low | `check_tables` | A group column absent from the FOV table was reported as "lists a field more than once". | Reported as missing from the FOV table. |
| A-9 | medium | plane rule | Skipped with one row per file, and when z was only in the FOV table. | Decision: keep the guard, warn for one row per field, read z/c/t from the FOV table. |
| A-10 | low | control check | Not checked without a FOV table; `control={}` accepted. | Checked on the observation table; an empty control refused. |
| A-11 | low | messages | `pd.NA`/`NaT` printed `<NA>`/`NaT`. | "empty". |
| A-12 | low | messages | `'1'` against `'1.0'` was called spaces or case. | "writes the same number differently". |
| A-13 | low | messages | 1 in 5000 missing printed "0.0%". | 0.02%, through `_wording.percent`. |
| A-14 | low | messages | A duplicated index printed a Series as the example. | Examples read by position. |
| A-15 | low | `StrategyResult.explain` | "after 1 FOVs, 1 cells". | Singular for one. |
| A-16 | low | `RareTarget`, `EffortPolicy`, fixed baselines | numpy numbers refused. | Accepted and stored as Python numbers; same config ids. |
| A-17 | low | `provenance.config_hash` | `{1: x}` hashed as `{'1': x}`, and a tuple as a list. | Tagged; every plain-JSON id is unchanged (checked over the whole suite). |
| A-18 | low | `combine_decisions([])` | Returned STOP. | Raises. |
| A-19 | low | validator | Reset the observation index when groups came from the FOV table. | Index kept. |
| A-20 | low | validator vs `check_tables` | Rules written twice, diverging in A-4, A-5, A-7, A-8 and A-19. | Decision: keep both, fix each divergence. A 13-case agreement test requires the same verdict from both. |
| A-21 | low | near-miss names | `Conditions`, `PositionID`, `technicalRepeat` were not recognised, and the groups were pooled silently. | Decision: plurals and camelCase are near misses. |

## B. Live session, sequential replay, benchmark, calibration, reservoir

Tests: `tests/test_audit2_b.py`.

| ID | Sev. | Where | Finding | Fix |
| --- | --- | --- | --- | --- |
| B-1 | **high** | `OnlineSession.add_fov` | 12 z-plane files counted as 12 fields, then STOP: the plane rule never ran live. | The session keeps each field's z/c/t values and runs the validator's plane check. |
| B-2 | medium | replay prefixes | Kept table row order within a prefix, while live receives fields in arrival order, so order-sensitive estimators disagreed. | Decision: prefixes in acquisition order (table order within a field). Live and replay now agree on every step. |
| B-3 | medium | acquisition-order column | Text was sorted as text ('10' before '2'). | Numeric, datetime or timedelta required. |
| B-4 | medium | benchmark permutation | Permuted the whole FOV table, so a group's order depended on the other groups and on row order. | Decision: one per-group permutation (`benchmark.permuted_fovs`), seeded from `random_state` (with J-3, I-8). Benchmark protocol version 3. |
| B-5 | medium | `calibrate_next_fov_novelty` | One generator across groups. | One per group and count. |
| B-6 | medium | `add_fov` | A field was half added when an estimator failed. | Atomic: the field is removed and state rebuilt. |
| B-7 | medium | `benchmark._area` | Text areas were concatenated; inf, zero or negative areas summed. | Refused. |
| B-8 | medium | `ReservoirSplit.split`, `reference_error_floor` | Pooled several groups silently. | Refused. |
| B-9 | low | `BenchmarkRecord` | `stop_on` not recorded; `final_decision` disagreed with `stopped`. | `stop_on` recorded; `final_decision` reads it. |
| B-10 | low | `extra_trials_for_bound` | Not monotone; half-even rounding. | Foundations. |
| B-11 | low | reservoir size | `ceil(0.07 × 100)` gave 8. | Rounded before the ceiling. |
| B-12 | low | calibration | A group too small for every count was dropped. | A row with `draws=0` and a reason. |
| B-13 | low | `spatial_blocks` | An infinite coordinate made a huge integer block. | Refused. |
| B-14 | low | several | numpy integers refused. | Accepted. |
| B-15 | low | `reference_error_floor` | Docstring said "half" with odd blocks. | Says which side is larger. |
| B-16 | low | seeds | `reservoir._group_seed` duplicated `group_identity`. | Foundations. |
| B-17 | low | private helpers | `sequential` helpers imported by replay and benchmark. | Decision: names kept, documented as shared-private. |
| B-18 | low | `FovSubsets.take` | Repeated rows listed a field twice but its observations once. | Refused. |

## C. Scalar precision, JCB planning, stabilization

Tests: `tests/test_audit2_cd.py`.

| ID | Sev. | Where | Finding | Fix |
| --- | --- | --- | --- | --- |
| C-1 | **high** | `fov_averaging_cv` | Empty fields counted in `k/(k-1)` and in the degrees of freedom: the upper-95 bound covered 49% (37%). | Decision: only fields with objects (k′) there; all fields k for effort and the pilot projection. Coverage 0.947. |
| C-2 | medium | `plan_objects_to_average` | A within-sample SD of 0 planned 0 objects and stopped at an empty field. | Decision: refused. |
| C-3 | medium | planned-count policy | Counted rows, missing values included. | Decision: `measured_objects` counts measured values. |
| C-4 | medium | pilot order | A categorical order column sorted two ways (full and incremental). | Refused (the leakage guard strips categories); both paths use one stable sort. |
| C-5 | medium | `DescriptorPlateau` | A missing order value was placed last silently. | Refused, through a shared `check_order_col`. |
| C-6 | medium | design effect | A constant feature gave noise over noise, different incrementally. | None below the rounding tolerance. |
| C-7 | low | pilot docstring | Said the pilot was held fixed when the order column is not the acquisition order. | Decision: docstring and reason corrected. |
| C-8, C-9 | low | JCB count, SD bound | Float ceiling (50 for 49); confidence unvalidated. | Foundations. |
| C-10 | low | `pilot_fovs` | 2.0 accepted, then a pandas crash. | Integer ≥ 2 required. |
| C-11 | low | `FixedAreaStrategy` | `area_col` checked late; text coerced. | Checked at construction. |
| C-12 | low | JCB planning | A missing sample value crashed converting NaN to int. | Clear error. |
| C-13 | low | docstring | Said a zero mean is not applicable; it is insufficient. | Corrected. |
| C-14 | medium | ScalarPrecision vs `object_rule_check` | Two "JCB per field" rules gave 7 and 343 fields. They estimate different means. The ratio CV was 4 times optimistic when field size and phenotype correlate. | Decision: both kept, named apart in `statistics` (`object_weighted_mean_se`, `field_weighted_mean_se`). ScalarPrecision reports the size–phenotype Spearman correlation and warns at p < 0.05. |
| C-15 | low | naming | Two "design effect" estimators, 24.4 and 21.8. | ScalarPrecision's is `ratio_design_effect`. |
| C-16 | low | duplication | Descriptors, JCB count, SD bound outside `statistics`; `required_fovs` rounding. | Foundations; `required_fovs` rounded like the object count. |
| C-17 | low | object-unit spread | Memory grew by 2.4 kB per object. | Chunked; same draws. |

## D. Distances, incidence coverage, bootstrap, transforms

Tests: `tests/test_audit2_cd.py`.

| ID | Sev. | Where | Finding | Fix |
| --- | --- | --- | --- | --- |
| D-1 | medium | `frechet_distance` | Clipped to 0 below an absolute 1e-12, so small units gave 0. | Relative tolerance on the scale of the two distributions. |
| D-2 | medium | incidence coverage at Q2 = 0 | Coverage used the uncorrected factor while Chao2 used the corrected Q0: 0.571 against iNEXT's 0.657, and E[new] = (U/T)(1 − C) broken. | Decision: iNEXT, A = T·Q0/(T·Q0 + Q1). |
| D-3 | low | FOV bootstrap with a transform | The C6 fix of the first audit was incomplete with a transform. | `FeatureTransform.output_width`. |
| D-4 | low | PCA rank test | Refused independent features with a scale ratio above 1e6 as collinear. | Numerical-rank rule. |
| D-5 | low | clustering | Its own closed form of `first_detection_fovs`. | Uses it (group H). |
| D-6 | low | `multivariate_outcomes._samples` | Re-implemented `Standardize`. | Uses it. |
| D-7 | low | placement | `fovs_to_detect` in a strategy module, used by four others. | Foundations. |

## E. Normalisation, outcomes, sufficiency, spatial

Tests: `tests/test_audit2_ef.py`.

| ID | Sev. | Where | Finding | Fix |
| --- | --- | --- | --- | --- |
| E-1 | **high** | `normalize.control_means` | A negative control mean flipped the sign of every effect. | Decision: a control mean ≤ 0 is refused, naming the feature and the repeat. |
| E-2 | medium | `repeat_col` | A misspelt column pooled the repeats silently. | Only the default may be absent. |
| E-3 | medium | zero-mean checks | `== 0` let a rounding-noise mean give errors of 1e16. | `mean_is_zero`. |
| E-4 | medium | "seen in few" | Counted new fields as recurrent, and planning pooled two one-field states. | Few but not new only. |
| E-5 | medium | `bootstrap_nested` | Double-counted the field variance. | Foundations; the plot title says how many wells and warns at 2–3. |
| E-6 | low | `fovs_beyond_enough` | Filled for a "no". | Only for "yes". |
| E-7 | medium | criteria | A duplicate criterion raised `MergeError`. | Refused clearly. |
| E-8 | low | field sampling | Silent with too few objects. | Foundations. |
| E-9 | low | wording | "over 1 orders". | Singular. |
| E-10 | low | semivariogram | Dropped infinite coordinates silently. | Refused. |
| E-11, E-12, E-13 | low | duplicates | `_fov_sample`, inline Clopper-Pearson, `srs_relative_rmse` outside `statistics`. | Foundations. |

## F. Retrospective criteria, reliability, protocol, adequacy, stage

Tests: `tests/test_audit2_ef.py`.

| ID | Sev. | Where | Finding | Fix |
| --- | --- | --- | --- | --- |
| F-1 | **high** | `OutcomeCriterion.assess` | An empty subset was NOT_PRESERVED even when the reference left the metric undefined: 0% instead of "could not be judged". | UNDEFINED unless the metric is defined on the reference. |
| F-2 | medium | reliability | Repeated reservoirs counted as distinct hold-outs. | Decision: "N splits over M distinct hold-outs", and a warning. |
| F-3 | medium | `redundancy_distance` | One small group aborted every group. | Decision: a row with a reason. |
| F-4 | medium | `order_efficiency` | Unreached runs filled with the shortest stream + 1. | Counted beyond their own stream; no median when half are unreached. |
| F-5 | medium | `order_efficiency` | Accepted adequacy from another replay. | Refused on a settings mismatch. |
| F-6 | medium | adequacy text | A criterion never judged read "not within the fields available". | Says it could not be judged. |
| F-7 | medium | protocol interval | Collapsed to 100%–100% at the boundary; its own bootstrap. | Decision: exact Clopper-Pearson over units at 0 or 100%; `bootstrap_fields` otherwise. |
| F-8 | medium | protocol text | Said held-back fields "could not judge" samples with a defined low ceiling. | Decision: reworded. |
| F-9 | medium | leave-one-unit-out | A unit with too few fields counted as 0 of 0. | Named, not counted. |
| F-10 | low | stage | Infinite coordinates accepted. | Refused. |
| F-11 | low | `fov_counts` | numpy integers refused. | Accepted. |
| F-12 | low | `technical_levels`, `strata` | A bare string was split into letters. | Refused. |
| F-13 | low | wording | "batchs". | "batches". |
| F-14 | low | wording | "about 0 min"; "for this criterion" for all. | Foundations; "for all the criteria together". |
| F-15 | low | reliability, replay | One single-field group made both raise for every group; "1 hold-outs". | Decision: skipped with a reason; singular. |
| F-16 | low | duplicates | Order phrases three times, median rules, protocol bootstrap. | One order-phrase table; `bootstrap_fields`. |

## G. Effort, report, controls, inspection, frontier

Tests: `tests/test_audit2_g.py`.

| ID | Sev. | Where | Finding | Fix |
| --- | --- | --- | --- | --- |
| G-1 | **high** | `effort._se`, `_contributions` | Sample and repeat terms assumed equal sizes: SE 0.42 against 0.69. | Decision: the acquired SE uses the actual sizes (now 0.690 = analytic 0.690, simulation 0.688). Alternatives are compared with the acquired allocation made balanced, and the imbalance cost is stated separately. |
| G-2 | medium | effort | Units counted with units that hold no observations. | Counted among those with observations (also the few-units gate). |
| G-3 | medium | effort text | "Every repeat holds a single sample" when one repeat held only empty fields. | The actual reason. |
| G-4 | medium | effort bootstrap | Failed resamples dropped silently (152 of 500). | Counted; below 90% success no interval. |
| G-5 | medium | allocations | Defaults below one field per sample; declared ones unvalidated. | At least one; validated. |
| G-6 | medium | effort | An empty or NaN stratum crashed the projection. | Skipped with a reason. |
| G-7 | low | wording | "Most of" at 46%. | Only above half. |
| G-8, G-9, G-10 | medium | inspection | A zero summary skipped silently; skipped summaries keyed by label only; "No field stands out" when nothing was checked. | Reported, keyed by group, "No field was checked". |
| G-11 | low | inspection | `influence_threshold=None` accepted, then `TypeError`. | Validated. |
| G-12 | medium | planted control | Silently absent when no donor field had enough objects. | Said so. |
| G-13 | **high** | report | "Expected by chance" and the adjusted unseen bound used the rate after `fovs_seen` fields, not leave one field out: 41% against 3%. | Decision: both rates are measured leave one field out. |
| G-14 | low | profile | A sampling-frame caveat marked question 3 as assessed. | Not assessed; the caveat goes to the notes. |
| G-15 | medium | frontier | "Never got there" when a setting met the risk but not lastingly. | Says where it met and where it was lost. |
| G-16 | low | frontier | "Saves N fields" when the interval includes 0. | Decision: "(not resolved)". |
| G-17 | low | placement | Variance components, design effect, "fields" helper. | Foundations. |
| G-18 | low | modularity | 200-, 165- and 135-line functions. | Split into `effort_design.py` and `report_profile.py`. |
| G-19 | low | report | Adjusted bound withheld under "positive only". | Decision: given. |

## H. Sampling-unit checks and plots

Tests: `tests/test_audit2_h.py`.

| ID | Sev. | Where | Finding | Fix |
| --- | --- | --- | --- | --- |
| H-1 | **high** | `_units.permutation_p` | Infinite shuffles never counted: perfect separation always got p = 1/(n + 1). | Counted; only NaN guarded. |
| H-2 | **high** | `_units.permutation_p` | Float ties dropped; no exact enumeration; p below the exact floor. | Relative tolerance; enumerated when few assignments exist (`p_exact`). |
| H-3 | **high** | contrast, ranking | One well on a side: fields shuffled as wells, 60% false positives. | Decision: no test, with the reason; the ranking does not gate on it. |
| H-4 | **high** | variability, contrast, ranking | A constant feature scored ICC 0.49, d 1.79, ranked first. | Rounding tolerance; not applicable. |
| H-5 | medium | ranking | The printed floor was 1/comb; with equal sides it is 2/comb. | The attainable floor (the G7 test: 0.014 became 0.029). |
| H-6 | medium | contrast | Used `cohen_d`, so perfect separation read "field means do not vary". | Same separation measure as the ranking. |
| H-7 | medium | `fov_clustering` | Unlabelled fields counted as fields without the state. | Left out and counted apart. |
| H-8, H-9, H-18 | medium | contrast, ranking, plots | A misspelt explicit column was dropped silently (repeats pooled); an absent control column gave a bare `KeyError`. | Raised; only defaults may be absent. |
| H-10 | medium | pooled-repeat plots | The interval never resampled repeats: 60% coverage. | Decision: repeats, then wells, then fields (0.93). |
| H-11 | medium | `object_rule_check` | JCB on negative values, which the strategy refuses. | Decision: not applicable. |
| H-12 | low | legend | Counted well names across repeats. | Per repeat. |
| H-13, H-14 | low | duplicates | Field sample, clustering closed form. | Foundations; `first_detection_fovs`. |
| H-15 | low | ranking | Untested features listed as failing. | Listed as untested. |
| H-16 | low | variability summary | Silent about missing values. | Stated. |
| H-17 | low | plots | One generator in call order. | One per cluster. |
| H-19 | low | placement | Shared statistics outside `statistics`. | Foundations. |

## I. Workflows and data flow

Tests: `tests/test_audit2_i.py`.

| ID | Sev. | Where | Finding | Fix |
| --- | --- | --- | --- | --- |
| I-1 | **high** | `report`, `optimise`, `benchmark` | With normalisation, rule replays read normalised values; live reads raw: 149 of 180 steps differed. | Decision: every rule replay reads raw values, as live does (`workflows._shared.replay_rules`); normalised values only for outcome criteria, the contrast, the ranking and the plots. The report's verdicts now equal live on 150 of 150 synthetic and 36 of 36 idr0028 pairs. |
| I-2 | medium | declared hierarchy | Default effort strata pooled conditions (well variance 80 against 17 and 32). | One stratum per condition; `effort_strata=()` pools explicitly. |
| I-3 | medium | `optimise` | Crashed when `group_cols` left out a hierarchy level or stratum. | Says what is not assessed and why. |
| I-4 | medium | `LiveTables` | Keyed by `group_cols + fov_col`, not the field identity; refused valid tables. | The validator's field identity. |
| I-5 | medium | report | Hard-coded `few_fovs=3`, crashing for `min_other_fovs ≥ 3`. | `Setup.few_fovs`. |
| I-6 | medium | report with a control | Crashed on pooled `group_cols`. | Decision: "not assessed: group_cols pool …". |
| I-7 | medium | permutation ties | = H-2. | Group H. |
| I-8 | low | seeds | `random_state` not passed to the protocol bootstrap; orders seeded 0..n−1. | Passed; orders per group (B-4); documented. |
| I-9 | low | report | Added a novelty rule not in `Setup.rules`, not live, not calibrated. | Decision: labelled report-only and declared. |
| I-10 | low | wording | "objects" for nuclei; unusual objects claimed without a control. | The unit's word; not assessed without a control. |
| I-11 | low | `watch(until_stop)` | Used the objective's decision, not the rule's. | The rule's own verdict. |
| I-12 | low | presets | Undocumented `protocol_threshold`, `alphas`; docstrings claimed "no scientific choice". | Decision: kept, printed, reworded. |

## J. Structure

Most structural findings are fixed in the foundations or in a group above:
J-1 = H-6, J-2 = C-14, J-3 = B-4, J-4 = B-5/H-17, J-5 = G-4, J-6 = G-8,
J-7 = E-2/H-8/H-9, J-10 = D-5, J-13, J-14, J-15, J-16 and J-19 in the
foundations. The rest:

| ID | Sev. | Where | Finding | Fix |
| --- | --- | --- | --- | --- |
| J-8 | medium | plots | Plot units left out empty fields, which `bootstrap_fields` includes. | Decision: included. |
| J-9 | medium | lower bound | Two conventions (501 against 342 fields). | Foundations: one, the low end of the two-sided interval. |
| J-11 | medium | imports | Package cycle `retrospective` ↔ `sampling_unit`. | Foundations. |
| J-12 | medium | `unusual_objects` | A group with no complete features: "ok", "0 of 0 (nan%)". | Not assessed, with the reason. |
| J-17 | low | clustering | Singular by `removesuffix('s')`: "no patche". | Proper plurals. |
| J-18 | low | private names | Used across modules. | Decision: left, documented. |
| J-20 | low | long functions | Six functions of 130–200 lines. | `effort_projection`, `microscopist_report` and `_profile` split (G-18); the others left. |
| J-21 | low | deprecated shims | Used only by tests. | Decision: kept. |

## What changed for users

These are behaviour changes, not only wording:

- Refused where they used to pass or pool silently:
  - a control mean ≤ 0;
  - an absent explicit column (`repeat_col`, `technical_col`, a control
    column, plot columns);
  - a text acquisition-order column;
  - a categorical pilot order;
  - a text or non-positive area;
  - infinite coordinates;
  - a plan whose samples are constant;
  - repeated column names;
  - `combine_decisions([])`;
  - several groups in a reservoir split;
  - duplicate criteria.
- The live session applies the plane rule, and `add_fov` is atomic.
- Replays feed estimators in acquisition order and draw permuted orders per
  group from `random_state`. With normalisation, the workflows replay rules
  on raw values.
- With normalisation, only the outcome criteria, the contrast, the ranking
  and the plots read normalised values. Variability, object rules, unusual
  objects, controls, inspection and effort read raw values: question 7's
  between-repeat variance is no longer removed by the normalisation.
- `Setup.effort_strata` defaults to one stratum per condition; `()` pools.
  `Setup` gains `few_fovs`.
- New or renamed columns and fields:
  - ScalarPrecision's diagnostic `design_effect` is now `ratio_design_effect`,
    and it adds `field_size_correlation` and a `field_size_warning`;
  - the planned-count policy reads `measured_objects`;
  - the effort `projections` table gains a `design` column (acquired,
    balanced), and `components` gain `samples_acquired`, `repeats_acquired`
    and `repeat_dropped_reason`;
  - contrast and ranking gain `p_exact`;
  - clustering gains `fovs_acquired` and `fovs_unlabelled`;
  - `ReplayResult` and `ReliabilityResult` gain `skipped`;
  - `BenchmarkRecord` gains `stop_on`;
  - the calibration table gains `reason`;
  - smaller additions: `distinct_holdouts`, `longest_stream_fovs`,
    `judged_fovs`, and `reason` in `redundancy_distance`.
- Permutation p-values are exact when few assignments exist, and never go
  below the attainable floor.

The benchmark was rerun under protocol version 3
(`data/crick_idr/benchmark_protocol.py`), locked before its first run.
Version 2's lock no longer matched its protocol, because the well column had
been renamed `technical_repeat` after the lock. Version 3 keeps every
setting and records the replay changes above. Every tracked table of
`data/crick_idr/results` was regenerated with `data/crick_idr/run_all.py`;
the changed numbers are in `docs/preliminary_findings.md`.
