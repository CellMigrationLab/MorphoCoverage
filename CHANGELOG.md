# Changelog

## Unreleased

### Removed (compatibility sweep, 2026-11-13)

The code is not in production yet, so compatibility paths for earlier outputs are removed.
The scientific refusals are unchanged. The "sweep" labels are the compatibility-sweep
inventory's, not the Stage 2 MC-1/MC-2 of the experiment description.

- **Sweep MC-1.** `coverage.json` `schema_version` 1 and 2 are no longer read. Only 3 is read. Any
  other version is refused with `coverage_version_unknown`. A schema-3 export without the
  `experiment` block still defaults no control (`experiment_block_missing`).
- **Sweep MC-2.** `html_report(..., check_source=False)` is removed. The page always checks that
  `objects` and `fovs` are the tables the report was run on, and refuses a report that records
  no source. The separate row-count check is gone too: the source digest already covers it.
- **Sweep MC-3.** `next_fov` and `final` sufficiency tables are no longer read in their older forms.
  A `next_fov` table without `uninformative` and a `final` table without `warning_list`
  are no longer accepted. The joined `warnings` text is no longer read as a fallback.

### Fixed

- **#577.** The lab converters hash a downloaded source archive in chunks
  (`hashlib.file_digest`) instead of reading it whole into memory (381 MB for CCT8 CDM
  morphology); the md5 and the member's sha256 are checked as before.
- **#583.** The reference check (`data/reference/reference.py`) refuses a job whose kind is
  not one it runs or whose name is not one plain word (a separator, `.` or `..`), and a job
  listed twice, before anything runs; the owner's job names are checked the same way, and
  `run_job` refuses a target outside `OUT` before removing it. A job list could make it
  remove a folder outside `OUT` (`../.. victim`).
- **#591.** The benchmark's Colab notebook (`data/simulations/decisions/benchmark_colab.ipynb`)
  no longer puts the GitHub token in the clone URL, where it was in the command, in the
  clone's `.git/config` and in the error of a failed clone: git reads it through a
  `GIT_ASKPASS` helper from the clone's environment, the remote is checked to be the
  token-free URL, and a failed clone raises with git's message (token masked), not the
  command. A clone made by the earlier cell keeps the token in
  `/content/MorphoCoverage/.git/config` until that Colab session ends.
- **#547.** `BenchmarkRecord.as_row` (and so `records_to_frame`) exports each outcome
  metric's config id as `outcome_id.<name>` beside `outcome.<name>`: two metrics given the
  same label (`RelativeMeanError("feature_a", label="readout_error")` and the same for
  `feature_b`) could not be told apart once exported. The existing columns are unchanged.
- **#449.** The detection-adjusted unseen-state bound names each group by reading its rows by
  column name: a group column that is not a Python identifier (`"plate id"`, `"class"`)
  raised `AttributeError`, since `itertuples` renames it, whenever an adjusted bound was
  available. The bound and its wording are unchanged.
- **#448.** The guided session asks again, instead of keeping only the last condition, when
  treatment keys entered for one control repeat or are empty, and when a condition given a
  control later finds both its proposed key and its own name taken under that control (the
  condition holding the key was overwritten). Every condition assigned to a control is kept.
- **#430.** `plan_objects_to_average` refuses `sample_cols` under which one sample holds
  both control and other cells, naming the samples: each was centred on its own mean, so the
  treatment's shift entered the within-sample SD (a plan of 107 cells instead of 1 in the
  issue's example). Add the column that tells them apart, such as the condition.
- **#581.** A `FrozenDict` or `FrozenList` built by its constructor now freezes what it is
  given, so `freeze`, which returns such a container unchanged, no longer keeps a caller's
  inner list or dict that could still change a snapshot. Pickling, copying and `freeze` of
  plain containers behave as before.
- **#551.** `document` compares the parts' Setups field by field with a contrast table
  compared by its values (`DataFrame.equals`): a Setup declared with `contrasts=` raised
  "the truth value of a DataFrame is ambiguous" even against itself. Different contrast
  tables are still refused as different Setups.
- **#532.** The descriptor-spread resample helpers (`strategies._descriptor_resamples`)
  refuse with a `ValueError` what they cannot resample: no value (a `ZeroDivisionError`
  before), no FOV, no resample, or positions that are not one FOV of `k` per value. Their
  caller, `DescriptorSpreadEstimator`, already reports these cases as not yet possible, so
  no result changes.
- **#485.** A complex column of another feature table is refused by `attach_feature_tables`
  ("has complex columns"), before any float cast dropped its imaginary part: an infinity in
  the imaginary part went unseen and a varying one was listed as constant. Nullable real
  columns are read as before.
- **#495.** `ColumnWeights` refuses booleans and text as weights (`True` became 1.0 and
  `"2.0"` 2.0 through `float`); `BlockWeights` refuses an `inertia_radius` that is not a
  boolean (the text `"False"` turned the radius correction on while its provenance said
  `"False"`) and block feature names that are not non-empty text (they were recast with
  `str`). Valid settings give the same results.
- **#491.** `whole_table_tests` refuses `permutations` that is not an integer of at least 1
  (0 drew no shuffle and still gave `p_shuffle = 1.0`; negative, fractional, boolean and
  text values were not checked) and a `random_state` that is not a non-negative integer.
- **#528.** `repeat_effect` refuses a `level` outside (0, 1) (1 gave an infinite interval, a
  negative level an invalid quantile) and an infinite per-repeat contrast (it gave NaN bounds
  after runtime warnings). A missing contrast still gives no interval (NaN), as documented.
- **#538.** Stage coordinates a float cannot hold are refused with the contract's
  `ValueError`: a Python integer too large for a float (it raised a bare `OverflowError`) as
  not finite, and an integer beyond 2**53 that a float would round (`2**53` and `2**53 + 1`
  became one position) as not held exactly, in `stage_matrix` and `check_xy` alike. Nothing
  is rounded or rescaled.
- **#544.** Imaged areas that are each finite but whose total overflows a float (two fields
  of `1e308`) are now a problem of `_areas.area_problem`: `ImagedAreaEstimator` reports the
  area `not_applicable` with that reason, so `FixedAreaStrategy` no longer stops on an
  infinite total, and a benchmark refuses the dataset instead of recording an infinite
  `area`/`total_area`.
- **#418.** `robust_sd(values, "geometry")` of a 1-D sample of one value returns 0 (no
  spread) instead of raising `IndexError`: `constant_columns` sized its zero SD by a second
  axis that a 1-D sample does not have.
- **#578.** The glossary's "Sign-flip test" no longer calls it "the exact test on the signs":
  it is a two-sided paired permutation test on the mean of the per-repeat differences, each
  difference keeping its magnitude while its sign is flipped, exact up to 16 repeats and
  Monte Carlo (20,000 assignments) above, as `sign_flip_test` computes it. The source is
  `glossary.py`; `docs/common/glossary.md` is re-rendered from it. The test is unchanged.
- **#90 (part).** A paired test across repeats that gives no p because 3 or more repeats all
  give the same change is now "not assessed across repeats" (`reading_text.NOT_ASSESSED`),
  never "not significant": in the reading, the summary, and the page's table across repeats,
  where the verdict gives the reason (`contrast_repeats.NO_SPREAD`). Panel C of the figures
  marks such a pair "=" with its reason, apart from "–" (fewer than 3 repeats imaged both),
  in the static figure, its legend and the interactive hover, which said "fewer than 3" for
  both. "Plan the next experiment" keeps a condition whose repeat plan has no estimable
  answer, as "not assessable" with the plan's status, instead of leaving it out (and then
  saying every condition was significant). The shared machine-readable status vocabulary of
  #90 is not part of this change.
- **#557.** The PCA figure's loading labels are placed beside their own arrows. Labels were
  moved apart from the lowest up and returned in that order, while the names followed the
  loadings' order, so a feature's name could sit at another feature's arrow; the positions
  now come back in the loadings' order (`plots_embedding._spread_labels`).
- **#555.** `percent` (and `percent_fine`) no longer raises a math domain error for a share
  just below 100% (1 - 1e-12, or the float just below 1): the gap to 100% was rounded to 0
  before its log was taken. The gap is now kept to the digits a float holds of it, so such a
  share reads `99.9999999999%`, never `100%`; ordinary shares read as before.
- **#628 (item 3).** On exports that name conditions by `condition_id` or pair repeats by
  `repeat_group`, the acquisition anomalies' rows now name their folders, as the comparison
  rows and the fields to check do: a `folder` column and the folders in the row's field
  (`ctrl · repeat rep1 · img0 (folder control/rep1)`), in the report and on its page
  (`_pairing.with_folder_labels`). A table whose folders are their IDs is unchanged.
- **#642.** `html_report`, `plots` and `plots_of_data` with `group_cols=("condition",
  "repeat_group")`, the groups of MorphoFeature's notebooks, no longer raise "group_cols ...
  hold ['repeat_group'], which the figure does not place". The figures place `repeat_group`
  as their biological-repeat level (`_repeat_groups.repeat_level`), labelled by its values;
  the pairwise matrix pairs by it and the HTML report's interactive figures read it. Nothing
  falls back to the folder label `biological_repeat`; `group_cols` naming both columns are
  refused, since a figure has one repeat level. The `biological_repeat` path is unchanged.

### Changed

- **Q-MC-a.** `microscopist_report` refuses (`ValueError`) a result that records no acquisition
  identity, alone or beside others. It used to assemble such a result with a note saying it
  could not be checked against the others; `_acquisition.unchecked_note` is removed and
  `check_one_acquisition` returns nothing. Results of two acquisitions are still refused.
- `retrospective.redundancy_distance` returns a `RedundancyResult` (`table`, the former
  per-group DataFrame, and `settings`, with the acquisition identity, the features, stage
  columns, bins, share, unit, permutations, seed and observation unit) instead of a bare
  DataFrame, so it is checked against the other results like the other 18 result types.
  `microscopist_report(redundancy=...)` takes that result (a DataFrame is refused), and
  `optimise(..., redundancy_bins=...)` stores it in `Plan.results["redundancy"]`.
- `representation_control` without a positive control ("not assessed") validates the tables
  and records their acquisition identity before it returns, so the report accepts that
  result instead of refusing it. Tables it used to return early on without checking are now
  validated like every other call.

- **Q-MC-b.** `repeat_plan` refuses (`ValueError`) a contrast table without `mean`,
  `control_mean`, `log_ratio_se` or `difference_se`. It used to plan such a table on Cohen's d
  without saying so. The plan still rests on d when a value or its sampling error is not
  finite.
- `repeat_plan` also refuses a table without `d_low`, `d_high` or `between_share`. Without
  `d_low` it used to return an empty plan without saying why, and without `between_share` it
  used to halve the whole sampling error with twice the fields. `across_repeats` refuses a
  table without `mean` or `control_mean` (it used to return an empty table). Missing
  values in these columns are handled as before.
