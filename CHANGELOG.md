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

- **Q-MC-b.** `repeat_plan` refuses (`ValueError`) a contrast table without `mean`,
  `control_mean`, `log_ratio_se` or `difference_se`. It used to plan such a table on Cohen's d
  without saying so. The plan still rests on d when a value or its sampling error is not
  finite.
- `repeat_plan` also refuses a table without `d_low`, `d_high` or `between_share`. Without
  `d_low` it used to return an empty plan without saying why, and without `between_share` it
  used to halve the whole sampling error with twice the fields. `across_repeats` refuses a
  table without `mean` or `control_mean` (it used to return an empty table). Missing
  values in these columns are handled as before.
