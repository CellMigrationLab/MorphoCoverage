# Changelog

## Unreleased

### Removed (compatibility sweep, 2026-11-13)

The code is not in production yet, so compatibility paths for earlier outputs are removed.
The scientific refusals are unchanged.

- **MC-1.** `coverage.json` `schema_version` 1 and 2 are no longer read. Only 3 is read. Any
  other version is refused with `coverage_version_unknown`. A schema-3 export without the
  `experiment` block still defaults no control (`experiment_block_missing`).
- **MC-2.** `html_report(..., check_source=False)` is removed. The page always checks that
  `objects` and `fovs` are the tables the report was run on, and refuses a report that records
  no source. The separate row-count check is gone too: the source digest already covers it.
- **MC-3.** `next_fov` and `final` sufficiency tables are no longer read in their older forms.
  A `next_fov` table without `uninformative` and a `final` table without `warning_list`
  are no longer accepted. The joined `warnings` text is no longer read as a fallback.

### Changed

- **Q-MC-b.** `repeat_plan` refuses (`ValueError`) a contrast table without `mean`,
  `control_mean`, `log_ratio_se` or `difference_se`. It used to plan such a table on Cohen's d
  without saying so. The plan still rests on d when a value or its sampling error is not
  finite.
