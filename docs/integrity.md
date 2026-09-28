# Data integrity: copied data to inspect before reading the answers

Issue #49 asked which ideas of [PaperGuard](https://github.com/exergyleizhou-ux/PaperGuard)
(an MIT-licensed statistical and data-integrity screener) are useful for
microscopy feature tables. This note answers its five questions and describes
what was built: `morphocoverage.integrity_checks`, reported in the overview of
`workflows.report(..., integrity=True)`.

A finding is **structure to inspect**, never a judgment of intent: the report
says "check where these fields came from (a file read or exported twice, a
field identifier reused, a copy in another unit)".

## 1. Which PaperGuard ideas apply to image-derived feature tables?

Only the **deterministic duplication** detectors: exact duplicates, reordered
duplicates, affine (scaled or shifted) copies, repeated blocks, repeated
summaries. They answer "is this the same data twice?" exactly, whatever the
feature's distribution.

Its **statistical screens do not apply**, and were left out:

| PaperGuard screen | Why not for microscopy features |
| --- | --- |
| Benford's law | Features are bounded, scaled measurements (areas, intensities, ratios), not quantities spanning orders of magnitude. |
| Terminal digits | Digits come from floating-point arithmetic, quantization and export precision, not from people writing numbers. |
| Zero or missing patterns | Missing values mark segmentation or feature-extraction failures, and empty fields are real acquisitions (the FOV contract). |
| Low ICC or "too smooth" data | A low between-field share is a property of the phenotype; it is what `fov_variability` measures and reports, with its uncertainty. |

## 2. What does MorphoCoverage already cover?

| Failure | Where it is caught |
| --- | --- |
| A row that repeats another row exactly, identifiers included | `check_tables` (a warning) |
| A `label_id` twice in one field | `check_tables` |
| A field listed twice in the FOV table | the validator and `check_tables` (an error) |
| Plane or tile files counted as fields | the plane rule (validator, live session) |
| Fields unlike the others (outliers, crowded or empty fields, new states) | `inspect_fovs`, `fov_variability`'s unusual fields |
| Between-field variation, including a collapsed one | `fov_variability` (ICC, design effect) |

None of these catches a field **copied under another name**: its rows carry
another field's identifiers, so they are not duplicate rows, and its values
look like any field's.

## 3. What the new checks catch

`integrity_checks(objects, fovs, feature_cols=...)` compares every acquired
field with every other one, across groups, on the declared features:

- **copied field**: the same observations in the same order (a file read or
  exported twice under another name);
- **reordered copy**: the same observations in another order (a copy that
  was sorted, or merged back in another order);
- **scaled copy**: the same observations under a change of units,
  `a * x + b` per feature, matched row for row (pixels against micrometres, a
  re-export with another intensity scale);
- **repeated block**: 10 consecutive observations (`block_length`) that
  appear in two fields, or twice in one (rows pasted into another field);
- **repeated field summary**: two fields with the same number of
  observations and the same mean of every feature (to 12 significant digits)
  although their observations differ (a summary row copied).

Each check is exact up to floating-point rounding, so a finding is not a
probability. A check says **not applicable**, with its reason, when a
coincidence could produce it: when fewer than half the observations are
distinct (`min_distinct_share`: quantized or integer features repeat by
themselves), or, for summaries, when a field holds only integers (counts).
Fields with fewer than 5 observations (`min_objects`) are not compared as
copies. There is no global integrity or suspicion score.

## 4. Where they live

In their own module, `morphocoverage.integrity`, exported at the top level,
and as an **opt-in** part of the report's overview (`report(...,
integrity=True)`): what the tables hold includes the structure to inspect
before reading any answer. They do not change any other analysis.

## 5. How they were validated

- **Synthetic copies** (`tests/test_integrity.py`): into the committed idr0028
  table, one field is replaced by each kind of copy of another field (an
  exact copy, a shuffled copy, a copy with one feature scaled and another
  shifted, 12 rows pasted into another field, a field with two values
  swapped so the rows differ but every mean is the same). Each is found as
  exactly one finding of its kind.
- **Negative controls**: the committed idr0028 (336 fields, 9550 nuclei) and
  idr0139 (40 fields, 10748 nuclei) tables and `data/synthetic` give no
  finding on every declared feature, in under a second each.
- **Not applicable**: a feature taking three integer values makes every copy
  check and the summary check say why they cannot be read.

## Not built, on purpose

Near-identical (not identical) field distributions, "over-smooth" data and
collapsed between-field variation are exploratory: they are not
deterministic and would need calibration on imaging datasets before any
threshold could be trusted. The between-field share is already reported by
`fov_variability` with its uncertainty; a calibrated near-duplicate
detector stays open work (`docs/questions.md`).
