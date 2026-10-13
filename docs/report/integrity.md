# Data integrity: copied data to inspect before reading the answers

Issue #49 asked which ideas of [PaperGuard](https://github.com/exergyleizhou-ux/PaperGuard)
(an MIT-licensed statistical and data-integrity screener) are useful for
microscopy feature tables. This note answers its five questions and describes
what was built: `morphocoverage.integrity_checks`, reported in the overview of
`workflows.report(..., integrity=True)`.

A finding is **structure to inspect**, never a judgment of intent: the report
says "check where these fields came from (a file read or exported twice, a
field identifier reused, a copy in another unit)".

`IntegrityResult.summary()` opens with ❌ when something was found, ✅ when
nothing was, and ➖ when no check could be applied to the features; the short
form gives the findings as one table row per check and pair of groups, and
`summary(detail="full")` every finding with its detail. Both state what the
checks could not compare and what they cannot see, once.

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
| A row that repeats another row exactly, identifiers included (rows with no feature value, such as failed segmentations, are left out) | `check_tables` and the live sources (a warning) |
| An observation that repeats another of the same field in every feature while another column differs (a merge with a table that has one row per channel, a table appended to itself and saved with a fresh index) | `check_tables` (a warning) |
| An object label (`label_id`, CellProfiler's `ObjectNumber`, `object_id`, `label`) twice in one field | `check_tables` (a warning, also when a copy warning comes first); the live sources warn for a `label_id` twice |
| A field listed twice in the FOV table | the validator and `check_tables` (an error) |
| Plane or tile files counted as fields | the plane rule (validator, live session) |
| Planes of one field counted as new observations of its objects, or several stage positions of a multi-position file counted as one field | the validator (an error; see `docs/common/data_format.md`) |
| A manifest line written twice, or a manifest rewritten under the live watcher | `LiveFolder` (a warning; an error) |
| Fields unlike the others (outliers, crowded or empty fields, new states) | `inspect_fovs`, `fov_variability`'s unusual fields |
| Between-field variation, including a collapsed one | `fov_variability` (ICC, design effect) |

None of these catches a field **copied under another name**: its rows carry
another field's identifiers, so they are not duplicate rows, and its values
look like any field's.

## 3. What the new checks catch

`integrity_checks(objects, fovs, feature_cols=...)` compares every acquired
field with every other one, across groups, on the declared features. With
[other feature tables](feature_tables_and_maps.md#other-feature-tables-embeddings),
the report and the scan run the checks on the explainable features and on
each table apart, and give the union, naming the table each copy was found in
(owner, 2026-10-06): pooled, a copy confined to one table, an embedding
computed twice for one image, was missed.
Observations are compared **to 6 significant digits**, so a copy that went
through float32 (parquet, HDF5, `.npy`) or a six-digit export still matches:

- **copied field**: the same observations in the same order (a file read or
  exported twice under another name);
- **reordered copy**: the same observations in another order (a copy that
  was sorted, or merged back in another order);
- **scaled copy**: the same observations under a change of units,
  `a * x + b` per feature (`a` may be negative, so a sign convention counts),
  in any order, with missing values in the same places (pixels against
  micrometres, a re-export in raw grey levels);
- **repeated block**: a run of consecutive observations shared, in table
  order, by two fields (rows pasted into another field, or a copy with an
  observation dropped or added); the finding gives the whole run and where it
  starts;
- **shared observations**: two fields share many observations (at least 10,
  `block_length`, or half the smaller field) but not in one run: a copy with
  observations dropped or added and reordered, or a block pasted in another
  order;
- **copied small field**: every observation of a field too small for the
  checks above (fewer than 5, down to one cell per field) is also in one other
  field, each with values in at least 3 continuous features
  (`DISTINCTIVE_FEATURES`), so a coincidence to 6 significant digits is not
  plausible: positions of 2-4 tracks exported twice under two names, or a
  manually measured cell's row written twice (Fiji's Measure pressed twice).
  With fewer than 3 continuous features this check is not applicable, since
  one shared observation can then be a coincidence. Added after the lab
  datasets (`data/lab/FINDINGS.md`): the default thresholds missed 3 of 6
  copied track positions of 4 tracks, and could never see a copied cell with
  one cell per field;
- **repeated within a field**: a field holds many of its observations twice
  ("the field holds its observations 1 to 28 again from observation 29": a table
  appended to itself, or merged with one that has a row per channel);
- **repeated field summary**: two fields with the same number of
  observations and the same mean of every feature (to 12 significant digits)
  although their observations differ (a summary row copied).

Each finding says what was found, not a probability; there is no global
integrity or suspicion score. Coincidences are kept out in three ways, each
stated in the summary with what it set aside:

- a feature is **continuous** when its distinct values are at least half
  (`min_distinct_share`) of the distinct observations; with no continuous
  feature (quantized or integer features only) every check says **not
  applicable**, with its reason. Wide-range integers (pixel areas, raw
  intensity sums) are continuous;
- a field is compared **as a whole** only with at least 5 observations
  (`min_objects`) and 5 distinct values of a continuous feature: two fields of
  one or two values map onto each other by construction. The summary counts
  the fields compared and why the others were not;
- an observation found in **three or more fields** (identical hot pixels,
  empty background patches) is low-information: it is set aside and counted,
  so a few repeated background objects neither make copies of every field nor
  hide a copy of a real one.

Every step is one pass over the table or a sort: each field has a fingerprint
per feature that no order, change of units, sign flip or six-digit rounding
changes (the absolute skewness and the kurtosis of its standardised values);
fields with near-equal fingerprints, found by sorting, are then verified by
aligning their rows on every standardised feature. No pair of fields is
compared unless its fingerprints agree, so fields of one fixed size (a
CellProfiler or CellPatcher grid) cost no more than fields of any size.

**Out of reach**, and said in every summary: a copy re-rounded to fewer than
6 significant digits (exported with 4 decimals), a block of fewer than 10
observations that is not the whole of the smaller field, a copy with two feature columns
swapped or a feature (or a value) missing in one copy only, and observations
pasted into two or more other fields (they read as recurring).

## 4. Where they live

In their own module, `morphocoverage.integrity`, exported at the top level,
and as an **opt-in** part of the report's overview (`report(...,
integrity=True)`): what the tables hold includes the structure to inspect
before reading any answer. Findings are grouped by check and pair of groups
(28 copies from one well into another make one line, not 28) and the list is
capped. They do not change any other analysis, but the report says so: the
fields they name are listed under "6. Which fields should I look at?" and in
"What this dataset cannot tell me", since every answer counts them as
independent fields.

## 5. How they were validated

- **Synthetic copies** (`tests/test_integrity.py`, `tests/test_audit4_a.py`):
  into the committed idr0028 table and into constructed tables, one field is
  replaced by each kind of copy of another field (exact, shuffled, through
  float32 or a six-digit export, scaled with a sign flip on one feature, with
  ties and with missing values, with a row dropped and shuffled, a pasted
  block in and out of order, a table merged with one row per channel). Each is
  found as one finding of its kind.
- **Negative controls**: the committed idr0028 (336 fields, 9550 nuclei) and
  idr0139 (40 fields, 10748 nuclei) tables and `data/synthetic` give no
  finding on every declared feature (measured: 0.4 s and 1.0 s). So do
  fields sharing the same hot-pixel objects, one-row field summaries, 8-bit
  quantized intensities (not applicable), an integer count beside a
  continuous area, and the same field imaged twice with 1% noise.
- **Scale**: 2000 fields of 16 observations (every pair of equal size)
  finishes in under a second, and 10,000 fields of about 100 observations
  (10^6 observations, three features) in about 10 s on a shared machine.
- **Not applicable**: a feature taking three integer values makes every
  check say why it cannot be read.

## Not built, on purpose

Near-identical (not identical) field distributions, "over-smooth" data and
collapsed between-field variation are exploratory: they are not
deterministic and would need calibration on imaging datasets before any
threshold could be trusted. The between-field share is already reported by
`fov_variability` with its uncertainty; a calibrated near-duplicate
detector stays [open work](../common/open_work.md).
