# MorphoBricks FOV contract: what upstream repositories need to provide

MorphoCoverage treats the field of view (FOV) as the acquisition unit. It can only do that correctly if upstream MorphoBricks tools emit FOV identity consistently. This document states the contract MorphoCoverage relies on and lists the changes needed in sibling repositories. It is a proposal: the other repositories are not changed by this document. MorphoFeature's own side of the same table interface is its [feature-handoff contract](https://github.com/CellMigrationLab/MorphoFeature/blob/main/docs/contracts/feature-handoff.md).

Evidence was first gathered from CellPatcher, fiberfeatures and NucleiSky at their default branches on 2026-09-23. The CellPatcher and fiberfeatures items were last checked on 2026-10-02 against CellPatcher `783d8a1` and fiberfeatures `6651566` (both of 2026-10-01, their `main`). Work on an open branch is named as such, not counted as done. References name files and functions rather than line numbers, which move; NucleiSky was not rechecked.

## 1. Proposed shared contract

### FOV identity

- **One FOV is one independently selected spatial acquisition position (XY position).** This is the unit MorphoCoverage counts when asking how many positions are required.
- **Z, C and T are nested within a FOV.** A z-stack, multichannel acquisition, or time series at one position is still one FOV. Extra planes/channels/time points increase information within that position; they do not increase the number of sampled positions.
- **Use `position_id` when file layout is not 1:1 with positions.** This is the preferred explicit identifier. Examples:
  - one OME-TIFF containing one complete z-stack at one position: `image_file` may identify the FOV;
  - one TIFF per z-plane: every plane file shares the same `position_id`;
  - one ND2/CZI/LIF containing multiple stage positions: each position must have its own `position_id`, even though `image_file` is shared.
- **`image_file` is storage provenance, not the fundamental scientific definition of an FOV.** It remains a valid fallback identifier only when one file corresponds to exactly one spatial position.
- **FOV identity is therefore normally `(condition, biological_repeat, position_id)` when `position_id` exists, otherwise `(condition, biological_repeat, image_file)` when the one-file/one-position assumption holds** (with `technical_repeat` after `biological_repeat` when present).
- **`image_file` holds the file name (basename), not a machine-specific path.** If a path is needed, store it separately as `image_path`.

### Slice-by-slice or multidimensional feature extraction

Feature extraction may be 2D, 3D, per-slice, per-channel, per-timepoint, or any combination. MorphoCoverage does not prescribe how features are computed.

If features are computed on individual planes or frames, the object/feature rows must preserve the relevant lowercase metadata:

- `z` for z-plane or z-index;
- `c` for channel;
- `t` for timepoint.

All rows from the same spatial position must carry the same `position_id`. MorphoCoverage treats `z`, `c`, and `t` as within-FOV acquisition metadata when they encode acquisition dimensions; because phenotype features are explicit, these columns should simply be omitted from `feature_cols` in that case.

Sequential replay reveals an entire spatial position atomically. It must never reveal one z-plane from a stack as if it were a newly sampled FOV.

### Column names

The shared MorphoBricks columns are `position_id`, `image_file`, `z` or `z0`/`z1`, `c`, `t` (all 0-based), `condition`, `biological_repeat` and `technical_repeat`, with `mask_file` and `patch_file` as provenance where used.

- `condition` and `biological_repeat` are lowercase, exact, string-valued, and present in every table that leaves a tool, including per-image tables. `biological_repeat` was formerly named `repeat`; MorphoCoverage still reads `repeat` as `biological_repeat`, with a `DeprecationWarning`, and refuses a table holding both unless they are identical (`docs/common/data_format.md`).
- **A missing hierarchy level is written as the value `single`** (CellPatcher's convention: the level exists and has one member). A condition without repeats uses `biological_repeat="single"` (CellPatcher's `SINGLE_BIOLOGICAL_REPEAT_LABEL`, formerly `SINGLE_REPEAT_LABEL`, a name it keeps), a root folder without condition folders `condition="single"` (`SINGLE_CONDITION_LABEL`), and a repeat without wells `technical_repeat="single"`. An empty cell keeps meaning "not recorded": an optional column (`technical_repeat`, `position_id`) empty in every row is read as absent, and an empty group value otherwise raises.
- `technical_repeat` (optional) names the technical repeat within a biological repeat: a well, a dish, a coverslip. It is never a biological repeat, which stays in `biological_repeat`. Every row of one technical repeat carries the same value, and the value is identified together with `condition` and `biological_repeat` (well `J23` of two plates is two technical repeats). MorphoCoverage uses it as a default analysis group, as part of FOV identity, and as the level it resamples or shuffles before fields.
- Other spellings (`Biological_Repeat`, `bio_repeat`, `biological repeat`, `tec_repeat`, `tech_repeat`, `technical repeat`, `Repeat`, `Condition`) are not accepted by MorphoCoverage's defaults. Nor are typing errors of the shared names (`biological_repat`, `techincal_repeat`, `conditon`, `image_flie`: one edit, or two for the repeats' longer names). They raise with the column meant ("did you mean 'biological_repeat'?"), so they cannot silently bypass repeat separation.
- `label_id` identifies an object within an image, as both CellPatcher and fiberfeatures already use.

### A per-image table (the FOV table)

An observation table cannot represent a spatial position in which nothing was measured or retained: that position simply has no rows. Every tool that iterates over images should therefore also write one row per image it was given:

| Column | Required | Notes |
| --- | --- | --- |
| `condition`, `biological_repeat` | yes | analysis grouping / identity context |
| `technical_repeat` | optional | the well, dish or coverslip within the biological repeat; a technical repeat, never a biological one |
| `position_id` | preferred | explicit spatial acquisition-position identity; required when file != position |
| `image_file` | yes | storage provenance; may identify FOV only when one file = one position |
| `status` | yes | `processed`, or `failed`/`skipped` with a `reason` |
| `n_objects` | yes | Objects retained after the tool's own filters; may be 0 |
| `n_objects_detected` | recommended | Before size/border filters, so filtering is visible |
| `acquisition_order` | if known | Integer, no ties within `condition × biological_repeat` |
| `stage_x`, `stage_y` | if known | Physical units stated in run metadata |
| `pixel_size`, `height`, `width` | if known | Allows imaged area to be computed |

**A failed image is not a zero-object FOV.** A segmentation failure, missing mask or unreadable file must be marked as such, not reported as an image with `n_objects = 0`. Otherwise MorphoCoverage would count an acquisition failure as evidence that part of the sample contains no cells.

### Features

Feature columns should be listed explicitly in the run's provenance (for example in the run-config JSON), so downstream tools can pass `feature_cols` without inferring them from numeric columns. Any dataset-level normalization or projection (z-scoring across all images, a PCA fitted on the whole dataset) must be recorded. It makes sequential benchmarking leak future FOVs.

## 2. Changes needed per repository

### CellPatcher

Already compatible:

- `image_file` is the basename (`orchestration.py`, the per-image metadata of folder runs).
- `condition` and `biological_repeat` are real `PatchRegion` fields (`types.py`) and columns of `patches.csv`; `repeat` is read and written only as a deprecated alias (`_deprecation.py`). A condition folder without repeat folders is labelled `"single"` (`SINGLE_BIOLOGICAL_REPEAT_LABEL`, `io.py`), and so is a root without condition folders (`SINGLE_CONDITION_LABEL`).
- `position_id` is a real `PatchRegion` field and column, set per image from `image_metadata` (`_stamp_image_metadata`, `pipeline.py`), and the `t`/`z0`/`z1`/`c` of each plane are columns beside it. Planes of one position share it when the caller gives them the same value; nothing derives it from the files.
- `label_id` exists.

All four changes this contract once asked of CellPatcher are done (re-verified 2026-11-07 against local CellPatcher `031bba9`, not fetched; the contract of record is CellPatcher's [configuration reference](https://github.com/CellMigrationLab/CellPatcher/blob/main/docs/configuration_reference.md), which this page does not restate):

1. **Per-image table:** `images.csv` is written beside every `patches.csv`, one row per image considered including images without patches (`IMAGE_TABLE_FILENAME`, `export.py`).
2. **`image_id` is not an FOV identifier:** the reference calls it the image's place in the run, "not an identity across runs"; the stable identifier is `position_id` or `image_file`.
3. **`technical_repeat`** is a `PatchRegion` field (`types.py`) and the `<condition>/<biological_repeat>/<technical_repeat>/` folder level exists (`io.py`).
4. **Unset `condition`/`biological_repeat`:** `CommonConfig` now defaults them to the `"single"` labels (`config.py`) instead of `None`.

### fiberfeatures

Already compatible:

- `image_file` is the basename and the path is in `image_path` (`build_metadata_from_condition_folders`, `batch.py`).
- `condition`, `biological_repeat` and `technical_repeat` come from the `<condition>/<biological_repeat>/<technical_repeat>/` folder levels, a missing level being `"single"` (`_FOLDER_LEVELS`, `SINGLE_LABEL`, `batch.py`). Only `biological_repeat` is written; `Repeat` and `repeat` stay in `_COMMON_METADATA_COLUMNS` (`features.py`) so that an older input table's column is never scored as a feature.
- The patch table (`data/experiment_001/reference/patches.csv`) passes MorphoCoverage's data check as written, with `image_file` as the field; a cut of it is MorphoCoverage's fixture `data/morphobricks/fiberfeatures_patches.csv`. Its `patch_*` columns (where a patch was cut) are not offered as features by `scan_design`.

Needed:

1. **Emit a per-image table.** In cellular mode, cells below `min_region_size` are skipped (`extract_cell_features`, `features.py`), so an image with no qualifying cells contributes no rows and its acquisition is lost; patch skips are counted for the whole run (`patch_skip_counts`), not per image.
2. **Clarify `sample_id`.** It is still the filename stem by default (`sample_id_from="stem"`, with `"name"` and `"path"` as options and a duplicate check, `batch.py`), and the glossary still calls a sample "one image". Either rename it to say it is an image identifier, or reserve `sample_id` for a biological sample.
3. **Export the explicit feature list.** Features are chosen as numeric columns minus an exclusion set (`get_numeric_feature_columns`, `features.py`), and no run file records the resulting list. Writing it into run provenance lets MorphoCoverage use it explicitly, without re-implementing that inference.

fiberfeatures should prefer `position_id` for FOV-level blocking when it is present and fall back to `image_file` only when one file equals one spatial position. Plane/channel/time metadata remain nested within that position.

### NucleiSky

NucleiSky does not produce per-object phenotype feature tables, so it needs no change for MorphoCoverage's FOV handling. Its registration outputs (per-FOV transforms) are a natural source of `stage_x`/`stage_y` for the FOV table if they are exported keyed by `image_file`.

### Ecosystem-level

- **Keep the shared column contract in one place.** CellPatcher's and fiberfeatures' AGENTS.md copies agree today (`biological_repeat` "Always", `technical_repeat` "Optional", `position_id` "If needed"), after disagreeing on 2026-09-28. A single spec (e.g. in the organization `.github` or `docs-hub` repository), copied by reference into each AGENTS.md, would keep them from drifting again.
- **Add a contract test to each producer:** a fixture with a zero-object image and duplicate basenames across two repeats, asserting the per-position table has every acquired spatial position and that `(condition, biological_repeat, position_id)` is unique when `position_id` is used (otherwise use the documented one-file/one-position `image_file` fallback).

## 3. What MorphoCoverage does today with non-conforming input

| Upstream situation | MorphoCoverage behaviour |
| --- | --- |
| `condition`/`biological_repeat` in the observations but not in the FOV table | raises |
| `condition`/`biological_repeat` in the FOV table only | attached to the observations when the FOV identifier (with the biological repeat, when names restart per repeat) maps to one value; raises otherwise |
| `repeat`, the former name of `biological_repeat` | read as `biological_repeat` with a `DeprecationWarning` (the caller's tables are not changed); a table holding both raises unless they are identical |
| `Biological_Repeat`/`bio_repeat`/`tec_repeat`/`Repeat`/`Condition` spelling, or a typing error (`biological_repat`, `conditon`) | raises with "did you mean ...?" unless named explicitly in `group_cols` |
| A patch table's geometry (`patch_y0`, `patch_center_x`, `y0`, `center_x`) | never offered as features by `scan_design`; declared features are used as given |
| A level recorded as `single` | one member of that level (one biological or technical repeat), like any other label |
| Same `image_file` in different repeats, pooled | kept as distinct FOVs |
| Several objects/planes from one `position_id` | one FOV, regardless of file layout |
| One file per z-plane with shared `position_id` | one FOV; all planes revealed together |
| One multidimensional file containing several positions | requires explicit `position_id`; `image_file` alone is not sufficient |
| `image_file` in objects but not in FOV table | raises (no fallback to `fov_id`) |
| Objects whose FOV is missing from the FOV table | raises |
| No FOV table supplied | FOVs derived from objects with a `MissingFovTableWarning` (`fovs_from_observations=True`); zero-object FOVs cannot be represented |
| Full path in one table, basename in the other | raises as objects absent from the FOV table; cannot be auto-repaired |
| Failed image reported as zero objects | **not detectable**; must be fixed upstream |
