# MorphoBricks FOV contract: what upstream repositories need to provide

MorphoCoverage treats the field of view (FOV) as the acquisition unit. It can only do that correctly if upstream MorphoBricks tools emit FOV identity consistently. This document states the contract MorphoCoverage relies on and lists the changes needed in sibling repositories. It is a proposal: the other repositories are not changed by this document.

Evidence was first gathered from CellPatcher, fiberfeatures and NucleiSky at their default branches on 2026-09-23, and the CellPatcher and fiberfeatures items were checked again on 2026-09-28 against CellPatcher `b704a8d` and fiberfeatures `10879b6` (both of 2026-09-26). References name files and functions rather than line numbers, which move; NucleiSky was not rechecked.

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

- `condition` and `biological_repeat` are lowercase, exact, string-valued, and present in every table that leaves a tool, including per-image tables. `biological_repeat` was formerly named `repeat`; MorphoCoverage still reads `repeat` as `biological_repeat`, with a `DeprecationWarning`, and refuses a table holding both unless they are identical (`docs/data_format.md`).
- **A missing hierarchy level is written as the value `single`** (CellPatcher's convention: the level exists and has one member). A condition without repeats uses `biological_repeat="single"` (CellPatcher's `SINGLE_REPEAT_LABEL`), and a repeat without wells `technical_repeat="single"`. An empty cell keeps meaning "not recorded": an optional column (`technical_repeat`, `position_id`) empty in every row is read as absent, and an empty group value otherwise raises.
- `technical_repeat` (optional) names the technical repeat within a biological repeat: a well, a dish, a coverslip. It is never a biological repeat, which stays in `biological_repeat`. Every row of one technical repeat carries the same value, and the value is identified together with `condition` and `biological_repeat` (well `J23` of two plates is two technical repeats). MorphoCoverage uses it as a default analysis group, as part of FOV identity, and as the level it resamples or shuffles before fields.
- Other spellings (`Biological_Repeat`, `bio_repeat`, `biological repeat`, `tec_repeat`, `tech_repeat`, `technical repeat`, `Repeat`, `Condition`) are not accepted by MorphoCoverage's defaults. They raise with the column meant ("did you mean 'biological_repeat'?"), so they cannot silently bypass repeat separation.
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
- `condition`/`repeat` are real fields, and `"single"` is used for conditions without repeats (`SINGLE_REPEAT_LABEL`, `io.py`). MorphoCoverage reads CellPatcher's `repeat` as `biological_repeat` for now, with a `DeprecationWarning`.
- `technical_repeat` and `position_id` are real `PatchRegion` fields (`types.py`) and columns of `patches.csv`: `technical_repeat` from a `<condition>/<repeat>/<well>/` folder level (`"single"` without one) or per image from `image_metadata`, and `position_id` per image from `image_metadata`.
- `label_id` exists.

Needed:

0. **Rename `repeat` to `biological_repeat`** in `PatchRegion` and `patches.csv`, the shared name of the biological repeat (a deliberate, versioned change of a shared column under CellPatcher's own AGENTS.md).
1. **Emit a per-image table** (e.g. `images.csv` next to `patches.csv`). It should include images that produced zero patches and give the reason (no labels, all objects filtered by border or size). Today `patches.csv` has one row per accepted patch only, so these images disappear.
2. **Do not present `image_id` as an FOV identifier.** It is positional within a run (`image_{idx:04d}` in `pipeline.py`), so it changes if the input list changes. The stable `position_id` for the spatial acquisition position is now available (see above); what remains is to fill it wherever one file is not one position, since nothing derives it from the files.
3. **Keep plane metadata as columns.** Multi-plane images already carry `t/z/c` metadata. These must remain nested within one spatial position. If CellPatcher emits one row per plane, all rows from the same position should share `position_id`. For time-lapse, the same cell can appear in many frames; downstream tools need `t` (and ideally a track id) to avoid counting it as independent objects.
4. **Warn or refuse when `condition`/`biological_repeat` are unset.** Direct `run_patching()` calls leave them `None` (the `PatchRegion` docstring in `types.py`). MorphoCoverage rejects missing group values, so this surfaces late rather than at the source.

### fiberfeatures

Needed:

1. **Store the basename in `image_file`.** `build_metadata_from_condition_folders` writes `str(path)` (`batch.py`). Move the path to `image_path`. This change alters an existing shared column, so it needs a deliberate, versioned change, as fiberfeatures' own AGENTS.md requires.
2. **Emit only `biological_repeat`.** `_COMMON_METADATA_COLUMNS` contains both `Repeat` and `repeat` (`features.py`); fiberfeatures' AGENTS.md already lists this as a known inconsistency. MorphoCoverage refuses `Repeat` as a near miss and reads `repeat` as the former name of `biological_repeat`.
3. **Populate `biological_repeat`.** Folder metadata builds `condition` but no repeat (`build_metadata_from_condition_folders`, `batch.py`). Use `<condition>/<repeat>/` folders as CellPatcher does, or `"single"`.
4. **Emit a per-image table.** Cell rows exist only for regions above `min_region_size` (`extract_cell_features`, `features.py`), and metadata is broadcast onto the per-image frame (`batch.py`). An image with no qualifying cells therefore contributes no rows, and its acquisition is lost.
5. **Clarify `sample_id`.** It defaults to the filename stem (`sample_id_from="stem"`, `batch.py`) and is described as "one image". Either rename it to reflect that it is an image/FOV identifier, or reserve `sample_id` for a biological sample. As it stands, "sample" means an FOV.
6. **Export the explicit feature list.** Features are chosen as numeric columns minus an exclusion set (`get_numeric_feature_columns`, `features.py`). Writing the resulting list into run provenance lets MorphoCoverage use it explicitly, without re-implementing that inference.

fiberfeatures should prefer `position_id` for FOV-level blocking when it is present and fall back to `image_file` only when one file equals one spatial position. Plane/channel/time metadata remain nested within that position.

### NucleiSky

NucleiSky does not produce per-object phenotype feature tables, so it needs no change for MorphoCoverage's FOV handling. Its registration outputs (per-FOV transforms) are a natural source of `stage_x`/`stage_y` for the FOV table if they are exported keyed by `image_file`.

### Ecosystem-level

- **Keep the shared column contract in one place.** CellPatcher's and fiberfeatures' AGENTS.md copies already disagree: `repeat` (now `biological_repeat`) is "Always" in one and "If used" in the other. A single spec (e.g. in the organization `.github` or `docs-hub` repository), copied by reference into each AGENTS.md, would stop further drift.
- **Add a contract test to each producer:** a fixture with a zero-object image and duplicate basenames across two repeats, asserting the per-position table has every acquired spatial position and that `(condition, biological_repeat, position_id)` is unique when `position_id` is used (otherwise use the documented one-file/one-position `image_file` fallback).

## 3. What MorphoCoverage does today with non-conforming input

| Upstream situation | MorphoCoverage behaviour |
| --- | --- |
| `condition`/`biological_repeat` in the observations but not in the FOV table | raises |
| `condition`/`biological_repeat` in the FOV table only | attached to the observations when the FOV identifier (with the biological repeat, when names restart per repeat) maps to one value; raises otherwise |
| `repeat`, the former name of `biological_repeat` | read as `biological_repeat` with a `DeprecationWarning` (the caller's tables are not changed); a table holding both raises unless they are identical |
| `Biological_Repeat`/`bio_repeat`/`tec_repeat`/`Repeat`/`Condition` spelling | raises with "did you mean ...?" unless named explicitly in `group_cols` |
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
