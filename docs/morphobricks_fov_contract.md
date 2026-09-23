# MorphoBricks FOV contract: what upstream repositories need to provide

MorphoCoverage treats the field of view (FOV) as the acquisition unit. It can only do that correctly if upstream MorphoBricks tools emit FOV identity consistently. This document states the contract MorphoCoverage relies on and lists the changes needed in sibling repositories. It is a proposal: the other repositories are not changed by this document.

Evidence was gathered from CellPatcher, fiberfeatures and NucleiSky at their default branches on 2026-09-23. File references are to those checkouts.

## 1. Proposed shared contract

### FOV identity

- **One `image_file` is one FOV.** A z-stack, time-lapse, or multichannel image acquired at one stage position is one FOV. Planes are described by separate metadata columns (`z`, `t`, `c`), never by a different `image_file`.
- **A container holding several stage positions is not a valid `image_file`.** Multi-scene `.czi`/`.nd2`/`.lif` files and tiled plate acquisitions must be split upstream into one file per position, with the position readable from the name (e.g. `A01_s03.tif`). The FOV must be understandable from the image name.
- **FOV identity is `(condition, repeat, image_file)`.** `image_file` only needs to be unique within one `condition × repeat`. Names such as `img_001.tif` may recur in every repeat folder; MorphoCoverage keeps them distinct even when the analysis pools repeats.
- **`image_file` holds the file name (basename), not a path.** A full path makes the same FOV look different across machines and tools, and breaks joins between tables. If a path is needed, store it in a separate column such as `image_path`.

### Column names

- `condition` and `repeat` are lowercase, exact, string-valued, and present in every table that leaves a tool, including per-image tables. A condition without repeats uses `repeat="single"` (CellPatcher's `SINGLE_REPEAT_LABEL`).
- Other spellings (`Repeat`, `Condition`) are not accepted by MorphoCoverage's defaults. They raise, so they cannot silently bypass repeat separation.
- `label_id` identifies an object within an image, as both CellPatcher and fiberfeatures already use.

### A per-image table (the FOV table)

An object table cannot represent an image in which nothing was detected: that image simply has no rows. Every tool that iterates over images should therefore also write one row per image it was given:

| Column | Required | Notes |
| --- | --- | --- |
| `condition`, `repeat`, `image_file` | yes | FOV identity |
| `status` | yes | `processed`, or `failed`/`skipped` with a `reason` |
| `n_objects` | yes | Objects retained after the tool's own filters; may be 0 |
| `n_objects_detected` | recommended | Before size/border filters, so filtering is visible |
| `acquisition_order` | if known | Integer, no ties within `condition × repeat` |
| `stage_x`, `stage_y` | if known | Physical units stated in run metadata |
| `pixel_size`, `height`, `width` | if known | Allows imaged area to be computed |

**A failed image is not a zero-object FOV.** A segmentation failure, missing mask or unreadable file must be marked as such, not reported as an image with `n_objects = 0`. Otherwise MorphoCoverage would count an acquisition failure as evidence that part of the sample contains no cells.

### Features

Feature columns should be listed explicitly in the run's provenance (for example in the run-config JSON), so downstream tools can pass `feature_cols` without inferring them from numeric columns. Any dataset-level normalization or projection (z-scoring across all images, a PCA fitted on the whole dataset) must be recorded. It makes sequential benchmarking leak future FOVs.

## 2. Changes needed per repository

### CellPatcher

Already compatible:

- `image_file` is the basename (`orchestration.py:315-318`).
- `condition`/`repeat` are real fields, and `"single"` is used for conditions without repeats (`io.py:11`).
- `label_id` exists.

Needed:

1. **Emit a per-image table** (e.g. `images.csv` next to `patches.csv`). It should include images that produced zero patches and give the reason (no labels, all objects filtered by border or size). Today `patches.csv` has one row per accepted patch only, so these images disappear.
2. **Do not present `image_id` as an FOV identifier.** It is positional within a run (`image_{idx:04d}`, `pipeline.py:331,345,369,397`), so it changes if the input list changes. Document that `(condition, repeat, image_file)` is the FOV key.
3. **Keep plane metadata as columns.** Multi-plane images yield several planes per `image_file` (`preprocess.py:343-387`), and `t/z/c` are already stamped on patches. The rule "one file = one FOV" should be documented. For time-lapse, the same cell can appear in many frames; downstream tools need `t` (and ideally a track id) to avoid counting it as independent objects.
4. **Warn or refuse when `condition`/`repeat` are unset.** Direct `run_patching()` calls leave them `None` (`types.py:26-33`). MorphoCoverage rejects missing group values, so this surfaces late rather than at the source.

### fiberfeatures

Needed:

1. **Store the basename in `image_file`.** `build_metadata_from_condition_folders` writes `str(path)` (`batch.py:253`). Move the path to `image_path`. This change alters an existing shared column, so it needs a deliberate, versioned change, as fiberfeatures' own AGENTS.md requires.
2. **Emit only `repeat`.** `_COMMON_METADATA_COLUMNS` contains both `Repeat` and `repeat` (`features.py:17-32`); fiberfeatures' AGENTS.md already lists this as a known inconsistency.
3. **Populate `repeat`.** Folder metadata builds `condition` but no `repeat` (`batch.py:236-258`). Use `<condition>/<repeat>/` folders as CellPatcher does, or `"single"`.
4. **Emit a per-image table.** Cell rows exist only for regions above `min_region_size` (`extract_cell_features`, `features.py:298-345`), and metadata is broadcast onto the per-image frame (`batch.py:410-416`). An image with no qualifying cells therefore contributes no rows, and its acquisition is lost.
5. **Clarify `sample_id`.** It defaults to the filename stem (`batch.py:246-250`) and is described as "one image". Either rename it to reflect that it is an image/FOV identifier, or reserve `sample_id` for a biological sample. As it stands, "sample" means an FOV.
6. **Export the explicit feature list.** Features are chosen as numeric columns minus an exclusion set (`get_numeric_feature_columns`, `features.py:78-90`). Writing the resulting list into run provenance lets MorphoCoverage use it explicitly, without re-implementing that inference.

Already consistent: FOV-level randomization defaults to `fov_col="image_file"` (`optimize.py:303`), which matches MorphoCoverage.

### NucleiSky

NucleiSky does not produce object feature tables, so it needs no change for MorphoCoverage's FOV handling. Its registration outputs (per-FOV transforms) are a natural source of `stage_x`/`stage_y` for the FOV table if they are exported keyed by `image_file`.

### Ecosystem-level

- **Keep the shared column contract in one place.** CellPatcher's and fiberfeatures' AGENTS.md copies already disagree: `repeat` is "Always" in one and "If used" in the other. A single spec (e.g. in the organization `.github` or `docs-hub` repository), copied by reference into each AGENTS.md, would stop further drift.
- **Add a contract test to each producer:** a fixture with a zero-object image and duplicate basenames across two repeats, asserting the per-image table has every image and that `(condition, repeat, image_file)` is unique.

## 3. What MorphoCoverage does today with non-conforming input

| Upstream situation | MorphoCoverage behaviour |
| --- | --- |
| `condition`/`repeat` present in one linked table only | raises |
| `Repeat`/`Condition` spelling | raises unless named explicitly in `group_cols` |
| Same `image_file` in different repeats, pooled | kept as distinct FOVs |
| Several objects/planes from one `image_file` | one FOV |
| `image_file` in objects but not in FOV table | raises (no fallback to `fov_id`) |
| Objects whose FOV is missing from the FOV table | raises |
| No FOV table supplied | FOVs derived from objects; zero-object FOVs cannot be represented |
| Full path in one table, basename in the other | raises as objects absent from the FOV table; cannot be auto-repaired |
| Failed image reported as zero objects | **not detectable**; must be fixed upstream |
