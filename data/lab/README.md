# Lab datasets

Published datasets of the lab, stored as MorphoBricks tables so MorphoCoverage can be tested on real
data with documented designs and published conclusions. `catalog.load(id)` returns the tables and the
card; `DATASET.md` in each folder is the description to read first.

| dataset | paper, figure | unit | design | role | licence | tables |
|---|---|---|---|---|---|---|
| [`cct7_cct8_glioma_cdm_tracks`](cct7_cct8_glioma_cdm_tracks/DATASET.md) | unpublished (project of [10.1242/jcs.265278](https://doi.org/10.1242/jcs.265278)) | track | 590 tracks in 133 fields, 5 conditions, 6 biological repeats | development | CC-BY-4.0 | 121 kB |
| [`cct8_cdm_morphology`](cct8_cdm_morphology/DATASET.md) | [10.1242/jcs.265278](https://doi.org/10.1242/jcs.265278), Fig. 5D-E | cell | 174 cells, 3 conditions, 3 biological repeats | development | CC-BY-4.0 | 19 kB |
| [`cct8_filopodia_screen`](cct8_filopodia_screen/DATASET.md) | [10.1242/jcs.265278](https://doi.org/10.1242/jcs.265278), Fig. 2C | cell | 1686 cells, 13 conditions, 18 biological repeats | development | CC-BY-4.0 | 17 kB |

## Rules (owner decisions, 2026-09-30)

- **Folder per dataset:** `card.toml` (facts, each sentence from the paper quoted with its source),
  `DATASET.md` (generated from the card: `python data/lab/catalog.py render <id>`), `convert.py` (rebuilds
  the tables from the pinned source, checksums verified, deterministic output), `objects.csv.gz` and
  `fovs.csv.gz`, `NOTICE` (licence and attribution).
- **Columns:** the MorphoBricks names (`condition`, `biological_repeat`, `technical_repeat`, `image_file`,
  `position_id`, `z`/`z0`/`z1`, `c`, `t`); metadata columns are always renamed, feature columns keep their
  source names. A level with one member is `single` only when the source or the owner says so; a level that exists but
  is not recorded is `unrecorded`, and the card says which analyses that rules out.
- **Sources:** CC BY 4.0 or CC0 only (the lab's own unpublished data too, once the owner licenses it); no patient-derived material; raw images are never stored. Data budget:
  150 MB for the whole repository (git keeps every committed file).
- **Reproduction:** the paper's numbers for the named figure are recomputed from the stored tables with
  tolerances fixed beforehand (counts exact, means to the stated decimals, p-values to their order of
  magnitude); every mismatch is reported to the owner and recorded, never silently fixed.
- **Role:** lab datasets are `development` (settings may be tuned on them); the `data/crick_idr/` idr0139 data stay the
  held-out dataset. Each card's hypotheses are written before any MorphoCoverage run; runs go to
  `results/<id>/` and `FINDINGS.md`, never into `DATASET.md`.
- **Contract changes:** tables are written against the contract version in the card; `catalog.load`
  translates older versions.

`convert.py` needs network access and `openpyxl` for spreadsheet sources; the stored tables and the tests do not.
