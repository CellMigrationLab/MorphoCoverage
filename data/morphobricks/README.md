# MorphoBricks-shaped tables

Small tables as the other MorphoBricks tools write them, for MorphoCoverage's tests
(`tests/test_morphobricks_tables.py`).

- `fiberfeatures_patches.csv`: 96 grid patches of 6 images (3 per condition) from
  fiberfeatures' reference patch table, `data/experiment_001/reference/patches.csv` at
  fiberfeatures `87d18d0` (2026-10-01, MIT licence), with every metadata and patch column in
  the order fiberfeatures writes them and 5 of its about 110 features (`nematic_order`,
  `linearity_mean`, `entropy_mean`, `bundle_evidence_mean`,
  `pixel_category_network_fraction`), rounded to 6 decimals. One row is one patch; an image is
  one field (`image_file`); the folders had no repeat levels, so `biological_repeat` and
  `technical_repeat` are `single`. The `patch_*` columns say where each patch was cut, not what
  it shows.
