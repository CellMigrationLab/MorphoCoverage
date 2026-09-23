# Crick Enhancing-Reproducibility feature tables (idr0028, idr0139)

Real microscopy feature tables used for **testing and exploration** in
MorphoCoverage. They are not a MorphoCoverage benchmark dataset and are not
intended for publication.

## Source and attribution

These files are derived from the CellProfiler feature tables and IDR
annotation files published in the `inputs/` folder of

> FrancisCrickInstitute/Enhancing-Reproducibility, companion code for
> Marcotti S, Gerontogianni L, Kelly G, Barry DJ. *Practical statistics for
> bioimage analysis – a guide to experimental design and data interpretation.*
> J Cell Sci 2026;139(10):jcs264367. https://doi.org/10.1242/jcs.264367
> https://github.com/FrancisCrickInstitute/Enhancing-Reproducibility

which is licensed under the Creative Commons Attribution 4.0 International
licence (CC BY 4.0, https://creativecommons.org/licenses/by/4.0/).

The underlying images come from the Image Data Resource:

- **idr0139**: Lawson et al., "Nuclear fascin regulates cancer cell survival",
  IDR study idr0139, licence CC BY 4.0, copyright Lawson et al.
- **idr0028** (screen B): Pascual-Vargas et al., RNAi screen for regulators of
  cell shape and YAP/TAZ localisation, IDR study idr0028. We did not reach
  the study's own licence record; the derived tables are redistributed here
  under the CC BY 4.0 licence of the Crick repository.

No images are included.

## Changes made (CC BY 4.0 notice)

Produced by `convert.py` from the Crick files:

- **Wells:** only the wells analysed in the paper are kept, as selected by the
  Crick scripts `subset_idr0139_data.py` and `subset_cp_data.py`. These are the
  same wells as the `data_subsets.zip` used by their notebooks.
  - idr0139: plate 1093711385, wells B02, E22, G15, H13, I19, J05, L08, L18,
    N12 and O02, with 4 fields each.
  - idr0028 screen B: plate 1A wells B6, I23, J23, M23 and N23; plate 2A wells
    C13, J23, K16 and N23; plate 2B wells C13, C17 and P13. There are 28
    fields each.
- **Layout:** the tables are reshaped into MorphoBricks-style linked tables.
  Each dataset has a FOV table (`*_fovs.csv.gz`, one row per acquired field)
  and an object table (`*_nuclei.csv.gz`, one row per nucleus). Each nucleus
  is joined one-to-one with its cytoplasm. All CellProfiler intensity
  measurements are kept, prefixed `Nuclei_` or `Cytoplasm_`.
- **Precision:** values are rounded to 6 significant digits.

## Columns

Common to both tables:

- `condition`: the paper's labels. idr0139 uses Untreated, DMSO, Leptomycin b
  or the compound ID. idr0028 uses the gene symbol, or MOCK for
  negative-control wells.
- `repeat`: the plate (`1A`, `2A`, `2B` or `1093711385`). It is not asserted to
  be a biological repeat.
- `well`, `image_file`: `(condition, repeat, image_file)` identifies a field.
  The same well name, for example J23, occurs on several plates.

FOV table only:

- `field`.
- `acquisition_order`.
  - idr0028: the Columbus acquisition timestamp (`acquisition_time_s`), with
    ties broken by field number. In some wells a few fields carry timestamps
    about an hour earlier than the rest, probably re-imaging. They are kept as
    recorded.
  - idr0139: the source has no timestamps, so the field number is the only
    available proxy.
- idr0028 only: `stage_x_um`, `stage_y_um`, `area_um2` (a 7 × 4 grid of
    adjacent fields).
- `control_type`, `quality_control`: from the IDR annotation.

Object table only:

- `label_id`: the CellProfiler object number.
- `nucleus_x_px`, `nucleus_y_px`: metadata, not features.
- Features: `Nuclei_Intensity_*` and `Cytoplasm_Intensity_*`. The papers'
  metrics are derived from these:
  - `Fascin_Ratio` = `Nuclei_Intensity_MeanIntensity_Fascin` /
    (`Nuclei_…Fascin` + `Cytoplasm_Intensity_MeanIntensity_Fascin`);
  - `YAPTAZ_Ratio` is built the same way from the YAPTAZ columns.

`tests/test_crick_idr.py` checks that the idr0139 tables reproduce the
paper's Table 1: mean, median, quartiles, SD and number of cells.

## Scripts

- `convert.py RAW_DIR OUT_DIR` downloads the source files (about 2 GB, via
  GitHub's LFS media host) and regenerates these tables.
- `fov_vs_cells.py RAW_DIR` runs the retrospective whole-FOV versus random-cell
  comparison on the complete plates, as summarized in
  `docs/preliminary_findings.md`.
