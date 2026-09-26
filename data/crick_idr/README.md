# Crick Enhancing-Reproducibility feature tables (idr0028, idr0139)

Real microscopy feature tables used for **testing, exploration and the
benchmark** in MorphoCoverage, stored ready to use. They are not intended for
publication.

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
  cell shape and YAP/TAZ localisation, IDR study idr0028. The IDR study page
  (https://idr.openmicroscopy.org/study/idr0028/) is reported to list CC BY 4.0;
  that page was not reachable from the environment that prepared these files,
  so this was not confirmed directly. The Crick repository redistributes the
  derived tables under CC BY 4.0 as well.

Source pinned at Crick commit `a81d20c9393485e57395cd4a1be9e7ff83c7f11d`.
`convert.py` downloads from that commit and verifies every file against its
SHA-256 checksum, which equals its Git LFS object id.

No images are included.

## Ready to use

Four datasets are stored here as MorphoBricks-style linked tables: a FOV
table (`*_fovs.csv.gz`, one row per acquired field, fields without nuclei
included) and an object table (`*_nuclei.csv.gz`, one row per nucleus, joined
one-to-one with its cytoplasm).

| name | wells | fields per well | nuclei | features | readout |
|---|---|---|---|---|---|
| `idr0028_screenB` | the paper's 12 wells on plates 1A, 2A, 2B | 28 | 9,550 | 30 intensity features | `YAPTAZ_Ratio` |
| `idr0139` | the paper's 10 wells on plate 1093711385 | 4 | 10,748 | 60 intensity features | `Fascin_Ratio` |
| `idr0028_screenB_plates` | every well of the 3 plates (1,144) | 28 | 1,400,399 | readout only | `YAPTAZ_Ratio` |
| `idr0139_plate` | every well of the plate (308) | 4 | 392,377 | readout only | `Fascin_Ratio` |

In Python, from a script in this directory or with it on `sys.path`:

```python
from datasets import load

dataset = load("idr0139")  # or any name above
objects, fovs = dataset.tables
dataset.features  # the explicit feature columns ("Nuclei_*", "Cytoplasm_*")
dataset.readout  # "Fascin_Ratio", the paper's metric
dataset.control  # "Untreated": the negative-control condition
dataset.groups  # ("condition", "repeat", "technical_repeat"): one analysis group per well
dataset.role  # "held_out"; idr0028 is "development"
```

`load` renames the stored `well` column to `technical_repeat`, the
MorphoBricks column for a technical repeat within a biological repeat (the
plate here), and attaches `condition` and `technical_repeat` to the nuclei of
the complete plates, whose object tables keep only `repeat`, `image_file`,
`label_id` and the readout to stay small. `datasets.registry()` declares the benchmark roles
(`idr0028_screenB` development, `idr0139` held out).

## Changes made (CC BY 4.0 notice)

Produced by `convert.py` from the Crick files:

- **Wells.** `idr0028_screenB` and `idr0139` keep only the wells analysed in
  the paper, as selected by the Crick scripts `subset_idr0139_data.py` and
  `subset_cp_data.py`. These are the same wells as the `data_subsets.zip` used
  by their notebooks:
  - idr0139: plate 1093711385, wells B02, E22, G15, H13, I19, J05, L08, L18,
    N12 and O02;
  - idr0028 screen B: plate 1A wells B6, I23, J23, M23 and N23; plate 2A wells
    C13, J23, K16 and N23; plate 2B wells C13, C17 and P13.

  `idr0028_screenB_plates` and `idr0139_plate` keep every well of those plates.
- **Layout.** The CellProfiler tables are reshaped into the linked tables
  above. The paper-well tables keep every CellProfiler intensity measurement,
  prefixed `Nuclei_` or `Cytoplasm_`. The complete-plate tables keep only the
  readout.
- **Readout.** The papers' metric is stored as a column: `Fascin_Ratio` =
  `Nuclei_Intensity_MeanIntensity_Fascin` / (`Nuclei_…Fascin` +
  `Cytoplasm_Intensity_MeanIntensity_Fascin`); `YAPTAZ_Ratio` is built the
  same way from the YAPTAZ columns. It is derived from two features and is not
  one of the `features`.
- **Precision.** Values are rounded to 6 significant digits. The files are
  written without a gzip timestamp, so rerunning `convert.py` gives identical
  files.

## Columns

Common to both tables:

- `condition`: the paper's labels. idr0139 uses Untreated, DMSO, Leptomycin b,
  the compound ID, or `unannotated` for wells without an annotation. idr0028
  uses the gene symbol, or MOCK for negative-control wells. The complete
  plates also hold untargeted `technical control` and `empty well` wells,
  labelled by their control type, so MOCK always means negative control.
- `repeat`: the plate (`1A`, `2A`, `2B` or `1093711385`). It is not asserted to
  be a biological repeat.
- `well` (`technical_repeat` once loaded), `image_file`: `(condition, repeat,
  image_file)` identifies a field.
  The same well name, for example J23, occurs on several plates, and idr0028
  **reuses image file names across plates** (for example `003013-1.tif` in
  well C13 of plates 2A and 2B): always key fields by plate and file.

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
- `nucleus_x_px`, `nucleus_y_px` (paper-well tables): metadata, not features.
- The readout, and in the paper-well tables the features
  `Nuclei_Intensity_*` and `Cytoplasm_Intensity_*`.

`tests/test_crick_idr.py` checks that the idr0139 tables reproduce the
paper's Table 1 (mean, median, quartiles, SD and number of cells), that every
dataset validates, that the stored readout is the paper's ratio and that the
complete plates contain the paper's wells with the same values.

## Scripts

- `convert.py RAW_DIR OUT_DIR` downloads the source files (about 2 GB, via
  GitHub's LFS media host, checked against pinned SHA-256 checksums) and
  regenerates every table here.
- `datasets.py`: `load(name)`, the benchmark `registry()`, and the helpers that
  rebuild FOVs from blocks of nuclei for the semi-synthetic controls.

The analyses below run on the stored tables and write to `results/`; each is
described in `docs/preliminary_findings.md` (entry in brackets). Run them from
the repository root, for example `python data/crick_idr/fov_vs_objects.py`.

- `fov_vs_objects.py` (entry 1): whole FOVs against random objects on the
  complete plates, `results/fov_vs_objects_*.csv`.
- `positive_control.py` (entry 2): `results/positive_control_idr0028.csv`.
- `frequency_control.py` (entry 3): `results/frequency_control*_idr0028.csv`.
- `reliability_idr0028.py` (entry 4): `results/reliability_idr0028.csv` and
  `results/reliability_ceiling_idr0028.csv`.
- `stopping_idr0028.py` (entry 5): `results/stopping_calibration_idr0028.csv`
  and `results/adequacy_idr0028.csv`.
- `report_steps_idr0028.py` (entry 6; run `reliability_idr0028.py` first):
  `results/protocol_idr0028.csv`, `inspection_idr0028.csv`,
  `control_idr0028.csv`, `effort_idr0028.csv`, `effort_comparisons_idr0028.csv`,
  `order_efficiency_idr0028.csv`, `redundancy_idr0028.csv`,
  `sampling_profile_idr0028.csv` and the assembled
  `microscopist_report_idr0028.txt`.
- `heldout_idr0139.py` (entry 7): `results/heldout_*_idr0139.csv`.
- `benchmark.py` (entry 8) with its locked `benchmark_protocol.py`: the
  benchmark score, `results/benchmark_*`.
- `objects_vs_fields.py` (entries 9 and 10): the report's sampling-unit checks
  on the paper's wells and the complete plates, `results/objects_vs_fields_*`.
- `jcs_effect_fovs.py` (entry 9): the JCS treatment comparisons with whole
  fields, `results/jcs_effect_fovs*`.
- `run_all.py`: every analysis above, in order.
