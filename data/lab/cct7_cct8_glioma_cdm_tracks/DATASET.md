# Migration of U-87 MG glioma cells on cell-derived matrices after CCT7 or CCT8 silencing (unpublished)

`cct7_cct8_glioma_cdm_tracks` · role: development · licence: CC-BY-4.0 · unpublished (project: [10.1242/jcs.265278](https://doi.org/10.1242/jcs.265278)) · provided by the owner, 2026-09-30

_Generated from `card.toml` by `python data/lab/catalog.py render cct7_cct8_glioma_cdm_tracks`; edit the card, not this file._

## 1. The experiment

U-87 MG glioblastoma cells migrating on cell-derived matrices (CDMs); siCTRL and two siRNAs against each of CCT7 (siCCT7#3, siCCT7#6) and CCT8 (siCCT8#5, siCCT8#8). Figure: none (the owner's statement, 2026-09-30: from the same project, not in the paper). One row is one track; one field is the imaged position, '<well>_<position>' from the results file name (e.g. 'B5_04-1'), verbatim. Readout: `Mean Speed` (mean instantaneous speed along the track (px per 10 min)).

## 2. Design at a glance

- Experiment: the experimental set, CCT7 or CCT8 (from the siRNA label); each has its own siCTRL well (B4 for CCT7, B1 for CCT8)
- Condition: the siRNA; both controls written 'siCTRL', the source label kept as source_condition
- Biological repeat: '<set>_<Repeat>' (Repeat 1-3): the two sets' movies have different lengths in every repeat (tracks of about 950 min for CCT7, 1200-1440 min for CCT8), so they are treated as separate acquisitions (assumption; to confirm by the owner)
- Technical repeat: the well (B1-B6), one well per siRNA per repeat; rules out any test that needs several wells within a biological repeat
- Field: the imaged position, '<well>_<position>' from the results file name (e.g. 'B5_04-1'), verbatim
- Field table: the results files named in the source (one per position); positions with no track are not known
- Field selection: not stated (positions within each well; how they were chosen is not recorded)
- Stage coordinates: not recorded; acquisition order: not recorded
- Counts: 590 tracks, 5 conditions, 6 biological repeats; 75-119 per condition (siCTRL_CCT7 75, siCCT7#3 92, siCCT7#6 92, siCTRL_CCT8 109, siCCT8#8 103, siCCT8#5 119)

Tracks per condition and biological repeat:

```text
repeat                repeat 1  repeat 2  repeat 3
experiment condition                              
CCT7       siCCT7#3         28        34        30
           siCCT7#6         29        27        36
           siCTRL           30        23        22
CCT8       siCCT8#5         37        40        42
           siCCT8#8         32        40        31
           siCTRL           31        39        39
```

## 3. What the paper concluded from these data

- None: these data are in no paper (the owner's statement, 2026-09-30: from the same project, not in the paper).

How the paper analysed them: not applicable

## 4. Curator's hypotheses for MorphoCoverage

_Written 2026-09-30, before any MorphoCoverage run. These are expectations to check, not answers._

- The integrity checks should find the 28 tracks that appear twice (identical track number and metrics in two results files of the same siRNA and repeat: B5_04/B5_04-1, B4_05/B4_052, B6_01-1/B6_02-1, B6_04-1/B6_05-1, B2_01/B2_02, B1_03/B1_04). A clean report would be a miss.
- Tracks cluster by field (several cells per position, cells on the same patch of matrix): the field-level design effect for Mean Speed should be above 1, unlike the two one-cell-per-field datasets.
- The condition contrast within a repeat cannot be tested with one well per siRNA; MorphoCoverage should say so and report effect sizes and directions per repeat instead.
- Direction per repeat, against the set's own siCTRL: Mean Speed lower for both CCT8 siRNAs in 3/3 repeats; for CCT7 no consistent direction (both lower in repeat 1, both higher in repeat 2, split in repeat 3: siCCT7#3 0.63 and siCCT7#6 0.56 against 0.63).
- The two siCTRL wells differ (mean speed 0.56 for the CCT7 set, 0.44 for CCT8, with different movie lengths): pooling them as one 'siCTRL' condition, which MorphoCoverage does because it has no experiment level, mixes two baselines.

## 5. Traps for an analyst

- 28 tracks are in the source twice, in two results files of the same siRNA and repeat (see the hypotheses); they are kept as in the source, so counts and means include them. Deduplicate on (experiment, condition, biological_repeat, label_id, metrics) before any biological use.
- Track lengths differ within a repeat: in CCT7 repeat 2 the siCTRL tracks mostly last 1200 min and the siRNA tracks 950 min, so path length, spatial coverage, turning angle and directionality are not comparable there; across sets, movies differ in length in every repeat.
- Distances and speeds are in pixels (px per 10 min); the pixel size is not recorded, so values are comparable within this dataset only.
- MorphoCoverage has no experiment level: both sets' controls are condition 'siCTRL'; the biological_repeat labels ('CCT7_1', 'CCT8_1', ...) keep the sets apart.
- The acquisition method and the track filters are the related paper's; that they apply here is an assumption. Fields were imaged in parallel (multi-position time-lapse).
- Some position names look like re-exports or typos (B4_052, '-1' and '-2' suffixes); they are kept verbatim as field identifiers.

## 6. MorphoCoverage results

Runs are stored in `data/lab/results/cct7_cct8_glioma_cdm_tracks/` and logged in `data/lab/FINDINGS.md`.

## Provenance

- Source: provided by the owner (2026-09-30), not deposited; CC-BY-4.0 (the lab's own unpublished data; licence to be confirmed by the owner); file `Tracking_data_CCT8_CCT7_silencing_Glioma.csv` (sha256 `dec991b3fdd52956d01ee64c2348ee4ec64afe4f16b7d0f67c549350b064233e`).
- Paper: none; project: the CCT8-MYO10 filopodia project (CCT8 paper, J. Cell Sci. 2026) (10.1242/jcs.265278).
- Acquisition: CellIQ, 10x objective, brightfield, a frame every 10 min (the results file names say 'px per 10min') — "For cell migration analyses, U-87 MG or MDA-MB-231 cells were seeded onto CDMs for 4 hours prior to imaging. Brightfield images were taken every 10 minutes over 16 hours using a CellIQ system with a 10× objective at 37°C and 5% CO2." (Methods of the related paper (owner's manuscript copy); that these data follow it is an assumption)
- Curated by Claude (on behalf of Guillaume Jacquemet) on 2026-09-30; verified by (not yet verified).
