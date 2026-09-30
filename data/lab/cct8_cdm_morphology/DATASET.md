# Shape of MDA-MB-231 cells on cell-derived matrices after CCT8 silencing (CCT8 paper, Fig. 5D-E)

`cct8_cdm_morphology` · role: development · licence: CC-BY-4.0 · paper: [10.1242/jcs.265278](https://doi.org/10.1242/jcs.265278) · data: [10.5281/zenodo.20704995](https://doi.org/10.5281/zenodo.20704995)

_Generated from `card.toml` by `python data/lab/catalog.py render cct8_cdm_morphology`; edit the card, not this file._

## 1. The experiment

MDA-MB-231 triple-negative breast cancer cells on cell-derived matrices (CDMs) made by telomerase-immortalised fibroblasts; siCTRL and two independent siRNAs against CCT8 (siCCT8#5, siCCT8#8). Figure: Fig. 5D (representative images) and Fig. 5E (area and aspect ratio). One row is one cell; one field is one cell per field (owner); position_id is synthetic ('<siRNA>|<repeat>|cellNNN', numbered by Fiji's row number); image_file is 'unrecorded'. Readout: `Area` (area of the cell outline (um^2)).

## 2. Design at a glance

- Experiment: not applicable (one experiment)
- Condition: the siRNA (source column 'Condition')
- Biological repeat: '1'-'3' (source column 'Repeat')
- Technical repeat: single: each siRNA was imaged in one well per biological repeat (owner's statement, 2026-09-30); rules out any test that needs technical repeats within a biological repeat
- Field: one cell per field (owner); position_id is synthetic ('<siRNA>|<repeat>|cellNNN', numbered by Fiji's row number); image_file is 'unrecorded'
- Field table: one field per cell (owner's statement); empty fields: not applicable (fields were images of chosen cells)
- Field selection: not stated (cells were segmented manually; one cell per image by the owner's account)
- Stage coordinates: not recorded; acquisition order: not recorded (Fiji's row number is kept as source_roi, not as an acquisition order)
- Counts: 174 cells, 3 conditions, 3 biological repeats; 53-63 per condition (siCTRL 53, siCCT8#5 63, siCCT8#8 58)

Cells per condition and biological repeat:

```text
repeat     repeat 1  repeat 2  repeat 3
condition                              
siCCT8#5         25        21        17
siCCT8#8         27        16        15
siCTRL           18        24        11
```

## 3. What the paper concluded from these data

- **CCT8 silencing made MDA-MB-231 cells on CDM smaller** (Fig. 5E (area)).
  > "CCT8 knockdown led to smaller, more rounded cells (Fig. 5D and 5E), consistent with impaired protrusive activity." — Results, owner's manuscript copy
  n: n > 53 cells per condition, three biological repeats; test: randomization test (Cohen's d, 10,000 iterations), Bonferroni-corrected.
  Reproduced from the stored tables: **partly** — mean area 419 (siCCT8#5) and 404 (siCCT8#8) against 533 um^2 (siCTRL); Cohen's d 0.4362 and 0.5682, identical to the deposit's Stats table; Bonferroni-corrected randomization p 0.061 (siCCT8#5) and 0.0084 (siCCT8#8) in that table (both siRNAs are smaller in each of the 3 repeats, but siCCT8#5 does not reach p < 0.05 after the correction (0.056 on the repeat-normalised area). To discuss with the owner; not a correction of the paper.)
- **CCT8 silencing made MDA-MB-231 cells on CDM more rounded (lower aspect ratio)** (Fig. 5E (aspect ratio)).
  > "CCT8 knockdown led to smaller, more rounded cells (Fig. 5D and 5E), consistent with impaired protrusive activity." — Results, owner's manuscript copy
  n: n > 53 cells per condition, three biological repeats; test: randomization test (Cohen's d, 10,000 iterations), Bonferroni-corrected.
  Reproduced from the stored tables: **match** — mean AR 1.70 (siCCT8#5) and 1.64 (siCCT8#8) against 4.25 (siCTRL); Cohen's d 1.5631 and 1.5430, identical to the deposit's Stats table; Bonferroni-corrected randomization p 0.0003 for both (lower in each of the 3 repeats for both siRNAs; circularity (measured, not shown) rises too, d 0.99 and 0.77, but not in every repeat (siCCT8#8, repeat 2: 0.32 against 0.37))
- **Cell counts per condition** (Fig. 5 legend (E)).
  > "Quantification of cell area and aspect ratio for MDA-MB-231 cells (n > 53 cells per condition, three biological repeats)." — Fig. 5 legend, owner's manuscript copy
  n: n > 53 cells per condition; test: not stated.
  Reproduced from the stored tables: **partly** — 53, 63 and 58 cells (siCTRL, siCCT8#5, siCCT8#8); 3 repeats; counts, means and SDs identical to the deposit's summary table (siCTRL has exactly 53 cells, so the legend's 'n > 53' reads as n >= 53)

How the paper analysed them: randomization test with Cohen's d, 10,000 iterations, Bonferroni-corrected over the 3 pairwise comparisons; cells as the unit — "Randomization tests used Cohen's d as the effect-size metric, with 10,000 iterations per test." (Methods, Statistical analysis; the correction from the deposit's Stats/ tables)

## 4. Curator's hypotheses for MorphoCoverage

_Written 2026-09-30, before any MorphoCoverage run. These are expectations to check, not answers._

- Per-cell precision rules should fail for Area in most condition x repeat groups: Area is right-skewed (skewness 1.9-2.6) with a coefficient of variation of 0.43-0.68, and the groups hold only 11-27 cells. This is the opposite of the filopodia screen, where 76-113 cells per condition were collected.
- AR in siCTRL is far more variable (SD 2.28, mean 4.25) than in siCCT8 (SD 0.71-0.85): per-condition sufficiency should differ between control and knockdown, with siCTRL needing the most cells.
- With fields = cells, the field-level design effect should be about 1.
- The condition contrast within a repeat cannot be tested (one well per siRNA and repeat, one unit per side). The effect sizes should instead point the same way in every repeat: Area and AR lower in 3/3 repeats for both siRNAs.
- The AR effect (d about 1.55) needs far fewer cells to show than the Area effect (d 0.44-0.57); any effort estimate for the contrast should rank Area, and siCCT8#5 in particular, as under-sampled.
- Repeat 3 is the smallest (11 siCTRL cells, 15-17 per siRNA) and should be singled out as the weakest group.

## 5. Traps for an analyst

- One cell per field is the owner's statement; the images are not identified, so field identities are synthetic. Fiji's Slice column (source_slice) is the z-plane each outline was drawn on (owner's statement, 2026-09-30), not an image index; it is not a z coordinate of the tables, since the outline is one plane of a 3D cell.
- One well per siRNA and repeat (technical repeat 'single'): no test within a biological repeat is possible; cells are not independent replicates.
- Cells were segmented manually and the cell-selection rule is not stated: a chosen, not random, sample of cells.
- The intensity columns (Mean ... RawIntDen) were measured on processed RGB projections and are not features here; positions, orientations and bounding boxes are not features either.
- The published plots clip the axes (area to 1000 um^2, AR to 10); the statistics use every cell.
- The figure's statistics treat cells as independent; with 3 repeats the repeat-level evidence is 3 paired means per siRNA.

## 6. MorphoCoverage results

Runs are stored in `data/lab/results/cct8_cdm_morphology/` and logged in `data/lab/FINDINGS.md`.

## Provenance

- Source: Zenodo record 20704995 (10.5281/zenodo.20704995), CC-BY-4.0, retrieved 2026-09-30; file `Fig5 and FigS5/Fig5DE 231 CDM - shape/Cell Morphology 231s CDMs.xlsx` (sha256 `944227bc9eda234b2b7eeca92fbced55258f8c8fa4df006e85b12617cc6c530e`).
- Paper: CCT8 associates with the MYO10 motor domain and regulates filopodia and breast cancer cell invasion. Journal of Cell Science (2026) — quotes from revised manuscript for JCS provided by the owner (clean copy), 2026-09-30.
- Acquisition: Zeiss LSM880 confocal with Airyscan; 63x/1.4 NA objective, 0.1318 um pixels, 1024 x 1024 px (about 135 um) fields, 35-slice z-stacks, 3 channels (from the metadata of the deposited ctrl3-Processed.czi) — "Samples were imaged on a confocal microscope (LSM880; Zeiss). Cells were manually segmented, and area, aspect ratio, and circularity were measured in Fiji" (Methods, Cell migration on cell-derived matrices (owner's manuscript copy))
- Curated by Claude (on behalf of Guillaume Jacquemet) on 2026-09-30; verified by (not yet verified).
