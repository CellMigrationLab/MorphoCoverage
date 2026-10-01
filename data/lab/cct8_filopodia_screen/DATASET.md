# Filopodia per cell after siRNA silencing of six MYO10-proximal candidates (CCT8 paper, Fig. 2C)

`cct8_filopodia_screen` · role: development · licence: CC-BY-4.0 · paper: [10.1242/jcs.265278](https://doi.org/10.1242/jcs.265278) · data: [10.5281/zenodo.20704995](https://doi.org/10.5281/zenodo.20704995)

_Generated from `card.toml` by `python data/lab/catalog.py render cct8_filopodia_screen`; edit the card, not this file._

## 1. The experiment

U-87 MG glioblastoma cells stably expressing MYO10-GFP; siCTRL (AllStars negative control) and two independent siRNAs for each of MAP4K4, MINK1, NF2, SCRIB, CSNK1A1 and CCT8. Figure: Fig. 2C (images: Fig. 2B; knockdown validation: Fig. S2B-G). One row is one cell; one field is one cell per field (owner); position_id is synthetic ('<set>|<siRNA>|<repeat>|cellNNN', numbered in spreadsheet order); image_file is 'unrecorded'. Readout: `Filopodia number` (MYO10-positive filopodia counted per cell (count)).

## 2. Design at a glance

- Experiment: the experimental set, one per targeted gene (source column 'dataset'); each has its own siCTRL
- Condition: the siRNA (source column 'Condition')
- Biological repeat: '<set>_<Repeat>': repeat 1-3 within each set; repeat 1 of one set is not repeat 1 of another (each set was run as its own experiment: owner's statement, 2026-09-30)
- Technical repeat: single: one coverslip per siRNA per biological repeat (owner's statement, 2026-09-30); rules out any test that needs technical repeats within a biological repeat
- Field: one cell per field (owner); position_id is synthetic ('<set>|<siRNA>|<repeat>|cellNNN', numbered in spreadsheet order); image_file is 'unrecorded'
- Field table: one field per cell (owner's statement); empty fields: not applicable (fields were images of chosen cells)
- Field selection: operator: the operator chose the cells to image, one per field (owner's statement, 2026-09-30); not stated in the paper
- Stage coordinates: not recorded; acquisition order: not recorded (the spreadsheet order is kept as source_row, not as an acquisition order)
- Counts: 1686 cells, 13 conditions, 18 biological repeats; 76-113 per condition (76-113 (siCTRL: 84-104 per set, 555 pooled))

Cells per condition and biological repeat:

```text
repeat                   repeat 1  repeat 2  repeat 3
experiment condition                                 
siCCT8     siCCT8#5            34        49        30
           siCCT8#8            31        38        31
           siCTRL              33        39        32
siCSNK1A1  siCSNK1A1#11        34        34        34
           siCSNK1A1#5         35        34        34
           siCTRL              31        33        33
siMAP4K4   siCTRL              30        32        31
           siMAP4K4#5          31        13        32
           siMAP4K4#6          37        29        32
siMINK1    siCTRL              34        32        18
           siMINK1#10          33        29        27
           siMINK1#3           38        27        20
siNF2      siCTRL              31        29        31
           siNF2#17            32        31        30
           siNF2#7             30        31        31
siSCRIB    siCTRL              30        26        30
           siSCRIB#6           30        31        28
           siSCRIB#7           33        29        29
```

## 3. What the paper concluded from these data

- **Silencing MINK1, CSNK1A1, SCRIB or CCT8 reduced the number of MYO10-positive filopodia per cell** (Fig. 2C).
  > "depletion of MINK1 (~80% efficiency), CSNK1A1 (~80% efficiency), SCRIB (~80–90% efficiency), or CCT8 (~75% efficiency) significantly reduced the number of MYO10-positive filopodia (Fig. 2B and 2C)." — Results, owner's manuscript copy
  n: n = 76-113 cells per condition, three biological repeats; the siCTRL condition pools data from 6 experimental sets; test: randomization test (Cohen's d, 10,000 iterations), Bonferroni-corrected, against pooled siCTRL.
  Reproduced from the stored tables: **match** — all 8 siRNAs: Cohen's d 0.51-1.24 against pooled siCTRL, identical to the deposit's Stats table to 4 decimals; randomization p < 1e-4 for each; means 30.6-45.5 against 56.9 (also lower than their own set's siCTRL in every one of the 3 repeats)
- **Silencing MAP4K4 or NF2 did not significantly alter filopodia number** (Fig. 2C).
  > "While silencing MAP4K4 (~85% efficiency) or NF2 (~87% efficiency) did not significantly alter filopodia number" — Results, owner's manuscript copy
  n: as above; test: as above.
  Reproduced from the stored tables: **partly** — siMAP4K4#5 d 0.10 (Bonferroni p 1.0), siNF2#7 d 0.23 and siNF2#17 d 0.19 (Bonferroni p 1.0): no significant change against pooled siCTRL; but siMAP4K4#6 d 0.50, mean 68.2 against 56.9, Bonferroni p 0.0078 (an increase), as in the deposit's Stats table (the claim holds if a hit needs both siRNAs; against each set's own siCTRL, both MAP4K4 siRNAs and both NF2 siRNAs are higher in all 3 repeats (MAP4K4 set siCTRL mean 42.0 against the pooled 56.9). To discuss with the owner; not a correction of the paper.)
- **Cell counts per condition and pooled control** (Fig. 2C (legend)).
  > "n = 76-113 cells per condition, three biological repeats); the siCTRL condition pools data from 6 experimental sets." — Fig. 2 legend, owner's manuscript copy
  n: 76-113 cells per condition; test: not stated.
  Reproduced from the stored tables: **match** — 76-113 cells per siRNA; 555 pooled siCTRL cells; 3 repeats in every set; counts, means and SDs identical to the deposit's summary table

How the paper analysed them: randomization test with Cohen's d, 10,000 iterations, Bonferroni-corrected; cells as the unit; siCTRL pooled over the six sets — "Randomization tests used Cohen's d as the effect-size metric, with 10,000 iterations per test." (Methods, Statistical analysis; the pooling from the Fig. 2 legend and the deposit's Stats/ tables)

## 4. Curator's hypotheses for MorphoCoverage

_Written 2026-09-30, before any MorphoCoverage run. These are expectations to check, not answers._

- Within each set and repeat, the per-cell rules should judge the counts precise with few cells: the per-object JCB rule at 5% should be met in every condition and repeat.
- With fields = cells, the field-level design effect should be about 1, so objects and fields should agree (no clustering by construction).
- The condition contrast (each siRNA against its own set's siCTRL, within each repeat) cannot be tested: one coverslip per siRNA and repeat (one unit per side). MorphoCoverage should say so, and report effect sizes and their direction per repeat instead.
- Direction per repeat: decreases for CCT8, CSNK1A1, MINK1 and SCRIB in 3/3 repeats; increases for MAP4K4 and NF2 in 3/3 repeats against their own set's control.
- Pooling siCTRL across sets (as the figure does) hides the set-to-set difference in control level (42-65 filopodia per cell); the per-set pairing should expose it.

## 5. Traps for an analyst

- One cell per field is the owner's statement; the images and their names are not in the deposit, so field identities are synthetic.
- One coverslip per siRNA and repeat (technical repeat 'single'): no test within a biological repeat is possible; cells are not independent replicates.
- Cells were chosen by the operator and counted manually (the paper states no selection rule): a chosen, not random, sample of cells.
- Counts are integers (7-178): quantized features can trip exact-copy and novelty checks by coincidence.
- The figure pools siCTRL across the six sets; these tables keep each set's control, so numbers against 'the control' depend on the choice.

## 6. MorphoCoverage results

Runs are stored in `data/lab/results/cct8_filopodia_screen/` and logged in `data/lab/FINDINGS.md`.

## Provenance

- Source: Zenodo record 20704995 (10.5281/zenodo.20704995), CC-BY-4.0, retrieved 2026-09-30; file `Fig2 and FigS2/Fig2C filopodia numbers after silencing/FILOPODIA NUMBER ALL.xlsx` (sha256 `4efb5c9913a47ad1f40be801ab12932ffdd697db4db6893b75b30cebfdb17b56`).
- Paper: CCT8 associates with the MYO10 motor domain and regulates filopodia and breast cancer cell invasion. Journal of Cell Science (2026) — quotes from revised manuscript for JCS provided by the owner (clean copy), 2026-09-30.
- Acquisition: spinning-disk confocal (Marianas, Yokogawa CSU-W1, Zeiss Axio Observer Z1) — "To quantify filopodia, images were captured on a spinning-disk confocal microscope with a 100× objective, and filopodia per cell were manually counted." (Methods (owner's manuscript copy))
- Curated by Claude (on behalf of Guillaume Jacquemet) on 2026-09-30; verified by (not yet verified).
