# Glossary

Every test, rule and term the report uses, in plain words, with where it appears and the papers it comes from. Generated from `morphocoverage.glossary` (the HTML report prints the same entries); edit them there.

## Reading the report

**Status markers.** ✅ answered; ⚠️ answered, with a caution about these data; ❌ check the data first; ➖ not applicable or not assessed. A marker says how far to trust an answer, not what it found: ✅ can mean 'no difference'.

*Where:* At a glance, every answer.

**Assumptions.** What an answer rests on that these data cannot check (cells as independent draws, fields as a random sample of the well). Stated once, after the answer, in grey.

*Where:* Every answer.

## The design

**Field (FOV).** One imaged position of the stage: the unit a microscopist acquires, and the unit most answers count, because cells of one field are not independent of each other.

*Where:* Everywhere.

**Biological repeat.** An independent experiment (often its date). The replicate that a conclusion about the conditions must hold across.

*Where:* Question 9, the design scan.

**Technical repeat.** A well, dish or coverslip within a biological repeat; never a biological repeat. 'single' when there is one.

*Where:* Question 9, the plots (colours), question 7.

**Pilot.** A dataset of one or two biological repeats. It gets every answer its design supports and advice for the next experiment, not a verdict across repeats.

*Where:* Question 9, the plan.

**Normalised values.** Each value divided by the mean of the control of its own biological repeat, so a scale shared by a repeat (a staining or imaging batch) is removed. Needs a positive control mean.

*Where:* Question 9, the plots, the PCA and UMAP.

## Did I image enough?

**Mean-precision rule.** Stops once the mean of the readout is precise enough: its averaging CV (one standard error as a share of the mean) at or below the target, e.g. 5%. 'Per cell' treats cells as independent; 'per field' uses field means. A CV of 5% is one standard error of 5%, not a ±5% bound.

*Where:* Question 2, the stopping rules.
*See:* [Zweifach A. Determining how many cells to average for statistical testing of microscopy experiments. J Cell Biol. 2024.](https://doi.org/10.1083/jcb.202401074)

**Mean-precision count check.** Whether the number of cells the mean-precision rule asks for still gives that precision when the cells come as whole fields, as they are acquired.

*Where:* Foundation.
*See:* [Zweifach A. Determining how many cells to average for statistical testing of microscopy experiments. J Cell Biol. 2024.](https://doi.org/10.1083/jcb.202401074)

**JCS spread and plateau rules.** Stop once the mean, median, SD and IQR of a feature stabilise as cells are added (within a tolerance you choose, in the feature's units).

*Where:* Question 2, the stopping rules.
*See:* [Marcotti et al. J Cell Sci. 2026.](https://doi.org/10.1242/jcs.264367)

**Held-back fields.** Fields set aside before a rule is replayed, so whether the rule stopped too early is judged on fields it never saw.

*Where:* Questions 4, 5 and the rules' reliability (optimise).

**Resolved.** The 95% interval of the difference from the control, from the fields resampled as they were acquired, excludes zero.

*Where:* Question 2.

## Do the conditions differ?

**Cohen's d.** The difference of two means in units of their pooled standard deviation: 0.2 small, 0.5 medium, 0.8 large by Cohen's convention. Sign: negative is lower than the control.

*Where:* Question 9, the feature ranking.
*See:* Cohen J. Statistical Power Analysis for the Behavioral Sciences. 2nd ed. 1988.

**Cell-level randomization test.** Shuffles the condition labels over every cell of a condition and its control, as the lab's Plot&Stats notebook does. It asks whether these wells differ, not whether the effect repeats: when the difference varies between repeats it gives p ≤ 0.05 more often than 5% without an effect, and the report prints that chance beside it.

*Where:* Question 9, the discovery gate.
*See:* Plot&Stats, the Cell Migration Lab's analysis notebook.

**Balanced test.** The cell-level test when the repeats hold different numbers of cells: each draw takes the same number from every repeat, so none weighs more; the median d and p over 100 draws.

*Where:* Question 9.

**Sparse repeat.** A repeat with under a quarter of the median repeat's cells, or fewer than 5: left out of the balanced test only, and named so it can be checked.

*Where:* Question 9, the design scan.

**Paired t-test on the repeat means.** Each biological repeat's condition mean paired with its own control mean, the repeat as the replicate (as in SuperPlots). Keeps false positives at 5%, but with 3 repeats it confirms only large, consistent effects. Given raw and on log ratios to the control (normalised).

*Where:* Question 9: the only test that confirms a difference.
*See:* [Lord SJ, Velle KB, Mullins RD, Fritz-Laylin LK. SuperPlots: communicating reproducibility and variability in cell biology. J Cell Biol. 2020;219(6):e202001064.](https://doi.org/10.1083/jcb.202001064)

**Sign-flip test.** The exact test on the signs of the per-repeat differences; given only from 6 repeats, where its smallest p can reach 0.05 after the correction.

*Where:* Question 9, full detail.

**Bonferroni correction.** Each p-value multiplied by the number of conditions tested, so testing several does not raise the chance of one false positive.

*Where:* Question 9.

**Benjamini-Hochberg q-value.** A p-value corrected for the many features tested together: among the features with q ≤ 0.05, about 5% are expected to be false discoveries.

*Where:* Question 9, the feature ranking.
*See:* [Benjamini Y, Hochberg Y. Controlling the false discovery rate: a practical and powerful approach to multiple testing. J R Stat Soc B. 1995;57(1):289-300.](https://doi.org/10.1111/j.2517-6161.1995.tb02031.x)

**Discovery gate and confirmed across repeats.** A feature passes the discovery gate when its cell-level test has q ≤ 0.05; it is confirmed across repeats when the paired test on the repeat means passes too. The first finds candidates; the second says the effect repeats.

*Where:* Question 9, the feature ranking.

**Reagents of one target.** Two siRNAs (or other reagents) of one gene, named in the 'target' column: when both go the same way in every repeat, an off-target effect of one of them is less likely. Described, not tested.

*Where:* Question 9.
*See:* [CCT8 associates with the MYO10 motor domain and regulates filopodia and breast cancer cell invasion. J Cell Sci. 2026 (the lab datasets).](https://doi.org/10.1242/jcs.265278)

**JCS effect size.** The treated mean minus the control mean over the control SD, from cells drawn as random cells and as whole fields.

*Where:* Question 9, full detail.
*See:* [Marcotti et al. J Cell Sci. 2026.](https://doi.org/10.1242/jcs.264367)

## Figures

**PlotsOfData-style plot.** Every cell a point coloured by technical repeat, the biological repeats side by side, each repeat's median with its 95% interval from resampling whole technical repeats and their fields as acquired (never single cells).

*Where:* Question 9's figures.
*See:* [Postma M, Goedhart J. PlotsOfData: a web app for visualizing data together with their summaries. PLoS Biol. 2019;17(3):e3000202.](https://doi.org/10.1371/journal.pbio.3000202); [Lord SJ, Velle KB, Mullins RD, Fritz-Laylin LK. SuperPlots: communicating reproducibility and variability in cell biology. J Cell Biol. 2020;219(6):e202001064.](https://doi.org/10.1083/jcb.202001064); Ren et al. Nonparametric bootstrapping for hierarchical data. J Appl Stat. 2010.

**PCA and loadings.** The two directions along which the standardised features vary most, each labelled with its share of the variance; an arrow shows how much a best-ranked feature weighs on each direction. Describes, does not test.

*Where:* Question 9's figures.

**UMAP.** A map that keeps each cell's nearest neighbours close, so it can show groups that PCA mixes. Its axes have no units and distances between groups are not to scale. Describes, does not test; on raw values a separation can be a batch between repeats.

*Where:* Question 9's figures.
*See:* [McInnes L, Healy J, Melville J. UMAP: Uniform Manifold Approximation and Projection for dimension reduction. arXiv:1802.03426. 2018.](https://arxiv.org/abs/1802.03426)

## The data

**Integrity checks.** Exact checks for accidents of image analysis: a field exported twice, rows pasted into another field, a change of units, a summary written twice. A finding is structure to inspect, not a judgement.

*Where:* The data, question 6, the design scan.

**Unusual cells and 'nothing new'.** Cells far from the control's, and whether later fields still showed states earlier fields did not (the definition of new is declared in the setup).

*Where:* Questions 2 and 3.

**Positive control (representation).** A condition known to differ, used to check that the features can see the phenotype at all before 'nothing new' is trusted.

*Where:* Question 1.

**Between-field share.** The part of a feature's variance that lies between fields rather than between cells of one field: when it is large, more fields help more than more cells per field.

*Where:* Foundation, question 3.
