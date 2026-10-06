# Glossary

Every test, rule and term the report uses, in plain words, with where it appears and the papers it comes from. Generated from `morphocoverage.glossary` (the HTML report prints the same entries); edit them there.

## Reading the report

**Status markers.** What each answer found: ✅ yes or a clear answer (a condition differs across repeats, enough was imaged, the data are clean); ⚠️ partly, a trend only, or something to look at; ❌ check the data first; ➖ no conclusion possible here (not assessed, one biological repeat, or no candidate group of cells passed the check). The word beside the marker says which.

*Where:* At a glance, every answer.

**Assumptions.** What an answer rests on that these data cannot check (cells as independent draws, fields as a random sample of the well). Stated once, after the answer, in grey.

*Where:* Every answer.

## The design

**Field (FOV).** One imaged position of the stage: the unit a microscopist acquires, and the unit most answers count, because cells of one field are not independent of each other.

*Where:* Everywhere.

**Biological repeat.** An independent experiment (often its date). The replicate that a conclusion about the conditions must hold across.

*Where:* Do the conditions differ?, the design scan.

**Sub-experiment.** The conditions that share their biological repeats and their control, as when a screen runs target by target, each with its own control: the tests are corrected within it, not over the whole report, so one part of a screen does not dilute another.

*Where:* Do the conditions differ?.

**Well (technical repeat).** A well, dish or coverslip within a biological repeat (the data's technical_repeat column); never a biological repeat. With one well per condition in each repeat, no within-repeat technical replication, the cells of a well are not independent replicates: the repeats are. 'single' in the data when there is one.

*Where:* The data, the plots (point shapes).

**Pilot.** A dataset of one or two biological repeats. It gets every answer its design supports and advice for the next experiment, not a verdict across repeats.

*Where:* Do the conditions differ?, the plan.

**Normalised values.** Each value divided by the mean of the control of its own biological repeat, so a scale shared by a repeat (a staining or imaging batch) is removed. Needs the control mean to be above zero.

*Where:* Do the conditions differ?, the plots, the PCA and UMAP.

## Did I image enough?

**Mean-precision rule.** Stops once the mean of the readout is precise enough: its standard error at most a share of the mean you choose, e.g. 5% (its 95% interval is then about ±10%). 'Per cell' treats cells as independent; 'per field' uses field means.

*Where:* Did I image enough?.
*See:* [Zweifach A. Determining how many cells to average for statistical testing of microscopy experiments. J Cell Biol. 2024.](https://doi.org/10.1083/jcb.202401074)

**Mean-precision count check.** Whether the number of cells the mean-precision rule asks for still gives that precision when the cells come as whole fields, as they are acquired.

*Where:* Why count fields, and not only cells?.
*See:* [Zweifach A. Determining how many cells to average for statistical testing of microscopy experiments. J Cell Biol. 2024.](https://doi.org/10.1083/jcb.202401074)

**JCS spread and plateau rules.** Stop once the mean, median, SD and IQR of a feature stabilise as cells are added (within a tolerance you choose, in the feature's units).

*Where:* Did I image enough?.
*See:* [Marcotti et al. J Cell Sci. 2026.](https://doi.org/10.1242/jcs.264367)

**Held-back fields.** Fields set aside before a rule is replayed, so whether the rule stopped too early is judged on fields it never saw.

*Where:* Plan the next experiment, the optimise workflow.

**Resolved.** The 95% interval of the difference from the control, from the fields resampled as they were acquired, excludes zero: the difference shows within that repeat.

*Where:* Did I image enough?.

## Do the conditions differ?

**Cohen's d.** The difference of two means in units of their pooled standard deviation: 0.2 small, 0.5 medium, 0.8 large by Cohen's convention. Sign: negative is lower than the control.

*Where:* Do the conditions differ?.
*See:* Cohen J. Statistical Power Analysis for the Behavioral Sciences. 2nd ed. 1988.

**Cell-level randomization test.** Shuffles the condition labels over every cell of a condition and its control, as the Plot&Stats notebook does. It asks whether these wells differ, not whether the effect repeats: when the difference varies between repeats it gives p ≤ 0.05 more often than 5% with no real effect, and the report prints how often beside it. It is named after the declared observation unit: the patch-level test for patches.

*Where:* Do the conditions differ?.
*See:* Plot&Stats, the Cell Migration Lab's analysis notebook.

**Balanced test.** The cell-level test when the repeats hold different numbers of cells: each draw takes the same number from every repeat, so none weighs more; d is the median over 100 draws, and p comes from the labels shuffled within each repeat, one permutation test.

*Where:* Do the conditions differ?.

**Sparse repeat.** A repeat with under a quarter of the median repeat's cells, or fewer than 5: left out of the balanced test only, and named so it can be checked.

*Where:* Do the conditions differ?, the design scan.

**Paired t-test on the repeat means.** Each biological repeat's condition mean paired with its own control mean, the repeat as the replicate (as in SuperPlots). Keeps false positives at 5%, but with 3 repeats only large, consistent effects reach q ≤ 0.05. Given raw and on log ratios to the control (normalised), corrected within each sub-experiment.

*Where:* Do the conditions differ?: the test behind 'significant across repeats', the result to report.
*See:* [Lord SJ, Velle KB, Mullins RD, Fritz-Laylin LK. SuperPlots: communicating reproducibility and variability in cell biology. J Cell Biol. 2020;219(6):e202001064.](https://doi.org/10.1083/jcb.202001064)

**Sign-flip test.** The exact test on the signs of the per-repeat differences; given only from 6 repeats, where its smallest p can reach 0.05 after the correction.

*Where:* Do the conditions differ?, full detail.

**Bonferroni correction.** Each p-value multiplied by the number of tests, so that across all of them the chance of even one false positive stays at 5%. Used for the tests within one repeat.

*Where:* Do the conditions differ?, within repeats.

**Benjamini-Hochberg q-value.** A p-value corrected for the many tests made together (the conditions of one experiment, the features ranked, or the pairs of a matrix): among the results with q ≤ 0.05, at most 5% are expected to be false discoveries, on average and when each p-value is valid (the cell-level screen's are not when the repeats differ).

*Where:* Do the conditions differ?.
*See:* [Benjamini Y, Hochberg Y. Controlling the false discovery rate: a practical and powerful approach to multiple testing. J R Stat Soc B. 1995;57(1):289-300.](https://doi.org/10.1111/j.2517-6161.1995.tb02031.x)

**Candidate and significant across repeats.** A feature is a candidate when its exploratory cell-level screen has q ≤ 0.05 for a condition; it is significant across repeats when the paired t-test on the repeat means gives q ≤ 0.05 for that condition too (with n repeats: n is said beside it). The first finds candidates; the second says the mean difference holds across the repeats. 'Same direction in every repeat, not significant', said from 6 repeats, is a consistent trend, not evidence of no effect; with fewer, every repeat goes the same way by chance too often (1 experiment in 4 with 3 repeats), so it reads 'not significant across repeats' (owner, 2026-10-06).

*Where:* Do the conditions differ?, the feature ranking.

**Reagents of one target.** Two siRNAs (or other reagents) of one gene, named in the 'target' column: when both go the same way in every repeat, an off-target effect of one of them is less likely. Described, not tested.

*Where:* Do the conditions differ?.
*See:* [CCT8 associates with the MYO10 motor domain and regulates filopodia and breast cancer cell invasion. J Cell Sci. 2026 (the lab datasets).](https://doi.org/10.1242/jcs.265278)

**JCS effect size.** The treated mean minus the control mean over the control SD, from cells drawn as random cells and as whole fields.

*Where:* Do the conditions differ?, full detail.
*See:* [Marcotti et al. J Cell Sci. 2026.](https://doi.org/10.1242/jcs.264367)

## Figures

**PlotsOfData-style plot.** Every cell a point in its condition's colour, shaped by technical repeat; the biological repeats side by side; each repeat's median with its 95% interval from resampling whole technical repeats with their fields as acquired (never single cells), or whole repeats when they are pooled.

*Where:* The figures of 'Do the conditions differ?'.
*See:* [Postma M, Goedhart J. PlotsOfData: a web app for visualizing data together with their summaries. PLoS Biol. 2019;17(3):e3000202.](https://doi.org/10.1371/journal.pbio.3000202); [Lord SJ, Velle KB, Mullins RD, Fritz-Laylin LK. SuperPlots: communicating reproducibility and variability in cell biology. J Cell Biol. 2020;219(6):e202001064.](https://doi.org/10.1083/jcb.202001064); Ren et al. Nonparametric bootstrapping for hierarchical data. J Appl Stat. 2010.

**PCA and loadings.** The two directions along which the standardised features vary most, each labelled with its share of the variance; an arrow shows how much a best-ranked feature weighs on each direction. Describes, does not test.

*Where:* The whole dataset at once.

**UMAP.** A map that keeps each cell's nearest neighbours close, so it can show groups that PCA mixes. Its axes have no units and distances between groups are not to scale. Describes, does not test; on raw values a separation can be a batch between repeats.

*Where:* The whole dataset at once.
*See:* [McInnes L, Healy J, Melville J. UMAP: Uniform Manifold Approximation and Projection for dimension reduction. arXiv:1802.03426. 2018.](https://arxiv.org/abs/1802.03426)

**Other feature tables and the combined map.** Features not read one by one, such as a deep-learning embedding, loaded as named tables beside the explainable features and never tested feature by feature: each is tested whole instead, its centre against its control's in every repeat. Each has its own map; the combined map weighs every table as much as the explainable features, each divided by its first singular value. Under each map, the share of each cell's nearest neighbours in its biological repeat, outside its field, that have its condition, against chance: how far the conditions separate in that table; and within each condition, the share that have its repeat: how far the repeats do. Describes, does not test.

*Where:* The whole dataset at once.

## The data

**Integrity checks.** Exact checks for accidents of image analysis: a field exported twice, rows pasted into another field, a change of units, a summary written twice. A finding is structure to inspect, not a judgement.

*Where:* The data, the fields to check, the design scan.

**Unusual cells.** Cells far from those of their own control in the same repeat, on values relative to that control; exploratory, never an error or a new state.

*Where:* What limits the sampling?.

**'Nothing new'.** Whether later fields still showed states earlier fields did not, by the definition of new declared in the setup; not the same as unusual cells.

*Where:* Did I image enough?.

**Positive control (representation).** A condition known to differ, used to check that the features can see the phenotype at all before 'nothing new' is trusted.

*Where:* Can these features detect what I care about?.

**Between-field share.** The part of a feature's variance that lies between fields rather than between cells of one field: when it is large, more fields help more than more cells per field.

*Where:* Why count fields, and not only cells?, What limits the sampling?.
