# Figures: every observation a point, the interval from fields

The figures show every observation, grouped by condition and biological repeat, with
each repeat's median and an interval from resampling how the data were sampled. This page
is `plots`, the figures of the HTML page, and the data each figure carries. The PCA and
UMAP are in [Feature tables and maps](feature_tables_and_maps.md).

## `plots`

```python
from morphocoverage.workflows import plots

figures = plots(objects, fovs, setup, folder="figures")   # the readout by default
figures = plots(objects, fovs, setup, features=["YAPTAZ_Ratio", "Nuclei_Area"])
```

PlotsOfData-style figures (`morphocoverage.plots`), one per feature:
within each condition the biological repeats (the `biological_repeat` column) sit side
by side, labelled above their cluster, so that a difference between
conditions can be read against the spread between repeats; each condition is
only as wide as the repeats it holds, so repeats imaged for other conditions
take no room. The figure's title is the feature; what it shows is its legend, at
the bottom. Under the plot (panel A), with two or more conditions, the matrices of
every pair of conditions, as Plot&Stats draws them (`plots(..., pairwise=False)`
leaves them out), side by side when the figure is wide enough, else one under the
other, the legend then lettered (A) to (C). Panel B: Cohen's d on the observations
below the diagonal, the row's condition
minus the column's (signed, blue lower, red higher), and above it the
randomization test on d (the condition labels shuffled over every observation of
the two conditions, pooled over the repeats, 1999 shuffles seeded from
`random_state`), Benjamini-Hochberg corrected over the pairs (q, green and bold
at 0.05). It treats every observation as an independent draw, so when the
repeats differ it passes more often than 5% without an effect; the table keeps
the paired t-test on the repeats' means beside it, drawn as panel C (`p_paired`
below the diagonal, uncorrected, and `q_paired`, Benjamini-Hochberg over the pairs
tested, above; with 3 repeats that imaged both), and a pair imaged in different repeats is marked †, since the
difference between batches is in it (`plots_pairwise.pairwise_contrasts` gives
the table, and each figure carries it as `figure.pairwise`). Every object is a
jittered point in its condition's colour (the HTML report's colours), shaped by
technical repeat (`technical_col`, by default
the MorphoBricks `technical_repeat` column; another column, such as a `well`
group column, is used only when named, `plots(..., technical_col="well")`,
and an analysis-group column the figure does not place raises: a well is a
technical repeat, never a biological one), and each
biological repeat's median (or `summary="mean"`) carries its 95% interval
from resampling that repeat's whole technical repeats, each with its fields
as acquired (Ren et al. 2010; whole fields alone when there is one technical
repeat; with only 2 or 3 technical repeats no resampling gives a reliable
interval, and the legend says so), never objects, so
it reflects how the data were sampled: a well-to-well shift widens the
interval instead of passing for field-to-field noise. More than 12 conditions
are split over several figures, more than 8 biological repeats are pooled
into one cluster per condition and the legend says so (a pooled cluster
resamples its repeats alone, each with all its fields, so the technical
repeats' variance is not counted twice), more than 8
technical repeats are drawn as small dots with a note, and more than 24 features
raise so that you choose; all four are arguments (`max_conditions`,
`max_repeats`, `max_technical`, `max_features`), and `max_technical` above 8,
the shapes that can be told apart, raises. The features must be
declared in the setup, and the plots use the same normalised values as the
report when `normalize` is set; the value axis then says "relative to
control" (`plots_of_data(..., value_note=...)` adds any such note). With `folder`, each figure is saved as a PNG.
Without `features`, the readout is plotted; given the report's
discovery (`plots(objects, fovs, setup, ranking=result.discovery)`), the five
best-ranked features are plotted instead (`top=`): those that pass the cell-level
gate, largest |d| first, as question 9 describes below, the readout among them
only when it ranks (and alone when none does: the report is neutral) (`ranking=result.ranking`
takes the ranking over whole technical repeats instead).

## On the page

The feature figures and the PCA and UMAP are interactive, drawn by plotly.js,
which the page carries (about 1.5 MB, MIT licence), so they work offline. A
feature figure shows every observation in its condition's colour, the biological
repeats side by side (each condition only as wide as the repeats it holds), and
each repeat's median with its interval: the numbers of the static figure; the
technical repeats of a biological repeat differ by shape. With a control
declared, "Raw" and "Normalised" switch the figure between the values as measured
and the values divided by each repeat's control mean (it opens on the values the
setup declares, and the caption and axis say which). Hover a point for its
repeat, technical repeat and field, or a median for its interval; zoom by
dragging; "Enlarge" fills the window. Beyond 4000 observations a figure
draws a seeded sample of each cluster, and says so (the summaries and tests use
every observation). Under it, as in Plot&Stats, the matrix of every pair of
conditions in two panels: (B) Cohen's d below the diagonal (the row's condition
minus the column's) and above it the randomization test on d, every observation of
the two conditions shuffled as Plot&Stats does, Benjamini-Hochberg corrected over
the pairs of that figure (a split figure tests its own conditions); (C) the paired t-test on the repeats' means, its uncorrected p below the
diagonal and its Benjamini-Hochberg q above (with 3 repeats that imaged both; it
holds its false-positive rate but has little power on three). A pair imaged in
different repeats is marked †, since the batch is in its difference
(`plots_pairwise`). The words are a legend under the panels, "(A)", "(B)", "(C)",
as in a paper. The static figure (PNG and PDF, for papers) is folded
under each, with the same matrix under the plot.

Up to 12 conditions, every condition, the control included, has a colour of its own (no
grey, hard to see);
past 12, the control and the conditions significant across repeats (else the
cell-level screen's best-ranked candidates), at most 11, keep a colour and the
others are grey, which the caption says, and "Find a condition" highlights any
one (each part of a split feature figure colours its own conditions, the control
always blue). Long names are cut with an ellipsis in the matrices, the heatmap
and the legends, whole on hover and in the CSV; the heatmap keeps its cells
readable and scrolls sideways, its rows the conditions when they outnumber the
features.

Every figure carries the data it draws as CSV, beside its PNG and PDF (in
`figures/`, inside the page and in the zip), to draw it again outside: per
feature figure `<feature>_data.csv` (every observation, raw and normalised, with
its condition, repeats and field), `<feature>_repeats.csv` (each biological
repeat's median and interval) and `<feature>_pairs.csv` (every pair of
conditions); for the maps `embedding_<values>.csv` (the PCA and UMAP coordinates
of every point, with its condition, repeat, field, `observation` (its index in the
observation table), `label_id` when the table has one, population and standardised
features) and `pca_loadings_<values>.csv` (with other feature tables
`embedding_<values>_<table>.csv`, the combined space's table named `Combined`), and
`separation.csv` (each space's share of neighbours sharing the condition, per biological
repeat); for the heatmap `heatmap_d.csv`.
