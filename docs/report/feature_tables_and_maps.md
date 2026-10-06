# Feature tables and maps

Feature tables come one per provenance, each tagged explainable (fiberfeatures,
CellProfiler: each column read and tested one by one) or non-explainable (an embedding such
as DINO, read only as a whole). This page is how they enter the report, the PCA and UMAP
maps of all the features, and the separation score under each map.

## Explainable and non-explainable tables

Each table passed to `report`, `html_report`, `scan_design` or `guide` is tagged (owner,
2026-10-06), any mix allowed:

```python
tables = {
    "fiberfeatures": (fiber, "explainable"),
    "CellProfiler": (profiler, "explainable"),
    "DINOv2": (dino, "non-explainable"),
}
setup = Setup(readout="area", control={"condition": "ctrl"})
result = report(objects, fovs, setup, feature_tables=tables)
html_report(result, objects, fovs, "results", feature_tables=tables)
```

The tag is `"explainable"` or `"non-explainable"` in any case (`"non_explainable"` too);
another tag is refused. A bare DataFrame is non-explainable, so every call written before
the tags gives the same output. The observation table's own numeric columns form an
implicit explainable table, as before.

- **Names.** A column name found in two tables is kept in the observation table and
  prefixed with its table's name in every other table, `"fiberfeatures: area"`; a name
  found once is kept. The line of the table in the overview and the scan says the renaming
  once. Metadata names stay refused.
- **Tests.** An explainable table's columns are explainable features: declared in
  `Setup.features` (or as the readout) under their final names, each is tested one by one
  exactly as an observation-table feature is, in one Benjamini-Hochberg family over every
  explainable feature of every table, per sub-experiment. **With no features declared,
  every column of the explainable tables is a feature**, the readout still declared (a
  Setup without a readout is refused when explainable tables are given, since it would not
  say which feature the stopping rules read); the overview says this in one sentence. The
  result tables gain a `table` column (`feature_discovery.csv`, `tests_across_repeats.csv`,
  `tests_cell_level.csv`, `heatmap_d.csv`, `tests_beyond_the_mean.csv`; the observation
  table's own features under `observation table`), and the page lists the features under
  their table (the significant features of each condition, the tabs of "Did I image
  enough?"). With one explainable table, nothing changes.
- **Novelty, unusual observations and the populations** read the declared explainable
  features. When they come from two tables or more, each table is weighted by MFA
  (robustly standardised, divided by its first singular value over `sqrt(n - 1)`, so its
  first axis has variance 1), so a table of many columns does not outweigh one of few:
  unusual observations and the populations on all the observations, the definition of new
  on the visible fields (`transforms.BlockWeights` after its own transform). The radius
  declared for `d` standardised features, `sqrt(d)`, then reads as the square root of the
  weighted space's total inertia, so the share of new cells stays that of one table rather
  than depending on how the features are split (owner, 2026-10-06, audit 8). A population
  rule may use any explainable feature, and with several tables names the table of each
  ("length from fiberfeatures, int0 from CellProfiler").
- **Maps and separation**: with two or more explainable tables, each has its own map and
  separation line; the "Explainable" map is all the explainable features MFA-weighted
  across their tables; the "Combined" map is MFA over every table, each explainable and each
  non-explainable table its own block. The declared fingerprint reads the explainable
  features as before; each non-explainable table keeps its own fingerprint, and only the
  non-explainable tables are tested whole.
- **Normalisation**: the explainable tables follow the features' control normalisation
  (`Setup.normalize`); the non-explainable tables the batch normalisation below. The copy
  checks run on each table apart, each explainable table on its own.
- **The Setup** records the explainable tables in `Setup.explainable_tables` (name to final
  columns) beside `Setup.feature_tables`, which keeps naming the non-explainable ones.
- **The guided session** asks, for each table it loads, whether it is explainable; the
  scan lists each table with its tag, and the saved answers record it (answers saved before
  the tags read as non-explainable).

## Other feature tables: embeddings

The features you test one by one (explainable features, such as the readout and the
shape and intensity measurements) are the observation table's and the explainable tables'.
Features that are not read one by one, such as the embedding of a deep-learning model, come
as non-explainable tables (a bare DataFrame), each with a name, so that several models can
be compared side by side (owner, 2026-10-05):

```python
tables = {"DINOv2": dino_v2, "DINOv3": dino_v3}
result = report(objects, fovs, setup, feature_tables=tables)
html_report(result, objects, fovs, "results", feature_tables=tables)
```

Each table is checked against the observation table at loading
([its rules](../common/data_format.md#other-feature-tables-embeddings-beside-the-explainable-features):
every observation matched exactly once by the same identity, numeric columns, no column
without a value, missing values counted, constant columns listed, a name found twice
prefixed, a name of its own), and `result.setup.feature_tables` names their columns. They are never tested
feature by feature: the discovery, the heatmap, the SuperPlots, "Did I image enough?" and
the contrasts read the explainable features as before. Each table is **tested whole**
instead, beside the per-feature tests (owner, 2026-10-06): for each condition against its
own control, does its centre in the table move away from the control's, the same way in
every biological repeat? The test, its calibration and its words are in
[Do the conditions differ?](conditions.md#the-whole-table-test-other-feature-tables). The
report says once what the tables join: the whole-table test, the maps, the populations, the
copy checks and their own fingerprints, not novelty or unusual observations, which read the
explainable features only (with a readout and no `features` declared, unusual observations
read the readout alone; owner, 2026-10-06, audit 8). With novelty on all the columns, an
audit found 99.6% of the cells of null data called new and a planted new state missed: in hundreds of dimensions the
distance to the nearest cell of another field does not follow a radius set for the
explainable features. A report with **no explainable feature** at all (an embedding-only
Setup, `readout=None`, owner 2026-10-06) reads the tables alone: see
[the Setup](../common/setup.md#embedding-only-reports-no-explainable-features); its unusual
observations are judged on the tables' weighted combined space (one table: its columns),
and the report says which space.

- **The maps**: one per table and a weighted combined map, shown first (below).
- **The populations** are searched in each space apart: the explainable features, each
  table, and the weighted combined space (owner, 2026-10-06), since a group living in one
  table is found there and can be drowned in the combined space. Each distinct group is
  reported once, naming the space or spaces that found it: groups found in two spaces are
  one when their cells overlap with a Jaccard index (the cells in both over the cells in
  either) of 0.5 or more; a group keeps the cells and description of the first space that
  found it (explainable, each table, combined). A group the explainable features found is
  described and given a rule with them; one found only in other spaces is said as "found in
  DINOv2 only; no rule in the explainable features describes it", never named after
  explainable features that barely differ there. The enrichment tests run on the merged
  list, one Benjamini-Hochberg family as before. Each space other than the explainable
  features is clustered on at most 5,000 cells (a seeded sample of the same share of every
  condition), to keep the runtime modest.
- **The copy checks** run on the explainable features and on each table apart, and the
  report gives the union, naming the table each copy was found in: pooled, a copy confined
  to one table (an embedding computed twice for one image) was missed.
- **A fingerprint** declared on them runs per table: the guided session declares one for the
  explainable features and one for each table at the same tolerance, each with its own
  verdict and line ("The fingerprint of DINOv2 (...): ..."). A fingerprint on all the
  columns at once needed about three times the fields to meet a tolerance chosen for the
  explainable features.
- **A wide table's fingerprint reads its first k principal components** (owner,
  2026-10-06). A covariance or quantiles of 384 columns cannot be estimated from a group of
  a few images: on the ECM tables the error of the DINO fingerprint read up to 192,282 and
  asked for "1e13 images per group". The principal components are fitted on every
  observation of the table holding all its values, robustly standardised as the maps are,
  and keep their variance (the first weighs most). The rule for k, one k for every group
  (owner, 2026-10-06, audit 8): at most a fifth of the observations of the median group the
  fingerprint judges (an analysis group of at least 4 fields) and at most the table's
  columns; the line says k ("read on its first 4 principal components (k at most a fifth of
  the median group's 23 images and at most its 384 columns)"). A judged group holding fewer
  than five times k observations is **not assessable** and named in that line ("not
  assessable in pilot R1, holding fewer than 5 k, 200 cells"), never lowering k: when k came
  from the smallest group, a pilot condition of 4 fields of 3 cells took k from 40 to 2 and
  turned every other group's verdict from 0-17% meeting the tolerance to 100% (with the rule,
  k stays 40 and the verdicts do not move). Below k = 2 for the median group the fingerprint
  is not assessable in any group, and the line says why (the median group's size, and that 2
  components need groups of 10); so it is, for a table's fingerprint, when no group holds 4
  fields: `optimise` then says it is not assessable rather than judging the full width
  (seed 9). The explainable features keep today's form when they number at most that fifth,
  or when no group holds 4 fields, and follow the same rule when they are wider. `optimise`
  judges the same number of components on held-back fields, never in a group too small for
  k (its draws not judged, the group named; `ComponentsError.excluded`)
  (`workflows._fingerprint_space`, `multivariate_outcomes.ComponentsError`). There the
  components are refitted in each split on the group's fields that are not held back, so no
  held-back field enters the fit that it judges (owner, 2026-10-06, plan v2: a fit on every
  field let the reference shape its own axes); the report's reading, from random halves of
  all the fields with nothing held back, keeps the fit on every observation.

In the weighted combined space each table, robustly standardised, is divided by its own
first singular value over `sqrt(n - 1)`, as in multiple factor analysis (MFA; owner,
2026-10-06): each table's first axis has variance 1, so none outweighs another, however
many columns it has, however correlated they are and whatever the order of the tables
(audit 8: when the first table kept weight 1 and set the others' scale, the share of new
cells went from 2.3% to 8.6% by swapping two tables; the order no longer matters, 3.5%
either way against 3.6% for the same columns as one table). With one table nothing is
weighted. Equal total variance per table, the
earlier weighting, gave the combined PCA to the more correlated table. Under the combined
PCA, each table's share of each axis shown is said ("PC1: explainable 42%, DINOv2 58%"):
the sum of its columns' squared loadings. The robust standardisation divides each column by
its MAD as an SD, or by its SD when the MAD is under a quarter of the SD (a column with just
under half its values at one value, such as an embedding's ReLU outputs, would otherwise
carry thousands of times the variance of the others); columns with an ordinary MAD are
scaled as before. Missing values in a table are said at loading and, in each analysis that
reads the table, the cells left out are counted (its maps, its populations, the separation
score, its fingerprint).

An other table is never divided by a control (an embedding has no ratio scale), so its map is
the same in raw and normalised values. **Batch normalisation** (owner, 2026-10-06): beside
them, its map **centred per repeat** shows each column minus a median of its biological
repeat, which removes a shift shared by a whole repeat, a batch, and keeps the differences
within it. The median is the same for every design (owner, 2026-10-06, audit 8; it
replaces the earlier rule of all the repeat's observations in a balanced design, else the
control's, from 5 observations): the median of the per-condition medians of the conditions
present in every biological repeat. No control is needed, and a condition imaged with many
more cells in one repeat does not move the centre, as it did with the median of all the
observations; nor does the noise of a small control, which the control's median carried into
every observation of its repeat and so set the repeats apart (audit 8: a control of one
field of 5 cells read as a batch in every simulation). When no condition is in every repeat,
the rule applies within each sub-experiment (the repeats linked by the conditions they
share), on the conditions in all of its repeats; the repeats of a sub-experiment without
one are left out of the centred values, and named once in that line: raw values beside
centred ones would read as a batch in the centred map, the populations and the separation
(owner, 2026-10-06). With no centre anywhere, there is no centred view and the tables'
analyses read them as given. The page and the text say in one line which conditions
set the centre ("Batch: the other tables centred per biological repeat on the median of the
per-condition medians of ctrl, a and b, the conditions in every repeat; ...").
When normalisation is declared
(`normalize="control_mean"`), the tables' analyses read these centred values, as the
explainable features read their control-normalised ones: the populations, unusual
observations (in a report without explainable features), the separation of the conditions
and the centred map; the raw map is still shown. The combined map, whose space the
populations search, follows them (owner, 2026-10-06, audit 8, seed 15): with `normalize` its
normalised view reads the normalised explainable features and the centred tables, and in a
report without explainable features it has a centred view, the tables' combined space
centred. These values never enter a test: the whole-table
test computes its own differences within each repeat on raw values, and the fingerprint
reads raw values too. On the ECM tables, whose repeats do not all hold the same conditions
(R4 lacks VEC and THBS5, R5 the SiNC controls), the tables are centred on the conditions
imaged in every repeat.
The live session and `optimise` read the columns already in `objects`: for
them, attach the tables first (`attach_feature_tables(objects, fovs, tables).objects`).

## The PCA and UMAP: `embedding_plots`

With three or more features, `embedding_plots(objects, fovs, setup,
discovery=result.discovery, folder=...)` draws two overviews of all of them,
coloured by condition (the control in the first colour, no grey; past 12 conditions only the control and the discovery's hits keep a colour) on the same values as the plots: a
PCA of the robustly standardised features, each axis labelled with its share
of the variance and the best-ranked features drawn as arrows of their
loadings, so the axes read in features; and a UMAP (`umap-learn`, seeded by
`random_state`; above 5000 observations on a seeded sample of the same share
of every condition). Both describe, they do not test: a separation is a
pattern to read with question 9's tests, and on raw values a separation can
be a batch between repeats, which `normalize="control_mean"` takes out.
Under the page's PCA, one line per axis names its three strongest loadings
with their signs on the axis as drawn, whatever the ranking, and the caption
says how many PCs hold 90% of the variance (counted by the populations) and
which arrows are drawn: the best-ranked features, or without a ranking the
five largest loadings.

## On the page

On the PCA and UMAP, hover a point for its condition, repeat and field, and its observation and `label_id` to
find it again in the table and the image, click a condition in the legend to
hide it, switch between PCA and UMAP and between raw values and values
normalised to each repeat's control, and draw around points (the lasso is on) to
see which conditions they come from: their number, their share of the selection,
the share chance would give and the ratio of the two. Under each map, one line says
how far the conditions, and the repeats, separate in it (owner, 2026-10-05 and 2026-10-06),
a description with no p-value:

- **by condition**: of each observation's 10 nearest neighbours among the other
  observations of its biological repeat, the share that have its condition, against that
  condition's share of those it could have (chance), the median over the repeats ("DINOv2:
  61% of neighbours share the condition, against 13% by chance; median over 3 repeats");
- **by repeat**: within each condition, the share of each observation's neighbours that
  share its repeat, against chance, the median over the conditions. When the repeats
  separate more than the conditions in a table (their score goes further from chance
  towards 100%, and at least a tenth of the way), the line says so plainly: there, a batch
  between repeats is larger than the difference between conditions. This score always
  reads the raw values, in every view (owner, 2026-10-06, audit 8): measured on values
  centred per repeat it would measure a batch after removing it, and a centre with noise of
  its own read as a batch (a control of one field of 5 cells: "the repeats separate more"
  in every simulation with no batch). The score by condition reads the view's values; under
  a normalised or centred map the line says so ("Within a condition, on the raw values, 35%
  share the repeat ..."), so it does not contradict the centred picture, and the two scores,
  reading different values, are not compared: no "the repeats separate more" under such a
  map (audit 10, owner, 2026-10-06). Cells with every raw value
  but none in the view are counted apart from the missing values and named: "18 cells of b1
  in R1 left out: no own control" (or "no condition to centre on", for a repeat left out of
  the centred values); the populations name them the same way (owner, 2026-10-06, audit 8).

With one well per condition and repeat, or no well column, a difference between wells is
also a difference between the conditions of a repeat, so the score by condition reads it as
a separation of the conditions (audit 8: a well shift of 0.5 SD with no condition effect
took the score 34% of the way from chance to 100%, and "the repeats separate more" in half
the simulations). One line under the maps says so for such a design ("Each condition is
imaged in one well per biological repeat, no within-repeat technical replication, so a
difference between wells reads here as a separation of the conditions."; wording of plan v2,
owner, 2026-10-06); with two wells or more per condition and repeat, a well is
never its own observations' neighbour and the line is left out.

Neighbours never come from the observation's own field, nor from its own well when its
condition has another well in the repeat, and chance is computed over what remains: cells of
one field resemble each other, and counted as neighbours they made a field effect read as a
separation of conditions (99% against 33% by chance at a field ICC of 0.2 with no condition
effect, 28% against 29% without them). Where a condition has fewer than 10 such neighbours
in a repeat (one image per field, six images a condition), the neighbours are as many as the
fewest any observation of the repeat has, so a perfect separation reads 100% rather than
50%.

With [other feature
tables](#other-feature-tables-embeddings), a selector switches between a map per table,
the weighted combined map first, and a table under the maps compares them (header "feature
table"; the combined space shown apart, last and not ranked, as it joins the others).
