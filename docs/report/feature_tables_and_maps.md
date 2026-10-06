# Feature tables and maps

Features that are not read one by one, such as a deep-learning embedding, come as other
feature tables beside the explainable features. This page is how they enter the report,
the PCA and UMAP maps of all the features, and the separation score under each map.

## Other feature tables: embeddings

The features you test one by one (explainable features, such as the readout and the
shape and intensity measurements) are the observation table's. Features that are not
read one by one, such as the embedding of a deep-learning model, come as other feature
tables, each with a name, so that several models can be compared side by side (owner,
2026-10-05):

```python
tables = {"DINOv2": dino_v2, "DINOv3": dino_v3}
result = report(objects, fovs, setup, feature_tables=tables)
html_report(result, objects, fovs, "results", feature_tables=tables)
```

Each table is checked against the observation table at loading
([its rules](../common/data_format.md#other-feature-tables-embeddings-beside-the-explainable-features):
every observation matched exactly once by the same identity, numeric columns, no column
without a value, missing values counted, constant columns listed, no clashing names, a name
of its own), and `result.setup.feature_tables` names their columns. They are never tested
feature by feature, and there is no whole-embedding test: the discovery, the heatmap, the
SuperPlots, "Did I image enough?" and the contrasts read the explainable features as
before. The report says once what the tables join (owner, 2026-10-06): the maps, the
populations, the copy checks and their own fingerprints, not novelty or unusual
observations, which read the explainable features only. With novelty on all the columns, an
audit found 99.6% of the cells of null data called new and a planted new state missed: in
hundreds of dimensions the distance to the nearest cell of another field does not follow a
radius set for the explainable features.

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

In the weighted combined space each table, robustly standardised, is divided by its own
first singular value, as in multiple factor analysis (MFA; owner, 2026-10-06), the
explainable features keeping weight 1: no table's first axis outweighs another's, however
many columns it has and however correlated they are. Equal total variance per table, the
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
the same in raw and normalised values. Beside them, its map **centred on control** shows each
column minus its median in the control of the same biological repeat (each condition's own
control with several controls; owner, 2026-10-06): it removes a shift shared by a whole
repeat, a batch, and keeps the differences within it. It is for the maps only: no test reads
these values. The live session and `optimise` read the columns already in `objects`: for
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
  between repeats is larger than the difference between conditions.

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
