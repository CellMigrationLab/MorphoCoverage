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
([its rules](data_format.md#other-feature-tables-embeddings-beside-the-explainable-features):
every observation matched exactly once by the same identity, numeric columns, missing
values counted, no clashing names, a name of its own), and `result.setup.feature_tables`
names their columns. They are never tested feature by feature, and there is no
whole-embedding test: the discovery, the heatmap, the SuperPlots, "Did I image enough?"
and the contrasts read the explainable features as before. With the explainable features
they make:

- the PCA and UMAP: one map per table and a weighted combined map, shown first;
- the populations, found on the combined space and still named, and given rules, with
  explainable features;
- unusual observations, novelty (with the positive control and the fields to inspect)
  and the copy checks;
- a fingerprint declared on their columns (the guided session's fingerprint reads the
  combined space).

In the weighted combined space each column is standardised as each analysis
standardises it, then each table is weighted so that it carries the explainable
features' total variance: a table of `p` columns times `sqrt(p_explainable / p)`, MFA's
equal weight per table with the square root of the number of columns in place of the
first singular value, so the weights do not depend on the data. The definition of new
keeps its meaning there with its radius times the square root of the number of tables.
An other table is never divided by a control (an embedding has no ratio scale), so its
map is the same in raw and normalised values. On the page, under each map, the share of
neighbours that share the condition against chance compares the tables: a higher share,
the conditions sit further apart in that table, by biology or by how the conditions were
imaged; near chance, that table does not tell them apart here, which does not show them
alike. The live session and `optimise` read the columns already in `objects`: for them,
attach the tables first (`attach_feature_tables(objects, fovs, tables).objects`).

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
how far the conditions separate in it (owner, 2026-10-05): of each observation's 10
nearest neighbours among the other observations of its biological repeat, the share that
have its condition, against that condition's share of the repeat (chance), the median over
the repeats ("DINOv2: 61% of neighbours share the condition, against 13% by chance; median
over 3 repeats"); a description, with no p-value. With [other feature
tables](#other-feature-tables-embeddings), a selector switches between a map per table,
the weighted combined map first, and a table under the maps compares them.
