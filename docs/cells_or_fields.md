# Cells or fields? The results in plain language

This page explains, without statistics, what the CHECK mode found on the
committed data (section 4 of `docs/questions.md`). The detailed numbers, the
methods and their limits are in `docs/preliminary_findings.md`, entries 1, 9
and 10. The code is `morphocoverage.sampling_unit`; the analyses on the
committed data are `data/crick_idr/cells_vs_fields.py` and
`data/crick_idr/jcs_effect_fields.py`.

## The question

Two published guides tell microscopists how many cells they need: Zweifach
in JCB (2024) and Marcotti and colleagues in JCS (2026). Both say, in effect,
"measure a few hundred cells and you are done; imaging more wastes time and
money."

Both guides assume the cells are picked at random from the whole well. The
JCS paper makes its case by imaging the whole well first, then picking cells
at random from all of it.

A microscope cannot do that. It takes pictures (fields) one at a time, and
every picture brings all the cells in it. Stopping at 250 cells does not
give you 250 random cells. It gives you the cells of the first one or two
pictures.

So we asked: does the advice still hold when the cells come a picture at a
time? We used the papers' own data, their own wells and their own methods,
and changed only that one thing.

## What we found

1. **Pictures are not random cells.** Cells in the same picture resemble
   each other more than cells from across the well. As a result, 250 cells
   from one or two pictures tell you less than 250 cells picked at random.
   In the JCS dataset, an average taken from pictures was typically about
   twice as uncertain as the JCB formula promises, and up to four times in
   some wells. In the second dataset (idr0028) it was about one and a half
   times.

2. **Whether that matters depends on the picture.** In the second dataset
   each picture holds only about 30 cells, so 200 cells means about 8
   pictures spread over the well. That behaves almost like random cells, and
   the papers' advice works. In the JCS dataset each picture holds about 270
   cells, so 250 cells is just one or two pictures, and the advice breaks
   down.

3. **The JCB rule misses its own target.** The rule is designed to keep the
   error of an average below 5%, or 2.5% for paired experiments. In the JCS
   dataset, the cells it asks for fit in one picture. That one picture met
   the 5% target in 8 of 10 wells, but the 2.5% target in only 3 of 10. The
   formula itself still works if you count pictures instead of cells. But
   with only 4 pictures per well, there is not enough information to say how
   many pictures you would need.

4. **The JCS stability check is fooled.** JCS watches the average and the
   spread of values settle down as more cells are added. With pictures, the
   average and the middle value settle more slowly than with random cells;
   the spread barely changes. In the second dataset you need about 250–320
   cells from pictures to get the stability that 200 random cells give.

5. **A small treatment effect turns into a coin flip.** JCS concluded that
   one compound raised fascin in the nucleus slightly more than its DMSO
   control, from about 250 cells per well. With random cells, that ranking
   comes out right 81% of the time. With the same number of cells taken as
   pictures, it comes out right 55% of the time, about a coin flip, and still
   only 66% with 500 cells. The large effect of the positive control,
   leptomycin B, came out right every time either way.

6. **Unusual cells often sit together.** We flagged cells that look unlike
   the control cells, a stand-in for rare cell states. In 5 of the 12 wells
   of the second dataset, those cells were packed into a few pictures
   instead of spread across the well. In those wells, seeing them reliably
   takes about twice as many pictures as counting cells would suggest. In
   one well, 7 unusual cells sat in just 3 of 28 pictures: seeing them with
   95% confidence would take about 27 pictures, where counting cells suggests
   12.

## What it means

- **The papers are not wrong.** Each gives correct advice under its stated
  assumption, and their main conclusions, which rest on large effects, still
  stand.
- **The assumption does not hold for real imaging.** Cells arrive a picture
  at a time, and pictures differ from each other.
- **Plan in pictures, not cells.** "250 cells" can mean one picture or ten,
  depending on the sample, and those give very different reliability. The
  question to ask is how many pictures, spread how widely.
- **Rare cells make this stronger.** What matters is how many pictures
  contain them, not how many cells you counted.

## Limits

- **Few wells.** These are two datasets and a few wells each, one plate for
  the JCS data.
- **Few pictures in the JCS data.** It has only 4 pictures per well, so the
  real problem is probably somewhat larger than we measured.
- **"Unusual" is not a named cell state.** It means "unlike the controls",
  which can include debris or imaging errors as well as rare biology. The
  flags also changed between repeat runs in the JCS data, so its clustering
  results are uncertain.
- **Some measures shift between runs.** The JCS stability measure varies from
  run to run, so only its direction is reliable: pictures are less stable
  than random cells.

## Checking your own data

Each function takes the observation table and the FOV table (every acquired
field, including empty ones), analyses each analysis group on its own, and
has a `summary()` in sentences.

```python
from morphocoverage.sampling_unit import (
    cell_rule_check,
    descriptor_spread_check,
    field_clustering,
    field_variability,
    unusual_cells,
)

groups = ["condition", "repeat", "well"]

# How different are my fields, and how much does a cell count understate the error?
print(field_variability(cells, fields, feature="ratio", group_cols=groups).summary())

# Would the JCB rule hold if its cells come as whole fields?
print(
    cell_rule_check(
        cells, fields, feature="ratio", control={"condition": "control"}, group_cols=groups
    ).summary()
)

# Does the JCS stabilisation hold with whole fields, at the threshold you would read?
print(
    descriptor_spread_check(
        cells, fields, feature="ratio", threshold=200, group_cols=groups
    ).summary()
)

# Which cells are unlike the controls, and are they clustered in a few fields?
unusual = unusual_cells(
    cells, fields, feature_cols=features, reference={"condition": "control"}, group_cols=groups
)
print(unusual.summary())
print(
    field_clustering(
        cells.assign(unusual=unusual.labels),
        fields,
        label_col="unusual",
        states=[True],
        group_cols=groups,
    ).summary()
)
```

`field_clustering` accepts any labels, not only unusual cells: clusters from
a mixture model, a gate on one feature, or manual annotations.
