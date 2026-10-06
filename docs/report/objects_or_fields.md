# Objects or fields? The results in plain language

This page explains, without statistics, what the foundation of the report
("Why count fields, and not only objects?", the [Foundation](questions.md#foundation-why-count-fields-and-not-only-objects)) and the clustering of rare objects (R4) found on the
committed data. An
object is one measured thing: here a cell, elsewhere a nucleus, a patch or a
slice. The detailed numbers, the methods and their limits are in
[preliminary findings](../preliminary_findings.md), entries 1, 9 and 10. The code is
`morphocoverage.sampling_unit`; the analyses on the committed data are
`data/crick_idr/objects_vs_fields.py` and `data/crick_idr/jcs_effect_fovs.py`.

## The question

Two published guides tell microscopists how many objects they need: Zweifach
in JCB (2024) and Marcotti and colleagues in JCS (2026). The JCB guide gives a
formula for the number of objects to average, which on our data asks for 2
to 8 objects in one dataset and 28 to 112 in the other. The JCS guide
observes, in its case study, that a few hundred cells were enough and that
imaging more wastes time and money.

Both guides assume the objects are picked at random from the whole well. The
JCS paper makes its case by imaging the whole well first, then picking cells
at random from all of it.

A microscope cannot do that. It takes pictures (fields) one at a time, and
every picture brings all the objects in it. Stopping at 250 objects does not
give you 250 random objects. It gives you the objects of the first one or two
pictures.

So we asked: does the advice still hold when the objects come a picture at a
time? We used the papers' own data, their own wells and their own methods,
and changed only that one thing.

## What we found

1. **Pictures are not random objects.** Objects in the same picture resemble
   each other more than objects from across the well. As a result, 250
   objects from one or two pictures tell you less than 250 objects picked at
   random. In the JCS dataset, an average taken from pictures was typically
   about twice as uncertain as the JCB formula promises. In the second dataset
   (idr0028) it was about one and a half times. The differences between wells
   are mostly the noise of measuring this from 4 (or 28) pictures per well,
   not wells that differ, so read the typical value only.

2. **Whether that matters depends on the picture.** In the second dataset
   each picture holds only about 30 objects, so 200 objects means about 8
   pictures spread over the well. The error is still about one and a half
   times what random objects give, but it stays small enough for the 5%
   target, and the papers' advice works. In the JCS dataset each picture holds about
   270 objects, so 250 objects is just one or two pictures, and the advice
   breaks down.

3. **The mean-precision rule misses its target when its objects come as whole
   pictures.** The rule is designed to keep the
   error of an average below 5%, or 2.5% for paired experiments. In the JCS
   dataset, the objects it asks for fit in one picture. That one picture met
   the 5% target in 8 of 10 wells, but the 2.5% target in only 2 of 10.
   Across all 308 wells of the plate, it misses the 2.5% target in 76% of
   wells; across all 1,144 wells of the second dataset, in 6%. The formula
   itself still works if you count pictures instead of objects. But
   with only 4 pictures per well, there is not enough information to say how
   many pictures you would need.

4. **The JCS stability check is fooled.** JCS watches the average and the
   spread of values settle down as more objects are added. With pictures, the
   average settles more slowly than with random objects in both datasets. In
   the JCS dataset the middle value does too, while the spread barely
   changes; in the second dataset the middle value and the standard
   deviation are only a little less stable, and the interquartile range not
   at all. There you need about 240–330 objects from
   pictures to get the stability that 200 random objects give.

5. **A small treatment effect turns into a coin flip.** JCS concluded that
   one compound raised fascin in the nucleus slightly more than its DMSO
   control, from about 250 cells per well. With random objects, that ranking
   comes out right 81% of the time. With the same number of objects taken as
   pictures, it comes out right 55% of the time, about a coin flip, and still
   only 66% with 500 objects. The large effect of the positive control,
   leptomycin B, came out right every time either way.

6. **Unusual objects sometimes sit together.** We flagged objects that look
   unlike the control objects, a stand-in for rare states. In 4 of the 12
   wells of the second dataset (3 in the previous run, with other random
   draws), one of them a control well, those objects were packed into a few
   pictures instead of spread across the well; by chance alone, fewer than
   one well would be. In those wells, seeing them reliably takes about twice
   as many pictures as counting objects would suggest (1.9 times). In
   one well, 7 unusual objects sat in just 3 of 28 pictures: seeing them with
   95% confidence would take about 27 pictures, where counting objects
   suggests 12.

## What it means

- **The papers are not wrong.** Each gives correct advice under its stated
  assumption, and their main conclusions, which rest on large effects, still
  stand.
- **The assumption does not hold for real imaging.** Objects arrive a picture
  at a time, and pictures differ from each other.
- **Plan in pictures, not objects.** "250 objects" can mean one picture or
  ten, depending on the sample, and those give very different reliability.
  The question to ask is how many pictures, spread how widely.
- **Rare objects make this stronger.** What matters is how many pictures
  contain them, not how many objects you counted.

## Limits

- **Few plates.** These are two datasets, one plate for the JCS data. The
  complete plates (308 and 1,144 wells) carry only the readout, so the
  stability, treatment and unusual-object checks use the paper's wells.
- **Few pictures in the JCS data.** It has only 4 pictures per well, so the
  real problem is probably somewhat larger than we measured.
- **"Unusual" is not a named state.** It means "unlike the controls", which
  can include debris or imaging errors as well as rare biology. The flags
  also changed between repeat runs in the JCS data, so its clustering results
  are uncertain. A state the controls already hold is never unusual, whatever
  its share: the populations question compares such shares instead. In a screen
  of sub-experiments, each condition is judged against the controls of its own
  sub-experiment (`unusual_objects(within_experiment=True)`, the default).
- **"Unusual" in the report is relative to the control of each repeat.** Called
  directly on raw values, as below, `unusual_objects` reads acquisition too: a repeat
  imaged brighter widens its threshold. The report reads each value relative to its own
  control in its repeat (normalised with `normalize`, else the log ratio or the difference
  to that control's mean), calls the result exploratory, and lists fields far from the rest
  of their condition and repeat on the raw values apart, as quality control in "What is in
  the dataset" (owner, 2026-10-06, plan v2).
- **One picture per object.** When every picture holds one object (manual single-cell
  imaging), pictures and objects coincide: the variability check gives the spread over
  pictures with a design effect of 1, the unusual objects are unusual pictures, and whether
  objects cluster in a few pictures, or a count rule holds with whole pictures, has no
  meaning; the report says so once (owner, 2026-10-06).
- **Some measures shift between runs.** The JCS stability measure varies from
  run to run, so only its direction is reliable: pictures are less stable
  than random objects.

## Checking your own data

Each function takes the object table and the FOV table (every acquired
field, including empty ones), analyses each analysis group on its own, and
has a `summary()` in sentences. `microscopist_report` accepts the results
and places them in the report: the first three in its Foundation section,
the unusual objects and their clustering in "What is limiting my sampling?".

```python
from morphocoverage.sampling_unit import (
    descriptor_spread_check,
    fov_clustering,
    fov_variability,
    object_rule_check,
    unusual_objects,
)

groups = ["condition", "biological_repeat", "technical_repeat"]  # the well is the technical repeat

# How different are my fields, and how much does an object count understate the error?
print(fov_variability(objects, fovs, feature="ratio", group_cols=groups).summary())

# Would the mean-precision rule hold if its objects come as whole fields?
print(
    object_rule_check(
        objects, fovs, feature="ratio", control={"condition": "control"}, group_cols=groups
    ).summary()
)

# Does the JCS stabilisation hold with whole fields, at the threshold you would read?
print(
    descriptor_spread_check(
        objects, fovs, feature="ratio", threshold=200, group_cols=groups
    ).summary()
)

# Which objects are unlike the controls, and are they clustered in a few fields?
unusual = unusual_objects(
    objects, fovs, feature_cols=features, reference={"condition": "control"}, group_cols=groups
)
print(unusual.summary())
print(
    fov_clustering(
        objects.assign(unusual=unusual.labels),
        fovs,
        label_col="unusual",
        states=[True],
        group_cols=groups,
    ).summary()
)
```

`fov_clustering` accepts any labels, not only unusual objects: clusters from
a mixture model, a gate on one feature, or manual annotations.
