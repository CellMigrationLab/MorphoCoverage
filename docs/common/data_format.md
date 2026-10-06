# Data format: the two tables MorphoCoverage reads

MorphoCoverage asks whether the fields of view (FOVs) of an acquisition were
enough. To answer, it needs to know which fields were acquired and what was
measured in each. It therefore reads two linked tables:

- the **FOV table**: one row per acquired field, *including fields in which
  nothing was measured*;
- the **observation table**: one row per phenotype observation (a cell, a
  nucleus, a patch, a slice, an embedding), with the field it came from.

Column names follow the MorphoBricks shared contract (`AGENTS.md` §5). The
validator is `morphocoverage.validate_input_tables`; every workflow
(`report`, `optimise`, `benchmark`, `plots`, `live_session`) calls it, so the
rules below hold everywhere. Check your tables before a first analysis with
`morphocoverage.check_tables` (see [Checking your tables](#checking-your-tables));
`report`, `optimise`, `benchmark` and `plots` also run it and pass its
warnings on (the report prints them at the end of its overview).

## Mandatory columns

| Column | Table | Type | What MorphoCoverage does with it |
| --- | --- | --- | --- |
| a field identifier: `position_id`, `image_file` or `fov_id` | both | string | Identifies the field, the unit that is counted, resampled and revealed whole. When `fov_col` is not given, the first of `position_id`, `image_file`, `fov_id` that holds values is used. Use `position_id` when one file is not one field (one file per z-plane, or several stage positions in one file). |
| feature columns | observations | numeric | The phenotype. Always named explicitly (`feature_cols`, or `Setup.readout` and `Setup.features`), never inferred. Missing values (NaN) are kept and reported; infinite values are refused. |

The FOV table itself is mandatory for the workflows. Without it, fields that
hold no observation vanish, effort is undercounted and every sampling answer
is optimistic. If no acquisition record exists,
`morphocoverage.schema.fov_table_from_observations` rebuilds one from the
observations. It warns every time, because it cannot know about empty fields.
`validate_input_tables` called without `fovs` does the same, with a
`MissingFovTableWarning`, and records `fovs_from_observations=True` on the
result.

## Optional columns

| Column | Table | Type | When present |
| --- | --- | --- | --- |
| `condition` | both, or FOV table only | string | Experimental group. A default analysis group and part of field identity. Needed to declare a control (`Setup.control`), to normalise to it, and for the condition contrast. |
| `biological_repeat` | both, or FOV table only | string | Biological repeat, `YYYY_MM_DD` or `single` in MorphoBricks; any label is accepted. A default analysis group and part of field identity. Normalisation and the condition contrast work within each repeat. Formerly named `repeat`: that name is still read, with a warning (see [The former name `repeat`](#the-former-name-repeat)). |
| `technical_repeat` | both, or FOV table only | string | Technical repeat within the biological repeat: a well, a dish, a coverslip. Never a biological repeat. A default analysis group and part of field identity. The overview counts technical repeats. Plots shape points by them and resample them before fields. The contrast shuffles whole technical repeats when its smallest p can reach 0.05 after the Bonferroni correction, 4 a side for one contrast. `Setup` derives the hierarchy "field within technical repeat within repeat" from it, for "more fields or more technical repeats?". `single` means one technical repeat per repeat (see [A level with one member](#a-level-with-one-member-single)). |
| `target` | FOV table (may also be in the observation table) | string | What each condition acts on, for example the gene two siRNAs silence (`CCT8` for `siCCT8#5` and `siCCT8#8`; `non-targeting` for the control). Not a group column. Question 9 then says, per target with two or more reagents, in how many repeats all of them are on the same side of the control; this is described, not tested. One target per condition, or the contrast raises. |
| other group columns | both, or FOV table only | any | Named with `group_cols`, for example `batch`. Each combination of group values is analysed apart. Without a `technical_repeat` column, one nested within condition and biological repeat (a `well` column) is the unit the condition contrast shuffles. |
| `z`, `c`, `t` (or `z0`, `z1`) | observations | integer | Plane, channel and time point within one position. Required when one position spans several files, so the files can be told apart. Never new fields, and never features: naming one in `feature_cols` raises. Rows of several planes, channels or time points in one field are refused unless each row measures a whole plane (see [Planes are not fields](#rules-the-validator-enforces) and [Features measured per plane](#features-measured-per-plane)). Read in lowercase only, so a column `Z`, `Metadata_Z`, `plane`, `slice`, `channel` or `frame` raises as a near miss. |
| acquisition order | FOV table | numeric, datetime or timedelta | Named with `acquisition_order_col` (MorphoBricks suggests `acquisition_order`). The recorded order in which fields are replayed; text is refused, since it would sort `'10'` before `'2'`. Without it only random orders are replayed; the order of the rows or of the file names is never taken for it. |
| stage coordinates | FOV table | numeric, finite | Named with `stage_cols` (MorphoBricks suggests `stage_x`, `stage_y`). Used for dispersed orders, stage travel, redundancy distance and spatial blocks; an infinite coordinate raises. |
| imaged area | FOV table | numeric, positive | Named on `FixedAreaStrategy`. The physical area of each field, for area-based stopping. A missing, infinite, zero or negative area makes the rule not applicable, and a text column raises; a benchmark record's effort refuses all of them. |
| a rare-state label | observations | boolean or 0/1 | Named with `Setup.rare_label`. An independent label of the rare phenotype, so held-back fields can judge whether it was seen. A missing value means unknown: the observation is left out of every share (owner, 2026-10-05). |
| `label_id`, `mask_file`, `patch_file` | observations | any | Provenance, kept for tracing an observation back to its image; never features. `check_tables` also reads CellProfiler's `ObjectNumber`, `object_id` or `label` as the object label. |
| `valid`, `reason` | observations | boolean, text | CellPatcher marks a patch it could not cut with `valid=False`. Such rows are still analysed as observations, so the validator warns (`InvalidRowsWarning`); filter them out first if they are not meant. |

## A level with one member: `single`

A hierarchy level that the experiment does not have is written as the value
`single`, CellPatcher's convention: the level exists and has one member.
`biological_repeat="single"` is one biological repeat, and
`technical_repeat="single"` one technical repeat in each biological repeat.
MorphoCoverage treats `single` as any other label: a group of one member,
counted as one repeat in the overview and resampled as one unit. An empty
cell means something else: the value was not recorded. An empty
`technical_repeat` (or `position_id`) in every row is read as the column
being absent; a partly empty one, or an empty `condition` or
`biological_repeat` value, is refused, since the observation's group would
be unknown.

## The former name `repeat`

MorphoBricks named the biological repeat `repeat` before it became
`biological_repeat` (`technical_repeat` is unchanged). Tables that still use
`repeat` are accepted for now, with a `DeprecationWarning`: wherever the
tables enter (`validate_input_tables`, `check_tables`, every workflow,
`OnlineSession.add_fov`, `LiveTables`, a `LiveFolder` manifest), a `repeat`
column in a table without `biological_repeat` is read as
`biological_repeat`. Your tables are not changed, and every result names
the new column. A table with both columns is refused unless they hold the
same values (then `biological_repeat` is used, with the warning). A
`group_cols`, `fov_col`, `control`, `Setup` or `SamplingHierarchy` field
naming `repeat` is read the same way. Rename the column: the former name
will be removed in a later version.

## Rules the validator enforces

- **Exact names.** Column names are lowercase and exact. A near miss raises,
  with the column it probably means ("did you mean 'biological_repeat'?"),
  so a misspelt group cannot silently merge repeats: `Biological_Repeat`,
  `Technical_Repeat`, `biological repeat` or `technical repeat` (a space for
  the underscore), `biologicalrepeat` (no separator), `technicalRepeat` or
  `PositionID` (camelCase), `technical.repeat` (a dot), `Conditions` (a
  plural), `condition ` (a trailing space), `Repeat.1` (the copy pandas
  names when a file has two `Repeat` columns), `Z` for `z`, or a byte-order
  mark that a spreadsheet wrote before the first header. Short and other
  names of the two repeats are near misses too: `bio_repeat`, `biol_rep`,
  `biological_replicate` for `biological_repeat`; `tec_repeat`,
  `tech_repeat`, `tech_rep`, `technical_replicate` for `technical_repeat`;
  and `Repeat`, `repeats` or `Metadata_repeat` for the former `repeat`.
  So are typing errors of the shared names: one letter added, missing,
  changed or swapped with its neighbour (`conditon`, `image_flie`,
  `postion_id`), or two in the repeats' longer names (`biological_repat`,
  `techincal_repeat`); a longer name (`technical_repeat_id`) is not one.
  CellProfiler's `Metadata_` prefix before such a name (`Metadata_Z`,
  `Metadata_condition`) and the other names of a plane, channel or time
  point (`plane`, `slice`, `channel`, `frame`, `timepoint`, ...) are near
  misses too, since such a column would otherwise be ignored and its planes
  counted as fields. Name the column in `group_cols` or `fov_col` to use it
  anyway.
- **Each column name once.** A table with two columns of the same name (as
  `pd.concat(axis=1)` can make) raises before anything else, and so does a
  column named twice in `feature_cols` or `group_cols`. Features are an
  ordered list, not a set (whose order changes between runs) or a mapping (whose
  keys alone would be read), and must be real
  numbers (complex values raise).
- **One row per field.** Every field appears once in the FOV table, and
  every observation's field must be listed there.
- **Field identity.** A field is identified by its group columns together with
  its identifier. Well `J23` of plate `1A` and well `J23` of plate `2A` are two
  technical repeats, and `field_001.tif` in two wells is two fields.
- **Group values live where they are unambiguous.** `condition`,
  `biological_repeat` and `technical_repeat` may be only in the FOV table.
  They are then copied onto the observations, which is allowed only when the
  field identifier maps to a single value; when field names restart in each
  repeat, the copy goes through the group columns the observations do carry
  (the biological repeat and the field name together). A MorphoBricks column
  in the observations but not in the FOV table raises.
- **No missing identity or group values.** A missing value in a field
  identifier or a group column raises, because the observation's field or
  group would be unknown.
- **Empty optional columns count as absent.** Producers that write every
  column, as CellPatcher does, leave `position_id` empty when one file is one
  field and `technical_repeat` empty when there are no wells. A column that is
  empty in every row is therefore read as absent. When only the observation
  table's copy is empty, the FOV table's values are used. A column that is
  only partly empty still raises, since it is ambiguous. A column you name
  explicitly in `group_cols` or `fov_col` is always kept.
- **Empty fields are effort.** A field with zero observations is a real
  result: the microscope looked and found nothing. Keep it in the FOV table.
  A failed image (unreadable, failed segmentation) is not an empty field. Mark
  it in its own column, or leave it out and say so.
- **Planes are not fields.** When the field identifier is found automatically
  as `image_file` or `fov_id`, and each field holds a single `z`, `c` or `t`
  value (on the observations, or in the FOV table) that varies across fields,
  each "field" may be one plane of a position: with several observations per
  field this raises, with one row per field it is a warning. A FOV-table `z`
  or `t` counts only when one of its values recurs in several fields, since a
  stage height or an acquisition time distinct in every field is not a plane.
  File names that differ only by a plane, channel or frame token (`pos_z01.tif`
  and `pos_z02.tif`, Operetta's `...p01-ch1sk1...`) give a warning. Supply
  `position_id`, or pass `fov_col` if each file really is one position.
- **Planes within a field are not new observations.** When the rows of one
  field carry several values of `z`, `c` or `t` (or `z0`), an object measured
  in five planes would count as five objects, so the table is refused, unless
  `observation_unit` is `"slice"`, `"plane"` or `"frame"` (each row measures a
  whole plane). Turn each feature into one per plane with
  [`planes_as_features`](#features-measured-per-plane), keep one plane, or
  declare the plane-level unit.
- **One field is one stage position.** When the field identifier was chosen
  automatically and a position-like column (`series`, `scene`, `site`,
  `position`, with or without `Metadata_`) varies within a field, one file
  holds several positions (a `.nd2`, `.czi` or `.lif`), and the table is
  refused: supply `position_id`, one value per stage position.
- **Declare what a row is.** Pass `observation_unit` (`"cell"`, `"nucleus"`,
  `"patch"`, ...) when rows are not ordinary objects. The report's words follow it
  (tracks, patches, the patch-level test). Otherwise the rows are analysed as
  generic objects, every result records that the unit was assumed
  (`observation_unit_declared=False`), and the report calls them cells (owner,
  2026-10-02: "cells could be the default").

## Checking your tables

`check_tables` runs every rule above without stopping at the first problem.
It says what it understood, then lists each error and each warning with
how to fix it; `summary(detail="full")` also says where each one is (the
table, the column, how many rows and the first few), as does the error
`raise_if_errors` and every refusal raise:

```python
from morphocoverage import check_tables

check = check_tables(objects, fovs, feature_cols=["area", "intensity"],
                     observation_unit="cell", control={"condition": "MOCK"})
print(check.summary(detail="full"))
check.raise_if_errors()  # or: if check.ok: ...
```

```text
❌ MorphoCoverage data check — 2 errors, 1 warning; the tables cannot be analysed until the errors
   are fixed

What was understood
  - 6 observations (cell); 5 rows in the FOV table.
  - A field is identified by 'image_file' within condition, technical_repeat.
  ...

Errors (must be fixed)
  ❌ 1. Feature 'intensity' is not numeric.
       Where: observation table, 2 rows: row 0 (intensity='12,5'); row 3 (intensity='n/a').
       Fix: Convert the column to numbers (a decimal point, not a comma; empty cells for missing
       values).
  ...
```

The first line's marker is ❌ with an error, ⚠️ with warnings only and ✅
with neither.

Errors stop the analysis. They cover every rule above: misspelt MorphoBricks
columns, a missing or unmatched field identifier, empty identifiers or group
values, a field listed twice, observations of fields missing from the FOV
table, missing, non-numeric or infinite features, a column with two roles,
and planes or channels counted as fields. Where it can tell, an error says
what was probably meant:
- a label typed differently in the two tables (`' drug'` or `'Drug'`
  against `'drug'`);
- a number in one table against text in the other (`5` against `'05'`);
- a feature name that differs only in case or spaces (`'Area '` for
  `'area'`);
- the values that exist when the control matches nothing.

Warnings do not stop it but are often mistakes:
- no FOV table (the workflows need one);
- an undeclared observation unit;
- fields with no observation, so you can confirm that they are real empty
  fields and not failed images;
- an observation table with no rows;
- rows that are exact copies of another row (a file read twice; rows with no
  feature value, such as failed segmentations, are left out);
- observations that repeat another of the same field in every feature while
  another column differs (a merge with a table that has one row per channel,
  or a table appended to itself with a fresh index);
- an object label (`label_id`, `ObjectNumber`, `object_id` or `label`) twice
  in one field, and, for fields named by multi-position files (`.nd2`,
  `.czi`, `.lif`, ...), a `label_id` that repeats within one file;
- group labels within one table that differ only in spaces, case, zero
  padding or number against text (`'drug'`, `'drug '` and `'Drug'`; `'B2'`
  and `'B02'`; `1` and `'1'`), which are analysed as separate groups;
- a column that is not an analysis group but looks like one: constant within
  each field and splitting a group into blocks (an undeclared `well` or
  `plate` column, whose fields are then pooled);
- a technical repeat (a well) that holds no observation in any field, which
  was probably never imaged;
- a plate well named in several conditions of one repeat (a plate-map
  typo, or different plates with the same well names);
- features with missing values;
- repeats without the declared control (checked against the fields in the
  observations when there is no FOV table);
- one row per file with its own `z`, `c` or `t`, or file names that differ
  only by a plane token, which may be planes counted as fields;
- observations marked `valid=False` by CellPatcher.

`data/synthetic` holds a small acquisition whose answers are known, and
`tests/test_human_errors.py` breaks it in each of these ways; read the test
to see the message each mistake gives.

`validate_input_tables` and every workflow run the same check when they
refuse tables, so their error carries the full report after its first line.

**Features that repeat one measurement** (owner, 2026-10-05). The report's
"The data" also compares the declared features of the observation table pair
by pair, on the rows where both are finite. A copy, an exact affine relation
(`1 − x`, a change of units; the largest residual within 1e-4 of the column's
range) or a near copy (|Pearson r| ≥ 0.999) counts one measurement several
times in each test family, the heatmap and the discovery. Related features
are grouped and each group is said once as a ⚠️ caution, for example
"Porosity X (mean) and Porosity Y (mean) are copies of Porosity (mean), and
Porosity (mean) = 1 − HDM: one measurement counted 4 times in each test
family; keep one of them." Nothing is dropped or merged and no statistic
changes: declare the features again with one of each group. Constant
features are left to the few-distinct-values check, and other feature tables
are not compared, since their dimensions are expected to correlate. With
several controls the check also runs within each experiment: a control, the
conditions declared under it and its parent when one is linked. A relation
found in some experiments but not on all rows is said once, naming where it
holds: "in VEC and THBS5 only, Porosity (mean) = 1 − HDM; elsewhere they are
not one measurement, so the experiments may have been processed
differently." One that holds in every experiment, with other coefficients
in each, is said as such. One that holds on all rows is said once, not again per
experiment, and an experiment with fewer than 10 rows where both values are
finite is not checked for that pair (owner, 2026-10-05).

## Minimal example

```python
import pandas as pd
from morphocoverage import validate_input_tables

fovs = pd.DataFrame(
    {
        "condition": ["control"] * 3 + ["drug"] * 3,
        "biological_repeat": ["2026_01_01"] * 6,
        "technical_repeat": ["B02", "B02", "B03", "C02", "C02", "C03"],
        "image_file": ["f1.tif", "f2.tif", "f1.tif", "f1.tif", "f2.tif", "f1.tif"],
    }
)
# One row per cell; f2.tif of well C02 held no cell and appears only in fovs.
objects = pd.DataFrame(
    {
        "condition": ["control", "control", "control", "drug", "drug"],
        "biological_repeat": ["2026_01_01"] * 5,
        "technical_repeat": ["B02", "B02", "B03", "C02", "C03"],
        "image_file": ["f1.tif", "f2.tif", "f1.tif", "f1.tif", "f1.tif"],
        "area": [210.0, 188.5, 240.1, 305.2, 290.0],
    }
)
data = validate_input_tables(objects, fovs=fovs, feature_cols=["area"], observation_unit="cell")
print(data.group_cols)  # ('condition', 'biological_repeat', 'technical_repeat')
print(data.fov_key)  # ('condition', 'biological_repeat', 'technical_repeat', 'image_file')
print(len(data.fovs), "fields,", len(data.objects), "cells")  # 6 fields, 5 cells
```

## Other feature tables: embeddings beside the explainable features

The observation table holds the **explainable** features: each is tested one by
one (the discovery, the heatmap, the SuperPlots, "Did I image enough?" per
feature). Features that are not read one by one, such as the deep-learning
embedding of a DINO model, come as **other feature tables**, each with a name
(owner, 2026-10-05). Several can be loaded side by side, a DINOv2 and a DINOv3
embedding say, to compare the models. They are never tested feature by feature:
each is tested whole, beside the per-feature tests (owner, 2026-10-06,
[the whole-table test](../report/conditions.md#the-whole-table-test-other-feature-tables)),
and a dataset may have no explainable feature at all
([embedding-only reports](setup.md#embedding-only-reports-no-explainable-features)); they join the PCA and UMAP (one map per
table, and a weighted combined map first), the populations (searched in each
table, in the explainable features and in the combined space), the copy checks
(on each table apart) and their own fingerprints, never novelty or unusual
observations, which read the explainable features only (owner, 2026-10-06;
[Feature tables and maps](../report/feature_tables_and_maps.md)). Pass them by name to `report`, `html_report`, `scan_design`
and `guide` (`feature_tables={"DINOv2": dino}`); `attach_feature_tables` checks
them on their own.

One row of a table is one observation, identified as the observation table
identifies it: the field key columns the observation table holds (the analysis
groups and the field identifier, as the validator resolves them), then
`label_id` and `z`, `z0`, `z1`, `c`, `t` when the observation table has them.
Values are matched as text, a whole number written as a decimal read as that
whole number, so `1`, `1.0` and `"1"` are one label (a CSV column with one empty
cell is read as decimals; owner, 2026-10-06). The observations
must be told apart by these columns (several cells of one field need a
`label_id`). Each table must:

- hold every identity column and match every observation exactly once: no
  observation missing, no row matching none, no observation twice;
- hold numeric feature columns only: every column but the identity, none named
  like MorphoBricks metadata (`condition`, `z`, `label_id`, ...), and none
  without a single value;
- hold no infinite value; missing values are kept and counted, never imputed:
  each analysis that reads the table leaves out the observations missing one of
  its values, and says how many;
- use column names found neither in the observation table nor in another table;
- have a non-empty name of its own, other than `explainable` and `combined`
  (the names of the observation table's features and of the combined space),
  compared ignoring case.

Every problem is raised at once, naming the table; a column with one value
everywhere is accepted and listed, as no analysis can use it. In the combined
space, each table, robustly standardised, is divided by its own first singular
value (MFA; owner, 2026-10-06), so no table's first axis outweighs another's,
the explainable features keeping weight 1. An other table is never divided by a
control: an embedding has no ratio scale; its map centred on each repeat's
control is a view of the maps only.

```python
import numpy as np
import pandas as pd
from morphocoverage.workflows import attach_feature_tables

# One row per image; the DINO table names its images the same way.
key = {"condition": ["ctrl", "ctrl", "drug", "drug"], "biological_repeat": ["R1"] * 4,
       "image_file": ["a.tif", "b.tif", "a.tif", "b.tif"]}
objects = pd.DataFrame({**key, "area": [210.0, 190.0, 240.0, 236.0]})
fovs = pd.DataFrame(key)
dino = pd.DataFrame({**key, **{f"dino_{j}": np.arange(4.0) + j for j in range(3)}})
tables = attach_feature_tables(objects, fovs, {"DINOv2": dino})
print(tables.identity)  # ('condition', 'biological_repeat', 'image_file')
print(tables.columns["DINOv2"])  # ('dino_0', 'dino_1', 'dino_2')
print(tables.objects.shape)  # (4, 7)
```

## Features measured per plane

A feature extractor may measure every object once per z-plane, channel or
time point. The planes are not fields and their coordinate is not a feature,
so such a table is refused as it is. When the question is how an object
differs between planes, turn each feature into one feature per plane first,
then declare the new names (as `feature_cols`, or as `Setup.readout` and
`Setup.features`):

```python
import pandas as pd
from morphocoverage import planes_as_features, validate_input_tables

# Two cells, each measured in planes z = 0 and z = 1 of one field.
objects = pd.DataFrame(
    {
        "condition": ["control"] * 4,
        "biological_repeat": ["2026_01_01"] * 4,
        "position_id": ["P1"] * 4,
        "label_id": [1, 1, 2, 2],
        "z": [0, 1, 0, 1],
        "area": [210.0, 190.0, 240.0, 236.0],
    }
)
fovs = pd.DataFrame(
    {"condition": ["control"], "biological_repeat": ["2026_01_01"], "position_id": ["P1"]}
)
per_object, features = planes_as_features(objects, fovs, features=["area"])
print(features)  # ['area_z0', 'area_z1']
data = validate_input_tables(per_object, fovs=fovs, feature_cols=features, observation_unit="cell")
print(len(data.fovs), "field,", len(data.objects), "cells")  # 1 field, 2 cells
```

Objects are matched across planes by `label_id` within the field (`object_col`
names another identifier), so the same label must name the same object in
every plane, as a 3D segmentation or a 2D one tracked across planes gives.
Labels that restart in each plane without tracking do not name one object,
and nothing in the table can tell; the pivot refuses what it can see: a
missing identifier, a label twice in one plane of a field, or an object
missing from a plane that other objects were measured in. An object that
spans only some planes of a stack (or stacks of different depth) can be kept
with `allow_missing_planes=True` (opt-in): its features in the missing planes
are missing values (NaN), treated as any missing value. CellPatcher writes a
patch's plane as `z0` (with `z1 = z0 + 1`); without a `z` column, `z0` is the
`z` axis (names still say `area_z0`), and a patch spanning several planes is
refused. `axes=("z", "c")` splits along two axes (`area_z0_c1`). The FOV
table is unchanged, and each field stays one FOV.

## From a CellPatcher export

CellPatcher's `patches.csv` carries the shared columns: `condition`,
`biological_repeat`, `technical_repeat` (from a `<condition>/<repeat>/<well>/`
folder level, `single` without one), `position_id`, `image_file`, `mask_file`
and `patch_file`; a `patches.csv` written before the renaming names the
biological repeat `repeat`, which is still read (see
[The former name `repeat`](#the-former-name-repeat)). It can be read
directly as the observation table, one row per patch
(`observation_unit="patch"`), once a feature-extraction step has added
feature columns. For the FOV table, prefer the list of every image given to
CellPatcher. `fov_table_from_observations(patches, ["condition",
"biological_repeat", "technical_repeat", "image_file"])` is a fallback: it
loses images that gave no patch, such as images without any segmented cell
in `cell` mode.
