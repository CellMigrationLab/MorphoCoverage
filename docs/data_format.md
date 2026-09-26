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
`morphocoverage.check_tables` (see [Checking your tables](#checking-your-tables)).

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

## Optional columns

| Column | Table | Type | When present |
| --- | --- | --- | --- |
| `condition` | both, or FOV table only | string | Experimental group. A default analysis group and part of field identity. Needed to declare a control (`Setup.control`), to normalise to it, and for the condition contrast. |
| `repeat` | both, or FOV table only | string | Biological repeat, `YYYY_MM_DD` or `single` in MorphoBricks; any label is accepted. A default analysis group and part of field identity. Normalisation and the condition contrast work within each repeat. |
| `technical_repeat` | both, or FOV table only | string | Technical repeat within the biological repeat: a well, a dish, a coverslip. Never a biological repeat. A default analysis group and part of field identity. The overview counts technical repeats. Plots colour points by them and resample them before fields. The contrast shuffles whole technical repeats when each side has at least two. `Setup` derives the hierarchy "field within technical repeat within repeat" from it, for "more fields or more technical repeats?". `single` means one technical repeat per repeat. |
| other group columns | both, or FOV table only | any | Named with `group_cols`, for example `batch`. Each combination of group values is analysed apart. |
| `z`, `c`, `t` (or `z0`, `z1`) | observations | integer | Plane, channel and time point within one position. Required when one position spans several files, so the files can be told apart. Never features and never new fields. |
| acquisition order | FOV table | integer | Named with `acquisition_order_col` (MorphoBricks suggests `acquisition_order`). The recorded order in which fields are replayed. Without it only random orders are replayed. |
| stage coordinates | FOV table | numeric | Named with `stage_cols` (MorphoBricks suggests `stage_x`, `stage_y`). Used for dispersed orders, stage travel and redundancy distance. |
| imaged area | FOV table | numeric | Named on `FixedAreaStrategy`. The physical area of each field, for area-based stopping. |
| a rare-state label | observations | boolean or 0/1 | Named with `Setup.rare_label`. An independent label of the rare phenotype, so held-back fields can judge whether it was seen. |
| `label_id`, `mask_file`, `patch_file` | observations | any | Provenance, kept for tracing an observation back to its image; never features. |

## Rules the validator enforces

- **Exact names.** Column names are lowercase and exact. A near miss raises,
  so a misspelt group cannot silently merge repeats: `Repeat`,
  `Technical_Repeat`, `technical repeat` (a space for the underscore),
  `condition ` (a trailing space), or a byte-order mark that a spreadsheet
  wrote before the first header. Name the column in `group_cols` or
  `fov_col` to use it anyway.
- **One row per field.** Every field appears once in the FOV table, and
  every observation's field must be listed there.
- **Field identity.** A field is identified by its group columns together with
  its identifier. Well `J23` of plate `1A` and well `J23` of plate `2A` are two
  technical repeats, and `field_001.tif` in two wells is two fields.
- **Group values live where they are unambiguous.** `condition`, `repeat` and
  `technical_repeat` may be only in the FOV table. They are then copied onto
  the observations, which is allowed only when the field identifier maps to a
  single value.
- **No missing identity or group values.** A missing value in a field
  identifier or a group column raises, because the observation's field or
  group would be unknown.
- **Empty optional columns count as absent.** Producers that write every
  column, as CellPatcher does, leave `position_id` empty when one file is one
  field and `technical_repeat` empty when there are no wells. A column that is
  empty in every row is therefore read as absent. A column that is only
  partly empty still raises, since it is ambiguous. A column you name
  explicitly in `group_cols` or `fov_col` is always kept.
- **Empty fields are effort.** A field with zero observations is a real
  result: the microscope looked and found nothing. Keep it in the FOV table.
  A failed image (unreadable, failed segmentation) is not an empty field. Mark
  it in its own column, or leave it out and say so.
- **Declare what a row is.** Pass `observation_unit` (`"cell"`, `"nucleus"`,
  `"patch"`, ...) when rows are not ordinary objects. Otherwise every sentence
  says "object" and records that the unit was assumed.

## Checking your tables

`check_tables` runs every rule above without stopping at the first problem.
It says what it understood, then lists each error and each warning with
where it is (the table, the column, how many rows and the first few) and
how to fix it:

```python
from morphocoverage import check_tables

check = check_tables(objects, fovs, feature_cols=["area", "intensity"],
                     observation_unit="cell", control={"condition": "MOCK"})
print(check.summary())
check.raise_if_errors()  # or: if check.ok: ...
```

```text
MorphoCoverage data check: 2 errors, 1 warning; the tables cannot be analysed until the errors are fixed.

What was understood
  - 6 observations (cell); 5 rows in the FOV table.
  - A field is identified by 'image_file' within condition, technical_repeat.
  ...

Errors (must be fixed)
  1. Feature 'intensity' is not numeric.
     Where: observation table, 2 rows: row 0 (intensity='12,5'); row 3 (intensity='n/a').
     Fix: Convert the column to numbers (a decimal point, not a comma; empty cells for missing values).
  ...
```

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
- rows that are exact copies of another row (a file read twice), or a
  `label_id` twice in one field;
- features with missing values;
- repeats without the declared control.

`data/synthetic` holds a small acquisition whose answers are known, and
`tests/test_human_errors.py` breaks it in each of these ways; read the test
to see the message each mistake gives.

`validate_input_tables` and every workflow run the same check when they
refuse tables, so their error carries the full report after its first line.

## Minimal example

```python
import pandas as pd
from morphocoverage import validate_input_tables

fovs = pd.DataFrame(
    {
        "condition": ["control"] * 3 + ["drug"] * 3,
        "repeat": ["2026_01_01"] * 6,
        "technical_repeat": ["B02", "B02", "B03", "C02", "C02", "C03"],
        "image_file": ["f1.tif", "f2.tif", "f1.tif", "f1.tif", "f2.tif", "f1.tif"],
    }
)
# One row per cell; f2.tif of well C02 held no cell and appears only in fovs.
objects = pd.DataFrame(
    {
        "condition": ["control", "control", "control", "drug", "drug"],
        "repeat": ["2026_01_01"] * 5,
        "technical_repeat": ["B02", "B02", "B03", "C02", "C03"],
        "image_file": ["f1.tif", "f2.tif", "f1.tif", "f1.tif", "f1.tif"],
        "area": [210.0, 188.5, 240.1, 305.2, 290.0],
    }
)
data = validate_input_tables(objects, fovs=fovs, feature_cols=["area"], observation_unit="cell")
print(data.group_cols)  # ('condition', 'repeat', 'technical_repeat')
print(data.fov_key)  # ('condition', 'repeat', 'technical_repeat', 'image_file')
print(len(data.fovs), "fields,", len(data.objects), "cells")  # 6 fields, 5 cells
```

## From a CellPatcher export

CellPatcher's `patches.csv` carries the shared columns: `condition`,
`repeat`, `technical_repeat` (from a `<condition>/<repeat>/<well>/` folder
level, `single` without one), `position_id`, `image_file`, `mask_file` and
`patch_file`. It can be read directly as the observation table, one row per
patch (`observation_unit="patch"`), once a feature-extraction step has added
feature columns. For the FOV table, prefer the list of every image given to
CellPatcher. `fov_table_from_observations(patches, ["condition", "repeat",
"technical_repeat", "image_file"])` is a fallback: it loses images that gave
no patch, such as images without any segmented cell in `cell` mode.
