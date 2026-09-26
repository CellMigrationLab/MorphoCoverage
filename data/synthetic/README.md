# A synthetic acquisition with known answers

`make_synthetic.py` builds a small acquisition from declared parts, so every
answer MorphoCoverage gives on it can be checked against the truth.

## What was built

- **Design:** 2 conditions × 3 biological repeats × 5 wells × 6 fields, so 180
  fields.
- **Tables:** in the MorphoBricks format (`docs/data_format.md`):
  - `synthetic_fovs.csv`: one row per acquired field;
  - `synthetic_cells.csv`: one row per cell, 2,421 cells.
- **Empty fields:** field 4 of every well holds no cell, so 30 fields are
  empty.
- **Readout:** `area` is built from a repeat, well, field and cell effect of
  variance 9, 16, 25 and 100, and the drug adds 15.
- **Null features:** `intensity` and `eccentricity` have no condition effect.
- **Rare state:** 15 rare cells were planted in 3 fields of one drug well, and
  the `rare` column labels them.

`expected.json` holds these facts. It also holds what follows from them: the
intraclass correlation within a well (0.2), the design effect of whole fields,
the share of fields that differ more than random cells would, and the effect
size on cells.

## How it is used

- `tests/test_end_to_end.py` runs `check_tables`, `report`, `optimise`, a
  live session fed through `write_fov`, `plots` and the normalisation. Each
  answer is compared with the truth.
- `tests/test_human_errors.py` then makes about forty mistakes people make
  with such tables. Each is one of three kinds:
  - **refused:** the mistake is refused with a message that says what is
    wrong;
  - **warned:** the tables run, with a warning;
  - **harmless:** the answers are unchanged.

To regenerate the files, run this from the repository root:

```bash
python data/synthetic/make_synthetic.py
```

The first end-to-end test checks that the committed files are exactly what
the generator builds.
