# Workflows: a new dataset in a few lines

`morphocoverage.workflows` runs a dataset through the four modes of the
[README](../README.md#modes): REPORT, OPTIMISE / PLAN, ONLINE and BENCHMARK.
You declare the dataset once, in a `Setup`. Each workflow then calls the
package's analyses in the order of [`questions.md`](questions.md) and returns
its results with a `summary()` written in sentences.

A worked example of all four, with plots, is the notebook
[`notebooks/MorphoCoverage_Workflows_Example`](../notebooks/MorphoCoverage_Workflows_Example/MorphoCoverage_Workflows_Example.ipynb).
It runs on the bundled idr0028 dataset or on your own CSV files, locally or on
Google Colab. It follows the lab's LabConstrictor portability guide: settings
are ipywidgets (not Colab forms), and Colab-only setup sits behind a Colab check.
Locally it needs `ipywidgets` and, for the plots, `matplotlib`.

The workflows make no scientific choice of their own. Every choice is either a
`Setup` field or an argument without a default. The analyses they call are
the ones documented elsewhere, and their results stay available under
`results`.

## Declare the dataset

```python
from morphocoverage.workflows import PositiveControl, Setup, default_novelty

setup = Setup(
    readout="YAPTAZ_Ratio",                  # the scalar you report
    features=features,                        # explicit columns; never inferred
    control={"condition": "MOCK"},            # negative-control groups
    positive=PositiveControl({"condition": "YAP"}, fovs_seen=10, pair_cols=("repeat",)),
    novelty=default_novelty(features),        # the definition of "new"; None to skip
    tolerances=(0.05, 0.025),                 # the readout's mean within 5% and 2.5%
    acquisition_order_col="acquisition_order",
    stage_cols=("stage_x_um", "stage_y_um"),
    group_cols=("condition", "repeat", "well"),
    observation_unit="nucleus",
    n_jobs=-1,
)
```

Every field left at `None` switches off the analyses that need it. The report
then says "Not assessed" and names what would be needed. The stopping rules
default to the JCB rule (upper-95 SD, first tolerance), per object and per
field; pass `rules=(...)` to assess others.

A declaration the workflows could not use raises instead of being skipped.
A definition of new (`novelty`) needs `features`, and a positive control needs
`novelty`. Group columns beyond condition and repeat (a `well`, say) are part of
a field's identity everywhere, including in the control check. "New" is judged
on `features` only, never on the readout, and the report names those columns.

## REPORT: what can the finished acquisition be trusted for?

```python
from morphocoverage.workflows import report

result = report(objects, fovs, setup)        # fovs lists every acquired field
print(result.summary())                      # the Foundation, questions 1 to 3 and 6
result.results["sufficiency"]                # each analysis, for further use
```

## OPTIMISE / PLAN: what should the next experiment do?

```python
from morphocoverage.workflows import document, optimise

plan = optimise(objects, fovs, setup, fov_counts=[2, 4, 8, 12], reservoir_fraction=0.35)
print(plan.summary())                         # questions 4, 5, 7, 8 and the rule calibration
print(document(result, plan).summary())       # one report with every question
```

`reservoir_fraction` has no default. It sets the share of each group's fields
held back as the reference that subsets are judged against, and that share is
a design choice. Planning advice rests on what random subsets preserved in
those held-back fields. It never rests on the best subset.

## BENCHMARK: which rule needs the fewest fields for the same risk?

```python
from morphocoverage.workflows import benchmark, default_families

bench = benchmark(objects, fovs, setup, reservoir_fraction=0.35)
print(bench.summary())
```

`default_families(setup.readout)` holds the rule families and threshold grids
of the locked protocol (version 2): the JCB rule and the JCS descriptor
spread, each per object and per field, and the FOV bootstrap of the mean.
Pass `families=` to compare others.

Tuning belongs on development data and scoring on held-out data
([`benchmarking.md`](benchmarking.md)). The workflow does not know a
dataset's role: keeping the two apart is up to you.

## ONLINE: continue or stop while the microscope acquires

`live_session(setup)` returns an `OnlineSession` for the setup's rules
([`online.md`](online.md)). It has two sources of fields: files on disk, or
tables held in memory. Both give the same decisions as calling `add_fov`
yourself with the same fields.

### Files on disk: `LiveFolder`

```python
from morphocoverage.workflows import LiveFolder, live_session, write_fov

# acquisition side, once per completed field
write_fov(folder, field_objects, {"condition": "MOCK", "repeat": "1A", "image_file": "002006-1.tif"})

# analysis side
live = LiveFolder(folder, live_session(setup))
live.watch(interval=1.0, until_stop=True)    # or live.poll() from your own loop
```

The folder holds one object file per field, `objects/<name>.csv`, plus an
append-only manifest, `fovs.csv`, with one line per completed field:

1. `write_fov` writes the object file under a temporary name.
2. It renames the file into place with `os.replace`, which is atomic. A
   reader therefore never sees half a field.
3. Only then does it append the manifest line, with `fsync`.

A field is read only once its manifest line exists.

The watcher remembers its byte offset in the manifest, so each poll reads
only the new lines. A last line that is not yet terminated waits for the next
poll.

A field with no objects is either an empty object file or an empty
`objects_file` entry. It still counts as acquisition effort. An
`objects_file` that names a missing file raises an error; it is never read as
an empty field.

Metadata are read as text by default, so an identifier such as `001` keeps
its leading zeros. Pass `metadata_dtypes=` to cast a column to another type.

### Tables in memory: `LiveTables`

When features are computed live and never written to disk, pass the two
growing tables instead:

```python
from morphocoverage.workflows import LiveTables, live_session

live = LiveTables(live_session(setup))
# after each field's features are computed and appended to both tables:
for update in live.update(objects, fovs):
    print(update.summary())
```

Each call reads only the rows appended since the previous call. As on disk,
the FOV row signals that a field is complete:

- **Order.** A field's objects must be appended before its FOV row, or in the
  same call. Objects that arrive before their FOV row wait in a buffer
  (`pending_objects`).
- **Late objects.** An object row for a field that has already been added
  raises an error. Accepting it would change a field the rules have already
  judged.
- **Growth only.** The tables may only grow. Removing rows that have already
  been read raises an error.

`key` names the columns that link an object row to its field. By default it
is the session's FOV column plus the condition and repeat columns present in
both tables. A key column holding numbers in one table and text in the other
(`1` against `"1"`) raises an error, because those values never match. Objects
still waiting for their FOV row are counted in each update's summary.

**A field that fails is retried, never skipped** (both sources). A source
moves past a field only once the session has accepted it. If a field raises
(an infinite value, a missing file), the updates so far are kept, and the next
poll or update raises on the same field again. No later field is added past it,
so acquisition effort is never under-counted.

## File format and chunks

The question: for the live folder, is CSV the right format, and must a field
be split into chunks?

**The chunk is the field.** Every object of a field enters the analysis
together (AGENTS.md section 3), so the field is also the natural unit of
writing and reading. One growing object table would be a poor alternative, for
three reasons:

- it gives no signal that a field is complete;
- it can be read while half written;
- it must be read again, or tailed, as it grows. Reading a 300-field table of
  60 features (49 MB) again takes about 0.8 s, and that time grows with every
  field.

With one file per field, each read stays the size of one field. The read
times below were measured in this container with `pandas.read_csv`:

| One field | CSV read |
| --- | --- |
| 300 objects x 60 features | about 5 ms |
| 3,000 objects x 60 features | about 36 ms |
| 3,000 objects x 400 features (embedding-sized) | about 190 ms |

All of these fit comfortably in the seconds between fields, so no chunking
within a field is needed.

**CSV is enough.** It is what the feature extractors write, it needs no new
dependency, and you can open it by hand. A binary format such as Parquet
(which needs `pyarrow`, not a dependency) pays off only for very wide rows:

- `write_fov(..., suffix=".parquet")` writes Parquet, and
  `LiveFolder(..., reader=pandas.read_parquet)` reads it;
- `suffix=".csv.gz"` writes compressed CSV.

In memory, `LiveTables` reads nothing from disk, so the format question does
not arise there.

## Run time on real data

These times come from idr0028 screen B: 12 wells, 28 fields each, 30
features, 4 workers. They are approximate.

| Workflow | Settings | Time |
| --- | --- | --- |
| `report` | defaults | about 45 s |
| `optimise` | `fov_counts=[2, 4, 8, 12]` | about 20 s |
| `benchmark` | `n_boot=200` | about 85 s |

In the live runs, one field took about 1 ms to read and a few milliseconds to
decide.
