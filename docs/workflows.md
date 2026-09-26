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

The workflows' scientific choices are the `Setup` fields and explicit
arguments. Two provisional presets have defaults, `optimise`'s
`protocol_threshold=0.9` and `benchmark`'s `alphas=(0.05, 0.10)`, and the
summaries print them. The analyses they call are the ones documented
elsewhere, and their results stay available under `results`.

## Declare the dataset

```python
from morphocoverage.workflows import PositiveControl, Setup, default_novelty

setup = Setup(
    readout="YAPTAZ_Ratio",                  # the scalar you report
    features=features,                        # explicit columns; never inferred
    control={"condition": "MOCK"},            # negative-control groups
    normalize="control_mean",                 # or None: divide by the repeat's control mean
    positive=PositiveControl({"condition": "YAP"}, fovs_seen=10, pair_cols=("repeat",)),
    novelty=default_novelty(features),        # the definition of "new"; None to skip
    tolerances=(0.05, 0.025),                 # the readout's mean within 5% and 2.5%
    acquisition_order_col="acquisition_order",
    stage_cols=("stage_x_um", "stage_y_um"),
    group_cols=("condition", "repeat", "technical_repeat"),  # the well: a technical repeat
    observation_unit="nucleus",
    n_jobs=-1,
)
```

Every field left at `None` switches off the analyses that need it. The report
then says "Not assessed" and names what would be needed. The stopping rules
default to the JCB rule (upper-95 SD, first tolerance), per object and per
field; pass `rules=(...)` to assess others.

### Normalising to the control

`normalize="control_mean"` divides every declared column (the readout and
the features) by the mean of the control condition's objects of the same
repeat before any retrospective analysis, as the JCB rule and the lab's
Plot&Stats notebook express values (`morphocoverage.normalize`). It needs
`control` to name the condition column, every repeat must hold control
objects (a repeat without them raises, since mixing scales inside one
analysis would be worse than stopping), and the report's overview and the
declaration say that the values are normalised. The normalised values are used
only where values are compared across repeats: the outcome criteria judged on
held-back fields, the condition contrast, the feature ranking and the plots.
Every stopping rule runs on the raw values, in `report`, `optimise` and
`benchmark` as in the live session (which sees one field at a time and has not
seen the control), so a rule gives the same verdict everywhere; the other
analyses (variability, controls, inspection, effort projection) read the raw
values too.

### What must the acquisition preserve?

The `Setup` also declares the target, the front door of
[`questions.md`](questions.md): there is no "enough" without "enough for
what". Three targets can be declared, and each becomes an outcome criterion
that `optimise` and `benchmark` judge on held-back fields (`setup.criteria()`):

- **a population summary**: the mean of `readout` within each of
  `tolerances` (always declared);
- **a phenotypic fingerprint**: `fingerprint=(OutcomeCriterion(...), ...)`,
  distribution criteria on the declared features, for example

  ```python
  from morphocoverage.multivariate_outcomes import CovarianceError, QuantileError
  from morphocoverage.retrospective import OutcomeCriterion

  fingerprint = (
      OutcomeCriterion(CovarianceError(features, True), 0.10, "lower_is_better",
                       "the covariance of the features within 10%"),
  )
  ```

  Their features must be in `features`. No tolerance here has a published
  meaning: state yours, and read the reservoir ceiling the curve prints
  (held-back fields are themselves a sample; a criterion they cannot pin
  down says "cannot be read above" the ceiling rather than "more fields do
  not help");
- **a rare phenotype**: `rare=RareTarget(...)` (its definition, its object
  prevalence and/or field incidence, the detection probability, the fields it
  must be seen in) and, when an independent label exists, `rare_label`, the
  object column (booleans or 0/1) that marks it. With the label, held-back
  fields judge whether `k` fields showed the state; without it the target is
  stated and planned for (the binomial field count at the declared incidence)
  but not judged. Object prevalence is never converted to field incidence.

Every document opens with the declaration (`setup.declaration()`), one
sentence per target, and a target that is not declared says so, so that an
answer about the mean is never read as an answer about the distribution or
about rare states. A fourth target, the difference between conditions, is a
retrospective benchmark outcome and not yet a declaration
([`questions.md`](questions.md), open work).

A declaration the workflows could not use raises instead of being skipped.
A definition of new (`novelty`) needs `features`, and a positive control needs
`novelty`. Group columns beyond condition and repeat (the `technical_repeat`, say) are part of
a field's identity everywhere, including in the control check. "New" is judged
on `features` only, never on the readout, and the report names those columns.
The stopping rules see the same columns (the readout, then `features`) in
`report`, `optimise`, `benchmark` and the live session, so one rule gives one
verdict whichever workflow runs it.

## REPORT: what can the finished acquisition be trusted for?

```python
from morphocoverage.workflows import report

result = report(objects, fovs, setup)        # fovs lists every acquired field
print(result.summary())                      # overview, declaration, Foundation, questions 1 to 3, 6, 9, limits
result.results["sufficiency"]                # each analysis, for further use
```

Every document opens with **what is in the dataset**: objects, acquired
fields and empty fields, what one row is, the feature columns (the first 12
names, then how many more), the conditions and repeats with a table of fields
by condition and repeat (and of technical repeats, when the column exists),
and whether the values are normalised. It closes
with **what this dataset cannot tell me**: every question not assessed, with
what it would need (a control, a positive control, stage coordinates, a
hierarchy, a replay), and the notes to read the answers with. Missing
columns never stop the report: without `condition` or `repeat` every field is
one group and the sections that need them say so.

## Plots: every object a point, the interval from fields

```python
from morphocoverage.workflows import plots

figures = plots(objects, fovs, setup, folder="figures")   # the readout by default
figures = plots(objects, fovs, setup, features=["YAPTAZ_Ratio", "Nuclei_Area"])
```

PlotsOfData-style figures (`morphocoverage.plots`), one per feature:
within each condition the biological repeats (the `repeat` column) sit side
by side, labelled above their cluster, so that a difference between
conditions can be read against the spread between repeats. Every object is
a jittered point coloured by technical repeat (`technical_col`, by default
the MorphoBricks `technical_repeat` column, else the group column below the
repeat: a well is a technical repeat, never a biological one), and each
biological repeat's median (or `summary="mean"`) carries its 95% interval
from resampling that repeat's technical repeats and then their whole fields
(whole fields alone when there is one technical repeat), never objects, so
it reflects how the data were sampled: a well-to-well shift widens the
interval instead of passing for field-to-field noise. More than 12 conditions
are split over several figures, more than 8 biological repeats are pooled
into one cluster per condition and the title says so, more than 8
technical repeats go uncoloured with a note, and more than 24 features
raise so that you choose; all four are arguments (`max_conditions`,
`max_repeats`, `max_technical`, `max_features`). The features must be
declared in the setup, and the plots use the same normalised values as the
report when `normalize` is set. With `folder`, each figure is saved as a PNG.
Without `features`, the readout is plotted and, given the report's ranking
(`plots(objects, fovs, setup, ranking=result.ranking)`), the five
best-ranked other features (`top=`): those that separate the conditions
beyond chance, as question 9 describes below.

When `control` names a condition and the FOV table holds other conditions,
the report also answers question 9, "Do the conditions differ, and how
confidently?": each condition against the control, per repeat, with Cohen's
d and the JCS effect size computed on objects (the numbers of the lab's
Plot&Stats notebook) and with the field as the unit, and a randomization test
that shuffles condition labels over whole technical repeats when each side
has at least two (a `technical_repeat` column among the group columns), else
over whole fields. It describes what these fields support; generalizing
needs independent repeats. With several declared features the section also
ranks them (`sampling_unit.feature_ranking`), as fiberfeatures scores a
feature set: the same test for every feature and contrast, Benjamini-Hochberg
corrected together, a gate at q <= 0.05, and the passing features ordered by
the largest |d| on field means. The ranking says which features are worth
plotting; with two technical repeats a side no test can pass the gate, and
the section says so.

## OPTIMISE / PLAN: what should the next experiment do?

```python
from morphocoverage.workflows import document, optimise

plan = optimise(objects, fovs, setup, fov_counts=[2, 4, 8, 12], reservoir_fraction=0.35)
print(plan.summary())                         # questions 4, 5, 7, 8 and the rule calibration
print(document(result, plan).summary())       # one report with every question
```

When the FOV table has a `technical_repeat` column and no `hierarchy` is
declared, `optimise` derives one (the field within the technical repeat
within the repeat, one stratum per condition) and answers question 7, "more
fields or more technical repeats?", from the variance between technical
repeats and between fields at equal total fields; where every repeat holds a
single technical repeat the repeat cannot be a variance level and the answer
says so. On idr0028, with 28 fields per well, most of the uncertainty of a
condition's mean sits between wells wherever there are two to see it.
A declared `hierarchy` also keeps one stratum per condition unless
`effort_strata` names others; `effort_strata=()` pools the conditions, as an
explicit request. Protocol reliability across samples needs every sample and
stratum to be an analysis group: when `group_cols` pool one (the repeat, the
technical repeat, the condition), that part of question 4 says "not assessed"
and why, as question 7 does when `group_cols` leave the technical repeat out.

`reservoir_fraction` has no default. It sets the share of each group's fields
held back as the reference that subsets are judged against, and that share is
a design choice. Planning advice rests on what random subsets preserved in
those held-back fields. It never rests on the best subset. Every declared
criterion gets its own curve: the mean within each tolerance, each fingerprint
criterion, and the labelled rare state, so the answer to "how many fields?"
is read per target, and a target the held-back fields cannot judge says so.

## BENCHMARK: which rule needs the fewest fields for the same risk?

```python
from morphocoverage.workflows import benchmark, default_families

bench = benchmark(objects, fovs, setup, reservoir_fraction=0.35)
print(bench.summary())
```

`default_families(setup.readout)` holds the rule families and threshold grids
of the locked protocol (version 2): the JCB rule and the JCS descriptor
spread, each per object and per field, and the FOV bootstrap of the mean
(on the readout only, whatever other features are declared).
Pass `families=` to compare others.

Tuning belongs on development data and scoring on held-out data
([`benchmarking.md`](benchmarking.md)). The workflow does not know a
dataset's role: keeping the two apart is up to you.

## ONLINE: continue or stop while the microscope acquires

`live_session(setup)` returns an `OnlineSession` for the setup's rules
([`online.md`](online.md)), under the objective made of the declared targets
(`setup.objective()`). Each update ends with one view per target: the
population summary, the phenotypic fingerprint and the rare phenotype, each
with the decision of the rules that decide it, then the overall decision. The
default rules (the JCB rule per object and per field) decide the population
summary only; declare a fingerprint or a rare target and the view says that
no rule decides it until you pass `rules=(...)` with one that does (for
example a `FOVBootstrapStrategy` on the covariance), and the overall decision
stays unsupported rather than assuming the target adequate. It has two sources of fields: files on disk, or
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
is the session's FOV column plus its group columns present in both tables:
condition, repeat and, when it holds values, `technical_repeat`. A key column holding numbers in one table and text in the other
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
