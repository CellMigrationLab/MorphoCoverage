# The live session: continue or stop while the microscope acquires

LIVE answers "I am imaging now. Should I acquire another field?" from the fields
acquired so far, and nothing else. Use it on the microscope, after a report or a plan has
told you what to stop on. This page is `live_session` with its two sources of fields, a
folder and tables in memory, and its questions O1 to O9. What one update means for the
underlying `OnlineSession` is in [Online acquisition](online_acquisition.md); how a stop
is read and checked is in [Stopping rules](stopping_rules.md).

## The questions

Deliberately narrower: the microscope knows only what it has acquired, and
the question is continue or stop for representative acquisition. "Where
should I image next?" is not part of it; that is the later targeted-sampling
extension, which changes the sampling distribution and needs its own design.

| # | Question | Answered by | Status |
|---|---|---|---|
| O1 | **Should I continue or stop after this FOV?** | `OnlineSession.add_fov` | done |
| O2 | **Why?** A short explanation, such as "population representation looks adequate, but the rare-state objective is not yet met". | `StrategyResult.explain()`: the rule, its evidence and the effort in one line; every stop ends with what it rests on, the acquisition-order check (`order_check`) or the assumption that the fields are a random sample of the well's positions | partial: one line per rule; no combined sentence across objectives |
| O3 | **Which objective is driving the decision?** The population summary, the phenotypic fingerprint, the rare phenotype, or several. | `component_decisions` of each result; `rule_decision` for what the rule itself said | done |
| O4 | **If the objectives disagree, is that obvious?** For example: "Population summary: stop / Phenotypic fingerprint: continue / Overall: continue". | `results.decision_view` over the rules of one acquisition state, printed by every live update | done; a target no rule decides is printed as such |
| O5 | **How uncertain is the evidence?** A borderline quantity must not become an unconditional STOP. | upper bounds on some quantities (novelty bound, upper-95 SD) | gap: no margin or "borderline" state in the decision |
| O6 | **How does this stopping rule perform on previously acquired datasets?** When this exact rule said STOP in replayed deep datasets, how often was it adequate? This is PLAN 6 seen from the microscope. | `stopping_calibration` of the same `config_id` | partial: not attached to the live session. **Benchmark** |
| O7 | **Can the decision be made fast enough?** Is the analysis faster than the time before the next acquisition? | `Timing` on every result; measured latencies in `docs/online.md` | done |
| O8 | **Does it behave correctly in awkward real acquisitions?** Empty FOV, few objects, missing feature values, patches rather than cells, z/c/t of one position, unsupported estimators, a new condition or repeat, changed grouping or FOV identity. | input validation and the session contract; contract tests | done |
| O9 | **Is there any future-data leakage?** Scaling, PCA, thresholds and other preprocessing must use only FOVs already acquired. | transforms fitted inside `estimate`; the no-future-leakage contract | done |

## `live_session`

`live_session(setup)` returns an `OnlineSession` for the setup's rules
([`online.md`](online.md)), under the objective made of the declared targets
(`setup.objective()`), or the narrower `objective=` passed (for example
`ObjectiveConfig("rare_discovery", rare=target)`, to stop on a rare phenotype alone). Each update ends with one view per target: the
population summary, the phenotypic fingerprint and the rare phenotype, each
with the decision of the rules that decide it, then the overall decision. The
default rules (the mean-precision rule per object and per field; per field only for
observations that are not objects, such as patches or tracks) decide the population
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
write_fov(folder, field_objects, {"condition": "MOCK", "biological_repeat": "1A", "image_file": "002006-1.tif"})

# analysis side
live = LiveFolder(folder, live_session(setup))
live.watch(interval=1.0, until_stop=True)    # or live.poll() from your own loop
```

`until_stop=True` returns once the objective's overall decision is stop in
every group seen so far, each group judged by its latest field: on a plate
acquired well by well, the watch goes on while any well still needs fields.
`until_stop="latest_group"` returns as soon as the latest field's group is
done. The overall decision is the last line of each update's view, as in
`benchmark`: a declared target that no rule decides is unsupported and keeps
the watch going, so the watch never stops while the view says otherwise
(owner, 2026-10-05; before, the watch read each rule's own verdict).
`watch(..., stop_on="rule")` is the lower-level stop when every rule's own
verdict is stop, whatever the targets it does not decide. Fields are added one at a time, so a backlog stops at the stop: the
fields still waiting in the folder are not added, and the next `poll` or
`watch` reads them.

The folder holds one object file per field, `objects/<name>.csv`, plus an
append-only manifest, `fovs.csv`, with one line per completed field:

1. `write_fov` writes the object file under a temporary name.
2. It renames the file into place with `os.replace`, which is atomic. A
   reader therefore never sees half a field.
3. Only then does it append the manifest line, with `fsync`.

A field is read only once its manifest line exists.

The watcher remembers its byte offset in the manifest, so each poll reads
only the new lines. A last line that is not yet terminated waits for the next
poll. The manifest may only grow: one rewritten, truncated or replaced under
the watcher raises rather than skipping fields. A line identical to one
already read (an acquisition that retried) is skipped with a warning; a line
naming an accepted field with another object file raises, naming both lines.
A field whose rows repeat another row exactly, or that holds a `label_id`
twice (an object file appended twice), is added with a warning, as
`check_tables` warns.

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
is the field identity, resolved with the validator's own resolvers: the
session's group columns, then the MorphoBricks condition, repeat and, when it
holds values, `technical_repeat` present in both tables, then the FOV column,
so well `B02` of two repeats is two fields even when `group_cols` pool the
repeats. It is resolved once the FOV table has a row, so objects appended
before the first FOV row wait without fixing it. A key that names two fields
is refused before any field of the update is added: a FOV-table group column
the key leaves out holding two values for one key (file names that restart in
each well, with no well column in the key), or a key twice. A key column holding numbers in one table and text in the other
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

## Run time

In the live runs, one field took about 1 ms to read and a few milliseconds to
decide.
