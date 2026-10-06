# Running a report

You have finished imaging and want to know what the acquisition can be trusted for.
This page is how to get the report: scan the tables, answer the questions of the guided
session or declare a `Setup` yourself, run `report`, and write the HTML page. What each
section of the report answers is in [the report's questions](questions.md), in the order
of the page. The `Setup` itself is described in [the setup](../common/setup.md).

## Scan the design first

Before declaring anything, `scan_design` reads the tables as given and says
what they are, whether they were loaded correctly, which columns would say
more, and what to answer before the analysis:

```python
from morphocoverage.workflows import report, scan_design

scan = scan_design(objects, fovs)          # everything else may be left undeclared
print(scan.summary())                      # summary(detail="full") adds the checks in full
setup = scan.setup(readout="area", control={"condition": "siCTRL"}, observation_unit="cell")
result = report(scan.objects, scan.fovs, setup)
```

- **What the dataset is**: the conditions (the control, declared or named
  like one: siCTRL, MOCK, DMSO, NT, scramble, vehicle), the biological repeats
  and which conditions each holds (one experiment, or separate acquisitions
  sharing the control, as in a screen), the technical repeats, the fields and
  the observations per field, the planes, the targets, the features, the
  [other feature tables](feature_tables_and_maps.md#other-feature-tables-embeddings) given as
  `scan_design(..., feature_tables={"DINOv2": dino})`, each checked (owner,
  2026-10-05: one that cannot be used is a ❌ finding, with why), and the
  observations per condition and repeat. With a FOV table that passes the data
  check, the fields, conditions and repeats are those it lists, a condition or
  repeat held only by empty fields included, with 0 observations.
- **Loaded correctly?** The data check (`check_tables`), the copy checks
  (`integrity_checks`: a field read twice, rows pasted into another field,
  a change of units), declared features holding one value in every row, and
  the balance over the biological repeats. Unbalanced repeats (more than 25%
  apart) are stated, since the report then tests the observations both as
  provided and balanced; a sparse repeat (under a quarter of the median
  repeat, or under 5) is a caution, since the balanced test leaves it out.
  The usual design (three repeats, one well each) draws no caution.
- **Columns that would say more**: an image identifier (`image_file` or
  `position_id`; without one each row is assumed to be its own image, and the
  scan adds a `position_id` per row so the workflows run); a `target` when
  several conditions are compared, with the conditions whose names look like
  reagents of one target (`siCCT8#5` and `siCCT8#8`); a `technical_repeat`
  when a condition was imaged in more than one well; a FOV table; optionally
  the acquisition order and stage coordinates, for planning.
- **Before the analysis**: the goal (compare population averages, the
  default; a phenotypic fingerprint, declared as `fingerprint=...`; a rare
  population, declared as `rare=RareTarget(...)`), the readout and, when they
  were not declared, the control, the features and the observation unit.
  Without `feature_cols`, numeric columns that are not metadata are read as
  candidates for the copy checks only.

`scan.setup(readout, goal=..., **fields)` builds the `Setup` from the scan's
features, field identifier, groups and unit; a guessed control is never used
without being declared, and a fingerprint or rare goal needs its criterion,
whose tolerance has no default. Nothing in the scan is estimated or tested.

**An embedding-only report** (owner, 2026-10-06): when the observation table holds no
numeric feature and other feature tables are given (a table of images and a DINO
embedding), the scan checks the tables' columns, and `scan.setup(None, goal="fingerprint"
or "rare", ...)` builds a Setup without a readout
([the setup](../common/setup.md#embedding-only-reports-no-explainable-features)). The report
then answers "Do the conditions differ?" with the
[whole-table test](conditions.md#the-whole-table-test-other-feature-tables) of each table,
"Did I image enough?" with the fingerprint of each table and the labelled phenotype, and
says once per section what needs explainable features ("not applicable: no explainable
features", ➖). In every report with other feature tables, each table is also tested whole
beside the per-feature tests, and the page saves the tests as `tests_whole_table.csv`.

### A guided session

`guide()` asks those questions for you, one at a time, and keeps the answers:

```python
from morphocoverage.workflows import guide

session = guide("cells.csv", folder="results")   # paths or DataFrames; asked for when left out
```

or, from a terminal, `python -m morphocoverage.workflows cells.csv [fovs.csv] --folder results`.
The session loads the tables; given as paths (asked for, or from a terminal), it then
asks "Do you have other feature tables for the same observations, e.g. deep-learning
embeddings such as DINO?" and, for each, a path and a name, checking each against the
observation table and asking again when one cannot be used (owner, 2026-10-05;
`guide(..., feature_tables={"DINOv2": dino})` gives them directly). It scans them,
the other tables listed in the scan, and prints the scan, then asks: did
the dataset load correctly (if not, it stops); what one row is (owner, 2026-10-05: asked
here, so the scan's words follow it); whether all the conditions are
compared with the same control (owner, 2026-10-05; No is proposed, after a table of the
images per condition and biological repeat, when the likeliest control is missing from
repeats where other conditions are imaged). Yes: "Which condition is the control?" (the
scan's guess is the default, "no control" is an answer). No: [several
controls](conditions.md#several-controls), each with a proposal to confirm: which conditions are
controls (named like one, such as `VEC_SiNC`), the conditions of each with their treatment
keys (the name without the stem it shares with its control: `THBS5_SiCol6a1` under
`THBS5_SiNC` is `SiCol6a1`), a parent when a control's name extends another condition's
(`VEC_SiNC` under `VEC`, key `transfection` for an siRNA suffix), and an optional master
control, the control imaged in the most biological repeats proposed (ties to the first
declared). After the conditions of each control, a condition still compared with nothing is
asked "Which control should THBS5 be compared with?", none by default, so a condition not
named like a control, such as `VEC`, can become one (owner, 2026-10-05). The tests these
answers give are then listed, one per line with where it is corrected. Then
a positive control, a
condition known to change the phenotype, which checks that the features can see
a change ("none", the default, leaves that check out; it can be named later,
whenever such a condition was imaged); whether you want to know if the average of each
feature differs, the whole distribution (shape, spread, tails), or both; the features to
assess (several by number, range or name, or all the numeric columns); the readout among
them; for the features named like an intensity or an area whose values are all positive,
whether to test their changes as ratios to the control, yes by default, no testing them as
differences (`Setup.scales`, saved with the answers; plan v2, owner, 2026-10-06); an
optional column that marks a phenotype you care about (true/false or 0/1 per cell,
such as mitotic: `Setup.rare_label`, its share taken as the rare target's prevalence, to be
seen in 3 fields with 95% chance); and, for the whole distribution, its tolerance (the
covariance of the standardised features within a share you state; 10% is offered, with no
published meaning). The whole distribution alone declares no mean target, so no
mean-precision rule; both keeps the mean (owner, 2026-10-06). When the observation table holds no numeric feature (a 0/1 column
marking a phenotype aside) and other feature tables are given, it skips the feature and
readout questions (owner, 2026-10-06): it asks for the phenotype column, then offers the
aim among the whole distribution of each table (its fingerprint), the labelled phenotype and
both; without a phenotype column it says the aim is the whole distribution. Averages are not
offered: there is no readout. Last, it asks whether to run the report now (the default) or
to save the answers and stop, as `run_report=False` does. Enter takes each default.

The answers are saved in the folder as `morphocoverage_answers.json`, with a
checksum of the observation table and one of the FOV table (`None` without
one), and each other feature table's name, path and checksum, before the report runs; the folder then
holds `scan.txt`, `report_short.txt` and `report_full.txt`. Run in the same
folder again, the session shows the saved answers and offers to continue with
them (straight to the report) or to delete them and start over. Answers saved
for another table, or with another FOV table or none, or with other feature tables
(compared by name and checksum), are never reused: the
session offers to delete them, or stops so you can choose another folder.
Answers saved by an earlier version, which did not check the FOV table, are
not reused either: the session says so and asks the questions again
(owner, 2026-10-05). Extra `Setup` fields go through
`guide(..., n_jobs=4, normalize="control_mean")`; `load_answers(folder)` and
`answers.setup(scan)` rebuild the `Setup` without the questions.

## `report`: the text report

```python
from morphocoverage.workflows import report

result = report(objects, fovs, setup)        # fovs lists every acquired field
print(result.summary())                      # title, At a glance, overview, declaration, answers, limits
print(result.summary(detail="full"))         # every sentence and number
result.results["sufficiency"]                # each analysis, for further use
```

`summary()` is short by default: a title line, an **At a glance** block with
one line per answer (✅ answered and good for the goal, ⚠️ answered but look
at this, ❌ a problem that changes how the answers are read, ➖ not assessed;
the rules per question are in `retrospective/report_status.py`), then each
answer, with per-group numbers in tables and each caveat stated once at the
end of its section: a caution found in these data (a non-normal readout,
only three repeats, no acquisition order to check) as a ⚠️ line, which makes
the answer at least ⚠️, and the standing assumptions of the methods (true of
every dataset, such as "hold-outs are not independent experiments") under a
plain "Assumptions:" heading, which never change the marker.
`summary(detail="full")` keeps everything, each section header with its
marker.

Every document opens with **what is in the dataset**: objects, acquired
fields and empty fields, what one row is, the feature columns (the first 12
names, then how many more), the conditions and repeats with a table of fields
by condition and repeat (and of technical repeats, when the column exists),
and whether the values are normalised. It closes
with **what this dataset cannot tell me**: every question not assessed, with
what it would need (a control, a positive control, stage coordinates, a
hierarchy, a replay), and the notes to read the answers with. Missing
columns never stop the report: without `condition` or `biological_repeat` every field is
one group and the sections that need them say so.

Every workflow runs `check_tables` on the setup's columns and control before
any analysis. `optimise`, `benchmark` and `plots` issue its warnings (copied
rows, labels written two ways such as `B2` and `B02`, undeclared group-like
columns, repeats without a control, ...) as one `UserWarning`; `report`
prints them at the end of its overview instead. Warnings that only restate the
design (an assumed observation unit, empty fields) are left out. In the
report, as in the live session, each rule's stop is given with what it rests
on: the order check's caution, or the random-sample assumption.

With `report(objects, fovs, setup, integrity=True)` the overview also gives
the exact data-integrity checks (`integrity_checks`): a field copied under
another name, in another order or in other units, a block of rows pasted into
another field, observations shared by two fields or repeated within one, a
field summary written twice. A finding is structure to inspect, not a
judgment. The fields the checks name are also listed under "6. Which fields
should I look at?" and in "What this dataset cannot tell me", since every
other answer still counts them as independent fields; see
[`integrity.md`](integrity.md).

## The report as a web page

```python
from morphocoverage.workflows import html_report

result = report(objects, fovs, setup)
html_report(result, objects, fovs, "results")   # results/report.html, results/figures/
```

`report.html` is one file that opens in any browser, offline, and can be
emailed or archived as it is; the guided session writes it too. It carries its
figures and data files up to about 25 MB in all; past that, as with hundreds of
conditions, the largest open from `figures/` beside it, so keep the page with its
folder (or send the zip). It reads in the
order a user acts on it, with titles instead of the text report's question
numbers: the data, the fields to check (quality control), with three or more
features the whole dataset at once (the PCA and UMAP), whether there are distinct
populations, whether the conditions differ, whether enough was imaged, and the
rest. A section this design cannot answer ("not applicable", such as the
field-versus-cell check when every field holds one cell) is left out; its reason
stays in "What this dataset cannot tell me". What a section is for is under the
"?" beside its title (on hover or a tap); its answer comes first and the rest is
folded under "Show more", in the words of `Report.summary`, its aligned tables as
tables you can sort by clicking a header.

Every figure is also saved in `figures/` as PNG and as PDF (vector, for papers). On a
wide screen the page uses the width, the feature figures two to a row. The page follows
the light or dark setting of the viewer, with a switch, and prints with every section
open.

The glossary of every test, rule and term the page uses, with the papers they
come from ([Glossary](../common/glossary.md), the same entries), is a drawer on the
side with a search: its tab, the sidebar and every underlined term open it. In
the answers, the first mention of a term in each section is underlined and shows
its meaning on hover or focus (`glossary.PHRASES` lists the words matched; terms
inside tables are left plain, explained by the sentences around them). A Files
section holds every figure (PNG and PDF), both text reports, the feature table
as CSV and, from the guided session, the design scan and the saved answers, each
saved with its button, so the page alone is the file to share. A file passed in
`attach=` that is not found is named there, and two of one name are both carried,
the second as `<name>_2`. The guided session also zips the whole folder
(`zip_results(folder)`: `<folder>.zip`, any other zip in it included) for
archiving, without the figures an earlier session left in `figures/` (`since=`).

## Run time on real data

These times come from idr0028 screen B: 12 wells, 28 fields each, 30
features, 4 workers. They are approximate; those of `optimise` and `benchmark`
predate audit 4.

| Workflow | Settings | Time |
| --- | --- | --- |
| `report` | the example notebook's | about 2 min, 3.5 min on one process (2026-10-05) |
| `optimise` | `fov_counts=[2, 4, 8, 12]` | about 20 s |
| `benchmark` | `n_boot=200` | about 85 s |

**Before a run, roughly.** `report` and `html_report` together take about
0.6 s per feature and condition on 4 cores (the tests), about 8 s per condition
(the page's figures) and about a minute for the UMAP and the rest: 300 features
and 3 conditions about 10 min, 100 features and 41 conditions about 45 min, 5
features and 151 conditions about 30 min. The guided session says its estimate
before it starts. It is approximate, from constructed datasets of 5 to 300
features and 3 to 151 conditions (2026-10-02 and 2026-10-05), which took 13, 50
and about 30 min; a busy machine, the novelty question below and many objects
per field add to it. Other feature tables are never tested feature by feature, but
each adds its populations, maps, copy checks and whole-table test (a few matrix products): the estimate counts 40 s per table and
0.06 s per column (owner, 2026-10-06). Measured on 2026-10-06 on 3 processes, at 5,000
cells in 72 fields with 17 explainable features and 3 conditions, the report and its page
took 54 s and 16 s without a table (a page of 12.5 MB), 86 s and 49 s with a table of 400
columns, and 129 s and 70 s with one of 1,500 (pages of 23.7 MB, within the 25 MB the page
carries its files to); on the ECM tables (236 images, 8 conditions, 17 explainable features
and a DINO table of 384), 94 s on 1 process, a page of 7.3 MB.

`Setup.n_jobs` reaches every workflow, the report's replay of the stopping
rules included (`sufficiency_report(..., n_jobs=)`, since audit 4), and the
cell-level tests of the feature discovery (`feature_discovery(..., n_jobs=)`,
2026-10-05); results are the same for every value. With a novelty definition
the report's time grows faster than the number of objects, since the leave-one-field-out
novelty compares every object with the other fields' objects. In audit 4 (336
fields, 12 groups, 5 features, one process) it took about 90 s of CPU at
30,000 objects, 7 min at 100,000 and 86 min at 300,000, and did not finish in
2.5 h at 1,000,000. For plate-scale tables, set `n_jobs`, subsample the
objects per field for the novelty question, or leave `novelty=None`.
