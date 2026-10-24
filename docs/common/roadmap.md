# Roadmap

The schedule of MorphoCoverage. For each step, this page holds:

- what it needs;
- how we know it passed;
- the decisions that must be made before it.

It was written on 2026-10-13 and revised the same day after two adversarial reviews. The
tracking issue is [#407](https://github.com/CellMigrationLab/MorphoCoverage/issues/407):
it lists this page's ids and, against each, the pull request that closed it.

What this page does not hold:

- **What an item is.** That lives in [open work](open_work.md) and the GitHub issues, and
  for the simulation benchmark in [the validation protocol](validation.md).
- **Decisions.** They live in AGENTS.md and in the protocol's "Deviations".

How the orders relate:

- Open work is ordered by importance.
- AGENTS.md section 8 is the development sequence.
- This page orders the work in time, within that sequence.
- `docs/proposals/candidate_estimators_roadmap.md` is a proposal; an estimator enters this
  page only through an issue or open work.

## Codes used here

Four sets of codes start with the same letters, so this page always prefixes them:

| Prefix | Means | Defined in |
|---|---|---|
| VP0-VP11 | a module of the simulation benchmark ("P4" in the protocol is VP4 here) | [validation.md](validation.md) |
| OPT-P1-P10 | a question of OPTIMISE | [optimise.md](../optimise/optimise.md) |
| R1-R8, O1-O9 | a question of REPORT, of LIVE | [questions.md](../report/questions.md), [live_session.md](../live/live_session.md) |
| OW n | item n of open work | [open_work.md](open_work.md) |

This page's own ids:

- **Items:** a track name and a number. The tracks are ENG, DATA, BENCH, BRICK
  (MorphoBricks, packaging and the bridge), VAL, LIVE and REL.
- **Datasets to be made:** DS-1 to DS-5.
- **Milestones:** M0 to M5.
- **Audits:** AUD-1 to AUD-5.
- **Decisions:** DEC-1 to DEC-21, listed in "Decisions".

## How the tracks fit together

The tracks run in parallel. An item waits only for what its "Needs" names.

```
ENG    ENG-1..8 ─────────────────────────────────────────────── M0
BENCH  BENCH-1 (needs BRICK-0) ── BENCH-2, BENCH-3 ── M1 ── AUD-1
                                  BENCH-5 ───────────────── M2 ── AUD-2
DATA   DATA-1 ── DATA-2 acquire DS-1, DS-2 ── processed after BRICK-2 ── DATA-4 register
BRICK  DEC-21 ── BRICK-0 (benchmark app, its own repository)
       DEC-10 ── BRICK-3
       BRICK-1 ── BRICK-2 (needs DS-3) ── BRICK-4 installer beta (needs M1) ── BRICK-5
VAL    VAL-1, VAL-2, VAL-3 (need M1, DATA-4) · VAL-4 (needs M2) · VAL-5, VAL-6 ──── M3 ── AUD-3
LIVE   LIVE-1 (needs M1) · LIVE-3 (needs DATA-4) · LIVE-5 (needs VP4) ── DATA-7 ── LIVE-7 ── M4 ── AUD-4
REL    0.2 after M1 · 0.3 after M2 · AUD-5 after M3 and M4 ── 1.0 = M5
```

The critical path runs through the data. The acquisitions are the steps no software work
can shorten.

1. DATA-1, the specification. DS-1 and DS-3 are acquired now.
2. DATA-2: DS-1 acquired. Meanwhile BRICK-1 and BRICK-2 make MorphoBricks emit the FOV
   table.
3. DS-1 processed through MorphoBricks, then registered (DATA-4).
4. Two branches:
   - VAL, with M2 for VAL-4, leading to M3;
   - LIVE-3, then the pilot acquisition DATA-7, then LIVE-7, leading to M4.
5. AUD-5, then release 1.0.

## ENG: a clean, enforced repository (now)

| Id | What | Pass condition |
|---|---|---|
| ENG-1 | CI runs again (DEC-1). Until then, the local gate: each PR description pastes the output of `pytest -n 0 tests/test_guards.py tests/test_function_guards.py tests/test_function_tools.py` and `python tools/audit.py --check-against <base>`. AGENTS.md's CI sentence says what holds today. | One PR green on CI, or the local gate written in AGENTS.md and used on every PR. |
| ENG-2 | 38 test files import from another test file, for example `test_html_*` from `test_rare_fingerprint`, `test_human_errors` from `test_end_to_end`. Their shared helpers move to `tests/_helpers_*.py`, and the 27 helper names that collide are renamed. | No test file imports from another test file. |
| ENG-3 | Two groups of test files fold into the test files of the modules they test. The first is the 62 audit-, review-, refactor- and plan-named ones: `test_audit*`, `test_review_*`, `test_refactor_*`, `test_plan_v2_phase1.py`, `test_microscopist_audit.py`. The second is the ones named for a date or a pass: `test_feature_tables_1006`, `test_owner_1005_workflows`, `test_owner_decisions_1002`, `test_owner_decisions_1005_wording`, `test_style_r1`, `_r2`, `_R3`, `_r4`, `_R5`, `test_style_polish`. Their references follow: the tests tree in AGENTS.md, `pyproject.toml`'s filterwarnings comment, `docs/report/integrity.md`. | `pytest --collect-only -q` gives the same number of tests before and after. The old ids map one to one onto the new ones. The suite has the same passes and xfails (last full run: 4368 passed, 6 xfailed). |
| ENG-4 | The naming rule in AGENTS.md: no file, test, directory or document is named after a dated or numbered audit, a review, a pull request, a plan phase or a date. Allowed by name: `tools/audit.py`, `tools/jscpd_review.py`, `tools/baselines/duplicates_reviewed.json`, `docs/audit/`. The guard is part of `tools/audit.py`, so it runs in ENG-1's local gate, also on doc-only PRs, which CI skips. | The guard fails on a planted `tests/test_audit99.py` and a planted `docs/review_x.md`, and passes on the allowed names. |
| ENG-5 | The reference check of PR #405 becomes a repository script, outside `tools/`, which must not import the package. It removes the run time stamps and the benchmark's printed replay seconds, nothing else. Two tiers: committed data (the 50 report scenarios, the 3 lab datasets, synthetic, fiberfeatures, the `data/crick_idr` scripts), runnable by anyone; and an optional tier for the 5 further real-data runs of #405, on the owner's data outside the repository. | Two runs on main agree file for file. The run time is in the script's README. |
| ENG-6 | Records: OW 18's two rules: the held-out rule goes into AGENTS.md section 9, before any further held-out read, and the new-method rule into section 20. The ECM tables' role is recorded (DEC-2), and idr0139's status per choice: `object_rules`'s `min_fields=2` was set with it in view. The MAD rule left open for novelty (owner, 2026-10-13: `RobustStandardize` keeps its zero rule until VP10) goes into the protocol under VP10. | All of these written in AGENTS.md or the protocol. |
| ENG-7 | The audit history: every audit after the six on [the audit page](../audit/README.md) that the code or git history names (today audits 6, 7, 8, 10, 11 and 12) gets one paragraph, added after the existing ones, which stay as written. A table maps the old test file names to the new ones (DEC-3). | Every audit the code names has a paragraph. Every test file the page names can be found through the table. |
| ENG-8 | Old META issues: each open item of #373 and #101 moves to open work, or to "Not scheduled" with its issue number. Then the two issues are closed. | Checked item by item: nothing of #373 or #101 is lost. |

**M0:** ENG-1 to ENG-8 pass.

Added after M0 (owner, 2026-10-13), due before AUD-1:

| Id | What | Pass condition |
|---|---|---|
| ENG-9 | Every test file and test function is checked for whether it is still useful. Each test is classified:<br>- keep;<br>- merge with a test that checks the same thing;<br>- remove, because the behaviour it pins is gone, or another test pins it more directly;<br>- rename, when its name carries an audit, review or issue id such as `test_a3_…` or `test_issue_160_…`, so that it says what it guards against.<br>Removals and merges are listed with their reason. Renames get an old-to-new table, since they change the test ids. | Coverage of `src/` by the suite does not drop: every line covered before is covered after, measured with `coverage` before and after. The full suite passes. The suite's run time is recorded before and after. Every removal names the test, or the change, that makes it redundant. |

## DATA: the data the plan needs (starts now)

### What exists

| Data | Role | Depth | Enough for |
|---|---|---|---|
| idr0028 screen B, 12 wells × about 28 FOVs, 3 plates | development | 3 independent units | development; not a held-out number |
| idr0139, 10 wells × 4 FOVs, 1 plate | held out | 4 FOVs a well | refuting a development conclusion, not setting a number (OW 5) |
| lab `cct7_cct8_glioma_cdm_tracks` | development | median 7.5 fields a well, 6 repeats | development |
| lab `cct8_cdm_morphology` | development | median 18 fields a well, 3 repeats | development |
| lab `cct8_filopodia_screen` | development | one operator-chosen cell per field | contract tests; its fields are not field samples |
| ECM tables, outside the repository | development, by DEC-2 | the owner's | the protocol's real-data anchoring and the owner's reference runs |
| 50 report scenarios, synthetic, `data/morphobricks` | constructed | — | report-path tests; the contract scenarios are `tests/scenarios.py` (AGENTS.md section 9) |

None of the three lab cards is verified by the owner yet (`verified_by` is empty).

### What is needed

| Id | What | Specification | Needed by | Who |
|---|---|---|---|---|
| DATA-1 | The written specification of DS-1 to DS-3, approved (DEC-4) | below | DATA-2 | owner |
| DATA-2 | DS-1, the deep held-out set, and, if it costs little, DS-2 on one of its samples | DS-1: at least 30 FOVs per sample, ideally 60; at least 5 independent units, such as biological repeats or plates; a control and a condition with a known effect; a FOV table listing zero-object fields, stage coordinates and acquisition order. DS-2, optional: an overview of every candidate position, with covariates such as density or region, for the sampling frame, which comes later (DEC-16). Acquired now; processed through MorphoBricks once BRICK-2 passes. | VAL, LIVE-3 | the lab |
| DATA-3 | DS-3, a small real image set for the end-to-end test | 2 conditions, 2 repeats, a few images each, raw images with masks or a segmentation recipe, including one image with no object, one image that fails, and file names repeated across repeats | BRICK-2 | the lab |
| DATA-4 | DS-1, and DS-2 if acquired, registered as held out in `BenchmarkRegistry`, their digests locked before anyone runs a report on them | AGENTS.md section 9 | VAL, LIVE | maintainer |
| DATA-5 | DS-4, a labelled rare-state set, labelled independently of MorphoCoverage | rare states with known prevalence | VAL-4 | the lab, or a public source (none identified yet) |
| DATA-6 | The owner verifies the three lab cards | `verified_by` filled | BENCH-1 | owner |
| DATA-7 | DS-5, the prospective pilot: imaged with the LIVE session running, plus fields acquired past its stop | one microscope, one condition pair | LIVE-7 | the lab |

Development work and audits never read DS-1, DS-2, DS-4 or DS-5. Each is read once per
milestone, by that milestone's protocol.

## BENCH: the simulation benchmark (issue #381)

| Id | What | Needs | Pass condition |
|---|---|---|---|
| BENCH-1 | Five things before any confirmatory run:<br>1. the lab cards verified (DATA-6);<br>2. the protocol's provenance table filled, from `real_parameters.csv` and the owner's ECM summaries;<br>3. VP4's truth (`sequential.Sequence.truth`) changed to the estimand of DEC-5, the mean over the wells that could have been sampled;<br>4. the benchmark app installed (BRICK-0);<br>5. each module's pilot run with it, with its cost recorded. | DATA-6, BRICK-0 | All five done. The pilots' cost recorded in #381. |
| BENCH-2 | Confirmatory runs of VP1 to VP5 at full size and the confirmatory seed (`runner.CONFIRMATORY_SEED`). They run on the machine of DEC-7 with the benchmark app, every core a worker (`--workers`), resuming from the chunks on disk. The review's pilots of 2026-10-13, on 4 CPUs, extrapolate to about 160 CPU-hours, mostly VP1 and VP3: about 10 hours on 16 cores, about 80 on a 2-CPU Colab runtime. | BENCH-1, DEC-7 | The protocol's [selection rule](validation.md#selection-rule), applied by `summarise`. `summary.md`, `summary.csv` and `selection_<question>.csv` are archived for the paper. |
| BENCH-3 | Per module, a pull request ships what the protocol's "What ships" says: the winner, or, when no method is eligible, the limitation stated in the report.<br>Each PR also:<br>- lists every changed output with ENG-5's script;<br>- updates the exploratory labels (`workflows._exploratory` and the docs);<br>- re-versions and re-locks the real-data stopping benchmark (`data/crick_idr/benchmark_protocol.py`) when it changes a rule that benchmark replays, rerun on development data only. | BENCH-2, DEC-18 | Per module: the owner and the second reviewer of DEC-18 verify the implementation, and the full suite passes. |
| BENCH-4 | The six rotation-invariance xfails of `tests/test_validation_p0.py` go to their modules:<br>- VP3: the centring of the tables, the whole-table test's scaling, and the robust scaling where that test uses it;<br>- VP7: the populations' distance space;<br>- VP10: the unusual-observation score and MFA's robust scaling;<br>- VP11: the separation score. | its module | Each xfail removed in its module's PR, by an operation that passes, or kept with the reason. |
| BENCH-5 | VP6 to VP11, VP7 with #397 (the reproducibility of discovered groups over whole-field splits). Each module starts with its ADEMP section, committed as a deviation with the owner's sign-off; VP9 first needs its claim (DEC-8). Then candidates, scripts, pilots and runs. VP7 and VP8 are high cost. | M1 | The same selection rule. The result is "validated in simulation" until VAL-4. |

**M1:** VP1 to VP5 shipped, or each with its limitation stated in the report. AUD-1 follows.

**M2:** each of VP6 to VP10 validated in simulation or kept exploratory with its reason,
and VP11 run or kept "later". AUD-2 follows.

## BRICK: MorphoBricks end to end, packaging and the bridge

MorphoCoverage is step 3 of MorphoBricks: first CellPatcher, then feature extraction, then
MorphoCoverage. The feature extraction is fiberfeatures today; #404 names MorphoRelate.
Packaging uses LabConstrictor, whose installer builds one app per repository.

| Id | What | Needs | Pass condition |
|---|---|---|---|
| BRICK-0 | The benchmark app, in its own private repository, `guijacquemet/MorphoCoverage-Benchmark`, where GitHub Actions run (DEC-1, DEC-21). It is for the owner and reviewers rerunning the benchmark, not for users, so it never ships in the MorphoBricks installer. That keeps its comparators out of that installer, including the GPL-3.0 harmonypy, and lets one installer match one protocol commit, archived for the paper. It is a LabConstrictor app that pip-installs MorphoCoverage with its `validation` extra at a pinned tag, reading this private repository with a read-only token stored as an Actions secret. The benchmark code stays in MorphoCoverage, in `data/simulations/decisions/`, next to the protocol and its smoke tests. The app's `src/` carries a copy taken from the pinned commit, which each summary records. A notebook or command runs a module at a chosen size on every local core, resumes, and summarises. VP0 stays a repository run, since it is a test file. It is the first test of LabConstrictor packaging and needs none of the upstream MorphoBricks changes. | DEC-14 | On the owner's machine, the installed app runs each module's pilot and resumes after being stopped. Its `summary.csv` equals that of a run from the repository at the same seed in every column but the run times, and it selects the same methods. |
| BRICK-1 | The FOV table and identities upstream: the CellPatcher and fiberfeatures changes in [the FOV contract](morphobricks_fov_contract.md), section 2. CellPatcher's technical-repeat PR #9 is not merged yet (DEC-9), and this work proceeds without it. MorphoRelate is added to AGENTS.md section 18. | — | Each upstream main emits the per-image table: zero-object and failed images listed, `position_id` where a file is not one position, the shared column names. |
| BRICK-2 | The end-to-end test on DS-3: raw images, then CellPatcher, then feature extraction, then MorphoCoverage's `report` and `html_report`, every tool at a pinned commit. It also becomes a job of ENG-5's script, run on every pin change. | BRICK-1, DATA-3 | The FOV table lists every image: the zero-object image with 0 objects, the failed image marked failed, not dropped. Repeated file names stay distinct fields. `check_tables` reports no error. A second run is identical. An image removed from the inputs shows as a missing field. |
| BRICK-3 | Each module pip-installable at a pinned version: MorphoCoverage, CellPatcher and the feature extraction each get a version (`__version__` and git tags, DEC-14) and are installable from their git tags until 1.0, from PyPI after (DEC-10). Until the repositories are public, at paper submission (DEC-15), an install needs a read-only token. | — | `pip install` of each module at its pinned version works in a clean environment. The MorphoCoverage notebook runs from a clean Colab. |
| BRICK-4 | One MorphoBricks installer, from a single MorphoBricks repository that does the packaging (DEC-10). That repository is a LabConstrictor app. It holds the pipeline's notebooks and the bridge's tools module, and no science code. Its `requirements.yaml` pip-installs each module at its pinned version. The modules keep their own repositories, tests and releases; a new release of a module reaches users when the MorphoBricks repository moves its pin and builds a new installer. The beta is built after M1 and BRICK-2. The release installer is built from the 1.0 candidate and passes this item again before the tag. | M1, BRICK-2, BRICK-3, DEC-10 | The installer installs on Windows, macOS and Linux. On a clean machine, the end-to-end notebook gives on DS-3 the same report as BRICK-2. The App Centre manifest (`labconstrictor-app.yaml`) is valid and lists the hosts of BRICK-5. |
| BRICK-5 | The plugins, through LabConstrictor's Tools bridge. Tools are declared in the app's single tools module, `src/<app package>_lc_tools/` in the MorphoBricks app repository, which the installer registers. One module serves the three packages, and each tool imports its package inside the function. The tools appear in napari and Fiji, in QuPath where the bridge's host table supports the feature, and from the command line. The REPORT tools come first: a report and its page from a folder of tables, and the flagged fields. #398's lasso export follows if a host supports a selection on a scatter plot; otherwise it stays in the notebook. The bridge is experimental: `labconstrictor-tools` v0.2.0-rc.1, the app template 0.1.16 or later, and the napari and Fiji plugins not yet on PyPI or a Fiji update site. | BRICK-4, M1, DEC-17 | `labconstrictor-tools check` and `test --cases` pass. Each tool runs in napari and in Fiji. A tool's output equals the package API's on the same inputs. |
| BRICK-6 | The LIVE tool through the bridge: a session over a `LiveFolder` that a per-field CellPatcher and feature-extraction step fills with `write_fov`. The microscope only writes images; MorphoCoverage does not control it (AGENTS.md section 4). | M4, DEC-11 | As BRICK-5, and the same continue or stop as `live_session` on DS-5. |

## VAL: validation on real data

| Id | What | Needs | Pass condition |
|---|---|---|---|
| VAL-1 | OPT-P2: protocol reliability counted by independent unit, on DS-1. | M1, DATA-4 | An interval is printed, not withheld: 5 or more units. |
| VAL-2 | OW 1: the remaining Benchmark rows, R2's distribution questions, OPT-P1, OPT-P10 and O6, scored on DS-1 by the real-data stopping benchmark under its locked protocol. | BENCH-3, DATA-4 | Fields saved at matched held-out risk against fixed FOVs, with an interval. |
| VAL-3 | OW 7: the owner's reading test on DS-1, a dataset the report was not built on. | M1, DATA-4 | A written record, for every answer of R1-R8 and OPT-P1-P10, of what the owner would do differently. |
| VAL-4 | The real-data check of VP6 to VP10, VP7 with #397's promotion gate. Each passes on DS-1, or on DS-4 for the rare target, before its exploratory label is removed. | M2, DATA-4, DATA-5 | Per module, the criterion its ADEMP section states. |
| VAL-5 | OW 8: external audits of OPT-P2 and OPT-P6. Each auditor is asked for a dataset on which the answer misleads while every test passes. | DEC-12 | Each audit's findings answered: fixed, or recorded as limits. |
| VAL-6 | OW 2 with #81: the multi-group outcome, and #84's test 5, biological fidelity. | DEC-13, DATA-4 | #81's protocol and #84's test 5 hold on DS-1. |

**M3:**

- VAL-1 to VAL-6 pass.
- A written verdict on [the project aim](project_aim.md)'s six success criteria and its
  falsification list. The prospective criteria are completed at M4.

AUD-3 follows.

## LIVE: from OPTIMISE to the microscope

| Id | What | Needs | Pass condition |
|---|---|---|---|
| LIVE-1 | #83: a frozen sampling plan between OPTIMISE and LIVE. | M1 | #83's acceptance list, including the deterministic JSON round trip. |
| LIVE-2 | #262: spatially structured benchmark scenarios. | — | Each scenario committed with a test of its spatial property. |
| LIVE-3 | #84: the implemented strategies benchmarked for transfer to unseen samples. | LIVE-1, LIVE-2, DATA-4 | #84's tests 1 to 4 and 6 on held-out samples, with an interval: 5 or more units. |
| LIVE-4 | #265, anytime-valid stopping: re-scoped after VP4, which already compares a confidence sequence, and closed if VP4 decides it. | BENCH-3 for VP4 | Done, or closed with VP4's result. |
| LIVE-5 | OW 4 and 14: O5, a borderline state in the decision; O6, the calibration attached to the live session; the LIVE wording rule. | BENCH-3 for VP4 | The decision has a borderline state, O6's calibration is attached, and every live sentence follows the rule. |
| LIVE-7 | The prospective pilot on one microscope, DS-5: `live_session` over a `LiveFolder`, run from a notebook. | LIVE-1, LIVE-3, LIVE-5, DATA-7 | On every DS-5 sample the session reaches a decision. The stop's error on the fields past it is recorded beside O6's calibrated false-stop rate for that rule. The O7 latency, from field to decision and including feature extraction, is below the time between two acquisitions. |

**M4:**

- LIVE-1, LIVE-3, LIVE-5 and LIVE-7 pass.
- LIVE offers only a rule that was frozen, replayed on development data, scored on held-out
  samples and compared with fixed FOVs.

AUD-4 follows. #206, scout and target, comes after M4, with the sampling frame (AGENTS.md
sections 8 and 11).

## REL: releases and the paper

| Release | After | Holds |
|---|---|---|
| 0.2 | M1 | REPORT and OPTIMISE with VP1 to VP5's rules; the exploratory labels say what is not validated. |
| 0.3 | M2 | VP6 to VP11's results, and BRICK-5's REPORT tools. |
| 1.0 | M3, M4, AUD-5 | REPORT and OPTIMISE validated, LIVE, the bridge tools including BRICK-6's, and no `repeat` alias (DEC-20). |

**M5**, release 1.0, passes when:

- the full suite passes;
- `__all__` is frozen;
- a tag, a CHANGELOG, a CITATION file and an archived DOI exist;
- the exploratory labels match the module status table;
- the notebook runs from a clean Colab;
- the release installer has passed BRICK-4.

**The paper:**

- **Main claim:** the regime map, which says when fields, not objects, limit an answer. It
  needs OW 11, VAL-2 and LIVE-3.
- **Supplementary information:** the archived summaries of BENCH-2 and BENCH-5.
- **Before submission:** figure owners named. Data availability settled, since the ECM
  tables stay outside the repository and each dataset has its own licence.

## Any time

These depend on nothing above:

- OW 3, OPT-P3 printed as increments;
- OW 9, OPT-P10's effort-projection intervals, checked by simulation: the simultaneous
  lower bound of their coverage of the true components 0.93 or more, at the protocol's
  replicate counts;
- OW 10, a cost model, once DEC-19 gives its inputs;
- OW 11, excess fields in benchmark records;
- OW 12, a drift diagnostic;
- OW 13, uncertainty semantics;
- #264 (iNEXT) and #268 (SciKit-GStat, GSTools): external numerical references;
- OW 22, library references for the hand-written statistics (#101);
- OW 23, the assessment-state rule (#373);
- #404, `patch_id`, with MorphoRelate (done on `claude/pensive-ride-khcmug`: `patch_id` is an observation identity of the feature tables);
- #398, the lasso export, which BRICK-5 then uses.

Not scheduled: #401, #402, #195, OW 15 to 17 and 19 to 21, and the sampling frame (#194, OW 6, DEC-16), a new input contract with CellPatcher upstream, with DS-2.

## Decisions

The owner makes each one before the step it names. The answers are recorded in AGENTS.md
section 8 and, for the benchmark, in the protocol's deviations, not here. The owner answered
all of them on 2026-10-13; the three left for later are put again at the step they name.

| Id | Decision | Before | Status |
|---|---|---|---|
| DEC-1 | Restore CI, by billing or a self-hosted runner, or keep the local gate of ENG-1 | M0 | decided: the benchmark app on the owner's account; this repository keeps ENG-1's local gate |
| DEC-2 | The ECM tables' role: development, since they were read for anchoring | ENG-6 | decided |
| DEC-3 | That the audit page gains paragraphs for the later audits and a table of renamed test files | ENG-7 | decided |
| DEC-4 | The specification of DS-1 to DS-3 (DATA-1) | DATA-2 | decided |
| DEC-5 | VP4's estimand. The protocol says "given its wells". If it is kept, the package's well bootstrap, which resamples wells, may fail the estimand gate; if it is changed, that is a deviation. Either way it sets LIVE's mean rule | BENCH-1 | decided: a deviation |
| DEC-6 | Whether the protocol is deposited with a timestamp as a pre-registration | BENCH-2 | decided: no deposit, not a registered study |
| DEC-7 | The machine for BENCH-2: the owner's multi-core computer with the benchmark app, else Colab sessions or a paid runner | BENCH-2 | decided |
| DEC-8 | VP9's claim: the variance, or named quantiles | VP9's ADEMP section | after VP1 to VP5 |
| DEC-9 | When CellPatcher PR #9, the technical repeat, lands | BRICK-1 | decided: not yet |
| DEC-10 | The packaging design of the MorphoBricks installer. Proposed by the owner, 2026-10-13: a single MorphoBricks repository does the packaging and pip-installs the modules at pinned versions. To confirm: whether the modules come from PyPI or from git tags (LabConstrictor's `requirements.yaml` lists pinned pip packages; git URLs there are untested) | BRICK-3 | decided |
| DEC-11 | How LIVE reaches the microscope. Through a `LiveFolder` filled per field only, or also by a direct integration such as Micro-Manager, which the bridge does not provide and AGENTS.md section 4 excludes today. Also whether the bridge hosts a long-running watch, or the tool is one-shot, replaying the folder and deciding now | BRICK-6 | after M4 |
| DEC-12 | Who the external auditors of VAL-5 are | VAL-5 | decided |
| DEC-13 | The multi-group outcome contract and its registry | VAL-6 | decided: after M1 |
| DEC-14 | The version scheme: how versions are numbered and tagged, and from which version the interface is stable | BRICK-0, BRICK-3 | decided |
| DEC-15 | The repositories' visibility, and the licences: the `validation` extra holds the GPL-3.0 harmonypy, which the benchmark app bundles and the MorphoBricks installer does not | BRICK-0, BRICK-3 | decided: public at paper submission |
| DEC-16 | The sampling-frame input contract, with CellPatcher | DS-2's format | later: not scheduled |
| DEC-17 | AGENTS.md sections 3 and 18 gain the bridge's hosts and its command line as usage contexts; today they defer a command line | BRICK-5 | decided: all four |
| DEC-18 | Who the second reviewer is, verifying implementations beside the owner | BENCH-3 | decided: an AI review per PR |
| DEC-19 | The cost-model inputs `C_sample` and `C_FOV` | OW 10 | decided: user-supplied |
| DEC-20 | The 1.0 scope: which workflows ship validated, whether LIVE is in 1.0, whether the `repeat` alias is removed | AUD-5 | decided |
| DEC-21 | The benchmark app's repository: its name, proposed `MorphoCoverage-Benchmark`, and its visibility, public when reviewers are to rerun it | BRICK-0 | decided |

## Audits

An audit follows each of M1 to M4 (AUD-1 to AUD-4), and AUD-5 precedes release 1.0.

- **Scope:** what changed since the last audit, every report question it touches followed
  from the tables to the sentence, and a fixed regression set: ENG-5's script and the code
  tools (`python tools/audit.py`). Not the whole package each time.
- **Data:** development data only (see DATA).
- **Limits:** a time box and a findings budget set at the start. What does not fit becomes
  open work.
- **Evidence:** each finding reproduced before it is reported. A fix pinned by a test that
  fails on the old code. Behaviour that must not change proven with ENG-5's script.
- **Where things go:**
  - the findings in one GitHub issue per audit;
  - fixes in pull requests, their tests in the test file of the module they test;
  - scientific choices to the owner, recorded as decisions;
  - when it closes, one paragraph in [the audit history](../audit/README.md).
- **Never committed:** audit reports, scratch scripts, intermediate outputs, or data that is
  not ours to publish.

## Risks

| Risk | Answer |
|---|---|
| One owner decides, verifies, runs the benchmark and supplies data | decisions batched; a second reviewer (DEC-18) |
| The deep data come late | DATA starts now; only VAL, LIVE-3, LIVE-7 and 1.0 wait for it |
| A held-out dataset stops being held out | ENG-6 first; DATA-4 locks before any read |
| Compute for the benchmark | the benchmark app on every local core (BRICK-0); measured by the pilots |
| CI stays blocked | ENG-1's local gate, which includes ENG-4's guard; BRICK-0 builds on the owner's account; BRICK-4's repository needs Actions that run, chosen when it is created |
| The bridge changes, being experimental; its Windows hook has not run on a real Windows computer | its version pinned in the installer; BRICK-5 after M1, BRICK-6 after M4 |
| Upstream repositories change the FOV contract | BRICK-2 rerun on every MorphoBricks pin change |

## Keeping the repository clean

- A file is named for what it holds; ENG-4's guard enforces it.
- A fact lives in one place; other pages link to it.
- A test lives in the test file of the module it tests. A regression test's docstring says
  what it guards against, not where the finding came from.
- Generated files are regenerated by their tool, never edited by hand: the function index,
  the duplicate report, the baselines.
- Committed results under `data/` are regenerated in the pull request that changes them.
