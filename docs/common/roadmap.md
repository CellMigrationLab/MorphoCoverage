# Roadmap

The schedule of MorphoCoverage: what is done in which order, what each step needs, how we
know it passed, and when each decision must be made. It was written on 2026-10-13 and
revised after an adversarial review the same day. The tracking issue is
[#407](https://github.com/CellMigrationLab/MorphoCoverage/issues/407); it lists the ids of
this page and the pull request that closed each one.

What this page does not hold:

- what an item is: [open work](open_work.md), the GitHub issues, and for the simulation
  benchmark [the validation protocol](validation.md) with its selection rule and gates;
- decisions: AGENTS.md and the protocol's "Deviations".

Open work is ordered by importance. This page orders the work in time, and where the two
differ, this page decides what happens next. AGENTS.md section 8 is the history of how the
package was built.

## Codes used here

Four sets of codes start with the same letters, so this page always prefixes them:

| Prefix | Means | Defined in |
|---|---|---|
| VP0-VP11 | a module of the simulation benchmark ("P4" in the protocol is VP4 here) | [validation.md](validation.md) |
| OPT-P1-P10 | a question of OPTIMISE | [optimise.md](../optimise/optimise.md) |
| R1-R8, O1-O9 | a question of REPORT, of LIVE | [questions.md](../report/questions.md), [live_session.md](../live/live_session.md) |
| OW n | item n of open work | [open_work.md](open_work.md) |

The roadmap's own ids are a track letter and a number: ENG, DATA, BENCH, BRICK (MorphoBricks,
packaging and the bridge), VAL, LIVE and REL. Milestones are M0 to M5, and the audits
AUD-1 to AUD-5. Decisions are DEC-1 to DEC-16 in "Decisions", below.

## How the tracks fit together

The tracks run in parallel. Each item says what it needs; nothing waits for anything else.

```
ENG   ──M0──────────────────────────────────────────────────────────────┐
BENCH ──prerequisites, benchmark app──VP1-VP5──M1──AUD-1──VP6-VP11──M2  │
DATA  ──spec──acquire deep held-out data──process──register, lock───┐   │
BRICK ──upstream FOV table──end-to-end test──installer beta──bridge─┼───┤
VAL   ─────────────────────────────── (needs M1 and DATA) ──────────M3──AUD-3
LIVE  ─────────────────────── (needs VP4, M3) ─────── prospective pilot──M4──AUD-4
REL   ── releases 0.2 (after M1), 0.3 (after M2), 1.0 (after M3, M4) ──M5──AUD-5
```

**The critical path** is the new data:

1. the specification (DATA-1);
2. the acquisition (DATA-2);
3. processing through MorphoBricks, which needs the end-to-end test (BRICK-2);
4. registration (DATA-4);
5. real-data validation (M3);
6. the sampling plan and the transfer benchmark (LIVE-1, LIVE-3);
7. the prospective pilot (M4);
8. release 1.0.

The acquisition is the only step no software work can shorten, so it starts now.

## ENG: a clean, enforced repository (now)

| Id | What | Pass condition |
|---|---|---|
| ENG-1 | CI runs again. Until then, the local gate: each PR description pastes the output of `pytest -n 0 tests/test_guards.py tests/test_function_guards.py tests/test_function_tools.py` and `python tools/audit.py --check-against <base>`. AGENTS.md's CI sentence says what holds today. Decide DEC-1. | One PR green on CI, or the local gate written in AGENTS.md and used on every PR. |
| ENG-2 | Shared test helpers move to `tests/_helpers_*.py`. They are now imported across files: `test_one_per_field_everywhere.py`, `test_owner_decisions_1005_wording.py` and `test_style_foundation.py` import from audit-named files, and 27 helper names collide. | No test file imports another test file. |
| ENG-3 | The 62 audit-, review-, refactor- and plan-named test files (`test_audit*`, `test_review_*`, `test_refactor_*`, `test_plan_v2_phase1.py`, `test_microscopist_audit.py`) and the date- or pass-named ones (`test_*_1002`, `_1005`, `_1006`, `test_owner_1005_workflows.py`, `test_style_r1`, `_r2`, `_r4`, `test_style_polish.py`) fold into the test files of the modules they test. The references follow: AGENTS.md (the tests tree), `pyproject.toml`'s filterwarnings comment, `docs/report/integrity.md`. | `pytest --collect-only -q` gives the same number of tests before and after, the old ids map one to one onto the new ones, and the suite has the same passes and xfails (last full run: 4368 passed, 6 xfailed). |
| ENG-4 | The naming rule in AGENTS.md: no file, test, directory or document is named after a dated or numbered audit, a review, a pull request, a plan phase or a date. Allowed by name: `tools/audit.py`, `tools/jscpd_review.py`, `tools/baselines/duplicates_reviewed.json`, `docs/audit/`. The guard runs in `tools/audit.py`, so it also runs on doc-only PRs, which CI skips. | The guard fails on a planted `tests/test_audit99.py` and a planted `docs/review_x.md`, and passes on the allowed names. |
| ENG-5 | The reference check of PR #405 becomes a repository script, outside `tools/`, because `tools/` must not import the package. It normalises the run time stamps and the benchmark's printed replay seconds, nothing else. It has two tiers: committed data (the 50 scenarios, the 3 lab datasets, synthetic, fiberfeatures, the `data/crick_idr` scripts), runnable by anyone, and an optional tier for the owner's data outside the repository (the 5 ECM runs). Its run time is recorded. | Two runs on main agree file for file, and the run time is in its README. |
| ENG-6 | Records: OW 18, the held-out rule, is written into AGENTS.md section 9 before any further held-out read. The ECM tables are declared development data (DEC-2). idr0139's status per choice is recorded: `object_rules`'s `min_fields=2` was set with it in view. The MAD rule left open for novelty (owner, 2026-10-13: `RobustStandardize` keeps its zero rule until VP10) goes into the protocol under VP10. | All four written in AGENTS.md or the protocol. |
| ENG-7 | The audit history: audits 7 to 12 get one paragraph each in [the audit page](../audit/README.md), added after the existing paragraphs, which stay as written (DEC-3). Each paragraph says where its tests are now (after ENG-3). | Every audit cited in the code has a paragraph. |
| ENG-8 | Old META issues: each open item of #373 and #101 moves to open work or to "Not scheduled" with its issue number. Then the two issues are closed. | Nothing of #373 or #101 is lost. Check list against list. |

**M0** when ENG-1 to ENG-8 pass.

## DATA: the data the plan needs (starts now)

### What exists

| Data | Role | Depth | Enough for |
|---|---|---|---|
| idr0028 screen B, 12 wells × about 28 FOVs, 3 plates | development | 3 independent units | development; not a held-out number |
| idr0139, 10 wells × 4 FOVs, 1 plate | held out | 4 FOVs a well | refuting a development conclusion, not setting a number (OW 5) |
| lab `cct7_cct8_glioma_cdm_tracks` | development | median 7.5 fields a well, 6 repeats | development |
| lab `cct8_cdm_morphology` | development | median 18 fields a well, 3 repeats | development |
| lab `cct8_filopodia_screen` | development | one operator-chosen cell per field | contract tests; its fields are not field samples |
| ECM tables, outside the repository | development (DEC-2) | the owner's | the protocol's real-data anchoring, the owner's reference runs |
| 50 scenarios, synthetic, `data/morphobricks` | constructed | — | contract tests, AGENTS.md section 9 |

None of the three lab cards is verified by the owner yet (`verified_by` is empty).

### What is needed

| Id | Data | Specification | Needed by | Who |
|---|---|---|---|---|
| DATA-1 | The written specification of the deep held-out acquisition, approved (DEC-4). | At least 30 FOVs per sample, ideally 60, at least 5 independent units (biological repeats or plates), a control and a condition with a known effect, and a FOV table listing zero-object fields, stage coordinates and acquisition order. Acquired through MorphoBricks once BRICK-2 passes. | DATA-2 | owner |
| DATA-2 | Deep held-out dataset H1, acquired per DATA-1. | as DATA-1 | VAL, LIVE-3 | the lab |
| DATA-3 | H2: an overview of every candidate position, imaged with one H1 sample, for the sampling frame (OPT-P9, #194, OW 6). | covariates such as density or region per position | LIVE-6 | the lab |
| DATA-4 | H1 and H2 registered as held out in `BenchmarkRegistry` and their digests locked, before anyone runs a report on them. | per AGENTS.md section 9 and OW 18 | VAL, LIVE | maintainer |
| DATA-5 | A labelled rare-state set: labels made independently of MorphoCoverage. | rare states with known prevalence | VP6, VAL-4 | the lab, or a public source (none identified yet) |
| DATA-6 | H3: a small real image set for the MorphoBricks end-to-end test. | 2 conditions, 2 repeats, a few images each, raw images with masks or a segmentation recipe | BRICK-2 | the lab |
| DATA-7 | H4: the prospective pilot. A new acquisition imaged with the LIVE session running, plus fields acquired past the stop, to score the stop. | one microscope, one condition pair | LIVE-7 | the lab |
| DATA-8 | The owner verifies the three lab cards. | `verified_by` filled | BENCH-1 | owner |

Audits and development work never read H1 to H4. Each held-out dataset is read once per
milestone, by the protocol of that milestone.

## BENCH: the simulation benchmark (issue #381)

| Id | What | Pass condition |
|---|---|---|
| BENCH-1 | Before any confirmatory run: DATA-8; the provenance table of the protocol filled from `real_parameters.csv` and the owner's ECM summaries; the VP4 estimand settled (DEC-5); the protocol deposited publicly with a timestamp (DEC-6); the benchmark app installed (BRICK-0); each module's pilot run with it, with its cost recorded. | All six done. The pilots' cost recorded in #381. |
| BENCH-2 | Confirmatory runs of VP1 to VP5 at full size and the confirmatory seed (`runner.CONFIRMATORY_SEED`), on the owner's machine with the benchmark app of BRICK-0, every core a worker (`--workers`), resuming from the chunks on disk. Compute: the pilots on a 4-CPU machine extrapolate to about 160 CPU-hours for VP1 to VP5, mostly VP1 and VP3: about 10 hours on 16 cores, against about 80 on a 2-CPU Colab runtime (DEC-7). | The protocol's gates (below), applied by `summarise`; `summary.md`, `summary.csv` and `selection_<question>.csv` archived for the paper. |
| BENCH-3 | Per module, a pull request ships the winner, or, when no method passes, the limitation in the report as a recorded deviation. Each PR does three more things. It lists every changed output with the reference script (ENG-5). It updates the exploratory labels (`workflows._exploratory`, the docs). It re-versions and re-locks the real-data stopping benchmark (`data/crick_idr/benchmark_protocol.py`) when it changes a rule that benchmark replays, rerun on development data only. | Per module, the owner verifies the implementation; the full suite passes. |
| BENCH-4 | The six rotation-invariance xfails of `tests/test_validation_p0.py` go to their modules. VP3 resolves the centring of the tables and the whole-table test's scaling, and the robust scaling where that test uses it. VP7 resolves the populations' distance space. VP10 resolves the unusual-observation score and MFA's robust scaling. VP11 resolves the separation score. | Each xfail removed in its module's PR, by an operation that passes, or kept with the reason. |
| BENCH-5 | VP6 to VP11. First an ADEMP section per module, committed as a deviation, with the owner's sign-off; VP9 needs its claim defined first (DEC-8). Then candidates, scripts, pilots and runs. VP7 and VP8 are high cost. | The same gates. A module's result is "validated in simulation" until VAL-5. |

The protocol's gates, for each module of BENCH-2 and BENCH-5:

- No more than 1% failed or unconverged runs.
- On every gated null design, the simultaneous 95% upper bound of the false-positive rate is
  0.060 or less at nominal 0.05. For VP4 this is the false-stop rate; for VP5 the mean FDP
  at 0, 20, 50 and 80% non-null.
- The lower bound of interval coverage is 0.93 or more.
- The rotation gate for embeddings, and the estimability gate of VP2.
- On power: Pareto-dominated methods are dropped. A difference under 2 points or under 2
  paired Monte Carlo SE is a tie (for VP4, 1 field), broken by simplicity.

**M1:** VP1 to VP5 shipped, or each recorded as having no eligible method. **AUD-1**
follows.

**M2:** VP6 to VP11 each validated in simulation or kept exploratory, with the reason in
the report. **AUD-2** follows.

## BRICK: MorphoBricks end to end, packaging and the bridge

MorphoCoverage is step 3 of MorphoBricks: CellPatcher, then feature extraction (today
fiberfeatures; MorphoRelate is named in #404), then MorphoCoverage.

| Id | What | Needs | Pass condition |
|---|---|---|---|
| BRICK-0 | The benchmark app, packaged early with LabConstrictor: an installer that brings MorphoCoverage with its `validation` extra and the benchmark scripts (`data/simulations/decisions/`), and a notebook or command that runs a module at a chosen size on every local core, resumes, and summarises. It is also the first test of LabConstrictor packaging for BRICK-4. | BRICK-3, DEC-15 | On the owner's machine, the installed app runs each module's pilot, resumes after being stopped, and its `summary.csv` equals that of a run from the repository at the same seed. |
| BRICK-1 | The FOV table and identities upstream: the CellPatcher and fiberfeatures changes listed in [the FOV contract](morphobricks_fov_contract.md), section 2. CellPatcher's technical-repeat PR #9 lands when the owner says so (DEC-9). MorphoRelate is added to AGENTS.md section 18. | DEC-9 | Each upstream main emits the per-image table: zero-object and failed images listed, `position_id` where file ≠ position, the shared column names. |
| BRICK-2 | An end-to-end test on H3: raw images → CellPatcher → feature extraction → MorphoCoverage `report` and `html_report`, every tool at a pinned commit. It also becomes a job of the reference script. | BRICK-1, DATA-6 | The FOV table holds every image, with failed images marked, not dropped. `check_tables` reports no error. The report runs. A second run is identical. An image removed from the inputs shows as a missing field. H1 is processed only after this passes. |
| BRICK-3 | MorphoCoverage ready to package: a version (`__version__` and git tags, DEC-14), and the notebook installable when the repository is private (DEC-15). | DEC-14, DEC-15 | `pip install` from a tag works; the notebook runs from a clean Colab. |
| BRICK-4 | One MorphoBricks installer with LabConstrictor: one app repository whose installer brings CellPatcher, the feature extraction and MorphoCoverage, each pinned, with the notebooks for the whole pipeline. LabConstrictor builds one app per repository, so bundling three packages is checked first (DEC-10). The beta follows M1 and BRICK-2; the release installer follows M5. | M1, BRICK-2, BRICK-3, DEC-10 | The installer installs on Windows, macOS and Linux. On a clean machine, the end-to-end notebook gives on H3 the same report as BRICK-2. The App Centre manifest (`labconstrictor-app.yaml`) is valid. |
| BRICK-5 | The plugins, through LabConstrictor's Tools bridge: MorphoCoverage functions declared as tools in a `morphocoverage_lc_tools` module. They appear in napari, Fiji and QuPath and from the command line. The REPORT tools come first: a report and its page from a folder of tables, the flagged fields, and the lasso export of #398. The bridge is experimental (`labconstrictor-tools` v0.2.0-rc.1; its napari and Fiji plugins are not yet on PyPI or a Fiji update site). It has no microscope host (DEC-11). | BRICK-4, M1 | `labconstrictor-tools check` and `test --cases` pass. Each tool runs in napari and in Fiji, within the bridge's table of host features. A tool's output equals the package API's on the same inputs. |
| BRICK-6 | The LIVE tool through the bridge: a session over the folder the acquisition software writes into (`LiveFolder`), with no control of the microscope (AGENTS.md section 4). | M4, DEC-11 | As BRICK-5, and the same continue or stop as `live_session` on H4. |

## VAL: validation on real data

| Id | What | Needs | Pass condition |
|---|---|---|---|
| VAL-1 | OPT-P2, protocol reliability counted by independent unit on H1. | M1, DATA-4 | An interval is printed, not withheld: 5 or more units. |
| VAL-2 | OW 9: OPT-P10's effort-projection intervals. | DATA-4 | Coverage of the true components with a lower bound of 0.93 or more, in place of today's 80 to 88%. |
| VAL-3 | OW 1: the remaining Benchmark rows (R2's distribution questions, OPT-P1, OPT-P10, O6), scored on H1 by the real-data stopping benchmark, under its locked protocol. | BENCH-3, DATA-4 | Fields saved at matched held-out risk against fixed-FOV, with an interval. |
| VAL-4 | OW 7: the owner's reading test on H1, a dataset the report was not built on. | M1, DATA-4 | A written record, for every answer of R1-R8 and OPT-P1-P10, of what the owner would do differently. |
| VAL-5 | VP6 to VP11's real-data check. Each passes on H1, or with DATA-5 for the rare target, before its exploratory label is removed. | M2, DATA-4, DATA-5 | Per module, as the module's ADEMP section states. |
| VAL-6 | OW 8: external audits of OPT-P2 and OPT-P6. Each auditor is asked for a dataset on which the answer misleads while every test passes. | DEC-12 | Each audit's findings answered: fixed, or recorded as limits. |
| VAL-7 | OW 2 with #81: the multi-group outcome, once its contract is decided. | DEC-13 | The protocol of #81 holds on H1. |

**M3:**

- VAL-1 to VAL-7 pass.
- A verdict, in writing, on [the project aim](project_aim.md)'s six success criteria and
  its falsification list.

**AUD-3** follows.

## LIVE: from OPTIMISE to the microscope

| Id | What | Needs | Pass condition |
|---|---|---|---|
| LIVE-1 | #83: a frozen sampling plan between OPTIMISE and LIVE. | M1 | #83's acceptance list, including the deterministic JSON round trip. |
| LIVE-2 | #262: spatially structured benchmark scenarios. | — | They come before LIVE-3, as #84 orders it. |
| LIVE-3 | #84: the implemented strategies benchmarked for transfer to unseen samples. | LIVE-1, LIVE-2, DATA-4 | #84 tests 1 to 4 on held-out samples, with an interval: 5 or more units. |
| LIVE-4 | #265, anytime-valid stopping: re-scoped after VP4, which already compares a confidence sequence. Closed if VP4 decides it. | VP4 | Done or closed with VP4's result. |
| LIVE-5 | OW 4 and 14: O5, a borderline state; O6, the calibration attached to the live session; the LIVE wording rule. | VP4 | O6's calibration attached and every live sentence worded by the rule. |
| LIVE-6 | #194 and OW 6: the sampling frame, a new input contract with CellPatcher upstream (DEC-16). | DEC-16, DATA-3 | The contract's tests and H2's frame comparison. |
| LIVE-7 | The prospective pilot on one microscope (H4). | LIVE-1, LIVE-3, LIVE-5 | The stop is scored on the fields acquired past it. The decision latency (O7) is below the time between two acquisitions on that microscope. |

**M4:**

- LIVE-1, LIVE-3, LIVE-5 and LIVE-7 pass.
- Only a rule that was frozen, replayed on development data, scored on held-out samples and
  compared with fixed FOVs is offered in LIVE.

**AUD-4** follows. #206 (scout and target) comes after M4, with the sampling frame
(AGENTS.md section 11).

## REL: releases and the paper

| Release | After | Holds |
|---|---|---|
| 0.2 | M1 | REPORT and OPTIMISE with VP1 to VP5's rules; the exploratory labels say what is not validated. The installer beta follows (BRICK-4). |
| 0.3 | M2 | VP6 to VP11's results. The REPORT tools of the bridge (BRICK-5). |
| 1.0 | M3, M4, AUD-5 | The release scope of DEC-14: the interface frozen, the `repeat` alias removed if so decided. The installer and the bridge tools released. |

**M5**, release 1.0, passes when:

- the full suite passes;
- `__all__` is frozen;
- a tag, a CHANGELOG, a CITATION file and an archived DOI exist;
- the exploratory labels match the module status table;
- the notebook runs from a clean Colab;
- the installer passes BRICK-4 again.

The paper:

- **Main claim:** the regime map: when fields, not objects, limit an answer. It needs OW 11
  and VAL-3.
- **Supplementary information:** BENCH-2 and BENCH-5's archived summaries.
- **Before submission:** figure owners named; data availability settled, since the ECM
  tables stay outside the repository and each dataset has its own licence.

## Any time

These items depend on nothing above:

- OW 3, OPT-P3 printed as increments;
- OW 10, a cost model, once its inputs are given;
- OW 11, excess fields in benchmark records;
- OW 12, a drift diagnostic;
- OW 13, uncertainty semantics;
- #264 (iNEXT) and #268 (SciKit-GStat, GSTools), external numerical references;
- #404, `patch_id`, with MorphoRelate;
- #398, the lasso export, which BRICK-5 then uses.

Not scheduled: #401, #402, #195 and OW 15 to 17 and 19 to 21.

## Decisions

The owner makes each one before the step it names. Each is recorded in AGENTS.md or the
protocol, not here.

| Id | Decision | Before |
|---|---|---|
| DEC-1 | Restore CI (billing or a self-hosted runner), or keep the local gate of ENG-1 | ENG-4 |
| DEC-2 | The ECM tables' role: development, since they were read for anchoring | BENCH-1 |
| DEC-3 | Whether the audit page may gain paragraphs for audits 7 to 12 | ENG-7 |
| DEC-4 | The deep held-out specification of DATA-1 | DATA-2 |
| DEC-5 | VP4's estimand: the protocol says "given its wells". A change is a deviation, and the package's own well bootstrap may then fail the estimand gate | the VP4 pilot |
| DEC-6 | Where the protocol is deposited with a timestamp (Zenodo or OSF), since the repository is private | BENCH-2 |
| DEC-7 | The machine for BENCH-2: the owner's multi-core computer with the benchmark app (BRICK-0), else Colab sessions or a paid runner | BENCH-2 |
| DEC-8 | VP9's claim: the variance, or named quantiles | VP9's ADEMP section |
| DEC-9 | When CellPatcher PR #9 (the technical repeat) lands | BRICK-1 |
| DEC-10 | The MorphoBricks app repository: a new one, and how LabConstrictor installs three packages in it | BRICK-4 |
| DEC-11 | The microscope: through `LiveFolder` only, or also a direct integration (for example Micro-Manager), which the bridge does not provide and AGENTS.md section 4 excludes today | BRICK-6, LIVE-7 |
| DEC-12 | Who the external auditors of VAL-6 are | VAL-6 |
| DEC-13 | The multi-group outcome contract and its registry | VAL-7 |
| DEC-14 | The version scheme and the 1.0 scope: which workflows ship validated, whether LIVE is in 1.0, whether the `repeat` alias is removed | BRICK-3, REL 1.0 |
| DEC-15 | The repository's visibility, and the licences: the `validation` extra holds the GPL-3.0 harmonypy, which the benchmark app would bundle | BRICK-0, BRICK-3 |
| DEC-16 | The sampling-frame input contract, with CellPatcher | LIVE-6 |

Decisions are batched: the open ones are put to the owner together at each milestone, and
before BENCH-1, which needs five of them.

## Audits

An audit follows each of M1 to M4 (AUD-1 to AUD-4), and AUD-5 precedes release 1.0.

- **Scope:** what changed since the last audit, every report question it touches followed
  from the tables to the sentence, plus a fixed regression set: the reference script of
  ENG-5 and the code tools (`python tools/audit.py`). Not the whole package each time.
- **Data:** development data only. Held-out data are read once per milestone, by that
  milestone's protocol.
- **Limits:** a time box and a findings budget are set at the start. What does not fit
  becomes open work.
- **Evidence:** each finding reproduced before it is reported. A fix pinned by a test that
  fails on the old code. Behaviour that must not change proven with the reference script.
- **Where things go:**
  - the findings in one GitHub issue per audit;
  - fixes in pull requests, their tests in the test file of the module they test;
  - scientific choices to the owner, recorded as decisions;
  - when it closes, one paragraph in [the audit history](../audit/README.md).
- **Never committed:** audit reports, scratch scripts, intermediate outputs, or data that
  is not ours to publish.

## Risks

| Risk | Answer |
|---|---|
| One owner decides, verifies, runs the benchmark and supplies data | decisions batched per milestone; a second reviewer named for verifying implementations |
| The deep data come late | DATA starts now; nothing else waits for it except VAL, LIVE-3, LIVE-7 and 1.0 |
| A held-out dataset stops being held out | ENG-6 first; DATA-4 locks before any read; audits read development data only |
| Compute for the benchmark | run locally on every core with the benchmark app (BRICK-0), measured by the pilots (BENCH-1); DEC-7 |
| CI stays blocked | ENG-1's local gate; ENG-4's guard runs locally |
| The bridge changes, being experimental | its version pinned in the installer; BRICK-5 after M1, BRICK-6 after M4 |
| Upstream repositories change the FOV contract | BRICK-2 is a reference job, run on every MorphoBricks pin change |

## Keeping the repository clean

- A file is named for what it holds; the guard of ENG-4 enforces it.
- A fact lives in one place; other pages link to it.
- A test lives in the test file of the module it tests. A regression test's docstring says
  what it guards against, not where the finding came from.
- Generated files (the function index, the duplicate report, the baselines) are regenerated
  by their tool, never edited by hand.
- Committed results under `data/` are regenerated in the pull request that changes them.
