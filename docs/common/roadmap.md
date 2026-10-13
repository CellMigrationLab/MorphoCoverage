# Roadmap

The order of work for MorphoCoverage, in phases, each ending in a milestone. This page
holds the order, the milestones and the rules that keep the repository clean. What each
item is lives in one place only: the numbered items of [open work](open_work.md), the
GitHub issues, and for the benchmark [the validation protocol](validation.md). The
checklist that follows the work is the tracking issue
[#407](https://github.com/CellMigrationLab/MorphoCoverage/issues/407) (owner,
2026-10-13).

An adversarial audit of the whole package closes each phase from Phase 1 on (see "Audits"
below), and a phase starts when the audit of the one before it is closed.

## Phase 0: a clean slate

The repository holds code, its tests and its documentation, each named for what it is,
not for when it was made.

1. The 63 test files named after an audit, a review, a refactor or a plan phase
   (`test_audit*`, `test_review_*`, `test_refactor_*`, `test_plan_v2_*`,
   `test_microscopist_audit.py`) are folded into the test files of the modules they test,
   test bodies unchanged: the same number of tests passes before and after.
2. A rule in AGENTS.md: no file, test, directory or document is named after an audit, a
   review, a pull request or a plan phase. A guard test fails on a new one.
3. The byte-identical check of the refactor (PR #405) becomes a repository tool: reference
   outputs from the scenarios, the lab datasets, synthetic, fiberfeatures and the
   `data/crick_idr` scripts, compared after removing time stamps only. Refactors and
   audits use it. The ECM data stays outside the repository.
4. One record per kind of fact: what is open in [open work](open_work.md), decisions in
   AGENTS.md and the validation protocol, the audit history in
   [one page](../audit/README.md), findings in GitHub issues. The older META issues (#373,
   #101) are closed or re-scoped against this page.

Milestone M0: no audit-named file in the repository, and the reference tool in `tools/`.

## Phase 1: the statistical benchmark, P0-P5

The pre-registered benchmark of issue #381 and [the validation protocol](validation.md)
chooses the rules the package ships; the paper's supplementary information reports it.

1. P0, the analytical checks: done, as `tests/test_validation_p0.py`.
2. Before the confirmatory runs: the P4 estimand settled (the mean of the wells imaged, or
   of the wells sampled), the protocol's numbers confirmed, and each module's pilot run to
   measure its cost.
3. The confirmatory runs of P1-P5 on Colab (`data/simulations/decisions/`), summaries
   checked, and the owner's decision per module.
4. Each module's winner ships in its own pull request, with the reference tool listing
   every output it changes; a departure from the winner is a documented deviation in the
   protocol.
5. What waits for a module: the 6 embedding operations that fail rotation invariance (P3),
   the moderated paired t (P1), the contrast layer for designs without a control (P2).

Milestone M1: P0-P5 decided and shipped, and their tables ready for the paper. Audit A1.

## Phase 2: the exploratory analyses, P6-P11

Each stays labelled exploratory in the reports until its module validates it.

1. P6, the rare target: prevalence and occupancy kept apart.
2. P7, populations: stability across biological repeats (#397) and their enrichment.
3. P8, the covariance fingerprint.
4. P9, beyond the mean: the claim defined first, else it stays exploratory.
5. P10, novelty and combined spaces: a calibrated threshold, and novelty's robust scale
   (the MAD rule left open on 2026-10-13).
6. P11, the separation score.

Milestone M2: P6-P11 decided, each either validated or kept exploratory with its reason.
Audit A2.

## Phase 3: validation on real data

Open-work items 5 (deeper held-out acquisitions), 7 (the owner's reading test), 8 (external
audits of P2 and P6), 9 (P10's intervals), 2 with #81 (multi-group preservation), 10 (a
cost model) and 11 to 13 (excess fields, technical drift, uncertainty semantics).

Milestone M3: each REPORT and OPTIMISE answer read by the owner on a dataset it was not
built on, and the held-out numbers recorded. Audit A3.

## Phase 4: from OPTIMISE to LIVE

A frozen sampling plan between OPTIMISE and LIVE (#83), the implemented strategies
benchmarked for transfer to unseen samples (#84), anytime-valid stopping for LIVE (#265),
open-work items 4 and 14 (O5, O6 and the LIVE wording), the sampling frame (#194 and
open-work item 6), and the external numerical references (#262, #264, #268, #206).

Milestone M4: a rule promoted to LIVE only after it was frozen, replayed on development
data, scored on held-out samples and compared with fixed-FOV baselines. Audit A4.

## Phase 5: release

A freeze of the public interface, a documentation pass, the release audit, and the release
with its notes and the paper's supplementary information.

## Any time

Small, self-contained items that wait for no phase: #404 (`patch_id` as an observation
identity) and #398 (exporting a lasso selection with its source FOVs). Ideas kept for
later, not scheduled: #401, #402, #195, and open-work items 15 to 21.

## Audits

An adversarial audit of the whole package closes each phase (A1 to A4) and precedes the
release.

- **Scope:** every report question followed from the tables to its sentence and to the
  decision it supports, on the scenarios, the lab datasets and the held-out data, plus the
  code-quality tools (`python tools/audit.py`).
- **Evidence:** each finding reproduced before it is reported. Its fix is pinned by a test
  that fails on the old code. Behaviour that should not change is proven with the
  reference tool.
- **Where things go:** the findings in one GitHub issue per audit. Fixes in pull requests,
  their tests in the test file of the module they test. Scientific choices go to the owner
  and are recorded as decisions. When the audit closes, one paragraph in the
  [audit history](../audit/README.md).
- **What is never committed:** audit reports, scratch scripts, intermediate outputs, or
  data that is not ours to publish.

## Keeping the repository clean

- A file is named for what it holds, never for the audit, review, pull request or phase
  that made it.
- A fact lives in one place; other pages link to it.
- Each test lives with the module it tests; a regression test says in its docstring what
  it guards against, not where the finding came from.
- Generated files (the function index, the duplicate report, the baselines) are
  regenerated by their tool, never edited by hand.
- Committed results under `data/` are regenerated in the pull request that changes them.
