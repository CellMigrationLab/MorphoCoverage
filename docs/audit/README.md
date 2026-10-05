# Audits: a short history

Six audits of the package were made between 2026-09-25 and 2026-09-28.
Each read the code against what it claims and prints, reproduced every
finding before fixing it, and pinned each fix with a regression test that
fails on the old code. This page keeps what each one was, its most important
fixes and where its tests are. The full texts of the first four, with every
finding, are in git history, and the commands below print them; the fifth and
sixth were never committed, so this page and their commit messages are their
record. The
work they left open is in [`docs/questions.md`](../questions.md), "Open work".

**2026-09-25: the microscopist audit.** An adversarial audit of the three
moments of use (report, plan, online) at commit `ec80e53`: every question was
followed through the code that computes it, the sentence that prints it and
the decision it supports, then run on idr0028 and on constructed acquisitions
whose truth is known. The most important fixes:

- F1 (critical): the per-object JCB verdict said "enough" after one field
  without its assumption; it now says it treats observations as independent
  draws from a spatially uniform sample.
- F2, F3: the protocol turned a reservoir ceiling into "every well needs more
  than 18 fields", and the shipped reliability curves omitted their ceiling;
  samples capped by their reference are now named apart, and a curve without
  its ceiling says so.
- F4: "98% of the variance is between samples" from three wells is now marked
  "not established".
- Later passes: states seen in few fields and a planning line for them
  (`few_fovs`, F5, F11), a weak positive control qualifies the answers (F6),
  a one-line reason on every result (`StrategyResult.explain`, F7), and
  idr0139 registered as a held-out dataset.

Tests: `tests/test_microscopist_audit.py`. Added in `9658c29` (merged in PR
#34, `4865096`); its second and third passes came in PR #36 (`f503570`). Full
text: `git show de08769:docs/audit/microscopist_audit_2026-09-25.md`.

**2026-09-26: the foundation assessment.** An assessment, not an audit of
defects: every claim of the proposal to make "why FOVs matter" the foundation
of the report, instead of one question among others, was checked against the
code, the documents and the committed results; a second part checked ten
principles of the *Points of Significance* series the same way. It
recommended the Foundation section, the declared target in `Setup` (the
population summary, the phenotypic fingerprint through a new `distribution`
component, and the rare phenotype), the condition contrast as a report
section, and "what this dataset cannot tell me". These were built in PR #43
(`210f227`), with tests in the modules they touched (`tests/test_workflows.py`,
`tests/test_sampling_unit.py`, `tests/test_objectives.py` and others); the
assessment itself fixed nothing and has no regression tests. Added in
`22eb0e9` (its second part in `87b9be2`), merged in PR #43 (`210f227`). Full
text: `git show de08769:docs/audit/foundation_proposal_assessment_2026-09-26.md`.

**2026-09-26: the first function audit.** Every module read function by
function at commit `0728583`, looking for places where the code, its
docstring and its sentence disagree, or where a reasonable input gives a
wrong number, a crash or a silent omission; a ninth pass traced one dataset
from the tables to the report, and a synthetic sweep made the mistakes people
make with such tables. The high-severity fixes:

- C1: the per-field precision rule stopped after one field when the other
  fields were empty.
- E1: the sufficiency report printed "yes, met" and "not met at any point"
  for the same rule (`run_benchmark(stop_on="rule")`).
- F1: the effort projection understated the field term when fields hold
  unequal numbers of objects.
- H1: pooled plots merged same-named wells across repeats, so their
  intervals were too narrow.
- H2, H3: one rule was judged on three different feature sets depending on
  the workflow; every rule now reads `Setup.columns`.

Tests: `tests/test_audit_strategies.py`, `tests/test_audit_sufficiency.py`,
`tests/test_audit_planning.py`, `tests/test_audit_report.py` and
`tests/test_audit_workflows.py` (groups A and B in the module tests, the sweep
in `tests/test_end_to_end.py` and `tests/test_human_errors.py`). Added in
`81ae29f` (merged in PR #46, `e78b520`). Full text:
`git show de08769:docs/audit/function_audit_2026-09-26.md`.

**2026-09-26: the second function audit.** Ten passes at commit `07c20f5`:
nine read the package module by module, checking each function against its
docstring, its sentence and the mathematics it claims; the tenth read the
package as a whole for silent fallbacks, duplicated functions, import cycles
and misplaced functions. Shared statistics, bounds, wording and seeds were
moved into one place each (`statistics`, `bounds`, `_wording`, `_seeds`,
`_checks`) before the per-group fixes. The high-severity fixes:

- B-1: a live session counted 12 z-plane files as 12 fields; the session now
  applies the plane rule.
- C-1: empty fields entered the per-field precision rule's variance factor
  and degrees of freedom, so its nominal 95% bound covered 37–49% of the time.
- E-1: normalising to a negative control mean flipped the sign of every
  effect; such a control mean is refused.
- F-1: an empty subset was scored "not preserved" when the reference left the
  metric undefined.
- G-1, G-13: the effort projection assumed equal sample and repeat sizes, and
  "expected by chance" used the wrong false-alarm rate.
- H-1 to H-4: permutation p-values ignored infinite shuffles and float ties,
  one technical repeat on a side was shuffled as if its fields were wells,
  and a constant feature passed as variable through rounding noise.
- I-1: with normalisation, rule replays read normalised values while the
  live session reads raw ones; every rule now reads raw values.

The benchmark was rerun under protocol version 3 afterwards. Tests:
`tests/test_audit2.py` (the shared foundations) and `tests/test_audit2_a.py`,
`_b`, `_cd`, `_ef`, `_g`, `_h` and `_i` (one file per group of findings; the
test names carry the finding ids). Added in `6a42b43` (merged in PR #48,
`de08769`). Full text: `git show de08769:docs/audit/function_audit_2_2026-09-26.md`.

**2026-09-26 to 2026-09-28: the third function audit.** Eleven passes, A to K,
from commit `2abde25`: nine read the package module by module as the second
audit did, J read its structure (duplicated logic, private names used across
modules, misplaced functions, the 600-line rule) and K read every document
against the code. They reported about 200 findings, four of them
high:

- B-1, B-2: the live session read `z`/`c`/`t` only from the observations and
  resolved the field and groups from the FOV row alone, so it accepted tables
  the validator refuses and counted plane files as fields.
- G-1: the report's "expected by chance" used a false-alarm rate measured over
  another reference set (the control's pairing stratum, by default the whole
  table, not the analysis group the inspection judges each field in).
- H-1: when `group_cols` left the wells out, the condition contrast shuffled
  fields as if they were wells, and 42% of null contrasts came out
  significant at p <= 0.05.

The fixes began with the foundations the groups share (`5e80d25`): bootstrap
intervals over few units are widened by `sqrt(k / (k - 1)) * t_{k-1} / z`, so
a nominal 95% covers 92-95% at 3 to 20 wells instead of 74-93%; groups are
looked up by `_seeds.group_key`, not by the 32-bit seeding hash; each split
is seeded with `split_seed(random_state, split)`, so neighbouring seeds no
longer share reservoirs; and one helper writes plurals and percentages
(`_wording`), declared rates exactly. Then, by group: the live session
resolves identity with the validator's resolvers over the observations and
the FOV row, and applies the plane rule to both (B); `z`, `c` and `t` are
never features, and `planes_as_features` pivots a feature per plane on
request (A); the control rates behind "expected by chance" are measured
within the inspection's groups (G); the FOV table's technical repeat is
always the contrast's and the ranking's unit, which brought the null false
positives from 42% to 0% (H); and the JCS rules report location and spread
quantities as population and distribution evidence (C). The project
owner decided the findings that needed a choice: the FOV-table technical
repeat is the unit whatever `group_cols` says (H-1); the small-sample
widening applies to every whole-field and nested bootstrap (E-2, H-3); a plane
coordinate is refused as a feature and the per-plane pivot is opt-in (A-3); a
feature constant so far is not yet enough data (C-1); no finite-population
correction against a reference that does not contain the stop (D-1); a
control holding negative values is still divided, with a warning (E-7); a
subset too small for its metric counts as not preserved (F-11); and the
report clusters the declared rare label (I-7). Version 4 of the benchmark
protocol was locked for the replays these fixes changed (`a65a226`).

The audit also ended the series. The bug classes it kept finding again, in a
place the last fix had not reached, are now guarded package-wide by
`tests/test_guards.py` (`61a7314`): the live session gives the replay's
result at every step of every contract scenario, `check_tables` accepts
exactly what the validator accepts on mutated tables, counts and numbers are
checked only through `_checks` and percentages written only through
`_wording`, the seeding hash is never a lookup key, only `metadata` resolves
the field, modules stay under 600 lines, and the Clopper-Pearson and
bootstrap intervals cover their level. No further broad audit is planned:
the guard tests catch the next instance of each class, and later audits read
only the code that changed. Tests: `tests/test_guards.py`, and
`tests/test_audit3.py` (the foundations) with `tests/test_audit3_a.py` to
`_i` (one file per group; the test names carry the finding ids). Committed in
`5e80d25` to `a65a226`, whose messages summarise the fixes group by group.

**2026-09-28: the fourth, adversarial audit.** Nine passes from commit
`2f5847a` (with PR #51, the integrity checks), each trying to make the
package give a wrong answer a microscopist would act on, on data a lab could
plausibly produce and whose truth was known: the report (P1), live stopping
(P2), the calibration of the bounds (P3), the stopping rules (P4), input
traps (P5), novelty and rare states (P6), integrity and `check_tables` (P7),
the documents' claims against their evidence (P8) and reproducibility (P9).
They reported 109 findings (18 high, 45 medium, 46 low), every high one
reproduced before fixing. Among the high ones: the contrast shuffled fields
as independent draws unless the wells were named `technical_repeat`; a raster
over a spatial trend let the per-field rules stop early with nothing saying
the fields must be a random sample; the per-field JCB bound under-covered
with unequal field sizes; the JCS and bootstrap rules stopped on a readout
constant so far; rows of every z-plane in one field counted as objects; the
rare-state field count was driven by ordinary unusual cells; and the scaled-copy
check compared every pair of fields.

The fixes came in eight groups: integrity and duplicate checks (A), input
traps (B), the sampling-unit analyses (C), the stopping rules (D), the
acquisition-order check beside every stop (E), the live sources (F),
novelty and rare states (G), and the retrospective analyses and the benchmark
(H). The project owner decided the choices: a well column in `group_cols` is
the contrast's unit, and without a unit there is no p-value unless
`fields_independent=True` (Q1); an order check flags a trend beside every
stop, never vetoing it, and the documents recommend random or dispersed
orders (Q2); the per-field bound takes Kish's effective number of fields as
its degrees of freedom (Q3; replaced by a bootstrap of whole fields, owner,
2026-10-05); and the recommended defaults D1 to D20, among
them robust standardization in `default_novelty`, per-plane rows refused,
`watch(until_stop=True)` waiting for every group, and Monte Carlo errors
beside every reliability share, with each group seeded from its labels
written canonically (D16). Benchmark protocol version 5 was locked for the
changed rules (`e30b8a6`). Tests: `tests/test_audit4_a.py` to `_h` (one file
per group; the test names carry the finding ids). Committed in `98f6006` to
`e30b8a6`, whose messages summarise the fixes group by group.
