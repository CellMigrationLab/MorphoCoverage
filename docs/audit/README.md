# Audits: a short history

Four audits of the package were written between 2026-09-25 and 2026-09-26.
Each read the code against what it claims and prints, reproduced every
finding before fixing it, and pinned each fix with a regression test that
fails on the old code. This page keeps what each one was, its most important
fixes and where its tests are. The full texts, with every finding, are in git
history; the commands below print them. The work they left open is in
[`docs/questions.md`](../questions.md), "Open work".

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
