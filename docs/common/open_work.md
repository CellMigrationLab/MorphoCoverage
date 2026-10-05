# Open work, in order

What is not yet built or not yet shown, across REPORT, OPTIMISE and LIVE, most important
first. The question codes are those of [the report](../report/questions.md),
[optimise](../optimise/optimise.md) and [the live session](../live/live_session.md).

1. The benchmark behind the remaining **Benchmark** rows: the distribution
   questions of R2, P1 (a method's recommended number against the realized
   need), P10 and O6, which need deeper held-out data. The score and the mean
   and rare-state rows are done (`docs/optimise/benchmarking.md`, "The score").
2. A multi-group outcome for the benchmark: whether a treatment comparison
   survives whole-field sampling. Its first part is
   `retrospective.effect_preservation` (owner, 2026-10-05): for one feature,
   how often `k` fields a side preserve each repeat's difference from the
   control, judged on held-back fields, "not evaluable" when those do not
   establish it (`docs/optimise/optimise.md`). It is standalone; feeding it to
   `optimise` and the report waits for the owner's decision on the
   multi-group outcome contract, which needs its own contract registry
   (AGENTS.md section 9). The object-count version is
   `data/crick_idr/jcs_effect_fovs.py` (the JCS Fig. 3G-I analysis with whole
   fields, `docs/preliminary_findings.md` entry 9).
3. P3 printed as increments.
4. O5, a borderline state in the decision, and O6, the calibration of the
   same rule attached to the live session.
5. Data: deeper held-out acquisitions (30 or more FOVs per sample, ideally 60,
   and five or more independent units) for P2, P10 and the distribution and
   rare rows of R2: the one held-out dataset, idr0139, has 4 FOVs per well,
   so it can refute a development conclusion but not set a number;
   independent labels for rare states; an overview for P9.
6. P9, sampling-frame bias: comparing the imaged positions with every
   candidate position of an overview (with covariates such as density or
   region) needs a new input contract (AGENTS.md section 15) and upstream
   support in CellPatcher, so it is not started until the owner approves it.
7. The owner's reading test: on a real dataset, the project owner reads each
   report answer without the code and says what they would do differently;
   not yet done for any section, nor for the whole report on a dataset it was
   not built on.
8. External audits of the stopping-rule calibration (P6) and the protocol
   reliability (P2), each asked for a dataset on which the answer misleads
   while every test passes (AGENTS.md section 14): planned, and run only as
   part of the whole-report audit of 2026-09-25 (`docs/audit/README.md`).
9. P10's intervals: the effort projection's bootstrap intervals covered the
   true sample and field components in only 80–88% of 40 simulated runs
   (a Monte Carlo standard error of about 5 to 6 points), not 95%
   (`docs/preliminary_findings.md`, entry 6), so a stated difference
   between allocations also needs five or more independent units; a better
   interval is open.
10. A cost model for P10 (`C = n_samples * C_sample + n_FOV * C_FOV`, supplied
    by the user): today allocations are compared at equal total FOVs, which
    is not equal cost, since a sample costs preparation that a field does not.
11. Excess FOVs beyond the adequacy points in benchmark records, and their
    aggregation into a regime map (`docs/optimise/benchmarking.md`): the adequacy
    points exist, the benchmark does not yet report what was imaged past
    them.
12. A diagnostic of technical drift between repeats, batches or plates: a
    shift pooled across groups is invisible to the novelty definition (the
    shifted fields support each other), and the report says "not assessed".
13. Uncertainty semantics: a rule that every interval, bound or percentage
    says what it covers and at what unit, a table in
    `docs/common/scientific_contract.md` of the five kinds (parameter confidence,
    next-field prediction, population tolerance, empirical calibration,
    protocol generalization) with the function behind each, and one
    field-to-field tolerance sentence in the Foundation ("representative
    fields here span X to Y"). Most sentences already say it; the rule is not
    written.
14. The LIVE wording rule: every live sentence names a bound as a nominal
    per-look bound and a stopping risk as empirically calibrated, since
    checking a fixed-sample bound after every field does not keep its nominal
    coverage (with O6).
15. The positive predictive value of a flagged field at a declared field
    incidence (`Setup.rare`), from the control's detection and false-alarm
    rates, printed with its two cautions: it holds for fields flagged by this
    definition of new, and it counts fields, not objects.
16. Representation robustness: a workflow helper that runs the control check,
    and a fingerprint reliability once deeper data allow it, over several
    representations (feature sets, PCA sizes) and tabulates them, with the
    rule that unsupervised clusters are hypotheses about phenotype structure,
    not biological states.
17. A "validated across" block in the protocol summary: for each declared
    level of the hierarchy (field, sample, technical level, repeat), whether
    the protocol curve spans more than one unit of it, or "not assessed".
18. Two process rules for AGENTS.md: a dataset stops being held out for a
    choice once its results influence that choice (section 9 and
    `docs/optimise/benchmarking.md`; no setting has yet been chosen on idr0139), and a
    new method names the microscopist's question it improves that the simpler
    tools cannot answer reliably (section 20).
19. Smaller items left open by the 2026-09-25 audit: advice when missing
    values make the novelty question not estimable (drop those observations
    or those features, and say so); a cap on the support-novelty cost, which
    grows about as `k · n²` per decision (5.9 s at 60 fields of 300 objects);
    a warning in `sufficiency_report` when only per-object rules are given,
    and the README example with both units; and a sentence that the
    next-field calibration is noisy when few fields remain.
20. The candidate estimators not yet built (learning curves, kNN and PRDC
    support, abundance-based and diversity estimators, occupancy models,
    spatial ICC, pMoSS, biological-fidelity outcomes, continual-monitoring
    methods) and the benchmark they must pass:
    `docs/proposals/candidate_estimators_roadmap.md`.
21. A calibrated near-duplicate detector (fields whose distributions are
    near-identical but not identical), opt-in and calibrated on imaging
    datasets before any threshold is used; the exact integrity checks are in
    `docs/report/integrity.md`.
