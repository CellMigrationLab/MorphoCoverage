# Stopping rules: reading a stop, and how often it was too early

A stopping rule says continue or stop after each field. This page is how to read its
decision, what a stop rests on (the order the fields were acquired in), and how its stops
are checked on completed acquisitions: how often a rule stopped before held-back fields
showed the target was met.

## Reading a decision

Every `StrategyResult` has `explain()`: one line with the decision, each
policy's rule beside its evidence, and the effort, for example
`stop: lofo_novel_fov_upper = 0.24 (stop at <= 0.25, distribution); after 11
FOVs, 1100 cells; treats the fields as a random sample of the well's
positions; a raster over a spatial trend can stop early`. Every stop ends with
what it rests on (see [Acquisition order](#acquisition-order-random-or-dispersed)),
and a rule that treats objects as independent draws (the mean-precision rule per object)
says so. When the rule decides some of the objective's targets but not all (a
mean-precision rule under a declared rare target) the line says
`stop (decides the population summary only): ...`; when it decides none of
them (a novelty rule under the population objective) it says
`unsupported (the rule itself: continue): ...`, so the rule's own verdict is
never hidden behind the objective's.

Several rules on one acquisition state are read together with
`morphocoverage.results.decision_view(results)`: one entry per target
(population summary, phenotypic fingerprint, rare phenotype), the decision
combined over the rules that decide it and the criteria that decided it,
then the overall decision for the objective; a required target no rule
decides is printed as such and keeps the overall at `unsupported`. The JCB
criterion is named as what it is: an averaging CV of 5% is one standard error
of 5% of the mean (the paper's criterion), not a bound of ±5% on the mean's
error; whether the mean was within the declared tolerances is judged on
held-back fields by `report` and `optimise`. `LiveUpdate.summary(detail="full")` ends with
that view, one entry per line ([O4](live_session.md#the-questions)); the default short block opens with
the field and its decision (`✅ field 8 · MOCK · 1A · B6 — stop: all 3 rules say enough`, `⏳
continue: ...` with what is not yet met, `⚠️` at a stop the order check flags, `➖` while a
declared target has no rule), then one table of the rules and, at a stop, each caveat once. `live_session(setup)` runs under
`setup.objective()`, whose components are the declared targets. Read the quantity, not only the word: with nothing new among `k`
FOVs the novelty bound is `1 - 0.05 ** (1 / k)` whatever the data, so a stop
on it after 11 FOVs is a statement about the next field, not about rare
states (`docs/preliminary_findings.md`, entry 3). A rule's threshold is its
own; whether its stops were premature is the retrospective calibration
(`morphocoverage.retrospective.stopping_calibration`).

## Acquisition order: random or dispersed

Every stopping rule reads the fields seen so far as a random sample of the
well's positions. A raster that tiles the well row by row breaks that when
the readout follows a spatial trend (a gradient, an edge, an empty region) or
drifts over the acquisition: the first rows look settled, and the default JCB
pair can say stop after one or two rows while the well's mean is far off. In
audit 4 (a 10 x 10 well, a gradient of ±20%), the overall stop was more than
5% off in every raster run, against 16% of runs with the same wells in random
order; a rare state confined to the last rows was declared ruled out in every
raster run before the raster reached it, and in no random-order run.

**Acquire fields in a random or dispersed order** (for example a random
permutation of the planned positions, or a farthest-point order of them,
`morphocoverage.retrospective.stage.dispersed_order`); that makes the rules' premise true instead of hoping
it holds.

`live_session(setup)` also checks the fields of each group so far, and
`LiveUpdate.summary()` prints the result once at a stop (`detail="full"`: beside every stop,
in each rule's line and after the overall decision):

- with `Setup.acquisition_order_col`, a trend of the field means of the
  readout with acquisition order (Spearman);
- with `Setup.stage_cols`, alike means in neighbouring fields (Moran's I,
  neighbours within 1.5 times the median nearest-neighbour distance, a
  permutation p-value with a fixed seed), and beside a rare-phenotype stop the
  stage region the fields have covered (in the report: the share of the
  acquired fields' stage extent covered at the recorded order's stop);
- with neither, the stop says that it treats the fields as a random sample of
  the well's positions, and that a raster over a spatial trend can stop early.

The two tests share a 5% level (Bonferroni), and each needs 4 fields with a
mean. The check **flags, it never vetoes**: a flagged stop stays a stop. A
check that finds nothing does not show the fields to be representative: in a
raster over a gradient it has little power in the first rows, where the
early stops happen. It reads only the fields seen so far, so a replay of the
same prefix gives the same check (`order_check.check_acquisition_order`).
`report` prints the same caution beside each rule's stop ("Did each
criterion say enough?"), computed on the group's acquired fields.

## Does this stopping rule work? Calibration and adequacy points

A stopping rule can depend on the order in which FOVs arrive, so it is checked sequentially, separately from sampling reliability (P6 for the calibration; R2 and P4 for the adequacy points). `morphocoverage.retrospective.replay_acquisitions` holds back the same fixed reservoir per split as `sampling_reliability` and replays the stream in the recorded order (when an acquisition-order column is given) and in seeded random orders (10 by default). After every FOV it records each strategy's decision, computed from the visible prefix only, and, separately, whether the prefix preserves each `OutcomeCriterion` against the reservoir. Outcomes never depend on a strategy.

- `stopping_calibration` finds each rule's first stop and reports, over replayed acquisitions, how often it stopped, the median FOVs at the stop (with its Monte Carlo standard error over the reservoir splits, each split's median stop counting once, since its orders share one reservoir; owner, 2026-10-05), and the shares of correct stops (preserved at the stop), stops that could not be judged, and runs that never stopped, with whether those had it by the last FOV. The headline risk is the share **not preserved at the stop**. It splits into premature stops (the whole stream was not shown to fail the criterion) and stops the whole stream did not fix either (not preserved by every stream FOV together). The held-back reference is itself a finite sample, so its noise alone can put the whole stream outside the tolerance: the premature share is a lower bound on the stops that came too early, and the total not preserved at the stop an upper bound; the summary prints both. For a fixed-FOV rule in random orders the share of correct stops estimates the same quantity as `sampling_reliability` at that `k`; a test checks this.
- `adequacy_points` gives, per replayed acquisition, the first and the sustained apparent adequacy (from decisions only; a later continue, unsupported or undefined decision breaks a sustained stop) and the first and the sustained realized adequacy (from outcomes only, for all criteria together and each on its own; a later failure breaks it, an undefined outcome is skipped). Sustained realized adequacy is the preferred answer to "how many FOVs did this experiment really need?". For example, decisions that say stop at FOV 16, stay stop to 23, continue at 24 and stop from 25 on give a first apparent adequacy of 16 and a sustained one of 25; outcomes preserved at 10 and 11, not at 12 and from 13 on give a first realized adequacy of 10 and a sustained one of 13. The apparent points use only decisions and the realized points only outcomes; they are never mixed.

Both have a `summary()` in plain sentences. Replayed acquisitions are hold-outs and orders of the same FOVs, not independent experiments.
