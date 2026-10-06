# The setup: declare the dataset once

REPORT, OPTIMISE and LIVE all start from one `Setup`: the readout, the features, the
control, the targets the acquisition must preserve and the columns that place each field.
This page is every field of it. `scan_design` and the guided session build one for you
([Running a report](../report/running_a_report.md)).
[What happens for what you provide](what_happens.md) maps each field to the analyses it
switches on.

## The `Setup`

The workflows' scientific choices are the `Setup` fields and explicit
arguments. Two provisional presets have defaults, `optimise`'s
`protocol_threshold=0.9` and `benchmark`'s `alphas=(0.05, 0.10)`, and the
summaries print them. The analyses they call are the ones documented
elsewhere, and their results stay available under `results`.

```python
from morphocoverage.workflows import PositiveControl, Setup, default_novelty

setup = Setup(
    readout="YAPTAZ_Ratio",                  # the scalar you report
    features=features,                        # explicit columns; never inferred
    control={"condition": "MOCK"},            # negative-control groups
    normalize="control_mean",                 # or None: divide by the repeat's control mean
    positive=PositiveControl({"condition": "YAP"}, fovs_seen=10, pair_cols=("biological_repeat",)),
    novelty=default_novelty(features),        # the definition of "new"; None to skip
    tolerances=(0.05, 0.025),                 # the readout's mean within 5% and 2.5%
    acquisition_order_col="acquisition_order",
    stage_cols=("stage_x_um", "stage_y_um"),
    group_cols=("condition", "biological_repeat", "technical_repeat"),  # the well: a technical repeat
    observation_unit="nucleus",
    n_jobs=-1,
)
```

Every field left at `None` switches off the analyses that need it. The report
then says "Not assessed" and names what would be needed. `observation_unit`
names what a row is, and the report's words follow it: "every patch a point",
"Are there groups of tracks?", the nucleus-level test (`Setup.unit_words`,
`Setup.unit_word`); left at `None`, the rows are analysed as generic objects
(`observation_unit_declared=False`) and the report calls them cells. The stopping rules
default to the mean-precision rule (upper-95 SD, first tolerance), per object and per
field; pass `rules=(...)` to assess others. `random_state` (default 0) seeds
every random procedure the workflows run, the report's random orders
included, and `minimum_detection` (default 0.9, a provisional preset) is the
share of positive-control checks the representation must recognize before the
adequacy answers are read without a warning. The control's rates are pooled over
its strata (the repeats, or `pair_cols`), and the report says "pooled over all
biological repeats" wherever it applies one to a group: the warning, the
detection-adjusted bound, the false alarms expected among the fields to check,
and question 1 (owner, 2026-10-05). With planted objects declared, their rate is the
warning's; when none could be planted, the warning on questions 2, 4 and 5 reads
"not assessed: the planted check could not run", never the whole-field rate
(owner, 2026-10-05). A `PositiveControl` without
`pair_cols` pairs its fields with reference fields within each repeat when the
FOV table has a repeat column; `pair_cols=()` pools the repeats, as an
explicit request. The declaration prints every such preset.

`default_novelty(features)` standardizes robustly (median and MAD, the
`RobustStandardize` transform), so a few extreme objects such as clumps or
debris cannot widen the scale and hide a new state elsewhere. It was chosen
on idr0028 with the mean and SD, the version behind the numbers of
[`preliminary_findings.md`](../preliminary_findings.md) until they are
regenerated after audit 4. Its radius is `sqrt(len(features))`: with 8 or more standardized
features two objects of one ordinarily variable state often sit farther
apart than that, and `Setup` warns (reduce the dimension, for example
`Chain([RobustStandardize(), PCA(k)])`, or declare fewer features).

`fields_independent` (default `False`) declares that the fields are
independent draws of their condition. Only then do the condition contrast and
the feature ranking give a p-value when the fields have no unit to shuffle
(see [question 9](../report/conditions.md#within-each-repeat)): fields of one dish or well are not independent draws.

Features measured plane by plane are split before the `Setup`:
`morphocoverage.planes_as_features` turns `area` into `area_z0`, `area_z1`,
... (objects matched by `label_id`), and those names are the `readout` and
`features` declared here ([`data_format.md`](data_format.md#features-measured-per-plane)).

## Features from explainable tables

Features may come from several tables, each tagged explainable or not (owner, 2026-10-06;
[feature tables](../report/feature_tables_and_maps.md#explainable-and-non-explainable-tables)).
An explainable table's columns are declared in `features` (or as the `readout`) under their
names in the report (a name also in another table is prefixed, `"fiberfeatures: area"`).
With explainable tables given and no `features` declared, every column of the explainable
tables is a feature; the `readout` is still declared, and a Setup without one is refused
when explainable tables are given. The workflows record the tables in
`Setup.explainable_tables` (name to final columns); `Setup.feature_tables` keeps naming the
non-explainable ones.

```python
setup = Setup(readout="area", control={"condition": "ctrl"})  # every fiberfeatures column too
result = report(objects, fovs, setup, feature_tables={"fiberfeatures": (fiber, "explainable")})
```

## Embedding-only reports (no explainable features)

A dataset may have no feature to read one by one, only an embedding such as DINO (owner,
2026-10-06). Declare no readout and no features, and give the tables to the workflows:

```python
setup = Setup(control={"condition": "MOCK"}, observation_unit="image", fov_col="image_file",
              fingerprint=fingerprint)              # and/or rare=..., rare_label="mitotic"
result = report(fovs, fovs, setup, feature_tables={"DINO": dino})   # the table of images
html_report(result, fovs, fovs, "results", feature_tables={"DINO": dino})
```

`readout=None` (its default) with no `features` makes such a Setup (`Setup.embedding_only`);
features without a readout and `intensity_features` are refused. `normalize` is accepted:
it centres the tables per biological repeat (for the maps, the populations, unusual
observations and the separation, never a test; without it, unusual observations read each
column relative to its own control's mean in its repeat) and never divides them, since the other
tables are never divided by a control. The targets are the user's choice among those
that need no single readout: the **whole distribution**, the fingerprint of each table
(read on its first principal components,
[feature tables](../report/feature_tables_and_maps.md#other-feature-tables-embeddings)), a
**labelled phenotype** (`rare`, `rare_label`) when a column marks one, or both; averages
are not offered. The mean-precision rules read the readout, so they are not applicable,
which the declaration says once, and there is no default stopping rule.

`report`, `html_report`, `scan_design` and `guide` accept such a Setup with
`feature_tables=`. In the report, "Do the conditions differ?" is answered by the
[whole-table test](../report/conditions.md#the-whole-table-test-other-feature-tables) of
each table, its only test, which the answer leads with; "Did I image enough?" by the
declared fingerprint and labelled phenotype; unusual observations are judged on the tables'
weighted combined space (one table: its columns), and the maps, the populations, the
separation and the copy checks read the tables as in any report. Each part that needs
explainable features says "not applicable: no explainable features" once, with the ➖
marker: the Foundation, question 1, the fields to check, and per feature the discovery, the
heatmap, the SuperPlots and "Did I image enough?". The guided session, when the
observation table holds no numeric feature (a 0/1 column marking a phenotype aside) and
other tables are given, skips the feature and readout questions, asks for the phenotype
column, and offers the aim among the whole distribution, the labelled phenotype and both.

`optimise` judges the declared targets on held-back fields (the tables' columns already in
`objects`: `attach_feature_tables(objects, fovs, tables).objects`, with
`Setup(feature_tables={name: columns})`); there is no stopping rule to replay or effort
projection (both read the readout), and it refuses a Setup with no target it can judge.
`live_session` has no default rule without a readout: pass `rules=(...)` on the tables'
columns (a `FOVBootstrapStrategy`) or the label (a `RareIncidenceStrategy`); it refuses
otherwise, saying so. With both the fingerprint and the rare phenotype the objective is those
two, `distribution_rare`, with no population component, and the declaration lists only them
(owner, 2026-10-06); `objective=` may still narrow it.

## Normalising to the control

`normalize="control_mean"` divides every declared column (the readout and
the features) by the mean of the control condition's objects of the same
repeat, for the analyses that compare values across repeats (below), as the
mean-precision rule and the lab's Plot&Stats notebook express values
(`morphocoverage.normalize`). It needs
`control` to name the condition column, every repeat must hold control
objects (a repeat without them raises, since mixing scales inside one
analysis would be worse than stopping), and the report's overview and the
declaration say that the values are normalised. A control mean that is zero
(up to rounding), negative or undefined by missing values is refused, naming
the feature and the repeat: dividing by it has no meaning or flips the sign of
every effect, so compare a signed feature (a velocity, a log-ratio) raw. When
the control objects hold negative values but their mean is positive, the
values are still divided, with a warning that they are then not multiples of
the control, and the sentence on normalised values says so (`Caution: ...`). What reads
the normalised values is listed once, in `_contrasts.normalised_readers`, which the
overview and the declaration print (audit 8B). With one control, the figures, the
normalised maps, the populations, the separation, unusual observations, new states for the positive control and the fields to check, the
fingerprint and the outcomes judged on held-back fields read the normalised values; the
condition contrast, the feature ranking and the discovery read the raw values and
normalise them themselves; the stopping rules, the acquisition anomalies and every other
analysis read the raw values. With several controls, the figures, the normalised maps, the
populations, the separation, unusual observations, new states for the positive control and
the fields to check, the fingerprint and the outcomes judged on held-back fields read the
normalised values; the condition contrast, the feature ranking and the discovery read the
raw values, each normalising to its own control; the stopping rules, the acquisition
anomalies and every other analysis read the raw values, where levels across experiments
are read. Unusual observations (biological unusualness) and the new states judged for the
positive control and the fields to check read the normalised values, the acquisition
anomalies the raw ones, as quality control (owner, 2026-10-06, plan v2). The fingerprint and the held-back outcomes read the normalised values
whatever the number of controls (owner, 2026-10-06, audit 8; before, the raw values with
several controls, so their scale changed with the number of controls). The condition
contrast reads the raw values and normalises them itself, so its raw columns hold raw values and its normalised ones the values divided by
the control mean of the same repeat (owner, 2026-10-05; before, it read the normalised
values, so its "raw" d and paired t were the normalised ones).

Several controls are declared as `control={"condition": {control: {treatment: condition}
or [conditions], ...}}`, with `control_parents` and `master_control`
([conditions.md](../report/conditions.md#several-controls)). One control declared that way,
as a list, a keyed dict or with itself as the master, and without parents, is
`{"condition": control}`: the `Setup` reduces it, so the output is identical to the one
control's (owner, 2026-10-05).
Every stopping rule runs on the raw values, in `report`, `optimise` and
`benchmark` as in the live session (which sees one field at a time and has not
seen the control), so a rule gives the same verdict everywhere; the other
analyses (variability, controls, inspection, effort projection) read the raw
values too.

## What must the acquisition preserve?

The `Setup` also declares the target, the front door of every
workflow: there is no "enough" without "enough for
what". Three targets can be declared, and each becomes an outcome criterion
that `optimise` and `benchmark` judge on held-back fields (`setup.criteria()`):

- **a population summary**: the mean of `readout` within each of
  `tolerances` (declared with a readout unless `tolerances=()`, which declares no mean
  target and so no default mean-precision rule, beside a fingerprint or a rare target: the
  guided "whole distribution", owner 2026-10-06; the declaration says so once, `optimise`
  then judges the other targets and `live_session` needs `rules=`; none in an
  [embedding-only Setup](#embedding-only-reports-no-explainable-features));
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
  object column (booleans or 0/1) that marks it; a missing value is unknown, never a
  negative: the cell is left out of every share's numerator and denominator (the label
  check, the groups, the share compared, the plan, the fields holding it), and the report's
  notes count such cells once (owner, 2026-10-05). With the label, held-back
  fields judge whether `k` fields showed the state; without it the target is
  stated and planned for (the binomial field count at the declared incidence)
  but not judged. Object prevalence is never converted to field incidence.

A fourth target, the **difference between conditions**, is reported
whenever a control condition is declared and others exist (R7); as a
preservation outcome for OPTIMISE it waits on the multi-group outcome
contract (open work item 2).
The live decision runs under the objective made of the declared targets
(`Setup.objective()`: `population`, `phenotype`, `balanced` or `all`), and
`decision_view` shows each target's decision and the overall (O4).

The report answers the declared targets beside the mean, from the fields acquired: in
[Did I image enough?](../report/did_i_image_enough.md), [Do the conditions
differ?](../report/conditions.md), [populations](../report/questions.md) and [whether the
features detect what you care about](../report/questions.md).

Every document opens with the declaration (`setup.declaration()`), one
sentence per target, and a target that is not declared says so, so that an
answer about the mean is never read as an answer about the distribution or
about rare states. The declaration also names the JCB criterion as what it
is (an averaging CV of 5% is one standard error of 5% of the mean, not a
bound of ±5% on its error; the tolerances are judged on held-back fields) and
has an "Acquisition order:" line: the order check printed beside every stop
(with `acquisition_order_col`, a trend of the readout's field means with
order; with `stage_cols`, alike means in neighbouring fields), or, with
neither column, that every stop treats the fields as a random sample of the
well's positions ([Stopping rules](../live/stopping_rules.md#acquisition-order-random-or-dispersed)).
A flagged trend is a caution beside the stop, never a veto; image fields in
a random or dispersed order.

A declaration the workflows could not use raises instead of being skipped,
and so does a column named explicitly that the tables do not hold (a repeat
or technical-repeat column, a control column, a plot column): only a column
left at its MorphoBricks default may be absent, so a misspelt name never pools
what it should keep apart.
A definition of new (`novelty`) needs `features`, and a positive control needs
`novelty` and at least as many reference fields seen (`fovs_seen`) as the definition
judges a field after (its `min_fovs`, 5 for `default_novelty`): with fewer, no field
would ever be judged. With `novelty`, the report also counts the fields holding a state
that fewer than `few_fovs` other fields show (seen, but not established);
`few_fovs` must exceed the definition's `min_other_fovs` and defaults to two
more than it. When every field holds one observation the report reads the definition
across fields (owner, 2026-10-06, plan v2): a new state's `min_novel_observations` similar
observations cannot sit in one field, so a field's observation is new when no other field
holds one within the radius, a state seen once (`support_novelty.per_field`:
`min_other_fovs` and `min_novel_observations` 1), exploratory; the estimator applies it
itself on such data, so `optimise`, `benchmark` and a live session judge alike, and the
declaration says "new phenotypes, exploratory (new: a cell that no other field shows)".
Like every stopping rule, the novelty rule reads the raw values, as live does (owner,
2026-10-06). Group columns beyond condition and repeat (the `technical_repeat`, say) are part of
a field's identity everywhere, including in the control check. "New" is judged
on `features` only, never on the readout, and the report names those columns.
The stopping rules see the same columns (the readout, then `features`) in
`report`, `optimise`, `benchmark` and the live session, so one rule gives one
verdict whichever workflow runs it.
