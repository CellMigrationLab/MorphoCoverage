# Did I image enough?

Question 2 of the report asks whether the fields acquired were enough for what you
declared: the difference from the control, the mean of each feature, a rare phenotype,
a fingerprint, and whether one more field would likely have shown something new (R2 and
R3 of [the report's questions](questions.md)). This page gives the answer on the page and in the text report, then the function
behind the stopping rules and the next field, `sufficiency_report`.

## On the page

"Did I image enough?" answers in one sentence (owner, 2026-10-02): within each
biological repeat, for which features the fields imaged show the difference from
the control in every comparison (a condition against the control in one repeat), in
some or in none; with fewer than 3 biological repeats it first says that no
difference can be called significant. With a rare phenotype or a fingerprint
declared, a line each says in how many wells it was seen in enough fields, or is
within its tolerance. A compact table follows, one row per feature: the comparisons
shown and what would show the rest (how many comparisons would need about how many
fields a side, or cells, tracks, patches when the comparison counts those, how
many are too small to show even with 4 times as many, and how many more fields will
not resolve: more wells or biological repeats). The fields a side are projected from
the part of the interval of d within wells only (owner, 2026-10-05): the interval
comes from resampling the wells and then their fields as acquired, fields without
cells included, and a nested bootstrap splits its variance into a part between wells,
which more fields of the same wells do not reduce, and a part within wells, which
shrinks as 1/fields. When the part between wells alone is as wide as the difference,
the answer is that more fields will not resolve it: more wells or biological repeats.
With one well a side the interval holds no variation between wells, and the
projection is from its fields alone. The plan's repeats needed with twice the fields halve
only that part within wells too. Then one feature at a time,
as tabs: its comparisons, the precision of each well's own mean and its table within
each repeat; the rare phenotype and the fingerprint have their own tabs after the
features. What every tab would repeat (that this is per repeat, what each well's own
mean is for) is said once, in the card's "?", and the assumptions of these answers
once, at its end. With one feature the card is the same, the comparison table given
once, in full. The stopping rules, the repeats a new experiment would need and the
plan `optimise` would make are not in this card: they are in the text report
(`report_full.txt`) and, on the page, in its last section, "Plan the next experiment".

## A rare phenotype and a fingerprint

The covariance fingerprint and the rare target are exploratory (plan v2, owner, 2026-10-06):
not yet validated by the benchmark of [the validation protocol](../common/validation.md); the
answer says so once, in one line, as it does for novelty below.

With a rare label or a fingerprint declared, the answer says, per well, whether the labelled state was
seen in at least `k` fields and how many fields would see it there with the
declared probability, at the share of fields holding it in the data; and
the fingerprint's error with the fields acquired (estimated from random
halves of each well's fields) against its tolerance, with the fields that
would bring it there. A well where more than 10% of the 20 random halves
have no defined error is not judged, and named; below that share the error
comes from the other halves, and the text says the share (owner,
2026-10-05). When no well can be judged, it reads "not judged in any well"
with the reasons, marked ➖. When the fields `optimise` holds back (35% of a well's)
would themselves err by more than the tolerance, the text report says so
once: the held-back judgement cannot confirm that target, and the estimate is
the guide. On the page each target has its own tab, after the features, and a
line at the top of the card. A covariance criterion, the one the guided session declares,
is named the covariance fingerprint ("The covariance fingerprint of DINOv2 (...)"): it
reads how the features vary together, not their whole distribution (plan v2, owner,
2026-10-06). When every declared target is not assessable, an embedding alone with "the whole
distribution" (`tolerances=()`) on groups too small for its components, the answer gives
each reason and says that no declared target can be judged on these data; `optimise` refuses
with the same words.

## The stopping rules and the next field: `sufficiency_report`

Novelty, the chance that one more field shows something new, is exploratory (plan v2, owner,
2026-10-06) until [the validation protocol](../common/validation.md) validates it.

`sufficiency_report(objects, fovs, feature_cols=..., strategies=..., objective=..., orders=..., novelty=..., novelty_target=..., calibration=...)` answers two questions per analysis group of an analysed dataset:

1. **Did each criterion say enough?** (`enough`) For each criterion, judged by its own threshold and not by held-out FOVs: `sampled_enough` (`yes`, `no` or `undefined` on the acquired FOVs; a fixed-effort rule, an `EffortPolicy` or a strategy flagged `effort_rule`, reports `target reached`, since reaching a fixed effort says nothing about the data), after how many FOVs it was first met across the declared acquisition orders (median and 10-90% range), and the FOVs imaged beyond that point. The recorded and the permuted orders are kept apart (`order_kind` in the stops table; the answer gives each kind in its own clause, for example "in 2 of 9 random orders, first met after ..."), never pooled into one median. A rule that cannot be judged yet gives the reason (`insufficient_reason`), and a rule's caveats (the FOV-size warning of the per-field mean-precision rule, the object-unit JCS assumption of random objects) are carried into its sentence (`warnings`, printed as `Caution:`). Verdicts hold only for each criterion and threshold, so compare several.
2. **Would one more FOV show something new?** (`next_fov`, given a `SupportNoveltyEstimator` that defines "new") The chance, its nominal upper bound (Clopper–Pearson on leave-one-FOV-out outcomes, which are not independent, so its coverage is approximate and is checked by the calibration below), and, for a `novelty_target`, the extra FOVs needed before the bound can fall below it: at least this many if none of them shows anything new, and this many if new FOVs keep appearing at the current rate. The target is also added as a criterion to question 1. `new_observations` lists the observations counted as new so they can be inspected in the images: debris or segmentation errors count as new too.

With `few_fovs`, it also counts the FOVs holding a state that fewer than `few_fovs` other FOVs show: a state recurring in two neighbouring FOVs is not new to the next FOV, yet not established either. It also states what the FOVs rule out: a state that no FOV has shown yet is present in at most `1 - (1 - confidence) ** (1 / k)` of FOVs (about `3 / k` at 95%; `unseen_state_max_frequency`), with the extra FOVs needed to rule out states above `novelty_target`. A small chance for the next FOV is not evidence that no rare state remains (preliminary findings, entry 3). `summary()` states both answers in short (the rules in one table across the groups, each caveat once); `summary(detail="full")` gives every number group by group, and the `answer` columns keep one sentence per group and rule. `final` and `stops` keep the full evidence and stop distributions.

The chance can be checked on the dataset itself (`calibration=CalibrationSettings(...)`, or `morphocoverage.calibration.calibrate_next_fov_novelty`). Random subsets of `k` FOVs predict the chance, and the remaining FOVs, judged against the same subset with the same definition, give the actual rate. A FOV shows something new when at least `min_novel_observations` (`m`) of its unsupported cells form one cluster (linked within `epsilon`), so scattered outliers do not add up to a phenotype. Support mirrors that rule: another FOV covers a cell only with at least `m` cells of its own within `epsilon`, so a single stray cell cannot cancel a new state. The two are not exactly symmetric, since support counts cells in a ball of radius `epsilon` while a new state is a single-linkage chain that can reach beyond it (the module docstring of `support_novelty` gives a case that is counted new in two FOVs although it recurs).

No committed script or results table holds this check on the committed data. The table below was computed on 2026-09-28, at the code of the third audit, with the call beneath it (it replaces a table from an unscripted run of 2026-09-24, before later changes to the calibration and to support novelty, whose values were close but not identical). On idr0028 (12 wells of 28 FOVs; 30 standardized intensity features; `epsilon = sqrt(30)`), pooled over wells with 20 draws each:

| FOVs imaged | `m = 2`: predicted | actual | mean nominal 95% bound | bound held | `m = 3`: predicted | actual | bound held |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 5 | 0.159 | 0.134 | 0.61 | 100% | 0.054 | 0.052 | 100% |
| 10 | 0.091 | 0.079 | 0.38 | 99.6% | 0.025 | 0.025 | 100% |
| 20 | 0.060 | 0.057 | 0.23 | 96% | 0.013 | 0.016 | 99.6% |

```python
# from the repository root, with data/crick_idr on sys.path
import numpy as np
import datasets
from morphocoverage.calibration import calibrate_next_fov_novelty
from morphocoverage.strategies import SupportNoveltyEstimator
from morphocoverage.transforms import Standardize

d = datasets.load("idr0028_screenB")
for m in (2, 3):
    novelty = SupportNoveltyEstimator(np.sqrt(30), 1, Standardize(), 2, m)
    table = calibrate_next_fov_novelty(
        d.objects, d.fovs, feature_cols=list(d.features), novelty=novelty,
        fov_counts=[5, 10, 20], draws=20, random_state=0,
        group_cols=list(d.groups), observation_unit="nucleus",
    )
    print(m, table.groupby("fovs_imaged")[["predicted", "actual", "bound", "bound_held"]].mean())
```

The prediction tracks the actual rate at these settings (at 20 FOVs with `m = 3` it is slightly below it). The bound is conservative here, which is not the same as calibrated: the mean bound sits far above the actual rate, and a bound that ignores the data, `1 - 0.05 ** (1 / k)`, would hold about as often, so "bound held" cannot tell the two apart; that needs settings where the actual rate approaches the bound. Each row pools 12 wells of 20 draws, so a share near 96% has a Monte Carlo standard error of at least about 1.3 points (more, since draws within a well are not independent), and one well alone far more. The fourth audit reproduced the table exactly at `random_state=0`, with similar values at other seeds; it is still not tracked as a script and results table. The unscripted run of 2026-09-24 also found that the definition moves the answer a lot: counting any unsupported cells rather than a cluster gave much higher rates, mostly from scattered cells; letting one cell of another FOV support a cluster gave rates near zero, because stray cells cancelled real groups; and at half the distance the bound held less often. Those variants were not rerun, so their numbers are not given here. With no new FOVs among `k` the bound is `1 - 0.05 ** (1 / k)` (0.14 at 20 FOVs) whatever the data. Random subsets treat FOVs as exchangeable, so this does not test drift along the recorded order, and the actual rate is measured on few remaining FOVs when `k` is close to the group size.

In the calibration, a group too small for a count gets a row with `draws=0` and its `reason` rather than being dropped; a criterion passed twice to `sufficiency_report` is refused. It is self-assessment of one dataset; whether a criterion is right in general is a benchmark question.
