"""The owner's decisions of 2026-10-05 on the strategies' precision estimates.

#97: the FOV unit's upper-95 averaging CV is a bootstrap of whole fields, not the chi-square
bound on Kish's effective fields; its coverage is simulated here.
#315 and #333: the FOV bootstrap's SE and the fingerprint's random-half error are not
available when more than 10% of their draws are undefined, and the share is said.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

from morphocoverage import ObjectiveConfig, SamplingObjective, validate_input_tables
from morphocoverage._style import span
from morphocoverage._wording import count
from morphocoverage.outcomes import OutcomeMetric
from morphocoverage.retrospective import OutcomeCriterion
from morphocoverage.statistics import kish_effective_fields
from morphocoverage.strategies import FOVBootstrapEstimator
from morphocoverage.strategies._averaging_cv import fov_averaging_cv
from morphocoverage.strategies.fov_bootstrap import UNDEFINED_TOLERANCE
from morphocoverage.workflows._enough_targets import (
    fingerprint_lines,
    fingerprint_precision,
    not_judged,
)


def _wells(
    rng: np.random.Generator, n: int, k: int, dispersion: float | None, sd: float
) -> tuple[np.ndarray, np.ndarray]:
    """Per-field sums and counts of ``n`` wells of ``k`` fields of mean 20 objects.

    Field sizes are Poisson, or negative binomial (over-dispersed) with ``dispersion``; field
    means are lognormal around 100 (SD ``sd`` on the log scale) and objects normal around their
    field's mean (SD 30), so a field's sum is drawn from its size and mean directly.
    """
    if dispersion is None:
        sizes = rng.poisson(20, (n, k)).astype(float)
    else:
        sizes = rng.negative_binomial(dispersion, dispersion / (dispersion + 20), (n, k))
        sizes = sizes.astype(float)
    means = 100 * np.exp(sd * rng.standard_normal((n, k)))
    return sizes * means + np.sqrt(sizes) * 30 * rng.standard_normal((n, k)), sizes


def _coverage(k: int, dispersion: float | None, sd: float, wells: int) -> tuple[float, float]:
    """How often the bootstrap bound and the former Kish bound (at 95%) cover the truth."""
    rng = np.random.default_rng(97)
    sums, counts = _wells(rng, 20000, k, dispersion, sd)
    usable = (counts > 0).sum(axis=1) >= 2
    pooled = sums[usable].sum(axis=1) / counts[usable].sum(axis=1)
    truth = pooled.std() / pooled.mean()  # the averaging CV of k fields
    sums, counts = _wells(rng, wells, k, dispersion, sd)
    bootstrap, kish = [], []
    for well in range(wells):
        if (counts[well] > 0).sum() < 2 or kish_effective_fields(counts[well]) < 2:
            continue  # insufficient_data for both bounds
        found = fov_averaging_cv(sums[well], counts[well], random_state=well)
        upper = found["cv_mean_upper_95"]
        bootstrap.append(not math.isfinite(upper) or truth <= upper)
        df = found["effective_fovs"] - 1
        kish.append(truth <= found["cv_mean"] * math.sqrt(df / stats.chi2.ppf(0.05, df)))
    return float(np.mean(bootstrap)), float(np.mean(kish))


def test_the_fov_upper_bound_covers_near_95_with_over_dispersed_field_sizes():
    # Regression (#97): the chi-square bound on Kish's effective fields covered 70-85% with
    # over-dispersed field sizes. Negative binomial sizes (dispersion 1), 8 fields.
    bootstrap, kish = _coverage(8, 1.0, 0.3, 600)
    assert 0.90 <= bootstrap <= 0.97
    assert bootstrap > kish
    # Poisson sizes, 10 fields: close to nominal.
    bootstrap, _ = _coverage(10, None, 0.3, 600)
    assert 0.93 <= bootstrap <= 0.99


def test_the_fov_upper_bound_needs_spread_in_95_percent_of_its_resamples():
    # Two fields of one mean among four: 1/16 of the resamples draw only those two (over 5%),
    # so the bound is not defined yet, while the point CV is.
    sums, counts = np.array([200.0, 200.0, 150.0, 260.0]), np.full(4, 2.0)
    found = fov_averaging_cv(sums, counts)
    assert found["cv_mean"] > 0 and math.isnan(found["cv_mean_upper_95"])
    five = fov_averaging_cv(np.r_[sums, 230.0], np.full(5, 2.0))
    assert five["cv_mean_upper_95"] > five["cv_mean"]


def _fields(filled: int, empty: int) -> object:
    """``filled`` fields of 5 objects each, then ``empty`` acquired fields without objects."""
    rng = np.random.default_rng(315)
    objects = pd.DataFrame(
        {
            "fov": np.repeat([f"f{j:02d}" for j in range(filled)], 5),
            "x": rng.normal(10, 2, 5 * filled) + np.repeat(rng.normal(0, 1, filled), 5),
        }
    )
    fovs = pd.DataFrame({"fov": [f"f{j:02d}" for j in range(filled + empty)]})
    return validate_input_tables(
        objects, fovs=fovs, feature_cols=["x"], fov_col="fov", group_cols=[]
    )


def _bootstrap(data: object, n: int = 2000) -> object:
    estimator = FOVBootstrapEstimator("centroid", None, n, 0, 2)
    return estimator.estimate(data, objective=ObjectiveConfig(SamplingObjective.POPULATION))


def test_the_fov_bootstrap_is_not_available_above_10_percent_undefined_resamples():
    # Regression (#315, #316): 2 fields with objects among 15 acquired: (13/15) ** 15 = 12% of
    # the resamples draw no observation. Their SE was taken from the rest, which is
    # conditional on a resample holding observations.
    assert UNDEFINED_TOLERANCE == 0.10
    evidence = _bootstrap(_fields(2, 13))
    quantity = evidence.quantities["fov_bootstrap_centroid_max_se"]
    assert quantity.availability == "insufficient_data"
    share = evidence.diagnostics["undefined_fraction"]
    assert 0.10 < share < 0.14
    assert evidence.diagnostics["undefined_resamples"] == round(share * 2000)
    assert quantity.reason == (
        f"{round(share * 2000)} of 2000 resamples ({round(100 * share)}%; more than 10%) draw "
        "too few observations to define the summary"
    )


def test_below_10_percent_undefined_the_fov_bootstrap_se_is_from_the_defined_resamples():
    # 3 fields with objects among 12: (9/12) ** 12 = 3% undefined. The SE is computed as before
    # from the defined resamples, and the share is in the diagnostics beside it.
    evidence = _bootstrap(_fields(3, 9))
    assert evidence.quantities["fov_bootstrap_centroid_max_se"].availability == "available"
    assert 0.01 < evidence.diagnostics["undefined_fraction"] < 0.06
    assert _bootstrap(_fields(4, 0)).diagnostics["undefined_fraction"] == 0


class _UndefinedWith(OutcomeMetric):
    """The difference of the mean area of two halves; undefined when ``fields`` all fall in
    the second half (invalidity tied to which fields a half holds)."""

    name = "undefined with fields"

    def __init__(self, fields: tuple[str, ...]) -> None:
        self.fields = fields

    def evaluate(self, full: object, stopped: object) -> float | None:
        held = set(stopped.fovs["image_file"])
        if held and all(any(name.endswith(f) for name in held) for f in self.fields):
            return None
        return abs(
            float(np.mean(full.feature_values("area")))
            - float(np.mean(stopped.feature_values("area")))
        )


def _precision(fields: tuple[str, ...]) -> pd.DataFrame:
    from _helpers_rare_fingerprint import _screen, _setup

    objects, fovs = _screen()
    criterion = OutcomeCriterion(_UndefinedWith(fields), 10.0, "lower_is_better", "loose")
    return fingerprint_precision(objects, fovs, _setup(fingerprint=(criterion,)))


def test_the_fingerprint_is_not_judged_above_10_percent_undefined_halves():
    # Regression (#333): halves whose error was undefined were dropped, so the estimate came
    # from the halves that happened to be defined. Here a half holding field 0 is undefined:
    # about half of every well's halves.
    precision = _precision(("_0.tif",))
    assert (precision["splits"] == 20).all()
    shares = precision["undefined_splits"] / precision["splits"]
    assert (shares > 0.10).all() and precision["met"].isna().all()
    assert precision["error"].isna().all()
    text = " ".join(" ".join(fingerprint_lines(precision, "well", "fields")).split())
    assert text.startswith("The fingerprint, loose: not judged in any well: ")
    said = not_judged(precision, "well", "fields")
    assert said.startswith(
        f"Not judged in 18 wells: no defined error on {span(shares, 'percent')} of the random "
        "halves, more than 10%."
    )


def test_below_10_percent_undefined_halves_the_error_is_from_the_others_and_said():
    # A half holding fields 0, 1 and 2 together is undefined: 1 in 20 halves of 6 fields.
    precision = _precision(("_0.tif", "_1.tif", "_2.tif"))
    undefined = precision["undefined_splits"]
    assert undefined.max() > 0 and (undefined == 0).any()
    judged = undefined <= 2
    assert precision.loc[judged, "error"].notna().all()
    assert precision.loc[~judged, "met"].isna().all()
    text = " ".join(" ".join(fingerprint_lines(precision, "well", "fields")).split())
    some = precision[judged & (undefined > 0)]
    shares = some["undefined_splits"] / some["splits"]
    assert (
        f"In {count(len(some), 'well')}, {span(shares, 'percent')} of the random halves have no "
        "defined error; the error there is from the others."
    ) in text
    # Wells whose halves are all defined keep the estimate as before.
    whole = _precision(("_never.tif",))
    assert (whole["undefined_splits"] == 0).all() and whole["error"].notna().all()


def test_the_wells_not_judged_are_named_with_each_reason():
    # Fields 0 and 1 both in the second half: 4 in 20 halves of 6 fields; a well left with 3
    # fields is too small to be judged at all.
    from _helpers_rare_fingerprint import _screen, _setup

    objects, fovs = _screen()
    small = fovs["image_file"].str.startswith("arrest_2026_01_05_C1_")
    drop = set(fovs.loc[small, "image_file"].iloc[3:])
    fovs, objects = fovs[~fovs["image_file"].isin(drop)], objects[~objects["image_file"].isin(drop)]
    criterion = OutcomeCriterion(_UndefinedWith(("_0.tif", "_1.tif")), 10.0, "lower_is_better")
    precision = fingerprint_precision(objects, fovs, _setup(fingerprint=(criterion,)))
    missing = precision[precision["met"].isna()]
    halves = missing[missing["fields"] >= 4]
    shares = halves["undefined_splits"] / halves["splits"]
    assert len(missing) - len(halves) == 1 and (shares > 0.10).all()
    assert not_judged(precision, "well", "fields") == (
        f"Not judged in {count(len(missing), 'well')}: 1 with fewer than 4 fields; "
        f"{len(halves)} with no defined error on {span(shares, 'percent')} of the random "
        "halves, more than 10%."
    )


def test_the_report_states_the_field_bounds_simulated_coverage():
    # #97 (owner, 2026-10-05): where the report gives the per-field upper-95 rule, it says how
    # often its bootstrap bound covered the truth in simulations; the point rule has no bound.
    from scenarios import FEATURE_COLS, SCENARIOS

    from morphocoverage.benchmark import AcquisitionOrder
    from morphocoverage.strategies import ScalarPrecisionStrategy
    from morphocoverage.sufficiency import sufficiency_report

    built = SCENARIOS["homogeneous"]()
    sentence = (
        "in simulations its upper bound, a bootstrap of whole fields, covered the true "
        "averaging error about 85 to 95% of the time with uneven field sizes and 93 to 99% "
        "with Poisson sizes"
    )
    for sd, said in (("upper_95", True), ("point", False)):
        report = sufficiency_report(
            built.objects,
            built.fovs,
            feature_cols=list(FEATURE_COLS),
            strategies=[ScalarPrecisionStrategy("feature_a", 0.05, "fov", sd)],
            objective=ObjectiveConfig(SamplingObjective.POPULATION),
            orders=[AcquisitionOrder.permuted(0)],
        )
        for detail in ("short", "full"):
            text = " ".join(report.summary(detail).split())
            assert (sentence in text) is said, (sd, detail)


def test_a_fingerprint_judged_in_no_well_is_said_and_not_assessed():
    # Regression (#333): with no well judged, the text said nothing and the page's tab could
    # show ✅ over an empty set. It now says so, and the marker is ➖ (not assessed).
    from types import SimpleNamespace

    from _helpers_rare_fingerprint import _setup

    from morphocoverage._style import Status
    from morphocoverage.workflows._enough_targets import targets_status
    from morphocoverage.workflows._html_enough import _targets

    precision = _precision(("_0.tif",))
    shares = precision["undefined_splits"] / precision["splits"]
    [line] = fingerprint_lines(precision, "well", "fields")
    assert " ".join(line.split()) == (
        "The fingerprint, loose: not judged in any well: no defined error on "
        f"{span(shares, 'percent')} of the random halves, more than 10%."
    )
    setup = _setup()
    assert targets_status(None, precision, setup) == (
        [Status.NOT_ASSESSED],
        ["the fingerprint judged in none of 18"],
    )
    result = SimpleNamespace(setup=setup, results={"fingerprint_precision": precision})
    lines, tabs = _targets(result, "well", "field")
    assert lines == ["The fingerprint is not judged in any well."]
    assert [(name, mark) for name, mark, _ in tabs] == [("fingerprint", "➖")]
