"""Regression tests of the stress test of irregular biological-repeat designs (2026-10-05):
more permutations where the correction family needs them, conditions and repeats without the
control said once, a condition that joins two sets of repeats, the population enrichment's
empty repeats, the declared label's group tested once, and counts said per condition.

The designs follow the stress test's ``gen.py`` (E1 one repeat, E2 3/2/4/1 repeats, E4 a
condition in repeats without the control, E5 a condition repeat without the control, E6 a
condition imaged in two sets of repeats), with few fields and cells to stay fast.
"""

import re
import warnings

import numpy as np
import pandas as pd
import pytest

from morphocoverage._style import Status
from morphocoverage.sampling_unit import condition_contrast, contrast_cells, feature_discovery
from morphocoverage.sampling_unit._experiments import bridges
from morphocoverage.sampling_unit.contrast_cells import (
    balanced_draws,
    cell_randomization,
    permutations_for,
    refined,
)
from morphocoverage.sampling_unit.contrast_text import contrast_summary
from morphocoverage.sampling_unit.reading_text import condition_readings, readings_parts
from morphocoverage.workflows._html_cards import chip_words, significant_summary
from morphocoverage.workflows._populations import (
    enrichment_tests,
    labelled_populations,
    populations_in_report,
    populations_text,
)

R = ["2026_01_05", "2026_01_12", "2026_01_19", "2026_01_26"]
OTHER = ["2026_02_01", "2026_02_08", "2026_02_15"]
SET_B = ["2026_03_02", "2026_03_09", "2026_03_16"]
CONTROL_ONLY = "2026_04_06"
DESIGNS = {
    "E1": {"ctrl": R[:1], "cond1": R[:1], "cond2": R[:1]},
    "E2": {"ctrl": R, "cond1": R[:3], "cond2": R[:2], "cond3": R, "cond4": R[:1]},
    "E4": {"ctrl": R[:3] + [CONTROL_ONLY], "cond1": R[:3], "cond2": R[:3], "cond3": OTHER},
    "E5": {"ctrl": R[:3], "cond1": R, "cond2": R[:3]},
    "E6": {
        "ctrl": R[:3] + SET_B,
        "condA": R[:3],
        "condB": SET_B,
        "bridge": [R[0], SET_B[0], R[1], SET_B[1]],
    },
}
EFFECTS = {"cond1": 2.0, "condA": 2.0}


def _make(design, fields=3, cells=15, seed=0):
    """Two features (area, the effect; intensity, none), a batch shift per repeat, one well
    per condition and repeat."""
    rng = np.random.default_rng(seed)
    plan = DESIGNS[design]
    repeats = sorted({r for held in plan.values() for r in held})
    batch = {r: rng.normal(0, 0.3, 2) for r in repeats}
    rows, fovs = [], []
    for index, (condition, held) in enumerate(plan.items()):
        for repeat in held:
            well = f"{'ABCDEFG'[index]}1"
            for field in range(fields):
                image = f"{condition}_{repeat}_{field}.tif"
                z = rng.normal(0, 1, (cells, 2)) + batch[repeat]
                z[:, 0] += EFFECTS.get(condition, 0.0)
                for i in range(cells):
                    rows.append(
                        {
                            "condition": condition,
                            "biological_repeat": repeat,
                            "technical_repeat": well,
                            "image_file": image,
                            "label_id": i + 1,
                            "area": float(np.exp(5 + 0.3 * z[i, 0])),
                            "intensity": float(np.exp(7 + 0.25 * z[i, 1])),
                        }
                    )
                fovs.append(
                    {
                        "condition": condition,
                        "biological_repeat": repeat,
                        "technical_repeat": well,
                        "image_file": image,
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(fovs)


CONTROL = {"condition": "ctrl"}


@pytest.fixture(scope="module")
def designs():
    """Each design's contrast on area and its feature discovery, at few draws and shuffles."""
    found = {}
    with pytest.MonkeyPatch.context() as patch, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        patch.setattr(contrast_cells, "BALANCED_DRAWS", 3)
        for name in DESIGNS:
            objects, fovs = _make(name)
            contrast = condition_contrast(
                objects,
                fovs,
                feature="area",
                control=CONTROL,
                n_permutations=99,
                n_draws=2,
                object_counts=(10,),
            )
            discovery = feature_discovery(
                objects, fovs, features=["area", "intensity"], control=CONTROL, n_permutations=99
            )
            found[name] = contrast, discovery
    return found


def _flat(lines):
    return " ".join(" ".join(lines).split())


# --- 1. More permutations where the correction family needs them --------------------------


def test_the_shuffles_let_the_smallest_q_reach_alpha():
    assert permutations_for(100, 999) == 2000
    assert permutations_for(10, 999) == 999
    for family in (50, 210, 300):
        shuffles = permutations_for(family, 999)
        assert family / (shuffles + 1) < 0.05


def test_only_a_small_p_is_rerun_seeded_by_its_own_identity():
    """A family of 100 at 99 shuffles: the clear difference is rerun with 2000 and its q can
    reach 0.05; the null one, above 0.05, keeps its p and shuffles. The rerun is seeded from
    the test, so the order of the tests does not change it."""
    rng = np.random.default_rng(0)
    clear = (rng.normal(3, 1, 40), rng.normal(0, 1, 40))
    null = (rng.normal(0, 1, 40), rng.normal(0, 1, 40))
    _, p_clear = cell_randomization(*clear, 99, rng)
    _, p_null = cell_randomization(*null, 99, rng)
    index = [f"t{i}" for i in range(100)]
    p = pd.Series([p_clear, p_null] + [0.5] * 98, index=index)
    family = pd.Series(0, index=index)
    used = pd.Series(99, index=index)
    retests = {"t0": [clear], "t1": [null]}

    def seeded(name):
        return np.random.default_rng(int(name[1:]))

    values, shuffles = refined(p, family, used, retests, seeded)
    assert p_clear == pytest.approx(1 / 100) and values[0] == pytest.approx(1 / 2001)
    assert shuffles[0] == 2000 and shuffles[1] == 99 and values[1] == p_null
    assert 100 * values[0] < 0.05
    again, _ = refined(p[::-1], family, used, retests, seeded)
    assert again[-1] == values[0]


def test_the_balanced_test_is_rerun_with_more_labellings():
    """#89: the balanced test is one permutation test; its rerun draws new labellings and
    keeps the observed statistic, so its p moves to the finer grid of ``B`` and its d stays."""
    rng = np.random.default_rng(1)
    treated = [rng.normal(0.8, 1, n) for n in (60, 30, 30)]
    control = [rng.normal(0, 1, 30) for _ in range(3)]
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(contrast_cells, "BALANCED_DRAWS", 5)
        d, p, per_t, per_c, rerun = balanced_draws(treated, control, np.random.default_rng(2))
        assert (per_t, per_c) == (30, 30) and d > 0.5
        assert p == pytest.approx(1 / (contrast_cells.BALANCED_PERMUTATIONS + 1))
        assert rerun(1999, np.random.default_rng(3)) == pytest.approx(1 / 2000)


def test_the_screen_reruns_its_small_p_values_and_says_so(designs):
    """E4 holds 4 tests in its sub-experiment: at 99 shuffles none needs more, the family
    being small; the head says nothing then. With 100 features the screen states them."""
    _, discovery = designs["E4"]
    assert set(discovery.tests["permutations"].dropna()) <= {99, 999}
    assert "rerun with" not in discovery.summary()


def test_a_large_family_reruns_the_screen(monkeypatch):
    """40 null features beside area: 84 tests in E5's one sub-experiment, so the p-values at
    or below 0.05 are rerun with 1680 shuffles and area's q reaches 0.05."""
    objects, fovs = _make("E5")
    rng = np.random.default_rng(3)
    extra = [f"null{i}" for i in range(40)]
    objects = objects.assign(**{name: rng.lognormal(3, 0.2, len(objects)) for name in extra})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        discovery = feature_discovery(
            objects,
            fovs,
            features=["area", "intensity", *extra],
            control=CONTROL,
            n_permutations=99,
        )
    tests = discovery.tests.set_index(["feature", "condition"])
    area = tests.loc[("area", "cond1")]
    assert area["permutations"] == 1680 and area["q_cells"] <= 0.05
    # Rerun or not: the shuffles are the first pass's or the family's, nothing between.
    assert set(tests["permutations"].dropna()) <= {99, 999, 1680}
    assert (tests.loc[tests["p_cells"] < 0.01, "permutations"] == 1680).all()
    assert "the p-values at or below 0.05 rerun with 1680 shuffles" in " ".join(
        discovery.summary().split()
    )


# --- 2-3. Conditions and repeats without the control ---------------------------------------


def test_a_condition_without_the_control_is_named_once(designs):
    contrast, discovery = designs["E4"]
    line = "Not compared: cond3 (no biological repeat holds both it and ctrl)."
    reading = _flat(readings_parts(contrast))
    assert reading.count(line) == 1
    assert " ".join(discovery.summary().split()).count(line) == 1
    summary = " ".join(contrast_summary(contrast, "full").split())
    assert "No well-level test within a repeat" not in summary
    assert summary.count("Not compared: cond3") == 1

    class Result:
        pass

    result = Result()
    result.discovery, result.results = discovery, {}
    assert significant_summary(result, str).count(line) == 1


def test_a_repeat_without_the_control_is_said_where_the_repeats_are(designs):
    contrast, _ = designs["E5"]
    reading = condition_readings(contrast).set_index("condition")["reading"]
    assert "3 of its 4 biological repeats hold ctrl; 2026_01_26 left out" in reading["cond1"]
    assert "left out" not in reading["cond2"]
    assert "No well-level test" not in contrast_summary(contrast, "full")


def test_a_control_only_repeat_is_said_in_the_balance_line(designs):
    contrast, _ = designs["E4"]
    assert contrast.settings["control_only_repeats"] == [CONTROL_ONLY]
    text = " ".join(contrast_summary(contrast, "full").split())
    assert f"The ctrl of biological repeat {CONTROL_ONLY}, which holds no other condition" in text


# --- 4. A condition joining two sets of repeats --------------------------------------------


def test_a_bridging_condition_is_said_in_the_design_sentence(designs):
    assert bridges({"a": [1, 2], "b": [3, 4], "x": [1, 3]}) == {"x": 2}
    assert bridges({"a": [1, 2], "b": [1]}) == {}
    contrast, _ = designs["E6"]
    reading = _flat(readings_parts(contrast))
    assert reading.count("bridge, imaged in both sets of biological repeats, joins them") == 1
    assert "into one sub-experiment of 3 tests" in reading


# --- 5-6. Population enrichment ------------------------------------------------------------


def test_a_repeat_where_the_group_is_absent_on_both_sides_counts_as_no_change():
    rows = []
    for r, (control_k, drug_k) in enumerate(((5, 30), (6, 28), (0, 0), (4, 25))):
        for condition, k in (("ctrl", control_k), ("drug", drug_k)):
            rows += [{"condition": condition, "biological_repeat": f"r{r}", "P2": 1.0}] * k
            rows += [{"condition": condition, "biological_repeat": f"r{r}", "P2": 0.0}] * (300 - k)
    [row] = enrichment_tests(pd.DataFrame(rows), ["P2"], "ctrl").to_dict("records")
    assert row["repeats"] == 4 and row["higher"] == 3 and row["lower"] == 0


@pytest.fixture(scope="module")
def labelled():
    from _helpers_rare_fingerprint import _screen, _setup

    objects, fovs = _screen()
    setup = _setup(fingerprint=())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        found = populations_in_report(objects, fovs, setup)
    return found, setup


def test_the_declared_labels_group_is_tested_once_as_the_label(labelled):
    found, setup = labelled
    own = labelled_populations(found.label)
    assert own and not set(own) & set(found.enrichment["feature"])
    status, headline, text = populations_text(found, setup)
    flat = " ".join(text.split())
    name = own[0]
    said = f"{name} is the declared rare phenotype, mitotic: most of its cells carry the label"
    # Said once, in the headline (the page's In short); the text gives its numbers only.
    assert said in headline and said not in flat
    held = rf"{name}: \d+% of its cells carry the label mitotic, and it holds \d+% of the "
    assert re.search(held + "labelled ones; not tested again here.", flat)
    assert "its share is compared in “Do the conditions differ?”" in headline
    assert "q " not in headline and status is Status.OK

    class Result:
        pass

    result = Result()
    result.results = {"populations": found}
    assert chip_words(result)["populations"] == "The declared label"


# --- 7-8. Counts said per condition; one repeat ---------------------------------------------


def test_counts_are_each_conditions_own(designs):
    contrast, discovery = designs["E2"]
    text = " ".join(contrast_summary(contrast, "full").split())
    assert "Across the biological repeats, 1 to 4 per condition (each repeat's value" in text
    assert "With 2 and 1 biological repeats holding both sides, cond2 and cond4 are" in text
    assert "the control as provided is pooled from biological repeats the condition was not" in (
        text
    )
    assert "(1, 2 and 3 of them)" in text

    class Result:
        pass

    result = Result()
    result.discovery, result.results = discovery, {}
    words = significant_summary(result, str)
    assert "significant across the biological repeats, 1 to 4 per condition" in words


def test_one_repeat_is_read_as_one_repeat(designs):
    contrast, _ = designs["E1"]
    reading = _flat(readings_parts(contrast))
    assert "Significant across repeats:" not in reading
    assert "in one biological repeat: the direction and Cohen's d of that repeat" in reading
    assert "Cohen's d +" in reading and "per repeat, mean" not in reading
    text = " ".join(contrast_summary(contrast, "full").split())
    assert "each of 1 repeat" not in text and "the fields a side the one repeat needs" in text


def test_comparisons_too_small_are_counted_over_the_features_named(designs):
    from morphocoverage.workflows._enough_features import FeatureEnough, features_status

    contrast, _ = designs["E4"]
    table = pd.DataFrame(
        {
            "feature": ["area", "intensity", "perimeter"],
            "verdict": ["partly"] * 3,
            "too_small": [2, 3, 0],
            "needing": [0, 0, 1],
            "fields_high": [np.nan, np.nan, 12],
            "precision_met": [np.nan] * 3,
        }
    )
    _, headline = features_status(FeatureEnough(table, {"area": contrast}, {}))
    assert "5 comparisons over area and intensity too small to show even with" in headline
    one = table.assign(too_small=[2, 0, 0])
    _, headline = features_status(FeatureEnough(one, {"area": contrast}, {}))
    assert "2 comparisons of area too small to show" in headline
