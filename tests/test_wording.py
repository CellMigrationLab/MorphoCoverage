"""Tests of ``_wording``: percentages, counts, plurals and the words of the summaries."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
from _helpers_retrospective import _plate_curve
from _helpers_sampling_unit import _signed
from _helpers_tables import _error_titles

from morphocoverage import check_tables
from morphocoverage._wording import (
    change,
    count,
    duration,
    fields,
    listing,
    number,
    percent,
    rows_of,
)
from morphocoverage.retrospective import SamplingHierarchy, protocol_reliability
from morphocoverage.retrospective._text import plural


def test_j16_wording_helpers():
    assert percent(0.003) == "0.3%" and percent(0.5) == "50%" and percent(0) == "0%"
    assert number(7.0) == "7" and number(7.25) == "7.2" and number(1234567) == "1234567"
    assert fields(1) == "1 field" and fields(2.5) == "2.5 fields"
    assert fields(1.047) == "1 field" and fields(0.96) == "1 field" and fields(1.06) == "1.1 fields"
    assert count(1, "batch", "batches") == "1 batch" and count(3, "batch", "batches") == "3 batches"
    assert duration(20) == "20 s" and duration(600) == "10 min" and duration(5400) == "1.5 h"


def test_j16_rows_of_matches_missing_values():
    import pandas as pd

    frame = pd.DataFrame({"condition": ["A", None, "B"], "value": [1, 2, 3]})
    assert rows_of(frame, {"condition": None})["value"].tolist() == [2]
    assert rows_of(frame, {"condition": "B"})["value"].tolist() == [3]


def test_f13_unit_names_take_their_english_plural():
    assert [plural(w) for w in ("batch", "well", "repeat", "technical repeat", "assay")] == [
        "batches",
        "wells",
        "repeats",
        "technical repeats",
        "assays",
    ]
    assert plural("colony") == "colonies"
    curve = {(f"b{i}", "A1"): {5: 0.95} for i in range(2)}
    text = protocol_reliability(
        _plate_curve(curve, unit="batch"),
        SamplingHierarchy("image_file", "well", "batch"),
        threshold=0.9,
        strata=[],
    ).summary()
    assert "batches" in text and "batchs" not in text


def test_a6_e19_percent_never_reads_0_or_100_for_a_share_between_and_rounds_half_up():
    assert percent(199 / 200) == "99.5%" and percent(0.999) == "99.9%"
    assert percent(1 / 5000) == "0.02%" and percent(0.003) == "0.3%"
    assert percent(0.125) == "13%" and percent(0.025) == "3%"
    assert percent(0) == "0%" and percent(1) == "100%"
    assert percent(float("nan")) == "undefined"


def test_j7_one_plural_helper():
    assert [plural(w) for w in ("nucleus", "colony", "branch", "patch", "cell")] == [
        "nuclei",
        "colonies",
        "branches",
        "patches",
        "cells",
    ]
    assert plural("organoids") == "organoids" and plural("fov_summary") == "field summaries"
    assert count(1, "nucleus") == "1 nucleus" and count(3, "colony") == "3 colonies"
    assert listing(["a", "b", "c"]) == "a, b and c"
    assert listing(list("abcdefg"), limit=3) == "a, b, c and 4 more"


def test_a6_percent_never_rounds_to_0_or_100_between():
    assert percent(0.996) == "99.6%" and percent(0.004) == "0.4%"
    cells = pd.DataFrame(
        {"image_file": [f"f{i % 10}" for i in range(1000)], "area": [np.nan] * 999 + [1.0]}
    )
    fovs = pd.DataFrame({"image_file": [f"f{i}" for i in range(10)]})
    check = check_tables(cells, fovs, feature_cols=["area"], observation_unit="cell")
    assert "Feature 'area' is missing in 99.9% of observations." in _error_titles(check, "warning")


def test_e19_percent_rounds_half_up():
    assert percent(0.125) == "13%" and percent(0.135) == "14%"


def test_seed13_the_label_share_is_counted_apart_from_the_features():
    from morphocoverage._wording import features_counted
    from morphocoverage.sampling_unit._experiments import family_words
    from morphocoverage.sampling_unit.discovery import feature_discovery

    assert features_counted(5, "mitotic share") == "4 features and the mitotic share"
    assert features_counted(["area", "mitotic share"], "mitotic share", ", ") == (
        "1 feature, the mitotic share"
    )
    assert features_counted(["mitotic share"], "mitotic share") == "the mitotic share"
    assert features_counted(["area", "perimeter"], "mitotic share") == "2 features"
    settings = {"family_features": 5, "family_share": "mitotic share"}
    assert family_words(settings).endswith(
        "over its 4 features, the mitotic share and their conditions"
    )
    objects, fovs = _signed(offset_b=3.0)
    objects["mitotic share"] = (objects["area"] > 2.5).astype(float)
    found = feature_discovery(
        objects,
        fovs,
        features=["area", "positive", "mitotic share"],
        control={"condition": "ctrl"},
        n_permutations=9,
        fov_col="image_file",
    )
    found = replace(found, settings={**found.settings, "share": "mitotic share"})
    text = " ".join(found.summary().split())
    assert "of 3 (2 features and the mitotic share)" in text  # never "of 2 features and" (C20)


def test_the_change_is_written_so_a_tiny_change_never_reads_as_none():
    assert change(1.25) == "×1.25"
    assert change(0.987) == "×0.987"
    assert change(1.002) == "×1.002"
    assert change(-3.21, "raw") == "-3.21"
    assert change(np.nan) == "–"
