"""The definition of new judged against its own null rate (owner, 2026-10-06, audit 10: B2,
A23, C1, A20, B3): a list of new states is "not informative" when the definition calls more
than 20% of a homogeneous population's fields new at the group's size and feature count; the
count chance alone gives is said beside the new states, with or without a positive control;
a positive control recognized no more often than the reference's false alarms is "cannot
tell", not a weak representation; question 6's new states are exploratory."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from morphocoverage.retrospective import inspect_fovs
from morphocoverage.retrospective.report_profile import _new_state_rows
from morphocoverage.strategies.novelty_null import NOT_INFORMATIVE, null_rate
from morphocoverage.workflows import Setup, default_novelty, report

SCENARIOS = Path(__file__).resolve().parents[1] / "data" / "scenarios"


def _flat(text: str) -> str:
    return " ".join(text.split())


def _homogeneous(features: int, fields: int, cells: int, seed: int = 0, groups: int = 2):
    """``groups`` conditions of ``fields`` fields of ``cells`` observations, all drawn alike."""
    rng = np.random.default_rng(seed)
    names = [f"x{j}" for j in range(features)]
    rows = []
    for g in range(groups):
        for f in range(fields):
            values = rng.normal(10, 1, size=(cells, features))
            frame = pd.DataFrame(values, columns=names)
            rows.append(frame.assign(condition=f"c{g}", biological_repeat="R1", position_id=f))
    objects = pd.concat(rows, ignore_index=True)
    fovs = objects[["condition", "biological_repeat", "position_id"]].drop_duplicates()
    return objects, fovs.reset_index(drop=True), names


def test_the_null_rate_of_a_definition_follows_the_group_size_and_features():
    wide = [f"f{j}" for j in range(17)]
    # Audit 10B: 17 features, one image per field, a radius of sqrt(17): nearly every field new.
    for fields in (5, 10, 15):
        assert null_rate(default_novelty(wide), [1] * fields, wide) > 0.8
    narrow = [f"f{j}" for j in range(5)]
    cells = null_rate(default_novelty(narrow), [30] * 8, narrow)
    assert cells is not None and cells < NOT_INFORMATIVE
    # Reproducible, and none without a transform (a radius in the features' own units).
    assert null_rate(default_novelty(narrow), [30] * 8, narrow) == cells
    raw = default_novelty(narrow).__class__(1.0, 1, None, 5, 3)
    assert null_rate(raw, [30] * 8, narrow) is None


def test_new_states_are_not_listed_where_the_definition_is_not_informative():
    objects, fovs, names = _homogeneous(8, 12, 1)
    result = inspect_fovs(objects, fovs, feature_cols=names, novelty=default_novelty(names))
    assert (result.fovs["reason"] != "new state").all() if len(result.fovs) else True
    uninformative = result.settings["uninformative_novelty"]
    assert len(uninformative) == 2 and result.settings["n_fovs_judged"] == 0
    assert uninformative[0]["reason"].startswith("not informative with 12 fields in 8 features")
    # One line for every group, in the short and the full forms (owner, 2026-10-06).
    for detail in ("short", "full"):
        text = _flat(result.summary(detail=detail))
        assert text.count("New states are not listed") == 1
        assert "New states are not listed in 2 groups" in text
        assert "not informative with 12 fields in 8 features each" in text
        assert "of the fields of a homogeneous population of these sizes new" in text


def test_the_count_chance_gives_is_said_without_a_positive_control():
    objects, fovs, names = _homogeneous(3, 10, 30, seed=2)
    # One field holds a distinct state of 10 cells.
    state = (objects["condition"] == "c0") & (objects["position_id"] == 4)
    rows = objects.index[state][:10]
    objects.loc[rows, names] = 20 + np.random.default_rng(3).normal(0, 0.1, (10, 3))
    result = inspect_fovs(objects, fovs, feature_cols=names, novelty=default_novelty(names))
    assert not result.settings["uninformative_novelty"]
    rates = result.settings["novelty_null"]
    assert len(rates) == 2 and all(r["rate"] <= NOT_INFORMATIVE for r in rates)
    new = result.fovs[result.fovs["reason"] == "new state"]
    assert ((new["condition"] == "c0") & (new["position_id"] == 4)).any()
    statement = _new_state_rows(result, None)[0]["statement"]
    assert "; false alarms alone would give about" in statement
    assert "of the 20 fields judged, so these are" in statement
    assert "a homogeneous population of the same sizes" in statement


@pytest.fixture(scope="module")
def a02():
    """Scenario a02: a positive control of 4 cells a field, never told from the reference."""
    sys.path.insert(0, str(SCENARIOS))
    try:
        import make_scenarios
    finally:
        sys.path.remove(str(SCENARIOS))
    loaded = make_scenarios.load("a02_expl_several_norm_cells_tech_positive")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return report(
            loaded.objects,
            loaded.fovs,
            Setup(**loaded.setup),
            feature_tables=loaded.feature_tables,
        )


def test_a_positive_control_no_better_than_false_alarms_is_not_judged(a02):
    """Audit 10C (C1): the positive control recognized in 20% of checks, as the false alarms,
    read "weak representation" (❌); it is "cannot tell", not judged (➖)."""
    marker = a02.document.statuses["controls"].marker
    assert marker == "➖"
    assert a02.document.headlines["controls"].startswith(
        "cannot tell the positive control from the reference with this definition of new"
    )
    full = _flat(a02.summary(detail="full"))
    assert "weak representation" not in full.lower()
    assert "Cannot tell the positive control from the reference" in full
    for key in ("enough", "next_time", "bought"):
        assert a02.document.statuses[key].marker != "❌"


def test_question_6_new_states_are_exploratory(a02):
    """Audit 10B (B3): question 6's new states are labelled exploratory once."""
    full = a02.summary(detail="full")
    section = full[full.index("6. Which fields should I look at?") :]
    section = section[: section.index("9. Do the conditions differ")]
    assert _flat(section).count("Novelty is exploratory") == 1
