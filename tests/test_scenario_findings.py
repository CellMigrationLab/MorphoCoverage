"""The package problems the scenarios found (audit 8, F10), each pinned on the scenario that
showed it: what the package says now, and that it never says the old, wrong thing.

The fourth finding, ``rare_share_pairs.csv`` without the ``table`` column of the other pairs
CSVs, is pinned by ``test_scenarios_matrix`` on ``a09_mix_one_cells_tech_rare``: its page
check reads every pairs CSV, the label share's included.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pytest

from morphocoverage.workflows import PositiveControl, Setup, default_novelty, report

SCENARIOS = Path(__file__).resolve().parents[1] / "data" / "scenarios"
sys.path.insert(0, str(SCENARIOS))
try:
    import make_scenarios
finally:
    sys.path.remove(str(SCENARIOS))

FEATURES = ("area", "length", "ecc")


def test_a_positive_control_seen_in_too_few_fields_is_refused_naming_the_minimum() -> None:
    # Fewer reference fields than the definition of new needs: no field was ever judged,
    # and the report blamed missing values.
    declared = {
        "readout": "area",
        "features": FEATURES,
        "control": {"condition": "ctrl"},
        "novelty": default_novelty(FEATURES),
    }
    for seen in (1, 4):
        positive = PositiveControl({"condition": "a"}, fovs_seen=seen)
        with pytest.raises(ValueError, match=r"fovs_seen=5 or more") as refused:
            Setup(**declared, positive=positive)
        assert f"fovs_seen={seen} reference fields" in str(refused.value)
        assert "min_fovs" in str(refused.value)
    assert Setup(**declared, positive=PositiveControl({"condition": "a"}, fovs_seen=5))


def test_one_condition_declared_the_control_gives_the_real_reason_in_question_9() -> None:
    loaded = make_scenarios.load("e03_one_condition")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = report(
            loaded.objects, loaded.fovs, Setup(**loaded.setup), feature_tables=loaded.feature_tables
        )
    headline = result.document.headlines["conditions"]
    assert "the FOV table holds one condition, ctrl, so no other condition" in headline
    assert "needs a control condition" not in headline


def test_two_tables_of_the_same_column_names_are_refused_with_the_names_to_use() -> None:
    loaded = make_scenarios.load("a05_embed2_one_cells_tech_fp_dino")
    columns = [f"d{j}" for j in range(8)]
    tables = {"DINO1": columns, "DINO2": columns}
    with pytest.raises(ValueError, match="'DINO1: d0' for 'DINO1' and 'DINO2: d0' for 'DINO2'"):
        Setup(**{**loaded.setup, "feature_tables": tables})


def test_the_positive_control_names_several_controls_readably() -> None:
    loaded = make_scenarios.load("a02_expl_several_norm_cells_tech_positive")
    line = next(s for s in Setup(**loaded.setup).declaration() if "Positive control" in s)
    assert "judged against the controls cA and cB pooled" in line
    assert "{'condition': 'cA/cB'}" not in line
