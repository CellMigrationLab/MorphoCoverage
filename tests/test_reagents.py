"""Independent reagents for one target: do they agree across the repeats? (question 9)."""

import sys
from pathlib import Path

import pandas as pd
import pytest

from morphocoverage.retrospective.report_status import conditions_status
from morphocoverage.sampling_unit.contrast import condition_contrast
from morphocoverage.sampling_unit.contrast_repeats import reagent_agreement

LAB = Path(__file__).resolve().parents[1] / "data" / "lab"
sys.path.insert(0, str(LAB))
import catalog


def _table(signs):
    """Per-repeat means of two reagents of KD, against a control at 10."""
    rows = []
    for repeat, pair in enumerate(signs):
        for reagent, sign in zip(("siKD#1", "siKD#2"), pair, strict=True):
            rows.append(
                {
                    "biological_repeat": f"r{repeat}",
                    "condition": reagent,
                    "mean": 10 + sign,
                    "control_mean": 10.0,
                }
            )
    return pd.DataFrame(rows)


TARGETS = {"siKD#1": "KD", "siKD#2": "KD"}


def test_reagents_that_go_the_same_way_in_every_repeat_agree():
    [row] = reagent_agreement(_table([(-1, -2), (-1, -1), (-3, -1)]), TARGETS).to_dict("records")
    assert (row["repeats"], row["agree"], row["direction"]) == (3, 3, "lower")
    assert row["reagents"] == ["siKD#1", "siKD#2"]


def test_disagreement_and_flipping_sides_are_counted():
    [split] = reagent_agreement(_table([(-1, 1), (-1, -1), (1, 1)]), TARGETS).to_dict("records")
    assert (split["agree"], split["direction"]) == (2, "mixed")
    [none] = reagent_agreement(_table([(-1, 1), (1, -1)]), TARGETS).to_dict("records")
    assert (none["agree"], none["direction"]) == (0, "none")


def test_a_target_with_one_reagent_or_no_column_gives_nothing():
    assert reagent_agreement(_table([(-1, -1)]), {"siKD#1": "KD"}).empty
    assert reagent_agreement(_table([(-1, -1)]), {}).empty


@pytest.fixture(scope="module")
def screen():
    data = catalog.load("cct8_filopodia_screen")
    return data, condition_contrast(
        *data.tables, feature="Filopodia number", control=data.control, observation_unit="cell"
    )


def test_the_screens_reagents_agree_for_every_target(screen):
    _, contrast = screen
    targets = contrast.targets.set_index("target")
    assert set(targets.index) == {"CCT8", "CSNK1A1", "MAP4K4", "MINK1", "NF2", "SCRIB"}
    assert (targets["agree"] == 3).all()
    assert set(targets.loc[["MAP4K4", "NF2"], "direction"]) == {"higher"}
    short = contrast.summary()
    assert "Independent reagents for one target (target)" in short
    assert "all higher in 3 of 3" in short and "⚠️" not in short
    _, headline = conditions_status(contrast, None)
    assert headline.endswith("reagents agree in every repeat for 6 of 6 targets")


def test_without_the_column_the_contrast_is_unchanged(screen):
    data, contrast = screen
    objects, fovs = data.tables
    plain = condition_contrast(
        objects.drop(columns="target"),
        fovs.drop(columns="target"),
        feature="Filopodia number",
        control=data.control,
        observation_unit="cell",
    )
    assert plain.targets.empty and "Independent reagents" not in plain.summary()
    pd.testing.assert_frame_equal(plain.table, contrast.table)


def test_a_condition_with_two_targets_is_refused(screen):
    data, _ = screen
    objects, fovs = data.tables
    fovs = fovs.copy()
    fovs.loc[fovs.index[0], "target"] = "OTHER"
    with pytest.raises(ValueError, match="more than one target for condition"):
        condition_contrast(
            objects, fovs, feature="Filopodia number", control=data.control, observation_unit="cell"
        )
