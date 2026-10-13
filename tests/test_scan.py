"""The design scan run before any analysis: what the dataset is, whether it was loaded
correctly, which columns would say more, and the questions to ask (workflows.scan)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from morphocoverage._style import Status
from morphocoverage.workflows import DesignScan, Setup, scan_design
from morphocoverage.workflows.scan import reagent_groups

LAB = Path(__file__).resolve().parents[1] / "data" / "lab"
sys.path.insert(0, str(LAB))
import catalog


def _usual(n_repeats=3, per_field=10, fields=6, seed=0):
    """Three conditions in every repeat, one well each, balanced: the usual design."""
    rng = np.random.default_rng(seed)
    rows = []
    for condition in ("siCTRL", "drugA", "drugB"):
        for repeat in [f"2026_0{r + 1}_01" for r in range(n_repeats)]:
            for field in range(fields):
                for label in range(per_field):
                    rows.append(
                        {
                            "condition": condition,
                            "biological_repeat": repeat,
                            "image_file": f"{condition}_{repeat}_{field}.tif",
                            "label_id": label + 1,
                            "area": rng.normal(100, 10),
                            "speed": rng.normal(5, 1),
                            "aspect": rng.normal(2, 0.2),
                        }
                    )
    return pd.DataFrame(rows)


def _flat(text):
    """The text with its line wrapping undone."""
    return " ".join(text.split())


def _texts(scan, status=None):
    return [f.text for f in scan.findings if status is None or f.status is status]


def test_a_usual_design_has_no_caution():
    scan = scan_design(
        _usual(),
        feature_cols=["area", "speed", "aspect"],
        control={"condition": "siCTRL"},
        observation_unit="cell",
    )
    assert isinstance(scan, DesignScan)
    assert scan.status is Status.OK
    assert _texts(scan, Status.CAUTION) == []
    assert scan.design["conditions"] == ["siCTRL", "drugA", "drugB"]
    text = _flat(scan.summary())
    assert "3 biological repeats (2026_01_01, 2026_02_01 and 2026_03_01); every condition" in text
    assert "54 fields, 10 cells per field." in text
    assert "No copied field or row" in text
    # The goal and the readout are always asked; nothing else was left undeclared.
    assert [q.key for q in scan.questions] == ["goal", "readout"]
    assert scan.questions[0].options == ("averages", "fingerprint", "rare")
    assert scan.questions[0].suggested == "averages"


def test_cdm_cells_a_copy_and_the_columns_that_exist():
    data = catalog.load("cct8_cdm_morphology")
    scan = scan_design(
        *data.tables,
        feature_cols=data.feature_cols,
        control=data.control,
        observation_unit=data.observation_unit,
    )
    # The cell measured twice that the curation planted (FINDINGS.md, run 7).
    [copy] = [f for f in scan.findings if f.about == "copies"]
    assert copy.status is Status.PROBLEM
    assert "cell007" in copy.text and "cell008" in copy.text
    text = scan.summary()
    assert "174 fields, one cell per field." in text
    assert "Targets with several conditions: CCT8 (siCCT8#5 and siCCT8#8)." in text
    assert "Technical repeats: none recorded (one well per condition and repeat)." in text
    # Unbalanced repeats are stated, not warned about: the report tests both ways.
    [balance] = [f for f in scan.findings if f.about == "balance"]
    assert balance.status is Status.OK
    assert "up to 2.2× in siCTRL" in balance.text
    assert not any("'target'" in s for s in scan.suggestions)
    assert any("'technical_repeat'" in s for s in scan.suggestions)


def test_without_image_names_target_or_wells_the_scan_says_what_to_add():
    data = catalog.load("cct8_cdm_morphology")
    objects = data.objects.drop(columns=["position_id", "image_file", "target", "technical_repeat"])
    scan = scan_design(objects)
    assert scan.settings["fov_added"] and scan.settings["fov_col"] == "position_id"
    assert scan.objects["position_id"].nunique() == len(objects)
    assert len(scan.fovs) == len(objects)
    suggestions = " ".join(scan.suggestions)
    assert "Add 'image_file' (or 'position_id'" in suggestions
    assert "each observation is assumed to be on its own image" in suggestions
    assert "siCCT8#5 and siCCT8#8 look like reagents of CCT8" in suggestions
    assert "add 'technical_repeat' if a condition was imaged in more than one well" in suggestions
    # Candidate features are offered, not used as declared, and a constant column is not one.
    assert not scan.settings["features_declared"]
    assert "source_channel" not in scan.settings["feature_cols"]
    keys = {q.key: q for q in scan.questions}
    assert set(keys) == {"goal", "readout", "control", "features", "observation_unit"}
    assert keys["control"].suggested == "siCTRL"
    assert "siCTRL (named like a control; not declared)" in scan.summary()


def test_setup_needs_a_declared_control_and_a_tolerance_for_other_goals():
    data = catalog.load("cct8_cdm_morphology")
    objects = data.objects.drop(columns=["position_id", "image_file"])
    scan = scan_design(objects, feature_cols=data.feature_cols)
    with pytest.raises(ValueError, match="the scan suggests 'siCTRL'"):
        scan.setup("Area")
    with pytest.raises(ValueError, match="fingerprint=..."):
        scan.setup("Area", goal="fingerprint", control={"condition": "siCTRL"})
    with pytest.raises(ValueError, match="goal must be one of"):
        scan.setup("Area", goal="mean")
    setup = scan.setup("Area", control={"condition": "siCTRL"}, observation_unit="cell")
    assert isinstance(setup, Setup)
    assert setup.fov_col == "position_id"
    assert setup.features == tuple(data.feature_cols)
    assert setup.group_cols == ("condition", "biological_repeat", "technical_repeat")


def test_the_screen_is_separate_acquisitions_sharing_the_control():
    data = catalog.load("cct8_filopodia_screen")
    objects = data.objects.drop(columns=["target"])
    scan = scan_design(objects, feature_cols=data.feature_cols, observation_unit="cell")
    sets = scan.design["repeat_sets"]
    assert len(sets) == 6 and all(len(s["repeats"]) == 3 for s in sets)
    text = scan.summary()
    assert "holding different conditions (separate acquisitions)" in text
    assert "siCCT8_1, siCCT8_2 and siCCT8_3: siCCT8#5 and siCCT8#8 with the control" in text
    assert len(scan.design["reagent_groups"]) == 6
    # 18 repeats are too many columns: the layout is summarised per condition.
    assert "condition     repeats  per repeat  total" in text


def test_the_glioma_tracks_exported_twice_are_found():
    data = catalog.load("cct7_cct8_glioma_cdm_tracks")
    scan = scan_design(
        *data.tables,
        feature_cols=data.feature_cols,
        control=data.control,
        observation_unit=data.observation_unit,
    )
    assert len(scan.integrity.findings) == 6
    assert scan.status is Status.PROBLEM


def test_a_sparse_repeat_is_a_caution():
    objects = _usual()
    drop = (objects["condition"] == "drugB") & (objects["biological_repeat"] == "2026_03_01")
    objects = objects[
        ~drop | (objects["image_file"].str.endswith("_0.tif") & (objects["label_id"] < 4))
    ]
    scan = scan_design(
        objects,
        feature_cols=["area", "speed", "aspect"],
        control={"condition": "siCTRL"},
        observation_unit="cell",
    )
    [caution] = _texts(scan, Status.CAUTION)
    assert caution.startswith("Few cells in biological repeat 2026_03_01: drugB (3 cells)")


def test_one_repeat_is_a_pilot_and_errors_skip_the_copy_checks():
    objects = _usual(n_repeats=1)
    scan = scan_design(objects, feature_cols=["area"], control={"condition": "siCTRL"})
    assert "a pilot; the report describes it, and nothing is tested across repeats" in _flat(
        scan.summary()
    )
    broken = scan_design(objects, feature_cols=["area", "nope"], observation_unit="cell")
    assert broken.integrity is None
    [error] = _texts(broken, Status.PROBLEM)
    assert "not in the observation table. Where: missing: 'nope'." in error


def test_reagent_groups_by_name():
    names = ["siCTRL", "siCCT8#5", "siCCT8#8", "shNF2_1", "shNF2_2", "YAP", "drug"]
    assert reagent_groups(names, "siCTRL") == {
        "CCT8": ["siCCT8#5", "siCCT8#8"],
        "NF2": ["shNF2_1", "shNF2_2"],
    }
    assert reagent_groups(["MOCK", "YAP", "LATS1"], "MOCK") == {}


def test_the_full_summary_adds_the_checks_as_they_print():
    scan = scan_design(_usual(), feature_cols=["area", "speed", "aspect"], observation_unit="cell")
    full = scan.summary(detail="full")
    assert "MorphoCoverage data check" in full
    assert full.startswith(scan.summary().split("\n")[0])


def test_the_fields_are_those_of_the_fov_table_empty_ones_included():
    objects = _usual()
    fovs = objects[["condition", "biological_repeat", "image_file"]].drop_duplicates()
    empty = {"condition": "siCTRL", "biological_repeat": "2026_01_01", "image_file": "e.tif"}
    fovs = pd.concat([fovs, pd.DataFrame([empty])], ignore_index=True)
    scan = scan_design(objects, fovs)
    assert scan.design["fields"] == len(fovs) == 55
    assert scan.design["per_field"] == (0, 10.0, 10)
    assert "55 fields, 0 to 10 observations per field (median 10)." in scan.summary()
    assert scan_design(objects).design["fields"] == 54
    # No observation at all: zero fields, not a crash.
    assert scan_design(objects.iloc[:0]).design["per_field"] == (0, 0.0, 0)


def test_a_condition_or_repeat_held_only_by_empty_fields_is_counted():
    """The conditions and repeats are those of the FOV table, as its fields are (#363)."""
    objects = _usual()
    fovs = objects[["condition", "biological_repeat", "image_file"]].drop_duplicates()
    extra = [
        {"condition": "drugC", "biological_repeat": "2026_01_01", "image_file": "c.tif"},
        {"condition": "siCTRL", "biological_repeat": "2026_04_01", "image_file": "r.tif"},
    ]
    fovs = pd.concat([fovs, pd.DataFrame(extra)], ignore_index=True)
    scan = scan_design(objects, fovs, control={"condition": "siCTRL"})
    assert scan.design["fields"] == len(fovs) == 56
    assert scan.design["conditions"] == ["siCTRL", "drugA", "drugB", "drugC"]
    assert scan.design["repeats"][-1] == "2026_04_01" and len(scan.design["repeats"]) == 4
    # Acquired but empty is 0 observations; never acquired stays blank.
    assert scan.layout.loc["drugC", "2026_01_01"] == 0
    assert scan.layout.loc["siCTRL", "2026_04_01"] == 0
    assert pd.isna(scan.layout.loc["drugC", "2026_02_01"])
    text = _flat(scan.summary())
    assert "Conditions: siCTRL (the control), drugA, drugB and drugC." in text
    assert "2026_04_01: the control only" in text
    # Without the FOV table, only what the observations hold.
    assert scan_design(objects).design["conditions"] == ["siCTRL", "drugA", "drugB"]
