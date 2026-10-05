"""The guided session's control gate: one control, or several with their conditions
(owner, 2026-10-05)."""

import pytest
from ecm_design import CONTROLS, MASTER, PARENTS, tables
from test_guided import Script, _cells

from morphocoverage.workflows import Setup, guide, load_answers, save_answers
from morphocoverage.workflows._guided_controls import treatment_key

# loaded; unit; one control? (No is proposed); the controls; the conditions of THBS5_SiNC, its
# keys, those of VEC_SiNC, its keys, those of VEC, its keys; THBS5_SiNC under THBS5, its key;
# VEC_SiNC under VEC, its key; the master; then the positive control, aim, features and
# readout as for one control.
SEVERAL = [
    "y",
    "",
    "",
    "THBS5_SiNC, VEC, VEC_SiNC",
    "",
    "",
    "",
    "",
    "THBS5",
    "",
    "",
    "",
    "",
    "",
    "VEC_SiNC",
    "",
    "",
    "",
    "",
]


def _guide(objects, fovs, folder, answers):
    script = Script(answers)
    result = guide(objects, fovs, folder=folder, ask=script.ask, say=script.say, run_report=False)
    return result, script


def test_treatment_keys_strip_the_shared_stem():
    assert treatment_key("THBS5_SiNC", "THBS5_SiCol6a1") == "SiCol6a1"
    assert treatment_key("VEC", "THBS5") == "THBS5"


def test_one_control_keeps_the_single_control_question(tmp_path):
    result, script = _guide(_cells(), None, tmp_path, ["y", "", "", "", "", "", "all", ""])
    assert "Are all your conditions compared with the same control?" in " ".join(script.prompts)
    assert result.answers.control == {"condition": "siCTRL"}
    assert result.setup.contrast_table is None
    assert "Each condition against siCTRL" in script.text
    assert "The images per condition" not in script.text


def test_several_controls_are_proposed_confirmed_and_saved(tmp_path):
    objects, fovs = tables()
    result, script = _guide(objects, fovs, tmp_path, SEVERAL)
    # The likeliest control misses R5: No is proposed, after the table of images.
    assert "is not imaged in every biological repeat" in script.text
    assert "[y/N]" in next(p for p in script.prompts if "same control" in p)
    answers = result.answers
    assert answers.control["condition"]["THBS5_SiNC"] == {
        "SiCol6a1": "THBS5_SiCol6a1",
        "SiFBN2": "THBS5_SiFBN2",
    }
    assert answers.control["condition"]["VEC"] == {"THBS5": "THBS5"}
    assert answers.control_parents == {
        "THBS5_SiNC": ("THBS5", "transfection"),
        "VEC_SiNC": ("VEC", "transfection"),
    }
    assert answers.master_control == "VEC_SiNC"
    owner = Setup(readout="area", control=CONTROLS, control_parents=PARENTS, master_control=MASTER)
    guided = result.setup.contrast_table
    assert sorted(guided["label"]) == sorted(owner.contrast_table["label"])
    assert len(guided) == 11
    # The contrast table is shown, one test per line, with its family.
    assert "transfection effect: THBS5_SiNC/THBS5 vs VEC_SiNC/VEC" in script.text
    assert "(family vs_master_raw)" in script.text
    # Saved and read back, the declaration is the same.
    save_answers(tmp_path, answers)
    again = load_answers(tmp_path)
    assert again.control_parents == answers.control_parents
    assert again.version == 2
    assert "VEC_SiNC under VEC" in " ".join(again.lines())


def test_the_unit_comes_first_and_the_aim_is_asked_in_plain_words(tmp_path):
    # loaded; unit; one control?; control; positive; aim (2: the whole distribution);
    # features; readout; tolerance
    answers = ["y", "", "", "", "", "2", "all", "", ""]
    result, script = _guide(_cells(), None, tmp_path, answers)
    said = script.text
    # The unit is the first choice after "did it load", before the gate question.
    assert "What is one row?" in said
    assert "load correctly" in script.prompts[0] and script.prompts[1] == "> "
    assert "same control" in script.prompts[2]
    assert "whether the whole distribution differs (shape, spread, tails), or both?" in said
    assert result.answers.aim == "fingerprint"


def test_a_phenotype_column_is_offered_and_declared(tmp_path):
    cells = _cells()
    cells["mitotic"] = (cells.index % 7 == 0).astype(int)
    # loaded; unit; one control?; control; positive; aim; features (area); readout; mitotic
    result, script = _guide(cells, None, tmp_path, ["y", "", "", "", "", "", "1", "mitotic"])
    assert "Is there a column that marks a phenotype you care about" in script.text
    assert result.answers.rare_label == "mitotic"
    assert result.setup.rare_label == "mitotic"
    assert result.setup.rare.object_prevalence == pytest.approx((cells.index % 7 == 0).mean())
    assert load_answers(tmp_path).rare_label == "mitotic"


def test_the_session_can_stop_before_the_report(tmp_path):
    script = Script(["y", "", "", "", "", "", "all", "", "save and stop"])
    result = guide(_cells(), folder=tmp_path, ask=script.ask, say=script.say)
    assert result.report is None and result.answers is not None
    assert "Stopped with the answers saved" in script.text
    assert "Run the report now, or save the answers and stop?" in script.text
