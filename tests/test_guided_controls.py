"""The guided session's control gate: one control, or several with their conditions
(owner, 2026-10-05)."""

from ecm_design import CONTROLS, MASTER, PARENTS, tables
from test_guided import Script, _cells

from morphocoverage.workflows import Setup, guide, load_answers, save_answers
from morphocoverage.workflows._guided_controls import treatment_key

# loaded; one control? (No is proposed); the controls; the conditions of THBS5_SiNC, its
# keys, those of VEC_SiNC, its keys, those of VEC, its keys; THBS5_SiNC under THBS5, its key;
# VEC_SiNC under VEC, its key; the master; then the positive control, aim, features, readout
# and unit as for one control.
SEVERAL = [
    "y",
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
    result, script = _guide(_cells(), None, tmp_path, ["y", "", "", "", "", "all", "", ""])
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
