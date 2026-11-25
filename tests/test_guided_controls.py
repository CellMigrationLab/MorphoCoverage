"""The guided session's control gate: one control, or several with their conditions
(owner, 2026-10-05)."""

from __future__ import annotations

import warnings
from types import SimpleNamespace

import pandas as pd
import pytest
from _helpers_embedding import NOTHING_UNDER
from _helpers_guided import Script, _cells
from _helpers_workflows import SCREEN, _sets
from ecm_design import CONTROLS, MASTER, PARENTS, tables

from morphocoverage.workflows import Setup, guide, load_answers, report, save_answers
from morphocoverage.workflows._guided_controls import ask_experiment, treatment_key

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
    assert "Each condition against siCTRL, within each biological repeat." in script.text
    assert "one family" not in script.text and "want to log" not in script.text
    assert "Which condition is the control?" in script.text
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
    assert "corrected in the family vs_master_raw" in script.text
    assert "THBS5 vs VEC: against its own control, the main verdict, corrected within its" in (
        script.text
    )
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


# loaded; unit; one control? (No); the controls proposed (THBS5_SiNC, VEC_SiNC); the
# conditions of THBS5_SiNC, its keys, those of VEC_SiNC, its keys; THBS5, compared with
# nothing yet: VEC; THBS5_SiNC under THBS5, its key; VEC_SiNC under VEC, its key; the master
# (THBS5_SiNC proposed, a tie with VEC_SiNC); then the positive control, aim, features and
# readout.
PROPOSED = ["y", "", "", "", "", "", "", "", "VEC", "", "", "", "", "VEC_SiNC", "", "", "", ""]


def test_a_condition_compared_with_nothing_is_asked_for_its_control(tmp_path):
    objects, fovs = tables()
    result, script = _guide(objects, fovs, tmp_path, PROPOSED)
    assert "Which control should THBS5 be compared with?" in script.text
    # VEC, not named like a control, becomes THBS5's: the owner's declaration.
    assert result.answers.control["condition"]["VEC"] == {"THBS5": "THBS5"}
    assert "Which control should VEC be compared with?" not in script.text
    master = script.text[script.text.index("(the master)") :]
    assert "THBS5_SiNC  (default)" in master.split("\n4.")[0]
    owner = Setup(readout="area", control=CONTROLS, control_parents=PARENTS, master_control=MASTER)
    assert sorted(result.setup.contrast_table["label"]) == sorted(owner.contrast_table["label"])


def test_the_master_proposed_is_the_control_imaged_in_most_repeats(tmp_path):
    from types import SimpleNamespace

    from morphocoverage.workflows._guided_controls import most_imaged

    _, fovs = tables()
    scan = SimpleNamespace(objects=fovs, fovs=fovs)
    # VEC in 3 repeats, the SiNC controls in 5: a tie goes to the first declared.
    assert most_imaged(scan, ["VEC", "VEC_SiNC", "THBS5_SiNC"]) == "VEC_SiNC"
    assert most_imaged(scan, ["THBS5_SiNC", "VEC_SiNC"]) == "THBS5_SiNC"
    fewer = fovs[~((fovs["condition"] == "THBS5_SiNC") & (fovs["biological_repeat"] == "R1"))]
    assert (
        most_imaged(SimpleNamespace(objects=fewer, fovs=fewer), ["THBS5_SiNC", "VEC_SiNC"])
        == "VEC_SiNC"
    )


def test_the_report_runs_with_declared_experiments_and_the_guided_session_asks() -> None:
    # B11: the Setup's experiment column reaches every family; the guided session asks
    # whether sets of conditions are separate experiments, No by default.
    layout = {"ctrl": SCREEN["ctrl"], "siA": SCREEN["siA"], "siB": SCREEN["siB"]}
    objects, fovs = _sets(layout, {"siA": 0.4, "siB": 0.3})
    fovs = fovs.assign(screen="one")
    setup = Setup(
        readout="area",
        features=("area", "length"),
        control={"condition": "ctrl"},
        experiment="screen",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = report(objects, fovs, setup)
    tests = result.discovery.tests
    assert tests["experiment"].nunique() == 1 and tests["q_screen"].isna().all()
    assert "screen-wide" not in result.summary("full")
    scan = SimpleNamespace(objects=fovs, fovs=fovs)
    prompts: list[str] = []

    def answer(text: str) -> str:
        prompts.append(text)
        return "y"

    found = ask_experiment(answer, lambda text: None, scan, {"condition": "ctrl"})
    assert found == {"siA": "screen", "siB": "screen"}
    assert prompts[0].startswith("Are they one experiment against ctrl")
    assert ask_experiment(lambda text: "", print, scan, {"condition": "ctrl"}) is None
    several = {"condition": {"c1": ["a"], "c2": {"k": "b"}}}
    assert ask_experiment(answer, print, scan, several) == {"a": "c1", "b": "c2"}


def test_b3_the_guided_controls_question_has_no_all_default_and_asks_again():
    from types import SimpleNamespace

    from morphocoverage.workflows._guided_controls import ask_several

    fovs = pd.DataFrame(
        {"condition": ["cA", "a1", "cB", "b1"], "biological_repeat": "R1", "repeat_group": "R1"}
    )
    scan = SimpleNamespace(design={"conditions": list(fovs["condition"])}, objects=fovs, fovs=fovs)
    # Enter (asked again); cA and cB with nothing under them, a1 and b1 compared with
    # nothing, the master (refused, asked again); then a1 under cA and b1 under cB.
    replies = iter(["", "cA, cB", "", "", "", "", "", "cA, cB", "a1", "", "b1", "", ""])
    said: list[str] = []
    control, parents, master = ask_several(lambda prompt: next(replies), said.append, scan)
    text = "\n".join(said)
    assert "'all' (the default)" not in text and "Nothing was chosen." in text
    assert NOTHING_UNDER in text
    assert control == {"condition": {"cA": {"a1": "a1"}, "cB": {"b1": "b1"}}}
    assert parents is None and master == "cA"


def test_treatment_keys_entered_twice_are_asked_again_not_overwritten_448():
    # Two conditions given the key KD made {"KD": "drugB"}: drugA vanished silently.
    from morphocoverage.workflows._guided_controls import _keys

    script = Script(["n", "KD", "KD", "KDa", "KDb"])
    keys = _keys(script.ask, script.say, "ctrl", ["drugA", "drugB"])
    assert keys == {"KDa": "drugA", "KDb": "drugB"}
    assert "its own treatment key" in script.text
    script = Script(["n", "", "KDb"])  # Enter keeps the proposed key
    assert _keys(script.ask, script.say, "ctrl", ["drugA", "drugB"]) == {
        "drugA": "drugA",
        "KDb": "drugB",
    }


def test_a_taken_key_and_a_taken_condition_name_ask_for_a_key_448():
    # The fallback to the condition's name overwrote the condition already holding it.
    from morphocoverage.workflows._guided_controls import _unlinked

    found = {"ctrl": {"drugA": "drugB"}}
    script = Script(["ctrl", "drugA2"])
    controls = _unlinked(
        script.ask, script.say, ["ctrl", "drugA", "drugB"], [], found, {"ctrl", "drugB"}
    )
    assert controls == ["ctrl"]
    assert found == {"ctrl": {"drugA": "drugB", "drugA2": "drugA"}}
