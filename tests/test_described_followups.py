"""MC-2 follow-ups (#631): the experiment block of MorphoFeature's coverage.json.

1. References that form a cycle (``A -> B -> A``, or longer) are refused as malformed.
2. A ``coverage.json`` that names a key twice in one object is refused as malformed, never
   read last-wins.
3. Tables that carry ``condition_id`` get no control guessed from the names, even without
   their ``coverage.json``: the scan says the controls come from the description or are
   declared by ID.
4. ``guide(description=...)`` skips the control questions when the description gives a
   default, and asks the user to confirm the target suggestions (Enter leaves them).
5. The golden export of Spec fixture ``v10`` (a reference chain).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from _helpers_morphofeature_exports import EXPORTS, provenance, with_feature

from morphocoverage.described import (
    MALFORMED,
    DescriptionRefused,
    parse_coverage,
    read_coverage,
    read_experiment,
)
from morphocoverage.workflows import described_setup, scan_design
from morphocoverage.workflows._guided_controls import proposed_controls
from morphocoverage.workflows.guided import guide, load_answers

FAST = {"readout": "x", "features": ("x",)}
CHAIN = {
    "THBS5": None,
    "THBS5_SiCol6a1": "THBS5_SiNC",
    "THBS5_SiNC": "THBS5",
    "VEC": None,
    "VEC_SiCol6a1": "VEC_SiNC",
    "VEC_SiNC": "VEC",
}


def _record(name: str) -> dict:
    return json.loads((EXPORTS / name / "coverage.json").read_text(encoding="utf-8"))


def _refused(code: str, call, *args) -> DescriptionRefused:
    with pytest.raises(DescriptionRefused) as refused:
        call(*args)
    assert refused.value.code == code, str(refused.value)
    return refused.value


# --- 1. Reference cycles ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "references", "cycle"),
    [
        ("v03", {"CTRL": "DrugA", "DrugA": "CTRL"}, "CTRL -> DrugA -> CTRL"),
        ("v10", {"VEC": "VEC_SiCol6a1"}, "VEC -> VEC_SiCol6a1 -> VEC_SiNC -> VEC"),
        ("v10", {"THBS5": "THBS5_SiNC"}, "THBS5 -> THBS5_SiNC -> THBS5"),
    ],
)
def test_references_that_form_a_cycle_are_refused(name, references, cycle) -> None:
    record = _record(name)
    record["experiment"]["references"].update(references)
    error = _refused(MALFORMED, parse_coverage, record)
    assert f"experiment.references form a cycle ({cycle})" in str(error)


def test_a_chain_that_ends_at_a_root_is_no_cycle() -> None:
    # v10: VEC_SiCol6a1 -> VEC_SiNC -> VEC, and a second chain beside it: both accepted.
    assert dict(read_coverage(EXPORTS / "v10").experiment.references) == CHAIN


# --- 2. Duplicate JSON keys -------------------------------------------------------------------


def _with_duplicate(tmp_path: Path, old: str, new: str) -> Path:
    text = (EXPORTS / "v03" / "coverage.json").read_text(encoding="utf-8")
    assert text.count(old) == 1
    (tmp_path / "coverage.json").write_text(text.replace(old, new, 1), encoding="utf-8")
    return tmp_path


@pytest.mark.parametrize(
    ("old", "new", "key"),
    [
        # Read last-wins, the block would be valid and DrugA's reference silently CTRL.
        ('"references": {', '"references": {\n      "DrugA": null,', "DrugA"),
        ('"experiment": {', '"experiment": {"conditions": {},', "conditions"),
        ('"schema_version": 3', '"schema_version": 2,\n  "schema_version": 3', "schema_version"),
    ],
)
def test_a_key_named_twice_is_refused(tmp_path: Path, old, new, key) -> None:
    folder = _with_duplicate(tmp_path, old, new)
    error = _refused(MALFORMED, read_coverage, folder)
    assert f"the key {key!r} appears twice in one JSON object" in str(error)


def test_the_golden_exports_name_no_key_twice() -> None:
    for folder in sorted(p for p in EXPORTS.iterdir() if p.is_dir()):
        assert read_coverage(folder).experiment is not None


# --- 3. No control guessed from the names of a described table --------------------------------


def _blockless_experiment(name: str):
    objects, fovs = with_feature(name)
    record = _record(name)
    record.pop("experiment")
    record["schema_version"] = 2
    return objects, fovs, read_experiment(parse_coverage(record, f"{name}.json"), fovs)


CONTROLS_TEXT = (
    "Controls: the tables carry condition_id (a described run), so no control is guessed from "
    "the condition names; the controls come from the experiment description"
)


# Without condition_id the folder labels are read, as before: v33's folder is 'control'.
@pytest.mark.parametrize(("name", "named"), [("v03", "CTRL"), ("v27", "CTRL"), ("v33", "control")])
def test_without_the_coverage_json_no_control_is_guessed_by_name(name, named) -> None:
    objects, fovs = with_feature(name)
    scan = scan_design(objects, fovs, feature_cols=["x"])
    assert scan.design["control_guess"] is None and scan.design["control_guesses"] == []
    found = [f.text for f in scan.findings if f.about == "controls"]
    assert len(found) == 1 and found[0].startswith(CONTROLS_TEXT)
    assert "declared by their ID" in found[0]
    (question,) = [q for q in scan.questions if q.key == "control"]
    assert question.suggested is None
    assert "named like a control" not in scan.summary()
    assert proposed_controls(scan) == []  # the guided session proposes none by name either
    with pytest.raises(ValueError, match="by its condition_id") as refused:
        scan.setup("x", features=("x",))
    assert "suggests" not in str(refused.value)
    # Without condition_id the same names are still read, as before (the guess is a hint).
    plain = fovs.drop(columns=["repeat_group", "condition_id", "technical_repeat_id"])
    assert scan_design(objects, plain, feature_cols=["x"]).design["control_guess"] == named


def test_condition_id_in_the_observation_table_alone_is_a_described_table() -> None:
    objects, fovs = with_feature("v03")
    ids = fovs[["condition", "biological_repeat", "technical_repeat", "position_id"]]
    joined = objects.merge(fovs, on=list(ids.columns), how="left", validate="many_to_one")
    scan = scan_design(joined, feature_cols=["x"])
    assert scan.design["control_guess"] is None
    assert any(f.text.startswith(CONTROLS_TEXT) for f in scan.findings)


def test_a_declared_control_needs_no_controls_finding() -> None:
    objects, fovs = with_feature("v03")
    scan = scan_design(objects, fovs, feature_cols=["x"], control={"condition": "CTRL"})
    assert not any(f.about == "controls" for f in scan.findings)


def test_without_the_block_the_description_lists_no_name_guess_either() -> None:
    objects, fovs, experiment = _blockless_experiment("v33")
    scan = scan_design(objects, fovs, feature_cols=["x"], description=experiment)
    assert scan.design["control_guess"] is None
    assert not any(f.text.startswith(CONTROLS_TEXT) for f in scan.findings)  # MISSING says it


def test_the_scan_lists_the_descriptions_control_first() -> None:
    # v33's control 'ctrl' sorts after 'DrugA'; the description's control is listed first.
    objects, fovs = with_feature("v33")
    experiment = read_experiment(EXPORTS / "v33", fovs)
    scan = scan_design(objects, fovs, feature_cols=["x"], description=experiment)
    assert scan.design["conditions"] == ["ctrl", "DrugA"]
    assert list(scan.layout.index) == ["ctrl", "DrugA"]
    # A description whose control is not the one named like a control: its control is first.
    record = _record("v03")
    record["experiment"]["references"] = {"CTRL": "DrugA", "DrugA": None}
    objects, fovs = with_feature("v03")
    swapped = read_experiment(parse_coverage(record, "v03.json"), fovs)
    scan = scan_design(objects, fovs, feature_cols=["x"], description=swapped)
    assert scan.design["conditions"] == ["DrugA", "CTRL"]
    assert "DrugA (the description's control; not declared) and CTRL" in scan.summary()


def test_a_described_scan_takes_the_setups_own_experiment_field() -> None:
    # Setup(experiment=...) names the correction families; it reached described_setup's own
    # parameter and failed (TypeError) when the scan had a description.
    objects, fovs = with_feature("v27")
    experiment = read_experiment(EXPORTS / "v27", fovs)
    scan = scan_design(objects, fovs, feature_cols=["x"], description=experiment)
    families = {"DrugA": "one", "DrugB": "one", "DrugC": "two"}
    setup = scan.setup("x", features=("x",), experiment=families)
    assert setup.experiment == families and setup.control == {"condition": "CTRL"}
    assert described_setup(experiment, **FAST, experiment=None).control == {"condition": "CTRL"}


# --- 4. guide() reads the description ---------------------------------------------------------


class Answers:
    """Answers prompts by their start (Enter otherwise) and records what was asked and said."""

    def __init__(self, by_prompt: dict[str, list[str]] | None = None) -> None:
        self.by_prompt = {k: list(v) for k, v in (by_prompt or {}).items()}
        self.prompts: list[str] = []
        self.said: list[str] = []

    def ask(self, prompt: str) -> str:
        # A choice says its question and asks "> "; the other questions ask their prompt.
        prompt = self.said[-1] if prompt.strip() == ">" else prompt
        self.prompts.append(prompt)
        assert len(self.prompts) < 200, "the session keeps asking"
        for start, answers in self.by_prompt.items():
            if prompt.startswith(start) and answers:
                return answers.pop(0)
        return ""

    def say(self, text: str) -> None:
        self.said.append(text)

    @property
    def text(self) -> str:
        return "\n".join([*self.said, *self.prompts])


def _guide(name: str, folder: Path, answers: Answers, *, describe: bool = True, record=None):
    objects, fovs = with_feature(name)
    objects = objects.drop(columns=["Cells__area"])
    coverage = EXPORTS / name if record is None else parse_coverage(record, f"{name}.json")
    experiment = read_experiment(coverage, fovs) if describe else None
    return guide(
        objects,
        fovs,
        folder=folder,
        ask=answers.ask,
        say=answers.say,
        run_report=False,
        description=experiment,
    )


CONTROL_PROMPTS = ("Which condition is the control?", "Are all your conditions compared")


def test_guide_skips_the_control_questions_when_the_description_gives_them(tmp_path) -> None:
    answers = Answers()
    result = _guide("v27", tmp_path, answers)
    assert not any(p.startswith(CONTROL_PROMPTS) for p in answers.prompts)
    assert "The controls come from the experiment description (description b3fbac76f566)" in (
        answers.text
    )
    assert "DrugB against CTRL, its control in the description" in answers.text
    assert result.answers.control is None
    assert result.setup.control == {"condition": "CTRL"}
    assert result.setup.described.chosen is None
    assert (
        "Control: the experiment description's references (description b3fbac76f566)."
        in result.answers.lines()
    )
    # The positive control is still asked, among the conditions that are not the control.
    assert any(p.startswith("Is one of the conditions a positive control?") for p in answers.said)


def test_guide_compares_a_chain_as_the_description_gives_it(tmp_path) -> None:
    result = _guide("v10", tmp_path, Answers())
    pairs = set(
        zip(result.setup.contrasts["condition"], result.setup.contrasts["reference"], strict=True)
    )
    assert pairs == {(c, r) for c, r in CHAIN.items() if r is not None}
    assert result.setup.control is None


def test_guide_asks_the_control_when_the_description_gives_none(tmp_path) -> None:
    record = _record("v03")
    record["experiment"]["references"]["DrugA"] = None
    answers = Answers({"Which condition is the control?": ["CTRL"]})
    result = _guide("v03", tmp_path, answers, record=record)
    assert any(p.startswith("Which condition is the control?") for p in answers.said)
    assert result.setup.control == {"condition": "CTRL"}


def test_guide_asks_a_blockless_export_for_a_control_until_one_is_declared(tmp_path) -> None:
    record = _record("v33")
    record.pop("experiment")
    record["schema_version"] = 2
    # Enter ("no control") is not taken: the Setup would refuse it (experiment_block_missing).
    answers = Answers({"Which condition is the control?": ["", "ctrl"]})
    result = _guide("v33", tmp_path, answers, record=record)
    assert "No control is defaulted:" in answers.text
    assert "needs its controls declared by their IDs" in answers.text
    assert result.setup.control == {"condition": "ctrl"}


def test_guide_never_confirms_the_target_suggestions_on_enter(tmp_path) -> None:
    answers = Answers()
    result = _guide("v27", tmp_path, answers)
    asked = [s for s in answers.said if s.startswith("Group the reagents of one target")]
    assert asked and "leave them unconfirmed (not used)  (default)" in asked[0]
    assert "SRC: DrugA and DrugC" in answers.text and "src: DrugB" in answers.text
    assert result.answers.targets is None and result.setup.targets is None


def test_guide_uses_the_targets_the_user_confirms(tmp_path) -> None:
    answers = Answers({"Group the reagents of one target": ["1"]})
    result = _guide("v27", tmp_path, answers)
    confirmed = {"DrugA": "SRC", "DrugB": "src", "DrugC": "SRC"}
    assert result.answers.targets == confirmed and result.setup.targets == confirmed
    assert load_answers(tmp_path).targets == confirmed
    assert "Targets: as confirmed by the user: SRC (DrugA and DrugC); src (DrugB)." in (
        result.answers.lines()
    )


def test_guide_uses_the_targets_the_user_edits(tmp_path) -> None:
    edits = {"Target of DrugB": ["SRC"], "Target of DrugC": ["none"]}
    answers = Answers({"Group the reagents of one target": ["2"], **edits})
    result = _guide("v27", tmp_path, answers)
    assert result.setup.targets == {"DrugA": "SRC", "DrugB": "SRC"}


def test_guide_without_the_description_reuses_no_answers_saved_with_it(tmp_path) -> None:
    _guide("v27", tmp_path, Answers())
    saved = load_answers(tmp_path)
    assert (
        saved.dataset["description"] == read_coverage(EXPORTS / "v27").experiment.description_sha256
    )
    answers = Answers()
    _guide("v27", tmp_path, answers, describe=False)
    assert "what differs: the experiment description" in answers.text


# --- 5. The golden v10 ------------------------------------------------------------------------


def test_the_v10_golden_is_the_chain_of_its_description() -> None:
    record = provenance("v10")
    assert record["fixture"] == "fixtures/valid/v10_reference_chain.toml"
    block = read_coverage(EXPORTS / "v10").experiment
    assert dict(block.references) == CHAIN
    assert {c for c, (t, _) in block.suggestions.items() if t == "COL6A1"} == {
        "THBS5_SiCol6a1",
        "VEC_SiCol6a1",
    }
    _, fovs = with_feature("v10")
    experiment = read_experiment(EXPORTS / "v10", fovs)
    default = experiment.default_control()
    pairs = set(
        zip(default["contrasts"]["condition"], default["contrasts"]["reference"], strict=True)
    )
    assert pairs == {(c, r) for c, r in CHAIN.items() if r is not None}
