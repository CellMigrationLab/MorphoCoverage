"""Regression tests of audit 8B (2026-10-06): the paths the matrix of tables, controls,
normalisation, units and targets found crashing or printing what did not happen."""

import json
import re
import warnings

import pandas as pd
import pytest
from test_embedding_only import _cells

from morphocoverage.objectives import RareTarget
from morphocoverage.workflows import (
    Answers,
    Setup,
    html_report,
    load_answers,
    report,
    save_answers,
)

MITOTIC = RareTarget(
    detection_probability=0.95, min_fov_occurrences=2, object_prevalence=0.2, phenotype="mitotic"
)


def _visible(page: str) -> str:
    text = re.sub(r"<script.*?</script>", "", page, flags=re.DOTALL)
    return " ".join(re.sub(r"<[^>]+>", " ", text).split())


def _embedding_only(**fields):
    objects, fovs, table = _cells()
    setup = Setup(control={"condition": "ctrl"}, observation_unit="cell", **fields)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = report(objects, fovs, setup, feature_tables={"DINO": table})
    return objects, fovs, table, result


@pytest.mark.parametrize(
    "fields",
    [{"normalize": "control_mean"}, {"rare": MITOTIC, "rare_label": "mitotic"}],
    ids=["normalize", "rare_label"],
)
def test_b1_b2_an_embedding_only_page_is_built_with_normalize_or_a_rare_label(fields, tmp_path):
    objects, fovs, table, result = _embedding_only(**fields)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        page = html_report(result, objects, fovs, tmp_path, feature_tables={"DINO": table})
    visible = _visible(page.read_text())
    assert "The whole tables" in visible and "Plan the next experiment" in visible
    if "normalize" in fields:  # the populations counted the centred values: said on them
        assert (tmp_path / "figures" / "separation.csv").exists()


EMPTY = {"condition": {"cA": {}, "cB": {}}}
NOTHING_UNDER = "have no condition under them and no parent"


def test_b3_several_controls_with_nothing_under_them_are_refused_naming_what_to_declare():
    with pytest.raises(ValueError, match=NOTHING_UNDER) as error:
        Setup(readout="area", control=EMPTY, master_control="cA")
    assert "control_parents" in str(error.value) and "'cA' and 'cB'" in str(error.value)
    # One member, or a parent, is enough: the declaration compares something.
    Setup(readout="area", control={"condition": {"cA": ["a1"], "cB": {}}})
    Setup(readout="area", control=EMPTY, control_parents={"cB": "cA"})


def test_b3_the_guided_controls_question_has_no_all_default_and_asks_again():
    from types import SimpleNamespace

    from morphocoverage.workflows._guided_controls import ask_several

    fovs = pd.DataFrame({"condition": ["cA", "a1", "cB", "b1"], "biological_repeat": "R1"})
    scan = SimpleNamespace(design={"conditions": list(fovs["condition"])}, fovs=fovs)
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


def test_b14_saved_answers_with_nothing_under_the_controls_are_refused_on_reload(tmp_path):
    answers = Answers(True, {"condition": "cA"}, "population", ["area"], "area", "cell")
    path = save_answers(tmp_path, answers)
    saved = json.loads(path.read_text())
    path.write_text(json.dumps({**saved, "control": EMPTY}))
    with pytest.raises(ValueError, match=NOTHING_UNDER):
        load_answers(tmp_path)
