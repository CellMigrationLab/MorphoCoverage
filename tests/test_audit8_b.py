"""Regression tests of audit 8B (2026-10-06): the paths the matrix of tables, controls,
normalisation, units and targets found crashing or printing what did not happen."""

import json
import re
import warnings

import numpy as np
import pandas as pd
import pytest
from test_embedding_only import _cells

from morphocoverage.objectives import RareTarget
from morphocoverage.workflows import (
    Answers,
    PositiveControl,
    Setup,
    default_novelty,
    html_report,
    load_answers,
    report,
    save_answers,
    scan_design,
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


def test_b11_the_scan_accepts_several_controls_as_setup_declares_them():
    objects, fovs, _ = _cells()
    controls = {"condition": {"ctrl": ["a"], "b": []}}
    scan = scan_design(objects, fovs, observation_unit="cell", control=controls)
    assert scan.data_check.ok and scan.status.marker != "❌"
    assert scan.design["controls"] == ["ctrl", "b"]
    text = scan.summary()
    assert "ctrl (a control)" in text and "b (a control)" in text
    assert "Several conditions" not in text  # one condition besides the controls
    setup = scan.setup("mitotic", control_parents={"b": "ctrl"})
    assert setup.own_references == {"a": "ctrl", "b": "ctrl", "ctrl": "ctrl"}


FEATURES = ("f1", "f2", "f3")


def _one_per_field(seed: int = 0):
    """3 conditions in 3 repeats, 6 fields of one image each; features f1, f2, f3 with
    ``a`` shifted, and a table of 6 other columns."""
    rng = np.random.default_rng(seed)
    rows = [
        {"condition": c, "biological_repeat": f"R{r}", "image_file": f"{c}_{r}_{f}.tif"}
        for c in ("ctrl", "a", "b")
        for r in range(3)
        for f in range(6)
    ]
    objects = pd.DataFrame(rows)
    fovs = objects.copy()
    shift = objects["condition"].eq("a").to_numpy() * 1.0
    for name in FEATURES:
        objects[name] = 10 + rng.normal(size=len(objects)) + shift
    table = objects[["condition", "biological_repeat", "image_file"]].assign(
        **{f"e{j}": rng.normal(size=len(objects)) for j in range(6)}
    )
    return objects, fovs, table


def test_seed3_b4_one_image_per_field_says_why_and_declares_only_what_runs():
    objects, fovs, _ = _one_per_field()
    setup = Setup(
        readout="f1",
        features=FEATURES,
        control={"condition": "ctrl"},
        novelty=default_novelty(FEATURES),
        positive=PositiveControl({"condition": "a"}, fovs_seen=3),
        observation_unit="image",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = report(objects, fovs, setup)
    full = " ".join(result.summary(detail="full").split())
    short = " ".join(result.summary().split())
    assert "no check could be made" not in full and "controls" not in result.results
    assert "the positive control is judged by the definition of new" in short
    for text in (full, short):
        assert "for unusual images" not in text and "Report-only rule" not in text
        assert "Positive control: fields of" not in text
        assert "unusual images are not assessed, as each field holds one image" in text


def test_seed3_seed18_b12_embedding_only_gives_the_real_reason():
    objects, fovs, table = _one_per_field()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        alone = report(objects, fovs, Setup(observation_unit="image"), feature_tables={"T": table})
        single = report(
            objects,
            fovs,
            Setup(control={"condition": "ctrl"}, observation_unit="image"),
            feature_tables={"T": table},
        )
    short = " ".join(alone.summary().split())
    assert "9. Do the conditions differ? — not applicable: no control declared" in short
    assert "2. Did I image enough? — not applicable: no target declared; declare the" in short
    assert "fingerprint) or a labelled phenotype (rare with rare_label)" in short
    assert "unusual images not assessed without a control" in short
    # One image per field: the reason, in question 3 and the Foundation, as in any report.
    short = " ".join(single.summary().split())
    assert "3. What limits my sampling? — not applicable: with one image per field" in short
    assert "Foundation. Why count fields? — not applicable: each field holds one image" in short
    assert "unusual images not assessed, as each field holds one image" in short
    assert "the copy checks, unusual observations" not in short


def test_b8_b9_embedding_only_values_and_space_are_said_as_they_are():
    *_, result = _embedding_only(normalize="control_mean")
    full = " ".join(result.summary(detail="full").split())
    assert "Values: as given (not normalised)" not in full
    assert "Values: the other feature tables centred per biological repeat" in full
    # One table: its columns, not a weighted combined space.
    assert "unusual cells on the 12 columns of DINO, no definition of new" in full
    assert "weighted combined space" not in full


def test_seed6_the_batch_line_says_unusual_observations_read_the_explainable_features():
    objects, fovs, table = _cells()
    objects["area"] = np.random.default_rng(1).normal(10, 1, len(objects))
    setup = Setup(
        readout="area",
        features=("area", "mitotic"),
        control={"condition": "ctrl"},
        normalize="control_mean",
        observation_unit="cell",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = report(objects, fovs, setup, feature_tables={"DINO": table})
    full = " ".join(result.summary(detail="full").split())
    assert "the centred map, the populations and the separation read them, never a test" in full
