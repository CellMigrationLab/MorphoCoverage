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
from morphocoverage.retrospective import dataset_overview
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
from morphocoverage.workflows._contrasts import normalised_readers
from morphocoverage.workflows._declaration import declaration_short
from morphocoverage.workflows._feature_space import (
    centred_tables,
    centring_line,
    separation_in_report,
)

KEY = ["condition", "biological_repeat", "image_file", "label_id"]

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
        positive=PositiveControl({"condition": "a"}, fovs_seen=5),
        observation_unit="image",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = report(objects, fovs, setup)
    full = " ".join(result.summary(detail="full").split())
    short = " ".join(result.summary().split())
    # One observation per field (owner, 2026-10-06, plan v2, replacing the earlier "not
    # applicable"): the positive control is judged by the field version of new, and the
    # unusual images run; only the clustering within fields is said not to apply.
    assert "no check could be made" not in full and "controls" in result.results
    assert result.results["controls"].status != "not assessed"
    for text in (full, short):
        assert "for unusual images" in text and "Positive control: fields of" in text
        assert "exploratory (new: an image that no other field shows)" in text
        assert "unusual images are not assessed" not in text
    assert "cluster in a few fields is not applicable: each field holds one image" in short


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
    # One image per field: the unusual images run (plan v2), only their clustering within
    # fields is said not to apply; the Foundation needs explainable features.
    short = " ".join(single.summary().split())
    assert "3. What limits my sampling? — unusual images" in short
    assert "cluster in a few fields is not applicable: each field holds one image" in short
    assert "Foundation. Why count fields? — not applicable: no explainable features" in short
    assert "unusual images on the 6 columns of T (relative to the control of each repeat" in short


def test_b8_b9_embedding_only_values_and_space_are_said_as_they_are():
    *_, result = _embedding_only(normalize="control_mean")
    full = " ".join(result.summary(detail="full").split())
    assert "Values: as given (not normalised)" not in full
    assert "Values: the other feature tables centred per biological repeat" in full
    # One table: its columns, not a weighted combined space.
    space = "unusual cells on the 12 columns of DINO (relative to the control of each repeat; "
    assert space in full
    assert "exploratory), no definition of new" in full
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
    assert (
        "the centred map, the populations and the separation of the conditions read them, "
        "never a test" in full
    )


SEVERAL = {"condition": {"cA": ["a1"], "cB": ["b1"]}}


def _several(seed: int = 0):
    """cA, a1, cB, b1 in 3 repeats (cB not imaged in R2), 3 fields of 6 cells; ``area`` in
    the observation table, an explainable table ``fiber`` (fib_len, fib_w) and a table
    ``DINO`` of 4 columns, a1 and b1 shifted."""
    rng = np.random.default_rng(seed)
    rows = [
        {
            "condition": c,
            "biological_repeat": f"R{r}",
            "image_file": f"{c}_{r}_{f}.tif",
            "label_id": k,
        }
        for c in ("cA", "a1", "cB", "b1")
        for r in range(3)
        for f in range(3)
        for k in range(1, 7)
        if (c, r) != ("cB", 1)
    ]
    objects = pd.DataFrame(rows)
    fovs = objects[KEY[:3]].drop_duplicates().reset_index(drop=True)
    shift = objects["condition"].isin(["a1", "b1"]).to_numpy() * 0.8
    objects["area"] = 10 * np.exp(rng.normal(0, 0.2, len(objects)) + shift)
    fiber = objects[KEY].assign(
        fib_len=5 + rng.normal(size=len(objects)) + shift, fib_w=2 + rng.normal(size=len(objects))
    )
    dino = objects[KEY].assign(**{f"d{j}": rng.normal(size=len(objects)) for j in range(4)})
    return objects, fovs, {"fiber": (fiber, "explainable"), "DINO": dino}


def test_b7_the_csvs_name_each_features_table(tmp_path):
    objects, fovs, tables = _several()
    setup = Setup(
        readout="area",
        features=("area", "fib_len", "fib_w"),
        control=SEVERAL,
        master_control="cA",
        observation_unit="cell",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = report(objects, fovs, setup, feature_tables=tables)
        html_report(result, objects, fovs, tmp_path, feature_tables=tables)
    figures = tmp_path / "figures"
    across = pd.read_csv(figures / "tests_across_experiments.csv")
    assert list(across.columns[2:4]) == ["feature", "table"]
    assert set(across.loc[across["feature"] == "fib_len", "table"]) == {"fiber"}
    pairs = sorted(figures.glob("*_pairs.csv"))
    assert pairs and all(pd.read_csv(p).columns[0] == "table" for p in pairs)
    loadings = pd.read_csv(next(figures.glob("pca_loadings_raw_Combined.csv")))
    assert dict(zip(loadings["feature"], loadings["table"], strict=True))["d0"] == "DINO"
    assert dict(zip(loadings["feature"], loadings["table"], strict=True))["fib_w"] == "fiber"


def test_b10_embedding_only_says_the_rows_across_experiments_are_not_tested():
    objects, fovs, tables = _several()
    setup = Setup(control=SEVERAL, master_control="cA", observation_unit="cell")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = report(objects, fovs, setup, feature_tables={"DINO": tables["DINO"]})
    full = " ".join(result.summary(detail="full").split())
    assert "tested feature by feature only" not in full
    assert "there are no explainable features to test them on" in full


def test_seed5_a_repeat_without_its_control_is_centred_on_the_conditions_in_every_repeat():
    # Audit 8 (owner, 2026-10-06): no control decides the centre, so no repeat is left out.
    objects, fovs, tables = _several()
    names = tuple(f"d{j}" for j in range(4))
    merged = objects.merge(tables["DINO"], on=KEY)
    setup = Setup(control=SEVERAL, observation_unit="cell", feature_tables={"DINO": names})
    assert centred_tables(merged, fovs, setup)[list(names)].notna().all().all()
    line = centring_line(merged, fovs, setup)
    assert "per-condition medians of cA, a1 and b1, the conditions in every repeat" in line
    assert "uncentred" not in line


def test_seed22_separation_only_on_spaces_with_a_map():
    objects, fovs, tables = _several()
    merged = objects.merge(tables["DINO"], on=KEY)
    names = tuple(f"d{j}" for j in range(4))
    two = Setup(readout="area", features=("area",), feature_tables={"DINO": names})
    found = separation_in_report(merged, fovs, two)
    assert set(found.table["space"]) == {"Combined", "DINO"}  # no one-column map
    alone = Setup(readout="area", features=("area",))
    assert separation_in_report(merged, fovs, alone) is None


def test_b6_a_condition_label_is_never_capitalised():
    objects, fovs, _ = _several()
    overview = dataset_overview(objects, fovs, feature_cols=["area"], observation_unit="cell")
    assert "\ncB in 2 of the 3 repeats." in overview.summary(detail="full")


def test_b5_without_a_technical_repeat_the_fields_alone_are_resampled():
    objects, fovs, _ = _several()
    setup = Setup(readout="area", control={"condition": "cA"}, observation_unit="cell")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = report(objects, fovs, setup)
    full = " ".join(result.summary(detail="full").split())
    assert "from resampling the fields as acquired" in full
    assert "resampling the technical repeats" not in full and "within wells" not in full


def test_seed7_the_overview_and_the_declaration_print_the_one_list():
    objects, fovs, _ = _several()
    one = Setup(
        readout="area",
        control={"condition": "cA"},
        normalize="control_mean",
        observation_unit="cell",
    )
    several = Setup(
        readout="area", control=SEVERAL, normalize="control_mean", observation_unit="cell"
    )
    for setup, words in ((one, normalised_readers()), (several, normalised_readers(True))):
        assert words in setup.normalized_sentence(fovs, objects)
        assert words in " ".join(" ".join(setup.declaration()).split())
        assert words in " ".join(declaration_short(setup).split())
