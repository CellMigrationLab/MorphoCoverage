"""Other feature tables (owner, 2026-10-05): validated against the observation table, attached
as named blocks, weighted in the combined space (each table by its first singular value, owner
2026-10-06); every failure has its message."""

import json
import warnings

import numpy as np
import pandas as pd
import pytest

from morphocoverage.feature_tables import (
    FeatureTables,
    attach_feature_tables,
    mfa_weights,
    observation_identity,
)
from morphocoverage.multivariate_outcomes import CovarianceError
from morphocoverage.retrospective.criteria import OutcomeCriterion
from morphocoverage.separation import separation
from morphocoverage.workflows import Setup, html_report, optimise, report, scan_design
from morphocoverage.workflows._feature_space import attached, novelty_space, space
from morphocoverage.workflows._shared import check_workflow_inputs
from morphocoverage.workflows.setup import default_novelty


def _tables(cells: int = 2) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Two conditions, two repeats, two fields each, ``cells`` labelled cells a field; an
    embedding of 3 columns in another row order."""
    rows = [
        {"condition": c, "biological_repeat": r, "image_file": f"f{f}.tif", "label_id": k}
        for c in ("ctrl", "drug")
        for r in ("R1", "R2")
        for f in range(2)
        for k in range(1, cells + 1)
    ]
    objects = pd.DataFrame(rows)
    rng = np.random.default_rng(0)
    objects["area"] = rng.normal(size=len(objects))
    objects["shape"] = rng.normal(size=len(objects))
    fovs = objects[["condition", "biological_repeat", "image_file"]].drop_duplicates()
    embedding = objects[["condition", "biological_repeat", "image_file", "label_id"]].copy()
    for j in range(3):
        embedding[f"e{j}"] = rng.normal(size=len(embedding))
    return objects, fovs.reset_index(drop=True), embedding.iloc[::-1].reset_index(drop=True)


def _message(objects, fovs, tables) -> str:
    with pytest.raises(ValueError) as error:
        attach_feature_tables(objects, fovs, tables)
    return str(error.value)


def test_a_table_is_attached_in_the_observation_tables_order():
    objects, fovs, embedding = _tables()
    objects.index = objects.index + 100  # the caller's index is kept
    result = attach_feature_tables(objects, fovs, {"DINO": embedding})
    assert isinstance(result, FeatureTables)
    assert result.identity == ("condition", "biological_repeat", "image_file", "label_id")
    assert result.columns == {"DINO": ("e0", "e1", "e2")}
    assert list(result.objects.index) == list(objects.index)
    key = ["condition", "biological_repeat", "image_file", "label_id"]
    expected = objects.merge(embedding, on=key, how="left")["e1"].to_numpy()
    np.testing.assert_array_equal(result.objects["e1"].to_numpy(), expected)
    assert result.missing == {"DINO": 0}
    assert "3 feature columns, every observation matched once" in result.summary()


def test_one_row_per_image_is_identified_by_the_field_key():
    objects, fovs, embedding = _tables(cells=1)
    objects = objects.drop(columns="label_id")
    embedding = embedding.drop(columns="label_id")
    assert observation_identity(objects, fovs) == ("condition", "biological_repeat", "image_file")
    assert attach_feature_tables(objects, fovs, {"DINO": embedding}).objects.shape[1] == 8


def test_labels_given_as_text_match_numbers():
    objects, fovs, embedding = _tables()
    embedding["label_id"] = embedding["label_id"].astype(str)
    assert "e0" in attach_feature_tables(objects, fovs, {"DINO": embedding}).objects


def test_missing_values_are_counted_and_kept():
    objects, fovs, embedding = _tables()
    embedding.loc[[0, 1], "e0"] = np.nan
    embedding.loc[1, "e2"] = np.nan
    result = attach_feature_tables(objects, fovs, {"DINO": embedding})
    assert result.missing == {"DINO": 3} and result.incomplete == {"DINO": 2}
    assert int(result.objects["e0"].isna().sum()) == 2
    assert "3 missing values in 2 rows, which every analysis reading DINO leaves out" in (
        result.summary()
    )


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        (lambda e: e.iloc[1:], "lacks a row for 1 observation of the observation table"),
        (
            lambda e: pd.concat([e, e.iloc[[0]].assign(label_id=99)]),
            "holds 1 row matching no observation",
        ),
        (lambda e: pd.concat([e, e.iloc[[0]]]), "holds 1 observation more than once"),
        (lambda e: e.assign(e1="high"), "columns that are not numbers: e1"),
        (lambda e: e.assign(e1=np.inf), "holds 16 infinite values"),
        (lambda e: e.rename(columns={"e0": "area"}), "has columns of the observation table"),
        (lambda e: e.drop(columns="label_id"), "lacks the identity columns ['label_id']"),
        (lambda e: e.assign(z=0), "holds metadata columns ['z']"),
        (lambda e: e[["condition", "biological_repeat", "image_file", "label_id"]], "no feature"),
    ],
)
def test_each_problem_of_a_table_is_named(change, expected):
    objects, fovs, embedding = _tables()
    message = _message(objects, fovs, {"DINO": change(embedding)})
    assert expected in message and "'DINO'" in message


def test_two_tables_may_not_share_a_column():
    objects, fovs, embedding = _tables()
    message = _message(objects, fovs, {"DINOv2": embedding, "DINOv3": embedding})
    assert "'DINOv2' and 'DINOv3' share columns e0, e1 and e2" in message


@pytest.mark.parametrize(
    ("names", "expected"),
    [
        (("",), "non-empty text"),
        (("combined",), "names the observation table's features or the combined space"),
        (("Explainable",), "names the observation table's features or the combined space"),
        (("dino", "DINO"), "'DINO' names two tables"),
    ],
)
def test_each_table_has_a_name_of_its_own(names, expected):
    objects, fovs, embedding = _tables()
    tables = {
        name: embedding.rename(columns=lambda c, i=i: f"{c}_{i}" if c.startswith("e") else c)
        for i, name in enumerate(names)
    }
    assert expected in _message(objects, fovs, tables)


def test_observations_without_an_identity_of_their_own_cannot_be_matched():
    objects, fovs, embedding = _tables()
    with pytest.raises(ValueError, match="add a label_id column"):
        attach_feature_tables(objects.drop(columns="label_id"), fovs, {"DINO": embedding})


def test_every_problem_is_said_at_once():
    objects, fovs, embedding = _tables()
    message = _message(objects, fovs, {"DINO": embedding.iloc[1:].assign(e2="x")})
    assert "lacks a row" in message and "not numbers" in message


def test_each_table_is_divided_by_its_first_singular_value():
    # MFA (owner, 2026-10-06): a table of 16 correlated columns and one of 2 independent
    # ones end up with the explainable table's first singular value; explainable keeps 1.
    rng = np.random.default_rng(0)
    explainable = rng.normal(size=(200, 4))
    shared = rng.normal(size=(200, 1))
    correlated = shared + 0.1 * rng.normal(size=(200, 16))
    small = 3.0 * rng.normal(size=(200, 2))
    z = np.hstack([explainable, correlated, small])
    names = [*(f"x{i}" for i in range(4)), *(f"c{i}" for i in range(16)), "s0", "s1"]
    blocks = {"explainable": names[:4], "corr": names[4:20], "small": names[20:]}
    weights = mfa_weights(z, names, blocks)
    assert weights[:4] == pytest.approx(1.0)

    def first(columns: slice) -> float:
        part = (z * weights)[:, columns]
        return float(np.linalg.svd(part - part.mean(axis=0), compute_uv=False)[0])

    assert first(slice(4, 20)) == pytest.approx(first(slice(0, 4)))
    assert first(slice(20, 22)) == pytest.approx(first(slice(0, 4)))
    # The correlated table, whose first axis held its whole variance, is weighed down most.
    assert weights[4] < weights[20]


def test_the_setup_names_its_tables_and_refuses_a_clash():
    setup = Setup(readout="area", features=("area", "shape"), feature_tables={"DINO": ["e0"]})
    assert setup.feature_tables == {"DINO": ("e0",)}
    with pytest.raises(ValueError, match="also a declared feature"):
        Setup(readout="area", features=("area", "shape"), feature_tables={"DINO": ["shape"]})
    with pytest.raises(ValueError, match="reserved or names two tables"):
        Setup(readout="area", feature_tables={"combined": ["e0"]})
    with pytest.raises(ValueError, match="of both"):
        Setup(readout="area", feature_tables={"A": ["e0"], "B": ["e0"]})
    # Novelty never reads the tables (owner, 2026-10-06): a novelty rule is allowed with them.
    rule = Setup(readout="area", features=("area", "shape"), novelty=default_novelty(("area",)))
    Setup(
        readout="area",
        features=("area", "shape"),
        novelty=rule.novelty,
        rules=(rule.novelty_rule(),),
        feature_tables={"DINO": ["e0"]},
    )


def test_without_other_tables_nothing_is_weighted():
    setup = Setup(readout="area", features=("area", "shape"))
    assert space(setup) == (["area", "shape"], None)
    novelty = default_novelty(("area", "shape"))
    assert novelty_space(Setup(readout="area", features=("area", "shape"), novelty=novelty)) == (
        novelty,
        ["area", "shape"],
    )


def test_a_readout_alone_with_another_table_has_no_novelty_to_weigh():
    setup = Setup(readout="area", feature_tables={"DINO": ("e0", "e1")})
    assert novelty_space(setup) == (None, [])
    assert space(setup) == (
        ["area", "e0", "e1"],
        {"explainable": ("area",), "DINO": ("e0", "e1")},
    )


def test_novelty_reads_the_explainable_features_only():
    # Owner, 2026-10-06 (audit B, finding 1): the radius times sqrt(tables) called 99.6% of
    # null cells new; novelty reads the explainable features, whatever the tables.
    novelty = default_novelty(("area", "shape"))
    setup = Setup(
        readout="area",
        features=("area", "shape"),
        novelty=novelty,
        feature_tables={"DINO": ("e0", "e1", "e2", "e3", "e4", "e5", "e6", "e7")},
    )
    assert novelty_space(setup) == (novelty, ["area", "shape"])


def test_a_fingerprint_may_read_another_table_with_weights():
    metric = CovarianceError(("area", "e0"), True, weights=(1.0, 0.5))
    assert metric.parameters["weights"] == [1.0, 0.5]
    assert "weights" not in CovarianceError(("area", "e0"), True).parameters
    fingerprint = (OutcomeCriterion(metric, 0.2, "lower_is_better"),)
    setup = Setup(readout="area", feature_tables={"DINO": ("e0",)}, fingerprint=fingerprint)
    assert setup.judged_columns == ["area", "e0"]
    with pytest.raises(ValueError, match="another feature table"):
        Setup(readout="area", fingerprint=fingerprint)


def test_the_workflows_attach_the_tables_and_name_them_in_the_setup():
    objects, fovs, embedding = _tables()
    setup = Setup(readout="area", features=("area", "shape"))
    merged, named, tables = attached(objects, fovs, setup, {"DINO": embedding})
    assert named.feature_tables == {"DINO": ("e0", "e1", "e2")} and "e2" in merged
    assert tables is not None
    with pytest.raises(ValueError, match="pass the tables themselves"):
        attached(objects, fovs, named, None)
    assert attached(merged, fovs, named, None)[0] is merged


# --- The workflows on the weighted combined space ------------------------------------------


def _experiment(seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Three conditions, three repeats, four fields of 12 cells; three explainable features
    with a small shift, an embedding whose blocks separate the conditions (``sep``) and one
    of noise (``noise``)."""
    rng = np.random.default_rng(seed)
    rows = [
        {
            "condition": c,
            "biological_repeat": r,
            "image_file": f"{c}_{f}.tif",
            "label_id": k,
            "f1": rng.normal(shift),
            "f2": rng.normal(),
            "f3": rng.normal(shift),
        }
        for c, shift in (("ctrl", 0.0), ("a", 0.3), ("b", -0.3))
        for r in ("R1", "R2", "R3")
        for f in range(4)
        for k in range(1, 13)
    ]
    objects = pd.DataFrame(rows)
    fovs = objects[["condition", "biological_repeat", "image_file"]].drop_duplicates()
    key = objects[["condition", "biological_repeat", "image_file", "label_id"]]
    centre = objects["condition"].map({"ctrl": 0, "a": 1, "b": 2})
    sep = key.assign(
        **{f"s{j}": rng.normal(size=len(key)) + 3.0 * (centre == j % 3) for j in range(6)}
    )
    noise = key.assign(**{f"n{j}": rng.normal(size=len(key)) for j in range(6)})
    return objects, fovs.reset_index(drop=True), sep, noise


def test_the_separation_score_orders_a_separating_table_above_noise():
    objects, fovs, sep, noise = _experiment()
    merged = attach_feature_tables(objects, fovs, {"sep": sep, "noise": noise}).objects
    scores = separation(
        merged,
        fovs,
        spaces={"sep": list(sep.columns[4:]), "noise": list(noise.columns[4:])},
    )
    by_space = scores.spaces.set_index("space")
    # Never a neighbour from the cell's own field (owner, 2026-10-06): 36 cells of the same
    # condition in the other fields among the 132 others of a repeat, chance 36/132.
    assert by_space.loc["noise", "chance"] == pytest.approx(36 / 132)
    assert by_space.loc["sep", "share"] > 0.9
    assert by_space.loc["noise", "share"] < 0.4
    # A row per repeat by condition and per condition by repeat, in each space.
    assert scores.table["k"].eq(10).all() and len(scores.table) == 12
    assert scores.line("sep").startswith("sep: 9") and "by chance; median over 3 repeats" in (
        scores.line("sep")
    )
    rows = [line.split()[0] for line in scores.comparison().splitlines()]
    assert rows.index("sep") < rows.index("noise")  # the most separating first


def test_a_repeat_of_one_condition_is_not_scored():
    objects, fovs, sep, _ = _experiment()
    keep = objects["condition"].eq("ctrl").to_numpy()
    merged = attach_feature_tables(objects, fovs, {"sep": sep}).objects[keep]
    scores = separation(merged, fovs[fovs["condition"] == "ctrl"], spaces={"sep": ["s0", "s1"]})
    assert scores.spaces.loc[0, "single"] == 3
    assert scores.line("sep") == "sep: not scored, as no biological repeat holds two conditions."


@pytest.fixture(scope="module")
def combined_report():
    objects, fovs, sep, noise = _experiment()
    setup = Setup(
        readout="f1",
        features=("f1", "f2", "f3"),
        control={"condition": "ctrl"},
        observation_unit="cell",
    )
    tables = {"sep": sep, "noise": noise}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return report(objects, fovs, setup, integrity=True, feature_tables=tables)


def test_the_report_names_the_tables_and_tests_only_the_explainable_features(combined_report):
    setup = combined_report.setup
    assert setup.feature_tables == {
        "sep": tuple(f"s{j}" for j in range(6)),
        "noise": tuple(f"n{j}" for j in range(6)),
    }
    assert set(combined_report.discovery.ranking["feature"]) == {"f1", "f2", "f3"}
    text = " ".join(combined_report.summary().split())
    assert "sep: 6 feature columns, every observation matched once" in text
    # Said once: what the tables join, and not novelty (owner, 2026-10-06).
    assert text.count("not novelty or unusual observations") == 1
    assert "3 explainable features and 12 columns of 2 other tables" in text


def test_the_distance_analyses_read_their_spaces(combined_report):
    results = combined_report.results
    tables = [*(f"s{j}" for j in range(6)), *(f"n{j}" for j in range(6))]
    # Unusual observations read the explainable features alone (owner, 2026-10-06).
    unusual = results["unusual"].settings
    assert unusual["feature_cols"] == ["f1", "f2", "f3"] and not unusual["weights"]
    # The populations are searched in each space; the combined one weighs each table.
    populations = results["populations"].populations.settings
    assert list(populations["spaces"]) == ["explainable", "sep", "noise", "combined"]
    assert populations["named_by"] == ["f1", "f2", "f3"]
    assert set(results["integrity"].settings["feature_cols"]) >= set(tables)
    spaces = results["separation"].spaces.set_index("space")
    assert list(spaces.index) == ["Combined", "Explainable", "sep", "noise"]
    assert (
        spaces.loc["sep", "share"] > spaces.loc["Combined", "share"] > spaces.loc["noise", "share"]
    )


def _page_data(page) -> dict:
    text = page.read_text()
    start = text.index('id="embedding-data">') + len('id="embedding-data">')
    return json.loads(text[start : text.index("</script>", start)]), text


def test_the_page_draws_a_map_per_table_the_combined_one_first(combined_report, tmp_path):
    objects, fovs, sep, noise = _experiment()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        page = html_report(
            combined_report, objects, fovs, tmp_path, feature_tables={"sep": sep, "noise": noise}
        )
    data, text = _page_data(page)
    assert data["tables"] == ["Combined", "Explainable", "sep", "noise"]
    assert set(data["sets"]["raw"]) == set(data["tables"])
    combined = data["sets"]["raw"]["Combined"]["PCA"]["caption"]
    assert "each table divided by its first singular value" in combined
    # Each table's share of each axis, not the loadings of a table's columns (2026-10-06).
    assert "\nPC1: explainable " in combined and ", sep " in combined
    assert combined.splitlines()[-1].startswith("Combined: ")
    assert data["sets"]["raw"]["sep"]["UMAP"]["caption"].splitlines()[-1].startswith("sep: 9")
    # The explainable features colour every map, an other table's included.
    assert set(data["sets"]["raw"]["noise"]["PCA"]["z"]) == {"f1", "f2", "f3"}
    assert 'aria-label="Feature table"' in text
    # Beside raw and normalised, the other tables centred on each repeat's control
    # (owner, 2026-10-06): maps only, its one line said.
    # (No normalised view here: the explainable features hold negative values.)
    assert data["values"] == ["raw", "centred"]
    assert set(data["sets"]["centred"]) == {"sep", "noise"}
    assert "Centred on control: each column of an other feature table minus" in text
    assert (tmp_path / "figures" / "pca_centred_sep.png").exists()
    assert "How far the conditions and the repeats separate in each feature table" in text
    assert (tmp_path / "figures" / "pca_raw_sep.png").exists()
    with pytest.raises(ValueError, match="pass the tables themselves"):
        html_report(combined_report, objects, fovs, tmp_path)


def test_the_scan_lists_the_tables_before_loaded_correctly_and_names_them_in_the_setup():
    objects, fovs, sep, _ = _experiment()
    scan = scan_design(objects, fovs, feature_cols=["f1", "f2", "f3"], feature_tables={"sep": sep})
    text = scan.summary()
    assert text.index("Other feature table sep: 6 feature columns") < text.index(
        "Loaded correctly?"
    )
    assert set(scan.integrity.settings["feature_cols"]) >= {"s0", "s5"}
    setup = scan.setup("f1", control={"condition": "ctrl"})
    assert setup.feature_tables == {"sep": tuple(f"s{j}" for j in range(6))}
    refused = scan_design(objects, fovs, feature_tables={"sep": sep.iloc[3:]})
    (finding,) = [f for f in refused.findings if f.about == "feature_tables"]
    assert finding.status.marker == "❌" and "'sep' lacks a row for 3 observations" in finding.text
    assert "s0" not in refused.objects and refused.settings["feature_tables"] is None


def test_a_workflow_without_the_tables_attached_says_how_to_attach_them(combined_report):
    objects, fovs, sep, noise = _experiment()
    with pytest.raises(ValueError, match="attach them first"):
        optimise(objects, fovs, combined_report.setup, fov_counts=[2], reservoir_fraction=0.5)
    merged = attach_feature_tables(objects, fovs, {"sep": sep, "noise": noise}).objects
    checked, _, _ = check_workflow_inputs(merged, fovs, combined_report.setup)
    assert "n5" in checked
