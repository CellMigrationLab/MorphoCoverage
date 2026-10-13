"""Other feature tables (owner, 2026-10-05): validated against the observation table, attached
as named blocks, weighted in the combined space (each table by its first singular value, owner
2026-10-06); every failure has its message."""

from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
import pytest
from _helpers_embedding import KEY, SEVERAL, _cells, _several
from _helpers_feature_tables import (
    _batch,
    _fingerprint_data,
    _fingerprint_setup,
    _setup,
    _table,
    _tables,
    _two_tables,
    _without,
)
from _helpers_feature_tables import _cells as _table_cells
from _helpers_tables import _four_cell_tables

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
from morphocoverage.workflows._feature_space import (
    analysed_tables,
    attached,
    centred_tables,
    centring_line,
    features_words,
    novelty_space,
    separation_in_report,
    space,
    unusual_space,
)
from morphocoverage.workflows._fingerprint_space import fingerprint_space
from morphocoverage.workflows._html_map_text import comparison_table
from morphocoverage.workflows._html_maps import _tables as _map_tables
from morphocoverage.workflows._shared import check_workflow_inputs
from morphocoverage.workflows.setup import default_novelty


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
        (lambda e: e.drop(columns="label_id"), "lacks the identity columns ['label_id']"),
        (lambda e: e.assign(z=0), "holds metadata columns ['z']"),
        (lambda e: e[["condition", "biological_repeat", "image_file", "label_id"]], "no feature"),
    ],
)
def test_each_problem_of_a_table_is_named(change, expected):
    objects, fovs, embedding = _tables()
    message = _message(objects, fovs, {"DINO": change(embedding)})
    assert expected in message and "'DINO'" in message


def test_two_tables_sharing_columns_load_with_their_names_before_them():
    # Owner, 2026-10-06: a name found in two tables is prefixed, no longer refused.
    objects, fovs, embedding = _tables()
    both = attach_feature_tables(objects, fovs, {"DINOv2": embedding, "DINOv3": embedding})
    assert both.columns["DINOv3"] == ("DINOv3: e0", "DINOv3: e1", "DINOv3: e2")


@pytest.mark.parametrize(
    ("names", "expected"),
    [
        (("",), "non-empty text"),
        (
            ("combined",),
            "names the explainable features, the observation table or the combined space",
        ),
        (
            ("Explainable",),
            "names the explainable features, the observation table or the combined space",
        ),
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
    # MFA (owner, 2026-10-06): every table's first axis ends with variance 1, the explainable
    # features' too (audit 8: before, the first table kept weight 1 and set the others' scale).
    rng = np.random.default_rng(0)
    explainable = rng.normal(size=(200, 4))
    shared = rng.normal(size=(200, 1))
    correlated = shared + 0.1 * rng.normal(size=(200, 16))
    small = 3.0 * rng.normal(size=(200, 2))
    z = np.hstack([explainable, correlated, small])
    names = [*(f"x{i}" for i in range(4)), *(f"c{i}" for i in range(16)), "s0", "s1"]
    blocks = {"explainable": names[:4], "corr": names[4:20], "small": names[20:]}
    weights = mfa_weights(z, names, blocks)

    def first(columns: slice) -> float:
        part = (z * weights)[:, columns]
        return float(np.linalg.svd(part - part.mean(axis=0), compute_uv=False)[0])

    for columns in (slice(0, 4), slice(4, 20), slice(20, 22)):
        assert first(columns) == pytest.approx(np.sqrt(len(z) - 1))
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
    # Beside raw and normalised, the other tables centred per repeat (owner, 2026-10-06):
    # on the median of the condition medians of those in every repeat, said in one line
    # (audit 8, owner 2026-10-06).
    # (No normalised view here: the explainable features hold negative values.)
    assert data["values"] == ["raw", "centred"]
    assert set(data["sets"]["centred"]) == {"sep", "noise"}
    assert "Centred: each column of an other feature table minus a median" in text
    assert "centred per biological repeat on the median of the per-condition medians" in text
    # In the interactive maps only: no static figure or CSV of its own.
    assert not (tmp_path / "figures" / "pca_centred_sep.png").exists()
    assert "How far the conditions and the repeats separate in each feature table" in text
    assert (tmp_path / "figures" / "pca_raw_sep.png").exists()
    with pytest.raises(ValueError, match="pass the tables themselves"):
        html_report(combined_report, objects, fovs, tmp_path)


def test_the_scan_lists_the_tables_before_loaded_correctly_and_names_them_in_the_setup():
    objects, fovs, sep, _ = _experiment()
    scan = scan_design(objects, fovs, feature_cols=["f1", "f2", "f3"], feature_tables={"sep": sep})
    text = scan.summary()
    assert text.index("Other feature table sep (non-explainable): 6 feature columns") < text.index(
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


def test_the_repeats_separation_reads_the_raw_values_and_the_conditions_the_centred_ones():
    merged, fovs, names = _batch()
    setup = _setup(names)
    centred = centred_tables(merged, fovs, setup)
    raw = separation_in_report(merged, fovs, setup, ["DINO"]).spaces.iloc[0]
    alone = separation_in_report(centred, fovs, setup, ["DINO"]).spaces.iloc[0]
    both = separation_in_report(centred, fovs, setup, ["DINO"], raw=merged).spaces.iloc[0]
    # A batch measured after the centring that removes it would be circular (audit 8 A2).
    assert both["repeat_share"] == raw["repeat_share"]
    assert both["repeat_share"] > alone["repeat_share"] + 0.3
    assert both["share"] == alone["share"]


def test_the_mfa_weights_do_not_depend_on_the_order_of_the_tables():
    z, names = _two_tables()
    first = mfa_weights(z, names, {"A": names[:4], "B": names[4:]})
    second = mfa_weights(z, names, {"B": names[4:], "A": names[:4]})
    np.testing.assert_allclose(first, second)
    # Each table's first axis has variance 1 (audit 8 A1).
    for columns in (slice(0, 4), slice(4, 6)):
        part = (z * first)[:, columns]
        top = np.linalg.svd(part - part.mean(axis=0), compute_uv=False)[0]
        assert top**2 / (len(z) - 1) == pytest.approx(1.0)
    # One table: nothing weighted, as before.
    np.testing.assert_allclose(mfa_weights(z, names, {"A": names}), np.ones(6))


def test_a_readout_without_features_beside_a_table_reads_unusual_observations_on_it_alone():
    objects, fovs, table = _cells()
    objects["area"] = np.random.default_rng(1).normal(10, 1, len(objects))
    setup = Setup(readout="area", control={"condition": "ctrl"}, observation_unit="cell")
    merged, setup, _ = attached(objects, fovs, setup, {"DINO": table})
    columns, weights, words = unusual_space(merged, fovs, setup)
    assert (columns, weights, words) == (["area"], None, "the readout area alone")
    line = next(line for line in setup.declaration() if line.startswith("Representation"))
    assert line == (
        "Representation: readout area; no feature columns declared, so new phenotypes are not "
        "assessed; unusual cells on the readout alone, the other tables never entering them."
    )


def test_the_normalised_combined_map_reads_the_centred_tables_as_the_populations_do():
    objects, fovs, table = _cells()
    rng = np.random.default_rng(2)
    objects["area"] = rng.lognormal(3, 0.2, len(objects))
    objects["shape"] = rng.lognormal(0, 0.2, len(objects))
    setup = Setup(
        readout="area",
        features=("area", "shape"),
        control={"condition": "ctrl"},
        normalize="control_mean",
        observation_unit="cell",
    )
    merged, setup, _ = attached(objects, fovs, setup, {"DINO": table})
    tables, _ = _map_tables(setup, merged, fovs)
    read = analysed_tables(setup.prepared(merged, fovs), fovs, setup)
    pd.testing.assert_frame_equal(tables["normalised"], read)
    assert not np.allclose(tables["normalised"]["e0"], merged["e0"])  # centred, not raw


def test_without_a_group_of_four_fields_a_tables_fingerprint_is_not_assessable_anywhere():
    objects, fovs = _fingerprint_data(False, fields=3)
    setup = _fingerprint_setup()
    space = fingerprint_space(objects, fovs, setup)
    assert not space.setup.fingerprint
    reason = "no group holds 4 fields, so no group sets the number of its principal components"
    assert list(space.not_assessable.values()) == [reason]
    # optimise never judges the full width (seed 9): here it has nothing to judge.
    with pytest.raises(ValueError, match="no group holds 4 fields"):
        optimise(objects, fovs, setup, fov_counts=(1, 2), reservoir_fraction=0.35)


def test_under_a_centred_map_the_repeat_line_says_it_reads_the_raw_values():
    merged, fovs, names = _batch()
    setup = _setup(names)
    centred = centred_tables(merged, fovs, setup)
    found = separation_in_report(centred, fovs, setup, ["DINO"], raw=merged)
    line = found.line("DINO")
    assert "Within a condition, on the raw values, " in line
    # Raw repeats against centred conditions are not compared (audit 10, B15).
    assert "separate more" not in line
    plain = separation_in_report(merged, fovs, setup, ["DINO"]).line("DINO")
    assert "on the raw values" not in plain and "Here the repeats separate more" in plain
    assert "separate more than the conditions" not in comparison_table(found)


def test_the_normalised_map_line_names_the_cells_without_their_own_control():
    objects, fovs, tables = _several()
    objects.loc[objects.index[:5], "area"] = np.nan  # a true missing value, counted as one
    setup = Setup(
        readout="area",
        features=("area", "fib_len", "fib_w"),
        control=SEVERAL,
        normalize="control_mean",
        observation_unit="cell",
    )
    merged, setup, _ = attached(objects, fovs, setup, {"fiber": tables["fiber"]})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prepared = setup.prepared(merged, fovs)
    line = separation_in_report(prepared, fovs, setup, raw=merged).line("Explainable")
    assert (
        "5 cells with a missing value left out. 18 cells of b1 in R1 left out: no own control."
        in line
    )


def test_the_centred_view_subtracts_the_median_of_the_condition_medians_in_every_repeat():
    objects, fovs, rng = _table_cells(fields=2, cells=3)
    # b not imaged in R1: ctrl and a, in every repeat, set the centre (owner, 2026-10-06,
    # audit 8; test_feature_tables_batch).
    objects, fovs = _without(objects, "b", "R1"), _without(fovs, "b", "R1")
    values = rng.normal(size=(len(objects), 2))
    table = _table(objects, values, "d")
    merged = attach_feature_tables(objects, fovs, {"DINO": table}).objects
    merged["x0"] = 1.0
    setup = Setup(
        readout="x0", control={"condition": "ctrl"}, feature_tables={"DINO": ("d0", "d1")}
    )
    centred = centred_tables(merged, fovs, setup)
    for _, rows in merged.groupby("biological_repeat"):
        medians = [rows.loc[rows["condition"] == c, "d0"].median() for c in ("ctrl", "a")]
        np.testing.assert_allclose(centred.loc[rows.index, "d0"], rows["d0"] - np.median(medians))
    assert (centred["x0"] == 1.0).all()  # the explainable features are left as they are
    assert centred_tables(merged, fovs, Setup(readout="x0")) is None


def test_a_column_without_a_value_is_refused_and_a_constant_one_listed():
    objects, fovs, embedding = _tables()
    with pytest.raises(ValueError, match="columns without a single value: e1"):
        attach_feature_tables(objects, fovs, {"DINO": embedding.assign(e1=np.nan)})
    tables = attach_feature_tables(objects, fovs, {"DINO": embedding.assign(e2=3.0)})
    assert tables.constant == {"DINO": ("e2",)}
    assert "1 column with one value everywhere (e2), which no analysis can use" in (
        tables.summary()
    )


@pytest.mark.parametrize("which", ["table", "objects"])
def test_a_label_written_as_a_decimal_matches_its_whole_number(which):
    # Audit B, finding 11: a CSV column with one empty cell is read as 1.0, 2.0, ...
    objects, fovs, embedding = _tables()
    if which == "table":
        embedding["label_id"] = embedding["label_id"].astype(float)
    else:
        objects["label_id"] = objects["label_id"].astype(float)
    assert "e0" in attach_feature_tables(objects, fovs, {"DINO": embedding}).objects


def test_the_columns_are_said_as_explainable_features_and_other_tables():
    setup = Setup(
        readout="a",
        features=("a", "b"),
        feature_tables={"DINOv2": ("d0", "d1", "d2"), "DINOv3": ("v0", "v1")},
    )
    assert features_words(setup) == "2 explainable features and 5 columns of 2 other tables"
    one = Setup(readout="a", features=("a", "b"), feature_tables={"DINOv2": ("d0",)})
    assert features_words(one) == "2 explainable features and 1 column of the other table DINOv2"


def test_attach_feature_tables_refusals():
    objects, fovs = _four_cell_tables()
    with pytest.raises(TypeError) as error:
        attach_feature_tables([1], fovs, {"t": objects})
    assert str(error.value) == "objects must be a pandas DataFrame"
    with pytest.raises(TypeError) as error:
        attach_feature_tables(objects, fovs, {})
    assert str(error.value) == (
        "feature_tables must map each table's name to its DataFrame, or to (DataFrame, "
        "'explainable' or 'non-explainable')"
    )
    with pytest.raises(ValueError) as error:
        attach_feature_tables(objects, fovs, {"t": [1]})
    assert str(error.value) == (
        "the other feature tables cannot be used: the table 't' must be a pandas DataFrame"
    )
