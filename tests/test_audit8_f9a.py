"""The owner's decisions after audit 8 on the feature spaces (F9a, 2026-10-06): the separation
of the repeats on raw values, MFA weights free of the tables' order, the populations' cells
without their own control, unusual observations on the readout alone, the label check's
scale, the combined map, the wide fingerprint's k and the values the fingerprint reads."""

import re
import warnings

import numpy as np
import pandas as pd
import pytest
from test_audit8_b import SEVERAL, _several
from test_embedding_only import _cells
from test_feature_tables_batch import _batch, _setup

from morphocoverage.embedding import robust_scale
from morphocoverage.feature_tables import mfa_weights
from morphocoverage.multivariate_outcomes import CovarianceError
from morphocoverage.objectives import ObjectiveConfig, SamplingObjective
from morphocoverage.retrospective.criteria import OutcomeCriterion
from morphocoverage.schema import validate_input_tables
from morphocoverage.separation import separation
from morphocoverage.transforms import BlockWeights
from morphocoverage.workflows import Setup, default_novelty, html_report, optimise, report
from morphocoverage.workflows._contrasts import normalised_readers
from morphocoverage.workflows._enough_targets import fingerprint_lines, fingerprint_precision
from morphocoverage.workflows._feature_space import (
    analysed_tables,
    attached,
    centred_tables,
    novelty_space,
    separation_in_report,
    unusual_space,
)
from morphocoverage.workflows._fingerprint_space import fingerprint_space
from morphocoverage.workflows._html_map_text import comparison_table
from morphocoverage.workflows._html_maps import _maps, _tables
from morphocoverage.workflows._label_control import _robust
from morphocoverage.workflows._populations import populations_in_report, populations_text

KEY = ["condition", "biological_repeat", "image_file", "label_id"]


def _visible(page: str) -> str:
    text = re.sub(r"<script.*?</script>", "", page, flags=re.DOTALL)
    return " ".join(re.sub(r"<[^>]+>", " ", text).split())


# --- 2. The separation of the repeats on raw values; one well per condition ----------------


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


def test_one_well_per_condition_says_a_well_difference_reads_as_a_condition_separation():
    merged, fovs, names = _batch()
    spaces = {"DINO": names}
    unknown = separation(merged, fovs, spaces=spaces).well_line()
    assert unknown.startswith("The wells are not given, so a difference between the wells")
    one = fovs.assign(technical_repeat=fovs["condition"] + fovs["biological_repeat"])
    assert separation(merged, one, spaces=spaces).well_line() == (
        "Each condition is imaged in one well per biological repeat, so a difference between "
        "wells reads here as a separation of the conditions."
    )
    field = fovs["image_file"].str.extract(r"_(\d+)\.tif$")[0].astype(int)
    two = one.assign(technical_repeat=one["technical_repeat"] + (field % 2).astype(str))
    assert separation(merged, two, spaces=spaces).well_line() == ""


def test_the_page_says_the_repeats_are_scored_on_raw_values_and_the_one_well_line(tmp_path):
    objects, fovs, table = _cells()
    setup = Setup(control={"condition": "ctrl"}, observation_unit="cell", normalize="control_mean")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = report(objects, fovs, setup, feature_tables={"DINO": table})
        page = html_report(result, objects, fovs, tmp_path, feature_tables={"DINO": table})
    visible = _visible(page.read_text())
    assert "Within a condition, the same with the repeat, always on the raw values" in visible
    assert "The wells are not given, so a difference between the wells" in visible


# --- 3. MFA weights free of the tables' order; the novelty radius after weighting -----------


def _two_tables(seed: int = 0) -> tuple[np.ndarray, list[str]]:
    """4 correlated columns (A) and 2 independent ones (B), robustly standardised enough."""
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=(300, 1))
    first = latent + 0.3 * rng.normal(size=(300, 4))
    return np.hstack([first, rng.normal(size=(300, 2))]), ["a0", "a1", "a2", "a3", "b0", "b1"]


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


def test_the_novelty_radius_reads_the_square_root_of_the_weighted_inertia():
    z, names = _two_tables()
    blocks = (tuple(names[:4]), tuple(names[4:]))
    state, _ = BlockWeights(blocks, inertia_radius=True).fit_matrix(z, tuple(names))
    plain, _ = BlockWeights(blocks).fit_matrix(z, tuple(names))
    scaled = np.asarray(state["weights"])
    # The declared radius sqrt(d) in the scaled space is sqrt(total inertia) in the MFA one.
    assert (np.var(z * scaled, axis=0, ddof=1)).sum() == pytest.approx(len(names))
    inertia = (np.var(z * np.asarray(plain["weights"]), axis=0, ddof=1)).sum()
    np.testing.assert_allclose(scaled, np.asarray(plain["weights"]) * np.sqrt(6 / inertia))


def test_novelty_with_two_explainable_tables_does_not_depend_on_their_order():
    z, names = _two_tables(1)
    rows = pd.DataFrame(z, columns=names)
    rows["image_file"] = [f"f{i // 20}" for i in range(len(rows))]
    rows["label_id"] = np.arange(len(rows)) % 20 + 1
    rows["condition"], rows["biological_repeat"] = "c", "R1"
    fovs = rows[["condition", "biological_repeat", "image_file"]].drop_duplicates()
    found = []
    for tables in ({"TA": names[:4], "TB": names[4:]}, {"TB": names[4:], "TA": names[:4]}):
        setup = Setup(
            readout="a0",
            features=tuple(names),
            novelty=default_novelty(names),
            explainable_tables=tables,
        )
        estimator, columns = novelty_space(setup)
        data = validate_input_tables(rows, fovs=fovs, feature_cols=columns)
        evidence = estimator.estimate(data, objective=ObjectiveConfig(SamplingObjective.POPULATION))
        found.append(evidence.quantities["lofo_novel_fraction"].value)
    assert found[0] == pytest.approx(found[1])


# --- 4. The populations name the cells left out for want of their own control ---------------


def test_the_populations_name_the_cells_without_their_own_control_never_a_missing_value():
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
        found = populations_in_report(prepared, fovs, setup, merged)
    settings = found.populations.settings
    # b1 in R1, where cB was not imaged: 3 fields of 6 cells (audit 8, seed 4).
    assert [(b["reason"], b["observations"]) for b in settings["left_out_by"]] == [
        ("no own control", 18)
    ]
    assert settings["incomplete_observations"] == 5
    text = " ".join(populations_text(found, setup)[2].split())
    assert "(18 cells of b1 in R1 left out: no own control)" in text


# --- 5. A readout without features beside other tables: unusual observations on it alone ----


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


# --- 6. The label check standardises as the other explainable spaces ------------------------


def test_the_label_check_reads_the_robust_scale_and_mfa_of_the_other_explainable_spaces():
    rng = np.random.default_rng(3)
    spread = rng.normal(0, 5, 400)
    spread[:190] = 0.0  # a mostly-zero column: its MAD is far under a quarter of its SD
    matrix = np.column_stack([rng.normal(size=400), spread, rng.normal(size=400)])
    one = Setup(readout="a", features=("a", "b", "c"))
    centre, scale = robust_scale(matrix)
    np.testing.assert_allclose(_robust(matrix, one), (matrix - centre) / scale)
    assert scale[1] == pytest.approx(matrix[:, 1].std(ddof=1))  # the MAD floor (seed 14)
    two = Setup(readout="a", features=("a", "b", "c"), explainable_tables={"T": ("c",)})
    z = (matrix - centre) / scale
    weights = mfa_weights(z, ["a", "b", "c"], {"observation table": ("a", "b"), "T": ("c",)})
    np.testing.assert_allclose(_robust(matrix, two), z * weights)


# --- 7. The populations' combined space gets its map ----------------------------------------


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
    tables, _ = _tables(setup, merged, fovs)
    read = analysed_tables(setup.prepared(merged, fovs), fovs, setup)
    pd.testing.assert_frame_equal(tables["normalised"], read)
    assert not np.allclose(tables["normalised"]["e0"], merged["e0"])  # centred, not raw


def test_an_embedding_only_combined_space_has_its_centred_map():
    objects, fovs, table = _cells()
    halves = {"D1": table[[*KEY, *(f"e{j}" for j in range(6))]]}
    halves["D2"] = table[[*KEY, *(f"e{j}" for j in range(6, 12))]]
    setup = Setup(control={"condition": "ctrl"}, normalize="control_mean", observation_unit="cell")
    merged, setup, _ = attached(objects, fovs, setup, halves)
    tables, _ = _tables(setup, merged, fovs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        maps = _maps(setup, tables, fovs, (), [])
    assert set(maps["centred"]) == {"Combined", "D1", "D2"}


# --- 8. One common k for a wide fingerprint ---------------------------------------------------

COLUMNS = tuple(f"d{j}" for j in range(30))


def _fingerprint_data(pilot: bool, fields: int = 6, cells: int = 25):
    """ctrl and a in 3 repeats, ``fields`` fields of ``cells`` cells, a table of 30 columns
    on 3 latent axes; with ``pilot``, a condition of 4 fields of 3 cells in R1."""
    rng = np.random.default_rng(0)
    load = rng.normal(size=(3, len(COLUMNS)))
    groups = [(c, r, fields, cells) for c in ("ctrl", "a") for r in ("R1", "R2", "R3")]
    groups += [("pilot", "R1", 4, 3)] if pilot else []
    frames = []
    for c, r, n_fields, n_cells in groups:
        for field in range(n_fields):
            values = rng.normal(size=(n_cells, 3)) @ load
            values += 0.3 * rng.normal(size=(n_cells, len(COLUMNS)))
            frame = pd.DataFrame(values, columns=list(COLUMNS))
            frame.insert(0, "label_id", np.arange(1, n_cells + 1))
            frame.insert(0, "image_file", f"{c}{r}{field}")
            frame.insert(0, "biological_repeat", r)
            frame.insert(0, "condition", c)
            frames.append(frame)
    objects = pd.concat(frames, ignore_index=True)
    return objects, objects[KEY[:3]].drop_duplicates().reset_index(drop=True)


def _fingerprint_setup() -> Setup:
    criterion = OutcomeCriterion(
        CovarianceError(COLUMNS, True), 0.15, "lower_is_better", "DINO covariance within 15%"
    )
    return Setup(
        fingerprint=(criterion,), feature_tables={"DINO": COLUMNS}, observation_unit="cell"
    )


def test_a_small_pilot_group_is_not_assessable_and_named_never_lowering_k():
    setup = _fingerprint_setup()
    alone = fingerprint_precision(*_fingerprint_data(False), setup)
    with_pilot = fingerprint_precision(*_fingerprint_data(True), setup)
    # k = a fifth of the median group's 150 cells, capped at the 30 columns (audit 8 A6).
    assert set(alone["components"]) == set(with_pilot["components"]) == {30}
    pilot = with_pilot[with_pilot["condition"] == "pilot"].iloc[0]
    assert pilot["small"] and pilot["met"] is None
    others = with_pilot[with_pilot["condition"] != "pilot"].reset_index(drop=True)
    assert others["met"].tolist() == alone["met"].tolist()
    text = " ".join(" ".join(fingerprint_lines(with_pilot, "group", "fields", unit="cell")).split())
    assert "not assessable in pilot R1, holding fewer than 5 k, 150 cells)" in text
    assert "Not judged in 1 group: too few observations for the common k." in text


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


# --- 9. The fingerprint and optimise read normalised values whenever normalize is declared --


def test_the_fingerprint_reads_normalised_values_with_several_controls_too():
    objects, fovs, _ = _several()
    for control in (SEVERAL, {"condition": "cA"}):
        setup = Setup(readout="area", control=control, normalize="control_mean")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            values = setup.compared_values(objects, fovs)
            expected = setup.prepared(objects, fovs)
        pd.testing.assert_frame_equal(values, expected)
        assert not values["area"].equals(objects["area"])
    plain = Setup(readout="area", control=SEVERAL)
    assert plain.compared_values(objects, fovs) is objects
    several = normalised_readers(several=True)
    assert "the fingerprint and the outcomes judged on held-back fields read the normalised" in (
        several
    )


def test_with_several_controls_the_fingerprint_names_the_cells_without_their_own_control():
    objects, fovs, tables = _several()
    criterion = OutcomeCriterion(
        CovarianceError(("area", "fib_len"), True), 0.5, "lower_is_better", "shape"
    )
    setup = Setup(
        readout="area",
        features=("area", "fib_len", "fib_w"),
        control=SEVERAL,
        normalize="control_mean",
        fingerprint=(criterion,),
        observation_unit="cell",
    )
    merged, setup, _ = attached(objects, fovs, setup, {"fiber": tables["fiber"]})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rows = fingerprint_precision(merged, fovs, setup)
    assert int(rows["left_out"].sum()) == 0 and int(rows["no_own_control"].sum()) == 18
    text = " ".join(" ".join(fingerprint_lines(rows, "group", "fields", unit="cell")).split())
    assert "18 cells of b1 in R1 left out: no own control." in text
    assert "missing value" not in text


def test_optimise_does_not_judge_a_group_too_small_for_the_common_k_and_names_it():
    objects, fovs = _fingerprint_data(True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plan = optimise(
            objects, fovs, _fingerprint_setup(), fov_counts=(2,), reservoir_fraction=0.35
        )
    curve = plan.results["reliability"].curve
    pilot = curve[curve["condition"] == "pilot"]
    assert len(pilot) and (pilot["defined_draws"] == 0).all() and pilot["preserved"].isna().all()
    others = curve[curve["condition"] != "pilot"]
    assert (others["defined_draws"] > 0).all()
    assert (
        "The fingerprint, DINO covariance within 15%, is not assessable in pilot R1: fewer than "
        "5 k, 150 cells." in plan.document.notes
    )


def test_under_a_centred_map_the_repeat_line_says_it_reads_the_raw_values():
    merged, fovs, names = _batch()
    setup = _setup(names)
    centred = centred_tables(merged, fovs, setup)
    found = separation_in_report(centred, fovs, setup, ["DINO"], raw=merged)
    line = found.line("DINO")
    assert "Within a condition, on the raw values, " in line
    assert line.endswith("On the raw values, the repeats separate more than the conditions.")
    plain = separation_in_report(merged, fovs, setup, ["DINO"]).line("DINO")
    assert "on the raw values" not in plain and "Here the repeats separate more" in plain
    assert "On the raw values, the repeats separate more than the conditions in DINO" in (
        comparison_table(found)
    )


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
