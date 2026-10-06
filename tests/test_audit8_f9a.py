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

from morphocoverage.feature_tables import mfa_weights
from morphocoverage.objectives import ObjectiveConfig, SamplingObjective
from morphocoverage.schema import validate_input_tables
from morphocoverage.separation import separation
from morphocoverage.transforms import BlockWeights
from morphocoverage.workflows import Setup, default_novelty, html_report, report
from morphocoverage.workflows._feature_space import (
    attached,
    centred_tables,
    novelty_space,
    separation_in_report,
)
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
    assert settings["no_own_control"]["observations"] == 18
    assert settings["incomplete_observations"] == 5
    text = " ".join(populations_text(found, setup)[2].split())
    assert "(18 cells of b1 in R1 left out: no own control)" in text
