"""The owner's decisions of 2026-10-06 on wide feature tables: the fingerprint of a table on
its first principal components (k at most a fifth of the median group, audit 8; not assessable
below 2), and the centred map view (audit 8: no repeat left out for a small control)."""

import numpy as np
import pandas as pd
from _helpers_feature_tables import _cells, _table, _wide, _without

from morphocoverage.feature_tables import attach_feature_tables
from morphocoverage.multivariate_outcomes import ComponentsError, CovarianceError
from morphocoverage.retrospective.criteria import OutcomeCriterion
from morphocoverage.schema import validate_input_tables
from morphocoverage.workflows import Setup
from morphocoverage.workflows._enough_targets import (
    fingerprint_lines,
    fingerprint_precision,
    targets_status,
)
from morphocoverage.workflows._feature_space import centred_tables, centring_line
from morphocoverage.workflows._fingerprint_space import fingerprint_space
from morphocoverage.workflows._html_map_text import centred_note


def _setup(names: tuple[str, ...], explainable=("area", "shape")) -> Setup:
    words = "the covariance within 50%"
    fingerprint = (
        OutcomeCriterion(CovarianceError(explainable, True), 0.5, "lower_is_better", words),
        OutcomeCriterion(CovarianceError(names, True), 0.5, "lower_is_better", f"DINO {words}"),
    )
    return Setup(
        readout="area",
        features=("area", "shape"),
        fingerprint=fingerprint,
        feature_tables={"DINO": names},
        observation_unit="cell",
    )


def test_a_wide_tables_fingerprint_reads_its_first_components_and_says_k():
    # 4 fields of 3 cells: groups of 12 cells, so k = 12 // 5 = 2 of the 60 columns.
    merged, fovs, names = _wide(fields=4, cells=3)
    precision = fingerprint_precision(merged, fovs, _setup(names))
    dino = precision[precision["table"] == "DINO"]
    assert set(dino["components"]) == {2}
    assert dino["met"].notna().all() and (dino["error"] < 5).all()
    explainable = precision[precision["table"] == "the explainable features"]
    assert explainable["components"].isna().all()  # 2 columns: today's form
    text = " ".join(" ".join(fingerprint_lines(precision, "well", "fields", unit="cell")).split())
    assert (
        "The covariance fingerprint of DINO (DINO the covariance within 50%), read on its first 2 "
        "principal components (k at most a fifth of the median group's 12 cells and at "
        "most its 60 columns)"
    ) in text


def test_the_raw_covariance_of_a_wide_table_on_few_cells_is_the_absurd_number_avoided():
    # The former form on 60 columns and 12 cells a group: the error reads noise.
    merged, fovs, names = _wide(fields=4, cells=3)
    setup = _setup(names)
    raw = Setup(**{**setup.__dict__, "fingerprint": setup.fingerprint[1:]})
    data = validate_input_tables(merged, fovs=fovs, feature_cols=list(names), **raw.common)
    space = fingerprint_space(merged, fovs, raw)
    read = space.setup.fingerprint[0].metric
    assert isinstance(read, ComponentsError) and len(read.loadings) == 2
    first = data.objects.index[: len(data.objects) // 2]
    half = validate_input_tables(
        merged.loc[first], fovs=fovs, feature_cols=list(names), **raw.common
    )
    assert np.isfinite(read.evaluate(data, half))


def test_groups_too_small_for_two_components_are_not_assessable_and_say_why():
    merged, fovs, names = _wide(fields=4, cells=2)  # groups of 8 cells: k = 1
    setup = _setup(names)
    precision = fingerprint_precision(merged, fovs, setup)
    dino = precision[precision["table"] == "DINO"]
    assert dino["met"].isna().all() and dino["not_assessable"].notna().all()
    text = " ".join(" ".join(fingerprint_lines(precision, "well", "fields", unit="cell")).split())
    assert (
        "The covariance fingerprint of DINO (DINO the covariance within 50%): not assessable: its "
        "median group holds 8 cells, and 2 principal components need groups of at least 10"
    ) in text
    _statuses, parts = targets_status(None, precision, setup)
    assert "the covariance fingerprint of DINO not assessable" in parts


def test_wide_explainable_features_follow_the_same_rule():
    merged, fovs, names = _wide(fields=4, cells=3)
    for j in range(4):
        merged[f"x{j}"] = merged[f"d{j}"] + merged["area"]
    explainable = ("area", "shape", "x0", "x1", "x2", "x3")
    setup = Setup(
        readout="area",
        features=explainable,
        fingerprint=(
            OutcomeCriterion(CovarianceError(explainable, True), 0.5, "lower_is_better", "f"),
        ),
        feature_tables={"DINO": names},
        observation_unit="cell",
    )
    precision = fingerprint_precision(merged, fovs, setup)
    assert set(precision["components"]) == {2}  # 6 columns, more than 12 // 5


def test_components_error_is_its_metric_on_the_projected_values():
    rng = np.random.default_rng(3)
    objects = pd.DataFrame(rng.normal(size=(40, 3)), columns=["a", "b", "e"])
    objects["position_id"] = np.repeat(np.arange(8), 5)
    data = validate_input_tables(objects, feature_cols=["a", "b", "e"], fov_col="position_id")
    half = validate_input_tables(
        objects.iloc[:20], feature_cols=["a", "b", "e"], fov_col="position_id"
    )
    loadings = ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    metric = ComponentsError(
        CovarianceError(("p1", "p2"), False), ("a", "b", "e"), (0, 0, 0), (1, 1, 2), loadings
    )
    projected = objects.assign(p1=objects["a"], p2=objects["e"] / 2)
    inner = CovarianceError(("p1", "p2"), False)
    full = validate_input_tables(projected, feature_cols=["p1", "p2"], fov_col="position_id")
    part = validate_input_tables(
        projected.iloc[:20], feature_cols=["p1", "p2"], fov_col="position_id"
    )
    assert np.isclose(metric.evaluate(data, half), inner.evaluate(full, part))
    assert metric.parameters["components"] == 2 and metric.parameters["features"] == list("abe")


def test_a_repeat_whose_control_is_small_is_centred_like_the_others():
    # Audit 8 (owner, 2026-10-06): the centre is the median of the condition medians of the
    # conditions in every repeat, so a small control no longer leaves its repeat out.
    objects, fovs, rng = _cells(fields=2, cells=3)
    objects, fovs = _without(objects, "b", "R1"), _without(fovs, "b", "R1")
    small = (objects["condition"] == "ctrl") & (objects["biological_repeat"] == "R2")
    objects = objects[~(small & (objects["label_id"] > 2))].reset_index(drop=True)  # 4 cells
    table = _table(objects, rng.normal(size=(len(objects), 2)), "d")
    merged = attach_feature_tables(objects, fovs, {"DINO": table}).objects
    merged["x0"] = 1.0
    setup = Setup(
        readout="x0",
        control={"condition": "ctrl"},
        feature_tables={"DINO": ("d0", "d1")},
        observation_unit="cell",
    )
    assert centred_tables(merged, fovs, setup)["d0"].notna().all()
    line = centring_line(merged, fovs, setup)
    note = centred_note(line)
    assert note.endswith(
        "Batch: the other tables centred per biological repeat on the median of the "
        "per-condition medians of ctrl and a, the conditions in every repeat; the centred map "
        "reads them, never a test."
    )
