"""Wording found by running the package on a real fibre-feature acquisition (audit 11)."""

from __future__ import annotations

import pandas as pd

from morphocoverage.workflows import Setup, scan_design


def _tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for cond, repeat, n in (
        ("ctrl", "R1", 6),
        ("ctrl", "R2", 1),
        ("drug", "R1", 6),
        ("drug", "R2", 6),
    ):
        for i in range(n):
            rows.append(
                {
                    "position_id": f"{cond}{repeat}{i}",
                    "condition": cond,
                    "biological_repeat": repeat,
                    "area": 1.0 + i / 10,
                }
            )
    fovs = pd.DataFrame(rows)[["position_id", "condition", "biological_repeat"]]
    return pd.DataFrame(rows), fovs


def test_the_normalised_sentence_names_the_declared_unit():
    _, fovs = _tables()
    declared = Setup(
        readout="area",
        control={"condition": "ctrl"},
        normalize="control_mean",
        observation_unit="image",
    )
    assert "control ctrl images of the same biological repeat" in declared.normalized_sentence(fovs)
    undeclared = Setup(readout="area", control={"condition": "ctrl"}, normalize="control_mean")
    assert "control ctrl objects of the same biological repeat" in undeclared.normalized_sentence(
        fovs
    )


def test_the_scan_says_one_image_not_one_images():
    objects, fovs = _tables()
    scan = scan_design(
        objects, fovs, feature_cols=["area"], observation_unit="image", fov_col="position_id"
    )
    text = " ".join(scan.summary().split())
    assert "(1 image)" in text and "1 images" not in text


def test_the_maps_have_raw_and_normalised_buttons_and_a_method_menu(tmp_path):
    import warnings

    from test_audit8_b import _embedding_only

    from morphocoverage.workflows import html_report

    objects, fovs, table, result = _embedding_only(normalize="control_mean")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        page = html_report(result, objects, fovs, tmp_path, feature_tables={"DINO": table})
    text = page.read_text()
    assert 'data-value="@corrected"' in text and 'data-value="raw"' in text
    assert "Divided by the control mean" in text or "Minus the repeat median" in text
    assert 'aria-label="Values"' in text
