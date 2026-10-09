"""The code the page writes matches typed field identities, never their text (issue 338)."""

from __future__ import annotations

import html
import re
from types import SimpleNamespace

import pandas as pd
import pytest

from morphocoverage.workflows._html_next import fields_next, fields_value


def _code(block: str) -> str:
    return html.unescape(re.search(r"<code>(.*?)</code>", block, re.DOTALL).group(1))


def _tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    fovs = pd.DataFrame(
        {
            "condition": ["a", "a", "a", "a"],
            "position_id": pd.Series([1, "1", 2, "x,y=z"], dtype=object),
        }
    )
    objects = fovs.loc[fovs.index.repeat(2)].reset_index(drop=True).assign(v=1.0)
    return objects, fovs


def _run(fields) -> tuple[list[bool], list[bool]]:
    objects, fovs = _tables()
    code = _code(fields_next(fields, ["condition", "position_id"]))
    uncommented = re.sub(r"^    # (\(.*\)),$", r"    \1,", code, flags=re.MULTILINE)
    before_report = uncommented.split("result = report(")[0]
    namespace = {"objects": objects, "fovs": fovs, "result": SimpleNamespace(setup=None)}
    exec(before_report, namespace)  # noqa: S102
    return list(namespace["keep"]), list(namespace["inside"])


def test_an_integer_and_a_text_identity_that_print_alike_are_two_fields():
    keep, inside = _run([("a", 1)])  # field 1 (an integer), not "1"
    assert keep == [False, True, True, True]
    assert inside == [True, True, False, False, False, False, False, False]
    keep, _ = _run([("a", "1")])
    assert keep == [True, False, True, True]


def test_values_with_separators_are_matched_exactly():
    keep, _ = _run([("a", "x,y=z")])
    assert keep == [True, True, True, False]


def test_the_code_lists_typed_literals_and_refuses_types_it_cannot_write():
    block = fields_next([("a", 1), ("a", "1")], ["condition", "position_id"])
    code = _code(block)
    assert "# ('a', 1)," in code and "# ('a', '1')," in code
    assert fields_value(1.5) == 1.5 and fields_value(True) is True
    with pytest.raises(TypeError, match="cannot be written as code"):
        fields_value(pd.Timestamp("2026-01-01"))


def test_integrity_findings_carry_the_typed_key_of_each_field_they_name():
    import numpy as np

    from morphocoverage import integrity_checks

    rng = np.random.default_rng(0)
    block = rng.normal(size=(12, 4))
    rows = []
    for fov in (1, "1", 2, 3):
        values = block if fov in (1, "1") else rng.normal(size=(12, 4))
        for row in values:
            rows.append(
                {
                    "condition": "a",
                    "position_id": fov,
                    **dict(zip(["a1", "a2", "a3", "a4"], row, strict=True)),
                }
            )
    objects = pd.DataFrame(rows)
    result = integrity_checks(
        objects,
        objects[["condition", "position_id"]].drop_duplicates(),
        feature_cols=["a1", "a2", "a3", "a4"],
        observation_unit="cell",
        fov_col="position_id",
    )
    keys = result.settings["field_keys"]
    assert len(result.findings) >= 1
    assert {("a", 1), ("a", "1")} <= set(keys.values()) or set(keys.values()) <= {
        ("a", 1),
        ("a", "1"),
    }
    named = {*result.findings["field"], *result.findings["other_field"]}
    assert named <= set(keys)
