"""The golden MorphoFeature exports of the Spec fixtures, read as MorphoCoverage reads them.

``tests/morphofeature_exports/<fixture>/`` holds the ``coverage/Cells`` folder MorphoFeature
wrote for one vendored Spec fixture (``tests/spec_fixtures``), unchanged, beside its
``provenance.json`` (the MorphoFeature commit and the fixture's git blob hash; how they were
made: ``README.md`` there). The only feature MorphoFeature's test provider writes,
``Cells__area``, is the label value (1 or 2) in every field, so the tests of a pairing add a
feature of their own, ``x``, built from the **description** (the vendored TOML), never from the
export's ``repeat_group``: each repeat shifts both conditions by its batch and every treated
condition adds ``EFFECT``. A pairing by folder label then compares the wrong batches.
"""

from __future__ import annotations

import hashlib
import json
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
EXPORTS = HERE / "morphofeature_exports"
SPEC = HERE / "spec_fixtures"
NAMES = ("v03", "v10", "v27", "v30", "v33")
KEY = ["condition", "biological_repeat", "technical_repeat", "position_id"]
IDS = ["repeat_group", "condition_id", "technical_repeat_id"]
FEATURE = "Cells__area"
# The added feature's batch per repeat (in the description's order) and treatment effect.
BATCH_STEP = 6.0
EFFECT = 1.5


def git_blob_id(data: bytes) -> str:
    """The git blob-object ID of ``data`` (``git hash-object``)."""
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def provenance(name: str) -> dict:
    """The export's ``provenance.json``."""
    return json.loads((EXPORTS / name / "provenance.json").read_text(encoding="utf-8"))


def description(name: str) -> dict:
    """The vendored Spec fixture the export was made from, parsed."""
    return tomllib.loads((SPEC / provenance(name)["fixture"]).read_text(encoding="utf-8"))


def described_groups(name: str) -> dict[tuple[str, str], str]:
    """``(condition folder, repeat folder)`` -> repeat ID, read from the description."""
    data = description(name)
    folders = {cid: c["source_folder"] for cid, c in data["conditions"].items()}
    return {
        (folders[cid], member["source_repeat"]): rid
        for rid, repeat in data["repeats"].items()
        for cid, member in repeat["members"].items()
    }


def described_conditions(name: str) -> dict[str, str]:
    """Condition folder -> condition ID, read from the description."""
    return {c["source_folder"]: cid for cid, c in description(name)["conditions"].items()}


def read_export(name: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """The export's observation table, FOV table and ``shape`` feature table, as written."""
    folder = EXPORTS / name
    objects = pd.read_csv(folder / "objects.csv")
    fovs = pd.read_csv(folder / "fovs.csv")
    shape = pd.read_csv(folder / "tables" / "shape.csv")
    return objects, fovs, shape


def with_feature(name: str, x: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The observation table with ``Cells__area`` joined (and the added feature ``x``), and
    the FOV table, as written."""
    objects, fovs, shape = read_export(name)
    joined = objects.merge(shape, on=[*KEY, "label_id"], how="left", validate="one_to_one")
    if x:
        joined["x"] = added_feature(name, joined)
    return joined, fovs


def added_feature(name: str, objects: pd.DataFrame) -> np.ndarray:
    """``x``: the repeat's batch (from the description), plus ``EFFECT`` for every condition
    but the first declared, plus a small deterministic spread by field and label."""
    groups = described_groups(name)
    order = list(description(name)["repeats"])
    first = next(iter(description(name)["conditions"].values()))["source_folder"]
    batch = [
        BATCH_STEP * order.index(groups[(c, r)])
        for c, r in zip(objects["condition"], objects["biological_repeat"], strict=True)
    ]
    effect = np.where(objects["condition"] == first, 0.0, EFFECT)
    field = objects["position_id"].str.removeprefix("img").astype(int)
    spread = 0.05 * field + 0.1 * objects["label_id"]
    return np.asarray(batch) + effect + spread.to_numpy()
