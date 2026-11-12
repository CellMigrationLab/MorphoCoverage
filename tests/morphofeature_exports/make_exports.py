"""Write the golden MorphoFeature exports of the Spec fixtures v03, v30, v33 and v27 (MC-0, MC-2).

Not run by the test suite: it needs a MorphoFeature checkout and its pinned dependencies
(morphobricks-preprocessing, CellPatcher, normalization, the ``cli`` extra). It does what
MorphoFeature's own ``tests/test_coverage_repeat_group.py`` does (MF-5): each vendored fixture
goes through preprocessing's ``make_described_run`` with one ``Cells`` instance mask (two
labelled squares per image), then through MorphoFeature's run file (``run_from_config``) with
its fake ``shape`` provider (``area`` = the label value), tables written as CSV. The set's
``coverage/Cells`` folder is copied here unchanged, beside ``provenance.json``. Since MF-5b
(MorphoFeature #215) its ``coverage.json`` is ``schema_version`` 3 and ends with the
``experiment`` block MC-2 reads.

Usage, from the MorphoFeature checkout's root, in its environment::

    python -I <this file> --morphofeature . --spec-fixtures <here>/../spec_fixtures \
        --out <here>
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import shutil
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

FIXTURES = {
    "v03": "v03_paired_different_folder_names.toml",
    "v30": "v30_same_source_label_different_repeats.toml",
    "v33": "v33_condition_folder_on_forbidden_list.toml",
    "v27": "v27_descriptive_keys_collide.toml",
}
CELLS = """[masks.Cells]
source_type = "cells"
representation = "instances"
source_channels = ["DNA"]

[used_ids]"""


def git_blob_id(data: bytes) -> str:
    """Return the git blob-object ID of ``data`` (``git hash-object``)."""
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def main() -> int:
    """Write one folder per fixture under ``--out``."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--morphofeature", type=Path, required=True)
    parser.add_argument("--spec-fixtures", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    mf = args.morphofeature.resolve()
    sys.path[:0] = [str(mf / "tests"), str(mf / "tools")]
    from morphobricks_preprocessing import testing_described
    from morphofeature.cli import run_from_config
    from morphofeature.run_config import parse_run_config
    from test_brick_runner import _load  # MorphoFeature's fake provider loader
    from test_run_config import SHAPE  # its fake ``shape`` provider, feature ``area``

    commit = subprocess.run(
        ["git", "-C", str(mf), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(mf), "status", "--porcelain", "--untracked-files=no"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if dirty:
        raise SystemExit(f"the MorphoFeature checkout has local changes:\n{dirty}")
    pin = tomllib.loads((args.spec_fixtures / "PIN.toml").read_text(encoding="utf-8"))
    for name, file in FIXTURES.items():
        raw = (args.spec_fixtures / "fixtures" / "valid" / file).read_bytes()
        text = raw.decode("utf-8").replace("[used_ids]", CELLS)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            run = testing_described.make_described_run(tmp_path / name, text, masks=("Cells",))
            data = {
                "preprocessing_run": str(run.output),
                "experiment_root": str(run.experiment),
                "masks_root": str(run.masks),
                "masks": {"Cells": "Cells"},
                "requests": [{"provider": "shape", "feature": "area", "object_set": "Cells"}],
                "format": "csv",
                "output": str(tmp_path / name / "features"),
            }
            run_from_config(parse_run_config(data, tmp_path), [SHAPE], load=_load)
            source = tmp_path / name / "features" / "coverage" / "Cells"
            target = args.out / name
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(source, target)
        provenance = {
            "fixture": f"fixtures/valid/{file}",
            "fixture_blob": git_blob_id(raw),
            "spec_commit": pin["commit"],
            "morphofeature_commit": commit,
            "morphofeature_source": "CellMigrationLab/MorphoFeature main (MF-5b, #215)",
            "object_set": "Cells",
            "provider": "MorphoFeature's test fake `shape` (area = the label value)",
            "mask": "Cells, instances, two labelled squares per image (make_described_run)",
            "format": "csv",
            "installed": {
                package: importlib.metadata.version(package)
                for package in (
                    "morphobricks-preprocessing",
                    "cellpatcher",
                    "morphobricks-normalization",
                    "numpy",
                    "pandas",
                )
            },
            "python": sys.version.split()[0],
        }
        (target / "provenance.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(name, sorted(p.name for p in target.rglob("*") if p.is_file()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
