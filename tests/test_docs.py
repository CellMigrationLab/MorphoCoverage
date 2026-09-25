"""Documentation examples must run against the current API."""

import os
import re
import runpy
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def test_readme_current_status_example_runs(tmp_path):
    text = (ROOT / "README.md").read_text()
    section = re.search(r"## Current status.*?```python\n(.*?)```", text, flags=re.DOTALL)
    assert section, "README 'Current status' example not found"
    script = tmp_path / "readme_example.py"
    script.write_text(section.group(1))
    cwd = os.getcwd()
    try:
        os.chdir(ROOT)  # the example uses repository-relative data paths
        namespace = runpy.run_path(str(script))
    finally:
        os.chdir(cwd)
    results = namespace["results"]
    assert results and all(r.result.feature_cols for r in results)


def _constructed_wells() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Three wells of 8 fields with a modest field effect: two control, one treated."""
    rng = np.random.default_rng(0)
    rows, fields = [], []
    for well, condition in enumerate(["control", "control", "treated"]):
        for j in range(8):
            name = f"w{well}_f{j}"
            fields.append(
                {"condition": condition, "repeat": "r1", "well": f"w{well}", "image_file": name}
            )
            offset = rng.normal(0, 0.3)
            for _ in range(30):
                rows.append(
                    {
                        "condition": condition,
                        "repeat": "r1",
                        "well": f"w{well}",
                        "image_file": name,
                        "ratio": 5 + offset + rng.normal(),
                        "b": rng.normal(),
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(fields)


def test_objects_or_fields_example_runs(tmp_path, capsys):
    """The usage example of docs/objects_or_fields.md runs against the current API."""
    text = (ROOT / "docs" / "objects_or_fields.md").read_text()
    section = re.search(r"## Checking your own data.*?```python\n(.*?)```", text, flags=re.DOTALL)
    assert section, "docs/objects_or_fields.md example not found"
    script = tmp_path / "objects_or_fields_example.py"
    script.write_text(section.group(1))
    objects, fovs = _constructed_wells()
    runpy.run_path(
        str(script), init_globals={"objects": objects, "fovs": fovs, "features": ["ratio", "b"]}
    )
    printed = capsys.readouterr().out
    assert "How variable are the fields?" in printed
    assert "Are the states spread over the fields?" in printed
