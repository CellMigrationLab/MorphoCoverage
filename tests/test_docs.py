"""Documentation examples must run against the current API."""

import os
import re
import runpy
from pathlib import Path

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
