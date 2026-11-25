"""The example notebooks stay in step with the package (they are run by hand, not here).

Executing a notebook needs Jupyter and matplotlib, which are not dependencies; these
checks only make sure every code cell still compiles, every name it imports from
MorphoCoverage exists, and no outputs are committed.
"""

from __future__ import annotations

import ast
import importlib
import json
from pathlib import Path

import pytest

NOTEBOOKS = sorted((Path(__file__).resolve().parents[1] / "notebooks").glob("*/*.ipynb"))


def _code_cells(path: Path) -> list[str]:
    notebook = json.loads(path.read_text())
    return ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]


def _python(source: str) -> str:
    # Shell escapes and magics (``!git ...``, ``%pip ...``) are IPython syntax, not Python.
    return "\n".join(
        line[: len(line) - len(line.lstrip())] + "pass"
        if line.lstrip().startswith(("!", "%"))
        else line
        for line in source.splitlines()
    )


def test_there_is_an_example_notebook():
    assert NOTEBOOKS


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.stem)
def test_every_code_cell_compiles_and_has_a_title(path):
    for source in _code_cells(path):
        compile(_python(source), str(path), "exec")
        assert source.lstrip().startswith(("# @title", "#@title")), source[:60]


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.stem)
def test_every_imported_morphocoverage_name_exists(path):
    for source in _code_cells(path):
        for node in ast.walk(ast.parse(_python(source))):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
                "morphocoverage"
            ):
                module = importlib.import_module(node.module)
                for alias in node.names:
                    assert hasattr(module, alias.name), f"{node.module}.{alias.name}"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.stem)
def test_no_outputs_are_committed(path):
    notebook = json.loads(path.read_text())
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert not cell.get("outputs") and cell.get("execution_count") is None


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.stem)
def test_notebooks_run_in_both_colab_and_jupyterlab(path):
    # LabConstrictor's portability guide: no Colab forms (they do nothing in JupyterLab),
    # and Colab-only code and paths only behind the ``'google.colab' in sys.modules`` guard.
    for source in _code_cells(path):
        assert "@param" not in source, "use ipywidgets, not Colab forms"
        guarded = "'google.colab' in sys.modules" in source
        for line in source.splitlines():
            stripped = line.lstrip()
            if "/content" in line or "from google.colab" in line or stripped.startswith(("!", "%")):
                assert guarded, line


def test_the_workflows_example_pairs_the_positive_control_by_repeat_group_640():
    # The folder label (biological_repeat) is never a pairing (#617): the example taught it.
    path = next(p for p in NOTEBOOKS if p.stem == "MorphoCoverage_Workflows_Example")
    declaration = next(s for s in _code_cells(path) if "PositiveControl(" in s and "Setup(" in s)
    calls = [
        node
        for node in ast.walk(ast.parse(_python(declaration)))
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "PositiveControl"
    ]
    assert calls
    for call in calls:
        pair_cols = next(k.value for k in call.keywords if k.arg == "pair_cols")
        assert ast.literal_eval(pair_cols) == ("repeat_group",)
    # Without the column the declaration stops, never falling back to the folder label.
    assert '"repeat_group" not in fovs.columns' in declaration and "raise ValueError" in declaration
