"""What ``benchmark_colab.ipynb`` calls: run P0 and the module scripts, summarise, zip.

The notebook stays thin (AGENTS.md: notebooks call code, they hold none): it finds or fetches
the repository, chooses an output folder (a mounted Google Drive folder on Colab) and calls
``run`` and ``summarise`` here. Every module resumes from what is already in the output
folder, so a run cut by a Colab timeout continues where it stopped when the cell is run
again (``runner``).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from . import runner
from .summary import summarise as summarise_module

MODULES = {
    "P1": "decisions.p1_repeat_inference",
    "P2": "decisions.p2_contrasts",
    "P3": "decisions.p3_embedding",
    "P4": "decisions.p4_sequential",
    "P5": "decisions.p5_multiple_testing",
}
P0_TESTS = "tests/test_validation_p0.py"


def repository() -> Path:
    """The repository root (this file is in ``data/simulations/decisions``)."""
    return Path(__file__).resolve().parents[3]


def run_p0(out: Path) -> int:
    """The P0 tests (pytest, serially), their report written to ``<out>/p0/p0.txt``; returns
    pytest's exit code (0: every test passed or failed as recorded, ``xfail``)."""
    root = repository()
    environment = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([str(root / "src"), str(root / "data/simulations")]),
    }
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-n",
        "0",
        "-q",
        "-rxX",
        "-p",
        "no:cacheprovider",
        str(root / P0_TESTS),
    ]
    found = subprocess.run(
        command, cwd=root, env=environment, capture_output=True, text=True, check=False
    )
    folder = Path(out) / "p0"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "p0.txt").write_text(found.stdout + found.stderr)
    print(found.stdout[-3000:])
    return found.returncode


def run(
    modules: Sequence[str],
    size: str,
    out: Path,
    workers: int | None = None,
    chunk: int = 100,
) -> None:
    """Run each of ``modules`` (``"P0"``, ``"P1"`` ... ``"P5"``) at ``size`` into ``out``.
    A failed P0 stops the run: the modules after it are not run on a baseline that failed
    its gate (#520)."""
    out = Path(out)
    for name in modules:
        if name.upper() == "P0":
            code = run_p0(out)
            if code:
                raise RuntimeError(
                    f"P0 failed (pytest exit {code}; {out / 'p0' / 'p0.txt'}): the modules "
                    "after it were not run, and no result of this run is confirmatory"
                )
            continue
        runner.run(MODULES[name.upper()], out, size, workers, chunk)


def summarise(modules: Sequence[str], out: Path) -> list[Path]:
    """``summary.csv`` and ``summary.md`` of each module run in ``out``; prints each
    ``summary.md``."""
    written = []
    for name in modules:
        if name.upper() == "P0":
            continue
        files = summarise_module(MODULES[name.upper()], Path(out))
        print(files[1].read_text())
        written += files
    return written


def archive(out: Path) -> Path:
    """``out`` zipped (``<out>.zip`` beside it), to download and send back."""
    out = Path(out)
    return Path(shutil.make_archive(str(out), "zip", root_dir=out))


__all__ = ["MODULES", "archive", "repository", "run", "run_p0", "summarise"]
