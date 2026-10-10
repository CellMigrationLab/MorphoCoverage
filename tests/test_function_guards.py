"""The function index and its ratchets (tools/function_index.py): the index is current, and no
function is longer, more complex, wider, undocumented or more silent than the baselines allow."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "function_index", ROOT / "tools/function_index.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TOOL = _tool()
REGENERATE = "run `python tools/function_index.py`"
RATCHET = "run `python tools/function_index.py --write-baselines` only to record a REMOVAL"


@pytest.fixture(scope="module")
def current() -> dict[str, dict[str, int]]:
    return TOOL.findings()


def test_the_function_index_is_current():
    rows = TOOL.build_index()
    committed = TOOL.read_index()
    now, then = set(map(tuple, TOOL.index_key(rows))), set(map(tuple, TOOL.index_key(committed)))
    added, removed = sorted(now - then), sorted(then - now)
    assert not added and not removed, (
        f"docs/common/function_index.jsonl is out of date ({len(added)} new or changed, "
        f"{len(removed)} gone or changed; first: {(added or removed)[0][:2]}): {REGENERATE}"
    )


@pytest.mark.parametrize("kind", ["lines", "complexity", "arguments", "no_docstring", "fallbacks"])
def test_nothing_grows_beyond_its_baseline(current, kind):
    baseline = TOOL.read_baseline(kind)
    grown = {k: (baseline.get(k, 0), v) for k, v in current[kind].items() if v > baseline.get(k, 0)}
    assert not grown, f"{kind}: new or larger (was, now): {dict(list(grown.items())[:8])}"


@pytest.mark.parametrize("kind", ["lines", "complexity", "arguments", "no_docstring", "fallbacks"])
def test_a_baseline_never_holds_what_was_fixed(current, kind):
    stale = {k: v for k, v in TOOL.read_baseline(kind).items() if current[kind].get(k, 0) < v}
    assert not stale, (
        f"{kind}: fixed or shrunk, tighten the baseline ({RATCHET}): {list(stale)[:8]}"
    )
