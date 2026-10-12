"""The function index and its ratchets (tools/function_index.py, tools/ratchet.py): the index is
current, and no function is longer, more complex, wider, undocumented or more silent than the
baselines allow."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import duplicates as DUPLICATES
import function_index as TOOL
import layers as LAYERS
import ratchet as RATCHETS

REGENERATE = "run `python tools/audit.py`"
RATCHET = "run `python tools/audit.py --tighten` to record a removal"
KINDS = ["lines", "complexity", "arguments", "no_docstring", "fallbacks"]


@pytest.fixture(scope="module")
def rows() -> list[dict[str, object]]:
    return TOOL.build_index()


@pytest.fixture(scope="module")
def current(rows) -> dict[str, dict[str, int]]:
    return RATCHETS.findings(rows)


def test_the_limits_are_pinned():
    # Raising a limit would admit every function under it: a change of rule, not of code.
    assert (TOOL.MAX_LINES, TOOL.MAX_COMPLEXITY, TOOL.MAX_ARGUMENTS) == (60, 15, 8)


def test_every_index_key_is_unique(rows):
    repeated = [k for k, n in Counter(map(TOOL.key, rows)).items() if n > 1]
    assert not repeated, f"index keys must be unique (name@setter, name#2): {repeated[:5]}"


def test_the_function_index_is_current(rows):
    now = Counter(map(TOOL.row_text, rows))
    then = Counter(map(TOOL.row_text, TOOL.read_index()))
    added, removed = sorted((now - then).elements()), sorted((then - now).elements())
    first = json.loads((added or removed or ["{}"])[0])
    assert not added and not removed, (
        f"docs/common/function_index.jsonl is out of date ({len(added)} new or changed, "
        f"{len(removed)} gone or changed; first: {first.get('module')}:{first.get('name')}): "
        f"{REGENERATE}"
    )


@pytest.mark.parametrize("kind", KINDS)
def test_nothing_grows_beyond_its_baseline(current, kind):
    baseline = RATCHETS.read_baseline(kind)
    grown = {k: (baseline.get(k, 0), v) for k, v in current[kind].items() if v > baseline.get(k, 0)}
    hint = (
        "mark a deliberate fallback `# fallback: <reason>` on its line"
        if kind == "fallbacks"
        else 'meet the limit, or `python tools/audit.py --accept KIND:key --reason "..."`'
    )
    assert not grown, f"{kind}: new or larger (was, now): {dict(list(grown.items())[:8])}; {hint}"


@pytest.mark.parametrize("kind", KINDS)
def test_a_baseline_never_holds_what_was_fixed(current, kind):
    baseline = RATCHETS.read_baseline(kind)
    stale = {k: v for k, v in baseline.items() if current[kind].get(k, 0) < v}
    assert not stale, (
        f"{kind}: fixed or shrunk, tighten the baseline ({RATCHET}): {list(stale)[:8]}"
    )


def test_every_acceptance_has_a_reason_and_a_date():
    for entry in RATCHETS.read_accepted():
        assert set(entry) >= {"kind", "key", "value", "reason", "date"}, entry
        assert entry["kind"] in KINDS and str(entry["reason"]).strip(), entry


def _baseline_file(name: str):
    return json.loads((TOOL.BASELINES / f"{name}.json").read_text(encoding="utf-8"))


def test_no_new_exact_duplicate_functions():
    new = sorted(set(DUPLICATES.exact_groups()) - set(_baseline_file("duplicates")))
    assert not new, f"new exact duplicates (merge them, or explain): {new[:5]}"


def test_a_duplicate_baseline_never_holds_what_was_merged():
    stale = sorted(set(_baseline_file("duplicates")) - set(DUPLICATES.exact_groups()))
    assert not stale, f"merged: tighten the baseline ({RATCHET}): {stale[:5]}"


def test_no_import_crosses_a_layer_it_must_not():
    allowed = set(_baseline_file("layers")["violations"])
    new = sorted(set(LAYERS.violations()) - allowed)
    assert not new, f"a layer imports one it must not (tools/layers.py RULES): {new[:5]}"
    stale = sorted(allowed - set(LAYERS.violations()))
    assert not stale, f"fixed, tighten the baseline ({RATCHET}): {stale[:5]}"


def test_no_new_import_cycle():
    allowed = set(_baseline_file("layers")["cycles"])
    new = sorted(set(LAYERS.cycles()) - allowed)
    assert not new, f"new import cycle between modules: {new[:3]}"
    stale = sorted(allowed - set(LAYERS.cycles()))
    assert not stale, f"cycle gone, tighten the baseline ({RATCHET}): {stale[:3]}"
