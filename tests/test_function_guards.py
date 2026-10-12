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


@pytest.fixture(scope="module")
def identical() -> dict[str, list[str]]:
    return DUPLICATES.exact_groups()


def test_no_new_identical_function_bodies(identical):
    accepted = DUPLICATES.read_baseline()
    new = {k: v for k, v in identical.items() if k not in accepted}
    assert not new, (
        "identical function bodies (merge them, or accept with a reason: "
        f"`python tools/duplicates.py --accept HASH --reason ...`): {list(new.values())[:5]}"
    )


def test_no_accepted_duplicate_group_gains_a_member(identical):
    count = {key.split(":")[0]: len(members) for key, members in identical.items()}
    grown = {
        key: count[key.split(":")[0]]
        for key in DUPLICATES.read_baseline()
        if count.get(key.split(":")[0], 0) > int(key.split(":")[1])
    }
    assert not grown, f"an accepted identical group gained a member (now): {grown}"


def test_every_accepted_duplicate_group_has_a_reason_and_still_exists(identical):
    accepted = DUPLICATES.read_baseline()
    assert all(reason.strip() for reason in accepted.values())
    stale = sorted(set(accepted) - set(identical))
    assert not stale, f"merged or shrunk, run `python tools/duplicates.py --tighten`: {stale[:5]}"


def test_no_import_crosses_a_layer_it_must_not():
    allowed = LAYERS.read_baseline()["violations"]
    found = LAYERS.violations()
    new = sorted(set(found) - set(allowed))
    assert not new, f"a layer imports one it must not (tools/layers.py RULES): {new[:5]}"
    stale = sorted(set(allowed) - set(found))
    assert not stale, f"fixed, run `python tools/layers.py --tighten`: {stale[:5]}"


def test_no_runtime_import_cycle():
    found = LAYERS.cycles()
    assert not found, f"modules import each other at import time (no baseline): {found[:3]}"


def test_no_new_deferred_import_inside_a_cycle():
    allowed = LAYERS.read_baseline()["back_edges"]
    found = LAYERS.back_edges()
    new = sorted(set(found) - set(allowed))
    assert not new, (
        "a function-local or TYPE_CHECKING import closes an import cycle (break the cycle, "
        f"or record the edge with a reason in tools/baselines/layers.json): {new[:5]}"
    )
    stale = sorted(set(allowed) - set(found))
    assert not stale, f"gone, run `python tools/layers.py --tighten`: {stale[:5]}"
