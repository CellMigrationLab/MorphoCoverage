"""The ratchets on the function findings: they only tighten, and a new entry needs a reason.

``tools/baselines/<kind>.json`` maps ``module:name`` to the value allowed now (lines,
complexity, arguments over the limits of ``function_index``; 1 for a public function without
a docstring; the number of unmarked silent fallbacks). ``tests/test_function_guards.py``
fails when a function is over its baseline, and when a baseline still holds what was fixed.

- ``tighten`` lowers every entry to the current value and drops the fixed ones; it never
  adds. A function renamed or moved keeps its entry when its body is unchanged (same body
  hash in the old and the new index) or when ``renames.json`` records the move.
- ``accept`` adds or raises one entry and records why in ``accepted.json`` (kind, key,
  value, reason, date): the only way in.
- ``check_against`` compares the baselines with those of a git ref (the merge base in CI):
  a new or larger entry needs a recorded acceptance, a new entry of the duplicate and layer
  baselines (which map each key to its reason) needs a reason, and the limits never change.
"""

from __future__ import annotations

import datetime
import json
import re
import subprocess
from pathlib import Path

import fallbacks as fb
import function_index as fi

KINDS = ("lines", "complexity", "arguments", "no_docstring", "fallbacks")
LIMITS = {"lines": "MAX_LINES", "complexity": "MAX_COMPLEXITY", "arguments": "MAX_ARGUMENTS"}
FOLDER = fi.BASELINES  # where the baselines are (a test points it elsewhere)
ACCEPTED, RENAMES = "accepted.json", "renames.json"


def findings(
    rows: list[dict[str, object]] | None = None, found: list[fb.Fallback] | None = None
) -> dict[str, dict[str, int]]:
    """Every baseline-able finding: ``{kind: {"module:name": value}}``."""
    rows = fi.build_index() if rows is None else rows
    out: dict[str, dict[str, int]] = {kind: {} for kind in KINDS}
    for row in fi.function_rows(rows):
        for kind, attribute in LIMITS.items():
            value = int(row[kind])  # type: ignore[call-overload]
            if value > getattr(fi, attribute):
                out[kind][fi.key(row)] = value
        if row["public"] and not row["summary"]:
            out["no_docstring"][fi.key(row)] = 1
    out["fallbacks"] = fb.unmarked(fb.fallbacks() if found is None else found)
    return out


def _read(path: Path, default: object) -> object:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def read_baseline(kind: str) -> dict[str, int]:
    """One baseline, ``{}`` when absent."""
    return dict(_read(FOLDER / f"{kind}.json", {}))  # type: ignore[call-overload]


def write_baseline(kind: str, values: dict[str, int]) -> None:
    """Write one baseline, keys sorted."""
    _write(FOLDER / f"{kind}.json", dict(sorted(values.items())))


def read_accepted() -> list[dict[str, object]]:
    """The recorded acceptances (kind, key, value, reason, date)."""
    return list(_read(FOLDER / ACCEPTED, []))  # type: ignore[call-overload]


def read_renames() -> dict[str, str]:
    """``{new key: old key}`` of the recorded moves with edits."""
    entries: list[dict[str, str]] = _read(FOLDER / RENAMES, [])  # type: ignore[assignment]
    return {entry["to"]: entry["from"] for entry in entries}


def moved(
    old_rows: list[dict[str, object]], new_rows: list[dict[str, object]], renames: dict[str, str]
) -> dict[str, str]:
    """``{new key: old key}`` of the functions renamed or moved: the body hash of a key that
    disappeared found once among the keys that appeared, or a recorded move."""
    old = {fi.key(r): r.get("hash") for r in old_rows if r.get("kind") != "module"}
    new = {fi.key(r): r.get("hash") for r in new_rows if r.get("kind") != "module"}
    gone: dict[object, list[str]] = {}
    for key, digest in old.items():
        if key not in new:
            gone.setdefault(digest, []).append(key)
    appeared: dict[object, list[str]] = {}
    for key, digest in new.items():
        if key not in old:
            appeared.setdefault(digest, []).append(key)
    out = {
        keys[0]: gone[digest][0]
        for digest, keys in appeared.items()
        if len(keys) == 1 and len(gone.get(digest, [])) == 1
    }
    out.update({new_key: old_key for new_key, old_key in renames.items() if new_key in new})
    return out


def tighten(
    current: dict[str, dict[str, int]],
    old_rows: list[dict[str, object]],
    new_rows: list[dict[str, object]],
) -> dict[str, tuple[int, int]]:
    """Lower every baseline to the current values, drop what was fixed, carry renamed keys;
    never add. Returns ``{kind: (entries before, entries after)}``."""
    renamed = {old: new for new, old in moved(old_rows, new_rows, read_renames()).items()}
    summary = {}
    for kind in KINDS:
        before = read_baseline(kind)
        after = {}
        for key, allowed in before.items():
            now = key if key in current[kind] else renamed.get(key, key)
            if now in current[kind]:
                after[now] = min(allowed, current[kind][now], after.get(now, allowed))
        write_baseline(kind, after)
        summary[kind] = (len(before), len(after))
    return summary


def accept(target: str, reason: str, current: dict[str, dict[str, int]]) -> str:
    """Add or raise one baseline entry (``kind:module:name``) to its current value and record
    the reason. Returns what was done; raises ``ValueError`` when there is nothing to accept."""
    kind, _, key = target.partition(":")
    if kind not in KINDS or not key:
        raise ValueError(f"--accept takes KIND:module:name with KIND one of {', '.join(KINDS)}")
    if not reason.strip():
        raise ValueError("--accept needs --reason: why this function may stay over the limit")
    if key not in current[kind]:
        raise ValueError(f"{key} has no {kind} finding: nothing to accept")
    baseline = read_baseline(kind)
    value = current[kind][key]
    if baseline.get(key, 0) >= value:
        raise ValueError(f"{key} is already allowed {kind} {baseline[key]}: nothing to accept")
    baseline[key] = value
    write_baseline(kind, baseline)
    entry = {
        "kind": kind,
        "key": key,
        "value": value,
        "reason": reason.strip(),
        "date": _today(),
    }
    _write(FOLDER / ACCEPTED, [*read_accepted(), entry])
    return f"accepted {kind} {key} = {value}"


def _today() -> str:
    return datetime.datetime.now(tz=datetime.UTC).date().isoformat()


def record_rename(old: str, new: str, reason: str) -> str:
    """Record a move with edits (``renames.json``): the baselines of ``old`` carry to ``new``."""
    entries: list[dict[str, str]] = _read(FOLDER / RENAMES, [])  # type: ignore[assignment]
    entry = {"from": old, "to": new, "reason": reason, "date": _today()}
    _write(FOLDER / RENAMES, [*entries, entry])
    return f"recorded the move {old} -> {new}"


def _git(*args: str) -> str:
    done = subprocess.run(["git", *args], cwd=fi.ROOT, capture_output=True, text=True, check=False)
    if done.returncode:
        raise RuntimeError(f"git {' '.join(args)} failed: {done.stderr.strip()}")
    return done.stdout


def _at(commit: str, path: Path, default: str) -> str:
    try:
        return _git("show", f"{commit}:{path.relative_to(fi.ROOT).as_posix()}")
    except RuntimeError:
        return default


def _limits(text: str) -> dict[str, str]:
    return dict(re.findall(r"^(MAX_[A-Z]+) = (\S+)", text, flags=re.MULTILINE))


def compare(
    base: dict[str, dict[str, int]],
    head: dict[str, dict[str, int]],
    renamed: dict[str, str],
    accepted: list[dict[str, object]],
) -> list[str]:
    """The baseline entries of ``head`` that are new or larger than in ``base`` without a
    recorded acceptance of at least their value."""
    allowed = {(str(e["kind"]), str(e["key"])): int(e["value"]) for e in accepted}  # type: ignore[call-overload]
    problems = []
    for kind in KINDS:
        for key, value in sorted(head.get(kind, {}).items()):
            before = base.get(kind, {}).get(key, base.get(kind, {}).get(renamed.get(key, "")))
            if before is not None and value <= before:
                continue
            if value <= allowed.get((kind, key), 0):
                continue
            change = "new entry" if before is None else f"raised from {before}"
            problems.append(f"{kind}: {key} = {value} ({change}) without `--accept` reason")
    return problems


def _reasoned(text: str, name: str) -> dict[str, str]:
    """``{key: reason}`` of a baseline that records reasons (``duplicates.json``: key to reason;
    ``layers.json``: a section of edges to reasons), keys prefixed by file and section."""
    value = json.loads(text) if text.strip() else {}
    if not isinstance(value, dict):  # an older baseline, a list without reasons
        return {f"{name}: {key}": "" for key in value}
    out = {}
    for key, reason in value.items():
        if isinstance(reason, dict):
            out.update({f"{name} {key}: {k}": str(r) for k, r in reason.items()})
        else:
            out[f"{name}: {key}"] = str(reason)
    return out


def reasoned_changes(base: dict[str, str], head: dict[str, str]) -> tuple[list[str], list[str]]:
    """``(problems, admitted)``: an entry new on the branch needs a reason; with one it is
    listed for the reviewer."""
    problems, admitted = [], []
    for key, reason in sorted(head.items()):
        if key in base:
            continue
        if reason.strip():
            admitted.append(f"{key} ({reason.strip()})")
        else:
            problems.append(f"{key}: new entry without a reason")
    return problems, admitted


def check_against(ref: str) -> tuple[list[str], list[str]]:
    """``(problems, admitted)`` of the baselines on disk against those at the merge base with
    ``ref``: unaccepted growth and changed limits; the entries admitted with their reasons."""
    commit = _git("merge-base", "HEAD", ref).strip()
    base = {kind: json.loads(_at(commit, fi.BASELINES / f"{kind}.json", "{}")) for kind in KINDS}
    head = {kind: read_baseline(kind) for kind in KINDS}
    old_rows = [json.loads(line) for line in _at(commit, fi.INDEX, "").splitlines() if line]
    renamed = moved(old_rows, fi.read_index(), read_renames())
    # Only an acceptance recorded on this branch admits an entry: an older one was for the
    # value then, and the ratchet may have lowered it since.
    earlier = json.loads(_at(commit, fi.BASELINES / ACCEPTED, "[]"))
    accepted = [e for e in read_accepted() if e not in earlier]
    problems = compare(base, head, renamed, accepted)
    admitted = [f"{e['kind']}: {e['key']} = {e['value']} ({e['reason']})" for e in accepted]
    for name in ("duplicates", "layers"):
        path = fi.BASELINES / f"{name}.json"
        then = _reasoned(_at(commit, path, ""), name)
        now = _reasoned(path.read_text(encoding="utf-8") if path.exists() else "", name)
        found, listed = reasoned_changes(then, now)
        problems += found
        admitted += listed
    tool = Path(fi.__file__)
    before, after = _limits(_at(commit, tool, "")), _limits(tool.read_text(encoding="utf-8"))
    for name in sorted(set(before) | set(after)):
        if before and before.get(name) != after.get(name):
            problems.append(f"limit {name} changed: {before.get(name)} -> {after.get(name)}")
    return problems, admitted
