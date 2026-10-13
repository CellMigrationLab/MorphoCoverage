"""The naming rule (AGENTS.md section 7, roadmap ENG-4): a file or directory is named for what
it holds, never for the audit, review, pull request, plan phase or date that made it.

Read from the tracked paths alone (``git ls-files``). A path breaks the rule when one of its
parts, without its extension, holds the word "audit" or "review", a pull request or issue number
(``pr_12``, ``issue412``), a plan phase (``plan_v2``, ``phase1``), a date (``2026-10-13``, or a
month-day number such as ``_1005``) or a pass number (``_r3``, ``round2``). Dataset files under
``data/`` keep their own numbers, so only the words, pull requests and plan phases are read
there. ``ALLOWED`` lists the tools that carry the words by design, ``ALLOWED_DIRECTORIES``
the record directories, whose files are still read.
"""

from __future__ import annotations

import re
import subprocess
from collections.abc import Iterable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

ALLOWED = (
    "tools/audit.py",
    "tools/jscpd_review.py",
    "tools/baselines/duplicates_reviewed.json",
)
# directories whose own name carries a word by design; the files in them are still read
ALLOWED_DIRECTORIES = ("docs/audit",)

WORDS = {
    "an audit or a review": re.compile(r"(^|[_-])(audit|review)[a-z]*\d*($|[_-])"),
    "a pull request or an issue": re.compile(r"(^|[_-])(pr|issue)[_-]?\d+"),
    "a plan phase": re.compile(r"plan[_-]?v\d|phase[_-]?\d"),
}
NUMBERS = {
    "a date": re.compile(
        r"(19|20)\d\d[-_]?\d\d[-_]?\d\d|(^|[_-])(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])($|[_-])"
    ),
    "a pass": re.compile(r"(^|[_-])(r|round|pass)\d+($|[_-])"),
}


def tracked(root: Path = ROOT) -> list[str]:
    """The paths git tracks under ``root``, relative to it, with forward slashes."""
    listed = subprocess.run(
        ["git", "ls-files"], cwd=root, capture_output=True, text=True, check=True
    )
    return listed.stdout.split()


def problem(path: str) -> str | None:
    """Why ``path`` breaks the rule, or None when it does not."""
    if path in ALLOWED:
        return None
    rules = WORDS if path.startswith("data/") else {**WORDS, **NUMBERS}
    parts = path.split("/")
    for depth, part in enumerate(parts):
        if "/".join(parts[: depth + 1]) in ALLOWED_DIRECTORIES:
            continue
        stem = part.rsplit(".", 1)[0].lower()
        for what, pattern in rules.items():
            if pattern.search(stem):
                return f"{path}: named after {what} ('{part}'); name it for what it holds"
    return None


def violations(paths: Iterable[str] | None = None) -> list[str]:
    """Every tracked path that breaks the rule, with the reason."""
    found = (problem(path) for path in (tracked() if paths is None else paths))
    return sorted(line for line in found if line)


if __name__ == "__main__":
    print("\n".join(violations()) or "every name says what it holds")
