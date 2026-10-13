"""Compare two reference output trees file for file, after removing the run's stamps only.

What is removed, and nothing else:

- the page's ``Generated <date> UTC`` line (``*.html``);
- each PDF's ``/CreationDate``, also in the PDFs the page carries (base64 inside ``*.html``);
- the run time and package lines of a ``run.txt`` (``run time: ...``, ``morphocoverage: ...``);
- the replay seconds ``benchmark_summary.txt`` prints (``replay 870 s)``, ``replays 12 s)``).

Logs (``*.log``) and ``traceback.txt`` (file paths and line numbers, which move with any
refactor; the error's type and message are compared through ``record.json``) are skipped.
"""

from __future__ import annotations

import base64
import re
from pathlib import Path

PDF_STAMP = re.compile(rb"/CreationDate \(D:\d+Z\)")
PAGE_STAMP = re.compile(rb"Generated \d{4}-\d\d-\d\d \d\d:\d\d UTC")
EMBEDDED_PDF = re.compile(rb'"application/pdf","([A-Za-z0-9+/=]+)"')
REPLAY_TIME = re.compile(rb"(replays?) \d+ s\)")
RUN_STAMPS = (b"run time", b"morphocoverage:")
SKIPPED_SUFFIXES = frozenset({".log"})
SKIPPED_NAMES = frozenset({"traceback.txt"})


def _without_pdf_stamp(match: re.Match[bytes]) -> bytes:
    """An embedded PDF, decoded, its creation date removed, encoded again."""
    pdf = PDF_STAMP.sub(b"", base64.b64decode(match.group(1)))
    return b'"application/pdf","' + base64.b64encode(pdf) + b'"'


def normalised(path: Path) -> bytes:
    """The bytes of ``path`` with the stamps of its kind of file removed."""
    data = path.read_bytes()
    if path.suffix == ".pdf":
        return PDF_STAMP.sub(b"", data)
    if path.suffix == ".html":
        return EMBEDDED_PDF.sub(_without_pdf_stamp, PAGE_STAMP.sub(b"", data))
    if path.name == "benchmark_summary.txt":
        return REPLAY_TIME.sub(rb"\1 <time>)", data)
    if path.name == "run.txt":
        lines = data.split(b"\n")
        return b"\n".join(line for line in lines if not line.startswith(RUN_STAMPS))
    return data


def compared_files(root: Path) -> set[Path]:
    """Every file under ``root`` that is compared, relative to ``root``."""
    return {
        path.relative_to(root)
        for path in root.rglob("*")
        if path.is_file() and path.suffix not in SKIPPED_SUFFIXES and path.name not in SKIPPED_NAMES
    }


def differences(before: Path, after: Path) -> list[str]:
    """One line per file found in one tree only, or differing after normalisation."""
    old, new = compared_files(before), compared_files(after)
    lines = [f"only in {before}: {path}" for path in sorted(old - new)]
    lines += [f"only in {after}: {path}" for path in sorted(new - old)]
    lines += [
        f"differs: {path}"
        for path in sorted(old & new)
        if normalised(before / path) != normalised(after / path)
    ]
    return lines


def compare(before: Path, after: Path) -> int:
    """Print every difference and their count; 1 when there is any, else 0."""
    for root in (before, after):
        if not root.is_dir():
            raise FileNotFoundError(f"{root} is not a folder of reference outputs")
    found = differences(before, after)
    for line in found:
        print(line)
    print(f"{len(found)} differences")
    return 1 if found else 0
