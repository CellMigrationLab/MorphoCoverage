"""The shape of the ``experiment`` block of MorphoFeature's ``coverage.json`` (MC-2, private).

MorphoFeature's MF-5b writes the block as the last key of each exported ``coverage.json``
(``schema_version`` 3). Its format is fixed by the design note
``docs/restart/designs/coverage-experiment-block-2026-11-12.md`` (MorphoFeature, branch
``claude/restart-doc``), section 2: exactly these keys at every level, none added and none
left out (a value is ``null`` instead), every SHA-256 and digest 64 lower-case hex digits,
every condition ID of ``references``, ``repeat_groups`` and ``suggestions`` a key of
``conditions``, and every ``references`` value a key of ``conditions`` or ``null``.

``parsed_block`` reads a block or refuses it (``DescriptionRefused``, note section 4): an
``description_schema_version`` other than 1 (``DESCRIPTION_VERSION_UNKNOWN``), and anything
else that departs from the format (``MALFORMED``), naming where. Strings are kept exactly as
written: nothing is trimmed, case-folded or normalised.

Shared-private: used by ``described``.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from ._checks import is_integer

# Why a coverage.json or its experiment block is refused, by code (``DescriptionRefused.code``).
COVERAGE_VERSION_UNKNOWN = "coverage_version_unknown"
DESCRIPTION_VERSION_UNKNOWN = "description_version_unknown"
NOT_EXPORTED = "coverage_set_not_exported"
MALFORMED = "experiment_block_malformed"
MISSING = "experiment_block_missing"
DISAGREES = "experiment_block_disagrees_with_fovs"
TWO_DESCRIPTIONS = "experiment_descriptions_differ"
UNCONFIRMED = "suggestion_not_confirmed"

# The versions this MorphoCoverage reads: coverage.json up to 3 (MF-5b), descriptions 1.
COVERAGE_VERSION = 3
DESCRIPTION_VERSION = 1

BLOCK_KEYS = (
    "description_sha256",
    "description_schema_version",
    "section_digests",
    "conditions",
    "references",
    "repeat_groups",
    "suggestions",
)
DIGEST_KEYS = ("conditions", "repeats")
SUGGESTION_KEYS = ("target", "treatment_key")
_HEX64 = re.compile(r"[0-9a-f]{64}")


class DescriptionRefused(ValueError):
    """The experiment description of an export cannot be read or does not fit the tables;
    ``code`` names the reason and the message says what to fix."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


def _malformed(where: str, what: str, source: str) -> DescriptionRefused:
    return DescriptionRefused(
        MALFORMED,
        f"{source}: the experiment block is malformed: {where} {what}; the block is read "
        "exactly as MorphoFeature writes it (coverage.json schema_version 3), never in part: "
        "re-export the set with MorphoFeature",
    )


def _object(value: Any, keys: tuple[str, ...] | None, where: str, source: str) -> dict:
    """``value`` as a JSON object (with exactly ``keys`` when given), else refused."""
    if not isinstance(value, Mapping):
        raise _malformed(where, f"must be a JSON object, not {type(value).__name__}", source)
    if keys is not None and set(value) != set(keys):
        missing = [k for k in keys if k not in value]
        extra = sorted(str(k) for k in value if k not in keys)
        found = "; ".join(
            part
            for part in (
                f"missing {missing}" if missing else "",
                f"not part of the format {extra}" if extra else "",
            )
            if part
        )
        raise _malformed(where, f"must hold exactly the keys {list(keys)} ({found})", source)
    return dict(value)


def _text(value: Any, where: str, source: str, *, empty: bool = True) -> str | None:
    """A string, or ``None`` when ``empty`` allows it; anything else refused."""
    if value is None and empty:
        return None
    if not isinstance(value, str):
        kind = "a string or null" if empty else "a string"
        raise _malformed(where, f"must be {kind}, not {value!r}", source)
    return value


def _hex(value: Any, where: str, source: str) -> str:
    if not isinstance(value, str) or not _HEX64.fullmatch(value):
        raise _malformed(where, f"must be 64 lower-case hex digits, not {value!r}", source)
    return value


def _ids(table: Mapping, conditions: Mapping, where: str, source: str) -> None:
    """Refuse a condition ID of ``table`` that is not a key of ``conditions``."""
    unknown = [str(c) for c in table if c not in conditions]
    if unknown:
        raise _malformed(
            where, f"names {unknown}, which are not keys of experiment.conditions", source
        )


def _every_condition(table: Mapping, conditions: Mapping, where: str, source: str) -> None:
    """Refuse ``table`` unless its keys are exactly the conditions."""
    _ids(table, conditions, where, source)
    missing = [str(c) for c in conditions if c not in table]
    if missing:
        raise _malformed(where, f"has no entry for the conditions {missing}", source)


def _version(block: Mapping, source: str) -> int:
    """The block's ``description_schema_version``, refused unless it is 1."""
    version = block.get("description_schema_version")
    if not is_integer(version):
        raise _malformed(
            "experiment.description_schema_version", f"must be an integer, not {version!r}", source
        )
    if int(version) != DESCRIPTION_VERSION:
        raise DescriptionRefused(
            DESCRIPTION_VERSION_UNKNOWN,
            f"{source}: the experiment block comes from a description of schema_version "
            f"{version}; this MorphoCoverage reads version {DESCRIPTION_VERSION} only and never "
            "reads another version in part: update MorphoCoverage",
        )
    return int(version)


def _conditions(value: Any, source: str) -> dict[str, str]:
    conditions = _object(value, None, "experiment.conditions", source)
    folders = {}
    for cid, entry in conditions.items():
        where = f"experiment.conditions[{cid!r}]"
        entry = _object(entry, ("source_folder",), where, source)
        folders[_text(cid, where, source, empty=False)] = _text(
            entry["source_folder"], f"{where}.source_folder", source, empty=False
        )
    return folders


def _references(value: Any, conditions: Mapping, source: str) -> dict[str, str | None]:
    references = _object(value, None, "experiment.references", source)
    _every_condition(references, conditions, "experiment.references", source)
    for cid, reference in references.items():
        where = f"experiment.references[{cid!r}]"
        _text(reference, where, source)
        if reference is not None and reference not in conditions:
            raise _malformed(where, f"is {reference!r}, not a key of experiment.conditions", source)
        if reference == cid:
            raise _malformed(where, "names the condition itself as its reference", source)
    return {cid: references[cid] for cid in conditions}


def _repeat_groups(value: Any, conditions: Mapping, source: str) -> dict[str, dict[str, str]]:
    groups = _object(value, None, "experiment.repeat_groups", source)
    parsed: dict[str, dict[str, str]] = {}
    for rid, members in groups.items():
        where = f"experiment.repeat_groups[{rid!r}]"
        _text(rid, where, source, empty=False)
        members = _object(members, None, where, source)
        _ids(members, conditions, where, source)
        parsed[rid] = {
            cid: _text(
                _object(member, ("source_repeat",), f"{where}[{cid!r}]", source)["source_repeat"],
                f"{where}[{cid!r}].source_repeat",
                source,
                empty=False,
            )
            for cid, member in members.items()
        }
    return parsed


def _suggestions(
    value: Any, conditions: Mapping, source: str
) -> dict[str, tuple[str | None, str | None]]:
    suggestions = _object(value, None, "experiment.suggestions", source)
    _every_condition(suggestions, conditions, "experiment.suggestions", source)
    parsed = {}
    for cid in conditions:
        where = f"experiment.suggestions[{cid!r}]"
        entry = _object(suggestions[cid], SUGGESTION_KEYS, where, source)
        target, key = (_text(entry[k], f"{where}.{k}", source) for k in SUGGESTION_KEYS)
        parsed[cid] = (target, key)
    return parsed


def parsed_block(value: Any, source: str) -> dict[str, Any]:
    """The fields of an ``experiment`` block, checked (see the module), or refused."""
    if not isinstance(value, Mapping):
        raise _malformed("experiment", f"must be a JSON object, not {value!r}", source)
    version = _version(value, source)  # another version may have another shape
    block = _object(value, BLOCK_KEYS, "experiment", source)
    digests = _object(block["section_digests"], DIGEST_KEYS, "experiment.section_digests", source)
    conditions = _conditions(block["conditions"], source)
    return {
        "description_sha256": _hex(
            block["description_sha256"], "experiment.description_sha256", source
        ),
        "description_schema_version": version,
        "section_digests": {
            k: _hex(digests[k], f"experiment.section_digests.{k}", source) for k in DIGEST_KEYS
        },
        "conditions": conditions,
        "references": _references(block["references"], conditions, source),
        "repeat_groups": _repeat_groups(block["repeat_groups"], conditions, source),
        "suggestions": _suggestions(block["suggestions"], conditions, source),
    }


def checked_version(record: Mapping, source: str) -> int:
    """The record's ``schema_version``, refused when unknown (above ``COVERAGE_VERSION``, below
    1, or not an integer), and a record of an unexported set refused."""
    version = record.get("schema_version")
    if not is_integer(version) or not 1 <= int(version) <= COVERAGE_VERSION:
        raise DescriptionRefused(
            COVERAGE_VERSION_UNKNOWN,
            f"{source}: coverage.json schema_version {version!r} is not one this MorphoCoverage "
            f"reads (1 to {COVERAGE_VERSION}); it is never read in part: update MorphoCoverage",
        )
    if record.get("exported") is False:
        raise DescriptionRefused(
            NOT_EXPORTED,
            f"{source}: MorphoFeature did not export this set (exported: false), so it has no "
            "FOV table to analyse",
        )
    if int(version) < COVERAGE_VERSION and "experiment" in record:
        raise _malformed("experiment", f"is present in a schema_version {version} record", source)
    return int(version)


__all__ = [
    "BLOCK_KEYS",
    "COVERAGE_VERSION",
    "COVERAGE_VERSION_UNKNOWN",
    "DESCRIPTION_VERSION",
    "DESCRIPTION_VERSION_UNKNOWN",
    "DISAGREES",
    "MALFORMED",
    "MISSING",
    "NOT_EXPORTED",
    "TWO_DESCRIPTIONS",
    "UNCONFIRMED",
    "DescriptionRefused",
    "checked_version",
    "parsed_block",
]
