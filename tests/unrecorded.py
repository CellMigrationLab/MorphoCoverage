"""Test helper: results from separate fixtures, assembled for their wording only.

``microscopist_report`` refuses results that record different acquisitions (issue 277). Some
wording tests deliberately assemble results built on unrelated tables; they drop the record
so the assembly is not checked, which is what the report itself does for rebuilt results.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from morphocoverage._acquisition import ACQUISITION


def unrecorded(result: Any) -> Any:
    """``result`` without the acquisition identity it recorded."""
    if hasattr(result, "acquisition"):
        return dataclasses.replace(result, acquisition=None)
    settings = {k: v for k, v in result.settings.items() if k != ACQUISITION}
    return dataclasses.replace(result, settings=settings)
