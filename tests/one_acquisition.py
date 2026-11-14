"""Test helper: results from separate fixtures, assembled for their wording only.

``microscopist_report`` refuses a result that records no acquisition identity, and results
that record different ones (issue 277, Q-MC-a). Some wording tests deliberately assemble
results built on unrelated tables, or tables built by hand; they stamp every result with the
same declared test identity, so the assembly is checked and passes on purpose. Nothing in the
package does this: a result's identity is always the one its analysis function recorded.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from morphocoverage._acquisition import ACQUISITION

# Not a SHA-256: it can never equal an identity the package computes.
TEST_ACQUISITION = "test-acquisition"


def stamped(result: Any, identity: str = TEST_ACQUISITION) -> Any:
    """``result`` recording ``identity`` as its acquisition instead of its own."""
    if hasattr(result, "acquisition") and not hasattr(result, "settings"):
        return dataclasses.replace(result, acquisition=identity)
    return dataclasses.replace(result, settings={**result.settings, ACQUISITION: identity})
