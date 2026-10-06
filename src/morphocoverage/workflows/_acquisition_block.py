"""The acquisition block of "What is in the dataset" (private; owner, 2026-10-06, plan v2).

Quality control on the raw values, reported apart from the biology: the fields far from the
other fields of their condition and repeat (``sampling_unit.acquisition``) and, with several
controls, the controls apart in intensity (``_acquisition_gap``, moved here from the checks
of the values so that the acquisition is said in one place). Neither is a biological claim;
biological unusualness reads values relative to each own control (``_unusual_values``).

Shared-private: used by ``workflows.retrospective``.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any

import pandas as pd

from .._style import Status, worst, wrap
from .._wording import count
from ..retrospective.report import MicroscopistReport
from ..sampling_unit.acquisition import TITLE, AcquisitionAnomalies, acquisition_anomalies
from ._acquisition_gap import NAME, gap_caution

if TYPE_CHECKING:
    from .setup import Setup

# The block's name in the text, the values it reads said once.
HEADING = f"{TITLE} (raw values, quality control)"


def acquisition_in_report(
    objects: pd.DataFrame, fovs: pd.DataFrame, setup: Setup
) -> AcquisitionAnomalies | None:
    """The acquisition anomalies on the declared explainable columns (the readout and the
    features), raw; ``None`` without any (an embedding-only report)."""
    if not setup.columns:
        return None
    return acquisition_anomalies(
        objects,
        fovs,
        feature_cols=list(setup.columns),
        fov_col=setup.fov_col,
        group_cols=setup.group_cols,
        observation_unit=setup.observation_unit,
    )


def gap_moved(results: dict[str, Any]) -> None:
    """The controls apart in intensity taken out of the checks of the values: the acquisition
    block says them (``with_acquisition``)."""
    overview = results.get("overview")
    if overview is None:
        return
    kept = tuple(c for c in overview.value_checks if not c.startswith(f"{NAME}:"))
    results["overview"] = replace(overview, value_checks=kept)


def with_acquisition(
    document: MicroscopistReport, anomalies: AcquisitionAnomalies | None, gaps: list[dict]
) -> MicroscopistReport:
    """The report with the acquisition block at the end of "What is in the dataset", short and
    full; ⚠️ when it finds a field or a gap (unchanged when neither ran)."""
    caution = gap_caution(gaps)
    if anomalies is None and caution is None:
        return document
    found = (anomalies is not None and anomalies.found > 0) or caution is not None
    lines, rows = [], []
    if anomalies is not None:
        judged = Status.OK if anomalies.judged else Status.NOT_ASSESSED
        marker = (Status.CAUTION if anomalies.found else judged).marker
        lines.append(f"{marker} {HEADING}: {anomalies.line()}.")
        if anomalies.found:
            # Only the table under its line: the full summary starts with that line, wrapped.
            line = anomalies.summary(detail="short")
            table = anomalies.summary(detail="full").removeprefix(line).strip("\n")
            rows = [f"  {row}" for row in table.splitlines()]
    if caution is not None:
        lines.append(f"{Status.CAUTION.marker} {NAME}: {caution}")
    short = "\n".join(wrap(line, 2, hang=5) for line in lines)
    full = "\n".join([wrap(lines[0], 2, hang=5), *rows, *(wrap(x, 2, hang=5) for x in lines[1:])])
    statuses, headlines = dict(document.statuses), dict(document.headlines)
    if found:
        statuses["overview"] = worst([statuses.get("overview", Status.OK), Status.CAUTION])
        named = []
        if anomalies is not None and anomalies.found:
            named.append(f"{count(anomalies.found, 'field')} far from their condition and repeat")
        if caution is not None:
            named.append("controls apart in intensity")
        parts = (headlines.get("overview", ""), f"acquisition: {', '.join(named)}")
        headlines["overview"] = "; ".join(p for p in parts if p)
    previous = document.short.get("overview", "")
    return replace(
        document,
        overview="\n\n".join(p for p in (document.overview, full) if p),
        short={**document.short, "overview": "\n".join(p for p in (previous, short) if p)},
        statuses=statuses,
        headlines=headlines,
    )


__all__ = ["HEADING", "acquisition_in_report", "gap_moved", "with_acquisition"]
