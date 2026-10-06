"""The results of ``report`` and ``optimise`` (private; re-exported by ``retrospective``):
``Report`` and ``Plan``, each with its ``summary``, and the questions each answers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .._caveats import caveat_lines
from .._style import check_detail, headline, wrap
from ..retrospective.calibration import calibration_short_parts
from ..retrospective.report import LIMITS_TITLE
from ..retrospective.report_status import calibration_status
from ..sampling_unit import FeatureDiscovery, FeatureRanking
from ._document import plan_settings, report_settings
from .setup import Setup

PLAN_TITLE = "Plan for the next experiment"
CALIBRATION_TITLE = (
    "Would a simple rule have been sufficient, and how reliable is each stopping rule?"
)
# The rule check as the "At a glance" block names it.
CALIBRATION_SHORT = "Rules. How reliable was each stopping rule?"
REPORT_QUESTIONS = (
    "unit",
    "controls",
    "enough",
    "profile",
    "inspect",
    "conditions",
    "populations",
)
PLAN_QUESTIONS = ("next_time", "bought", "effort", "order")
# The random orders in which the report replays each rule for "Did I image enough?".
REPORT_ORDERS = 5


@dataclass(frozen=True)
class Report:
    """The REPORT answers: ``results`` by analysis, ``document`` the assembled report.

    ``not_assessed`` holds, by question, why an analysis was not run;
    ``settings`` the replay settings of the run.
    """

    results: dict[str, Any]
    document: Any
    setup: Setup
    not_assessed: dict[str, str] = field(default_factory=dict)
    settings: dict[str, Any] = field(default_factory=dict)

    def summary(self, detail: str = "short") -> str:
        """The report questions (``detail``: ``"short"`` or ``"full"``, as
        ``MicroscopistReport.summary``), then the settings of the run."""
        parts = [self.document.summary(REPORT_QUESTIONS, detail=detail)]
        if self.settings:
            parts.append(report_settings(self.settings))
        return "\n\n".join(parts)

    @property
    def ranking(self) -> FeatureRanking | None:
        """The ranking over whole technical repeats (question 9), when several features were
        declared."""
        return self.results.get("ranking")

    @property
    def discovery(self) -> FeatureDiscovery | None:
        """The features the cell-level gate ranks, the paired test beside it (question 9),
        when several features were declared; ``plots`` draws its best-ranked by default."""
        return self.results.get("discovery")


@dataclass(frozen=True)
class Plan:
    """The OPTIMISE answers: ``results`` by analysis, ``document`` the assembled report.

    ``not_assessed`` holds, by question, why an analysis was not run;
    ``settings`` the computational settings and presets of the run.
    """

    results: dict[str, Any]
    document: Any
    setup: Setup
    not_assessed: dict[str, str] = field(default_factory=dict)
    settings: dict[str, Any] = field(default_factory=dict)

    def summary(self, detail: str = "short") -> str:
        """The planning questions, with the calibration of each stopping rule (``detail``:
        ``"short"`` or ``"full"``, as ``MicroscopistReport.summary``), then the settings.

        The rule check ("Would a simple rule have been sufficient, and how reliable
        is each stopping rule?", P5 and P6) is a section of its own, before "What
        this dataset cannot tell me", with its status (``calibration_status``) in
        the "At a glance" block of the short form.
        """
        short = check_detail(detail) == "short"
        text = self.document.summary(PLAN_QUESTIONS, detail=detail)
        calibration = self.results.get("calibration")
        if calibration is not None:
            status, answer = calibration_status(calibration)
            if short:
                lines, cautions = calibration_short_parts(calibration, indent=2)
                body = "\n".join([*lines, *caveat_lines(cautions, indent=2)])
                glance = self.document.at_a_glance(PLAN_QUESTIONS)
                line = wrap(headline(status, CALIBRATION_SHORT, answer), "  ", hang=5)
                text = text.replace(glance, f"{glance}\n{line}", 1)
            else:
                full = calibration.summary(detail="full")
                body = "\n".join(f"  {row}" if row else row for row in full.splitlines())
            section = f"{headline(status, CALIBRATION_TITLE)}\n{body}"
            head, limits, tail = text.rpartition(f"\n\n{LIMITS_TITLE}\n")
            text = f"{head}\n\n{section}{limits}{tail}" if limits else f"{text}\n\n{section}"
        if self.settings:
            text = f"{text}\n\n{wrap(plan_settings(self.settings))}"
        return text


__all__ = [
    "CALIBRATION_SHORT",
    "CALIBRATION_TITLE",
    "PLAN_QUESTIONS",
    "PLAN_TITLE",
    "REPORT_ORDERS",
    "REPORT_QUESTIONS",
    "Plan",
    "Report",
]
