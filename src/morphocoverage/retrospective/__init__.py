"""Retrospective analyses of completed acquisitions: what the imaging achieved.

Separate from the prospective path (``Estimator -> Evidence -> DecisionPolicy``):
here, sets of FOVs are judged against FOVs no estimator saw, through
``OutcomeMetric`` values and declared ``OutcomeCriterion`` tolerances. See
``docs/proposals/microscopist_report_roadmap.md``.
"""

from .criteria import OutcomeCriterion, Preservation
from .hierarchy import SamplingHierarchy
from .reliability import ReliabilityResult, reliability_summary, sampling_reliability

__all__ = [
    "OutcomeCriterion",
    "Preservation",
    "ReliabilityResult",
    "SamplingHierarchy",
    "reliability_summary",
    "sampling_reliability",
]
