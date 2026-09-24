"""Retrospective analyses of completed acquisitions: what the imaging achieved.

Separate from the prospective path (``Estimator -> Evidence -> DecisionPolicy``):
here, sets of FOVs are judged against FOVs no estimator saw, through
``OutcomeMetric`` values and declared ``OutcomeCriterion`` tolerances. See
``docs/proposals/microscopist_report_roadmap.md``.
"""

from .adequacy import (
    ALL_CRITERIA,
    AdequacyResult,
    adequacy_points,
    first_apparent_adequacy,
    first_realized_adequacy,
    sustained_apparent_adequacy,
    sustained_realized_adequacy,
)
from .calibration import StoppingCalibration, calibrate_next_fov_novelty, stopping_calibration
from .criteria import OutcomeCriterion, Preservation
from .hierarchy import SamplingHierarchy
from .reliability import ReliabilityResult, reliability_summary, sampling_reliability
from .replay import ReplayResult, default_orders, replay_acquisitions

__all__ = [
    "ALL_CRITERIA",
    "AdequacyResult",
    "OutcomeCriterion",
    "Preservation",
    "ReliabilityResult",
    "ReplayResult",
    "SamplingHierarchy",
    "StoppingCalibration",
    "adequacy_points",
    "calibrate_next_fov_novelty",
    "default_orders",
    "first_apparent_adequacy",
    "first_realized_adequacy",
    "reliability_summary",
    "replay_acquisitions",
    "sampling_reliability",
    "stopping_calibration",
    "sustained_apparent_adequacy",
    "sustained_realized_adequacy",
]
