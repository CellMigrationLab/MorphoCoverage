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
from .controls import ControlResult, RepresentationControl, representation_control
from .criteria import OutcomeCriterion, Preservation
from .effort import EffortResult, design_effect, effort_projection, nested_variance_components
from .hierarchy import SamplingHierarchy
from .inspection import InspectionResult, Summary, inspect_fovs
from .protocol import ProtocolResult, protocol_reliability
from .reliability import ReliabilityResult, reliability_summary, sampling_reliability
from .replay import ReplayResult, default_orders, replay_acquisitions
from .overview import DatasetOverview, dataset_overview
from .report import MicroscopistReport, microscopist_report
from .spatial import (
    EfficiencyResult,
    dispersed_order,
    order_efficiency,
    redundancy_distance,
    stage_travel,
)

__all__ = [
    "DatasetOverview",
    "dataset_overview",
    "ALL_CRITERIA",
    "AdequacyResult",
    "ControlResult",
    "EfficiencyResult",
    "EffortResult",
    "InspectionResult",
    "MicroscopistReport",
    "OutcomeCriterion",
    "Preservation",
    "ProtocolResult",
    "ReliabilityResult",
    "ReplayResult",
    "RepresentationControl",
    "SamplingHierarchy",
    "StoppingCalibration",
    "Summary",
    "adequacy_points",
    "calibrate_next_fov_novelty",
    "default_orders",
    "design_effect",
    "dispersed_order",
    "effort_projection",
    "first_apparent_adequacy",
    "first_realized_adequacy",
    "inspect_fovs",
    "microscopist_report",
    "nested_variance_components",
    "order_efficiency",
    "protocol_reliability",
    "redundancy_distance",
    "reliability_summary",
    "replay_acquisitions",
    "representation_control",
    "sampling_reliability",
    "stage_travel",
    "stopping_calibration",
    "sustained_apparent_adequacy",
    "sustained_realized_adequacy",
]
