"""Why count fields, and not only objects? Checks of the sampling unit.

A microscope acquires phenotype observations through spatial fields, not as
a random sample of individual objects: every object of a FOV enters together.
Object-count rules, including the published JCB and JCS rules this package
reproduces, assume the objects they count are a random sample of the well.
These checks measure, on a completed acquisition, whether that assumption
holds and what it costs when it does not. They are the foundation of the
report, before its questions (``docs/questions.md``):

- ``fov_variability``: how much FOVs differ, the design effect and the FOVs
  that differ more than random objects would (Foundation);
- ``object_rule_check`` and ``descriptor_spread_check``: the JCB rule and the
  JCS Fig. 2A procedure, each run with random objects and with whole FOVs
  (Foundation; OPTIMISE, P5);
- ``unusual_objects`` and ``fov_clustering``: out-of-distribution objects
  against declared reference groups, and whether any labelled state, rare
  ones included, is spread over the FOVs or clustered in a few (REPORT, R4);
- ``condition_contrast``: each condition against the declared control, per
  repeat, with Cohen's d and the JCS effect size computed on objects (as the
  lab's Plot&Stats notebook does) and with the field as the unit, and a
  randomization test that shuffles condition labels over whole technical
  repeats (or a nested well column of ``group_cols``), over whole fields only
  when they are declared independent (``fields_independent``) (REPORT, R7);
- ``feature_ranking``: the same tests for every declared feature, gated by
  Benjamini-Hochberg q-values and ranked by |d| as fiberfeatures scores a
  feature set, to choose which features the report plots (REPORT, R7).

All are retrospective (they use every acquired FOV) and group-local: each
analysis group is assessed on its own data, except that ``object_rule_check``
may normalize by declared control groups and ``unusual_objects`` scores each
group against declared reference groups other than itself.
``retrospective.microscopist_report`` places their summaries in the report.
The results on the committed data, in plain language, are in
``docs/objects_or_fields.md``.
"""

from .clustering import FOVClustering, UnusualObjects, fov_clustering, unusual_objects
from .contrast import ConditionContrast, condition_contrast
from .discovery import FeatureDiscovery, feature_discovery
from .object_rules import (
    DESCRIPTORS,
    RANDOM_OBJECTS,
    WHOLE_FOVS,
    DescriptorSpreadCheck,
    ObjectRuleCheck,
    descriptor_spread_check,
    object_rule_check,
)
from .ranking import FeatureRanking, feature_ranking
from .variability import FOVVariability, fov_variability

__all__ = [
    "DESCRIPTORS",
    "RANDOM_OBJECTS",
    "WHOLE_FOVS",
    "ConditionContrast",
    "DescriptorSpreadCheck",
    "FOVClustering",
    "FOVVariability",
    "FeatureDiscovery",
    "FeatureRanking",
    "ObjectRuleCheck",
    "UnusualObjects",
    "condition_contrast",
    "descriptor_spread_check",
    "feature_discovery",
    "feature_ranking",
    "fov_clustering",
    "fov_variability",
    "object_rule_check",
    "unusual_objects",
]
