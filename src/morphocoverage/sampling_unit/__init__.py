"""Can objects be counted, or must FOVs be? Checks of the sampling unit.

A microscope samples FOVs, not objects: every object of a FOV enters
together. Object-count rules, including the published JCB and JCS rules this
package reproduces, assume the objects they count are a random sample of the
well. These checks measure, on a completed acquisition, whether that
assumption holds and what it costs when it does not (``docs/questions.md``):

- ``fov_variability``: how much FOVs differ, the design effect and the FOVs
  that differ more than random objects would (REPORT, R0);
- ``object_rule_check`` and ``descriptor_spread_check``: the JCB rule and the
  JCS Fig. 2A procedure, each run with random objects and with whole FOVs
  (REPORT, R0; OPTIMISE, P5);
- ``unusual_objects`` and ``fov_clustering``: out-of-distribution objects
  against declared reference groups, and whether any labelled state, rare
  ones included, is spread over the FOVs or clustered in a few (REPORT, R4).

All are retrospective (they use every acquired FOV) and group-local: each
analysis group is assessed on its own data, except that ``object_rule_check``
may normalize by declared control groups and ``unusual_objects`` scores each
group against declared reference groups other than itself.
``retrospective.microscopist_report`` places their summaries in the report.
The results on the committed data, in plain language, are in
``docs/objects_or_fields.md``.
"""

from .clustering import FOVClustering, UnusualObjects, fov_clustering, unusual_objects
from .object_rules import (
    DESCRIPTORS,
    RANDOM_OBJECTS,
    WHOLE_FOVS,
    DescriptorSpreadCheck,
    ObjectRuleCheck,
    descriptor_spread_check,
    object_rule_check,
)
from .variability import FOVVariability, fov_variability

__all__ = [
    "DESCRIPTORS",
    "RANDOM_OBJECTS",
    "WHOLE_FOVS",
    "DescriptorSpreadCheck",
    "FOVClustering",
    "FOVVariability",
    "ObjectRuleCheck",
    "UnusualObjects",
    "descriptor_spread_check",
    "fov_clustering",
    "fov_variability",
    "object_rule_check",
    "unusual_objects",
]
