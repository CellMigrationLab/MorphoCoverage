"""Cells or fields? Checks of the sampling unit (section 4 of ``docs/questions.md``).

A microscope samples fields, not cells: every cell of a field enters
together. Cell-number rules, including the published JCB and JCS rules this
package reproduces, assume the cells they count are a random sample of the
well. These checks measure, on a completed acquisition, whether that
assumption holds and what it costs when it does not:

- ``field_variability``: how much fields differ, the design effect and the
  fields that differ more than random cells would (F1, F2);
- ``cell_rule_check`` and ``descriptor_spread_check``: the JCB rule and the
  JCS Fig. 2A procedure, each run with random cells and with whole fields (F3);
- ``unusual_cells`` and ``field_clustering``: out-of-distribution cells
  against a declared reference, and whether any labelled state, rare ones
  included, is spread over the fields or clustered in a few (F4, F5).

All are retrospective (they use every acquired field) and group-local: each
analysis group is assessed on its own data, except that ``cell_rule_check``
may normalize by declared control groups and ``unusual_cells`` scores each
group against declared reference groups other than itself.
"""

from .cell_rules import (
    DESCRIPTORS,
    CellRuleCheck,
    DescriptorSpreadCheck,
    cell_rule_check,
    descriptor_spread_check,
)
from .rare_fields import FieldClustering, UnusualCells, field_clustering, unusual_cells
from .variability import FieldVariability, field_variability

__all__ = [
    "DESCRIPTORS",
    "CellRuleCheck",
    "DescriptorSpreadCheck",
    "FieldClustering",
    "FieldVariability",
    "UnusualCells",
    "cell_rule_check",
    "descriptor_spread_check",
    "field_clustering",
    "field_variability",
    "unusual_cells",
]
