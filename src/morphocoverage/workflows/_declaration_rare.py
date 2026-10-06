"""The rare phenotype's sentences of a ``Setup``'s declaration (private; split out of
``_declaration`` to keep it under 600 lines, no behaviour change).

Shared-private: used by ``_declaration``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .._caveats import assumption
from .._wording import count, rate
from ..statistics import fovs_to_detect

if TYPE_CHECKING:
    from .setup import Setup


def rare_line(setup: Setup) -> str:
    """The rare phenotype declared, in the full declaration's sentences."""
    target = setup.rare
    assert target is not None
    parts = []
    if target.object_prevalence is not None:
        parts.append(f"{rate(target.object_prevalence)} of {setup.unit_words}")
    if target.fov_incidence is not None:
        parts.append(f"{rate(target.fov_incidence)} of fields")
    fields = target.min_fov_occurrences
    line = (
        f"Rare phenotype: {target.phenotype}, present in {' and '.join(parts)}, to be seen "
        f"in at least {count(fields, 'field')} with "
        f"{rate(target.detection_probability)} probability."
    )
    if target.fov_incidence is not None:
        needed = fovs_to_detect(target.fov_incidence, target.detection_probability, fields)
        line += (
            f" At that field incidence the binomial plan asks for {needed} independent "
            "representative fields; fields that cluster the state need more."
        )
    elif setup.rare_label is not None:
        line += (
            " The fields to see it are counted from the share of fields holding it in the "
            "data (“Did I image enough?”); object prevalence is never converted to "
            "field incidence."
        )
    else:
        line += (
            " Without a field incidence or a label, no field count is planned: object "
            "prevalence is never converted to field incidence."
        )
    if setup.rare_label is not None:
        line += (
            f" Judged with the label column {setup.rare_label!r}: in the report from the "
            "fields acquired (“Did I image enough?”), on held-back fields by optimise "
            "and benchmark; the report also shows its clustering in few fields."
        )
    else:
        line += " No label column: stated and planned for, not judged on held-back fields."
    return line


def rare_short(setup: Setup) -> tuple[str | None, list[str]]:
    """The rare phenotype in one line, and its caveats."""
    target = setup.rare
    if target is None:
        # An option left at its default is a standing note, not a caution about the data.
        return None, [
            assumption(
                "no rare phenotype declared (rare), so no answer says whether a rare state was "
                "sampled enough"
            )
        ]
    parts = []
    if target.object_prevalence is not None:
        parts.append(f"{rate(target.object_prevalence)} of {setup.unit_words}")
    if target.fov_incidence is not None:
        parts.append(f"{rate(target.fov_incidence)} of fields")
    line = (
        f"Rare phenotype: {target.phenotype} in {' and '.join(parts)}, to be seen in at least "
        f"{count(target.min_fov_occurrences, 'field')} with "
        f"{rate(target.detection_probability)} probability"
    )
    cautions = []
    if target.fov_incidence is not None:
        needed = fovs_to_detect(
            target.fov_incidence, target.detection_probability, target.min_fov_occurrences
        )
        line += f"; the binomial plan asks for {needed} representative fields"
    elif setup.rare_label is not None:
        line += "; the fields to see it are counted from the data"
    if setup.rare_label is not None:
        line += f"; judged on held-back fields with the label {setup.rare_label!r}"
    else:
        cautions.append(
            "no rare_label column: the rare phenotype is not judged on held-back fields"
        )
    return f"{line}.", cautions


__all__ = ["rare_line", "rare_short"]
