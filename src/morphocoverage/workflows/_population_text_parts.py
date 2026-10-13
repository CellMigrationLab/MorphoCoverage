"""Parts of question 10's text (private; ``_populations``, whose ``PopulationsResult`` they
read): how the groups were searched (the method), the answer when none could be or none was
found, the conditions not compared and the assumptions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .._style import Status, wrap
from .._wording import count, listing, percent
from ._population_context import EXPERIMENTS_DIFFER
from ._population_spaces import no_own_control_words, spaces_method

if TYPE_CHECKING:
    from .setup import Setup


def population_method(result: Any, setup: Setup) -> str:
    """How the groups were searched: the observations pooled, the clustering and the check,
    with the observations left out for want of their own control and the normalisation."""
    pops = result.populations
    settings = pops.settings
    units = setup.unit_words
    method = (
        f"{count(settings['observations'], setup.unit_word)} in all, every condition pooled, "
        f"clustered on the {count(len(settings['features']), 'declared feature')}"
        f" (HDBSCAN on a {settings['map_components']}-dimensional UMAP of their robust "
        f"standardisation; a cluster holds at least {settings['min_cells']} {units}), each "
        "cluster then judged on all the features"
    )
    if settings.get("spaces"):  # other feature tables: each space apart (owner, 2026-10-06)
        method = (
            f"Every condition pooled, clustered {spaces_method(pops, setup)}; in each, HDBSCAN "
            "on a UMAP of the robust standardisation, each cluster then judged on all its "
            f"features; groups found in two spaces are one when their {units} overlap with a "
            f"Jaccard index of {settings['same_group']:g} or more"
        )
    # Observations without their own control, named once, never a missing value (audit 8).
    left = no_own_control_words(settings, setup.unit_word, units)
    method += f" ({left})" if left else ""
    method += f", {EXPERIMENTS_DIFFER}" if result.normalised else ""
    return method


def no_group_answer(method: str, pops: Any, setup: Setup) -> tuple[Status, str, str] | None:
    """Question 10's status, headline and text when no group could be searched (a missing
    value in every observation, too few of them) or none passed the check; ``None`` when
    groups were found."""
    settings = pops.settings
    units = setup.unit_words
    if not settings.get("observations") and not settings.get("spaces"):
        text = wrap(
            f"{method}: not assessed: no {setup.unit_word} has a value of every feature "
            f"({count(settings['incomplete_observations'], setup.unit_word)} with a missing "
            "value).",
            2,
            hang=4,
        )
        return Status.NOT_ASSESSED, f"not assessed: every {setup.unit_word} misses a value", text
    if not settings.get("clustered", True):
        text = wrap(
            f"{method}: not clustered: too few {units} (or no feature that varies) for two "
            f"clusters of {settings['min_cells']}.",
            2,
            hang=4,
        )
        return Status.NOT_ASSESSED, f"not clustered: too few {units}", text
    if not pops.names:
        # Not finding a group says nothing of a continuum, nor of rare states (#94; owner,
        # 2026-10-05): ➖, a neutral answer.
        text = wrap(
            f"{method}: no candidate group passed the neighbours-inside check on these "
            f"features ({percent(settings['purity'])} or more of a cluster's {units}' "
            f"{settings['neighbours']} nearest neighbours in it).",
            2,
            hang=4,
        )
        headline = "no group: no candidate group passed the check on these features"
        return Status.NOT_ASSESSED, headline, text
    return None


def unpaired_lines(result: Any) -> list[str]:
    """The conditions not compared, imaged in no biological repeat of the control."""
    unpaired = result.enrichment.attrs.get("unpaired", []) if result.enrichment is not None else []
    if not unpaired:
        return []
    return [
        wrap(
            f"Not compared: {listing([str(c) for c in unpaired], 4)} (no biological repeat "
            "shared with the control).",
            4,
            hang=6,
        )
    ]


def population_assumptions(setup: Setup, settings: dict[str, Any]) -> str:
    """Question 10's assumptions, as a wrapped paragraph."""
    units = setup.unit_words
    return wrap(
        f"Assumptions: The {units} of every condition are clustered together"
        + (  # compared only where a control is declared (audit 10, C24)
            f", then each condition is compared with its own control on its share of "
            f"{units} in each, per biological repeat (paired t-test on the log odds of the "
            "repeat shares, Benjamini-Hochberg within each sub-experiment over every "
            "population and condition)"
            if setup.reference_control is not None
            else ", and no condition is compared, as no control is declared"
        )
        + "; the main population, holding most of them, is the rest, listed and "
        "not tested. A cluster on the UMAP is kept as a group when "
        f"{percent(settings['purity'])} or more of its {units}' "
        f"{settings['neighbours']} nearest neighbours on all the features are in it too "
        "(neighbours inside). Whether a group is a separate state or the end of a "
        f"continuum (the most extreme {units} of a gradient) is not decided from the "
        "numbers: no test of a gap told the two apart reliably at these sizes, so open "
        "the images of its fields to validate it. Its rule is the simplest one on the "
        "features (at most two thresholds, a decision tree) that picks out its "
        f"{units}, fitted on all of them; the share of them it finds and the share right "
        "of those it picks describe that procedure refitted with whole fields held out "
        f"({units} held out for a group in one field), not the exact thresholds shown. "
        "Groups that overlap strongly on every feature "
        "are not told apart. The clusters depend on the declared features and their "
        "standardisation; a group held by few fields, or most of whose "
        f"{units} are in one (in largest field), may be an artefact of those images.",
        2,
        hang=4,
    )
