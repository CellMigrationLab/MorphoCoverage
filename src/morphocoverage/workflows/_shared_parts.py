"""Steps of the report's analyses against the control and between conditions (private;
split out of ``workflows._shared``): the unusual observations and their clustering in fields,
and the feature discovery and ranking beside the readout's contrast.

Shared-private: used by ``workflows._shared`` (``control_checks``, ``conditions``).
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pandas as pd

from .._wording import plural
from ..sampling_unit import (
    FOVClustering,
    feature_discovery,
    feature_ranking,
    fov_clustering,
    unusual_objects,
)
from ._feature_space import unusual_space, with_table
from ._one_per_field import field_unit
from ._rare_share import share_column, with_share
from ._unusual_values import biological_values
from .setup import Setup, default_novelty


def unusual_result(
    objects: pd.DataFrame,
    fovs: pd.DataFrame,
    setup: Setup,
    values: pd.DataFrame | None,
    per_field: bool,
    raw_columns: list[str],
) -> tuple[Any, list[str]]:
    """The unusual observations against each own control, on ``values`` (default
    ``objects``) made relative to it in its repeat, with the space they were read in; and
    the columns of that space."""
    relative, how = biological_values(
        objects if values is None else values, fovs, setup, raw_columns
    )
    columns, weights, space = unusual_space(relative, fovs, setup)
    unusual = unusual_objects(
        relative,
        fovs,
        feature_cols=columns,
        reference=setup.reference_control,
        own=setup.own_references,  # several controls: each against its own
        random_state=setup.random_state,
        weights=weights,
        **setup.common,
    )
    noted = {"space": f"{space}, {how}", "exploratory": True}
    if not per_field:  # unusual images: each field holds one observation
        noted["words"] = plural(field_unit(setup) or setup.unit_word)
    return replace(unusual, settings={**unusual.settings, **noted}), columns


def unusual_clustering(
    objects: pd.DataFrame,
    fovs: pd.DataFrame,
    setup: Setup,
    unusual: Any,
    columns: list[str],
    label: str,
) -> FOVClustering:
    """Whether the unusual observations (labelled ``label``) cluster in few fields."""
    # The label flags a share of ordinary objects by construction, so a field holds
    # the state only with the novelty rule's m of them, and the control groups'
    # own incidence is printed beside it (P6-1).
    novelty = setup.novelty or default_novelty(columns)
    return fov_clustering(
        objects.assign(**{label: unusual.labels}),
        fovs,
        label_col=label,
        states=[True],
        random_state=setup.random_state,
        min_objects_per_fov=novelty.min_novel_observations,
        reference=setup.reference_control,
        **setup.common,
    )


def feature_results(
    objects: pd.DataFrame, fovs: pd.DataFrame, setup: Setup, contrast: Any, compared: list[str]
) -> dict[str, Any]:
    """The feature discovery and ranking over the ``compared`` features, and the readout's
    ``contrast`` in the discovery's correction family."""
    shared = with_share(objects, setup)
    discovery = feature_discovery(
        shared,
        fovs,
        features=compared,
        random_state=setup.random_state,
        n_jobs=setup.n_jobs,
        scales=setup.scales,
        families=setup.families(fovs),
        **setup.contrast_arguments,
        **setup.common,
    )
    # The headline counts the share apart from the features (report_status_conditions).
    discovery = replace(discovery, settings={**discovery.settings, "share": share_column(setup)})
    # Each feature's table named (owner, 2026-10-06).
    discovery = replace(
        discovery,
        tests=with_table(discovery.tests, setup),
        ranking=with_table(discovery.ranking, setup),
    )
    # One correction family on the page: the readout's tests take the discovery's.
    from ._families import shared_across, shared_family

    contrast = shared_family(contrast, discovery, setup.readout)
    # The secondary rows for every feature, each family over all of them.
    contrast = shared_across(contrast, shared, fovs, compared, setup)
    ranking = feature_ranking(
        shared,
        fovs,
        features=compared,
        random_state=setup.random_state,
        fields_independent=setup.fields_independent,
        **setup.contrast_arguments,
        **setup.common,
    )
    # The share counted apart (C20).
    ranking = replace(ranking, settings={**ranking.settings, "share": share_column(setup)})
    return {"discovery": discovery, "contrast": contrast, "ranking": ranking}


__all__ = ["feature_results", "unusual_clustering", "unusual_result"]
