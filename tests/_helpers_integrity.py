"""Copied, replaced and checked fields, shared by the integrity tests."""

import pandas as pd

from morphocoverage import integrity_checks

FEATURES = [
    "Nuclei_Intensity_IntegratedIntensity_YAPTAZ",
    "Nuclei_Intensity_MeanIntensity_YAPTAZ",
    "Cytoplasm_Intensity_MeanIntensity_YAPTAZ",
]


def _field(objects, fovs, i):
    row = fovs.iloc[i]
    mask = pd.Series(True, index=objects.index)
    for column in ("condition", "biological_repeat", "technical_repeat", "image_file"):
        mask &= objects[column].eq(row[column])
    return mask


def _replace_field(objects, fovs, target, rows):
    """``objects`` with field ``target``'s observations replaced by ``rows``."""
    keep = objects[~_field(objects, fovs, target)]
    identity = fovs.iloc[target][
        ["condition", "biological_repeat", "technical_repeat", "image_file"]
    ]
    return pd.concat([keep, rows.assign(**identity.to_dict())], ignore_index=True)


def _check(objects, fovs):
    return integrity_checks(objects, fovs, feature_cols=FEATURES, observation_unit="nucleus")
