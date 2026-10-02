"""What each question of the report needs when it is not assessed (private; ``report``):
in a few words for its "At a glance" line, and in a sentence for its section."""

# What a question not assessed needs, in a few words for its "At a glance" line.
_SHORT_NEEDS = {
    "unit": "needs fov_variability",
    "controls": "needs a positive control: a condition known to change the phenotype",
    "enough": "needs sufficiency_report or a stopping calibration",
    "profile": "needs the controls, variability, clustering or effort analyses",
    "next_time": "needs sampling_reliability and protocol_reliability",
    "bought": "needs adequacy_points on a replay",
    "inspect": "needs inspect_fovs",
    "effort": "needs technical repeats or a sampling hierarchy",
    "order": "needs stage coordinates and an order replay",
    "conditions": "needs a control condition and condition_contrast",
    "populations": "needs at least two declared features",
}

_NEEDS = {
    "unit": "run fov_variability (and object_rule_check or descriptor_spread_check) "
    "from morphocoverage.sampling_unit",
    "controls": "name a condition known to change the phenotype as the positive control (the "
    "guided session asks for it; in Python, Setup(positive=PositiveControl(...))) and run the "
    "report again; it can be added after the acquisition, whenever such a condition was "
    "imaged",
    "enough": "run sufficiency_report, or stopping_calibration on a replay",
    "profile": "run representation_control, fov_variability, fov_clustering, "
    "effort_projection, redundancy_distance or inspect_fovs; the profile lists what they measured",
    "next_time": "run sampling_reliability, then protocol_reliability across samples",
    "bought": "run adequacy_points on a replay",
    "inspect": "run inspect_fovs",
    "effort": "declare a sampling hierarchy, or give the FOV table a technical_repeat column "
    "(a well, a dish) kept among the group columns, and run optimise (effort_projection)",
    "order": "give stage coordinates and run order_efficiency on a replay with dispersed orders",
    "conditions": "declare a control condition and run condition_contrast on data with more "
    "than one condition, with analysis groups that keep conditions and repeats apart",
    "populations": "declare at least two features: the populations are found on all of them",
}

__all__ = ["_NEEDS", "_SHORT_NEEDS"]
