"""The benchmark protocol, fixed before the benchmark is run (``docs/questions.md``).

Everything that decides a result is declared here, before any score exists:
the datasets and their roles, the questions and their criteria, each rule
family with its threshold grid, the risk levels, the replay design and the
independent unit. ``benchmark.py`` refuses to run when this protocol does not
match the digest locked for its version in ``benchmark_protocol.lock`` (one
``version digest`` line per version, written with
``python data/crick_idr/benchmark.py --lock`` before that version's first
run). Changing a setting afterwards needs a new version and a new lock,
recorded in the findings, so no threshold is chosen after seeing which rule
wins.

Versions:

- 1 (locked 2026-09-25, before the first run; ``docs/preliminary_findings.md``
  entry 8).
- 2 (2026-09-25): names only. The families "JCB per cell" and "JCS spread per
  cell" became "JCB per object" and "JCS spread per object", and the rare
  question's ``planted_cells`` became ``planted_objects``, when the package
  renamed cells to objects. Every setting, grid and seed is that of version 1.
  (2026-09-26, no new version: ``benchmark.py`` runs the "novelty bound"
  family under the distribution objective instead of the population
  objective, since novelty became distribution evidence when that component
  was added; the objective is not part of the protocol digest, and the
  frontier reads each rule's own verdict (``StrategyResult.rule_decision``),
  so the scores are unchanged.)
- 3 (2026-09-26, second audit, ``docs/audit/function_audit_2_2026-09-26.md``): the
  replay changed under the settings, not the settings. Permuted orders are
  drawn per analysis group, from the group's FOVs in key order, seeded from
  ``random_state``, the group identity, the order and the split
  (``benchmark.permuted_fovs``); version 2 permuted the whole table, so a
  group's order depended on the other groups and on row order. Replay
  prefixes are in acquisition order (field by field, table order within a
  field), as a live session receives them. The key ``unit_cols`` names the
  well ``technical_repeat`` (the column's new name; version 2's digest was
  taken when it was ``well``, which is why version 2 no longer matched its
  lock). Every threshold, grid, split count and seed is that of version 2.

Questions scored (codes of ``docs/questions.md``):

- **mean** (R2 average, P5, P6; stage 1): the well mean of the YAP/TAZ ratio
  within 5% and within 2.5% of the held-back fields. Families: the JCB rule
  (averaging CV, upper-95 SD) per object and per field, the JCS descriptor
  spread per object and per field, and the FOV bootstrap of the mean. The JCS
  plateau rule is left out: it orders visible fields by the recorded
  acquisition-order column, which a permuted replay does not rewrite.
- **rare** (R2 rare, P5): whether the acquisition, ended at the stop, saw a
  planted rare state that the held-back fields show. Semi-synthetic: 5 siYAP
  nuclei added to a share of the fields of a MOCK well (as in the frequency
  control, ``docs/preliminary_findings.md`` entry 3). Families: the support
  novelty bound (the report's definition of new) and the beta-binomial rare
  incidence rule, which reads the planted label and so stands for "when
  labels exist".

Threshold grids follow one rule, fixed before any score: each grid reaches
from a setting that never stops within the stream to one that stops at the
first allowed field, so every family can be compared with a fixed count at
every effort. A reduced dry run (one split, one order, not scored) showed
the first grids ending before the cheap end, and they were widened before
the lock; no setting was chosen on a result.

The independent unit is the well for idr0028 (12 wells; only 3 plates, so
plate-level claims are not made), the well for idr0139 (10 wells, one plate)
and the host well for the planted states (2 wells, so their interval is
indicative only).
"""

from __future__ import annotations

import hashlib
import json

PROTOCOL = {
    "version": 3,
    "order_permutation": "per analysis group, FOVs in key order, seeded from random_state",
    "prefix_order": "acquisition order",
    "alphas": [0.05, 0.10],
    "n_boot": 2000,
    "random_state": 0,
    "datasets": {
        "idr0028_screenB": {
            "role": "development",
            "reservoir_fraction": 0.35,
            "n_splits": 5,
            "permuted_orders": 10,
            "recorded_order": True,
            "unit_cols": ["repeat", "technical_repeat"],
        },
        "idr0139": {
            "role": "held_out",
            "reservoir_fraction": 0.5,
            "n_splits": 6,
            "permuted_orders": 5,
            "recorded_order": True,
            "unit_cols": ["repeat", "technical_repeat"],
        },
    },
    "questions": {
        "mean": {
            "datasets": ["idr0028_screenB", "idr0139"],
            "criteria": [
                {"tolerance": 0.05, "label": "the well mean within 5%"},
                {"tolerance": 0.025, "label": "the well mean within 2.5%"},
            ],
            "families": {
                "JCB per object": {
                    "kind": "scalar_precision",
                    "resampling_unit": "object",
                    "quantity": "averaging_cv",
                    "thresholds": [0.0025, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.4],
                },
                "JCB per field": {
                    "kind": "scalar_precision",
                    "resampling_unit": "fov",
                    "quantity": "averaging_cv",
                    "thresholds": [0.0025, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.4],
                },
                "JCS spread per object": {
                    "kind": "descriptor_spread",
                    "resampling_unit": "object",
                    "quantity": "max_descriptor_spread",
                    "thresholds": [0.001, 0.002, 0.005, 0.01, 0.02, 0.04, 0.08, 0.16],
                },
                "JCS spread per field": {
                    "kind": "descriptor_spread",
                    "resampling_unit": "fov",
                    "quantity": "max_descriptor_spread",
                    "thresholds": [0.001, 0.002, 0.005, 0.01, 0.02, 0.04, 0.08, 0.16],
                },
                "FOV bootstrap of the mean": {
                    "kind": "fov_bootstrap_centroid",
                    "quantity": "fov_bootstrap_centroid_max_se",
                    "thresholds": [0.0005, 0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064],
                },
            },
            "n_resamples": 200,
            "min_fovs": 1,
        },
        "rare": {
            "datasets": ["idr0028_screenB"],
            "design": [["1A", "B6", "M23"], ["2A", "K16", "N23"]],
            "frequencies": [0.1, 0.2],
            "layouts": ["spread", "clustered"],
            "replicates": 6,
            "planted_objects": 5,
            "reservoir_fraction": 0.35,
            "n_splits": 3,
            "permuted_orders": 3,
            "recorded_order": True,
            "unit_cols": ["plate"],
            "criterion": {"label": "the planted state seen, when the held-back fields show it"},
            "families": {
                "novelty bound": {
                    "kind": "support_novelty",
                    "quantity": "lofo_novel_fov_upper",
                    "thresholds": [0.1, 0.15, 0.2, 0.3, 0.45, 0.6],
                },
                "rare incidence (labels)": {
                    "kind": "rare_incidence",
                    "quantity": "rare_incidence_risk",
                    "thresholds": [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5],
                    "target_incidence": 0.1,
                },
            },
            "min_fovs": 2,
        },
    },
}


def digest(protocol: dict = PROTOCOL) -> str:
    """SHA-256 of the protocol as canonical JSON: what the lock file records."""
    text = json.dumps(protocol, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()


__all__ = ["PROTOCOL", "digest"]
