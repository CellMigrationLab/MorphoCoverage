"""Cells or fields? The sampling-unit checks of the package, on both committed datasets.

Runs section 4 of ``docs/questions.md`` (``morphocoverage.sampling_unit``) on
idr0139 (the JCS case study; fascin ratio; control Untreated) and idr0028
screen B (YAP/TAZ ratio; control MOCK), one analysis group per well:

- F1, F2: ``field_variability``, how much fields differ and the design effect;
- F3: ``cell_rule_check``, the JCB rule at its own 5% and 2.5% targets with
  whole fields, and ``descriptor_spread_check``, the JCS Fig. 2A procedure
  with random cells and with whole fields, at the paper's threshold (250
  cells for idr0139, Fig. 2; 200 for idr0028, Fig. S2);
- F4, F5: ``unusual_cells`` against the control wells on every intensity
  feature, and ``field_clustering`` of those unusual cells.

The committed tables hold every imaged field of each well (4 per idr0139
well, 28 per idr0028 well) and reproduce JCS Table 1 for idr0139. The
paper's treatment comparisons with whole fields, which compare wells, are in
``jcs_effect_fields.py``. Retrospective and exploratory. Usage::

    python data/crick_idr/cells_vs_fields.py

Tables are written to ``results/cells_vs_fields_*.csv`` and the summary quoted
in ``docs/preliminary_findings.md`` (entries 9 and 10) to
``results/cells_vs_fields_summary.txt``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from datasets import FASCIN_RATIO, load_idr0139
from idr0028_data import RATIO as YAPTAZ_RATIO
from idr0028_data import load as load_idr0028

from morphocoverage.sampling_unit import (
    cell_rule_check,
    descriptor_spread_check,
    field_clustering,
    field_variability,
    unusual_cells,
)

RESULTS = Path(__file__).resolve().parent / "results"
GROUPS = ["condition", "repeat", "well"]
DATASETS = {
    # name: (loader, metric, control condition, JCS threshold in cells)
    "idr0139": (load_idr0139, FASCIN_RATIO, "Untreated", 250),
    "idr0028": (load_idr0028, YAPTAZ_RATIO, "MOCK", 200),
}
COMMON = {"group_cols": GROUPS, "observation_unit": "cell"}


def run(name: str) -> dict[str, pd.DataFrame]:
    load, metric, control, threshold = DATASETS[name]
    objects, fovs, features = load()
    variability = field_variability(objects, fovs, feature=metric, **COMMON)
    rule = cell_rule_check(objects, fovs, feature=metric, control={"condition": control}, **COMMON)
    spread = descriptor_spread_check(
        objects, fovs, feature=metric, threshold=threshold, sizes=range(10, 501, 10), **COMMON
    )
    unusual = unusual_cells(
        objects, fovs, feature_cols=features, reference={"condition": control}, **COMMON
    )
    clustering = field_clustering(
        objects.assign(unusual=unusual.labels), fovs, label_col="unusual", states=[True], **COMMON
    )
    tag = {"dataset": name}
    return {
        "variability": variability.groups.assign(**tag),
        "jcb": rule.table.assign(**tag),
        "jcs": spread.at_threshold.assign(**tag),
        "jcs_curves": spread.curves.assign(**tag),
        "unusual": unusual.groups.drop(columns="standardize_state").assign(**tag),
        "rare": clustering.table.assign(**tag),
    }


def _fields(count: float) -> str:
    return "1 field" if count == 1 else f"{count:.0f} fields"


def _range(values: pd.Series, fmt: str) -> str:
    return f"median {values.median():{fmt}} (range {values.min():{fmt}}-{values.max():{fmt}})"


def summarize(tables: dict[str, pd.DataFrame]) -> str:
    lines = ["Cells or fields: the sampling-unit checks, one group per well", ""]
    for name, (_, metric, control, threshold) in DATASETS.items():
        v = tables["variability"].query("dataset == @name")
        j = tables["jcb"].query("dataset == @name")
        s = tables["jcs"].query("dataset == @name")
        u = tables["unusual"].query("dataset == @name")
        r = tables["rare"].query("dataset == @name")
        lines += [f"== {name} ({metric}; control {control}; {len(v)} wells) ==", ""]
        lines.append(
            f"F1-F2 field variability: between-field share of variance {_range(v.icc, '.1%')}; "
            f"design effect {_range(v.design_effect, '.1f')}; error ratio "
            f"{_range(v.error_ratio, '.1f')}; fields differing more than random cells "
            f"{_range(v.unusual_field_share, '.0%')} (5% expected)."
        )
        first = j[j.target == j.target.max()]
        lines.append(
            f"F3 JCB: cell SD {_range(first.cell_sd, '.1%')} of the control mean; for the "
            f"cells obtained, real / predicted error {_range(first.error_ratio, '.1f')}."
        )
        for target, t in j.groupby("target", sort=False):
            lines.append(
                f"  target {target:.1%}: rule cells {_range(t.rule_cells, '.0f')}, fields used "
                f"{_range(t.fields_used, '.1f')}; real error {_range(t.real_error, '.1%')}; "
                f"{int((~t.meets_target).sum())} of {len(t)} wells miss; with fields as the "
                f"unit the rule asks {_fields(t.field_rule_fovs.median())} (upper-95 "
                f"{t.field_rule_fovs_upper95.median():.0f}) of {int(t.fovs.iloc[0])} acquired."
            )
        lines.append(
            f"F3 JCS Fig. 2A at {threshold} cells, whole-field spread / random-cell spread:"
        )
        for descriptor, t in s.groupby("descriptor", sort=False):
            matched = t.fields_cells_to_match.dropna()
            lines.append(
                f"  {descriptor}: {_range(t.ratio, '.1f')}; fields matched the {threshold}-cell "
                f"spread in {len(matched)} of {len(t)} wells"
                + (f", from a median of {matched.median():.0f} cells" if len(matched) else "")
                + " (up to 500 tried)."
            )
        ok = u[u.status == "ok"]
        seen = r[r.state_cells > 0]
        clustered = seen[seen.p_clustered <= 0.05]
        lines.append(
            f"F4-F5 unusual cells (all intensity features, against the other {control} wells, "
            f"1% reference quantile): {_range(ok.unusual_share, '.1%')} of cells per well; "
            f"seed agreement {_range(ok.seed_agreement, '.0%')}."
        )
        lines.append(
            f"  seen in {len(seen)} of {len(r)} wells, clustered by field (p <= 0.05) in "
            f"{len(clustered)}; clustering ratio {_range(seen.clustering_ratio, '.1f')}; fields "
            f"to see them with 95% chance {_range(seen.fovs_needed, '.0f')} from field "
            f"incidence, against {_range(seen.fovs_needed_if_random, '.0f')} if cells were "
            f"random."
        )
        if len(clustered):
            factor = clustered.fovs_needed / clustered.fovs_needed_if_random
            lines.append(
                f"  in the clustered wells, fields needed from field incidence / if cells were "
                f"random: {_range(factor, '.1f')}."
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    runs = [run(name) for name in DATASETS]
    tables = {key: pd.concat([r[key] for r in runs], ignore_index=True) for key in runs[0]}
    for key, table in tables.items():
        table.to_csv(RESULTS / f"cells_vs_fields_{key}.csv", index=False)
    summary = summarize(tables)
    (RESULTS / "cells_vs_fields_summary.txt").write_text(summary + "\n")
    print(summary)


if __name__ == "__main__":
    main()
