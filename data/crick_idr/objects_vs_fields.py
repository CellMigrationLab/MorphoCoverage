"""Objects or fields? The sampling-unit checks of the report, on the stored datasets.

Runs the report's premise (R0 of ``docs/questions.md``) and the clustering of
unusual objects (R4) with ``morphocoverage.sampling_unit``, one analysis group
per well:

- on the paper's wells (``idr0139``: the JCS case study, fascin ratio,
  control Untreated; ``idr0028_screenB``: YAP/TAZ ratio, control MOCK):
  ``fov_variability``; ``object_rule_check``, the JCB rule at its own 5% and
  2.5% targets with whole fields; ``descriptor_spread_check``, the JCS Fig. 2A
  procedure at the paper's threshold (250 nuclei for idr0139, Fig. 2; 200 for
  idr0028, Fig. S2); ``unusual_objects`` against the control wells on every
  intensity feature, and ``fov_clustering`` of those unusual objects;
- on the complete plates (``idr0139_plate``, ``idr0028_screenB_plates``,
  readout only): ``fov_variability`` and ``object_rule_check``, so the
  paper's wells can be read against every well of their plates.

The tables hold every imaged field of each well (4 per idr0139 well, 28 per
idr0028 well). The paper's treatment comparisons with whole fields, which
compare wells, are in ``jcs_effect_fovs.py``. Retrospective and exploratory.
Usage::

    python data/crick_idr/objects_vs_fields.py

Tables are written to ``results/objects_vs_fields_*.csv`` and the summary
quoted in ``docs/preliminary_findings.md`` (entries 9 and 10) to
``results/objects_vs_fields_summary.txt``.
"""

from __future__ import annotations

import pandas as pd
from datasets import RESULTS, load

from morphocoverage.sampling_unit import (
    descriptor_spread_check,
    fov_clustering,
    fov_variability,
    object_rule_check,
    unusual_objects,
)

# dataset: JCS threshold in objects (the paper's reading, Fig. 2 and Fig. S2)
PAPER_WELLS = {"idr0139": 250, "idr0028_screenB": 200}
PLATES = ("idr0139_plate", "idr0028_screenB_plates")
PLATE_DRAWS = 1000  # fewer than on the paper's wells: 308 and 1,144 wells


def _common(dataset) -> dict:
    return {"group_cols": list(dataset.groups), "observation_unit": dataset.observation_unit}


def run_paper_wells(name: str, threshold: int) -> dict[str, pd.DataFrame]:
    dataset = load(name)
    objects, fovs = dataset.tables
    common = _common(dataset)
    control = {"condition": dataset.control}
    variability = fov_variability(objects, fovs, feature=dataset.readout, **common)
    rule = object_rule_check(objects, fovs, feature=dataset.readout, control=control, **common)
    spread = descriptor_spread_check(
        objects,
        fovs,
        feature=dataset.readout,
        threshold=threshold,
        sizes=range(10, 501, 10),
        **common,
    )
    unusual = unusual_objects(
        objects, fovs, feature_cols=list(dataset.features), reference=control, **common
    )
    clustering = fov_clustering(
        objects.assign(unusual=unusual.labels),
        fovs,
        label_col="unusual",
        states=[True],
        **common,
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


def run_plate(name: str) -> dict[str, pd.DataFrame]:
    dataset = load(name)
    objects, fovs = dataset.tables
    common = _common(dataset)
    variability = fov_variability(objects, fovs, feature=dataset.readout, **common)
    rule = object_rule_check(
        objects,
        fovs,
        feature=dataset.readout,
        control={"condition": dataset.control},
        n_draws=PLATE_DRAWS,
        **common,
    )
    tag = {"dataset": name}
    return {"variability": variability.groups.assign(**tag), "jcb": rule.table.assign(**tag)}


def _fields(count: float) -> str:
    return "1 field" if count == 1 else f"{count:.0f} fields"


def _range(values: pd.Series, fmt: str) -> str:
    return f"median {values.median():{fmt}} (range {values.min():{fmt}}-{values.max():{fmt}})"


def _unit_lines(v: pd.DataFrame, j: pd.DataFrame) -> list[str]:
    ok = v[v.status == "ok"]
    lines = [
        (
            f"R0 field variability ({len(ok)} wells): between-field share of variance "
            f"{_range(ok.icc, '.1%')}; design effect {_range(ok.design_effect, '.1f')}; error "
            f"ratio {_range(ok.error_ratio, '.1f')}; fields differing more than random objects "
            f"{_range(ok.unusual_fov_share, '.0%')} (5% expected)."
        )
    ]
    j = j[j.status == "ok"]
    first = j[j.target == j.target.max()]
    lines.append(
        f"R0 JCB: object SD {_range(first.object_sd, '.1%')} of the control mean; for the "
        f"objects obtained, real / predicted error {_range(first.error_ratio, '.1f')}."
    )
    for target, t in j.groupby("target", sort=False):
        lines.append(
            f"  target {target:.1%}: rule objects {_range(t.rule_objects, '.0f')}, fields used "
            f"{_range(t.fovs_used, '.1f')}; real error {_range(t.real_error, '.1%')}; "
            f"{int((~t.meets_target).sum())} of {len(t)} wells miss; with fields as the unit "
            f"the rule asks {_fields(t.fov_rule_fovs.median())} (upper-95 "
            f"{t.fov_rule_fovs_upper95.median():.0f}) of {int(t.fovs.median())} acquired."
        )
    return lines


def summarize(tables: dict[str, pd.DataFrame]) -> str:
    lines = ["Objects or fields: the sampling-unit checks, one group per well", ""]
    for name in (*PAPER_WELLS, *PLATES):
        dataset = load(name)
        v = tables["variability"].query("dataset == @name")
        j = tables["jcb"].query("dataset == @name")
        lines += [f"== {name}: {dataset.description} ({dataset.readout}) ==", ""]
        lines += _unit_lines(v, j)
        if name in PAPER_WELLS:
            threshold = PAPER_WELLS[name]
            s = tables["jcs"].query("dataset == @name")
            lines.append(
                f"R0 JCS Fig. 2A at {threshold} objects, whole-field spread / random-object spread:"
            )
            for descriptor, t in s.groupby("descriptor", sort=False):
                matched = t.objects_to_match.dropna()
                lines.append(
                    f"  {descriptor}: {_range(t.ratio, '.1f')}; fields matched the "
                    f"{threshold}-object spread in {len(matched)} of {len(t)} wells"
                    + (f", from a median of {matched.median():.0f} objects" if len(matched) else "")
                    + " (up to 500 tried)."
                )
            u = tables["unusual"].query("dataset == @name")
            r = tables["rare"].query("dataset == @name")
            ok = u[u.status == "ok"]
            seen = r[r.state_objects > 0]
            clustered = seen[seen.p_clustered <= 0.05]
            lines.append(
                f"R4 unusual objects (all intensity features, against the other "
                f"{dataset.control} wells, 1% reference quantile): "
                f"{_range(ok.unusual_share, '.1%')} of objects per well; seed agreement "
                f"{_range(ok.seed_agreement, '.0%')}."
            )
            lines.append(
                f"  seen in {len(seen)} of {len(r)} wells, clustered by field (p <= 0.05) in "
                f"{len(clustered)}; clustering ratio {_range(seen.clustering_ratio, '.1f')}; "
                f"fields to see them with 95% chance {_range(seen.fovs_needed, '.0f')} from "
                f"field incidence, against {_range(seen.fovs_needed_if_random, '.0f')} if "
                "objects were random."
            )
            if len(clustered):
                factor = clustered.fovs_needed / clustered.fovs_needed_if_random
                lines.append(
                    "  in the clustered wells, fields needed from field incidence / if objects "
                    f"were random: {_range(factor, '.1f')}."
                )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    runs = [run_paper_wells(name, t) for name, t in PAPER_WELLS.items()]
    runs += [run_plate(name) for name in PLATES]
    keys = dict.fromkeys(k for r in runs for k in r)
    tables = {k: pd.concat([r[k] for r in runs if k in r], ignore_index=True) for k in keys}
    for key, table in tables.items():
        table.to_csv(RESULTS / f"objects_vs_fields_{key}.csv", index=False)
    summary = summarize(tables)
    (RESULTS / "objects_vs_fields_summary.txt").write_text(summary + "\n")
    print(summary)


if __name__ == "__main__":
    main()
