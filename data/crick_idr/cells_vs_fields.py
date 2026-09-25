"""Cells or fields? Each paper's own procedure, run with random cells and with whole fields.

Both reference papers are faithful to what they state, and both state that the
cells they count are a random, spatially uniform sample of the well:

- Zweifach (2024, JCB) "further assume[s] that samples are spatially
  uniform" and gives the number of cells to average as
  ``ceil((SD / target) ** 2)``, with SD the between-cell SD in % of the
  control mean and a target averaging error of 5% CV (independent samples)
  or 2.5% CV (paired samples). The promise is that the average of that many
  cells has an averaging error (its standard error) at the target.
- Marcotti et al. (2026, JCS) assume "this small number of cells is
  representative of the broader population" and, in Fig. 2A, draw 100 random
  samples of ``n`` cells from the whole well and plot the max - min of their
  IQR against ``n``; descriptors stabilise at about 250-300 cells in the
  untreated idr0139 well (Fig. 2) and about 200 in idr0028 (Fig. S2).

A microscope does not draw random cells: it images whole fields, and every
cell of a field enters together. So each check runs the paper's procedure
twice, with cells drawn at random from the well (the paper's sampling) and
with the same number of cells obtained by imaging whole fields in random
order (the microscope's sampling). Nothing else changes.

1. **JCB.** For each well, the JCB number of cells ``n`` at 5% and 2.5%, from
   the SD of all cells of the well (so the point and upper-95 estimates
   agree). Then image random fields until at least ``n`` cells, and measure
   the real averaging error: the SD of that average over 4000 draws, in % of
   the control mean. Random cells give the target by construction. The JCB
   formula with the field as the unit (the SD of field means, and its
   upper-95 bound as the paper advises for small pilots) gives the number of
   fields that would meet it.
2. **JCS.** For each well, the Fig. 2A curve: 100 random samples of ``n``
   cells, max - min of the IQR, the median, the SD and the mean, for ``n``
   from 10 to 500. Field samples take fields in random order until ``n``
   cells, with a random part of the last field. The comparison is the
   spread at the paper's threshold (250 cells for idr0139, 200 for idr0028),
   and the smallest ``n`` from which field samples stay within the spread that
   random cells reach at that threshold.

Metrics are the papers' own: the fascin ratio for idr0139 (whose committed
tables reproduce JCS Table 1: wells J05, E22, L08 and O02) and the YAP/TAZ
ratio for idr0028. Both use the committed tables, which hold every imaged
field of each well: 4 per idr0139 well, 28 per idr0028 well. Draws come from
the imaged fields only, so the field errors are, if anything, optimistic:
a well has more positions than were imaged.

Retrospective and exploratory. Usage::

    python data/crick_idr/cells_vs_fields.py

Tables are written to ``data/crick_idr/results/cells_vs_fields_*.csv`` and a
plain summary to ``results/cells_vs_fields_summary.txt``.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import FASCIN_RATIO, load_idr0139
from idr0028_data import RATIO as YAPTAZ_RATIO
from idr0028_data import load as load_idr0028
from scipy import stats

RESULTS = Path(__file__).resolve().parent / "results"
SEED = 0
JCB_DRAWS = 4000
JCS_DRAWS = 100  # as in JCS Fig. 2A
JCS_SIZES = list(range(10, 501, 10))
TARGETS = (5.0, 2.5)
DATASETS = {
    # name: (loader, metric, negative-control condition, JCS threshold in cells)
    "idr0139": (load_idr0139, FASCIN_RATIO, "Untreated", 250),
    "idr0028": (load_idr0028, YAPTAZ_RATIO, "MOCK", 200),
}
DESCRIPTORS = ("IQR", "median", "SD", "mean")


def _wells(name: str) -> tuple[list[dict], float]:
    """Each well's cell values grouped by field, and the control mean (JCB normalization)."""
    load, metric, control, _ = DATASETS[name]
    objects, _, _ = load()
    wells = []
    for (condition, repeat, well), cells in objects.groupby(["condition", "repeat", "well"]):
        fields = [g[metric].to_numpy() for _, g in cells.groupby("image_file")]
        wells.append({"condition": condition, "repeat": repeat, "well": well, "fields": fields})
    control_means = [np.concatenate(w["fields"]).mean() for w in wells if w["condition"] == control]
    return wells, float(np.mean(control_means))


def _jcb_n(sd_pct: float, target: float) -> int:
    return math.ceil((sd_pct / target) ** 2)


def _upper_sd(sd: float, dof: int) -> float:
    """Upper bound of the 95% interval of an SD (chi-square), as in the JCB calculator."""
    return sd * math.sqrt(dof / stats.chi2.ppf(0.025, dof))


def _fields_until(
    fields: list[np.ndarray], n: int, rng: np.random.Generator
) -> tuple[np.ndarray, int]:
    """Whole fields in random order until at least ``n`` cells (the last field kept whole)."""
    taken, count = [], 0
    for i in rng.permutation(len(fields)):
        taken.append(fields[i])
        count += len(fields[i])
        if count >= n:
            break
    return np.concatenate(taken), len(taken)


def jcb(name: str, rng: np.random.Generator) -> pd.DataFrame:
    """The JCB number of cells, and the averaging error it really gives from whole fields."""
    wells, control_mean = _wells(name)
    rows = []
    for w in wells:
        cells = np.concatenate(w["fields"]) / control_mean * 100
        fields = [f / control_mean * 100 for f in w["fields"]]
        sd = cells.std(ddof=1)
        field_means = np.array([f.mean() for f in fields])
        sd_fields = field_means.std(ddof=1)
        for target in TARGETS:
            n = _jcb_n(sd, target)
            if n > len(cells):
                continue
            draws = [_fields_until(fields, n, rng) for _ in range(JCB_DRAWS)]
            means = np.array([d.mean() for d, _ in draws])
            got = np.array([len(d) for d, _ in draws])
            rows.append(
                {
                    "dataset": name,
                    "condition": w["condition"],
                    "repeat": w["repeat"],
                    "well": w["well"],
                    "n_fields": len(fields),
                    "cells_in_well": len(cells),
                    "cell_sd_pct": sd,
                    "target_pct": target,
                    "jcb_cells": n,
                    "cells_obtained": got.mean(),
                    "fields_used": float(np.mean([k for _, k in draws])),
                    # the averaging error the JCB formula predicts for the cells obtained
                    "predicted_error_pct": float(np.mean(sd / np.sqrt(got))),
                    "real_error_pct": means.std(ddof=1),
                    "jcb_fields_point": _jcb_n(sd_fields, target),
                    "jcb_fields_upper95": _jcb_n(_upper_sd(sd_fields, len(fields) - 1), target),
                }
            )
    return pd.DataFrame(rows)


def _descriptors(values: np.ndarray) -> tuple[float, ...]:
    q1, med, q3 = np.percentile(values, [25, 50, 75])
    return q3 - q1, med, values.std(ddof=1), values.mean()


def _field_sample(fields: list[np.ndarray], n: int, rng: np.random.Generator) -> np.ndarray:
    """Whole fields in random order until ``n`` cells, with a random part of the last one."""
    taken, count = [], 0
    for i in rng.permutation(len(fields)):
        need = n - count
        field = fields[i]
        if len(field) >= need:
            taken.append(rng.choice(field, size=need, replace=False))
            break
        taken.append(field)
        count += len(field)
    return np.concatenate(taken)


def jcs(name: str, rng: np.random.Generator) -> pd.DataFrame:
    """JCS Fig. 2A for every well: max - min of each descriptor over 100 samples of n cells."""
    wells, _ = _wells(name)
    rows = []
    for w in wells:
        cells = np.concatenate(w["fields"])
        for n in JCS_SIZES:
            if n > len(cells):
                break
            for sampling in ("random cells", "whole fields"):
                if sampling == "random cells":
                    samples = [rng.choice(cells, size=n, replace=False) for _ in range(JCS_DRAWS)]
                else:
                    samples = [_field_sample(w["fields"], n, rng) for _ in range(JCS_DRAWS)]
                values = np.array([_descriptors(s) for s in samples])
                spread = values.max(axis=0) - values.min(axis=0)
                rows.append(
                    {
                        "dataset": name,
                        "condition": w["condition"],
                        "repeat": w["repeat"],
                        "well": w["well"],
                        "n_cells": n,
                        "sampling": sampling,
                        **{f"{d}_spread": float(s) for d, s in zip(DESCRIPTORS, spread)},
                    }
                )
    return pd.DataFrame(rows)


def jcs_equivalent(curves: pd.DataFrame, threshold: int) -> pd.DataFrame:
    """Per well and descriptor: spreads at the threshold, and the n fields need to match cells."""
    rows = []
    keys = ["dataset", "condition", "repeat", "well"]
    for (dataset, condition, repeat, well), g in curves.groupby(keys):
        cells = g[g.sampling == "random cells"].set_index("n_cells")
        fields = g[g.sampling == "whole fields"].set_index("n_cells")
        if threshold not in cells.index:
            continue
        for d in DESCRIPTORS:
            reference = cells.loc[threshold, f"{d}_spread"]
            # the smallest n from which field samples stay at or below the reference
            above = fields.index[fields[f"{d}_spread"] > reference]
            later = fields.index[fields.index > above.max()] if len(above) else fields.index
            rows.append(
                {
                    "dataset": dataset,
                    "condition": condition,
                    "repeat": repeat,
                    "well": well,
                    "descriptor": d,
                    "threshold_cells": threshold,
                    "spread_random_cells": reference,
                    "spread_whole_fields": fields.loc[threshold, f"{d}_spread"],
                    "ratio": fields.loc[threshold, f"{d}_spread"] / reference,
                    "fields_n_to_match": int(later.min()) if len(later) else None,
                    "max_n": int(fields.index.max()),
                }
            )
    return pd.DataFrame(rows)


def summarize(jcb_table: pd.DataFrame, equivalent: pd.DataFrame) -> str:
    lines = ["Cells or fields: each paper's own procedure, random cells against whole fields", ""]
    for name, (_, metric, control, threshold) in DATASETS.items():
        t = jcb_table[jcb_table.dataset == name]
        lines.append(f"== {name} ({metric}; control {control}) ==")
        lines.append("")
        lines.append("JCB: n = ceil((cell SD / target)^2); averaging error when n cells come as")
        lines.append("whole random fields (random cells give the target by construction).")
        first = t[t.target_pct == TARGETS[0]]
        ratio = first.real_error_pct / first.predicted_error_pct
        lines.append(
            f"  cell SD median {first.cell_sd_pct.median():.1f}% of the control mean. For the "
            f"cells one draw really holds, SD / sqrt(n) predicts "
            f"{first.predicted_error_pct.median():.1f}%; the real error is "
            f"{first.real_error_pct.median():.1f}%, {ratio.median():.1f} times the prediction "
            f"(range {ratio.min():.1f}-{ratio.max():.1f})."
        )
        for target in TARGETS:
            u = t[t.target_pct == target]
            if u.empty:
                continue
            lines.append(
                f"  target {target}%: median {u.jcb_cells.median():.0f} cells "
                f"(range {u.jcb_cells.min()}-{u.jcb_cells.max()}), "
                f"about {u.fields_used.median():.1f} fields; "
                f"real error median {u.real_error_pct.median():.1f}% "
                f"(range {u.real_error_pct.min():.1f}-{u.real_error_pct.max():.1f}); "
                f"{(u.real_error_pct > target).sum()} of {len(u)} wells miss the target; "
                f"JCB per field asks {u.jcb_fields_point.median():.0f} fields "
                f"(upper-95 {u.jcb_fields_upper95.median():.0f}) of {u.n_fields.iloc[0]} imaged."
            )
        e = equivalent[equivalent.dataset == name]
        lines.append("")
        lines.append(
            f"JCS Fig. 2A: max - min over {JCS_DRAWS} samples of {threshold} cells "
            "(the paper's threshold)."
        )
        for d in DESCRIPTORS:
            u = e[e.descriptor == d]
            matched = u.fields_n_to_match.dropna()
            lines.append(
                f"  {d}: whole fields spread {u.ratio.median():.1f} times random cells "
                f"(range {u.ratio.min():.1f}-{u.ratio.max():.1f}); fields matched the "
                f"{threshold}-cell spread in {len(matched)} of {len(u)} wells"
                + (f", at a median of {matched.median():.0f} cells" if len(matched) else "")
                + f" (up to {u.max_n.min()} cells tried)."
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    rng = np.random.default_rng(SEED)
    RESULTS.mkdir(exist_ok=True)
    jcb_table = pd.concat([jcb(name, rng) for name in DATASETS], ignore_index=True)
    curves = pd.concat([jcs(name, rng) for name in DATASETS], ignore_index=True)
    equivalent = pd.concat(
        [jcs_equivalent(curves[curves.dataset == name], DATASETS[name][3]) for name in DATASETS],
        ignore_index=True,
    )
    jcb_table.to_csv(RESULTS / "cells_vs_fields_jcb.csv", index=False)
    curves.to_csv(RESULTS / "cells_vs_fields_jcs_curves.csv", index=False)
    equivalent.to_csv(RESULTS / "cells_vs_fields_jcs.csv", index=False)
    summary = summarize(jcb_table, equivalent)
    (RESULTS / "cells_vs_fields_summary.txt").write_text(summary + "\n")
    print(summary)


if __name__ == "__main__":
    main()
