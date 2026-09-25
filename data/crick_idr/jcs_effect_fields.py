"""Is "a few hundred cells is enough" still true when the cells come as whole fields?

Marcotti et al. (2026, JCS) argue that a random sample of about 200-300 cells
per well gives the same conclusion as every cell of the well, so imaging more
wastes time and money. Their Fig. 3G-I (idr0139) and Fig. S3G-I (idr0028)
show it: for each sample size, 100 random samples of ``n`` cells per well,
the effect size ``(mean_treated - mean_control) / SD_control`` (their Eqn 2,
as in ``plot_effect_size_v_sample_size`` of their companion code), with the
median and the IQR band over samples. At about 250-300 cells the SN0212398523
and DMSO bands barely overlap, so SN0212398523 reads above DMSO.

Their random cells are drawn from every imaged field of the well: the saving
is computed after the whole well was imaged. A microscope that stops early
acquires whole fields, so the sample it gets is not ``n`` random cells but
the cells of the first few fields. This script runs the same analysis twice,
on the paper's wells:

- **random cells:** the paper's procedure;
- **whole fields:** fields in random order until ``n`` cells, with a random
  part of the last field (the ``n`` cells a microscope would have).

For each ``n`` it reports the effect-size median and IQR band, whether the
bands of the two treatments the paper compares overlap, how often a sample
ranks the two treatments as the full wells do, and how many fields the
``n`` cells took. 1000 samples per size (the paper uses 100) keep the bands
stable; the conclusions do not depend on it.

Wells, from the companion notebooks: idr0139 plate 1093711385, J05
(Untreated, the control), O02 (DMSO), E22 (SN0212398523), L08 (Leptomycin B);
idr0028 plate 2A, K16 (MOCK, the control), C13 (ARAP2), N23 (YAP siRNA),
J23 (LATS1). Only the imaged fields are available: 4 per idr0139 well, 28
per idr0028 well, so whole-field samples are, if anything, optimistic.

Retrospective and exploratory. Usage::

    python data/crick_idr/jcs_effect_fields.py

Writes ``results/jcs_effect_fields.csv`` and ``results/jcs_effect_fields_summary.txt``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from datasets import FASCIN_RATIO, load_idr0139
from idr0028_data import RATIO as YAPTAZ_RATIO
from idr0028_data import load as load_idr0028

RESULTS = Path(__file__).resolve().parent / "results"
SEED = 0
DRAWS = 1000
SIZES = list(range(10, 501, 10))
PAPER = {
    # dataset: (repeat, control well, {condition: well}, comparisons, cell counts to report)
    "idr0139": (
        "1093711385",
        "J05",
        {"DMSO": "O02", "SN0212398523": "E22", "Leptomycin b": "L08"},
        [("SN0212398523", "DMSO"), ("Leptomycin b", "SN0212398523")],
        (50, 250, 300),
    ),
    "idr0028": (
        "2A",
        "K16",
        {"ARAP2": "C13", "YAP": "N23", "LATS1": "J23"},
        [("ARAP2", "YAP"), ("LATS1", "ARAP2")],
        (50, 200, 300),
    ),
}


LOADERS = {"idr0139": (load_idr0139, FASCIN_RATIO), "idr0028": (load_idr0028, YAPTAZ_RATIO)}


def _paper_wells(name: str) -> tuple[list[np.ndarray], dict[str, list[np.ndarray]]]:
    """The control well's fields and each treated well's fields, as arrays of the metric."""
    repeat, control, treated, _, _ = PAPER[name]
    load, metric = LOADERS[name]
    objects, _, _ = load()

    def fields(well: str) -> list[np.ndarray]:
        cells = objects[(objects["repeat"] == repeat) & (objects["well"] == well)]
        return [g[metric].to_numpy() for _, g in cells.groupby("image_file")]

    return fields(control), {c: fields(w) for c, w in treated.items()}


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


def _sample(fields: list[np.ndarray], n: int, how: str, rng: np.random.Generator):
    if how == "random cells":
        return rng.choice(np.concatenate(fields), size=n, replace=False)
    return _field_sample(fields, n, rng)


def _fields_holding(fields: list[np.ndarray], n: int, rng: np.random.Generator) -> float:
    """Mean number of fields, in random order, whose cells reach ``n`` (the last one partly)."""
    counts = []
    for _ in range(200):
        total, k = 0, 0
        for i in rng.permutation(len(fields)):
            k += 1
            total += len(fields[i])
            if total >= n:
                break
        counts.append(k)
    return float(np.mean(counts))


def effects(name: str, rng: np.random.Generator) -> pd.DataFrame:
    """Effect sizes against the control, per sample size and sampling, as in JCS Fig. 3G-I."""
    control, treated = _paper_wells(name)
    n_max = min(len(np.concatenate(f)) for f in [control, *treated.values()])
    rows = []
    for how in ("random cells", "whole fields"):
        for n in [s for s in SIZES if s <= n_max]:
            draws = {c: [] for c in treated}
            for _ in range(DRAWS):
                for c, fields in treated.items():
                    # a fresh control sample per treatment, as in the companion code
                    ctrl = _sample(control, n, how, rng)
                    t = _sample(fields, n, how, rng)
                    draws[c].append((t.mean() - ctrl.mean()) / ctrl.std(ddof=1))
            for c, values in draws.items():
                v = np.array(values)
                rows.append(
                    {
                        "dataset": name,
                        "sampling": how,
                        "n_cells": n,
                        "condition": c,
                        "median": float(np.median(v)),
                        "q25": float(np.percentile(v, 25)),
                        "q75": float(np.percentile(v, 75)),
                        "fields_treated": _fields_holding(treated[c], n, rng),
                        "values": v,
                    }
                )
    return pd.DataFrame(rows)


def full_wells(name: str) -> dict[str, float]:
    """The effect sizes from every imaged cell: the reference the samples should reproduce."""
    control, treated = _paper_wells(name)
    ctrl = np.concatenate(control)
    return {
        c: float((np.concatenate(f).mean() - ctrl.mean()) / ctrl.std(ddof=1))
        for c, f in treated.items()
    }


def compare(table: pd.DataFrame, name: str) -> pd.DataFrame:
    """For each compared pair: band overlap, and how often a sample ranks them as the wells do."""
    _, _, _, pairs, _ = PAPER[name]
    reference = full_wells(name)
    rows = []
    for (how, n), g in table[table.dataset == name].groupby(["sampling", "n_cells"]):
        g = g.set_index("condition")
        for high, low in pairs:
            a, b = g.loc[high], g.loc[low]
            rows.append(
                {
                    "dataset": name,
                    "sampling": how,
                    "n_cells": n,
                    "pair": f"{high} > {low}",
                    "full_well_difference": reference[high] - reference[low],
                    "bands_overlap": bool(a.q25 <= b.q75),
                    # draws are independent, so compare every pair of draws
                    "share_ranked_as_full_wells": float(
                        np.mean(a["values"][:, None] > b["values"][None, :])
                    ),
                    "fields_high": a.fields_treated,
                    "fields_low": b.fields_treated,
                }
            )
    return pd.DataFrame(rows)


def summarize(table: pd.DataFrame, comparisons: pd.DataFrame) -> str:
    lines = [
        "Is a few hundred cells enough when the cells come as whole fields? (JCS Fig. 3G-I, S3G-I)",
        "",
    ]
    for name, (repeat, control, treated, _, report) in PAPER.items():
        reference = full_wells(name)
        control_fields = _paper_wells(name)[0]
        lines.append(f"== {name}, plate {repeat}, control {control} ==")
        lines.append("cells per field, control: " + ", ".join(str(len(f)) for f in control_fields))
        lines.append(
            "effect sizes from every imaged cell: "
            + ", ".join(f"{c} {v:.2f}" for c, v in reference.items())
        )
        t = table[table.dataset == name]
        for n in report:
            lines.append(f"  {n} cells per well:")
            for c in treated:
                row = t[(t.n_cells == n) & (t.condition == c)].set_index("sampling")
                if row.empty:
                    continue
                cells, fields = row.loc["random cells"], row.loc["whole fields"]
                lines.append(
                    f"    {c}: random cells {cells['median']:.2f} "
                    f"[{cells.q25:.2f}, {cells.q75:.2f}]; whole fields "
                    f"{fields['median']:.2f} [{fields.q25:.2f}, {fields.q75:.2f}] "
                    f"from {fields.fields_treated:.1f} fields"
                )
            c = comparisons[(comparisons.dataset == name) & (comparisons.n_cells == n)]
            for pair, g in c.groupby("pair", sort=False):
                g = g.set_index("sampling")
                lines.append(
                    f"    {pair}: ranked as the full wells in "
                    f"{g.loc['random cells', 'share_ranked_as_full_wells']:.0%} of random-cell "
                    f"samples and {g.loc['whole fields', 'share_ranked_as_full_wells']:.0%} of "
                    f"whole-field samples; IQR bands overlap: "
                    f"{'yes' if g.loc['random cells', 'bands_overlap'] else 'no'} (cells), "
                    f"{'yes' if g.loc['whole fields', 'bands_overlap'] else 'no'} (fields)"
                )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    rng = np.random.default_rng(SEED)
    RESULTS.mkdir(exist_ok=True)
    table = pd.concat([effects(name, rng) for name in PAPER], ignore_index=True)
    comparisons = pd.concat([compare(table, name) for name in PAPER], ignore_index=True)
    table.drop(columns="values").to_csv(RESULTS / "jcs_effect_fields.csv", index=False)
    comparisons.to_csv(RESULTS / "jcs_effect_fields_pairs.csv", index=False)
    summary = summarize(table, comparisons)
    (RESULTS / "jcs_effect_fields_summary.txt").write_text(summary + "\n")
    print(summary)


if __name__ == "__main__":
    main()
