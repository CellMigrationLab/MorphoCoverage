"""Is "a few hundred cells is enough" still true when the objects come as whole FOVs?

Marcotti et al. (2026, JCS) argue that a random sample of about 200-300 cells
per well gives the same conclusion as every object of the well, so imaging more
wastes time and money. Their Fig. 3G-I (idr0139) and Fig. S3G-I (idr0028)
show it: for each sample size, 100 random samples of ``n`` objects per well,
the effect size ``(mean_treated - mean_control) / SD_control`` (their Eqn 2,
as in ``plot_effect_size_v_sample_size`` of their companion code), with the
median and the IQR band over samples. At about 250-300 objects the SN0212398523
and DMSO bands barely overlap, so SN0212398523 reads above DMSO.

Their random objects are drawn from every imaged field of the well: the saving
is computed after the whole well was imaged. A microscope that stops early
acquires whole fields, so the sample it gets is not ``n`` random objects but
the objects of the first few fields. This script runs the same analysis twice,
on the paper's wells:

- **random objects:** the paper's procedure;
- **whole FOVs:** fields in random order until ``n`` objects, with a random
  part of the last field (the ``n`` objects a microscope would have).

For each ``n`` it reports the effect-size median and IQR band, whether the
bands of the two treatments the paper compares overlap, how often a sample
ranks the two treatments as the full wells do, and how many fields the
``n`` objects took. 1000 samples per size (the paper uses 100) keep the bands
stable; the conclusions do not depend on it.

Wells, from the companion notebooks: idr0139 plate 1093711385, J05
(Untreated, the control), O02 (DMSO), E22 (SN0212398523), L08 (Leptomycin B);
idr0028 plate 2A, K16 (MOCK, the control), C13 (ARAP2), N23 (YAP siRNA),
J23 (LATS1). Only the imaged fields are available: 4 per idr0139 well, 28
per idr0028 well, so whole-field samples are, if anything, optimistic.

Retrospective and exploratory. Usage::

    python data/crick_idr/jcs_effect_fovs.py

Writes ``results/jcs_effect_fovs.csv``, ``results/jcs_effect_fovs_pairs.csv`` and
``results/jcs_effect_fovs_summary.txt``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from datasets import RESULTS, load

SEED = 0
DRAWS = 1000
SIZES = list(range(10, 501, 10))
PAPER = {
    # dataset: (repeat, control well, {condition: well}, comparisons, object counts to report)
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


DATASETS = {"idr0139": "idr0139", "idr0028": "idr0028_screenB"}


def _paper_wells(name: str) -> tuple[list[np.ndarray], dict[str, list[np.ndarray]]]:
    """The control well's fields and each treated well's fields, as arrays of the metric."""
    repeat, control, treated, _, _ = PAPER[name]
    dataset = load(DATASETS[name])
    objects, metric = dataset.objects, dataset.readout

    def fields(well: str) -> list[np.ndarray]:
        rows = objects[(objects["repeat"] == repeat) & (objects["well"] == well)]
        return [g[metric].to_numpy() for _, g in rows.groupby("image_file")]

    return fields(control), {c: fields(w) for c, w in treated.items()}


def _field_sample(fields: list[np.ndarray], n: int, rng: np.random.Generator) -> np.ndarray:
    """Whole fields in random order until ``n`` objects, with a random part of the last one."""
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
    if how == "random objects":
        return rng.choice(np.concatenate(fields), size=n, replace=False)
    return _field_sample(fields, n, rng)


def _fields_holding(fields: list[np.ndarray], n: int, rng: np.random.Generator) -> float:
    """Mean number of fields, in random order, whose objects reach ``n`` (the last one partly)."""
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
    for how in ("random objects", "whole FOVs"):
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
                        "n_objects": n,
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
    """The effect sizes from every imaged object: the reference the samples should reproduce."""
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
    for (how, n), g in table[table.dataset == name].groupby(["sampling", "n_objects"]):
        g = g.set_index("condition")
        for high, low in pairs:
            a, b = g.loc[high], g.loc[low]
            rows.append(
                {
                    "dataset": name,
                    "sampling": how,
                    "n_objects": n,
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
        "Is a few hundred objects enough when they come as whole fields? (JCS Fig. 3G-I, S3G-I)",
        "",
    ]
    for name, (repeat, control, treated, _, report) in PAPER.items():
        reference = full_wells(name)
        control_fields = _paper_wells(name)[0]
        lines.append(f"== {name}, plate {repeat}, control {control} ==")
        lines.append(
            "objects per field, control: " + ", ".join(str(len(f)) for f in control_fields)
        )
        lines.append(
            "effect sizes from every imaged object: "
            + ", ".join(f"{c} {v:.2f}" for c, v in reference.items())
        )
        t = table[table.dataset == name]
        for n in report:
            lines.append(f"  {n} objects per well:")
            for c in treated:
                row = t[(t.n_objects == n) & (t.condition == c)].set_index("sampling")
                if row.empty:
                    continue
                random, whole = row.loc["random objects"], row.loc["whole FOVs"]
                lines.append(
                    f"    {c}: random objects {random['median']:.2f} "
                    f"[{random.q25:.2f}, {random.q75:.2f}]; whole fields "
                    f"{whole['median']:.2f} [{whole.q25:.2f}, {whole.q75:.2f}] "
                    f"from {whole.fields_treated:.1f} fields"
                )
            c = comparisons[(comparisons.dataset == name) & (comparisons.n_objects == n)]
            for pair, g in c.groupby("pair", sort=False):
                g = g.set_index("sampling")
                lines.append(
                    f"    {pair}: ranked as the full wells in "
                    f"{g.loc['random objects', 'share_ranked_as_full_wells']:.0%} of random-object "
                    f"samples and {g.loc['whole FOVs', 'share_ranked_as_full_wells']:.0%} of "
                    f"whole-field samples; IQR bands overlap: "
                    f"{'yes' if g.loc['random objects', 'bands_overlap'] else 'no'} (objects), "
                    f"{'yes' if g.loc['whole FOVs', 'bands_overlap'] else 'no'} (fields)"
                )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    rng = np.random.default_rng(SEED)
    RESULTS.mkdir(exist_ok=True)
    table = pd.concat([effects(name, rng) for name in PAPER], ignore_index=True)
    comparisons = pd.concat([compare(table, name) for name in PAPER], ignore_index=True)
    table.drop(columns="values").to_csv(RESULTS / "jcs_effect_fovs.csv", index=False)
    comparisons.to_csv(RESULTS / "jcs_effect_fovs_pairs.csv", index=False)
    summary = summarize(table, comparisons)
    (RESULTS / "jcs_effect_fovs_summary.txt").write_text(summary + "\n")
    print(summary)


if __name__ == "__main__":
    main()
