"""The summary of a module's run and the pre-registered selection rule
(docs/common/validation.md, "Selection rule").

``summarise(script, out)`` merges the chunks in ``<out>/<module>/chunks``, computes per point
and method the rates and means with their Monte Carlo standard errors, applies each
question's ``runner.Rule`` and writes ``<out>/<module>/summary.csv`` (one row per point and
method) and ``summary.md`` (the gates, the power per alternative class, the ties and the
winner per question): what the owner sends back. The rule, in order:

1. Hard gates: no more than ``failure_bound`` of the core replicates failed (an exception or
   a missing or non-finite primary output); every output named in ``hard`` is 1 wherever it
   is given (rotation invariance in P3, correct "not estimable" detection in P2).
2. Calibration gate on the gated core points (``Point.null``): the simultaneous one-sided
   95% upper bound of the gate metric (Bonferroni over those ``K`` points: Clopper-Pearson
   at ``1 - 0.05 / K`` for a rate, a normal bound for a mean such as the FDP) at most
   ``bound``; no lower bound.
3. Coverage gate (methods giving 95% intervals): the simultaneous lower bound of the
   coverage over every core point at least ``coverage_bound``; the length is reported.
4. Objective over the alternative classes, each class weighing the same (its points
   averaged first); Pareto-dominated methods dropped (another at least as good in every
   class and better by more than ``tie`` in one); methods within ``tie`` (2 points) or
   2 Monte Carlo SE of the paired difference from the best are tied, and ties go to the
   simplest (``Method.simplicity``), then the fastest.
Stress and screening points are reported, never gated.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from .runner import FULL, RESERVED, Module, Rule

_RESERVED = RESERVED


def load(out: Path, module: str) -> pd.DataFrame:
    """Every row of a module's chunks in ``out`` (empty when there are none)."""
    files = sorted((Path(out) / module / "chunks").glob("*.csv"))
    if not files:
        return pd.DataFrame()
    rows = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    rows["alternative"] = rows["alternative"].fillna("").astype(str)
    return rows


def derived(rows: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """``rows`` with ``reject`` (p at most ``alpha``), ``covered`` (the truth in
    ``[lo, hi]``), ``length``, ``deviation`` (estimate minus truth; its mean is the bias) and
    ``squared`` (its mean the MSE) where their inputs are given."""
    rows = rows.copy()
    if "p" in rows:
        rows["reject"] = np.where(rows["p"].notna(), (rows["p"] <= alpha).astype(float), np.nan)
    if {"lo", "hi", "truth"} <= set(rows):
        given = rows[["lo", "hi", "truth"]].notna().all(axis=1)
        inside = (rows["lo"] <= rows["truth"]) & (rows["truth"] <= rows["hi"])
        rows["covered"] = np.where(given, inside.astype(float), np.nan)
        rows["length"] = rows["hi"] - rows["lo"]
    if {"estimate", "truth"} <= set(rows):
        rows["deviation"] = rows["estimate"] - rows["truth"]
        rows["squared"] = rows["deviation"] ** 2
    return rows


def metrics(rows: pd.DataFrame) -> list[str]:
    """The numeric outputs summarised (every numeric column but the bookkeeping ones)."""
    return [
        c for c in rows.columns if c not in _RESERVED and pd.api.types.is_numeric_dtype(rows[c])
    ]


def point_table(rows: pd.DataFrame) -> pd.DataFrame:
    """One row per point and method: replicates, failures, median runtime, and the mean and
    Monte Carlo SE of every metric."""
    keys = ["point", "kind", "null", "alternative", "method", "question"]
    out = []
    for key, group in rows.groupby(keys, sort=False, dropna=False):
        row = dict(zip(keys, key, strict=True))
        row.update(
            replicates=len(group),
            failure_rate=float(group["failed"].mean()),
            runtime=float(group["runtime"].median()),
        )
        for metric in metrics(rows):
            values = group[metric].dropna().to_numpy(dtype=float)
            row[metric] = float(values.mean()) if len(values) else np.nan
            row[f"{metric}_se"] = (
                float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else np.nan
            )
            row[f"{metric}_n"] = len(values)
        out.append(row)
    return pd.DataFrame(out)


def upper_bound(mean: float, n: int, se: float, kind: str, level: float) -> float:
    """The one-sided upper bound at ``level``: Clopper-Pearson for a rate, normal for a
    mean."""
    if n == 0 or not np.isfinite(mean):
        return np.nan
    if kind == "binomial":
        successes = round(mean * n)
        return 1.0 if successes >= n else float(stats.beta.ppf(level, successes + 1, n - successes))
    return float(mean + stats.norm.ppf(level) * (se if np.isfinite(se) else 0.0))


def lower_bound(mean: float, n: int, level: float) -> float:
    """The one-sided Clopper-Pearson lower bound of a rate at ``level``."""
    if n == 0 or not np.isfinite(mean):
        return np.nan
    successes = round(mean * n)
    return 0.0 if successes <= 0 else float(stats.beta.ppf(1 - level, successes, n - successes + 1))


def _class_means(values: pd.DataFrame, column: str) -> pd.Series:
    """Per method, the mean over the alternative classes of each class's mean over its
    points (equal weights)."""
    by_class = values.groupby(["method", "alternative"])[column].mean()
    return by_class.groupby("method").mean()


def _paired_se(rows: pd.DataFrame, rule: Rule, first: str, second: str) -> float:
    """The Monte Carlo SE of the difference in the objective between two methods, paired by
    replicate, with the classes weighted equally."""
    classes = set(rule.classes) or set(rows["alternative"]) - {""}
    chosen = rows[
        rows["method"].isin([first, second])
        & rows["alternative"].isin(classes)
        & (rows["kind"] == "core")
    ]
    wide = chosen.pivot_table(
        index=["alternative", "point", "replicate"], columns="method", values=rule.objective
    ).dropna()
    if wide.empty or first not in wide or second not in wide:
        return np.nan
    difference = (wide[first] - wide[second]).rename("d").reset_index()
    per_point = difference.groupby(["alternative", "point"])["d"].agg(["var", "count"])
    per_point["var_mean"] = per_point["var"] / per_point["count"]
    per_class = per_point.groupby("alternative").agg(
        var=("var_mean", "sum"), points=("var_mean", "size")
    )
    per_class["var"] /= per_class["points"] ** 2
    return float(np.sqrt(per_class["var"].sum()) / len(per_class))


def select(rows: pd.DataFrame, table: pd.DataFrame, module: Module, rule: Rule) -> pd.DataFrame:
    """One row per method of ``rule.question``: each gate, eligibility, the objective per
    class and over classes, Pareto, ties and the winner (see the module)."""
    methods = [m for m in module.methods if m.question == rule.question]
    names = [m.name for m in methods]
    here = table[table["method"].isin(names)]
    core = here[here["kind"] == "core"]
    classes = set(rule.classes) or set(core["alternative"]) - {""}
    mine_class = core["alternative"].isin(classes)
    gated = core[core["null"].astype(bool) & (~mine_class | rule.gate_alternatives)]
    k_gate = max(1, gated["point"].nunique())
    k_cover = max(1, core["point"].nunique())
    out = []
    for method in methods:
        mine = core[core["method"] == method.name]
        result: dict[str, Any] = {"question": rule.question, "method": method.name}
        result["role"], result["simplicity"] = method.role, method.simplicity
        result["runtime"] = float(mine["runtime"].median()) if len(mine) else np.nan
        failures = float((mine["failure_rate"] * mine["replicates"]).sum())
        result["failure_rate"] = failures / max(1, mine["replicates"].sum())
        hard = result["failure_rate"] <= rule.failure_bound
        for name in rule.hard:
            if name in mine and mine[name].notna().any():
                hard &= bool((mine[name].dropna() >= 1 - 1e-12).all())
        result["hard_gates"] = bool(hard)
        bounds = (
            []
            if not rule.gate
            else [
                upper_bound(
                    r[rule.gate],
                    int(r[f"{rule.gate}_n"]),
                    r.get(f"{rule.gate}_se", np.nan),
                    rule.gate_kind,
                    1 - 0.05 / k_gate,
                )
                for _, r in gated[gated["method"] == method.name].iterrows()
                if rule.gate in r and np.isfinite(r[rule.gate])
            ]
        )
        result["gate_upper"] = max(bounds) if bounds else np.nan
        result["calibrated"] = not rule.gate or (
            bool(bounds) and result["gate_upper"] <= rule.bound
        )
        covered = [
            lower_bound(r["covered"], int(r["covered_n"]), 1 - 0.05 / k_cover)
            for _, r in mine.iterrows()
            if "covered" in r and np.isfinite(r["covered"])
        ]
        result["coverage_lower"] = min(covered) if covered else np.nan
        result["covers"] = not (rule.coverage and covered) or min(covered) >= rule.coverage_bound
        result["eligible"] = result["hard_gates"] and result["calibrated"] and result["covers"]
        out.append(result)
    frame = pd.DataFrame(out)
    alternatives = core[core["alternative"].isin(classes)]
    if alternatives.empty or rule.objective not in alternatives:
        frame["objective"] = np.nan
        frame["winner"] = False
        return frame
    per_class = alternatives.groupby(["method", "alternative"])[rule.objective].mean().unstack()
    for column in per_class.columns:
        frame[f"class: {column}"] = frame["method"].map(per_class[column])
    frame["objective"] = frame["method"].map(_class_means(alternatives, rule.objective))
    return _rank(frame, per_class, rows, rule)


def _rank(
    frame: pd.DataFrame, per_class: pd.DataFrame, rows: pd.DataFrame, rule: Rule
) -> pd.DataFrame:
    """Pareto, ties and the winner among the eligible methods of ``frame``."""
    sign = 1.0 if rule.maximise else -1.0
    eligible = frame[frame["eligible"] & frame["objective"].notna()]["method"].tolist()
    scores = sign * per_class.loc[[m for m in eligible if m in per_class.index]]
    # Dominated: another method at least as good in every class and better by more than the
    # tie margin in one (a difference within the margin is a tie, never a domination).
    dominated = {
        a
        for a in scores.index
        for b in scores.index
        if a != b
        and (scores.loc[b] >= scores.loc[a]).all()
        and (scores.loc[b] > scores.loc[a] + rule.tie).any()
    }
    frame["pareto"] = frame["method"].isin([m for m in eligible if m not in dominated])
    frame["tied"] = False
    frame["paired_se"] = np.nan
    frame["winner"] = False
    front = frame[frame["pareto"]]
    if front.empty:
        return frame
    best = front.loc[(sign * front["objective"]).idxmax(), "method"]
    best_value = float(frame.loc[frame["method"] == best, "objective"].iloc[0])
    for index, row in front.iterrows():
        se = _paired_se(rows, rule, best, row["method"]) if row["method"] != best else 0.0
        gap = abs(best_value - row["objective"])
        frame.loc[index, "paired_se"] = se
        frame.loc[index, "tied"] = gap < rule.tie or (np.isfinite(se) and gap < 2 * se)
    tied = frame[frame["tied"]].sort_values(["simplicity", "runtime"])
    frame.loc[tied.index[0], "winner"] = True
    return frame


def _markdown(frame: pd.DataFrame, columns: list[str]) -> str:
    """A pipe table of ``columns`` of ``frame``, numbers to 4 significant digits."""

    def cell(value: Any) -> str:
        if isinstance(value, (bool, np.bool_)):
            return "yes" if value else "no"
        if isinstance(value, (float, np.floating)):
            return "" if not np.isfinite(value) else f"{value:.4g}"
        return str(value)

    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    lines += ["| " + " | ".join(cell(r[c]) for c in columns) + " |" for _, r in frame.iterrows()]
    return "\n".join(lines)


def summarise(script: str, out: Path) -> list[Path]:
    """Merge the chunks of ``script``'s module in ``out``, apply its rules and write
    ``summary.csv`` and ``summary.md`` (see the module); returns the files written."""
    module: Module = importlib.import_module(script).MODULE
    rows = load(out, module.name)
    folder = Path(out) / module.name
    folder.mkdir(parents=True, exist_ok=True)
    if rows.empty:
        raise FileNotFoundError(f"no chunks in {folder / 'chunks'}: run the module first")
    rows = derived(rows, module.alpha)
    table = point_table(rows)
    table.to_csv(folder / "summary.csv", index=False)
    seeds = ", ".join(str(s) for s in sorted(rows["seed"].unique()))
    counts = f"{int(table['replicates'].min())}-{int(table['replicates'].max())}"
    header = (
        f"{len(rows)} method runs, {rows['point'].nunique()} design points, "
        f"replicates per point {counts}, seed {seeds}. Selection rule: docs/common/validation.md."
    )
    gated = table[(table["kind"] == "core") & table["null"].astype(bool)]
    if gated.empty or gated["replicates"].min() < FULL["core_null"]:
        header += (
            " This run is smaller than the pre-registered size (runner.FULL): a development "
            "run, never a result."
        )
    parts = [f"# {module.name.upper()}: {module.title}", "", header]
    for rule in module.rules:
        frame = select(rows, table, module, rule)
        frame.to_csv(folder / f"selection_{rule.question}.csv", index=False)
        winner = frame[frame["winner"].astype(bool)]
        verdict = winner["method"].iloc[0] if len(winner) else "none: no method is eligible"
        columns = [c for c in frame.columns if c not in ("question",)]
        parts += ["", f"## Question: {rule.question}", "", f"Winner: **{verdict}**.", ""]
        parts.append(_markdown(frame, columns))
    stress = table[table["kind"] != "core"]
    if len(stress):
        shown = [
            c
            for c in (
                "reject",
                "covered",
                "deviation",
                "fdp",
                "tdp",
                "false_stop",
                "fields",
                "biology_error",
            )
            if c in stress
        ]
        parts += ["", "## Stress and screening (reported, not gated)", ""]
        parts.append(_markdown(stress, ["point", "method", "replicates", "failure_rate", *shown]))
    (folder / "summary.md").write_text("\n".join(parts) + "\n")
    return [folder / "summary.csv", folder / "summary.md"]


__all__ = ["derived", "load", "lower_bound", "point_table", "select", "summarise", "upper_bound"]
