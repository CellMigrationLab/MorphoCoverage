"""What to write in a figure legend for a plotted feature (question 9, ``figure_legend``).

One paragraph per plotted feature, under its figure on the page (owner, 2026-10-02), from
the numbers the report already holds (no new test), each one the page's table across
repeats gives for that feature and condition. First n, the number of data points shown
(owner, 2026-10-02: "N is always the number of data points of what is shown"): ``n = 174
cells shown, from 3 biological repeats, at least 11 per condition and repeat``, given per
condition when the conditions are not all compared in the same repeats, repeats counted
only where they hold both the condition and the control; wells only when there is more
than one per condition and repeat. Then the test behind "significant across repeats" with
its correction family counted, what the figure draws, and per condition the change (the
fold change the test compares, or the difference of the means), the mean per-repeat
Cohen's d, the adjusted p and p. With fewer than 3 biological repeats it says that there
is no test across repeats and describes the result as one experiment (owner, 2026-10-01)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .._wording import change, count, listing, p_value
from ..data_check import validate_input_tables
from ..statistics import repeat_values
from .contrast_cells import ALPHA
from .contrast_repeats import MIN_REPEATS


def table_counts(contrast: Any) -> pd.DataFrame:
    """Per condition and repeat, the readout's observations as the readout's ``contrast``
    counted them (``condition``, ``repeat``, ``n``; the control once per repeat): the counts
    a legend uses when it is not given the feature's own (``observation_counts``)."""
    table = contrast.table
    repeat_col = contrast.settings.get("repeat_col")
    repeats = table[repeat_col] if repeat_col in table else pd.Series(None, index=table.index)
    control = _control(contrast)
    treated = pd.DataFrame(
        {"condition": table["condition"], "repeat": repeats, "n": table["observations"]}
    )
    paired = table["control_observations"] > 0
    controls = pd.DataFrame(
        {"condition": control, "repeat": repeats[paired], "n": table["control_observations"]}
    ).drop_duplicates("repeat")
    return pd.concat([treated, controls], ignore_index=True).astype({"n": int})


def observation_counts(
    observations: pd.DataFrame, values: pd.Series, condition_col: str, repeat_col: str | None
) -> pd.DataFrame:
    """Per condition and repeat, the observations of one feature that have a value, the
    points its figure draws (``condition``, ``repeat``, ``n``)."""
    shown = observations[np.asarray(pd.notna(values))]
    keys = [condition_col] + ([repeat_col] if repeat_col in shown else [])
    sizes = shown.groupby(keys, sort=False, dropna=False).size().reset_index(name="n")
    if repeat_col not in sizes:
        sizes[repeat_col] = None
    return sizes.rename(columns={condition_col: "condition", repeat_col: "repeat"})[
        ["condition", "repeat", "n"]
    ]


def feature_counts(
    objects: pd.DataFrame,
    fovs: pd.DataFrame,
    features: list[str],
    contrast: Any,
    common: dict[str, Any],
) -> dict[str, pd.DataFrame]:
    """Each feature's ``observation_counts`` from the report's tables (``common``: the
    validation arguments of its setup)."""
    data = validate_input_tables(objects, fovs=fovs, feature_cols=features, **common)
    condition_col = contrast.settings.get("condition_col", "condition")
    repeat_col = contrast.settings.get("repeat_col")
    return {
        feature: observation_counts(
            data.observations, data.feature_values(feature), condition_col, repeat_col
        )
        for feature in features
    }


def _control(contrast: Any) -> Any:
    """The control condition's label."""
    control = contrast.settings.get("control") or {}
    return control.get(contrast.settings.get("condition_col", "condition"))


def _wells(contrast: Any) -> str:
    """``3 wells per condition in each repeat``, counted where the control is present; empty
    with one well per condition and repeat, the usual design."""
    table = contrast.table
    paired = table[table["control_observations"] > 0]
    wells = pd.concat([paired["technical"], paired["control_technical"]]).astype(int)
    if not len(wells) or (wells == 1).all():
        return ""
    low, high = int(wells.min()), int(wells.max())
    spread = f"{low}" if low == high else f"{low} to {high}"
    return f"{spread} wells per condition in each repeat"


def _sides(counts: pd.DataFrame, control: Any) -> dict[Any, tuple[dict, int]]:
    """Per condition, the control last: its observations in each repeat that holds both it
    and the control, and its observations shown in repeats without the control."""
    grouped = counts.groupby("condition", sort=False, dropna=False)
    by = {c: dict(zip(b["repeat"], b["n"], strict=True)) for c, b in grouped}
    own = by.get(control, {})
    sides: dict[Any, tuple[dict, int]] = {}
    paired_control: set = set()
    for condition, sizes in by.items():
        if condition == control:
            continue
        paired = {r: n for r, n in sizes.items() if n > 0 and own.get(r, 0) > 0}
        paired_control |= set(paired)
        sides[condition] = (paired, int(sum(sizes.values()) - sum(paired.values())))
    if control in by:
        paired = {r: n for r, n in own.items() if r in paired_control}
        sides[control] = (paired, int(sum(own.values()) - sum(paired.values())))
    return sides


def n_sentence(counts: pd.DataFrame, control: Any, units: str, wells: str = "") -> tuple[str, bool]:
    """The legend's n (owner, 2026-10-02: "N is always the number of data points of what is
    shown"): ``n = 174 cells shown, from 3 biological repeats, at least 11 per condition and
    repeat``; per condition when the conditions are not all compared in the same repeats
    (``siCCT8#5: n = 63 cells from 3 repeats, at least 17 per repeat``). Repeats count only
    where they hold both the condition and the control. Also says whether it is given per
    condition (then a sentence of its own)."""
    shown = int(counts["n"].sum())
    sides = _sides(counts, control)
    pairs = [frozenset(paired) for paired, _ in sides.values()]
    head = f"n = {shown} {units} shown"
    if pairs and len(set(pairs)) == 1 and pairs[0] and not any(u for _, u in sides.values()):
        repeats = len(pairs[0])
        least = min(n for paired, _ in sides.values() for n in paired.values())
        each = "per condition and repeat" if repeats > 1 else "per condition"
        parts = [head, f"from {count(repeats, 'biological repeat')}", wells]
        return ", ".join(p for p in parts if p) + f", at least {least} {each}", False
    entries = []
    for condition, (paired, unpaired) in sides.items():
        without = "another condition" if condition == control else "the control"
        if paired:
            entry = f"{condition}: n = {sum(paired.values())} {units} from "
            entry += count(len(paired), "repeat")
            if len(paired) > 1:
                entry += f", at least {min(paired.values())} per repeat"
            if unpaired:
                entry += f", and {unpaired} in repeats without {without}, not compared"
        else:
            entry = f"{condition}: n = {unpaired} {units}, in repeats without {without}, "
            entry += "not compared"
        entries.append(entry)
    return f"{head}. " + "; ".join(entries) + (f"; {wells}" if wells else ""), True


def _families(across: pd.DataFrame, features: int | None = None) -> str:
    """The correction family, counted: ``over 2 tests (siCCT8#5 and siCCT8#8 against
    siCTRL)``, or the tests in each sub-experiment."""
    if features and features > 1:
        return f"over {features} features and their conditions"
    tested = across[across["p_paired_t_normalised"].notna() | across["p_paired_t"].notna()]
    groups = list(tested.groupby("experiment", sort=False))
    sizes = sorted({len(block) for _, block in groups})
    if len(groups) == 1:
        block = groups[0][1]
        control = block["control_condition"].iloc[0]
        names = listing(list(block["condition"].astype(str)), 4)
        return f"over {count(len(block), 'test')} ({names} against {control})"
    size = f"{sizes[0]}" if len(sizes) == 1 else f"{sizes[0]} to {sizes[-1]}"
    return f"{size} tests in each of {len(groups)} sub-experiments, each with its own control"


def _values(rows: pd.DataFrame) -> str:
    """What the paired test compared: the log ratios, the differences, or each where it
    applies."""
    kinds = set(rows["values"])
    if kinds == {"normalised"}:
        return "normalised to each repeat's control, as log ratios"
    if kinds == {"raw"}:
        return "the differences of the repeat means, a mean not being positive"
    return (
        "normalised to each repeat's control as log ratios, the differences of the means "
        "where a mean is not positive"
    )


def _condition(row: Any) -> str:
    """``siCCT8#5: change ×0.77, d -0.57, adjusted p 0.26 (p 0.21)``, as the table has them."""
    parts = [f"change {change(row.change, row.values)}"] if pd.notna(row.change) else []
    if pd.notna(row.d_per_repeat_mean):
        parts.append(f"d {row.d_per_repeat_mean:+.2f}")
    if pd.notna(row.adjusted_p):
        parts.append(f"adjusted p {p_value(row.adjusted_p)} (p {p_value(row.p)})")
    return f"{row.condition}: " + ", ".join(parts)


def described_rows(contrast: Any) -> pd.DataFrame:
    """Without a table across repeats (one repeat, one feature): the readout's rows for its
    legend from the per-repeat table, the change and the mean d as the paired test would
    read them, no p."""
    table = getattr(contrast, "table", pd.DataFrame())
    if not len(table) or "mean" not in table:
        return pd.DataFrame()
    repeat_col = contrast.settings.get("repeat_col")
    rows = []
    for condition, block in table.groupby("condition", sort=False):
        block = block[block["mean"].notna() & block["control_mean"].notna()]
        if not len(block):
            continue
        values, _, fold = repeat_values(block["mean"], block["control_mean"])
        d = block["d_objects"].to_numpy(float)
        d = d[np.isfinite(d)]
        rows.append(
            {
                "condition": condition,
                "repeats": int(block[repeat_col].nunique()) if repeat_col in block else 1,
                "values": values,
                "change": fold,
                "d_per_repeat_mean": float(d.mean()) if len(d) else np.nan,
                "p": np.nan,
                "adjusted_p": np.nan,
            }
        )
    return pd.DataFrame(rows)


def figure_legend(
    contrast: Any, rows: pd.DataFrame, unit_words: str, counts: pd.DataFrame | None = None
) -> str | None:
    """The figure-legend paragraph for one feature (see the module), or ``None`` without a
    contrast or rows. ``rows`` are the feature's rows of the page's table across repeats
    (``condition``, ``repeats``, ``values``, ``change``, ``d_per_repeat_mean``, ``p``,
    ``adjusted_p``), so every number is the table's; ``unit_words`` names the observations;
    ``counts`` are the feature's observations shown per condition and repeat
    (``observation_counts``; the readout's ``table_counts`` when not given)."""
    if contrast is None or not len(getattr(contrast, "table", ())) or not len(rows):
        return None
    drawn = (
        f"Points: {unit_words}, coloured by condition; bars: the median with its 95% interval "
        "(the tests use means)."
    )
    counts = table_counts(contrast) if counts is None else counts
    n, by_condition = n_sentence(counts, _control(contrast), unit_words, _wells(contrast))
    # The n of every condition at once ends in a clause the test follows; n by condition is
    # its own sentence.
    join = ". " if by_condition else "; "

    def lead(text: str) -> str:
        return text[0].upper() + text[1:] if by_condition else text

    if rows["p"].isna().all():
        readings = [_condition(r) for r in rows.itertuples() if pd.notna(r.change)]
        described = f" Described as one experiment: {'; '.join(readings)}." if readings else ""
        return (
            f"{n}{join}{lead(f'no test across repeats (it needs {MIN_REPEATS})')}. {drawn}"
            f"{described} Repeat the experiment before testing; the cell-level screen is "
            "exploratory."
        )
    across = getattr(contrast, "repeats", None)
    features = contrast.settings.get("family_features")
    families = _families(across, features) if across is not None and len(across) else ""
    many = across is not None and "experiment" in across and across["experiment"].nunique() > 1
    within = " within each sub-experiment" if many else ""
    if families.startswith("over "):
        family = f"{',' if within else ''} {families}"
    else:
        family = f" ({families})" if families else ""
    conditions = "; ".join(_condition(r) for r in rows.itertuples())
    test = (
        f"paired t-test on the repeat means ({_values(rows)}), Benjamini-Hochberg{within}"
        f"{family}; significant at adjusted p ≤ {ALPHA:g}"
    )
    return (
        f"{n}{join}{lead(test)}. {drawn} d: Cohen's d, the mean of the per-repeat values. "
        f"{conditions}. Report the change and the adjusted p; the cell-level screen is "
        "exploratory."
    )


__all__ = [
    "described_rows",
    "feature_counts",
    "figure_legend",
    "n_sentence",
    "observation_counts",
    "table_counts",
]
