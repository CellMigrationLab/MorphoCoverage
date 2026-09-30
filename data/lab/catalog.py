"""The lab datasets, ready to use: ``load(id)`` returns the tables and the dataset's card.

Each dataset lives in ``data/lab/<id>/`` (see ``README.md``):

- ``card.toml``: the facts (source, paper, acquisition, mapping, the paper's claims with quotes,
  the curator's hypotheses), checked by ``check_card``;
- ``objects.csv.gz`` / ``fovs.csv.gz``: the MorphoBricks tables, built by ``convert.py`` from
  the pinned source;
- ``DATASET.md``: generated from the card (``python data/lab/catalog.py render <id>``), for
  people and agents interpreting what MorphoCoverage says about the data;
- ``NOTICE``: the source's licence and attribution.

Runs of MorphoCoverage on a dataset go to ``data/lab/results/<id>/`` and are logged in
``data/lab/FINDINGS.md``, never in DATASET.md. This is data-preparation code, not part of
the package.

Usage::

    python data/lab/catalog.py check            # check every card and its tables
    python data/lab/catalog.py render <id>      # rewrite <id>/DATASET.md from its card
"""

from __future__ import annotations

import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
# The shared MorphoBricks columns the tables are written with (card contract_version).
CONTRACT_VERSION = "2026-09-biological_repeat"
REQUIRED = {
    "": [
        "id",
        "title",
        "curator",
        "created",
        "contract_version",
        "role",
        "observation_unit",
        "feature_cols",
        "readout",
        "control",
    ],
    "source": ["repository", "record", "doi", "licence", "retrieved", "member_sha256"],
    "publication": ["doi", "citation", "version_quoted", "figure"],
    "biology": ["cells", "conditions", "control", "human_material", "identifiers_screened"],
    "acquisition": ["field_selection", "fields", "stage_coordinates", "acquisition_order"],
    "design": ["condition", "biological_repeat", "technical_repeat", "field", "fov_table_source"],
    "hypotheses": ["written", "items"],
}
ROLES = ("development", "held_out")
LICENCES = ("CC-BY-4.0", "CC0-1.0")
# The values a hierarchy level may hold when it is not a real label (data/lab/README.md).
SINGLE, UNRECORDED = "single", "unrecorded"


@dataclass(frozen=True)
class LabDataset:
    """One stored lab dataset with what an analysis needs to know about it."""

    id: str
    objects: pd.DataFrame
    fovs: pd.DataFrame
    card: dict[str, Any]

    @property
    def tables(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """``(objects, fovs)``, as every package analysis takes them."""
        return self.objects, self.fovs

    @property
    def feature_cols(self) -> list[str]:
        return list(self.card["feature_cols"])

    @property
    def observation_unit(self) -> str:
        return self.card["observation_unit"]

    @property
    def control(self) -> dict[str, str]:
        return dict(self.card["control"])


def ids() -> list[str]:
    """Every dataset folder holding a card, sorted."""
    return sorted(p.parent.name for p in HERE.glob("*/card.toml"))


def read_card(dataset_id: str) -> dict[str, Any]:
    """The parsed card of ``dataset_id``."""
    path = HERE / dataset_id / "card.toml"
    if not path.exists():
        raise ValueError(f"unknown lab dataset {dataset_id!r}; choose one of {ids()}")
    with open(path, "rb") as handle:
        return tomllib.load(handle)


def check_card(card: dict[str, Any]) -> list[str]:
    """The problems of a card (empty when it is complete and follows the rules)."""
    problems = []
    for section, keys in REQUIRED.items():
        block = card.get(section, {}) if section else card
        for key in keys:
            if block.get(key) in (None, "", []):
                where = f"{section}.{key}" if section else key
                problems.append(f"missing {where}")
    if card.get("role") not in ROLES:
        problems.append(f"role must be one of {ROLES}")
    if card.get("source", {}).get("licence") not in LICENCES:
        problems.append(f"licence must be one of {LICENCES} (owner decision)")
    if card.get("contract_version") != CONTRACT_VERSION:
        problems.append(f"contract_version {card.get('contract_version')!r} needs a translation")
    for claim in card.get("claims", []):
        for key in ("figure", "claim", "quote", "quote_source", "reproduction"):
            if not claim.get(key):
                problems.append(f"claim {claim.get('claim', '?')!r}: missing {key}")
        status = claim.get("reproduction", {}).get("status")
        if status not in ("match", "mismatch", "partly", "not-reproducible-from-deposit"):
            problems.append(f"claim {claim.get('claim', '?')!r}: reproduction status {status!r}")
    return problems


def load(dataset_id: str) -> LabDataset:
    """The stored dataset ``dataset_id``, ready for the package's analyses."""
    card = read_card(dataset_id)
    folder = HERE / dataset_id
    # Labels stay text (a repeat "1" is a label, not a number).
    objects = pd.read_csv(folder / "objects.csv.gz", dtype=str, keep_default_na=False)
    fovs = pd.read_csv(folder / "fovs.csv.gz", dtype=str, keep_default_na=False)
    numeric = [*card["feature_cols"], *card.get("design", {}).get("non_features", [])]
    for column in numeric:
        if column in objects.columns:
            objects[column] = pd.to_numeric(objects[column])
    return LabDataset(id=dataset_id, objects=objects, fovs=fovs, card=card)


def _bullets(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


def _quoted(entry: Any) -> str:
    if isinstance(entry, dict):
        text = entry.get("value", "")
        if entry.get("quote"):
            text += f' — "{entry["quote"]}" ({entry.get("source", "")})'
        return text
    return str(entry)


def render(dataset_id: str) -> str:
    """DATASET.md of ``dataset_id``, generated from its card and stored tables."""
    data = load(dataset_id)
    card = data.card
    objects = data.objects
    design, counts = card["design"], card["design"].get("counts", {})
    pub_doi = card["publication"]["doi"]
    groups = [c for c in ("experiment", "condition", "biological_repeat") if c in objects]
    table = objects.groupby(groups).size().rename("cells").reset_index()
    # A condition within its experiment (each experimental set has its own control).
    per_condition = objects.groupby([g for g in groups if g != "biological_repeat"]).size()
    lines = [
        f"# {card['title']}",
        "",
        (
            f"`{card['id']}` · role: {card['role']} · licence: {card['source']['licence']} · "
            f"paper: [{pub_doi}](https://doi.org/{pub_doi}) · "
            f"data: [{card['source']['doi']}](https://doi.org/{card['source']['doi']})"
        ),
        "",
        (
            "_Generated from `card.toml` by `python data/lab/catalog.py render "
            f"{card['id']}`; edit the card, not this file._"
        ),
        "",
        "## 1. The experiment",
        "",
        (
            f"{card['biology']['cells']}; {card['biology']['conditions']}. "
            f"Figure: {card['publication']['figure']}. One row is one {card['observation_unit']}; "
            f"one field is {design['field']}. Readout: `{card['readout']}` "
            f"({card['columns']['features'].get(card['readout'], '')})."
        ),
        "",
        "## 2. Design at a glance",
        "",
        _bullets(
            [
                f"Experiment: {design.get('experiment', 'not applicable')}",
                f"Condition: {design['condition']}",
                f"Biological repeat: {design['biological_repeat']}",
                f"Technical repeat: {design['technical_repeat']}",
                f"Field: {design['field']}",
                f"Field table: {design['fov_table_source']}",
                f"Field selection: {card['acquisition']['field_selection']}",
                (
                    f"Stage coordinates: {card['acquisition']['stage_coordinates']}; "
                    f"acquisition order: {card['acquisition']['acquisition_order']}"
                ),
                (
                    f"Counts: {len(objects)} {card['observation_unit']}s, "
                    f"{objects['condition'].nunique()} conditions, "
                    f"{objects['biological_repeat'].nunique()} biological repeats; "
                    f"{per_condition.min()}-{per_condition.max()} per condition "
                    f"({counts.get('cells_per_condition', '')})"
                ),
            ]
        ),
        "",
        "Cells per condition and biological repeat:",
        "",
        "```text",
        table.assign(
            repeat=table.groupby([g for g in groups if g != "biological_repeat"]).cumcount() + 1
        )
        .pivot(
            index=[g for g in groups if g != "biological_repeat"], columns="repeat", values="cells"
        )
        .rename(columns=lambda c: f"repeat {c}")
        .astype("Int64")
        .to_string(),
        "```",
        "",
        "## 3. What the paper concluded from these data",
        "",
    ]
    for claim in card.get("claims", []):
        rep = claim["reproduction"]
        lines += [
            f"- **{claim['claim']}** ({claim['figure']}).",
            f'  > "{claim["quote"]}" — {claim["quote_source"]}',
            (
                f"  n: {claim.get('stated_n') or 'not stated'}; test: "
                f"{claim.get('stated_test') or 'not stated'}."
            ),
            f"  Reproduced from the stored tables: **{rep['status']}** — {rep['value']}"
            + (f" ({rep['note']})" if rep.get("note") else ""),
        ]
    lines += [
        "",
        f"How the paper analysed them: {_quoted(card['analysis_in_paper'].get('statistics', ''))}",
        "",
        "## 4. Curator's hypotheses for MorphoCoverage",
        "",
        f"_Written {card['hypotheses']['written']}. These are expectations to check, not answers._",
        "",
        _bullets(card["hypotheses"]["items"]),
        "",
        "## 5. Traps for an analyst",
        "",
        _bullets([c["text"] for c in card.get("caveats", [])]),
        "",
        "## 6. MorphoCoverage results",
        "",
        (
            f"Runs are stored in `data/lab/results/{card['id']}/` and logged in "
            "`data/lab/FINDINGS.md`."
        ),
        "",
        "## Provenance",
        "",
        _bullets(
            [
                (
                    f"Source: {card['source']['repository']} record {card['source']['record']} "
                    f"({card['source']['doi']}), {card['source']['licence']}, retrieved "
                    f"{card['source']['retrieved']}; file `{card['source'].get('member', '')}` "
                    f"(sha256 `{card['source']['member_sha256']}`)."
                ),
                (
                    f"Paper: {card['publication']['citation']} — quotes from "
                    f"{card['publication']['version_quoted']}."
                ),
                f"Acquisition: {_quoted(card['acquisition'].get('microscope', ''))}",
                (
                    f"Curated by {card['curator']} on {card['created']}; verified by "
                    f"{card.get('verified_by') or '(not yet verified)'}."
                ),
            ]
        ),
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str]) -> None:
    if argv[:1] == ["render"] and len(argv) == 2:
        (HERE / argv[1] / "DATASET.md").write_text(render(argv[1]))
        return
    if argv[:1] == ["check"]:
        bad = {i: check_card(read_card(i)) for i in ids()}
        for dataset_id, problems in bad.items():
            print(dataset_id, "ok" if not problems else "; ".join(problems))
        raise SystemExit(1 if any(bad.values()) else 0)
    raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
