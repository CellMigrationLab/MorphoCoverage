"""Convert the Crick Enhancing-Reproducibility CellProfiler tables to MorphoBricks tables.

Source: https://github.com/FrancisCrickInstitute/Enhancing-Reproducibility
(``inputs/``, CC BY 4.0), CellProfiler outputs for IDR studies idr0028 (screen B)
and idr0139. See README.md in this directory for attribution and caveats.

Usage::

    python data/crick_idr/convert.py RAW_DIR OUT_DIR

RAW_DIR must contain the LFS-resolved files, named as downloaded by ``fetch()``.
Only the wells analysed in Marcotti et al. 2026 are kept (the same wells as
the Crick ``data_subsets``). The output is two linked tables per dataset: one
row per acquired field (including fields without nuclei) and one row per
nucleus. All nuclear and cytoplasmic CellProfiler intensity measurements are
kept, with their CellProfiler names prefixed by the compartment. Values are
written with 6 significant digits.

This is a data-preparation script, not part of the package.
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

import pandas as pd

MEDIA = (
    "https://media.githubusercontent.com/media/FrancisCrickInstitute/"
    "Enhancing-Reproducibility/main/inputs"
)
IDR0028_PLATES = ("1A", "2A", "2B")
IDR0139_PLATE = 1093711385
FLOAT_FORMAT = "%.6g"

# The wells analysed in Marcotti et al. 2026, exactly as selected by the Crick
# scripts subset_cp_data.py (idr0028) and subset_idr0139_data.py (idr0139),
# which build the data_subsets.zip used by their notebooks
# (https://zenodo.org/records/15727751).
IDR0028_PAPER_WELLS = {
    "1A": ["I23", "J23", "M23", "N23", "B6"],
    "2A": ["C13", "J23", "N23", "K16"],
    "2B": ["C13", "C17", "P13"],
}
IDR0139_PAPER_WELLS = ["J05", "I19", "G15", "O02", "B02", "N12", "L08", "L18", "H13", "E22"]
# Control-type labels used by the Crick notebook for idr0139 (companion_notebook_idr0139).
IDR0139_CONTROLS = {
    "Negative Control": "Untreated",
    "Neutral Control": "DMSO",
    "Stimulator Control": "Leptomycin b",
}


def normalize_well(well: str) -> str:
    """Zero-pad the column of a well name ("J5" -> "J05"), as the Crick notebooks do."""
    return f"{well[0]}{int(well[1:]):02d}"


def fetch(raw_dir: Path) -> None:
    """Download the LFS-resolved source files into ``raw_dir``."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "idr0028-screenB-annotation.csv": "idr/idr0028-screenB-annotation.csv",
        "idr0139-screenA-annotation.csv": "idr/idr0139-screenA-annotation.csv",
    }
    for plate in IDR0028_PLATES:
        base = f"cell_profiler_outputs/idr0028/screenB/LM2_GEFGAP_ONTARGETPlus_{plate}"
        files[f"idr0028_{plate}_index.tsv"] = (
            f"idr/LM2_GEFGAP_ONTARGETPlus_{plate}_ImageIndex.ColumbusIDX.csv"
        )
        for table in ("Image", "Nuclei", "Cytoplasm"):
            files[f"idr0028_{plate}_{table}.csv"] = f"{base}/{table}.csv"
    for table in ("Image", "Nuclei", "Cytoplasm"):
        files[f"idr0139_{table}.csv"] = f"cell_profiler_outputs/idr0139/{table}.csv"
    for name, path in files.items():
        target = raw_dir / name
        if not target.exists():
            urllib.request.urlretrieve(f"{MEDIA}/{path}", target)


def _objects(raw_dir: Path, prefix: str, image_numbers) -> pd.DataFrame:
    """Nuclei joined one-to-one with their cytoplasm; all intensity features, prefixed."""
    nuclei = pd.read_csv(raw_dir / f"{prefix}_Nuclei.csv")
    nuclei = nuclei[nuclei["ImageNumber"].isin(image_numbers)]
    intensity = [c for c in nuclei.columns if c.startswith("Intensity_")]
    nuclei = nuclei[["ImageNumber", "ObjectNumber", "Location_Center_X", "Location_Center_Y",
                     *intensity]].rename(columns={c: f"Nuclei_{c}" for c in intensity})
    cytoplasm = pd.read_csv(raw_dir / f"{prefix}_Cytoplasm.csv")
    cytoplasm = cytoplasm[cytoplasm["ImageNumber"].isin(image_numbers)]
    intensity = [c for c in cytoplasm.columns if c.startswith("Intensity_")]
    cytoplasm = cytoplasm[["ImageNumber", "Parent_Nuclei", *intensity]].rename(
        columns={"Parent_Nuclei": "ObjectNumber", **{c: f"Cytoplasm_{c}" for c in intensity}}
    )
    return nuclei.merge(
        cytoplasm, on=["ImageNumber", "ObjectNumber"], how="left", validate="one_to_one"
    )


def convert_idr0028(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    annotation = pd.read_csv(raw_dir / "idr0028-screenB-annotation.csv")
    fov_tables, object_tables = [], []
    for plate in IDR0028_PLATES:
        index = pd.read_csv(raw_dir / f"idr0028_{plate}_index.tsv", sep="\t")
        index = index[index["Channel"] == index["Channel"].min()]
        fovs = pd.DataFrame(
            {
                "repeat": plate,
                "well": index["WellName"].to_numpy(),
                "image_file": index["sourcefilename"].to_numpy(),
                "field": index["Field"].to_numpy(),
                "acquisition_time_s": index["AbsoluteTime@s"].to_numpy(),
                "stage_x_um": index["PositionX@um"].to_numpy(),
                "stage_y_um": index["PositionY@um"].to_numpy(),
                "area_um2": (
                    index["ImageWidth"] * index["ImageResolutionX@um"]
                    * index["ImageHeight"] * index["ImageResolutionY@um"]
                ).to_numpy(),
            }
        )
        wells = annotation[annotation["Plate"] == f"LM2_ONTARGETPlus_{plate}"].rename(
            columns={"Well": "well"}
        )
        # Condition labels as in the Crick idr0028 notebook: gene symbol, with
        # untargeted (negative-control) wells labelled MOCK.
        wells = wells.assign(
            condition=wells["Gene Symbol"].fillna("MOCK"),
            control_type=wells["Control Type"],
            quality_control=wells["Quality Control"],
        )[["well", "condition", "control_type", "quality_control"]]
        fovs = fovs[fovs["well"].isin(IDR0028_PAPER_WELLS[plate])]
        fovs = fovs.merge(wells, on="well", how="left", validate="many_to_one")
        # Acquisition order within a well: timestamp, ties broken by field number.
        fovs = fovs.sort_values(["well", "acquisition_time_s", "field"], kind="stable")
        fovs["acquisition_order"] = fovs.groupby("well").cumcount() + 1

        images = pd.read_csv(
            raw_dir / f"idr0028_{plate}_Image.csv", usecols=["ImageNumber", "Metadata_FileLocation"]
        )
        images["image_file"] = images["Metadata_FileLocation"].str.rsplit("/", n=1).str[-1]
        images = images[images["image_file"].isin(fovs["image_file"])]
        objects = _objects(raw_dir, f"idr0028_{plate}", images["ImageNumber"])
        objects = objects.merge(images[["ImageNumber", "image_file"]], on="ImageNumber",
                                validate="many_to_one")
        objects = objects.merge(fovs[["image_file", "condition", "repeat", "well"]],
                                on="image_file", validate="many_to_one")
        fov_tables.append(fovs)
        object_tables.append(objects)

    fovs = pd.concat(fov_tables, ignore_index=True)
    objects = pd.concat(object_tables, ignore_index=True).rename(
        columns={"ObjectNumber": "label_id", "Location_Center_X": "nucleus_x_px",
                 "Location_Center_Y": "nucleus_y_px"}
    )
    fov_cols = ["condition", "repeat", "well", "image_file", "field", "acquisition_order",
                "acquisition_time_s", "stage_x_um", "stage_y_um", "area_um2",
                "control_type", "quality_control"]
    feature_cols = [c for c in objects.columns if c.startswith(("Nuclei_", "Cytoplasm_"))]
    object_cols = ["condition", "repeat", "well", "image_file", "label_id",
                   "nucleus_x_px", "nucleus_y_px", *feature_cols]
    return fovs[fov_cols], objects[object_cols]


def convert_idr0139(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    annotation = pd.read_csv(raw_dir / "idr0139-screenA-annotation.csv")
    wells = annotation[annotation["Plate"] == IDR0139_PLATE].rename(columns={"Well": "well"})
    wells = wells.assign(well=wells["well"].map(normalize_well))
    control = wells["Control Type"].map(IDR0139_CONTROLS)
    wells = wells.assign(
        condition=control.fillna(wells["Proprietary Compound"]).fillna("unannotated"),
        control_type=wells["Control Type"],
        quality_control=wells["Quality Control"],
    )[["well", "condition", "control_type", "quality_control"]]

    images = pd.read_csv(
        raw_dir / "idr0139_Image.csv",
        usecols=["ImageNumber", "FileName_DNA", "Metadata_Well", "Metadata_Field"],
    )
    fovs = pd.DataFrame(
        {
            "repeat": str(IDR0139_PLATE),
            "well": images["Metadata_Well"],
            "image_file": images["FileName_DNA"],
            "field": images["Metadata_Field"],
            # No timestamps or stage positions in the source: field number is the
            # only available proxy for acquisition order.
            "acquisition_order": images["Metadata_Field"],
        }
    ).merge(wells, on="well", how="left", validate="many_to_one")
    keep = fovs["well"].isin(IDR0139_PAPER_WELLS).to_numpy()
    fovs, images = fovs[keep], images[keep]

    objects = _objects(raw_dir, "idr0139", images["ImageNumber"])
    objects = objects.merge(images[["ImageNumber", "FileName_DNA"]], on="ImageNumber",
                            validate="many_to_one").rename(columns={"FileName_DNA": "image_file"})
    objects = objects.merge(fovs[["image_file", "condition", "repeat", "well"]],
                            on="image_file", validate="many_to_one")
    objects = objects.rename(
        columns={"ObjectNumber": "label_id", "Location_Center_X": "nucleus_x_px",
                 "Location_Center_Y": "nucleus_y_px"}
    )
    feature_cols = [c for c in objects.columns if c.startswith(("Nuclei_", "Cytoplasm_"))]
    fov_cols = ["condition", "repeat", "well", "image_file", "field", "acquisition_order",
                "control_type", "quality_control"]
    object_cols = ["condition", "repeat", "well", "image_file", "label_id",
                   "nucleus_x_px", "nucleus_y_px", *feature_cols]
    return fovs[fov_cols], objects[object_cols]


def main(raw_dir: Path, out_dir: Path) -> None:
    fetch(raw_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, convert in (("idr0028_screenB", convert_idr0028), ("idr0139", convert_idr0139)):
        fovs, objects = convert(raw_dir)
        fovs.to_csv(out_dir / f"{name}_fovs.csv.gz", index=False, float_format=FLOAT_FORMAT)
        objects.to_csv(out_dir / f"{name}_nuclei.csv.gz", index=False, float_format=FLOAT_FORMAT)
        print(f"{name}: {len(fovs)} fields, {len(objects)} nuclei")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
