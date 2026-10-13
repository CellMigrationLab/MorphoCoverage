"""The reference check of data/reference: its job list, its normaliser and a run twice."""

from __future__ import annotations

import base64
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "data" / "reference"
sys.path.insert(0, str(REFERENCE))
import reference
import reference_compare

OWNER_JOBS = """
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd() / "data" / "scenarios"))
import make_scenarios
from morphocoverage.workflows import Setup


def unnormalised():
    loaded = make_scenarios.load("m05_normalize_without_control")
    setup = Setup(**loaded.setup)
    return {"objects": loaded.objects, "fovs": loaded.fovs, "setup": setup, "title": "m05"}


JOBS = {"unnormalised": unnormalised}
"""
PDF = b"%PDF-1.4\n<< /Producer (Matplotlib) /CreationDate (D:20261013112751Z) >>\n"


def _page(stamp: str, pdf: bytes) -> bytes:
    """A page as html_report writes one: the generated line and a PDF carried in base64."""
    carried = base64.b64encode(pdf).decode()
    return (
        f'<p>Generated {stamp} UTC</p><script>["fig.pdf","application/pdf","{carried}"]</script>'
    ).encode()


def _write(root: Path, files: dict[str, bytes]) -> Path:
    for name, data in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_bytes(data)
    return root


def _tree(root: Path, stamp: str, pdf_date: str, seconds: int) -> Path:
    pdf = PDF.replace(b"20261013112751", pdf_date.encode())
    return _write(
        root,
        {
            "job/report.html": _page(stamp, pdf),
            "job/figures/fig.pdf": pdf,
            "job/report_short.txt": b"the report\n",
            "idr/benchmark_summary.txt": f"== mean (replay {seconds} s) ==\nscore 3 s\n".encode(),
            "lab/run.txt": f"settings: x\nrun time: {seconds} s\nmorphocoverage: 0.1\n".encode(),
            "job.log": f"took {seconds} s\n".encode(),
            "job/traceback.txt": f'File "/tmp/{seconds}/x.py", line {seconds}\n'.encode(),
        },
    )


def test_the_job_list_holds_the_whole_committed_tier():
    jobs = reference.read_jobs(REFERENCE / "jobs.txt")
    scenarios = {path.parent.name for path in (ROOT / "data" / "scenarios").glob("*/scenario.json")}
    lab = {path.parent.name for path in (ROOT / "data" / "lab").glob("*/card.toml")}
    assert len(scenarios) == 50 and len(lab) == 3
    expected = {f"scenario {name}" for name in scenarios} | {f"lab {name}" for name in lab}
    expected |= {"idr all", "synthetic synthetic", "morphobricks fiberfeatures"}
    assert sorted(jobs) == sorted(expected)


def test_selectors_name_a_kind_or_a_pattern_of_jobs():
    assert reference.selected("idr all", ["idr"])
    assert reference.selected("scenario e01_two_repeats", ["scenario/e0*"])
    assert not reference.selected("scenario a01_x", ["scenario/e0*", "lab"])


def test_the_normaliser_removes_the_stamps_and_nothing_else(tmp_path):
    before = _tree(tmp_path / "before", "2026-10-13 11:27", "20261013112751", 870)
    after = _tree(tmp_path / "after", "2027-01-02 03:04", "20270102030405", 12)
    assert reference_compare.differences(before, after) == []
    # the stamps are really there, and only they differ
    for name in ("job/report.html", "job/figures/fig.pdf", "lab/run.txt"):
        assert (before / name).read_bytes() != (after / name).read_bytes()
    normalised = reference_compare.normalised(before / "idr" / "benchmark_summary.txt")
    assert normalised == b"== mean (replay <time>) ==\nscore 3 s\n"
    assert reference_compare.normalised(before / "lab" / "run.txt") == b"settings: x\n"
    assert reference_compare.normalised(before / "job/figures/fig.pdf") == PDF.replace(
        b"/CreationDate (D:20261013112751Z)", b""
    )
    assert b"Generated" not in reference_compare.normalised(before / "job/report.html")


def test_a_changed_byte_is_reported_with_the_count(tmp_path, capsys):
    before = _tree(tmp_path / "before", "2026-10-13 11:27", "20261013112751", 870)
    after = _tree(tmp_path / "after", "2026-10-13 11:27", "20261013112751", 870)
    pdf = after / "job" / "figures" / "fig.pdf"
    pdf.write_bytes(pdf.read_bytes().replace(b"Matplotlib", b"Matplotlic"))
    (after / "job" / "report_short.txt").write_bytes(b"the report.\n")
    (after / "job" / "new.csv").write_bytes(b"a\n")
    # a stamp-like line that is not one of the stamps is compared
    run = after / "lab" / "run.txt"
    run.write_bytes(run.read_bytes().replace(b"settings: x", b"settings: y"))
    assert reference_compare.compare(before, after) == 1
    printed = capsys.readouterr().out.splitlines()
    assert printed == [
        f"only in {after}: job/new.csv",
        "differs: job/figures/fig.pdf",
        "differs: job/report_short.txt",
        "differs: lab/run.txt",
        "4 differences",
    ]
    assert reference.main(["compare", str(before), str(before)]) == 0


def test_two_runs_of_the_same_code_give_the_same_files(tmp_path, capsys):
    """A scenario with a page and an owner's job that fails, run twice: no difference after the
    stamps."""
    owner_jobs = tmp_path / "owner_jobs.py"
    owner_jobs.write_text(OWNER_JOBS)
    # the jobs only read the tree, so the repository itself serves as the tree here
    jobs = ["scenario a14_embed1_one_norm_image_none_rare"]
    jobs += reference.owner_job_lines(owner_jobs, ROOT)
    assert jobs[1:] == ["owner unnormalised"]
    for run in ("first", "second"):
        assert reference.run_jobs(ROOT, tmp_path / run, jobs, 2, owner_jobs) == 0
    first = tmp_path / "first"
    assert (first / "scenario__a14_embed1_one_norm_image_none_rare" / "report.html").is_file()
    record = (first / "owner__unnormalised" / "record.json").read_text()
    assert "failed at setup: ValueError: normalize needs control" in record
    assert reference_compare.compare(first, tmp_path / "second") == 0
    assert capsys.readouterr().out.endswith("0 differences\n")


def test_the_idr_job_never_writes_into_a_checkout(tmp_path):
    with pytest.raises(ValueError, match="export only"):
        reference.run_idr(ROOT, tmp_path / "idr", tmp_path / "idr.log")
