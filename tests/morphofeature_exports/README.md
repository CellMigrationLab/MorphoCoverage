# Golden MorphoFeature exports of the Spec fixtures

**Do not edit these files by hand.** Each folder is the `coverage/Cells` folder that
MorphoFeature wrote for one vendored Spec fixture (`tests/spec_fixtures/fixtures/valid/`),
copied unchanged, beside a `provenance.json` that records:

- `morphofeature_commit`: the MorphoFeature commit that wrote it (MF-5, MorphoFeature PR #206,
  branch `claude/stage2-mf5`);
- `fixture` and `fixture_blob`: the fixture and its git blob hash, which
  `tests/test_morphofeature_exports.py` checks against the vendored copy;
- `spec_commit`: the Spec commit of `tests/spec_fixtures/PIN.toml` when the export was made;
- the versions of the packages installed beside MorphoFeature.

| Folder | Fixture | What it shows |
|---|---|---|
| `v03` | `v03_paired_different_folder_names` | `CTRL/rep1` and `DrugA/experiment_23` are one repeat (`Day1`) |
| `v27` | `v27_descriptive_keys_collide` | `target` and `treatment_key` are descriptive only: no table carries them |
| `v30` | `v30_same_source_label_different_repeats` | `CTRL/rep1` pairs with `DrugA/rep2` (`Day1`), never `rep1` with `rep1` |
| `v33` | `v33_condition_folder_on_forbidden_list` | the condition folder `control` has the ID `ctrl` |

The FOV table (`fovs.csv`) carries `repeat_group`, `condition_id` and `technical_repeat_id`
beside the unchanged key `condition, biological_repeat, technical_repeat, position_id`, whose
`condition` and `biological_repeat` stay the source folder labels (experiment-v1 section 3.4).

## How they were made

[`make_exports.py`](make_exports.py) does what MorphoFeature's own
`tests/test_coverage_repeat_group.py` does: preprocessing's `make_described_run` writes the
experiment of each fixture with one `Cells` instance mask (two labelled squares per image), and
MorphoFeature's run file (`run_from_config`) measures it with its test fake `shape` provider
(`Cells__area` = the label value), tables written as CSV. It is not run by the test suite: it
needs a MorphoFeature checkout and its pinned dependencies (preprocessing `v0.5.0`, CellPatcher
`v0.3.0`, normalization `v0.3.4`, the `cli` extra). From the MorphoFeature checkout's root, in
its environment:

    python <MorphoCoverage>/tests/morphofeature_exports/make_exports.py --morphofeature . \
        --spec-fixtures <MorphoCoverage>/tests/spec_fixtures \
        --out <MorphoCoverage>/tests/morphofeature_exports

Two runs give byte-identical files. Re-make them, in a reviewed pull request, when MorphoFeature's
export or the vendored fixtures change.
