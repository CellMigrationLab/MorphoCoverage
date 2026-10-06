# Scenarios: what you provide, and what the report does with it

`make_scenarios.py` builds one small acquisition for each kind of input a user gives
MorphoCoverage, so that every path through the report runs on data whose answers are known.
`docs/common/what_happens.md` says which analyses each input switches on; these folders
check it.

## What is in a folder

- `fovs.csv`: one row per acquired field;
- `observations.csv`: one row per observation, the explainable features declared;
- `<table>.csv`: each other feature table, explainable or not (an embedding);
- `scenario.json`: the `Setup` arguments as plain JSON, the feature tables with their tags,
  and the truth: the analyses that must run and those that must not, the reason each
  question that cannot be answered gives, the refusal expected, and the planted effects.

The tables come from a latent effect per observation: a condition's shift on a log scale,
plus a repeat's, a well's, a field's and the observation's own noise. `area` is
`10 exp(effect + noise)`, so a shift of 0.8 more than doubles it; `ecc` never depends on
the effect. Each table is a few hundred rows at most.

## Three kinds of scenario

- **axes** (`a01` to `a25`): the tables (explainable only; two tagged explainable tables;
  one embedding; two embeddings with the same column names; a mix) x the controls (none;
  one; several with a treatment key, a parent, a master and one repeat without the second
  control) x normalisation x the unit (cells; one image per field) x the wells (none;
  technical repeats) x the target (the default tolerances; a fingerprint on explainable
  features; a fingerprint on an embedding; a labelled rare phenotype; a positive control).
  Every pair of values is in at least one scenario, every value in two or more.
- **edge** (`e01` to `e13`): designs the axes do not reach.
- **mistake** (`m01` to `m07`): what users get wrong, and what the package does now.

## The scenarios

<!-- scenarios -->
| Scenario | Kind | Input | What it checks |
| --- | --- | --- | --- |
| `a01_tagged_several_image_none_rare` | axes | tables tagged, controls several, normalize False, unit image, wells none, target rare | features; own controls and the rows across experiments; the questions about fields not applicable; the rare label's share and clustering |
| `a02_expl_several_norm_cells_tech_positive` | axes | tables expl, controls several, normalize True, unit cells, wells tech, target positive | features; own controls and the rows across experiments; normalised and centred views; a positive control; the page |
| `a03_mix_one_norm_image_none_fp_dino` | axes | tables mix, controls one, normalize True, unit image, wells none, target fp_dino | features; each condition against the control; normalised and centred views; the questions about fields not applicable; a fingerprint on DINO1 |
| `a04_embed1_none_cells_tech_fp_dino` | axes | tables embed1, controls none, normalize False, unit cells, wells tech, target fp_dino | embedding-only report; no condition test; a fingerprint on DINO1 |
| `a05_embed2_one_cells_tech_fp_dino` | axes | tables embed2, controls one, normalize False, unit cells, wells tech, target fp_dino | embedding-only report; each condition against the control; a fingerprint on DINO1 |
| `a06_expl_none_image_none_tol` | axes | tables expl, controls none, normalize False, unit image, wells none, target tol | features; no condition test; the questions about fields not applicable; the mean's tolerances |
| `a07_tagged_one_norm_cells_tech_tol` | axes | tables tagged, controls one, normalize True, unit cells, wells tech, target tol | features; each condition against the control; normalised and centred views; the mean's tolerances; the page, its CSVs naming tables |
| `a08_embed1_several_norm_image_tech_fp_dino` | axes | tables embed1, controls several, normalize True, unit image, wells tech, target fp_dino | embedding-only report; own controls and the rows across experiments; normalised and centred views; the questions about fields not applicable; a fingerprint on DINO1 |
| `a09_mix_one_cells_tech_rare` | axes | tables mix, controls one, normalize False, unit cells, wells tech, target rare | features; each condition against the control; the rare label's share and clustering; the page, its CSVs naming tables |
| `a10_embed2_several_norm_cells_none_tol` | axes | tables embed2, controls several, normalize True, unit cells, wells none, target tol | embedding-only report; own controls and the rows across experiments; normalised and centred views; the mean's tolerances; the page |
| `a11_mix_several_image_none_positive` | axes | tables mix, controls several, normalize False, unit image, wells none, target positive | features; own controls and the rows across experiments; the questions about fields not applicable; a positive control |
| `a12_embed2_none_image_tech_rare` | axes | tables embed2, controls none, normalize False, unit image, wells tech, target rare | embedding-only report; no condition test; the questions about fields not applicable; the rare label's share and clustering |
| `a13_mix_none_cells_none_fp_expl` | axes | tables mix, controls none, normalize False, unit cells, wells none, target fp_expl | features; no condition test; a fingerprint on the explainable features |
| `a14_embed1_one_norm_image_none_rare` | axes | tables embed1, controls one, normalize True, unit image, wells none, target rare | embedding-only report; each condition against the control; normalised and centred views; the questions about fields not applicable; the rare label's share and clustering; the page |
| `a15_expl_one_cells_tech_rare` | axes | tables expl, controls one, normalize False, unit cells, wells tech, target rare | features; each condition against the control; the rare label's share and clustering |
| `a16_tagged_several_norm_cells_none_fp_dino` | axes | tables tagged, controls several, normalize True, unit cells, wells none, target fp_dino | features; own controls and the rows across experiments; normalised and centred views; a fingerprint on DINO1 |
| `a17_tagged_one_norm_image_tech_positive` | axes | tables tagged, controls one, normalize True, unit image, wells tech, target positive | features; each condition against the control; normalised and centred views; the questions about fields not applicable; a positive control |
| `a18_tagged_none_image_tech_fp_expl` | axes | tables tagged, controls none, normalize False, unit image, wells tech, target fp_expl | features; no condition test; the questions about fields not applicable; a fingerprint on the explainable features |
| `a19_expl_none_cells_tech_fp_expl` | axes | tables expl, controls none, normalize False, unit cells, wells tech, target fp_expl | features; no condition test; a fingerprint on the explainable features |
| `a20_expl_one_image_tech_fp_dino` | axes | tables expl, controls one, normalize False, unit image, wells tech, target fp_dino | features; each condition against the control; the questions about fields not applicable; a fingerprint on DINO1 |
| `a21_embed1_one_norm_cells_none_tol` | axes | tables embed1, controls one, normalize True, unit cells, wells none, target tol | embedding-only report; each condition against the control; normalised and centred views; the mean's tolerances |
| `a22_mix_none_cells_tech_tol` | axes | tables mix, controls none, normalize False, unit cells, wells tech, target tol | features; no condition test; the mean's tolerances |
| `a23_embed2_several_image_tech_fp_dino` | axes | tables embed2, controls several, normalize False, unit image, wells tech, target fp_dino | embedding-only report; own controls and the rows across experiments; the questions about fields not applicable; a fingerprint on DINO1 |
| `a24_expl_one_norm_image_none_fp_expl` | axes | tables expl, controls one, normalize True, unit image, wells none, target fp_expl | features; each condition against the control; normalised and centred views; the questions about fields not applicable; a fingerprint on the explainable features |
| `a25_mix_several_norm_cells_tech_fp_expl` | axes | tables mix, controls several, normalize True, unit cells, wells tech, target fp_expl | features; own controls and the rows across experiments; normalised and centred views; a fingerprint on the explainable features |
| `e01_two_repeats` | edge | 2 conditions, 2 repeats, 12 fields, 60 rows; DINO1 (non-explainable) | two repeats: the paired tests and the whole-table test need 3, said, nothing tested |
| `e02_control_in_other_repeats` | edge | 2 conditions, 4 repeats, 12 fields, 60 rows; DINO1 (non-explainable) | a control imaged in other repeats than its condition: not compared across repeats |
| `e03_one_condition` | edge | 1 condition, 4 repeats, 12 fields, 60 rows; DINO1 (non-explainable) | one condition: no comparison, no separation, said why; the rest runs |
| `e04_screen_30_conditions` | edge | 31 conditions, 3 repeats, 279 fields, 279 rows | a screen of 30 conditions against one control, the readout alone: its tests corrected over the conditions, every condition compared |
| `e05_no_condition_in_every_repeat` | edge | 3 conditions, 4 repeats, 27 fields, 135 rows; DINO1 (non-explainable) | no condition in every repeat: compared where both are; the other tables not centred |
| `e06_chain_of_parent_controls` | edge | 6 conditions, 3 repeats, 54 fields, 216 rows | controls in a chain of parents: each condition against its own control, each control against its parent |
| `e07_tables_of_one_and_two_columns` | edge | 2 conditions, 4 repeats, 32 fields, 160 rows; single (explainable), pair (non-explainable) | a table of 1 column and one of 2: tested, the 2 columns fingerprinted; no separation line of their own |
| `e08_counts_with_many_zeros` | edge | 2 conditions, 4 repeats, 24 fields, 120 rows | a count feature, mostly zeros, beside the readout: compared like any feature; two columns, so no separation score |
| `e09_signed_feature_normalised` | edge | 2 conditions, 4 repeats, 24 fields, 120 rows | a signed feature divided by its control mean: refused, the mean is negative |
| `e10_features_per_z_plane` | edge | 2 conditions, 4 repeats, 24 fields, 360 rows | one row per cell and plane, split per plane (planes_as_features): one feature per plane |
| `e11_unequal_fields_per_repeat` | edge | 2 conditions, 4 repeats, 46 fields, 230 rows | very unequal fields per repeat: the cell-level screen also given balanced |
| `e12_missing_values_in_one_table` | edge | 2 conditions, 4 repeats, 24 fields, 120 rows; DINO1 (non-explainable) | missing values in one table only: counted and kept, each analysis says how many it leaves out |
| `e13_known_effects_six_repeats` | edge | 3 conditions, 6 repeats, 54 fields, 270 rows; DINO1 (non-explainable) | a strong effect found on the features and the embedding, a null condition not |
| `m01_dino_tagged_explainable` | mistake | 2 conditions, 3 repeats, 18 fields, 90 rows; DINO1 (explainable) | an embedding tagged explainable: run, its columns tested one by one, as declared |
| `m02_table_keyed_by_wrong_identity` | mistake | 2 conditions, 4 repeats, 24 fields, 120 rows; DINO1 (non-explainable) | a table keyed by the wrong identity (labels from 0): refused, unmatched observations |
| `m03_unknown_tag` | mistake | 2 conditions, 4 repeats, 24 fields, 120 rows; fiber (explanable) | a misspelt tag: refused, naming the two tags |
| `m04_several_controls_no_members` | mistake | 4 conditions, 4 repeats, 48 fields, 240 rows | several controls with no condition under them: refused at the Setup |
| `m05_normalize_without_control` | mistake | 2 conditions, 4 repeats, 24 fields, 120 rows | normalisation without a control: refused at the Setup |
| `m06_planes_given_as_rows` | mistake | 2 conditions, 4 repeats, 24 fields, 360 rows | one row per cell and plane given as is: refused, naming planes_as_features |
| `m07_positive_control_seen_too_few_fields` | mistake | 2 conditions, 3 repeats, 36 fields, 180 rows | a positive control after 2 reference fields, where the definition of new needs 5: refused at the Setup, naming the minimum |
<!-- /scenarios -->

## How it is used

`tests/test_scenarios_matrix.py` loads each folder (`make_scenarios.load`), builds the
`Setup`, runs `report`, and `html_report` for a few scenarios that between them hold every
kind of table, several controls, normalisation and a rare label, then checks the truth. It
checks structure and reasons, not numbers; the planted effects only where the design makes
the answer near-certain (a strong effect, six repeats). Another test checks that the
generator writes the committed files byte for byte.

To regenerate the folders and the table above, run this from the repository root:

```bash
python data/scenarios/make_scenarios.py
```
