# MorphoCoverage documentation

MorphoCoverage has three workflows, one per moment of use. Each answers one question a
microscopist asks, and each fixes what information may answer it. Start with the one that
matches where you are; every workflow starts from the same declaration of the dataset,
[the setup](common/setup.md).

| Workflow | The microscopist asks | May use | May not use |
|---|---|---|---|
| **REPORT** | "I have finished imaging. What can I trust, and what can't I?" | every acquired FOV of this dataset | a statement about another FOV without a calibration of it; any claim about other samples |
| **OPTIMISE** | "What should I do next time?" | completed datasets, with outcomes judged on held-back FOVs | a best or minimal subset as a recommendation; tuning on held-out datasets; subset draws counted as samples |
| **LIVE** | "I am imaging now. Should I acquire another field?" | only the FOVs acquired so far | any future FOV: its identity, metadata, features, or its effect on scaling, PCA or thresholds |

## When to use which

- **REPORT** after an experiment: whether the fields were enough, whether the conditions
  differ, which fields to open, and what the dataset cannot tell you. `report` and
  `html_report`, or the guided session, which asks the questions for you.
- **OPTIMISE** once you have one or more completed datasets and plan the next experiment:
  how many fields, wells or repeats, and whether a stopping rule would have stopped too
  early. `optimise`, and `benchmark` to compare rules at the same risk of stopping too
  early, which compares methods and answers no question of its own.
- **LIVE** on the microscope, once you know what to stop on: after each field, continue
  or stop. `live_session` with a folder or tables in memory.

In one line each:

- **FOUNDATION: why count fields, and not only objects?** Cell-number rules
  assume a representative sample of objects; a microscope acquires objects
  through spatial fields. Does that difference matter for this dataset?
- **REPORT: can I trust the experiment I already performed?** Can I detect
  the phenotype? Did I image enough? What am I still missing? What limited me?
  Which images should I inspect? What can these data not tell me? Do the
  conditions differ, and how confidently?
- **OPTIMISE: how should I design the next experiment?** How many FOVs? How
  reliable is that number across samples? What did extra imaging buy? Could
  simpler rules work? Is the stopping rule safe? Could spatial sampling be
  better? More FOVs or more biological samples?
- **LIVE: should the microscope acquire another representative FOV now?**
  Continue or stop? For which objective? Why? How safe is that rule on
  retrospective validation? Is it fast and leakage-free?

A worked example of the workflows, with plots, is the notebook
[`notebooks/MorphoCoverage_Workflows_Example`](../notebooks/MorphoCoverage_Workflows_Example/MorphoCoverage_Workflows_Example.ipynb).
It runs on the bundled idr0028 dataset or on your own CSV files, locally or on
Google Colab. It follows the lab's LabConstrictor portability guide: settings
are ipywidgets (not Colab forms), and Colab-only setup sits behind a Colab check.
Locally it needs `ipywidgets` and, for the plots, `matplotlib`.

## Reading order

1. [The data format](common/data_format.md): the two tables MorphoCoverage reads.
2. [The setup](common/setup.md): the dataset declared once, for every workflow.
3. REPORT, in `report/`:
   - [Running a report](report/running_a_report.md): the design scan, the guided
     session, `report`, the HTML page;
   - [The report's questions](report/questions.md), in the order of the page;
   - [Did I image enough?](report/did_i_image_enough.md);
   - [Do the conditions differ?](report/conditions.md): the tests, several controls and
     the contrast table;
   - [Figures](report/figures.md) and [Feature tables and maps](report/feature_tables_and_maps.md),
     with the separation score;
   - [Data integrity](report/integrity.md);
   - [Objects or fields?](report/objects_or_fields.md), the Foundation's results in plain
     language.
4. OPTIMISE, in `optimise/`:
   - [Optimise](optimise/optimise.md): the workflow, its questions, the held-back outcomes
     and `effect_preservation`;
   - [Sampling objectives](optimise/sampling_objectives.md);
   - [Benchmarking](optimise/benchmarking.md): rules scored at matched risk.
5. LIVE, in `live/`:
   - [The live session](live/live_session.md): `live_session`, `LiveFolder`, `LiveTables`;
   - [Stopping rules](live/stopping_rules.md): reading a stop, and its calibration;
   - [Online acquisition](live/online_acquisition.md): the `OnlineSession` contract and
     timing.
6. Shared, in `common/`:
   - [What happens for what you provide](common/what_happens.md): which analyses each input
     switches on, the rules per type of feature, normalisation and testing in one place;
   - [Glossary](common/glossary.md), generated from `morphocoverage.glossary`;
   - [Validation protocol](common/validation.md): the pre-registered benchmark that chooses
     and validates each rule (P0-P11), and how the owner runs it;
   - [Project aim](common/project_aim.md) and [scientific contract](common/scientific_contract.md);
   - [Architecture](common/architecture.md);
   - [MorphoBricks FOV contract](common/morphobricks_fov_contract.md): what upstream tools
     need to emit;
   - [Roadmap](common/roadmap.md): the phases, milestones and audits, and how the
     repository stays clean;
   - [Open work](common/open_work.md), in order.

Records, kept as they were written: [preliminary findings](preliminary_findings.md),
[audits](audit/README.md) and [proposals](proposals/candidate_estimators_roadmap.md).

## Reading the question tables

Each workflow's page lists its questions with a code: R for REPORT, P for OPTIMISE, O for
LIVE. Status: **done** (implemented and reported), **partial** (implemented, but
part of the question is not answered or not reported), **gap** (not
implemented). **Benchmark** marks the questions where methods compete and
are scored against each other ([Benchmarking](optimise/benchmarking.md)); the others are
measurements or diagnostics to report, not to rank.

Two distinctions run through all three workflows:

- **What a rule says is not what the sample achieved.** A criterion's own
  threshold ("the averaging CV is below 5%") is apparent adequacy; held-back
  FOVs preserving the outcome is realized adequacy.
- **Retrospective representability is not the prospective sampling
  requirement.** A completed dataset may be represented by a few well-chosen
  FOVs, while a new experiment that samples FOVs at random needs more to meet
  the same criterion reliably.

Every answer is written for the microscopist, and three rules keep it so:

- **An answer, not only a table.** Each analysis prints the sentences of its
  answer (`summary()`), with the decision they support, and every sentence
  stays traceable to a table column.
- **Plain words.** The sentences say "fields", "held-back fields", "what you
  wanted to preserve" and "how often it would have worked"; reservoir,
  criterion, prefix, outcome metric and `config_id` appear only in the
  detailed tables. Every number comes with its unit (FOVs, % of acquisitions,
  µm, minutes).
- **Only scientific choices are asked.** What to preserve and how closely
  (for example "the well mean within 5%") are declared in words, with
  provisional presets that the summaries print; computational settings
  (outer splits, draws, seeds) have documented defaults sized for a result in
  minutes.
