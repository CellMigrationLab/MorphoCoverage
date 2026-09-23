# MorphoCoverage: project aim, scientific context, and validation strategy

## Executive aim

MorphoCoverage addresses a practical experimental-design problem in quantitative microscopy:

> **How can we determine whether a biological sample has been imaged sufficiently, and whether acquiring another representative field of view is likely to provide useful new phenotypic information?**

The project starts from a simple observation: microscopes acquire **fields of view (FOVs)**, not independent cells. Cells are observations nested inside FOVs, and cells from the same or nearby FOVs may be correlated because they share local environment, density, tissue region, illumination, preparation, and other spatial or technical factors.

MorphoCoverage therefore studies microscopy sampling at the acquisition level while retaining object-level phenotypic information.

The project has two connected aims:

1. **Sampling adequacy:** determine whether the FOVs acquired so far adequately represent a specified phenotypic representation of a sample.
2. **Smart microscopy:** determine whether that information can support prospective decisions about whether to acquire another representative FOV, and later whether acquisition can be allocated more efficiently.

The project deliberately does **not** assume in advance what mathematical method will solve these problems. Ecological coverage estimators, missing-mass methods, clustering, nearest-neighbour novelty, distributional distances, Bayesian approaches, information-theoretic quantities, and simple empirical stopping rules are candidate solutions to be compared, not assumptions built into the project definition.

## 1. Why this project is needed

### 1.1 Three different questions are often conflated

The phrase "how many cells are enough?" can refer to several distinct experimental-design questions.

#### Biological replication

Work such as SuperPlots emphasizes that individual cells from one experimental sample are generally not independent biological replicates. The number of biological replicates determines the experimental population to which inferential claims can be generalized.

This addresses:

> **How many independent experimental units support the biological conclusion?**

MorphoCoverage does not replace this level of experimental design.

#### Precision of a cell-derived sample summary

Zweifach's 2024 JCB article, *Determining how many cells to average for statistical testing of microscopy experiments*, addresses a second problem: once cells are being averaged within an experimental sample, how many cells should be measured so that the sample mean is estimated with sufficiently low averaging noise?

The framework is intentionally scoped to a parameter of interest that is a whole-cell property at a single time point, and explicitly assumes that samples are spatially uniform. It argues that reducing the cell-averaging contribution to approximately 2.5-5% coefficient of variation is often a reasonable target, depending on experimental design and other sources of variation.

This addresses:

> **How many cell measurements are needed to estimate a chosen scalar sample-level quantity sufficiently precisely?**

That is a useful problem and MorphoCoverage should reproduce it faithfully as a benchmark rather than caricature it.

#### Sampling a heterogeneous multidimensional population

MorphoCoverage addresses a third question:

> **How many FOVs are needed to adequately survey the phenotypic distribution represented in an imaging experiment?**

This is not necessarily equivalent to estimating a scalar mean precisely.

A sample mean can stabilize while a minority state, tail, spatially restricted phenotype, or alternative mode remains undiscovered. Conversely, some datasets may be sufficiently homogeneous that a conventional cell-number calculation and an FOV-aware phenotypic criterion give essentially the same recommendation.

A central goal of MorphoCoverage is therefore not to assume that existing cell-number guidance is inadequate, but to identify **when it is sufficient and when it is not**.

## 2. Central hypotheses to challenge

MorphoCoverage is motivated by two hypotheses that must be tested rather than protected.

### Hypothesis A: precision of a summary statistic is not the same as phenotypic sampling adequacy

For a scalar measurement, a sufficiently large number of cells can make the estimated sample mean highly precise.

However:

[
\text{precision of a sample mean}
\neq
\text{coverage of a multidimensional phenotype distribution}.
]

This difference should be demonstrated empirically, not asserted rhetorically.

### Hypothesis B: cell count is not always an adequate measure of microscopy sampling effort

Consider:

```text
Acquisition A
2 FOVs x 500 cells/FOV = 1000 cells

Acquisition B
20 FOVs x 50 cells/FOV = 1000 cells
```

If phenotype is spatially structured, these datasets may contain very different amounts of population information despite identical total cell counts.

But this is also a testable hypothesis. If realistic datasets show negligible within- and between-FOV structure, FOV-aware analysis may add little beyond cell count.

The aim is to identify **regimes** in which FOV structure matters.

## 3. Aim 1: quantify phenotypic sampling adequacy

The first aim is:

> **Given the FOVs observed so far, quantify the evidence that further representative acquisition would materially change our description of the phenotypic population.**

This is deliberately broader than "estimate percentage coverage." A universal percentage of biological phenotype may not be identifiable.

Candidate outputs may include:

- expected phenotypic novelty in another representative FOV;
- probability that another FOV contains substantially unsampled phenotype;
- expected change in the observed phenotype distribution;
- expected information gain;
- probability that additional FOVs materially alter an analysis result;
- a calibrated sampling-adequacy or stopping criterion.

Which target is most useful should be determined by validation.

### FOV is the operational acquisition unit, not automatically an independent unit

MorphoCoverage treats an FOV as the primary operational sampling unit because that is what the microscope samples spatially. Here, **one FOV means one independently selected XY acquisition position**. A z-stack, multichannel acquisition, or time series at that position remains one FOV.

This does **not** mean FOVs are assumed statistically independent. Adjacent, overlapping, or nearby FOVs can be strongly correlated.

When available, preserve:

- stage coordinates;
- FOV dimensions and physical imaged area;
- overlap or adjacency information;
- acquisition order;
- sampling design such as random, systematic-grid, manually selected, or targeted acquisition.

Whole-FOV resampling is safer than cell-level resampling for an FOV-level question, but it does not itself solve spatial autocorrelation.

### FOV number is not the only acquisition-effort variable

Preserve and report at least:

- number of FOVs;
- total imaged area when known;
- number of detected objects;
- distribution of objects per FOV.

Ten FOVs acquired with different magnification, field size, or tissue coverage are not automatically equivalent.

## 4. Aim 2: evidence-based smart microscopy

The second aim is:

> **Use validated sampling-adequacy estimates prospectively to decide whether additional representative imaging is worthwhile.**

The initial smart-microscopy problem is intentionally restricted to:

```text
Acquire representative FOV
        -> extract phenotype representation
        -> update sampling assessment
        -> continue or stop
```

A useful system should adapt acquisition effort to sample complexity:

- homogeneous samples should not automatically receive excessive imaging;
- heterogeneous or spatially structured samples should receive more imaging when additional FOVs continue to add information.

The aim is not simply to image less. A good system may deliberately image **more** than a fixed protocol when a sample is unusually heterogeneous.

### Later question: where to image next?

Only after representative stopping has been validated should the project investigate:

> **Which candidate FOV should be acquired next?**

This may use low-cost previews, stage coordinates, spatial modelling, active learning, or expected information gain.

Representative stopping and targeted acquisition must remain separate initially because targeted acquisition changes the sampling distribution.

A future system may therefore need separate streams:

```text
REPRESENTATIVE / SENTINEL FOVs
    used to estimate population sampling adequacy

TARGETED FOVs
    used to enrich rare or biologically interesting observations
```

Alternatively, acquisition probabilities must be modelled explicitly.

## 5. Keep the problem definition independent of the solution

MorphoCoverage must not begin by declaring one estimator to be "the MorphoCoverage method."

Candidate approaches include, but are not limited to:

- fixed cell-number rules;
- fixed FOV-number rules;
- scalar mean/variance stabilization;
- Zweifach-style averaging-noise criteria;
- empirical recent-FOV novelty;
- ecological richness, incidence, or sample-completeness estimators;
- Good-Turing or related missing-mass estimators;
- discrete morphotype accumulation;
- nearest-neighbour or radius-based feature-space coverage;
- density-based novelty;
- MMD;
- energy distance;
- Wasserstein or other optimal-transport distances;
- classifier-based two-sample tests;
- set/distribution representations of whole FOVs;
- information-theoretic criteria;
- Bayesian population models;
- calibrated predictive approaches.

A simpler method should be preferred if it performs as well as a complicated one.

## 6. Phenotype is representation-dependent

MorphoCoverage consumes precomputed numeric representations rather than defining phenotype directly from images.

Representations may include:

- interpretable morphometric features;
- intensity and texture features;
- Nyxus or CellProfiler measurements;
- DINO-family embeddings;
- microscopy-specific foundation-model embeddings;
- combined representations;
- other future feature extractors.

Sampling adequacy is therefore conditional on the representation:

[
\text{sampling adequacy}
=
f(
\text{population},
\text{acquisition design},
\text{representation},
\text{analysis method}
).
]

A sample may be well sampled for gross morphology while remaining incompletely sampled for texture or another representation.

MorphoCoverage must never present a result as a representation-independent percentage of "all biological phenotype."

## 7. Input model: preserve both FOVs and objects

An object-level table alone is insufficient because an acquired FOV containing zero detected objects would disappear.

MorphoCoverage should therefore support two linked logical tables.

### FOV table

One row per acquired FOV, including FOVs with zero objects.

Minimum information:

```text
fov_id
```

Optional information may include:

```text
condition
repeat
sample_id
well_id
batch_id
acquisition_order
stage_x
stage_y
fov_width_um
fov_height_um
imaged_area_um2
sampling_design
```

### Object table

One row per detected object:

```text
object_id
fov_id
feature_001
...
feature_N
```

Feature columns must always be explicitly supplied. Numeric metadata must never silently enter phenotype space.

## 8. Biological replicates and conditions

Sampling adequacy should normally be estimated within each biological replicate or sampling population.

For example:

```text
Control R1    adequately sampled
Control R2    adequately sampled
Control R3    additional imaging useful
```

must not become:

```text
Control       adequately sampled
```

simply because R1 and R2 contain many objects.

Condition-level inference is separate from acquisition completeness.

## 9. Direct benchmark against current "how many cells" practice

MorphoCoverage should explicitly reproduce and benchmark against the problem solved by the JCB cell-averaging framework.

### Reproduce the scalar-precision problem

For scalar object-level measurements, estimate how many cells are required to achieve chosen averaging-noise targets such as approximately 2.5% or 5% CV under the applicable assumptions.

Then ask:

> At the point where the scalar precision criterion says enough cells have been sampled, what relevant phenotypic information remains in unseen FOVs?

### Test the spatial-uniformity assumption

Create or identify datasets with the same overall cell-level distribution but different FOV structure:

- uniformly mixed cells;
- moderate FOV clustering;
- strong spatial clustering;
- rare states confined to a few FOVs.

This asks whether cell-number guidance remains adequate when spatial uniformity is violated.

### Test whether FOV clustering affects scalar mean precision

Simulate cells with controlled intraclass correlation within FOVs.

For the same total cell count, compare empirical scalar-mean error when cells are distributed across many FOVs versus concentrated within few FOVs.

If FOV clustering does not materially affect mean precision in realistic regimes, report that.

### Test scalar stability against multidimensional incompleteness

Construct populations in which conventional scalar summaries stabilize early while multidimensional structure remains undiscovered:

- a low-frequency distinct phenotype;
- mixtures with similar means but different modes;
- a spatially restricted phenotype;
- phenotypes separable only through combinations of features.

The comparison is not "JCB wrong versus MorphoCoverage right." It is:

> **Which experimental questions are answered by scalar cell-number guidance, and which require FOV-aware distributional sampling?**

### Allow the null result

The project must permit the possibility that, for many routine datasets, cell-number guidance or a fixed-FOV rule performs nearly as well as a more complex method.

Identifying those regimes is a useful outcome.

## 10. Core validation principle: predict unseen FOVs

The main validation question is:

> **Using only FOVs observed so far, can a method predict how much relevant information will be obtained from subsequently unseen FOVs?**

For a completed dataset containing FOVs (F_1,\ldots,F_n), simulate acquisition.

At step (k):

1. expose only the first (k) FOVs;
2. fit preprocessing using only information allowed at that point;
3. calculate a candidate sampling statistic;
4. make a prediction about additional acquisition;
5. reveal held-out FOVs;
6. quantify what was actually learned;
7. compare prediction with observation.

Repeat across many valid acquisition orders and datasets.

Future FOVs may be used to score the prediction but must not leak into the prospective calculation.

## 11. Retrospective, prospective, and deployment modes

These modes must remain distinct.

### Retrospective characterization

All FOVs already exist and may be used to describe realized accumulation after the fact.

### Prospective simulation

Only previously revealed FOVs may be used to calculate the quantity that would have been available during acquisition. Held-out FOVs are used only for evaluation.

### Online deployment

The method runs during live acquisition and must satisfy latency and computational constraints.

Hyperparameters, preprocessing, thresholds, dimensionality reduction, and estimator selection require appropriate train/validation/test separation when the goal is prospective use.

## 12. What does "new information" mean?

The project should test multiple operational targets.

### Object-level novelty

How many objects in a future FOV occupy regions of the selected representation that were insufficiently represented previously?

### FOV-distribution novelty

How different is the phenotype distribution in the new FOV from previously observed FOVs?

### Population-estimate change

Does adding the FOV materially change estimated means, variances, quantiles, clusters, densities, or another predefined population description?

### Decision stability

Would adding the FOV materially change an intended downstream biological analysis or conclusion?

No single target is guaranteed to dominate for all applications.

## 13. Real-data-first evaluation

Deeply sampled real microscopy datasets are the primary scientific benchmark. Constructed or synthetic examples remain useful for unit tests, explicit counterexamples, and regression tests after a real-data failure reveals a concrete weakness; they are not the main evidence that a strategy works.

Required scenarios include:

1. **Homogeneous population** — low incremental information should be reached rapidly.
2. **Multiple common phenotypes** — substantial modes should be discovered without arbitrary excessive imaging.
3. **Rare states** — e.g. 10%, 5%, 1%, 0.5%, 0.1%; quantify premature stopping.
4. **Same prevalence, different FOV clustering** — hold global prevalence constant while varying spatial organization.
5. **Variable cells per FOV** — compare sparse and dense images.
6. **Same total cells, different FOV count** — explicitly separate cell count from image count.
7. **Controlled within-FOV correlation** — quantify effects on scalar precision and multidimensional sampling.
8. **Correlated neighbouring FOVs** — show when FOV-block resampling remains optimistic.
9. **Empty FOVs** — preserve acquisition history when zero objects are detected.
10. **Unequal physical FOV area** — show when simple FOV count becomes misleading.
11. **Batch/acquisition drift** — reveal technical novelty.
12. **Segmentation censoring** — remove or distort rare states upstream.
13. **High-dimensional noise** — add irrelevant/correlated features.
14. **Representation disagreement** — compare acquisition recommendations across feature representations.

## 14. Baselines every candidate method must beat or justify exceeding

At minimum include:

- fixed cell counts;
- fixed FOV counts;
- fixed imaged area when available;
- Zweifach-style scalar precision;
- mean/variance stabilization;
- recent-FOV empirical novelty;
- distributional stabilization such as MMD, energy distance, or Wasserstein distance.

A complex estimator is justified only if it gives better calibration, safer stopping, better efficiency, or another clearly defined advantage.

## 15. Evaluation criteria

Candidate methods should be assessed using:

- predictive calibration;
- error in predicting held-out FOV information;
- premature-stopping rate;
- rare-state miss rate;
- over-imaging after effective saturation;
- acquisition efficiency;
- robustness to FOV order;
- robustness to cell density and spatial clustering;
- robustness across representations;
- computational cost for prospective use.

The benchmark should produce a **regime map**, not merely a leaderboard:

> When is cell-number guidance sufficient, when does simple FOV counting suffice, and when does a richer FOV-aware method add meaningful value?

## 16. Assumptions that must remain explicit

1. **Observed FOVs are representative of the target sampling process.** Manual selection can bias inference.
2. **Future FOVs follow the same sampling process** unless a changed process is explicitly modelled.
3. **The representation captures the phenotype of interest.** Information absent upstream cannot be recovered.
4. **Technical effects are controlled or identifiable.** Technical novelty can mimic biological novelty.
5. **Segmentation/patch extraction does not systematically censor important states**, unless that limitation is being tested.
6. **Phenotypic similarity is meaningfully represented by the chosen feature geometry.**
7. **FOVs may still be spatially correlated.** FOV-aware is not synonymous with independent.

## 17. Fundamental limitations

MorphoCoverage cannot prove that no undiscovered rare phenotype exists.

No finite sample can exclude arbitrarily rare unseen biology without strong assumptions.

Avoid claims such as:

> "95% of all biological phenotype has been captured."

unless that quantity has a precise, validated statistical definition.

A more defensible output is:

> **Under the specified representation and sampling design, additional representative FOVs are predicted to have low information yield.**

MorphoCoverage also cannot decide whether an unusual point in feature space is biologically meaningful rather than technical. Experimental QC and biological interpretation remain external requirements.

## 18. Falsification criteria

The central case for MorphoCoverage is weakened if:

- FOV structure adds negligible predictive value in realistic datasets;
- scalar precision criteria predict phenotypic saturation sufficiently well;
- fixed-FOV rules perform as well as richer estimators;
- candidate methods cannot predict unseen FOV information;
- representation dependence overwhelms any stable acquisition recommendation;
- online methods are too computationally expensive for practical use.

These are valid outcomes and should not be optimized away.

## 19. Initial success criterion

MorphoCoverage should be considered successful only if at least one approach can demonstrate:

1. useful prediction of information in unseen FOVs;
2. sensitivity to spatial/FOV structure when that structure matters;
3. improvement over simple baselines in identifiable regimes;
4. controlled premature stopping for user-defined rare-state discovery targets;
5. robustness across multiple representations or clear characterization of representation dependence;
6. computational feasibility for prospective use.

A particularly valuable result would be to establish **when the JCB cell-number approach is sufficient and when FOV-aware phenotypic sampling adds information it cannot capture**.

## 20. Ultimate vision

Current microscopy protocols often use fixed rules such as:

> "Measure at least 100 cells."

or:

> "Acquire 10 random fields."

MorphoCoverage aims to make this decision empirical:

[
\boxed{\textbf{Has this particular sample been imaged sufficiently?}}
]

and ultimately:

[
\boxed{\textbf{Would another image provide enough new biological information to be worthwhile?}}
]

The first question defines the statistical core of MorphoCoverage.

The second connects it to smart microscopy.


## 21. Sampling objective is part of the scientific question

MorphoCoverage must ask **enough for what?** before judging sampling adequacy. The core objectives are population representation, rare-phenotype discovery, and balanced sampling. Rare discovery should use explicit user-defined targets such as minimum phenotype prevalence, desired detection probability, and minimum independent FOV occurrences. See `sampling_objectives.md`.

## 22. Modular strategy comparison

MorphoCoverage is a strategy-comparison framework. Candidate methods implement one common interface and are executed through the same MorphoBricks-compatible validation, grouping, timing, and provenance layer. The same feature tables should therefore be evaluable by many strategies without changing data plumbing.

By default, canonical MorphoBricks `condition` and `repeat` columns define independent analysis groups, and `image_file` is preferred as the FOV identity. Pooling must be explicit.

Scientific performance and computational performance are benchmarked separately. Runtime is a first-class outcome because one intended application is online smart microscopy. See `benchmarking.md` and `architecture.md`.


### Multidimensional acquisitions

Feature extraction may be 2D, 3D, slice-by-slice, channel-specific, or time-resolved. MorphoCoverage does not prescribe how phenotype features are computed. If 2D features are generated over a stack or time series, preserve the relevant lowercase `z`, `c`, and/or `t` metadata and keep a shared spatial `position_id` for all rows from the same acquisition position. Sequential acquisition must reveal the whole position together.
