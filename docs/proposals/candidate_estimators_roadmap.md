# Proposal: MorphoCoverage candidate estimators and implementation roadmap

## Status

This document is a **proposal**. It records candidate methods, benchmark requirements, and an implementation order for discussion. It does not select a default estimator and does not imply that every listed method should be implemented.

**Implemented so far (2026-09-25):** fixed evaluation reservoirs and `reference_error_floor` (0.2, `morphocoverage.reservoir`); multivariate outcomes (0.3, `multivariate_outcomes`, `distances`); dataset roles (part of 0.6, `benchmark`); the fixed-area baseline (0.5, 5.3, `FixedAreaStrategy`); the FOV bootstrap of the centroid, covariance, quantiles, PCA eigenspectrum and subspace, with a spatial block option (1.1–1.3, 1.6–1.7, 8.4, `FOVBootstrapStrategy`); epsilon support novelty (4.2, `SupportNoveltyStrategy`); Chao–Jost incidence coverage, Chao2, the Good–Turing estimate and next-FOV extrapolation on fixed labels (5.4, 5.6, 5.8, `IncidenceCoverageStrategy`); FOV-incidence planning (7.1, `FOVIncidencePlanStrategy`); beta-binomial incidence with recurrence (7.3–7.4, `RareIncidenceStrategy`); the semivariogram (8.2, `morphocoverage.spatial`). Not implemented: biological-fidelity outcomes (0.4, Phase 10), learning curves (2.2, 6.1), C2ST, kNN/adaptive/PRDC support, abundance Good–Turing, Chao1, ICE, Hill numbers, FOV embeddings, occupancy models, spatial ICC, pMoSS and Phase 11. No held-out dataset is registered.

The central rule is:

> **MorphoCoverage should choose sampling methods by how well they preserve scientifically relevant information in unseen data, not by how smoothly their own internal statistic appears to converge.**

A useful final result may be a regime map rather than one universal winner.

---

## 1. Scientific question

MorphoCoverage asks:

> **Given the FOVs acquired so far, has the experiment adequately sampled the declared phenotypic representation, and is another representative FOV likely to provide enough useful information to justify acquiring it?**

This is not the same as asking:

- whether a scalar mean is precise;
- whether two conditions are statistically different;
- whether the most recent FOV resembles previous FOVs;
- whether no rare phenotype can possibly remain unseen.

MorphoCoverage therefore needs several estimator families and a benchmark that can falsify them.

The project should compare methods drawn from:

- microscopy experimental design;
- survey and cluster sampling;
- ecological sample coverage and unseen-species estimation;
- support and novelty estimation;
- distribution estimation and two-sample statistics;
- predictive modelling;
- spatial statistics;
- sequential decision theory.

The objective is to learn **which methods work under which experimental regimes**.

---

## 2. Three architectural roles must remain separate

A major design requirement is to distinguish:

### 2.1 Core prospective estimator

A core estimator operates on one analysis group at a time and sees only the currently visible FOVs.

It asks:

> **How much uncertainty or unrepresented phenotype remains in this sample under the stated sampling objective?**

Examples:

- FOV-bootstrap uncertainty;
- support novelty;
- incidence-based sample coverage;
- predictive FOV uncertainty.

The estimator must not inspect future FOVs or comparison groups.

### 2.2 Multi-group planning comparator

Some published sample-size methods require several samples or conditions.

These should not be forced into the one-group estimator interface.

Instead, they can be implemented as explicit **planning comparators** that use declared pilot data and then return a fixed sampling plan.

The existing multi-sample JCB implementation already follows this pattern:

```text
multi-group pilot data
        |
        v
planning function
        |
        v
fixed plan
        |
        v
group-local stopping strategy
```

pMoSS could eventually be treated similarly if we decide it is worth implementing.

### 2.3 Benchmark outcome

Benchmark outcomes are retrospective and may use information unavailable to the prospective estimator.

They ask:

> **If acquisition had stopped here, how much scientifically relevant information would have been lost?**

This may include:

- population-distribution error;
- rare-state loss;
- FOV-level representation error;
- effect-size error between conditions;
- preservation of treatment-response ranking;
- classification or retrieval stability where scientifically appropriate.

This distinction is essential.

MorphoCoverage does **not** need to become a general biological-statistics package, but it must be allowed to evaluate whether a sampling strategy preserves the biological conclusion the experiment was designed to support.

---

## 3. What "best" should mean

The best sampling method is not necessarily the method with the smallest value of its own internal statistic.

Candidate strategies should be judged on several independent dimensions:

$
\text{population fidelity}
$

$
\text{rare-state protection}
$

$
\text{downstream biological fidelity}
$

$
\text{acquisition effort}
$

$
\text{computational cost}
$

Different experiments may weight these differently.

For example:

- a treatment screen may prioritize preservation of treatment effects;
- an atlas may prioritize population representation;
- a rare-event experiment may prioritize detection probability;
- live smart microscopy may impose strict latency constraints.

MorphoCoverage should therefore report Pareto trade-offs and regime maps rather than collapse everything into one arbitrary score.

---

## 4. Benchmark design comes before the method zoo

The first implementation priority is not another estimator. It is a benchmark capable of rejecting bad estimators.

### 4.1 Deeply sampled real data

The current Crick/IDR test data are useful for integration tests and preliminary analysis, but they are not sufficient as the primary scientific benchmark:

- idr0139 has only 4 FOVs per well;
- the selected idr0028 screen-B wells have 28 FOVs per well.

The benchmark programme should identify several deeper real microscopy datasets spanning different regimes, ideally including:

- relatively homogeneous cultured cells;
- heterogeneous cell populations;
- strongly spatially structured samples;
- rare states;
- variable density;
- multiple phenotype representations;
- several biological or technical replicates;
- recorded stage coordinates where possible.

Dataset roles must be fixed as development or held-out before method tuning.

### 4.2 Fixed evaluation reservoir

A key problem with comparing the acquisition prefix $F_{1:k}$ directly with the full acquisition $F_{1:n}$ is that the reference contains the prefix itself. Error therefore approaches zero mechanically as $k \to n$.

For primary scientific evaluation, use a reference reservoir that is never visible to the prospective estimator.

```text
deep acquisition
      |
      +---- prospective acquisition stream
      |       F1 ... Fk
      |       estimator sees only these
      |
      +---- fixed evaluation reservoir
              never visible to estimator
```

The split must respect the acquisition structure.

- When stage coordinates are available, split by **spatial blocks**, not individual neighbouring FOVs, so correlated neighbours do not appear on opposite sides of the benchmark.
- Repeat evaluation over several **predeclared block splits** so conclusions do not depend on one fortunate partition.
- When coordinates are unavailable, state explicitly that exchangeability of FOVs is an assumption of the split.
- A second independently prepared acquisition or biological repeat should not be treated automatically as a same-population reservoir: batch and biological shifts introduce error that no within-sample stopping rule can remove. Such data are better used as a separate cross-acquisition robustness outcome unless the sampling population is demonstrably shared.

The reservoir is finite and therefore has its own sampling error. Benchmark code should estimate a **reference error floor**, for example by repeated split-half or resampling comparisons inside the reservoir. An adequacy tolerance must be meaningfully above that floor; otherwise methods are being asked to predict the empirical reference more accurately than the reference estimates itself.

The reference must remain fixed as $k$ increases.

### 4.3 Immediate next-FOV utility

A separate short-horizon outcome should quantify the realized utility of the next unseen FOV $F_{k+1}$.

For a predeclared held-out loss $L$ against the fixed evaluation reservoir $R$, one useful outcome is:

$$
\Delta L_{k+1}
=
L(P_{1:k},R)
-
L(P_{1:k+1},R).
$$

Positive $\Delta L_{k+1}$ means that adding the next FOV improved representation of the fixed reservoir; negative values are possible.

Other predeclared next-FOV utilities may include:

- discovery or recurrence of an independently labelled rare state;
- reduction in error of a scalar or multivariate population functional;
- reduction in downstream biological-fidelity error.

This is a **retrospective outcome**, not a prospective estimator. It may mechanically become small late in acquisition, so it is interpreted together with matched-effort curves and the fixed-reservoir outcomes rather than used alone as proof of completeness.

### 4.4 Compare methods at matched effort

Do not judge a method by correlation between its internal evidence and an outcome that also changes automatically with $k$.

Instead compare:

- outcome error versus FOV count;
- outcome error versus imaged area;
- outcome error versus observation count;
- threshold sweeps producing stop-effort/error curves;
- premature-stop risk at matched acquisition budgets;
- acquisition savings at matched scientific error.

### 4.5 "Adequate" must be explicit

Terms such as "premature stopping" and "excess FOVs" require a declared adequacy criterion.

Possible benchmark formulations include:

- absolute tolerance on held-out distribution error, set above the estimated finite-reservoir error floor;
- relative tolerance compared with a deep reference;
- tolerance on rare-state detection probability;
- tolerance on downstream effect-size error;
- Pareto-front analysis without imposing one universal adequacy threshold.

The benchmark must not choose tolerances after seeing which method wins.

---

## 5. Existing baselines and current terminology

Published methods should keep their published names only when the corresponding statistical problem is reproduced faithfully.

### 5.1 Fixed observation count

Stop after a fixed number of phenotype observations.

This represents common "measure N cells" practice.

### 5.2 Fixed FOV count

Stop after a fixed number of spatial acquisition positions.

### 5.3 Fixed imaged area

Implemented as `FixedAreaStrategy` (nominal summed area; overlapping FOVs count twice).

When physical FOV area is available, stop after

$
A_k = \sum_{j=1}^{k} A_j.
$

This is more comparable than raw FOV count when field size differs.

### 5.4 JCB/Zweifach scalar precision

MorphoCoverage already implements the single-sample scalar-precision problem described by Zweifach:

$
SE(\bar{x}) \approx \frac{s}{\sqrt{n}}.
$

The package contains:

- the published object-level criterion;
- an explicitly labelled FOV-unit MorphoCoverage variant;
- the multi-sample planning workflow;
- `PlannedCellCountStrategy`.

The FOV-unit implementation already reports a design effect. Future spatial work should extend or interpret this quantity rather than duplicate it as though it were absent.

Reference:

Zweifach A. Determining how many cells to average for statistical testing of microscopy experiments. *Journal of Cell Biology*. 2024;223(8):e202401074. https://doi.org/10.1083/jcb.202401074

### 5.5 Existing descriptor-stability strategies

Current code names should remain the primary names until a deliberate API rename is approved:

- `DescriptorPlateauStrategy`;
- `DescriptorSpreadStrategy`.

Documentation should describe what they compute:

- `DescriptorPlateauStrategy`: **cumulative descriptor range** over a recent window;
- `DescriptorSpreadStrategy`: **resampled descriptor spread**.

They are inspired by sample-size analyses in Marcotti and colleagues' bioimage-statistics work, but they are MorphoCoverage operationalizations rather than published stopping rules.

Reference:

Marcotti S, Gerontogianni L, Kelly G, Barry DJ. Practical statistics for bioimage analysis - a guide to experimental design and data interpretation. *Journal of Cell Science*. 2026;139(10):jcs264367. https://doi.org/10.1242/jcs.264367

A separate API rename should not be bundled into this proposal.

---

## 6. Important negative result: stationarity is not completeness

Several attractive methods can be misused if their estimand is not stated carefully.

Suppose FOVs are independent draws from one stationary phenotype population $P$.

Then an early block and a recent block both satisfy approximately:

$
X_{\mathrm{early}} \sim P,
\qquad
X_{\mathrm{recent}} \sim P.
$

A two-sample method such as:

- MMD;
- energy distance;
- sliced Wasserstein;
- classifier two-sample testing;

should therefore approach its null behaviour even if a rare component of $P$ has not yet been observed.

Thus:

> **Similarity between recent and earlier FOV blocks measures stationarity/consistency of observed sampling, not completeness of the latent phenotype population.**

These methods remain useful, but not automatically as standalone stopping rules.

They can serve as:

- drift diagnostics;
- independent benchmark outcomes;
- predictive-error components;
- distributional loss functions in cross-FOV prediction.

Observed data cannot identify the phenotype or location of a **specific** state that has never appeared. Frequency-based methods such as Good-Turing or Chao can nevertheless infer aggregate unseen mass or richness from observed rare-frequency patterns under their sampling assumptions.

---

## 7. Important negative result: cumulative next-step change is mechanically diluted

Do not define coverage from

$
d(P_{1:k}, P_{1:k+1})
$

without correcting for the fact that one FOV contributes a fraction of order $1/k$ to the cumulative sample.

That quantity tends to shrink even when FOV heterogeneity remains substantial.

Similarly, the expected distance between two independent FOV distributions is generally a property of the underlying sample heterogeneity and need not converge to zero with more acquisition.

Therefore the target for FOV-predictive methods should be **uncertainty in the inferred population representation**, not raw FOV-to-FOV distance.

Examples of better targets include:

- uncertainty of a population functional under whole-FOV resampling;
- cross-FOV predictive risk;
- posterior/predictive uncertainty where a probabilistic model is justified;
- expected error against a held-out FOV reservoir.

---

## 8. Proposed implementation phases

### Phase 0 — benchmark and data infrastructure

Implement before expanding the estimator zoo.

#### 0.1 Deep benchmark datasets

Find and register multiple deeply sampled real microscopy datasets.

#### 0.2 Fixed evaluation reservoirs

Support benchmark partitions in which one subset drives prospective replay and another remains untouched for scoring.

#### 0.3 Multivariate outcome metrics

Add strategy-independent outcomes that can evaluate:

- centroid error;
- covariance error;
- quantile error;
- Fréchet-type error;
- MMD/energy/sliced-Wasserstein error to held-out data.

These are outcomes first, not assumed stopping rules.

#### 0.4 Biological-fidelity outcomes

Design a benchmark extension that can evaluate several stopped groups jointly.

Examples:

- treatment-effect error;
- preservation of effect direction;
- ranking stability across perturbations;
- rare-state frequency difference;
- condition-separation performance.

These outcomes are retrospective and must never leak into the prospective estimator.

#### 0.5 Fixed imaged area baseline

Add when physical area is available. Implemented as `FixedAreaStrategy`.

#### 0.6 Benchmark protocol lock

Before tuning many estimators, predefine:

- dataset roles;
- allowed transforms;
- threshold-search procedure;
- primary outcome families;
- acquisition-effort measures;
- reporting format.

---

### Phase 1 — simple population-estimation uncertainty

The first new estimators should answer:

> **How uncertain is our current estimate of simple multivariate population properties when FOV is treated as the sampling unit?**

#### 1.1 FOV-bootstrap centroid uncertainty

Resample whole FOVs and estimate uncertainty of the population centroid.

For centroid $\hat{\mu}$, report a scale-normalized uncertainty quantity after an explicit feature transform. A Euclidean "radius" in raw feature units is not representation-independent: standardization or another declared transform is part of the estimator provenance.

This extends the scalar FOV-unit JCB baseline from uncertainty of **one mean** to joint uncertainty of a multidimensional phenotype summary. It is useful only if that multivariate information improves held-out or biological-fidelity outcomes beyond the scalar baseline.

#### 1.2 FOV-bootstrap covariance uncertainty

Estimate uncertainty in

$$
\hat{\Sigma}.
$$

A possible loss is Frobenius distance between bootstrap covariance estimates after an explicit feature transform.

#### 1.3 FOV-bootstrap quantile uncertainty

Bootstrap uncertainty in feature quantiles can be useful for tails, but extreme quantiles are poorly supported when only a small number of FOVs or observations are visible. Quantile-based stopping should therefore require an explicit minimum information requirement and should be benchmarked for coverage/calibration before use.

Candidate quantiles might include:

$$
Q_{0.05},\;
Q_{0.25},\;
Q_{0.50},\;
Q_{0.75},\;
Q_{0.95},
$$

but $Q_{0.05}$ and $Q_{0.95}$ should not be assumed reliable at small $k$.

#### 1.4 Limits of the ordinary FOV bootstrap

The ordinary nonparametric FOV bootstrap only resamples states that have already been observed. Therefore:

- an entirely unseen rare component contributes **zero** bootstrap variance;
- with few FOVs the bootstrap distribution is discrete and ordinary variance estimates can be downward biased;
- tail quantities are especially unstable;
- spatial correlation invalidates independent-FOV resampling unless blocks or another spatial correction are used.

For population coverage, bootstrap uncertainty must therefore be evaluated together with rare-state/support methods rather than interpreted as proof that no unseen phenotype remains.

Small-$k$ behaviour should be tested against analytic, jackknife or other bias-aware uncertainty estimates where available.

#### 1.5 Power-law uncertainty model

For an approximately regular estimator whose standard error follows the usual sampling law, the first model to test is:

$$
D(n) = a n^{-1/2}.
$$

The expected marginal improvement from one additional independent FOV is:

$$
D(n)-D(n+1)
\approx
\frac{D(n)}{2n}.
$$

A more general power law $a n^{-\alpha}$ may be considered when development data justify estimating $\alpha$. Add a non-zero floor $c$ only when there is a specified source of irreducible or technical error:

$$
D(n)=a n^{-\alpha}+c.
$$

Do not use an exponential plateau model by default for sampling uncertainty. Exponential fits may still be descriptive for quantities whose theory actually supports that form; that is a separate model choice, not a generic definition of saturation.

#### 1.6 PCA eigenspectrum uncertainty

Fit PCA only on visible data and use FOV-aware resampling to measure uncertainty in the explained-variance spectrum.

#### 1.7 PCA subspace uncertainty

For orthonormal PCA bases $U$ and $V$, singular values of

$
U^\top V
$

are the cosines of the principal angles:

$
\theta_i = \arccos(\sigma_i).
$

Use FOV-aware resampling/cross-validation to ask how uncertain the inferred phenotype subspace remains.

---

### Phase 2 — simple multivariate population geometry

These methods provide inexpensive loss functions and may become estimators only if predictive validation supports that use.

#### 2.1 Gaussian / Fréchet geometry

Approximate phenotype distributions using mean and covariance.

For $(\mu_1,\Sigma_1)$ and $(\mu_2,\Sigma_2)$,

$
d_F^2 =
\|\mu_1-\mu_2\|^2
+
\operatorname{Tr}
\left[
\Sigma_1+\Sigma_2
-
2
\left(
\Sigma_1^{1/2}\Sigma_2\Sigma_1^{1/2}
\right)^{1/2}
\right].
$

This deliberately ignores higher-order multimodal structure.

That makes it a useful cheap comparator.

#### 2.2 Cross-FOV Fréchet learning curve

Raw held-out FOV loss does not converge to zero because it contains irreducible between-FOV heterogeneity. Therefore do not threshold the raw Fréchet loss as a completeness criterion.

Instead estimate a **learning curve**: fit the population representation with several numbers of visible FOVs while evaluating against the same held-out FOV block. Candidate evidence is the improvement or excess risk attributable to limited training FOVs, for example:

$$
E_m
=
R(m)-R(k-1),
\qquad m < k-1,
$$

where $R(m)$ is held-out loss when the representation is fitted from $m$ visible FOVs using a predeclared subsampling scheme.

The comparison should use the same held-out block and, where distribution losses are sensitive to empirical sample size, the same observation budget per held-out FOV or an estimator with an appropriate finite-sample correction. This prevents "more cells in this FOV" from masquerading as better population prediction.

The stopping signal is reduction in **excess risk / learning-curve slope**, not the absolute held-out FOV loss.

---

### Phase 3 — nonparametric distributional losses

Initially treat these as held-out losses and predictive-risk building blocks.

#### 3.1 Maximum Mean Discrepancy

For distributions $P$ and $Q$,

$
\operatorname{MMD}^2(P,Q)
=
E[k(X,X')]
+
E[k(Y,Y')]
-
2E[k(X,Y)].
$

Initial variants:

- RBF MMD;
- multiscale RBF MMD.

Bandwidth must be explicit and tuned only on development data.

Reference:

Gretton A, Borgwardt KM, Rasch MJ, Schölkopf B, Smola AJ. A Kernel Two-Sample Test. *Journal of Machine Learning Research*. 2012;13:723-773.

#### 3.2 Energy distance

$
\mathcal{E}(P,Q)
=
2E\|X-Y\|
-
E\|X-X'\|
-
E\|Y-Y'\|.
$

This provides a non-kernel comparator with no bandwidth parameter.

#### 3.3 Sliced Wasserstein

Project high-dimensional observations onto many one-dimensional directions, compute one-dimensional Wasserstein distances, and average across projections.

Explicit parameters include:

- number of projections;
- random seed;
- preprocessing transform.

#### 3.4 Classifier two-sample testing

Use FOV-grouped cross-validation to test whether two data sources are distinguishable.

For AUC-based evidence, a one-sided excess-over-chance quantity is preferable to absolute deviation:

$
\max(\operatorname{AUC}-0.5, 0).
$

This method requires a classifier dependency. It should not be added until dependency cost is justified.

Reference:

López-Paz D, Oquab M. Revisiting Classifier Two-Sample Tests. ICLR 2017.

#### 3.5 Role of these methods

Do not initially implement MMD, energy, sliced Wasserstein, or C2ST as "low recent-block difference = stop" rules.

Test them first as:

- held-out outcome metrics;
- drift diagnostics;
- cross-FOV predictive losses.

Promote any of them to a stopping estimator only if real-data benchmarking supports a valid prospective construction.

---

### Phase 4 — support and novelty estimators

These ask a different question:

> **Are poorly represented regions of phenotype space still being encountered?**

#### 4.1 kNN novelty

For observation $x$,

$
N_k(x) = d_k(x,S),
$

where $S$ is the visible reference phenotype set.

Useful FOV summaries include:

- median novelty;
- 95th percentile novelty;
- 99th percentile novelty;
- fraction exceeding a declared threshold.

#### 4.2 Epsilon support coverage

For a declared phenotype resolution $\epsilon$,

$
C_\epsilon = P[d(x,S)\leq \epsilon].
$

The complementary quantity

$
1-C_\epsilon
$

is novelty at that resolution.

Prospective coverage must be estimated by FOV-level resampling or cross-FOV prediction, not future-data inspection.

#### 4.3 Adaptive kNN support

Use local kNN radii rather than one global $\epsilon$ to adapt to phenotype-density variation.

#### 4.4 PRDC-style support metrics

Adapt neighbourhood-based precision/density/recall/coverage ideas to phenotype space.

These are sensitive to:

- dimensionality;
- scaling;
- neighbourhood size.

They should be tested only after simpler kNN methods.

Reference:

Naeem MF, Oh SJ, Uh Y, Choi Y, Yoo J. Reliable Fidelity and Diversity Metrics for Generative Models. *Proceedings of Machine Learning Research*. 2020;119:7176-7185.

---

### Phase 5 — ecological sample coverage with fixed phenotype categories

Ecology provides a powerful analogy:

```text
ecology                     microscopy
------------------------------------------------
individual                   phenotype observation
species                      fixed phenotype state
sampling unit/quadrat        FOV
community                    phenotype population
traits                       continuous phenotype features
```

However, classical unseen-species methods require **stable category identities**.

#### 5.1 Critical category rule

Do **not** refit k-means, GMM, HDBSCAN, or another clustering model at every acquisition prefix and then feed those changing labels into Good-Turing/Chao/ICE/Good-Toulmin.

That breaks the interpretation of:

- singletons;
- doubletons;
- incidence frequencies;
- accumulation curves.

A frozen closed partition has the opposite failure mode: truly novel phenotypes are forced into existing categories and therefore can never become a new state.

Initial ecological benchmarks should therefore prioritize:

- external biological labels whose identities are fixed independently of the evaluated acquisition; or
- an explicitly **open-set, fixed-resolution** phenotype partition defined on development data that can return "unassigned/novel" rather than forcing every observation into an existing state.

Any open-set rule must declare its distance metric and resolution before held-out evaluation.

High-dimensional fixed grids are not a generic solution: at overly fine resolution almost every observation becomes a singleton and estimated coverage collapses toward zero; at coarse resolution distinct phenotypes are merged. **Resolution sensitivity is part of the scientific result and must be benchmarked explicitly.**

Classical ecology estimators then measure coverage of the declared state system under their exchangeability assumptions; they do not provide representation-independent coverage of all possible phenotype space.

#### 5.2 Morphotype accumulation

For fixed categories,

$
S(k)=\text{number of states observed after }k\text{ FOVs}.
$

Useful diagnostics include observed richness and new-state discovery rate.

#### 5.3 Good-Turing abundance coverage

Use object abundance frequencies to estimate unseen probability mass.

This is sensitive to within-FOV duplication and therefore may overstate evidence when one local state produces many cells.

#### 5.4 Good-Turing incidence coverage

Use the number of FOVs containing each fixed phenotype state.

This maps naturally to FOV sampling because repeated cells within one FOV do not count as repeated sampling-unit incidence.

A state found in one FOV is qualitatively different from a state recurring across many FOVs.

#### 5.5 Chao1

Estimate unseen richness from abundance frequencies, especially singletons and doubletons.

A familiar simplified form is:

$
\hat{S}
=
S_{\mathrm{obs}}
+
\frac{f_1^2}{2f_2},
$

with bias-corrected variants required for edge cases.

#### 5.6 Chao2

Use rare **sampling-unit incidences** rather than object abundances.

For MorphoCoverage, the sampling unit is the FOV.

#### 5.7 ICE

Incidence-based coverage estimation uses the rare incidence-frequency spectrum more broadly than Chao2.

#### 5.8 Good-Toulmin / smoothed Good-Toulmin

Predict how many new categories are expected after additional sampling from the observed frequency spectrum.

This is conceptually close to the question:

> **How many new fixed phenotype states should another block of FOVs reveal?**

Smoothed variants are required for stable extrapolation beyond small sampling increments.

#### 5.9 Hill diversity

For category probabilities $p_i$,

$
{}^qD
=
\left(\sum_i p_i^q\right)^{1/(1-q)}.
$

Important cases:

- $q=0$: richness;
- $q\to1$: exponential Shannon diversity;
- $q=2$: inverse Simpson diversity.

A profile across $q$ shows whether adequacy is driven by rare or common states.

#### 5.10 Functional Hill diversity

Functional diversity incorporates distances among states in a continuous trait/phenotype space.

This may bridge categorical ecology and continuous morphology.

#### 5.11 Independence caveat

Incidence estimators treat FOVs as sampling units, but spatially correlated FOVs are not equivalent to independent quadrats.

Therefore Chao2/ICE/incidence coverage must be benchmarked under:

- random/permuted FOV order;
- recorded spatial order;
- known neighbouring-field correlation;
- spatial block resampling where coordinates permit it.

The FOV unit is operational, not automatically independent.

References:

Chao A. Estimating the population size for capture-recapture data with unequal catchability. *Biometrics*. 1987;43(4):783-791.

Lee SM, Chao A. Estimating population size via sample coverage for closed capture-recapture models. *Biometrics*. 1994;50(1):88-97. https://doi.org/10.2307/2533199

Chao A, Jost L. Coverage-based rarefaction and extrapolation: standardizing samples by completeness rather than size. *Ecology*. 2012;93:2533-2547. https://doi.org/10.1890/11-1952.1

Chao A, Chiu C-H, Jost L. Unifying Species Diversity, Phylogenetic Diversity, Functional Diversity, and Related Similarity and Differentiation Measures Through Hill Numbers. *Annual Review of Ecology, Evolution, and Systematics*. 2014;45:297-324. https://doi.org/10.1146/annurev-ecolsys-120213-091540

Good IJ, Toulmin GH. The number of new species, and the increase in population coverage, when a sample is increased. *Biometrika*. 1956;43:45-63.

Orlitsky A, Suresh AT, Wu Y. Optimal prediction of the number of unseen species. *Proceedings of the National Academy of Sciences*. 2016;113:13283-13288.

---

### Phase 6 — explicitly FOV-predictive estimators

This family is especially important because it models uncertainty at the actual acquisition unit.

The target should **not** be raw expected distance between consecutive FOVs.

Instead ask:

> **How uncertain is the inferred population description because only these FOVs have been observed?**

#### 6.1 Cross-FOV learning-curve excess risk

Leave-one-FOV-out loss by itself is not a stopping quantity. Even with a perfectly estimated population, a future FOV remains different from the population model because of irreducible FOV-to-FOV heterogeneity.

Use cross-FOV prediction to estimate a **learning curve** instead.

For several training sizes $m < k$, repeatedly:

1. select $m$ of the currently visible FOVs using a predeclared scheme;
2. fit the population representation only on those FOVs;
3. evaluate on the same held-out visible FOV block;
4. repeat across admissible FOV splits.

Candidate evidence is:

- excess risk between smaller and larger training sets;
- the slope of risk versus number of training FOVs;
- uncertainty in that slope.

Do not threshold absolute predictive risk.

When the loss compares empirical phenotype distributions, control for the held-out FOV's number of observations by using a fixed observation budget, repeated subsampling, or a finite-sample-corrected estimator. The `unequal_objects_per_fov` scenario must explicitly test this failure mode.

#### 6.2 Population-estimate uncertainty

The FOV bootstrap introduced in Phase 1 is one way to estimate uncertainty in a population representation; it should not be implemented a second time here.

Phase 6 instead asks whether richer whole-FOV representations improve **excess predictive risk or its learning curve** beyond the simple centroid/covariance/quantile bootstrap baselines.

The ordinary bootstrap limitations still apply: unseen states are absent, small-$k$ uncertainty can be underestimated, and spatial dependence requires block-aware resampling.

#### 6.3 FOV distribution representations

Candidate whole-FOV representations include:

- centroid;
- mean + covariance;
- quantile profile;
- fixed-category composition;
- kernel mean embedding.

#### 6.4 Kernel mean embedding per FOV

For FOV distribution $P_j$,

$
\mu_{P_j}
=
E_{x\sim P_j}[\phi(x)].
$

Empirically,

$
\hat{\mu}_{P_j}
=
\frac{1}{n_j}
\sum_i \phi(x_{ij}).
$

This turns each FOV into a distribution-valued observation.

#### 6.5 FOV-type accumulation

If whole-FOV states can be defined using a stable, externally fixed representation, ask whether new FOV community types continue to appear.

As with cell-level ecology, the state definition must not be refitted in a way that destroys category identity across prefixes.

---

### Phase 7 — explicit rare-phenotype objectives

Population representation and rare-state discovery are not interchangeable.

#### 7.1 Exact FOV-incidence planning baseline

If a target phenotype has known per-FOV incidence $p$, and we require at least $r$ positive FOVs after $n$ independent representative FOVs,

$$
P(X\ge r)
=
1-
\sum_{i=0}^{r-1}
{n\choose i}
p^i(1-p)^{n-i}.
$$

For first detection ($r=1$) and desired detection probability $\beta$, the required number of independent FOVs is fixed in advance:

$$
n
=
\left\lceil
\frac{\log(1-\beta)}{\log(1-p)}
\right\rceil.
$$

This is therefore a **planning calculation / fixed-FOV comparator**, not an adaptive coverage estimator, when $p$ is supplied.

The independence assumption is substantive. If rare states cluster across neighbouring FOVs, the binomial calculation is anti-conservative and overstates detection probability. Spatial clustering should be tested explicitly, and a beta-binomial, occupancy, or spatial model is required when exchangeability/independence is not defensible.

#### 7.2 Object prevalence is not FOV incidence

`RareTarget` may define object prevalence without FOV incidence.

Do not silently convert object prevalence to per-FOV incidence.

Such a conversion requires assumptions about:

- observations per FOV;
- independence of objects;
- within-FOV clustering;
- potentially zero-object FOVs.

If prevalence-to-incidence conversion is implemented later, its assumptions must be explicit and benchmarked.

#### 7.3 Beta-binomial incidence uncertainty

When per-FOV incidence is estimated rather than known, use a beta-binomial predictive formulation or another explicit uncertainty model.

#### 7.4 Rare-state recurrence

Track recurrence across FOVs rather than only object count.

Many rare cells in one FOV do not provide the same evidence as recurrence across many independently sampled positions.

#### 7.5 Occupancy/detection models

Later, if the data contain identifiable repeated detection opportunities, consider occupancy models that separate true presence from imperfect detection.

Do not add occupancy modelling when detection probability is unidentifiable from the available data.

Reference:

MacKenzie DI, Nichols JD, Lachman GB, Droege S, Royle JA, Langtimm CA. Estimating site occupancy rates when detection probabilities are less than one. *Ecology*. 2002;83:2248-2255.

---

### Phase 8 — spatial diagnostics and correction

FOV-aware is not synonymous with independent.

#### 8.1 Existing design effect

The current FOV-unit scalar-precision implementation already reports a design effect relative to random-cell sampling.

This should remain part of the regime diagnostics.

#### 8.2 Semivariogram

For an FOV-level representation $Z(s)$,

$
\gamma(h)
=
\frac{1}{2}
E[(Z(s)-Z(s+h))^2].
$

Estimate spatial correlation length where stage coordinates are available.

#### 8.3 Spatial ICC / effective sample size

Extend existing design-effect thinking beyond one scalar feature where scientifically justified.

#### 8.4 Spatial block bootstrap

When neighbouring FOVs are correlated, resampling individual FOVs can underestimate uncertainty.

Use spatial blocks when coordinates and sample geometry support a defensible block definition.

---

### Phase 9 — multi-group planning comparators

These methods are **not core one-group coverage estimators**, but they are important comparators because biological experiments often aim to detect condition differences.

#### 9.1 Existing JCB multi-sample planning

Already implemented through:

- `plan_cells_to_average`;
- `CellCountPlan`;
- `PlannedCellCountStrategy`.

This is the model for separating multi-group planning from one-group prospective estimators.

#### 9.2 pMoSS

pMoSS repeatedly subsamples comparison groups at different sample sizes, evaluates a hypothesis-test p-value, and models its dependence on $n$, including an exponential model of the form

$
p(n)=a e^{-cn}.
$

Its question is approximately:

> **How much data are required for a specified between-group statistical comparison to become robust under the pMoSS procedure?**

That is not phenotype coverage, but it is a legitimate comparator when the experimental objective is a condition difference.

If implemented, pMoSS should be an explicit multi-group planning workflow rather than a normal group-local `Estimator`.

Reference:

Gómez-de-Mariscal E, et al. Use of the p-values as a size-dependent function to address practical differences when analyzing large datasets. *Scientific Reports*. 2021;11:20942. https://doi.org/10.1038/s41598-021-00199-5

---

### Phase 10 — downstream biological-fidelity outcomes

These are benchmark outcomes, not necessarily sampling estimators.

They matter because an estimator that reconstructs the marginal phenotype distribution well may still stop before the biology needed to distinguish conditions is represented.

#### 10.1 Effect-size fidelity at the biological-repeat level

The unit of biological inference must remain the biological sample/repeat, not the FOV or well merely because more FOVs were acquired.

For each biological repeat, prospective stopping determines the within-sample phenotype summary. Condition effects are then computed across the appropriate biological repeats exactly as the downstream analysis would do; oversampling one well must never create additional inferential replicates.

If the held-out/deep benchmark defines a reference effect $\Delta_{\mathrm{ref}}$ and stopped acquisition gives $\Delta_k$, report at least:

$$
\operatorname{bias}_k
=
\Delta_k-\Delta_{\mathrm{ref}},
$$

and

$$
\operatorname{absolute\ error}_k
=
|\Delta_k-\Delta_{\mathrm{ref}}|.
$$

Signed bias must be summarized across acquisition orders/replays because a data-dependent stopping rule can systematically bias an effect even when mean absolute error looks acceptable.

Also report uncertainty and effect direction where appropriate, and compare within-repeat sampling error with between-repeat variability rather than rewarding ever-deeper imaging of a single repeat.

#### 10.2 Phenotype-frequency differences

For predefined states, compare between-condition differences in state frequency at the stop against the reference benchmark.

This is particularly important when biologically informative states are rare.

#### 10.3 Feature-ranking stability

Where the scientific workflow ranks features or perturbations, measure rank fidelity against the reference.

#### 10.4 Classification/retrieval fidelity

For profiling workflows, measure whether stopping changes a predefined downstream classification or retrieval task.

These outcomes may require optional dependencies and should only be implemented when they correspond to real use cases.

#### 10.5 Statistical significance should not be the sole outcome

A p-value threshold is highly sample-size dependent and should not be the only definition of preserved biological insight.

Prefer effect magnitude, ranking, prediction and uncertainty alongside any test-based result.

---

### Phase 11 — continual-monitoring decision methods

These methods belong to the decision layer once useful estimator quantities have been identified.

#### 11.1 Segmented/plateau regression

Fit explicit saturation models to an evidence trajectory when appropriate.

#### 11.2 Confidence sequences

Repeated checking of fixed-sample confidence intervals can invalidate nominal coverage.

Confidence sequences may support continual monitoring once the scientific estimand is settled.

#### 11.3 Anytime-valid / e-value policies

Investigate optional-stopping-safe evidence only after the target estimator is validated.

#### 11.4 Drift diagnostics

CUSUM, EWMA, ADWIN or Bayesian change-point methods may detect:

- technical drift;
- sample gradients;
- changes in acquisition regime.

These are QC/stationarity tools, not direct population-coverage estimators.

---

## 9. Dependency policy

The package currently depends primarily on NumPy, pandas and SciPy.

Do not add a heavyweight dependency merely because a candidate method is convenient to obtain from a library.

For each proposed estimator, document whether it can be implemented with current dependencies.

Examples:

- centroid/covariance/quantiles: current stack;
- Fréchet: current stack;
- MMD: current stack for exact calculation, but exact kernel MMD is typically $O(n^2)$ in time/memory; live use may require subsampling, linear-time estimators or random-feature approximations;
- energy distance: current stack, but the direct pairwise form is typically $O(n^2)$; live use may require subsampling or linear/near-linear approximations;
- sliced Wasserstein: current stack;
- kNN support: current stack with SciPy spatial tools;
- classical Good-Turing/Chao/Hill: current stack;
- logistic-regression C2ST: likely scikit-learn unless implemented locally;
- GMM/k-means/HDBSCAN: typically scikit-learn/HDBSCAN;
- graph community methods: additional graph dependency.

Dependency-expanding candidates should remain deferred until benchmarking shows that the corresponding method family is scientifically worth the cost.

---

## 10. Required contract for every future estimator

Every estimator PR must state:

1. **Scientific question**  
   What exact meaning of "enough imaging" is addressed?

2. **Estimand**  
   What population quantity is being estimated?

3. **Null/limiting behaviour**  
   What should happen under a homogeneous stationary population? What does a low value actually mean?

4. **Sampling unit**  
   Observation, FOV, incidence, imaged area, or spatial block?

5. **Objective component**  
   `population` or `rare`.

6. **Observation-unit semantics**  
   Which `observation_unit` values are valid?

7. **Representation assumptions**  
   Continuous features, fixed categories, scalar feature, labelled rare state, etc.

8. **Missing-data behaviour**  
   How are NaNs handled? Never silently impute or drop.

9. **Evidence quantity and direction**  
   Exact numerical value, units and whether higher/lower is better.

10. **Parameters and provenance**  
    No hidden bandwidths, distances, neighbourhood sizes, state resolutions or thresholds.

11. **Randomness**  
    Explicit `random_state` or equivalent.

12. **Prospective leakage protection**  
    How is future-FOV information excluded?

13. **Group independence**  
    Core estimators must remain group-local unless explicitly implemented as a planning comparator outside the estimator interface.

14. **Computational complexity**  
    Relevant for later smart microscopy.

15. **Incremental feasibility**  
    Full recomputation remains the reference implementation.

16. **Known failure modes**  
    Spatial correlation, rare-state blindness, curse of dimensionality, density variation, category instability, etc.

17. **Independent falsification outcome**  
    What held-out real-data result would show that the method is misleading?

18. **Contracts**  
    Add strategies to `STRATEGIES_UNDER_CONTRACT`; add outcome metrics to `METRICS_UNDER_CONTRACT`; add current-practice baselines to `baseline_comparison` where applicable.

---

## 11. Revised implementation order

The order below is deliberately front-loaded with benchmark work.

| Order | Work | Purpose |
| --- | --- | --- |
| 1 | Deep real benchmark datasets | Prevent method selection on inadequate data |
| 2 | Fixed evaluation-reservoir benchmark support | Remove self-containing reference bias |
| 3 | Multivariate held-out outcome metrics | Give every estimator independent falsification targets |
| 4 | Multi-group biological-fidelity benchmark design | Ensure preserved biology can influence method choice |
| 5 | Fixed-area baseline | Complete simple effort baselines |
| 6 | FOV-bootstrap centroid/covariance/quantile uncertainty | First new population estimators |
| 7 | PCA spectrum/subspace uncertainty | Test multivariate geometry stability |
| 8 | Fréchet learning-curve excess risk | Cheap distributional predictive baseline |
| 9 | MMD/energy/sliced-Wasserstein held-out losses | Flexible independent distributional outcomes |
| — | **Interim benchmark** | Decide which continuous methods merit promotion to estimators |
| 10 | kNN novelty | Local support novelty |
| 11 | epsilon/adaptive support | Explicit phenotype support coverage |
| 12 | PRDC-style support metrics | Only if simpler support methods justify it |
| — | **Interim benchmark** | Stop if support methods add no value |
| 13 | Stable fixed-category transform contract | Required before ecology methods |
| 14 | Morphotype accumulation | Interpretable discovery baseline |
| 15 | Good-Turing abundance + incidence | Missing-mass coverage |
| 16 | Chao1/Chao2/ICE | Unseen richness/completeness |
| 17 | Good-Toulmin / smoothed Good-Toulmin | Future-state discovery |
| 18 | Hill / functional Hill diversity | Rare-to-common diversity spectrum |
| — | **Ecology benchmark** | Compare categorical coverage with continuous methods |
| 19 | Cross-FOV learning-curve excess risk | Whole-FOV predictive convergence beyond irreducible heterogeneity |
| 20 | Kernel/FOV distribution representations | Only if simple FOV summaries are insufficient |
| — | **FOV-predictive benchmark** | Test whether FOV modelling materially improves stopping |
| 21 | Exact FOV-incidence planning baseline | Transparent rare-state planning comparator |
| 22 | Beta-binomial incidence + recurrence | Rare-state uncertainty |
| 23 | Variogram/spatial diagnostics | Quantify spatial redundancy |
| 24 | Spatial block bootstrap | Correct uncertainty when coordinates support it |
| 25 | pMoSS planning comparator | Only if condition-comparison benchmarking justifies it |
| 26 | Downstream biological-fidelity outcomes | Expand only for demonstrated real workflows |
| 27+ | Sequential confidence methods / Bayesian nonparametrics / GP / topology | Research reserve |

The numbering is a scientific priority order, not a commitment to complete every item.

At every interim benchmark, a null result is acceptable. If simple baselines perform as well as richer methods, later phases may be unnecessary.

---

## 12. Research reserve

The following remain worthwhile ideas but should not delay the main programme:

- Dirichlet-process discovery probability;
- Pitman-Yor discovery probability;
- posterior missing mass;
- hierarchical Bayesian species sampling;
- Sinkhorn divergence;
- Gaussian-process spatial coverage;
- conformal FOV novelty;
- persistent-homology stabilization;
- deep density models.

Each should be introduced only to address a failure demonstrated by simpler methods.

---

## 13. Expected scientific product

The final result should not be prewritten as "method X wins."

A plausible **type** of result is:

```text
some datasets
    -> cell-number or fixed-FOV guidance is sufficient

some datasets
    -> simple FOV-bootstrap uncertainty is sufficient

some datasets
    -> support or ecological incidence methods protect rare states better

some datasets
    -> explicitly FOV-predictive methods improve calibration

some datasets
    -> spatial correlation invalidates naive FOV resampling

some downstream workflows
    -> the method with best global distribution fidelity is not the method
       that best preserves the biological effect
```

The benchmark must be capable of producing the opposite result for any proposed family.

---

## 14. Decision rule for promoting a default

A default MorphoCoverage estimator should be selected only if it shows, on held-out real datasets:

1. useful prediction of unseen information;
2. acceptable premature-stop risk;
3. robust behaviour across acquisition order;
4. clear advantages over fixed observations/FOV/area and JCB baselines in identifiable regimes;
5. no unacceptable loss of rare-state or downstream biological fidelity;
6. manageable sensitivity to representation and hyperparameters;
7. computational feasibility for the intended deployment mode.

If no single method satisfies these requirements across regimes, MorphoCoverage should expose a small set of objective-specific recommended methods rather than force one universal default.
