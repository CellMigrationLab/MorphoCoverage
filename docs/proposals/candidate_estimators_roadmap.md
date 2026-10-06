# Proposal: MorphoCoverage candidate estimators not yet built

## Status

This document is a **proposal**. It lists the candidate methods that are not yet built, the benchmark requirements every candidate must meet, and an order for discussion. It does not select a default estimator and does not imply that every listed method should be implemented.

**Implemented (checked against `src/`, 2026-09-26), and therefore no longer described here:** fixed evaluation reservoirs split by spatial block and the reference error floor (`morphocoverage.reservoir`: `ReservoirSplit`, `spatial_blocks`, `reference_error_floor`); dataset roles and tuning checks (`benchmark.BenchmarkRegistry`, `check_tuning`, `leave_one_dataset_out`) and the locked benchmark protocol (`data/crick_idr/benchmark_protocol.py`); matched-effort scoring (`retrospective.frontier`); multivariate held-out outcomes, including the Fréchet, energy, MMD and sliced-Wasserstein losses (`multivariate_outcomes`, `distances`); the fixed-area baseline (`FixedAreaStrategy`); the FOV bootstrap of the centroid, covariance, quantiles, PCA eigenspectrum and subspace, with a spatial block option (`FOVBootstrapStrategy`); epsilon support novelty (`SupportNoveltyStrategy`); Chao–Jost incidence coverage, Chao2, the Good–Turing incidence estimate and the one-FOV Good–Toulmin-type extrapolation on fixed labels (`IncidenceCoverageStrategy`); FOV-incidence planning (`FOVIncidencePlanStrategy`); beta-binomial incidence with recurrence (`RareIncidenceStrategy`); the semivariogram (`morphocoverage.spatial`); and JCB multi-sample planning (`plan_objects_to_average`, `PlannedObjectCountStrategy`). Their design, parameters and limits are in `docs/common/architecture.md` and `docs/optimise/benchmarking.md`. idr0139 is registered as a held-out dataset, but it has 4 FOVs per well; no deep dataset is registered.

The central rule is:

> **MorphoCoverage should choose sampling methods by how well they preserve scientifically relevant information in unseen data, not by how smoothly their own internal statistic appears to converge.**

A useful final result may be a regime map rather than one universal winner.

---

## 1. Three architectural roles must remain separate

- **Core prospective estimator.** It operates on one analysis group at a time, sees only the currently visible FOVs, and asks how much uncertainty or unrepresented phenotype remains in this sample under the stated objective. It must not inspect future FOVs or comparison groups.
- **Multi-group planning comparator.** Published sample-size methods that need several samples or conditions are not forced into the one-group estimator interface. They use declared pilot data and return a fixed plan for a group-local stopping strategy, as the JCB multi-sample planning already does. pMoSS (section 5.10) would follow this pattern.
- **Benchmark outcome.** Retrospective, and allowed to use information unavailable to the prospective estimator: if acquisition had stopped here, how much scientifically relevant information would have been lost (population-distribution error, rare-state loss, FOV-level representation error, effect-size error between conditions, treatment-ranking preservation, classification or retrieval stability)?

MorphoCoverage does **not** need to become a general biological-statistics package, but it must be allowed to evaluate whether a sampling strategy preserves the biological conclusion the experiment was designed to support.

## 2. What "best" should mean

The best sampling method is not necessarily the method with the smallest value of its own internal statistic. Candidates are judged on independent dimensions: population fidelity, rare-state protection, downstream biological fidelity, acquisition effort and computational cost. Different experiments weight these differently (a treatment screen, an atlas, a rare-event experiment, live smart microscopy under latency constraints), so MorphoCoverage reports Pareto trade-offs and regime maps rather than one arbitrary score.

---

## 3. Benchmark requirements that still apply

The first priority is a benchmark capable of rejecting bad estimators, not another estimator.

### 3.1 Deeply sampled real data

The committed data are useful for integration tests and preliminary analysis, but not sufficient as the primary scientific benchmark: idr0139 has only 4 FOVs per well, and the selected idr0028 screen-B wells have 28. The programme should register several deeper real datasets spanning:

- relatively homogeneous cultured cells;
- heterogeneous cell populations;
- strongly spatially structured samples;
- rare states;
- variable density;
- multiple phenotype representations;
- several biological or technical replicates;
- recorded stage coordinates where possible.

Dataset roles must be fixed as development or held-out before method tuning. A second independently prepared acquisition or biological repeat should not be treated automatically as a same-population reservoir: batch and biological shifts introduce error that no within-sample stopping rule can remove, so such data are better used as a separate cross-acquisition robustness outcome.

### 3.2 Immediate next-FOV utility

A short-horizon outcome should quantify the realized utility of the next unseen FOV $F_{k+1}$. For a predeclared held-out loss $L$ against the fixed evaluation reservoir $R$:

$$
\Delta L_{k+1}
=
L(P_{1:k},R)
-
L(P_{1:k+1},R).
$$

Positive values mean the next FOV improved representation of the reservoir; negative values are possible. Other predeclared next-FOV utilities include discovery or recurrence of an independently labelled rare state, reduction in error of a scalar or multivariate population functional, and reduction in downstream biological-fidelity error. This is a **retrospective outcome**, not a prospective estimator; it may mechanically become small late in acquisition, so it is read together with matched-effort curves and the fixed-reservoir outcomes. (`sampling_reliability` reports the gain of the median outcome between random `k`-FOV sets, not this sequential quantity.)

### 3.3 Compare methods at matched effort

Do not judge a method by correlation between its internal evidence and an outcome that also changes automatically with $k$. Compare outcome error against FOV count, imaged area and observation count; threshold sweeps producing stop-effort/error curves; premature-stop risk at matched acquisition budgets; and acquisition savings at matched scientific error.

### 3.4 "Adequate" must be explicit

"Premature stopping" and "excess FOVs" require a declared adequacy criterion: an absolute tolerance on held-out distribution error set above the finite-reservoir error floor; a relative tolerance against a deep reference; a tolerance on rare-state detection probability or on downstream effect-size error; or a Pareto-front analysis without one universal threshold. The benchmark must not choose tolerances after seeing which method wins.

### 3.5 Independence of incidence and resampling estimators

Incidence estimators and FOV resampling treat FOVs as sampling units, but spatially correlated FOVs are not independent quadrats. The built ones (incidence coverage, Chao2, the FOV bootstrap) and any new ones must be benchmarked under random and recorded orders, known neighbouring-field correlation, and spatial block resampling where coordinates permit it. The FOV unit is operational, not automatically independent.

### 3.6 Checks the built FOV bootstrap still awaits

Quantile-based stopping needs an explicit minimum information requirement and a benchmark of its coverage and calibration before use, since extreme quantiles ($Q_{0.05}$, $Q_{0.95}$) are poorly supported by few FOVs or observations. Its small-$k$ behaviour should be tested against analytic, jackknife or other bias-aware uncertainty estimates where they exist (a test already compares the centroid variant with the analytic FOV-unit standard error). An unseen rare component contributes zero bootstrap variance, so bootstrap uncertainty is evaluated together with rare-state and support methods.

---

## 4. Two negative results that constrain the candidates

### 4.1 Stationarity is not completeness

Suppose FOVs are independent draws from one stationary phenotype population $P$. Then an early block and a recent block both satisfy approximately $X_{\mathrm{early}} \sim P$ and $X_{\mathrm{recent}} \sim P$, and a two-sample method (MMD, energy distance, sliced Wasserstein, classifier two-sample testing) approaches its null behaviour even if a rare component of $P$ has not yet been observed.

> **Similarity between recent and earlier FOV blocks measures stationarity/consistency of observed sampling, not completeness of the latent phenotype population.**

These methods remain useful as drift diagnostics, independent benchmark outcomes, predictive-error components and distributional losses in cross-FOV prediction, but not automatically as standalone stopping rules. Observed data cannot identify the phenotype or location of a **specific** state that has never appeared; frequency-based methods such as Good–Turing or Chao can nevertheless infer aggregate unseen mass or richness from observed rare-frequency patterns under their sampling assumptions.

### 4.2 Cumulative next-step change is mechanically diluted

Do not define coverage from $d(P_{1:k}, P_{1:k+1})$ without correcting for the fact that one FOV contributes a fraction of order $1/k$ to the cumulative sample: that quantity shrinks even when FOV heterogeneity remains substantial. The expected distance between two independent FOV distributions is a property of the sample's heterogeneity and need not converge to zero. The target for FOV-predictive methods is therefore **uncertainty in the inferred population representation** (under whole-FOV resampling, cross-FOV predictive risk, posterior or predictive uncertainty where a model is justified, expected error against a held-out reservoir), not raw FOV-to-FOV distance.

---

## 5. Candidates not yet built

### 5.1 Power-law uncertainty model

For an approximately regular estimator whose standard error follows the usual sampling law, the first model to test is $D(n) = a n^{-1/2}$, with expected marginal improvement from one more independent FOV

$$
D(n)-D(n+1)
\approx
\frac{D(n)}{2n}.
$$

A general power law $a n^{-\alpha}$ may be considered when development data justify estimating $\alpha$; add a non-zero floor $c$ only for a specified source of irreducible or technical error: $D(n)=a n^{-\alpha}+c$. Do not use an exponential plateau model by default for sampling uncertainty.

### 5.2 Cross-FOV learning curves (Fréchet and other losses)

Raw held-out FOV loss does not converge to zero, because it contains irreducible between-FOV heterogeneity, so never threshold it as a completeness criterion. Leave-one-FOV-out loss by itself is not a stopping quantity either. Estimate a **learning curve** instead: for several training sizes $m < k$, repeatedly select $m$ of the visible FOVs by a predeclared scheme, fit the population representation on them only, evaluate on the same held-out visible FOV block, and repeat across admissible splits. Candidate evidence is the excess risk between smaller and larger training sets, for example

$$
E_m
=
R(m)-R(k-1),
\qquad m < k-1,
$$

the slope of risk against the number of training FOVs, and the uncertainty of that slope. When the loss compares empirical distributions, control for the held-out FOV's number of observations (a fixed observation budget, repeated subsampling, or a finite-sample-corrected estimator); the `unequal_objects_per_fov` scenario must test this failure mode. The FOV bootstrap is the baseline this must beat: the question is whether richer whole-FOV representations improve excess predictive risk beyond it.

### 5.3 Classifier two-sample testing

Use FOV-grouped cross-validation to test whether two data sources are distinguishable, with the one-sided excess-over-chance quantity $\max(\operatorname{AUC}-0.5, 0)$. It requires a classifier dependency and should not be added until that cost is justified (López-Paz D, Oquab M. Revisiting Classifier Two-Sample Tests. ICLR 2017).

Do not implement MMD, energy, sliced Wasserstein or C2ST as "low recent-block difference = stop" rules (section 4.1). The first three exist as held-out outcomes; promote any of them to a stopping estimator only if real-data benchmarking supports a valid prospective construction.

### 5.4 kNN, adaptive and PRDC-style support

- **kNN novelty.** For observation $x$ and visible reference set $S$, $N_k(x) = d_k(x,S)$, summarized per FOV by its median, 95th and 99th percentiles and the fraction above a declared threshold.
- **Adaptive kNN support.** Local kNN radii rather than one global $\epsilon$, to adapt to phenotype-density variation.
- **PRDC-style metrics.** Neighbourhood-based precision, density, recall and coverage adapted to phenotype space; sensitive to dimensionality, scaling and neighbourhood size, so tested only after simpler kNN methods (Naeem MF, Oh SJ, Uh Y, Choi Y, Yoo J. Reliable Fidelity and Diversity Metrics for Generative Models. *PMLR*. 2020;119:7176-7185).

Prospective coverage must be estimated by FOV-level resampling or cross-FOV prediction, not by inspecting future data.

### 5.5 Ecological coverage beyond incidence coverage and Chao2

Ecology offers an analogy (individual ↔ phenotype observation, species ↔ fixed phenotype state, quadrat ↔ FOV, community ↔ phenotype population, traits ↔ continuous features), but classical unseen-species methods need **stable category identities**.

**Critical category rule.** Do not refit k-means, GMM, HDBSCAN or another clustering model at every acquisition prefix and feed the changing labels into Good–Turing, Chao, ICE or Good–Toulmin: that breaks the meaning of singletons, doubletons, incidence frequencies and accumulation curves. A frozen closed partition has the opposite failure: truly novel phenotypes are forced into existing categories. Ecological benchmarks therefore use external biological labels fixed independently of the evaluated acquisition, or an explicitly **open-set, fixed-resolution** partition defined on development data that can return "unassigned/novel"; its distance metric and resolution are declared before held-out evaluation. At too fine a resolution almost every observation is a singleton and coverage collapses toward zero; at too coarse a resolution distinct phenotypes merge. **Resolution sensitivity is part of the scientific result and must be benchmarked explicitly.** The estimators then measure coverage of the declared state system under their exchangeability assumptions, not of all possible phenotype space.

Not yet built:

- **Morphotype accumulation**: $S(k)$, the number of states observed after $k$ FOVs, with the new-state discovery rate.
- **Good–Turing abundance coverage**: unseen probability mass from object abundances; sensitive to within-FOV duplication, so it may overstate evidence when one local state produces many cells.
- **Chao1**: unseen richness from abundance singletons and doubletons, $\hat{S} = S_{\mathrm{obs}} + f_1^2 / (2 f_2)$, with bias-corrected variants for edge cases.
- **ICE**: incidence-based coverage using the rare incidence-frequency spectrum more broadly than Chao2.
- **Smoothed Good–Toulmin**: how many new states another block of FOVs should reveal, stable beyond the one-FOV extrapolation already reported.
- **Hill diversity**: ${}^qD = (\sum_i p_i^q)^{1/(1-q)}$ (richness at $q=0$, exponential Shannon as $q\to1$, inverse Simpson at $q=2$); a profile across $q$ shows whether adequacy is driven by rare or common states. **Functional Hill diversity** adds distances among states in a continuous trait space.

References: Chao A. *Biometrics*. 1987;43(4):783-791. Lee SM, Chao A. *Biometrics*. 1994;50(1):88-97. https://doi.org/10.2307/2533199. Chao A, Jost L. *Ecology*. 2012;93:2533-2547. https://doi.org/10.1890/11-1952.1. Chao A, Chiu C-H, Jost L. *Annu Rev Ecol Evol Syst*. 2014;45:297-324. https://doi.org/10.1146/annurev-ecolsys-120213-091540. Good IJ, Toulmin GH. *Biometrika*. 1956;43:45-63. Orlitsky A, Suresh AT, Wu Y. *PNAS*. 2016;113:13283-13288.

### 5.6 FOV distribution representations

Candidate whole-FOV representations: centroid; mean and covariance; quantile profile; fixed-category composition; the kernel mean embedding per FOV, $\hat{\mu}_{P_j} = \frac{1}{n_j}\sum_i \phi(x_{ij})$, which turns each FOV into a distribution-valued observation. **FOV-type accumulation** asks whether new FOV community types keep appearing, when whole-FOV states can be defined with a stable, externally fixed representation that is not refitted across prefixes. Only if simple FOV summaries prove insufficient.

### 5.7 Object prevalence is not FOV incidence

`RareTarget` may define object prevalence without FOV incidence, and the package never converts one into the other. A conversion would need assumptions about observations per FOV, independence of objects, within-FOV clustering and zero-object FOVs; if one is ever implemented, those assumptions must be explicit and benchmarked.

### 5.8 Occupancy and detection models

If the data contain identifiable repeated detection opportunities, occupancy models can separate true presence from imperfect detection. Do not add them when detection probability is unidentifiable from the available data (MacKenzie DI et al. *Ecology*. 2002;83:2248-2255).

### 5.9 Spatial ICC and effective sample size

Extend the design-effect reasoning (the FOV-unit JCB variant's `ratio_design_effect`, `statistics.design_effect`) beyond one scalar feature where scientifically justified, and keep it part of the regime diagnostics.

### 5.10 pMoSS as a multi-group planning comparator

pMoSS repeatedly subsamples comparison groups at different sizes, evaluates a hypothesis-test p-value and models its dependence on $n$, including $p(n)=a e^{-cn}$. It asks how much data a specified between-group comparison needs to become robust under the procedure: not phenotype coverage, but a legitimate comparator when the objective is a condition difference. If implemented, it is an explicit multi-group planning workflow, not a group-local `Estimator` (Gómez-de-Mariscal E, et al. *Scientific Reports*. 2021;11:20942. https://doi.org/10.1038/s41598-021-00199-5).

### 5.11 Downstream biological-fidelity outcomes

These are benchmark outcomes, evaluated on several stopped groups jointly, and must never leak into a prospective estimator. They matter because an estimator that reconstructs the marginal distribution well may still stop before the biology that distinguishes conditions is represented. Today the report describes condition differences (`sampling_unit.condition_contrast`, `feature_ranking`) and `data/crick_idr/jcs_effect_fovs.py` checks whether a treatment ranking survives whole-field sampling, but no packaged outcome scores a stop this way; that needs the multi-group outcome contract (AGENTS.md section 9).

- **Effect-size fidelity at the biological-repeat level.** The unit of inference stays the biological sample or repeat; condition effects are computed across repeats exactly as the downstream analysis would, and oversampling one well never creates inferential replicates. Report $\operatorname{bias}_k = \Delta_k-\Delta_{\mathrm{ref}}$ and $|\Delta_k-\Delta_{\mathrm{ref}}|$ against the reference effect, the signed bias summarized across orders and replays (a data-dependent stop can bias an effect while its mean absolute error looks acceptable), effect direction and uncertainty, and within-repeat sampling error against between-repeat variability.
- **Phenotype-frequency differences** between conditions for predefined states, particularly when the informative states are rare.
- **Feature- or perturbation-ranking stability** against the reference.
- **Classification or retrieval fidelity** for profiling workflows, only for real use cases (it may need optional dependencies).
- A p-value threshold is highly sample-size dependent and should not be the only definition of preserved biology: prefer effect magnitude, ranking, prediction and uncertainty alongside any test.

### 5.12 Continual-monitoring decision methods

These belong to the decision layer once useful estimator quantities have been identified:

- segmented or plateau regression fitted to an evidence trajectory when appropriate;
- confidence sequences, since repeated checking of fixed-sample intervals invalidates nominal coverage, once the estimand is settled;
- anytime-valid or e-value policies, only after the target estimator is validated;
- drift diagnostics (CUSUM, EWMA, ADWIN, Bayesian change points) for technical drift, sample gradients and changes of acquisition regime: QC and stationarity tools, not coverage estimators.

---

## 6. Dependency policy

The package depends on NumPy, pandas, SciPy and, for plots only, matplotlib. Do not add a heavyweight dependency merely because a candidate is convenient to obtain from a library; document for each proposal whether the current stack suffices. kNN support (SciPy spatial tools), abundance Good–Turing, Chao1, ICE and Hill numbers need nothing new; a logistic-regression C2ST likely needs scikit-learn unless implemented locally; GMM, k-means and HDBSCAN typically need scikit-learn or HDBSCAN; graph community methods need a graph library. Dependency-expanding candidates stay deferred until benchmarking shows the method family is worth the cost.

## 7. Required contract for every future estimator

Every estimator PR must state:

1. **Scientific question**: what exact meaning of "enough imaging" is addressed?
2. **Estimand**: what population quantity is being estimated?
3. **Null and limiting behaviour**: what happens under a homogeneous stationary population, and what does a low value actually mean?
4. **Sampling unit**: observation, FOV, incidence, imaged area or spatial block?
5. **Objective component**: `population`, `distribution` or `rare`.
6. **Observation-unit semantics**: which `observation_unit` values are valid?
7. **Representation assumptions**: continuous features, fixed categories, a scalar feature, a labelled rare state, etc.
8. **Missing-data behaviour**: how NaNs are handled; never silently impute or drop.
9. **Evidence quantity and direction**: exact value, units, and whether higher or lower is better.
10. **Parameters and provenance**: no hidden bandwidths, distances, neighbourhood sizes, state resolutions or thresholds.
11. **Randomness**: explicit `random_state` or equivalent.
12. **Prospective leakage protection**: how future-FOV information is excluded.
13. **Group independence**: group-local unless implemented as a planning comparator outside the estimator interface.
14. **Computational complexity**, for later smart microscopy.
15. **Incremental feasibility**; full recomputation remains the reference implementation.
16. **Known failure modes**: spatial correlation, rare-state blindness, dimensionality, density variation, category instability, etc.
17. **Independent falsification outcome**: what held-out real-data result would show that the method is misleading?
18. **Contracts**: strategies in `STRATEGIES_UNDER_CONTRACT`, outcome metrics in `METRICS_UNDER_CONTRACT`, current-practice baselines in `baseline_comparison` where applicable.

## 8. Implementation order of what remains

| Order | Work | Purpose |
| --- | --- | --- |
| 1 | Deep real benchmark datasets | Prevent method selection on inadequate data |
| 2 | Multi-group biological-fidelity outcome contract | Ensure preserved biology can influence method choice |
| 3 | Cross-FOV learning-curve excess risk (Fréchet first) | Cheap distributional predictive baseline |
| — | **Interim benchmark** | Decide which continuous methods merit promotion to estimators |
| 4 | kNN novelty | Local support novelty |
| 5 | Adaptive kNN support | Support at varying phenotype density |
| 6 | PRDC-style support metrics | Only if simpler support methods justify it |
| — | **Interim benchmark** | Stop if support methods add no value |
| 7 | Stable fixed-category (open-set) transform contract | Required before further ecology methods |
| 8 | Morphotype accumulation | Interpretable discovery baseline |
| 9 | Good–Turing abundance coverage, Chao1, ICE | Missing mass and unseen richness |
| 10 | Smoothed Good–Toulmin | Future-state discovery beyond one FOV |
| 11 | Hill / functional Hill diversity | Rare-to-common diversity spectrum |
| — | **Ecology benchmark** | Compare categorical coverage with continuous methods |
| 12 | Kernel / FOV distribution representations | Only if simple FOV summaries are insufficient |
| — | **FOV-predictive benchmark** | Test whether FOV modelling materially improves stopping |
| 13 | Spatial ICC / effective sample size | Spatial redundancy beyond one feature |
| 14 | pMoSS planning comparator | Only if condition-comparison benchmarking justifies it |
| 15 | Downstream biological-fidelity outcomes | Expand only for demonstrated real workflows |
| 16+ | Sequential confidence methods / Bayesian nonparametrics / GP / topology | Research reserve |

The numbering is a scientific priority order, not a commitment to complete every item. At every interim benchmark a null result is acceptable: if simple baselines perform as well as richer methods, later items may be unnecessary.

## 9. Research reserve

Worthwhile ideas that should not delay the main programme, each introduced only to address a failure demonstrated by simpler methods: Dirichlet-process and Pitman–Yor discovery probability; posterior missing mass; hierarchical Bayesian species sampling; Sinkhorn divergence; Gaussian-process spatial coverage; conformal FOV novelty; persistent-homology stabilization; deep density models.

## 10. Expected scientific product

The final result should not be prewritten as "method X wins". A plausible **type** of result:

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

## 11. Decision rule for promoting a default

A default MorphoCoverage estimator should be selected only if it shows, on held-out real datasets:

1. useful prediction of unseen information;
2. acceptable premature-stop risk;
3. robust behaviour across acquisition order;
4. clear advantages over fixed observations/FOV/area and JCB baselines in identifiable regimes;
5. no unacceptable loss of rare-state or downstream biological fidelity;
6. manageable sensitivity to representation and hyperparameters;
7. computational feasibility for the intended deployment mode.

If no single method satisfies these requirements across regimes, MorphoCoverage should expose a small set of objective-specific recommended methods rather than force one universal default.
