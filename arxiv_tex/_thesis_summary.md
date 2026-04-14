# MTech Thesis Comprehensive Summary
## SA-AKI Mortality Prediction — IIT Madras MTech in Industrial AI

---

## 1. Introduction (Chapter 1)

### Thesis Scope
The thesis investigates two clinical decision support problems using MIMIC-IV:
- **Problem 1:** Automated ICD-10-CM code prediction for rare diseases (proposed but NOT fully executed due to computational costs — only literature review, problem formulation, dataset exploration, and KNN baseline completed).
- **Problem 2:** Early mortality risk stratification for SA-AKI (FULLY executed).

### Clinical Background — SA-AKI
- SA-AKI couples systemic infection with acute renal failure and is associated with sharply elevated in-hospital mortality.
- SA-AKI is labeled when AKI develops within 7 days of sepsis onset.
- **Sepsis** defined by Sepsis-3: life-threatening organ dysfunction from dysregulated host response to infection, operationalized as SOFA increase ≥2 points.
- **AKI** defined by KDIGO: absolute creatinine increase ≥0.3 mg/dL within 48h, or ≥1.5× baseline within 7 days, or oliguria ≤0.5 mL/kg/h for 6 hours.
- SA-AKI accounts for 40–70% of AKI in ICU; mortality rates commonly 40–60%.

### What the Literature Has Tried and Where It Falls Short (Key Passage)
The thesis identifies these recurring methodological problems in prior work:

1. **Temporal anchoring is vague / whole-stay aggregation used** — future information flows into features presented as "early predictors."
2. **Length of stay included as predictor** — direct leakage channel from outcome to covariates, inflates discrimination.
3. **Feature selection and imputation on full dataset** — not learned on development folds then applied to held-out data → optimistic estimates.
4. **SMOTE/ADASYN misuse** — synthetic oversampling without assessing whether generated examples concentrate in overlapped/noisy regions or near outliers where decision boundaries are unstable.
5. **Calibration and threshold selection inconsistently reported** — limits bedside interpretability.
6. **Single-center development** — performance losses under transport to other hospitals documented.

### Research Objectives
1. Systematic EDA to identify clinical phenotypes and risk factors.
2. Develop and validate a well-calibrated LightGBM classifier using only T0–T24 features.
3. Ensure transparency via SHAP explanations.
4. Complement binary prediction with survival models (Cox PH and DeepSurv).

---

## 2. Literature Review (Chapter 2)

### Part I: Automated ICD Coding (Summary)
- Covers evolution from rule-based systems → CNNs (CAML) → RNNs (LSTM/GRU) → Transformers (BERT, PLM-ICD, D2SBERT) → Modern BERT variants (MosaicBERT, ModernBERT).
- Key challenges: extreme multi-label classification, long-tailed code distribution (>9,217 ICD-10 codes appear <5 times in MIMIC-IV), long clinical documents, explainability.
- Curriculum learning (HiCu) and hyperbolic embeddings help with rare codes.
- Synthetic data generation via LLMs (GPT-3.5/4) for rare codes shows modest gains.

### Part II: SA-AKI Mortality Prediction — Detailed Criticisms of Prior Work

#### Conventional Severity Scores
- SOFA, SAPS II, APACHE II show only modest discrimination for SA-AKI mortality.
- Cannot capture nonlinear/interacting physiology linking infection, hemodynamics, renal injury, and treatment.

#### Datasets and Cohort Construction
- MIMIC-IV is the most common source.
- Studies typically restrict to first ICU admission, exclude <18y, remove LOS <48h.
- **Critical design decision:** temporal window for features. Fixed early windows (24h/48h) support timeliness; whole-stay aggregation creates risk of future information entering model inputs.

#### Feature Engineering and Leakage Risks
- Feature selection often informed by the full dataset → predictor set influenced by outcomes from evaluation records → optimism.
- VIF-based pruning claimed to enhance stability without pre/post diagnostics or external tests; VIF has limited relevance for models that tolerate correlated predictors.
- **Length of stay included as predictor** — strong proxy for time-to-event, not available prospectively.
- Redundancy: total urine output and average urine output become scalar multiples under fixed window; diverging importance profiles suggest non-fixed/whole-stay windows.
- Creatinine and derived eGFR both included without clear rationale.

#### Missing Data
- Mechanisms rarely examined in depth.
- MCAR/MAR assumptions stated without empirical assessment.
- Mean imputation used even for skewed biomarkers (e.g., creatinine is right-skewed, log-normal).
- MICE/random forest imputation fitted across entire dataset before train-test separation → leakage.

#### Class Imbalance
- SMOTE and ADASYN sometimes applied together without clear rationale.
- SMOTE may generate synthetic points in overlapped/noisy areas; performance degrades in high dimensions without filtering.
- ADASYN concentrates synthesis near low-density/borderline regions → can shift decision boundaries around outliers, harm stability.
- Comparisons with intrinsic class weighting or hybrid sampling not always provided.

#### Evaluation Strategy
- Single train-test split without separate validation split.
- Explicit calibration and principled threshold selection omitted.

#### Model Tuning
- Grid search CV described as "optimal" despite nonconvex search spaces where only local minima are guaranteed.
- Calibration reported inconsistently.
- SHAP stability not explored.
- Confidence intervals for ensembles and linear models often overlap.

#### External Validity
- MIMIC-IV reflects single tertiary-care center in Boston.
- Performance drift across hospitals documented (Rockenschaub 2024).
- Timing of features relative to decision point often unspecified.

### Detailed Appraisals of Specific Studies

#### Chen et al. (2025) — AUROC 0.878
**Flaws identified:**
- Feature selection on entire dataset → leakage.
- Length of stay included as predictor → direct leakage.
- Temporal window for biomarker aggregation not specified; signals suggest whole-stay aggregation.
- VIF filtering described as enhancing stability without diagnostics; limited relevance for ensembles.
- SMOTE + ADASYN applied together without rationale.
- Mean imputation for skewed variables (creatinine).
- Missingness indicators not created; mechanism assumed not studied.
- 75/25 split without separate validation set.
- No explicit calibration or threshold selection.
- Claimed generalizability from single-center data.
- t-tests used for non-normal laboratory variables.
- CI overlap: XGBoost AUROC 0.878 (0.859–0.897) vs. LR 0.849 (0.824–0.873).
- Internal inconsistency: criticized Li et al. for 44 features being "too narrow" but used only 24.
- Base estimator for RFE not stated.

#### Li et al. (2023) — AUROC 0.794
**Flaws identified:**
- Lasso feature selection on full dataset → leakage.
- MICE imputation fitted globally → leakage.
- Missingness indicators not constructed.
- Linear framework prioritized main effects, could miss interactions.
- 80/20 split; discrimination gap (XGBoost 0.794 vs. LR 0.730) hard to interpret given selection/imputation scope.

#### Hu et al. (2021) — C-index 0.72
**Flaws identified:**
- Whole-stay median aggregation → features include post-baseline information.
- MICE on full dataset → leakage.
- Features with >20% missingness dropped without mechanism investigation.
- Test set only 102 rows → unreliable estimates.
- Ethnicity used as predictor → fairness/transportability concerns.

#### Roknaldin et al. (2024) — AUC 0.887
**Flaws identified:**
- Feature selection and SMOTE described prior to final split → influenced by evaluation records.
- Multiple imputation fitted globally.
- Fixed correlation threshold risks removing weakly correlated but jointly informative variables.
- Min/max eGFR alongside creatinine = redundancy (eGFR derived from creatinine).
- t-tests for skewed markers.
- Half data to validation, 1/10 to testing → small test set.
- Calibration not tied to recalibration step excluding test cohort.

#### Chaudhary et al. (2020) — Phenotype Discovery
- Autoencoder + clustering on whole-stay trajectories → embeds future information.
- Sepsis cohort by ICD codes → timing misalignment.
- No external validation of cluster stability.

#### He et al. (2021) — AUROC 1.00 (internal)
- Perfect discrimination with 209 patients and >24 features → overfitting/leakage suspected.
- Not clear imputation/preprocessing confined to development data.

#### Lin et al. (2025) — CKD Progression
- Lasso/stepwise screening possibly on full dataset → leakage.
- SHAP thresholds sensitive to sampling variation.

#### Kantola et al. (2025) — Procalcitonin
- Admission procalcitonin: OR ~1.01/ng/mL, AUROC ~0.59 → limited standalone utility.
- Procalcitonin not available in MIMIC-IV.

### Literature Synthesis Key Takeaway
"Studies that relied on whole-stay aggregation, on inclusion of length of stay or other post-baseline correlates, and on global feature selection or imputation tend to report stronger discrimination, yet those gains are paired with ambiguity about practical timing and with higher risk of optimism."

---

## 3. Problem Definition (Chapter 3)

### SA-AKI Problem Formulation (Problem 2)

#### Decision Point
At end of first ICU day (T_obs = 24h after ICU admission), estimate risk of in-hospital death for SA-AKI patients using only information available by T_obs.

#### Three Sub-problems
- **Sub-problem A (Unsupervised):** EDA and phenotyping of training data.
- **Sub-problem B (Supervised, Primary):** Binary mortality classifier f_θ: X → [0,1].
- **Sub-problem C (Survival):** Time-to-death modeling with right censoring.

#### Target Population
- Sepsis: Sepsis-3 (suspected infection + daily SOFA increase ≥2).
- AKI: KDIGO serum creatinine and urine output criteria.
- SA-AKI: AKI onset within ±48h of sepsis onset.
- Adults ≥18y, ICU LOS ≥24h, first ICU stay only.
- t_sepsis and t_aki used only for cohort identification, NOT as predictors.

#### Data Dimensions
- Training set: (8,028 patients, 162 features including target).
- Test set: (2,008 patients, 162 features).
- Positive (death) rate in training: 26.74%.
- Survival analysis: (8,028, 124) and (2,008, 124) before time/event columns.

#### Temporal Discipline
"Any statistic requiring t > T_obs (e.g., total ICU length of stay, whole-stay extrema/means, post-baseline therapies) is deemed inadmissible for prediction and is excluded by a scripted leakage audit."

#### Performance Targets
- Discrimination: High AUROC and F1-score on held-out test set.
- Calibration: Reliable probabilities via isotonic regression.
- Threshold: Youden's J statistic on internal validation split.

#### Constraints and Guardrails
1. All features from [0, 24]h window only.
2. Only first ICU stay per patient.
3. All transformations learned on training set only.
4. Test set used only once for final evaluation.
5. Cohort timestamps not used as predictors.

---

## 4. Methodology (Chapter 4)

### SA-AKI Pipeline (Problem 2)

#### Cohort Construction (T0–T24)
- **ETL:** MIMIC-IV integrated with medical ontologies (RxNorm, ATC, SNOMED CT) via UMLS in PostgreSQL.
- **Validation suite:** Unit harmonization (F→C, lb→kg), referential integrity, logical consistency.
- **Cohort:** Adults ≥18y, first ICU stay, LOS ≥24h.
- **Sepsis-3:** Suspected infection (body-fluid culture near systemic antibiotic) + ΔSOFA ≥2. Onset time = start of first qualifying ICU day.
- **AKI (KDIGO 2012):** SCr changes or urine output. Hierarchical baseline SCr. AKI onset = earliest time any criterion met.
- **SA-AKI:** AKI onset within [t_sepsis − 48h, t_sepsis + 48h].
- **Exclusions:** Prior ESRD, RRT within first 24h.

#### Data Split and Leakage Control
- One-time stratified 80/20 train-test split.
- All data-dependent objects (imputation, scaling, encoding) fitted ONLY on D_train.
- Any validation set carved from WITHIN D_train.
- **Leakage audit:** Scripted audit ensuring no post-T24 variables included. Removed features like total ICU LOS and observation counts (_n) that proxy care intensity/illness duration.

#### Preprocessing (Train-Only)
1. **Missingness analysis:** Little's MCAR test → rejected → data not MCAR → careful handling justified.
2. **Missingness indicators:** Binary _missing indicators created; chi-square test for association with mortality; only significant (p<0.05) indicators retained as features.
3. **Imputation:** Median imputation from training set.
4. **Feature selection:** Pearson correlation >0.99 → remove one from pair.
5. **Scaling:** StandardScaler fit on training data only.

#### Model Development
- **Benchmarking:** LR, LDA, QDA, Naive Bayes, regularized linear models, Decision Tree, Random Forest, XGBoost, LightGBM.
- **Tuning:** Optuna for tree-based ensembles.
- **Imbalance:** class_weight='balanced' compared against SMOTE, ADASYN in sensitivity analyses.

#### Final Model Pipeline
1. Train optimized LightGBM on sub-split of training data.
2. Calibrate outputs on held-out validation split (from training set) using CalibratedClassifierCV with isotonic method.
3. Determine optimal threshold via Youden's J on validation split.
4. Retrain final calibrated LightGBM on entire D_train.
5. Evaluate on untouched D_test using predetermined threshold.

#### Interpretability
- SHAP beeswarm summary plot for global feature importance and direction of effects.

#### Survival Analysis
- **Cox PH:** lifelines library; top 50 predictors from initial full model; penalizer tuned; forest plot of hazard ratios.
- **DeepSurv:** Custom PyTorch neural Cox model; SELU activations, residual connections; negative Cox partial log-likelihood with Efron method.
- **Evaluation:** Harrell's C-index, time-dependent AUC, calibration plots at 90/180/365 hours, KM curves by risk quartiles, log-rank test.

#### Three Types of Data Leakage Addressed
1. **Temporal leakage (look-ahead bias):** Features from after T24, whole-stay aggregates, total ICU LOS. → Prevented by T0–T24 restriction + scripted audit.
2. **Preprocessing leakage:** Imputation/scaling/selection fitted on full dataset. → Prevented by train-only fitting.
3. **Selection leakage:** Feature screening on full dataset. → Prevented by confining screening to training partition.

#### 24-Hour Window Heterogeneity Discussion
- Patients arrive at different illness stages (ED vs. ward transfer).
- T0–T24 captures different disease phases across patients.
- Acknowledged as genuine limitation but reflects real bedside conditions.
- APACHE III, SOFA at 24h, Charlson comorbidity, first/last/delta/slope suffixes partially account for variation.
- Future work: stratified analyses by admission source; alternative anchoring to sepsis onset.

#### TRIPOD Adherence
Explicitly addresses: source of data, participants, outcome/predictors, sample size, missing data, statistical methods, model specification/performance.

---

## 5. Dataset Understanding and EDA (Chapter 5)

### SA-AKI Cohort (Part II)

#### Source Tables
- MIMIC-IV core, hosp, ICU modules: vitals, labs, medications, microbiology, encounter scaffolding.

#### Ontology-Driven Concept Identification
- Clinical concepts mapped across terminologies: SNOMED CT → UMLS CUIs → RxNorm ingredients, with ATC cross-reference.
- Underlies systemic antibiotic and vasopressor flags.

#### Cohort Construction (CONSORT-style)
- All ICU stays → adults, first ICU stay, LOS ≥24h → Sepsis-3 screening → SA-AKI temporal linkage → exclusions (ESRD, early RRT).
- **Final cohort: n = 10,036 ICU stays.**

#### Clinical Labeling Rules
- Sepsis-3: suspected infection AND ΔSOFA ≥2.
- AKI: KDIGO SCr (absolute/relative) and urine output criteria.
- SA-AKI: AKI onset within ±48h of sepsis onset.
- All predictors restricted to T0–T24.

#### Cohort Statistics
| Metric | Value |
|--------|-------|
| In-hospital death | 26.7% (2,684/10,036) |
| Male | 53.0% |
| AKI trigger = Creatinine | 40.7% |
| Mechanical ventilation (24h) | 53.0% |
| Vasopressor (24h) | 36.6% |
| RRT (24h) | 69.0% |

- Stratified 80/20 split: 8,028 train / 2,008 test.
- Training mortality rate: 26.74%.

#### EDA Findings

**Statistical Validation:**
- Key clinical variables: highly significant differences between survivors/non-survivors (all p < 0.001, Mann-Whitney U).
- No significant differences between train/test sets (all p > 0.05) → successful stratified split.

**Missing Data Analysis:**
- 101 of 161 features had missing values.
- Little's MCAR test: strongly rejected (p ≈ 0.0) → data NOT MCAR.
- Chi-square: 46 missingness indicators significantly associated with mortality (p < 0.05).
- Top associations: AST, alkaline phosphatase, total bilirubin, LDH, calcium missingness indicators.
- "Patterns of clinical data collection are themselves informative predictors of outcome."

**Clinical Phenotypes (Survivors vs. Non-Survivors):**
- Non-survivors: older, higher APACHE III, higher SOFA, higher max creatinine, higher max lactate.
- Strongest positive correlations with death: sofa_total_24hr (0.234), rdw_median_pct (0.188), sofa_hepatic_24hr (0.169).
- Strongest negative correlations (survival): urine_output_total_24hr (−0.113), ph_min (−0.101).

---

## 6. Results (Chapter 6)

### Model Benchmarking

| Model | ROC-AUC | F1-Score |
|-------|---------|----------|
| LightGBM | 0.7492 | 0.4129 |
| XGBoost | 0.7444 | 0.4527 |
| Logistic Regression | 0.7382 | 0.5227 |
| LDA | 0.7364 | 0.3756 |
| ElasticNet | 0.7363 | 0.5185 |
| Ridge | 0.7349 | 0.5174 |
| DNN++ | 0.7347 | 0.4471 |
| Lasso | 0.7331 | 0.5244 |
| Random Forest | 0.7137 | 0.3468 |

- Tree-based ensembles consistently outperformed linear models in discrimination.
- Oversampling methods produced higher F1 but at expense of precision, without superior AUROC.
- LightGBM with class weights selected as final model.

### Final Model Performance (Held-Out Test Set)

| Metric | Value |
|--------|-------|
| ROC-AUC | 0.7488 |
| Accuracy | 0.6648 |
| Precision | 0.4220 |
| Recall (Sensitivity) | 0.6853 |
| F1-Score | 0.5224 |

- Optimal threshold: 0.2704 (Youden's J on validation set).
- Correctly identified 368/537 deaths (recall 0.685).

### Why LightGBM Outperforms
- Leaf-wise tree growth concentrates splitting on highest-loss leaves → effective for mixed continuous/sparse binary features.
- Natively captures interaction structure (SOFA × hemodynamic instability × metabolic derangement) without explicit feature engineering.
- class_weight='balanced' more effective than SMOTE/ADASYN — oversampling inflated recall at cost of precision without improving discrimination, consistent with known risks of synthesis in overlapped regions.
- Optuna explored richer hyperparameter space than grid search.

### SHAP Interpretability
Top predictors driving mortality:
- sofa_total_24hr (global organ dysfunction)
- rdw_median_pct (hematological dysfunction)
- apache_iii_score (illness severity)
- Age
- bun_median (renal failure)

Top predictors of survival:
- urine_output_total_24hr (preserved kidney function)

"Concordance between model's learned patterns and clinical pathophysiology interpreted as evidence supporting validity."

### Five Dimensions of Trustworthiness

1. **Methodological integrity:** AUROC 0.749 obtained with all preprocessing on training data only, no post-T24 features, test set used once, no feature selection on test outcomes. Prior studies reporting 0.79–0.88 include LOS as predictor, global feature selection, or whole-stay aggregation.

2. **Calibration:** Isotonic regression on validation split. "When model assigns 30% risk, approximately 30% of patients in that bin died." Several prior studies did not report calibration at all.

3. **Clinical face validity (SHAP):** Top predictors are established clinical risk factors. Model driven by clinically interpretable signals, not dataset artifacts.

4. **Decision-curve analysis:** Positive net benefit over "treat all" and "treat none" across 8–30% risk thresholds.

5. **Cross-model concordance:** Consistent results across LightGBM (AUROC 0.749), CoxPH (C-index 0.693), DeepSurv (C-index 0.688). "If predictive signal were spurious or driven by leakage, it would be unlikely to manifest consistently across" three fundamentally different model families.

### Comparison with Prior Work

| Criterion | This Work | Chen (2025) | Li (2023) | Hu (2021) |
|-----------|-----------|-------------|-----------|-----------|
| Feature window | T0–T24 | Unspecified | T0–T24 | Whole-stay |
| LOS as predictor | No | Yes | No | No |
| Train-only preprocessing | Yes | No | No | No |
| Missingness indicators | Yes | No | No | No |
| Explicit calibration | Yes | No | No | No |
| Separate validation set | Yes | No | No | No |
| Reported AUROC | 0.749 | 0.878 | 0.794 | — |
| Reported C-index | 0.69 | — | — | 0.72 |

**Key argument:** "The AUROC of 0.749 should be interpreted not as an inferior result but as a more realistic and credible estimate of what is achievable for a genuine early warning system that relies exclusively on information available at the bedside within 24 hours."

### Survival Analysis Results

#### Cox PH
- Test C-index: 0.693.
- Top hazard predictors: metastatic cancer, high SOFA scores.
- RRT and vasopressor use associated with reduced hazard (confounding by indication, not causal).
- KM curves by risk quartiles: clear separation, log-rank p < 0.001.

#### DeepSurv
- Test C-index: 0.688.
- Strong risk stratification (log-rank p < 0.001).
- Calibration plots at 90/180/365 hours: good agreement, slight overestimation in highest-risk bins.
- DCA: net benefit over default strategies for 8–30% thresholds.

#### Survival Synthesis
"Both models delivered consistent and clinically interpretable predictions. Hu et al. (2021) reported C-index 0.72 but used whole-stay aggregated features, which introduces high risk of data leakage. Results obtained here from T0–T24 data provide a more reliable benchmark."

### Limitations
- No external validation (MIMIC-IV only, single center).
- eICU identified as natural candidate for external validation.
- Harmonization challenges: different lab coding, variable availability, population differences.
- "Many prior studies claimed broad generalizability without providing external evidence."

---

## 7. Conclusions and Future Work (Chapter 7)

### Key Contributions
1. **High-fidelity cohort:** 10,036 SA-AKI patients from MIMIC-IV with ontology-integrated PostgreSQL warehouse.
2. **Leakage-aware preprocessing:** Formal missingness analysis, significant missingness indicators as features, train-only transformer fitting.
3. **Interpretable model:** LightGBM with AUROC 0.749, F1 0.522; SHAP confirms clinical validity (SOFA, RDW, age).
4. **Dynamic risk assessment:** CoxPH (C-index 0.693) and DeepSurv (C-index 0.688) for time-to-death.

### Future Directions
1. **External validation:** eICU (200+ hospitals); re-apply Sepsis-3/KDIGO, re-derive T0–T24 features, evaluate frozen model.
2. **Monte Carlo synthetic augmentation:** Copula-based simulation or parametric bootstrap preserving correlation structure — unlike SMOTE/ADASYN which operate without regard to clinical plausibility.
3. **Multi-database risk framing:** Amsterdam UMCdb, HiRID, ANZICS for federated/meta-analytic approach.
4. **Different temporal windows:** 6h, 12h (earlier intervention) or 48h (evolving trajectories).
5. **Fairness audit:** Performance across sex, age, ethnicity subgroups; Δ AUROC ≤ 0.05.
6. **Prospective clinical impact study:** Deploy as decision support tool in real ICU.

---

## Summary of Methodological Criticisms of Prior Work (Cross-Chapter Synthesis)

The thesis systematically identifies these flaws across the SA-AKI prediction literature:

| Flaw | Description | Studies Affected |
|------|-------------|-----------------|
| **Temporal leakage** | Whole-stay aggregation or unspecified windows embed future information | Chen 2025, Hu 2021, Chaudhary 2020 |
| **LOS as predictor** | Direct proxy for whether/when death occurred | Chen 2025 |
| **Global feature selection** | Selection on full dataset before split → evaluation outcomes influence predictor set | Chen 2025, Li 2023, Roknaldin 2024, Lin 2025 |
| **Global imputation** | MICE/imputation fitted on full dataset → test info leaks into training | Chen 2025, Li 2023, Hu 2021, Roknaldin 2024 |
| **SMOTE/ADASYN misuse** | Applied without assessing synthesis in overlapped/noisy regions; sometimes combined without rationale | Chen 2025, Roknaldin 2024 |
| **No calibration** | Predicted probabilities not calibrated; limits clinical utility | Chen 2025, Li 2023, Hu 2021 |
| **No threshold selection** | No principled operating point determination | Chen 2025 |
| **Mean imputation for skewed data** | Creatinine is right-skewed/log-normal; mean imputation inappropriate | Chen 2025 |
| **Missingness not studied** | Mechanism assumed not tested; indicators not created | Chen 2025, Li 2023, Hu 2021 |
| **Small test sets** | Unreliable performance estimates | Hu 2021 (102 rows), Roknaldin 2024 |
| **No external validation** | Single-center claims of generalizability | All studies |
| **t-tests for non-normal data** | Parametric tests on skewed laboratory variables | Chen 2025, Roknaldin 2024 |
| **Redundant features** | eGFR + creatinine; total + average urine output | Chen 2025, Roknaldin 2024 |
| **Grid search called "optimal"** | Discrete grid in nonconvex space ≠ global optimum | Chen 2025 |
| **Perfect discrimination** | AUROC 1.00 with 209 patients → overfitting/leakage | He 2021 |

---

## Feature Engineering Details (from Chapters 4–5)

### Time-Series Aggregation (T0–T24 Window)
- 9 statistical aggregations applied to time-varying features: **first, last, median, IQR, range, delta, AUC, slope, count**.
- Static features: demographics, comorbidities (Charlson components).
- Binary flags: mechanical ventilation, vasopressor, RRT within 24h.
- Severity scores: SOFA (total + sub-scores at 24h), APACHE III.

### Final Feature Set
- 162 features (including target column) after preprocessing.
- 124 features for survival analysis.
- Highly correlated features (r > 0.99) removed.
- Significant missingness indicators appended.
- All scaling/imputation fitted on training set only.
