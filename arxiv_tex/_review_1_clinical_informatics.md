# Peer Review — Reviewer 1 (Clinical Informatics / Conformal Prediction)

**Manuscript:** "When to Alert, When to Defer: Conformal Selective Triage for ICU Mortality in Sepsis-Associated Acute Kidney Injury"

**Venue:** arXiv preprint (double-blind review)

---

## Summary

This paper presents a conformal selective triage framework for predicting in-hospital mortality among ICU patients with sepsis-associated acute kidney injury (SA-AKI), using 10,036 ICU stays from MIMIC-IV v3.1. The authors first establish that discrimination for first-24-hour SA-AKI mortality prediction is bounded (AUROC ~0.77 under honest subject-grouped evaluation), then argue that the productive research direction is not marginal AUROC improvement but rather uncertainty-aware decision governance. They apply Mondrian conformal prediction to convert calibrated XGBoost risk scores into three-way triage decisions (Alert / Defer / Clear) with finite-sample coverage guarantees. Multi-model union consensus across XGBoost, LightGBM, and logistic regression further concentrates decisions into a smaller, higher-reliability actionable region (alert PPV 0.729, clear NPV 0.962 at α = 0.05). The paper includes shift robustness experiments, subgroup heterogeneity analyses, decision-curve analysis, and a disagreement-based selective triage ablation. The authors explicitly disclaim bedside readiness and frame the work as defining a deployment boundary rather than claiming clinical utility.

---

## Strengths

1. **Methodologically rigorous evaluation design.** Subject-grouped splitting enforced throughout, with 21-seed repeated evaluation and bootstrap confidence intervals. The authors correctly identify that the 1,034 repeated-subject rows make stay-level splitting inappropriate and treat grouped evaluation as a hard constraint rather than a sensitivity check. This level of evaluation discipline is uncommon in the MIMIC prediction literature and sets a good standard.

2. **Transparent prediction ceiling analysis.** The leak-prone ceiling experiment (adding `time_to_event_hrs` to reach AUROC 0.810 vs. honest 0.763) is a clever and honest device for bounding what first-24-hour features can achieve. The negative/null probes appendix (missingness indicators, ratio features, stacking ensemble, HPO-tuned LightGBM, random survival forest — all failing to break the ceiling) further strengthens this narrative. Reporting null results is commendable and rare.

3. **Explicit data contract with conservative claims.** The paper defines an interpretive boundary upfront — no bedside-readiness, no external validation, no transportability, no exact thesis-cohort equivalence — and adheres to it consistently. This is a mature framing that avoids the overclaiming endemic to single-center MIMIC studies.

4. **Well-motivated conformal framework with clinical mapping.** The Alert/Defer/Clear trichotomy maps naturally onto ICU triage workflows. The formulation is clearly presented (Equations 1–4), and the distinction between marginal coverage guarantees and conditional coverage is correctly noted. The multi-model consensus extension (union and intersection strategies) is a genuine contribution that provides institutions with a governance surface to navigate the coverage–automation tradeoff.

5. **Shift robustness analysis is a standout contribution.** The comparison between conformal coverage stability and fixed-threshold silent recall degradation under simulated missingness is clinically compelling. The finding that fixed thresholds lose 42% relative recall at 30% missingness without signaling degradation, while conformal coverage remains stable by deferring more, is an important practical insight for deployment.

6. **Comprehensive subgroup heterogeneity reporting.** Stratification across five clinically relevant axes (age, gender, ventilation, SOFA, vasopressors) with honest reporting of coverage shortfalls in high-acuity strata (high-SOFA coverage 0.866 at α = 0.10). The paper does not hide the heterogeneity but uses it to motivate acuity-stratified deployment governance.

7. **Decision-curve analysis bridges discrimination and clinical utility.** The net benefit comparison at threshold 0.20 (XGBoost 0.133 vs. APACHE-III 0.082 vs. SOFA 0.082) provides a decision-theoretic anchor that is more informative than AUROC alone. The explicit framing of conformal policies as "action-governance layers" rather than replacements for continuous risk ranking is conceptually sound.

8. **Feature engineering is well-documented.** The nine-aggregation scheme over the T24 window, the ontology-driven drug identification pipeline (SNOMED-CT → UMLS → RxNorm → ATC crosswalk), and the three-step missingness handling are described with sufficient detail for reproduction.

---

## Weaknesses / Major Concerns

1. **The "prediction ceiling" claim is overstated relative to the evidence.** The ceiling experiment adds only one leaked variable (`time_to_event_hrs`) and reaches AUROC 0.810. The authors then claim this establishes a "hard ceiling" on first-24-hour prediction. However, this experiment bounds the ceiling *for this feature set and these model families*, not for the prediction problem itself. Unstructured clinical notes (which MIMIC-IV provides), imaging data, continuous waveform features, and more sophisticated temporal representations (e.g., transformer-based sequence models over the 24-hour window) could plausibly push discrimination higher. The paper should reframe the ceiling as "ceiling under the current feature engineering and model family constraints" rather than a property of the clinical problem. The discussion partially acknowledges this (mentioning free-text notes and imaging) but the results section language ("hard ceiling") is stronger than the evidence supports.

2. **No race/ethnicity subgroup analysis despite fairness claims.** The methods section states that fairness was assessed with a pre-specified tolerance of ΔAUROC ≤ 0.05 across demographic groups, and the evaluation protocol mentions "sex, ethnicity, AKI stage, age quartile" for subgroup analysis. However, the actual subgroup results (Table 6, Figure 9) report only gender, age band, SOFA band, ventilation, and vasopressor status — ethnicity is entirely absent. For a paper that explicitly invokes fairness criteria, this is a significant omission. MIMIC-IV has well-documented racial disparities in ICU outcomes, and conformal coverage could plausibly vary across racial/ethnic groups due to differential feature missingness patterns and care-process variation. This analysis must be included or the fairness language must be removed.

3. **The 68% defer rate at α = 0.05 undermines practical utility.** At the primary operating point, the conformal framework defers on 68.2% of patients. The paper frames this as a feature ("when it speaks, it is reliable"), but from a clinical workflow perspective, a system that provides no recommendation for two-thirds of patients has limited value over no system at all. The clinician must still evaluate every deferred patient using their existing workflow. The paper does not adequately address: (a) what incremental value the system provides for deferred patients (is the continuous risk score still shown?), (b) whether the 31.8% automation rate justifies the infrastructure cost of deployment, and (c) how clinicians would perceive and interact with a system that mostly says "I don't know." The sweet-spot analysis at α = 0.12 (53.2% certain-decision fraction) partially addresses this, but the primary operating point presentation buries the practical utility concern.

4. **No comparison with other uncertainty quantification methods.** Conformal prediction is compared only against fixed thresholds and the disagreement heuristic. The UQ literature offers several alternatives that should be benchmarked: Monte Carlo dropout, deep ensembles, Bayesian logistic regression, and calibration-based abstention (e.g., selective prediction via the Chow reject option). Without these comparisons, the reader cannot assess whether conformal prediction's coverage guarantee justifies its high defer rate relative to methods that might achieve better coverage–automation tradeoffs without formal guarantees. At minimum, a calibration-based selective classifier (abstain when max class probability < threshold) should be included as a baseline.

5. **Conditional coverage failure in high-acuity subgroups is insufficiently addressed.** High-SOFA patients achieve only 0.866 coverage at α = 0.10 — well below the nominal 0.90 target. The paper acknowledges this as a limitation of marginal coverage but does not explore solutions. Recent work on group-conditional conformal prediction (e.g., Romano et al. 2020, Barber et al. 2021) and Mondrian conformal prediction with finer class-conditional grouping could potentially address this. The paper should either implement SOFA-stratified conformal thresholds (which it suggests as a governance policy but does not evaluate) or discuss why this was not pursued.

6. **Missing feature importance / explainability analysis.** For a 162-feature XGBoost model intended for clinical triage, the absence of any feature importance or SHAP analysis is a notable gap. Clinicians need to understand which features drive Alert vs. Clear decisions to trust the system. The paper discusses clinical interpretability of the triage *actions* but not of the underlying *model*. Even a top-20 SHAP summary plot would substantially strengthen the clinical relevance argument.

7. **The disagreement baseline comparison is somewhat asymmetric.** The disagreement approach is presented as clearly inferior (heuristic threshold, no coverage guarantee), but the comparison is not entirely fair. The disagreement method uses two different feature subsets (full vs. physiology-only), which provides a qualitatively different uncertainty signal than conformal prediction's nonconformity scores. The paper does not explore whether combining disagreement signals with conformal prediction could improve the coverage–automation tradeoff. Additionally, the disagreement baseline's defer rate (0.740) is actually higher than conformal's (0.682), yet this is not prominently discussed.

8. **Calibration analysis lacks depth.** The paper reports ECE and calibration slope for the primary models but does not discuss: (a) calibration stability across the 21 seeds, (b) calibration within subgroups (are probabilities well-calibrated for high-SOFA patients specifically?), and (c) the interaction between isotonic calibration and conformal prediction. Since conformal prediction's practical performance depends heavily on calibration quality, and isotonic regression can overfit on small calibration sets, this interaction deserves more attention.

---

## Minor Issues

1. The methods section mentions "sex, ethnicity, AKI stage, age quartile" for subgroup analysis, but the results report "age band, gender, mechanical ventilation, SOFA band, vasopressor use." The mismatch between planned and reported subgroups should be reconciled — AKI stage and ethnicity are missing; ventilation and vasopressors were added.

2. The "sweet spot" criterion (NPV ≥ 0.90 and PPV ≥ 0.55) appears without clinical justification. Why these specific thresholds? Are they derived from clinical guidelines, expert consensus, or arbitrary? A brief justification or citation would strengthen this operating-point selection.

3. The paper uses the word "honest" repeatedly (e.g., "honest grouped setting," "honest AUROC") as a descriptor for the subject-grouped evaluation. While the intent is clear, this is somewhat loaded language that implicitly characterizes other studies as "dishonest." Consider replacing with "subject-grouped" or "leak-free" throughout.

4. The 48-hour mortality endpoint (prevalence 0.004, AUROC 0.705, no usable PPV ≥ 0.50 operating point) adds little value. With only ~40 events in the full cohort, this analysis is severely underpowered. Consider removing it or relegating it to a single sentence noting that the event rate was too low for meaningful analysis.

5. No CONSORT flow diagram is provided despite the methods section describing a "CONSORT-style reduction." A visual cohort flow diagram showing the sequential application of inclusion/exclusion criteria with patient counts at each step would improve reproducibility and is expected for observational cohort studies.

6. The paper is very long. The results section contains extensive interpretive prose that overlaps substantially with the discussion. Consider tightening the results to report findings with minimal interpretation, reserving the clinical contextualization for the discussion.

7. The paper does not report computational cost or inference time. For a deployment-oriented framework, knowing whether conformal triage adds meaningful latency over a simple threshold is relevant.

8. The intersection consensus coverage of 0.911 at α = 0.05 (below the nominal 0.95 target) is noted but not flagged as a potential concern for deployment. If an institution selects the intersection strategy, they should understand that the coverage guarantee does not hold. This deserves a more prominent warning.

9. Several figures are referenced but their quality and readability cannot be assessed from the LaTeX source alone. Ensure that all figures use colorblind-safe palettes and have sufficient resolution for print.

10. The reproducibility appendix is helpful but does not specify exact software versions (Python, XGBoost, LightGBM, scikit-learn, MAPIE/crepes or whichever conformal library was used). A `requirements.txt` or environment specification would strengthen reproducibility claims.

---

## Questions for Authors

1. **Why is race/ethnicity absent from the subgroup analysis?** The methods explicitly mention ethnicity as a planned stratification axis and invoke a ΔAUROC ≤ 0.05 fairness bound. Was this analysis performed and omitted, or was it not conducted? If performed, what were the results? If not, what prevented it?

2. **Have you considered or evaluated group-conditional conformal methods to address the high-SOFA coverage shortfall?** Specifically, computing separate nonconformity thresholds for high-SOFA vs. low-SOFA patients (or using Mondrian conformal prediction with SOFA band as the Mondrian class) could potentially restore conditional coverage. What are the practical barriers?

3. **For the 68% of patients deferred at α = 0.05, what information does the clinician receive?** Is the continuous risk score still displayed alongside the "Defer" label? If so, the system is effectively a risk score with a confidence flag rather than a three-way triage tool, and the framing should reflect this. If not, the clinician receives less information than a simple risk score would provide.

4. **Why was XGBoost selected as the primary model over CatBoost?** Both achieved AUROC 0.768 and Brier 0.161, but CatBoost had higher recall at PPV ≥ 0.50 (0.623 vs. 0.608). The stated reason — that CatBoost "did not materially simplify or improve the downstream conformal story" — is vague. Was the conformal operating curve actually compared between XGBoost and CatBoost?

5. **Can you provide evidence that the conformal coverage guarantee is not an artifact of the calibration–conformal interaction?** Isotonic regression on a finite validation set can produce poorly calibrated probabilities in low-density regions of the score distribution. Since conformal prediction's nonconformity scores are derived from these calibrated probabilities, miscalibration in the tails could affect the coverage guarantee in practice. Have you tested conformal coverage with uncalibrated scores or with Platt scaling as an alternative?

---

## Overall Assessment

**Recommendation: Minor Revision**

**Justification:**

This is a well-conceived and methodologically careful paper that makes a genuine contribution to the clinical informatics literature. The core insight — that bounded discrimination should redirect research effort from AUROC optimization toward uncertainty-aware decision governance — is important and well-argued. The conformal selective triage framework is clearly formulated, the evaluation design (subject-grouped, 21-seed repeated, with explicit data contract) is among the most rigorous I have seen in the MIMIC prediction literature, and the shift robustness analysis is a standout contribution.

However, several issues prevent acceptance in the current form:

- The missing race/ethnicity subgroup analysis is a significant gap that must be addressed, especially given the paper's own fairness claims.
- The prediction ceiling claim needs to be scoped more carefully to the current feature set rather than presented as a property of the clinical problem.
- The practical utility of a 68% defer rate needs more honest engagement — either through a more prominent presentation of the sweet-spot operating point or through a clinical workflow analysis.
- The absence of comparisons with other UQ methods (even a simple max-probability abstention baseline) weakens the claim that conformal prediction is the right tool for this task.
- Feature importance / explainability analysis should be included for clinical credibility.

None of these issues require fundamental redesign of the study. The race/ethnicity analysis can be added from existing data. The ceiling claim can be reframed with careful language. The UQ comparison and SHAP analysis are straightforward additions. I therefore recommend minor revision with the expectation that these issues can be addressed in a single revision cycle.

The paper's self-aware limitations section, conservative claims, and transparent reporting of null results reflect scientific maturity that should be recognized. With the requested revisions, this manuscript would make a strong contribution to the literature on uncertainty-aware clinical prediction.

---

*Reviewer expertise: Clinical informatics, ICU prediction models, conformal prediction, calibration methods, MIMIC-IV analyses.*

*Conflicts of interest: None declared.*
