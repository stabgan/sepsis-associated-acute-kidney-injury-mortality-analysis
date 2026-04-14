# Reviewer 2 — Machine Learning, Statistics, and Conformal Prediction

**Manuscript:** "When to Alert, When to Defer: Conformal Selective Triage for ICU Mortality in Sepsis-Associated Acute Kidney Injury"

**Reviewer expertise:** Conformal prediction, distribution-free inference, model calibration, uncertainty quantification in clinical ML.

---

## Summary

This paper applies Mondrian conformal prediction to convert calibrated mortality risk scores into three-way triage decisions (Alert / Defer / Clear) for SA-AKI patients in MIMIC-IV. The authors first establish that first-24-hour discrimination is bounded (AUROC ≈ 0.77 across model families), then argue that the scientific target should shift from incremental AUROC improvement to uncertainty-aware decision governance. Conformal prediction sets are evaluated at multiple miscoverage levels, extended to multi-model consensus (union and intersection), and stress-tested under simulated distribution shift. A disagreement-based deferral baseline is included as an ablation. The paper is framed as a clinical informatics contribution rather than a methods paper, and it explicitly disclaims bedside readiness.

---

## Strengths

1. **Correct core conformal formulation.** The Mondrian conformal prediction setup (Equations 2–4) is standard and correctly stated. The nonconformity score $s_i^{(y)} = 1 - f(x_i)^{[y]}$, the finite-sample quantile correction, and the class-conditional threshold estimation are all technically sound. The coverage guarantee is correctly described as marginal under exchangeability, not conditional.

2. **Subject-grouped evaluation as a hard constraint.** The paper treats subject-grouped splitting as a non-negotiable design requirement rather than an optional sensitivity analysis. With 1,034 repeated-subject rows among 10,036 stays, this is the correct decision. The 21-seed repeated evaluation provides meaningful stability estimates, and the paper avoids the common mistake of reporting stay-level metrics as if they were patient-level.

3. **Honest bounded-ceiling narrative.** The leak-augmented ceiling experiment (AUROC 0.810 with time-to-event as a feature vs. 0.763 honest) is a genuinely useful contribution. Quantifying the gap between achievable and honest discrimination is rare in clinical ML papers and provides a concrete upper bound that contextualizes all downstream results. The comprehensive negative-results appendix (Table B12) reinforces this narrative with admirable transparency.

4. **Clinically grounded three-way triage.** The Alert/Defer/Clear trichotomy is well-motivated for ICU workflows. The paper correctly frames the Defer region as a feature (explicit abstention channel) rather than a failure mode, and the operating-characteristic curve across α levels provides a practical governance tool for institutions with different risk tolerances.

5. **Decision-curve analysis with appropriate comparators.** The inclusion of SOFA, APACHE-III, treat-all, and treat-none as decision-curve comparators is methodologically appropriate. The net-benefit interpretation (0.133 vs. 0.082 at threshold 0.20) is correctly stated, and the paper correctly distinguishes between the continuous risk score (for decision-curve comparison) and the conformal/disagreement policies (as action-governance layers).

6. **Union consensus preserves coverage guarantee.** The union consensus strategy is theoretically well-motivated: the union of individually valid prediction sets is itself a valid prediction set, and the empirical coverage of 0.982 at α = 0.05 confirms this. The paper correctly identifies the coverage–automation tradeoff and presents it as a tunable governance parameter.

7. **Shift robustness experiments exceed the field norm.** While the shift experiments have limitations (discussed below), the inclusion of missingness injection, measurement-process dropout, and care-process dropout is more thorough than most clinical ML papers. The comparison between conformal self-correction (automatic deferral increase) and fixed-threshold silent degradation is a compelling demonstration.

8. **Unusually thorough limitations section.** The ten-item limitations section is among the most honest I have seen in a clinical ML manuscript. The explicit acknowledgment that conformal guarantees are marginal (Limitation 5), that process features may not transport (Limitation 4), and that no prospective evaluation exists (Limitation 7) sets appropriate expectations.

---

## Weaknesses / Major Concerns

### W1. Incorrect claim about intersection coverage in the Discussion

The Discussion states: "This guarantee extends naturally to multi-model consensus through set union and intersection operations." This is **incorrect for intersection**. The intersection of individually valid prediction sets does not inherit the marginal coverage guarantee. If $C_1(x)$, $C_2(x)$, $C_3(x)$ each satisfy $\Pr(y \in C_k(x)) \geq 1 - \alpha$, it does not follow that $\Pr(y \in C_1(x) \cap C_2(x) \cap C_3(x)) \geq 1 - \alpha$. The Results section correctly observes that intersection coverage drops to 0.911 at α = 0.05 (below the 0.95 target), but the Discussion makes a blanket theoretical claim that contradicts this empirical finding. This sentence must be corrected. The guarantee extends through union (superset property) but not through intersection. A Bonferroni-style correction (e.g., using α/K for K models) could restore the guarantee for intersection, but this is not discussed.

**Recommendation:** Correct the Discussion sentence. Add a brief theoretical remark in Section 2.5 or 3.4 explaining why union preserves coverage but intersection does not, and note that the intersection results should be interpreted as an empirical operating point without formal coverage guarantees.

### W2. Ambiguity in the calibration–conformal data pipeline

The Methods describe a three-way split $\mathcal{D} = \mathcal{D}_{\text{train}} \cup \mathcal{D}_{\text{cal}} \cup \mathcal{D}_{\text{test}}$ (Equation 1), and separately state that isotonic calibration is "benchmarked on the held-out grouped validation split." It is unclear whether the calibration set $\mathcal{D}_{\text{cal}}$ used for conformal quantile estimation is the same partition used for fitting the isotonic calibration map. If so, there is a data-leakage concern: the conformal nonconformity scores would be computed on data that was also used to fit the calibration function, potentially producing optimistically narrow prediction sets. The exchangeability assumption underlying the conformal guarantee requires that the calibration examples are not used in any step of model fitting, including post-hoc recalibration.

**Recommendation:** Clarify the data flow explicitly. State whether (a) isotonic calibration is fitted on the validation split and conformal quantiles are estimated on a separate calibration split, or (b) the same data serves both purposes. If (b), either introduce a four-way split (train / calibration-fit / conformal-cal / test) or provide an argument for why the leakage is negligible (e.g., citing results on the stability of isotonic calibration under sample splitting).

### W3. Simulated shift uses MCAR despite rejecting MCAR

Section 2.3 states that Little's MCAR test was performed on the training partition and "rejected the null hypothesis of completely random missingness." Yet the shift robustness experiments (Section 3.6) inject missingness uniformly at random (MCAR). This is internally inconsistent: the paper establishes that the real missingness mechanism is not MCAR, then simulates shift under the very assumption it has rejected. Real-world distribution shift between institutions is far more likely to produce missing-not-at-random (MNAR) patterns—e.g., sicker patients may have more frequent lab draws (informative observation), or certain institutions may not routinely measure specific biomarkers. The MCAR simulation therefore represents a best-case scenario for shift robustness and may overstate the framework's resilience to realistic deployment-time distributional changes.

**Recommendation:** (a) Acknowledge the MCAR/MNAR inconsistency explicitly in the shift robustness discussion. (b) Consider adding at least one MNAR-style simulation (e.g., missingness correlated with feature value or outcome) as a sensitivity analysis. (c) Temper the shift robustness claims to reflect that only MCAR perturbations were tested.

### W4. The "sweet spot" α = 0.12 is selected on test-set criteria

The Results state: "Under the manuscript sweet-spot criterion—requiring both NPV ≥ 0.90 and PPV ≥ 0.55—the best operating point occurs at α = 0.12." This criterion is evaluated on test-set performance, which constitutes a form of test-set peeking. If the sweet-spot α were selected on the calibration or validation set and then evaluated on the test set, the result would be more defensible. As stated, the α = 0.12 operating point is optimized to the test data and its reported metrics are therefore optimistically biased.

**Recommendation:** Either (a) select the sweet-spot α on the validation/calibration set and report its test-set performance as a held-out evaluation, or (b) present α = 0.12 explicitly as a post-hoc descriptive finding rather than a recommended operating point, and remove the term "sweet spot" which implies a prescriptive recommendation.

### W5. Confidence intervals for conformal metrics need clarification

Table 3 reports "means ± bootstrap confidence intervals across 21 subject-grouped seeds." It is unclear whether the reported intervals are (a) the standard deviation of the 21 per-seed point estimates, (b) percentile intervals of the 21 per-seed estimates, or (c) within-seed cluster-bootstrap intervals averaged across seeds. With only 21 seeds, the sampling distribution of the mean is estimated from a small sample, and the distinction matters. Furthermore, the conformal coverage guarantee is a per-split property, not a property of the average across splits. The paper should clarify that the mean coverage of 0.952 summarizes 21 individual experiments, each of which independently satisfies (or fails to satisfy) the 1 − α guarantee.

**Recommendation:** (a) Specify the exact CI construction method (percentile bootstrap, normal approximation, etc.) and whether it accounts for the grouped structure. (b) Report the per-seed coverage range (min, max) in addition to the mean, so readers can verify that the guarantee holds split-by-split. (c) Add a sentence clarifying that the coverage guarantee is per-split, not for the average.

### W6. Repeated-measures structure in the calibration set

The paper correctly groups by subject_id for train/test splitting, but does not discuss how repeated stays from the same subject are handled within the calibration set. If a subject contributes multiple ICU stays to $\mathcal{D}_{\text{cal}}$, the resulting nonconformity scores are not independent (they share patient-level characteristics). The finite-sample coverage guarantee of conformal prediction assumes exchangeability of calibration and test points, which is weaker than independence but still requires that the calibration scores are not systematically correlated. With 1,034 repeated-subject rows, this is not a negligible concern.

**Recommendation:** Report the number of repeated-subject stays in the calibration partition. Discuss whether the within-subject correlation could affect the quantile estimation, and consider a sensitivity analysis using only one stay per subject in the calibration set.

---

## Minor Issues

1. **Notation inconsistency for $f(x_i)^{[y]}$.** The superscript notation $f(x_i)^{[y]}$ for the model's estimated probability of class $y$ is non-standard. Most conformal prediction literature uses $\hat{p}_y(x_i)$ or $f_y(x_i)$. The bracket-superscript notation could be confused with an indexing operation. Consider adopting standard notation.

2. **Empty prediction sets.** The formulation in Equation 4 admits the possibility of an empty prediction set $C(x) = \emptyset$, which would mean neither class is included. The paper does not discuss this case. While it is rare in practice for well-calibrated models, it should be acknowledged and its clinical interpretation (if any) stated.

3. **Mondrian vs. standard conformal.** The paper uses Mondrian (class-conditional) conformal prediction, which provides class-conditional marginal coverage rather than unconditional marginal coverage. The distinction is important for imbalanced classification (26.7% mortality). The paper should briefly explain why Mondrian was chosen over standard (label-unconditional) conformal, and note that Mondrian provides the stronger class-conditional guarantee.

4. **Calibration metric: ECE with 10 bins.** ECE is known to be sensitive to the number of bins and can be biased in finite samples (see Vaicenavicius et al., 2019). The paper uses 10 bins without justification. Consider reporting an additional calibration metric that is less bin-dependent (e.g., calibration error via kernel density estimation, or the Hosmer-Lemeshow test).

5. **Decision-curve analysis at a single threshold.** The headline net-benefit comparison is reported at threshold 0.20. While the paper mentions that the DCA curve spans 0.10–0.30, the choice of 0.20 as the focal threshold is not formally justified. Different clinical contexts may warrant different thresholds. Consider reporting net benefit at 2–3 thresholds or the integrated net benefit across the clinically relevant range.

6. **No formal hypothesis testing for model comparison.** The paper states that the 0.003 AUROC difference between XGBoost and LightGBM is "not clinically meaningful," but no formal test (e.g., DeLong test, paired bootstrap) is reported. While the paper's narrative does not depend on this comparison being significant, a formal test would strengthen the bounded-ceiling claim.

7. **Subgroup analysis at a single α.** Subgroup heterogeneity is reported only at α = 0.10. The choice is justified as providing "balanced operating point," but the high-SOFA coverage shortfall (0.866) might look different at α = 0.05. Reporting subgroup coverage at the primary α = 0.05 level would be informative, even if the certain-decision fractions are small.

8. **Missing discussion of conformal prediction set size distribution.** The paper reports the aggregate certain-decision fraction but does not show the distribution of prediction set sizes across the test population. A histogram or density plot of nonconformity scores by class would help readers understand where the decision boundaries fall and how much "headroom" exists between the scores and the thresholds.

9. **The term "coverage" is overloaded.** In the conformal prediction literature, "coverage" has a precise meaning (probability that the true label is in the prediction set). In clinical literature, "coverage" can mean population reach or insurance coverage. The paper should define the term explicitly at first use in the Results section, not just in the Methods.

10. **Reproducibility appendix mentions seed 42 but evaluation uses 21 seeds.** The Reproducibility section states "the random seed for all reproducible operations is 42," but the evaluation uses 21 different seeds. Clarify whether seed 42 is the base seed from which the 21 evaluation seeds are derived, or whether it refers to a different operation (e.g., imputation, feature selection).

---

## Questions for Authors

**Q1.** Is the conformal calibration set $\mathcal{D}_{\text{cal}}$ the same partition used to fit the isotonic calibration map, or is it a separate held-out set? If the same, can you provide evidence (e.g., a split-sample sensitivity analysis) that the resulting coverage is not inflated by the shared data?

**Q2.** For the intersection consensus strategy, have you considered applying a Bonferroni-type correction (e.g., using α/K for K models) to restore the formal coverage guarantee? What would the resulting automation rate and reliability metrics look like?

**Q3.** The shift robustness experiments inject MCAR missingness, but your own Little's test rejects MCAR. Have you considered simulating MNAR missingness patterns (e.g., missingness correlated with SOFA score or mortality outcome) to test whether the conformal framework's self-correcting behavior persists under more realistic shift scenarios?

**Q4.** The subgroup analysis reveals that high-SOFA patients have coverage 0.866 at α = 0.10, well below the nominal 0.90. Have you considered applying group-conditional conformal prediction (e.g., separate Mondrian thresholds per SOFA band) to restore within-group coverage guarantees? What would be the cost in terms of calibration set size per group?

**Q5.** The paper reports 21-seed repeated evaluation. What is the per-seed coverage range (minimum and maximum)? Are there individual seeds where coverage falls substantially below 1 − α, and if so, what characterizes those splits?

---

## Overall Assessment

**Recommendation: Minor Revision**

This is a well-executed clinical ML paper that makes a genuine methodological contribution by applying conformal prediction to ICU mortality triage with appropriate rigor. The core conformal formulation is correct, the subject-grouped evaluation design is sound, the bounded-ceiling narrative is refreshingly honest, and the clinical framing of the Alert/Defer/Clear trichotomy is well-motivated. The paper is substantially more careful than the median clinical ML submission in its treatment of uncertainty quantification and deployment limitations.

The major concerns are all addressable without changing the paper's conclusions: (W1) the incorrect intersection coverage claim in the Discussion is a one-sentence fix; (W2) the calibration–conformal data pipeline ambiguity requires clarification but likely does not invalidate results if a proper split was used; (W3) the MCAR shift simulation is a limitation that should be acknowledged more explicitly; (W4) the sweet-spot selection can be reframed as descriptive; (W5–W6) the CI and repeated-measures concerns require additional reporting but are unlikely to change the headline findings.

The paper would benefit from a second pass focused on tightening the theoretical claims around multi-model consensus and adding the clarifications requested above. None of the issues identified require new experiments or fundamental redesign.
