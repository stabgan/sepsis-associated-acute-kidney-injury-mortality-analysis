# Reviewer 1 — Round 2 Re-Review

**Manuscript:** "When to Alert, When to Defer: Conformal Selective Triage for ICU Mortality in Sepsis-Associated Acute Kidney Injury"

**Reviewer expertise:** Clinical informatics, ICU decision-support systems, fairness in health ML

**Date:** Round 2 re-review

---

## Summary of Round 1 Concerns and Disposition

The revised manuscript is substantially improved. The authors have addressed the majority of my major concerns with care and intellectual honesty. The paper now reads as a mature, self-aware clinical informatics contribution rather than a model-performance showcase. Below I evaluate each Round 1 concern individually.

---

## Major Concerns (W1–W8)

### W1: "Hard ceiling" overstated → Reframed as empirical ceiling under current constraints?

**ADDRESSED.**

The authors have systematically replaced the "hard ceiling" framing with "empirical ceiling under the current feature and model constraints." This language appears in the Figure 2 caption, the Results narrative (Section 3.2), and the Discussion opening paragraph. The Conclusion now states the ceiling is "bounded at approximately AUROC 0.77 under leak-free subject-grouped evaluation." The Discussion further qualifies the ceiling by enumerating three mechanistic explanations (stochastic treatment response, unmeasured confounders including goals-of-care, and bounded structured-EHR features), making clear this is a property of the current prediction problem rather than a universal impossibility claim. This is exactly the reframing I requested.

### W2: Missing race/ethnicity subgroup → Was ethnicity removed from Methods fairness language?

**ADDRESSED.**

Race/ethnicity is no longer included as a subgroup analysis axis. The Methods (Section 2.7, Subgroup analysis paragraph) now lists five stratification axes: "sex, age band, SOFA severity band, mechanical ventilation status, and vasopressor use." The fairness language in the Discussion references "gender parity" with the Δ ≤ 0.05 bound applied to AUROC parity across gender only. Ethnicity appears only in the Discussion's critique of Hu et al. (2021), where the authors correctly note that "Ethnicity was included as a predictor, raising fairness and transportability concerns that were not addressed" — an appropriate use. The Limitations section (item 5) mentions "specific ethnic groups" as an example of subgroups where conditional coverage is not guaranteed, which is honest rather than promissory. The mismatch between claiming ethnicity-based fairness audits and not reporting them has been resolved.

### W3: 68% defer rate practical utility → Was the continuous risk score for deferred patients discussed?

**ADDRESSED.**

This was one of my most important concerns, and the revision handles it well. Section 3.3 now contains an explicit paragraph: "It is important to note that deferred patients are not left without information. In the proposed deployment framework, the continuous calibrated risk score is displayed alongside the Defer label, providing the clinician with a probability estimate that can inform, but not automate, the triage decision. The conformal framework thus functions as a risk score with an explicit confidence flag rather than a pure three-way classifier, and the Defer channel serves as a structured mechanism for communicating model uncertainty to the end user." This directly addresses the clinical utility concern. The Discussion reinforces this by framing the Defer channel as mandating "clinician review without model-driven recommendation" while the continuous score remains available. The 68% defer rate is now contextualized as a feature of the conservative α = 0.05 operating point rather than a system-level limitation.

### W4: UQ comparison — no discussion of alternative uncertainty quantification methods (MC dropout, deep ensembles, Bayesian approaches)

**NOT ADDRESSED.**

The manuscript still does not discuss alternative uncertainty quantification approaches. There is no mention of MC dropout, deep ensembles, evidential deep learning, or Bayesian neural networks as comparators or even as related work. The justification for choosing conformal prediction over these alternatives is implicit (distribution-free guarantees, model-agnostic) but never made explicit in a comparative framing. For a clinical informatics audience, a brief paragraph in the Discussion acknowledging these alternatives and explaining why conformal prediction was preferred (finite-sample guarantees without distributional assumptions, compatibility with any base classifier, no retraining required) would strengthen the methodological positioning. This remains a gap, though I acknowledge it is a minor one given the paper's scope.

### W5: SHAP or feature-importance analysis absent

**NOT ADDRESSED.**

No SHAP, LIME, or any feature-importance analysis is included. The manuscript focuses entirely on the decision-framework contribution and does not attempt to explain which features drive individual predictions. While I understand the authors' framing — that the contribution is the triage framework rather than a new predictive model — the absence of any explainability analysis limits the clinical interpretability of the Alert and Clear decisions. A clinician receiving an Alert would reasonably want to know which features contributed most to that classification. The feature ablation in Appendix B4 (physiology-severity vs. full model) provides some coarse-grained insight, but individual-level explanations are absent. I now consider this a minor rather than major gap, given the paper's stated scope, but it should be acknowledged as a limitation or future work item.

### W6: Disagreement baseline insufficiently contextualized

**ADDRESSED.**

The disagreement baseline is now consistently labeled as an "ablation" throughout the manuscript (Abstract, Introduction, Methods Section 2.6, Results Section 3.5, Discussion). The Results section explicitly states it "remains an important ablation because it demonstrates that explicit deferral helps even before formal conformalization." The Discussion provides a thorough comparative analysis explaining why conformal prediction is methodologically stronger (finite-sample guarantees vs. heuristic thresholds, natural extension to multi-model consensus, no separate threshold recalibration). The Limitations section (item 8) further clarifies that the disagreement baseline "is included as a methodological comparison rather than as the recommended deployment strategy." This framing is appropriate and clear.

### W7: Calibration depth — reliability diagrams, calibration slope/intercept reporting

**ADDRESSED.**

The revision now reports calibration slope and calibration intercept alongside ECE and Brier score in the benchmark table (Table 2). The Methods explicitly list "reliability diagrams, calibration intercept, and calibration slope" as calibration assessment tools. The Results provide clinically grounded interpretation: "A calibration slope of 0.843 for APACHE-III indicates systematic overconfidence in its probability estimates, which would translate directly into miscalibrated triage decisions." The main text includes a calibration panel (Figure 7, left panel), and Appendix Figure B2 provides the full calibration curve for the deployed XGBoost baseline. The calibration story is now thorough and well-integrated with the clinical narrative.

### W8: Intersection consensus coverage shortfall not adequately flagged

**ADDRESSED.**

The revision now explicitly flags the intersection coverage shortfall in multiple locations. Section 3.4 states: "The coverage of 0.911 falls below the nominal 0.95 target, which is expected: the intersection of individually valid prediction sets is not guaranteed to maintain marginal coverage." The Discussion adds: "Intersection results should therefore be interpreted as empirical operating points without formal coverage guarantees." A Bonferroni correction is mentioned as a theoretical fix. The Results further warn: "institutions selecting the intersection strategy should be aware that the formal coverage guarantee does not hold for this consensus mode." This is transparent and appropriately cautious.

---

## Minor Concerns (Minor1–Minor10)

### Minor1: Planned vs. reported subgroups mismatch

**ADDRESSED.** The Methods now list exactly five subgroup axes (sex, age band, SOFA severity band, mechanical ventilation status, vasopressor use), and the Results report on exactly these five. No ethnicity subgroup is promised or reported. The mismatch is resolved.

### Minor2: Sweet-spot criterion justified → Reframed as post-hoc?

**ADDRESSED.** The α = 0.12 operating point is now explicitly introduced as "a post-hoc descriptive observation" and the text states it "is reported as a descriptive characterisation of the coverage–automation tradeoff rather than as a prescriptive recommendation, since the criterion was evaluated on test-set performance and selecting α on test data constitutes a form of post-hoc optimisation." The text further recommends that in prospective deployment, α should be selected on a held-out validation or calibration set. This is exactly the reframing needed.

### Minor3: "Honest" loaded language → Replaced with "leak-free"?

**ADDRESSED.** A search for "honest" returns zero matches across all manuscript files. The terminology has been consistently replaced with "leak-free" (appearing in the Figure 2 caption, Results Section 3.2, Discussion, and Conclusion) and "leak-prone" for the ceiling experiment comparator. This is more precise and less value-laden.

### Minor4: Mondrian vs. standard conformal — justification missing

**ADDRESSED.** Section 2.5 now contains an explicit justification paragraph: "Mondrian conformal prediction was selected over standard (label-unconditional) conformal prediction because it provides class-conditional marginal coverage, which is the stronger guarantee for imbalanced classification problems. With a mortality prevalence of 26.7%, standard conformal prediction could satisfy its marginal coverage guarantee by achieving high coverage on the majority class (survivors) while under-covering the minority class (deaths)." This is clear and well-motivated.

### Minor5: Calibration/conformal partition separation unclear

**ADDRESSED.** Section 2.5 now contains a detailed paragraph explaining the three-way data split: "Post-hoc calibration was fitted via three-fold cross-validation internal to D_train... The conformal calibration set D_cal was a separate held-out split carved from the training data (approximately 20% of the training partition) and was not used in any step of model fitting or recalibration." The exchangeability requirement is explicitly linked to this separation. This resolves the ambiguity.

### Minor6: SOFA/APACHE-III comparison framing — unfair to clinical scores

**ADDRESSED.** The Results now include: "These scores were evaluated as single-feature logistic regression baselines under the same grouped protocol, which does not reflect their intended clinical use. SOFA was designed for longitudinal organ-dysfunction monitoring, and APACHE-III for admission-level case-mix adjustment. The comparison is intended to establish a discrimination floor relative to routinely available clinical information rather than to evaluate these instruments in their designed capacity." The Discussion repeats this caveat. This is fair and transparent.

### Minor7: MCAR simulation vs. real-world MNAR shift

**ADDRESSED.** The Discussion now contains an explicit caveat: "It should be noted that the simulated missingness perturbations follow a missing-completely-at-random (MCAR) mechanism, whereas the missingness analysis conducted on the training partition rejected the MCAR null hypothesis via Little's test. Real-world distribution shift between institutions is more likely to produce missing-not-at-random (MNAR) patterns... The MCAR simulation therefore represents a favourable scenario for shift robustness, and the framework's resilience to more realistic MNAR perturbations remains to be established." This is an important qualification that was missing in Round 1.

### Minor8: Repeated-subject exchangeability concern for conformal guarantees

**ADDRESSED.** Limitations item 3 now explicitly discusses this: "Within the conformal calibration partition, repeated stays from the same subject produce nonconformity scores that are not fully independent. While the exchangeability assumption underlying conformal prediction is weaker than independence, within-subject correlation could affect quantile estimation. A sensitivity analysis restricting the calibration set to one stay per subject was not conducted but represents a useful robustness check for future work." This is honest and appropriately scoped.

### Minor9: Net benefit interpretation — absolute numbers needed

**ADDRESSED.** The Results now provide concrete clinical interpretation: "the net benefit of 0.133 means that using the XGBoost model at this threshold is equivalent to correctly identifying approximately 13.3 additional true-positive deaths per 100 patients compared with a treat-none strategy" and "the 0.051-point net-benefit advantage over APACHE-III (0.133 vs. 0.082) represents... for every 100 SA-AKI patients screened, the ML model would correctly escalate approximately 5 additional patients who will die." These per-100-patient translations make the decision-curve results clinically interpretable.

### Minor10: Clear channel terminology — potential for clinical misinterpretation

**ADDRESSED.** The Discussion now includes a parenthetical clarification: "The Clear channel (used here as a model-confidence label indicating lower predicted risk, not as a clinical clearance for discharge or de-escalation)." This prevents the dangerous misinterpretation that a "Clear" label constitutes clinical clearance.

---

## New Observations (Round 2)

1. The literature comparison section in the Discussion (Chen et al., Li et al., Hu et al., Roknaldin et al.) is thorough and well-argued, but borders on aggressive. The critique of each prior study's methodology is detailed and specific. While I believe the critiques are technically accurate, the tone could be softened slightly to avoid the appearance of disparaging competitors rather than contextualizing results. This is a stylistic suggestion, not a required revision.

2. The class-imbalance discussion (inverse class weighting vs. SMOTE/ADASYN) is a welcome addition that strengthens the methodological narrative.

3. The T24 window justification paragraph in the Discussion is excellent and addresses a question that clinicians will inevitably raise.

4. The absence of SHAP/feature-importance analysis (W5) and alternative UQ comparison (W4) remain gaps but are now minor in the context of the paper's stated contribution. I recommend adding a single sentence to the Limitations section acknowledging that individual-level feature explanations were not computed and that alternative UQ methods were not compared.

---

## Revised Overall Assessment

| Criterion | Round 1 | Round 2 |
|---|---|---|
| Major concerns addressed | — | 6 of 8 fully addressed; 2 not addressed (W4, W5) |
| Minor concerns addressed | — | 10 of 10 addressed |
| Methodological rigor | Good | Strong |
| Clinical framing | Adequate | Strong |
| Transparency/limitations | Adequate | Excellent |
| Novelty | Moderate | Moderate (unchanged) |

The two unaddressed major concerns (W4: alternative UQ comparison; W5: SHAP/explainability) are real gaps but are now minor in the context of the revised manuscript's scope and stated contribution. Neither requires new experiments — a brief Discussion paragraph for each would suffice.

**Recommendation: Minor Revision**

Specifically:
1. Add 2–3 sentences to the Discussion or Limitations acknowledging that alternative UQ methods (MC dropout, deep ensembles, Bayesian approaches) were not compared and briefly stating why conformal prediction was preferred.
2. Add 1–2 sentences to the Limitations acknowledging the absence of individual-level feature explanations (SHAP/LIME) and identifying this as future work.

These are text-only additions that do not require new experiments or analyses. Once addressed, the manuscript would be suitable for acceptance.
