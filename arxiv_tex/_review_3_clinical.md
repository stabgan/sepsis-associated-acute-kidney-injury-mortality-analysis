# Reviewer 3 — Clinical Review

**Manuscript:** "When to Alert, When to Defer: Conformal Selective Triage for ICU Mortality in Sepsis-Associated Acute Kidney Injury"

**Reviewer expertise:** Nephrology and critical care medicine; daily clinical management of SA-AKI patients in a tertiary ICU.

---

## Summary

This paper presents a retrospective analysis of 10,036 SA-AKI ICU stays from MIMIC-IV v3.1, developing a conformal prediction–based triage framework that classifies patients into Alert (high mortality risk), Clear (lower risk), or Defer (insufficient model confidence) categories using 162 features from the first 24 hours of ICU admission. The authors argue—correctly, in my view—that the discrimination ceiling for this prediction task is bounded (AUROC ~0.77) and that the more important contribution is a principled uncertainty-quantification layer that tells clinicians *when* to trust the model. The best conventional model (XGBoost, isotonic calibration) achieved AUROC 0.768, substantially outperforming SOFA (0.646) and APACHE-III (0.579). At α=0.05, single-model conformal triage achieved coverage 0.952, alert PPV 0.643, and clear NPV 0.948, with multi-model union consensus pushing alert PPV to 0.729 and clear NPV to 0.962 at the cost of deferring 83.6% of patients.

From a clinical standpoint, this is a thoughtful informatics study that takes the right philosophical approach—acknowledging what the model cannot do rather than overselling what it can. However, several aspects of the clinical framing, cohort definition, and proposed workflow integration require scrutiny before this framework could be considered for real-world pilot testing.

---

## Strengths

1. **Intellectual honesty about the prediction ceiling.** The authors explicitly document that AUROC ~0.77 is a hard ceiling for first-24-hour SA-AKI mortality prediction and resist the temptation to claim otherwise. The leak-augmented ceiling experiment (AUROC 0.810 with time-to-event as a feature) is a clever and transparent way to quantify the information locked behind the prediction window. This kind of honesty is rare in clinical ML papers and is clinically valuable—it sets realistic expectations for what any model can deliver in this space.

2. **The Alert/Defer/Clear trichotomy is clinically intuitive.** As an intensivist, I find the three-way output far more useful than a raw probability score. ICU nurses and residents do not act on "your patient has a 37% mortality risk"—they act on "this patient needs escalation" or "this patient is stable." The explicit Defer channel is particularly important: it acknowledges that for many patients, the model simply does not know, and that is a safer message than a falsely confident probability.

3. **Rigorous subject-grouped evaluation.** The recognition that 1,034 repeated ICU stays from the same patients require subject-level splitting is methodologically sound and clinically important. Many MIMIC-based studies ignore this, leading to inflated performance estimates. The 21-seed repeated evaluation adds further credibility.

4. **Decision-curve analysis with clinically relevant comparators.** Comparing against SOFA and APACHE-III on a decision-curve framework is exactly the right approach. These are the scores we actually use at the bedside. The net benefit of 0.133 vs. 0.082 for APACHE-III at threshold 0.20 is a clinically meaningful difference—roughly 5 additional correctly identified deaths per 100 patients screened.

5. **Shift robustness analysis is deployment-relevant.** The simulated missingness experiments address a real concern: models trained at one institution often encounter different documentation practices elsewhere. The finding that conformal coverage remains stable while fixed thresholds silently lose recall is an important safety argument. In my experience, silent degradation is the most dangerous failure mode for clinical decision support tools.

6. **Transparent reporting of subgroup heterogeneity.** The honest disclosure that high-SOFA patients achieve only 0.866 coverage (below the 0.90 target) with clear NPV of 0.842 is clinically critical. These are precisely the patients where we most need help, and the paper correctly identifies this as a deployment governance issue rather than hiding it.

7. **Comprehensive limitations section.** The ten enumerated limitations are unusually thorough and self-aware. The explicit statement that no bedside-readiness claim is made, and that prospective clinician-in-the-loop evaluation is necessary, reflects appropriate scientific caution.

8. **Negative and null probes reported transparently.** Appendix Table B12 reporting that stacking ensembles, ratio features, missingness indicators, and HPO-tuned LightGBM all failed to break the ceiling is valuable. This saves future researchers from repeating dead-end experiments.

---

## Weaknesses / Major Concerns

### 1. The 24-hour prediction window may be too late for the highest-impact clinical decisions

The T0–T24 feature window is a reasonable engineering choice, but from a clinical workflow perspective, many of the most impactful decisions in SA-AKI management occur *within* the first 24 hours, not after. Decisions about early goal-directed fluid resuscitation, vasopressor initiation, timing of RRT consultation, and initial antibiotic adequacy are typically made in the first 6–12 hours. By T24, many patients have already declared themselves—the sickest are on vasopressors and ventilators, and the trajectory is often apparent to experienced clinicians. A 6-hour or 12-hour prediction window, while yielding lower discrimination, might be more clinically actionable because it would inform decisions that have not yet been made. The authors should discuss why T24 was chosen over earlier windows and whether a staged prediction approach (T6 → T12 → T24) might better serve clinical workflows.

### 2. Exclusion of early RRT patients introduces survivorship bias

Excluding patients who received RRT within the first 24 hours is understandable from a feature-engineering perspective (RRT alters downstream lab values), but it removes some of the sickest SA-AKI patients from the cohort—precisely the population where mortality prediction is most needed. These patients have Stage 3 AKI with clinical urgency sufficient to warrant emergent dialysis. Their exclusion means the model is trained and evaluated on a somewhat less severe SA-AKI population, and the reported mortality rate of 26.7% may underestimate the true SA-AKI mortality burden. The authors should quantify how many patients were excluded by this criterion and report their mortality rate. If this is a substantial fraction, the model's applicability to the full SA-AKI spectrum is limited.

### 3. The 26.7% mortality rate needs more epidemiological context

The reported 26.7% in-hospital mortality is at the lower end of published SA-AKI mortality ranges (25–45%, as the authors cite). This is plausible for a MIMIC-IV cohort that excludes ESRD and early RRT patients, but the authors should explicitly discuss how their cohort severity compares to published SA-AKI registries. Key questions: What is the distribution of KDIGO stages (1 vs. 2 vs. 3)? What fraction of patients had septic shock (SOFA cardiovascular ≥3) vs. sepsis without shock? What is the median SOFA score? Without this granularity, it is difficult to assess whether the 26.7% rate reflects a representative SA-AKI population or a selected lower-acuity subset.

### 4. The Alert/Defer/Clear framework needs workflow integration specificity

The paper describes the trichotomy in abstract terms but does not specify *what clinical actions* each category should trigger. In a real ICU:
- **Alert:** Does this trigger an automatic nephrology consult? A palliative care referral? A change in monitoring frequency? An ICU attending notification?
- **Clear:** Does this mean the patient can be considered for step-down? That routine monitoring is sufficient? That RRT can be deferred?
- **Defer:** What does the clinician actually *do* with a deferred patient that they would not already be doing?

At α=0.05, 68.2% of patients are deferred. In a 20-bed ICU with SA-AKI prevalence of ~30%, that means roughly 4 of 6 SA-AKI patients get a "we don't know" recommendation. This is honest, but it raises the question of whether the tool provides enough actionable signal to justify the implementation cost. The sweet-spot at α=0.12 (53.2% certain decisions) is more practical, but the paper does not adequately discuss which α level would be recommended for different clinical settings.

### 5. Clinical severity score comparisons may be unfair

SOFA and APACHE-III were evaluated as single-feature logistic regression baselines. This is a valid statistical comparison, but it is not how these scores are used clinically. SOFA is a *monitoring* tool designed to track organ dysfunction trajectories over time, not a one-shot mortality predictor. APACHE-III is an *admission* severity score designed for case-mix adjustment and benchmarking, not for individual patient triage. Comparing a 162-feature ML model against these scores used outside their intended purpose risks creating a straw-man comparison. The authors should acknowledge this distinction and ideally compare against purpose-built ICU mortality prediction scores (e.g., APACHE-IV predicted mortality, SAPS-3, or the MIMIC-derived OASIS score).

### 6. Important clinical variables appear to be missing from the feature set

The 162-feature set is comprehensive for structured EHR data, but several clinically important variables for SA-AKI prognosis are not mentioned:
- **Fluid balance trajectory:** Cumulative fluid balance is mentioned, but net fluid balance (intake minus output) and its trajectory are among the strongest predictors of AKI outcomes. Is this captured as a time-series feature with the 9 aggregations?
- **Antibiotic appropriateness:** Whether the initial empiric antibiotic regimen was appropriate for the cultured organism is a major determinant of sepsis outcomes. This is admittedly difficult to extract from structured data at T24.
- **Source of infection:** Urinary, pulmonary, abdominal, and bloodstream infections carry different SA-AKI prognoses. Is infection source captured?
- **Pre-admission functional status/frailty:** Baseline functional status is one of the strongest predictors of ICU mortality in elderly patients. Is this captured through Charlson comorbidity components, or is there a separate frailty measure?
- **Lactate clearance:** The rate of lactate decline in the first 6–12 hours is a well-established prognostic marker in sepsis. Is this captured by the delta and slope aggregations of lactate?

### 7. The conformal guarantee is marginal, not conditional—and this matters clinically

The authors acknowledge this in the limitations, but it deserves more prominence. The coverage guarantee holds *on average* across the test population, not for any specific patient subgroup. The subgroup analysis shows this concretely: high-SOFA patients get only 0.866 coverage at α=0.10. In clinical practice, the patients for whom we most need reliable predictions are precisely the high-acuity patients where coverage is worst. A clinician cannot know whether a given patient's prediction falls within the guaranteed coverage region. This is a fundamental limitation that should be discussed more prominently in the main text, not just in the limitations section.

### 8. No comparison with nephrology/intensivist clinical judgment

The paper compares the model against severity scores but not against clinician prediction. Studies have shown that experienced intensivists can predict ICU mortality with AUROC ~0.85 using gestalt clinical judgment. Without a clinician-prediction comparator (even a retrospective one using documented prognosis notes), it is impossible to know whether the model adds value beyond what an experienced clinician already knows at T24. This is particularly relevant given that 68% of patients are deferred to clinician judgment anyway.

---

## Minor Issues

1. **Table 1 (cohort summary) is referenced but the actual demographic breakdown is not shown in the text.** The paper would benefit from reporting median age, sex distribution, racial/ethnic composition, and KDIGO stage distribution in the main text, not just as a table reference. Readers need this to assess generalizability.

2. **The term "Clear" may be clinically misleading.** In ICU parlance, "clearing" a patient implies they are safe for discharge or de-escalation. With a clear NPV of 0.948 (single-model) or 0.962 (union), 4–5% of "cleared" patients will die. The terminology should be softened—perhaps "Lower Risk" rather than "Clear"—to avoid false reassurance.

3. **The paper does not discuss AKI recovery as an outcome.** For nephrologists, renal recovery (return to baseline creatinine, dialysis independence) is as important as mortality. A model that predicts mortality but ignores renal recovery trajectory misses half the clinical picture for SA-AKI patients.

4. **The ±48-hour SA-AKI temporal linkage window is broad.** Some SA-AKI definitions use a tighter ±24-hour window. The authors should justify the 48-hour window and discuss whether sensitivity analyses with a tighter window were performed.

5. **Calibration slope of 1.035 for XGBoost is reported as "near 1.0," but this still represents slight underconfidence** in the upper risk range. For a triage tool, even small calibration errors in the high-risk tail can affect Alert classification. The clinical impact of this residual miscalibration should be discussed.

6. **The paper does not discuss alert fatigue.** At α=0.05, the alert rate is 10.7%. In a busy ICU, even a 10% alert rate can contribute to alarm fatigue if alerts are not actionable. The paper should discuss how the Alert channel would be integrated with existing alarm systems.

7. **No SHAP or feature importance analysis in the main text.** Clinicians want to know *why* a patient was flagged. The paper focuses on the conformal framework but does not discuss explainability. Even a brief SHAP summary plot would strengthen clinical credibility.

8. **The 48-hour mortality endpoint (prevalence 0.004) is clinically meaningless** and should be removed or relegated to a footnote. Predicting death within 48 hours of ICU admission for SA-AKI patients is not a useful clinical task—these are patients who are either moribund on arrival or experience catastrophic complications, and no prediction model will change their management.

9. **The paper uses "in-hospital mortality" as the outcome but does not discuss discharge disposition.** Patients discharged to hospice or comfort care are censored as survivors, which may undercount true mortality. Was discharge to hospice tracked?

10. **The bibliography appears to cite Chen et al. 2025, Qu et al. 2024, Li et al. 2023, and others in the Discussion, but these are not visible in the main reference list.** The authors should ensure all cited works are included in refs.bib and that the comparison with prior SA-AKI studies is accurate and fair.

---

## Questions for Authors

1. **How many patients were excluded by the early-RRT criterion, and what was their mortality rate?** If this is a substantial and high-mortality subgroup, the model's clinical applicability is narrower than implied. Would you consider a sensitivity analysis that includes these patients (using pre-RRT features only)?

2. **Have you considered a dynamic prediction approach with updating predictions at T6, T12, and T24?** The clinical utility of a T24 prediction is limited by the fact that many management decisions have already been made. A staged approach could provide earlier actionable signals, even if individual time-point discrimination is lower.

3. **What is the distribution of KDIGO stages in your cohort, and does conformal triage performance vary by AKI stage?** Stage 3 AKI patients have fundamentally different trajectories than Stage 1 patients, and a single model may not serve both populations well. This is a clinically critical stratification that is absent from the subgroup analysis.

4. **How would you propose handling the high-SOFA coverage gap in a real deployment?** The 0.866 coverage for high-SOFA patients at α=0.10 means the model fails its guarantee for the sickest patients. Would you recommend SOFA-stratified α thresholds, mandatory clinician override for high-SOFA patients, or a different approach?

5. **Can you provide the number needed to screen (NNS) for the Alert channel at different α levels?** Clinicians think in terms of "how many patients do I need to review to find one true positive?" rather than PPV. Translating alert PPV into NNS would make the clinical value proposition more concrete.

---

## Overall Assessment

**Recommendation: Minor Revision**

This is a well-conceived and unusually honest clinical informatics study that makes a genuine methodological contribution through the conformal selective triage framework. The philosophical stance—that bounded discrimination should lead to uncertainty-aware decision governance rather than incremental AUROC chasing—is correct and important. The evaluation is rigorous, the limitations are transparently reported, and the paper avoids the overclaiming that plagues much of the clinical ML literature.

However, the clinical framing needs strengthening. The 24-hour prediction window should be explicitly justified against earlier alternatives. The exclusion of early-RRT patients and its impact on cohort representativeness must be quantified. The Alert/Defer/Clear framework needs concrete workflow integration examples. The severity score comparisons should acknowledge that SOFA and APACHE-III are being used outside their designed purpose. The subgroup analysis should include KDIGO stage stratification. And the high-SOFA coverage gap deserves more prominent discussion as a deployment-critical finding.

These are addressable issues that do not undermine the core contribution. With revisions to strengthen the clinical grounding, this paper would make a valuable addition to the SA-AKI and clinical decision support literature. The conformal triage framework is the right direction for the field, and the bounded-ceiling finding is an important empirical contribution that should temper expectations across the SA-AKI prediction community.
