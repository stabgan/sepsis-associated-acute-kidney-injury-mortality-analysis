# Reviewer 2 — Round 2 Re-Review

**Manuscript:** "When to Alert, When to Defer: Conformal Selective Triage for ICU Mortality in Sepsis-Associated Acute Kidney Injury"

**Reviewer expertise:** ML/statistics, conformal prediction, calibration methodology

---

## Disposition of Major Concerns (W1–W6)

### W1: Incorrect intersection coverage claim in Discussion

**ADDRESSED.**

The original manuscript incorrectly implied that both union and intersection consensus preserve the marginal coverage guarantee. The revised Discussion now contains an explicit, technically correct paragraph:

> "The union of individually valid prediction sets is itself a valid prediction set. The intersection of prediction sets, by contrast, does not inherit the marginal coverage guarantee; the intersection can exclude the true label when constituent models disagree, and the empirical coverage of 0.911 at α = 0.05 (below the nominal 0.95 target) confirms this theoretical expectation. Intersection results should therefore be interpreted as empirical operating points without formal coverage guarantees."

The authors further note that a Bonferroni correction (α/K) could in principle restore the guarantee for intersection at the cost of wider prediction sets. The Results section (§3.4) echoes this, stating that the coverage shortfall "is a known theoretical limitation" and that "the formal coverage guarantee does not hold for this consensus mode." This is exactly the correction I requested.

### W2: Calibration–conformal data pipeline ambiguity

**ADDRESSED.**

The revised Methods (§2.5) now contains a dedicated paragraph that unambiguously specifies the pipeline:

> "Post-hoc calibration was fitted via three-fold cross-validation internal to D_train using CalibratedClassifierCV; the calibration map was therefore learned entirely within the training partition. The conformal calibration set D_cal was a separate held-out split carved from the training data (approximately 20% of the training partition) and was not used in any step of model fitting or recalibration."

This resolves the ambiguity. The three-fold CV is internal to D_train, D_cal is a separate held-out split, and the separation is explicitly stated to preserve the exchangeability requirement. The pipeline is now clear and methodologically sound.

### W3: MCAR shift simulation despite rejecting MCAR

**ADDRESSED.**

The revised Discussion now contains an explicit acknowledgment of this inconsistency:

> "It should be noted that the simulated missingness perturbations follow a missing-completely-at-random (MCAR) mechanism, whereas the missingness analysis conducted on the training partition rejected the MCAR null hypothesis via Little's test. Real-world distribution shift between institutions is more likely to produce missing-not-at-random (MNAR) patterns... The MCAR simulation therefore represents a favourable scenario for shift robustness, and the framework's resilience to more realistic MNAR perturbations remains to be established."

This is a candid and appropriate acknowledgment. The authors correctly characterize the MCAR simulation as a favourable-case scenario and flag MNAR robustness as an open question. This is exactly the transparency I requested.

### W4: Sweet-spot α = 0.12 and test-set peeking

**ADDRESSED.**

The revised Results (§3.3) now explicitly reframes the α = 0.12 operating point:

> "As a post-hoc descriptive observation, the operating point at α = 0.12 satisfies both NPV ≥ 0.90 and PPV ≥ 0.55... This operating point is reported as a descriptive characterisation of the coverage–automation tradeoff rather than as a prescriptive recommendation, since the criterion was evaluated on test-set performance and selecting α on test data constitutes a form of post-hoc optimisation."

The authors further state that in a prospective deployment, α should be selected on a held-out validation or calibration set. This is the correct framing: the α = 0.12 result is retained for descriptive value but is no longer presented as a recommended operating point. The test-set peeking concern is fully resolved.

### W5: CI construction unclear; per-seed min/max not reported

**ADDRESSED.**

The Reproducibility appendix (Appendix D) now specifies:

> "Confidence intervals reported in Table 3 are 2.5th and 97.5th percentile intervals computed across the 21 per-seed point estimates."

This confirms percentile-based intervals rather than normal-approximation CIs. Furthermore, per-seed extremes are now reported:

> "Across the 21 seeds at α = 0.05, per-seed coverage ranged from a minimum of 0.938 to a maximum of 0.968, with 19 of 21 seeds achieving coverage at or above the nominal 0.95 target."

The fact that 2 of 21 seeds fell below 0.95 is transparently reported. This is exactly the level of detail I requested.

### W6: Repeated measures in calibration set

**ADDRESSED.**

The revised Limitations section (item 3, "Repeated subjects") now contains an explicit discussion of this concern:

> "Furthermore, within the conformal calibration partition, repeated stays from the same subject produce nonconformity scores that are not fully independent. While the exchangeability assumption underlying conformal prediction is weaker than independence, within-subject correlation could affect quantile estimation. A sensitivity analysis restricting the calibration set to one stay per subject was not conducted but represents a useful robustness check for future work."

This is an honest acknowledgment. The authors correctly note that exchangeability is weaker than independence but that within-subject correlation could still affect quantile estimation. The suggestion of a one-stay-per-subject sensitivity analysis is appropriate. I would have preferred the sensitivity analysis itself, but acknowledging the limitation and flagging it for future work is acceptable for this submission.

---

## Disposition of Minor Issues

### Minor 1: Notation f(x)^[y] changed to \hat{p}_y(x)?

**ADDRESSED.** The nonconformity score is now defined as $s_i^{(y)} = 1 - \hat{p}_y(x_i)$, with $\hat{p}_y(x_i)$ explicitly defined as "the model's estimated probability for class y." The notation is clean and standard.

### Minor 2: Empty prediction sets discussed?

**ADDRESSED.** The revised Methods (§2.5) now includes:

> "In principle, the prediction set may also be empty (C(x) = ∅) when neither class exceeds its nonconformity threshold. In practice, this case was not observed for any test example under the calibrated models used in this study, and it is not expected to arise when the base classifier is well-calibrated."

This is a satisfactory treatment. The empty-set case is acknowledged, its non-occurrence is documented, and the connection to calibration quality is noted.

### Minor 3: Mondrian vs standard conformal explained?

**ADDRESSED.** A new paragraph in §2.5 explains:

> "Mondrian conformal prediction was selected over standard (label-unconditional) conformal prediction because it provides class-conditional marginal coverage, which is the stronger guarantee for imbalanced classification problems. With a mortality prevalence of 26.7%, standard conformal prediction could satisfy its marginal coverage guarantee by achieving high coverage on the majority class (survivors) while under-covering the minority class (deaths). Mondrian conformal prediction avoids this failure mode by computing separate nonconformity thresholds for each class."

This is a clear, technically correct justification. The class-imbalance motivation for Mondrian over standard conformal is well articulated.

### Minor 4–9: (Various presentation and clarity issues)

**ADDRESSED** across the board. The manuscript has been substantially rewritten with improved clarity throughout. The Discussion section now properly contextualizes the SOFA/APACHE-III comparison (acknowledging these scores were not used in their intended capacity). The Clear channel is explicitly defined as a model-confidence label rather than a clinical clearance for discharge. The deferred patients are noted to still receive the continuous calibrated risk score alongside the Defer label.

### Minor 10: Seed 42 vs 21 seeds clarified?

**ADDRESSED.** The Reproducibility appendix now clearly distinguishes:

> "The base random seed for preprocessing operations (imputation, feature selection, scaling) is 42. The 21 evaluation seeds used for repeated grouped benchmarking and conformal headline estimates are a fixed sequence of distinct integers (listed in the source code) that ensure independent subject-grouped splits across repetitions."

This resolves the confusion: seed 42 is for preprocessing, and 21 distinct seeds are used for evaluation. The distinction is clear.

---

## Remaining Concerns

1. The one-stay-per-subject sensitivity analysis for the calibration set (W6) was acknowledged but not conducted. This is acceptable for this submission but should be prioritized in any follow-up work.

2. The MNAR robustness gap (W3) is acknowledged but not addressed experimentally. Given the complexity of simulating realistic MNAR patterns, this is a reasonable scope boundary for a single paper.

3. Two of 21 seeds fell below the nominal 0.95 coverage target (min 0.938). While this is transparently reported and within expected finite-sample variation, it is worth noting that the per-split guarantee is not uniformly satisfied. The authors handle this appropriately by reporting it rather than hiding it.

---

## Overall Assessment

**Accept (minor editorial pass recommended).**

All six major concerns (W1–W6) have been substantively addressed. The intersection coverage claim is corrected, the calibration–conformal pipeline is clarified, the MCAR/MNAR inconsistency is acknowledged, the α = 0.12 operating point is properly reframed as post-hoc descriptive, CI construction is specified with per-seed extremes, and repeated measures in the calibration set are discussed in the limitations. All minor issues have been resolved. The manuscript is now methodologically transparent and the conformal prediction framework is presented with appropriate caveats. The paper makes a genuine contribution to uncertainty-aware clinical ML and is suitable for publication.
