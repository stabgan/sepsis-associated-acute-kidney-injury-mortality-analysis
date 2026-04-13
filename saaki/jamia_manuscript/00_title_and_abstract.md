# Title Page and Abstract

## Title

When to Alert, When to Defer: Conformal Selective Triage for ICU Mortality in Sepsis-Associated AKI

## Running Title

Conformal selective triage for SA-AKI mortality

## Structured Abstract

### Objective

To develop and evaluate an uncertainty-aware triage framework for early in-hospital mortality prediction in sepsis-associated acute kidney injury (SA-AKI), with primary emphasis on safe automation boundaries rather than raw discrimination alone.

### Materials and Methods

We analyzed `10036` ICU-stay-level SA-AKI cohort records from `mimic_saaki_raw_v2.csv` scored at 24 hours after ICU admission, with event prevalence `0.267`. All deployment claims used subject-grouped train/validation/test splits by `subject_id`. We benchmarked logistic regression, LightGBM, XGBoost, and score-only baselines; calibrated the deployable models; evaluated a disagreement-based selective-triage baseline; and made Mondrian conformal selective triage the main method. We summarized coverage, alert positive predictive value (PPV), clear negative predictive value (NPV), calibration, subgroup heterogeneity, and robustness under simulated distribution shift.

### Results

The best conventional discriminative baseline was `lightgbm` with `sigmoid` calibration. The prediction ceiling remained limited: even when `time_to_event_hrs` was added as a leakage-like feature, AUROC reached only `0.818`. In repeated subject-grouped evaluation, single-model conformal triage achieved approximately `0.954` coverage at `alpha=0.05` while retaining a clinically meaningful defer region. The multi-model union consensus at `alpha=0.05` concentrated decisions into a smaller, more reliable actionable region with alert PPV `0.735` and clear NPV `0.953`. Under added missingness shift, conformal coverage remained stable while fixed thresholds lost recall.

### Discussion

The main contribution is not a new AUROC leader; it is a deployment-oriented Alert/Defer/Clear framework that quantifies when the model should defer. The disagreement-based policy remains informative as an ablation, but conformal selective triage is more rigorous because it supplies finite-sample coverage guarantees and more graceful degradation under shift.

### Conclusion

For this SA-AKI cohort, the main scientific opportunity lies in safe decision-making under bounded predictive signal. Conformal selective triage offers a realistic uncertainty-aware deployment framing for ICU mortality prediction and is a strong fit for a clinical informatics journal such as `JAMIA`.
