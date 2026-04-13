# Title Page and Abstract

## Title

When to Alert, When to Defer: Conformal Selective Triage for ICU Mortality in Sepsis-Associated AKI

## Running Title

Conformal selective triage for SA-AKI mortality

## Structured Abstract

### Objective

To develop and evaluate an uncertainty-aware triage framework for early in-hospital mortality prediction in sepsis-associated acute kidney injury (SA-AKI), with primary emphasis on safe automation boundaries rather than raw discrimination alone.

### Materials and Methods

We analyzed `10036` ICU-stay-level SA-AKI cohort records from `mimic_saaki_raw_v2.csv` scored at 24 hours after ICU admission, with event prevalence `0.267`. All deployment claims used subject-grouped train/validation/test splits by `subject_id`. Conventional baselines included logistic regression, XGBoost, a grouped-Optuna-tuned LightGBM benchmark (`40` trials), an exploratory CatBoost diversity comparator, and score-only baselines. Calibration was selected on grouped validation data. We evaluated disagreement-based selective triage as an ablation and Mondrian conformal selective triage as the main method. We summarized coverage, alert positive predictive value (PPV), clear negative predictive value (NPV), calibration, decision-curve utility, subgroup heterogeneity, and robustness under simulated distribution shift, with conformal headline estimates repeated across `21` grouped seeds.

### Results

The best conventional discriminative baseline was `xgboost` with `isotonic` calibration (mean AUROC `0.768`; recall at `PPV >= 0.50` `0.608`). Grouped Optuna improved LightGBM validation recall at `PPV >= 0.50` to `0.590`, but repeated grouped benchmarking still preserved only a narrow tree-model performance band. The prediction ceiling remained limited: even when `time_to_event_hrs` was added as a leakage-like feature, AUROC reached only `0.810`. In repeated subject-grouped evaluation across `21` seeds, single-model conformal triage achieved approximately `0.952` coverage at `alpha=0.05` while retaining a clinically meaningful defer region. The multi-model union consensus at `alpha=0.05` concentrated decisions into a smaller, more reliable actionable region with alert PPV `0.729` and clear NPV `0.962`. At a decision threshold of `0.20`, the selected continuous model achieved net benefit `0.133` versus `0.082` for `APACHE-III` and `0.082` for `SOFA`. Under added missingness shift, conformal coverage remained stable while fixed thresholds lost recall.

### Discussion

The main contribution is not a new AUROC leader; it is a deployment-oriented Alert/Defer/Clear framework that quantifies when the model should defer. The disagreement-based policy remains informative as an ablation, but conformal selective triage is more rigorous because it supplies finite-sample coverage guarantees and more graceful degradation under shift.

### Conclusion

For this SA-AKI cohort, the main scientific opportunity lies in safe decision-making under bounded predictive signal. Conformal selective triage offers a realistic uncertainty-aware deployment framing for ICU mortality prediction and is a strong fit for a clinical informatics journal such as `JAMIA`.
