# Appendix B: Extended Benchmarks

## Grouped Tuning And Diversity Benchmarks

**Appendix Table B1. Repeated grouped benchmark summary for the main tree-family comparison.**

| model_name | auroc__mean | brier__mean | ece__mean | recall_at_ppv_050__mean |
| --- | --- | --- | --- | --- |
| catboost | 0.768 | 0.161 | 0.022 | 0.623 |
| lightgbm | 0.765 | 0.161 | 0.021 | 0.575 |
| xgboost | 0.768 | 0.161 | 0.017 | 0.608 |

**Appendix Table B2. Grouped Optuna summary for the LightGBM benchmark.**

| model_name | n_trials | validation_auroc | validation_recall_at_ppv_050 | validation_brier | validation_ece |
| --- | --- | --- | --- | --- | --- |
| lightgbm | 40 | 0.752 | 0.590 | 0.177 | 0.093 |

**Appendix Table B3. Top grouped Optuna LightGBM trials.**

| model_name | trial_number | objective_value | auroc | brier | recall_at_ppv_050 | param_n_estimators | param_learning_rate | param_num_leaves | param_max_depth | param_min_child_samples |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lightgbm | 17 | 0.882 | 0.752 | 0.177 | 0.590 | 603 | 0.132 | 52 | 10 | 61 |
| lightgbm | 34 | 0.881 | 0.753 | 0.173 | 0.580 | 627 | 0.095 | 60 | 8 | 24 |
| lightgbm | 33 | 0.878 | 0.753 | 0.176 | 0.569 | 605 | 0.085 | 60 | 9 | 50 |
| lightgbm | 29 | 0.874 | 0.754 | 0.175 | 0.552 | 814 | 0.109 | 60 | 8 | 25 |
| lightgbm | 24 | 0.872 | 0.748 | 0.179 | 0.569 | 607 | 0.196 | 50 | 10 | 10 |

The grouped Optuna search improved the LightGBM validation operating point (`recall@PPV0.50 = 0.590`), but repeated grouped benchmarking still left XGBoost, tuned LightGBM, and CatBoost in a narrow performance band. CatBoost therefore remains an informative diversity check rather than a reason to rewrite the main conformal benchmark story.

## Feature Ablation

**Appendix Table B4. Feature ablation of the selected model.**

| feature_set | model_name | auroc | auprc | brier | ece | precision | recall | f1 | specificity | alert_rate | prevalence | tp | fp | tn | fn | calibration_intercept | calibration_slope | recall_at_ppv_050 | precision_at_ppv_050_threshold | threshold_ppv_050 | low_risk_coverage_npv_095 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| physiology_severity | xgboost | 0.756 | 0.539 | 0.163 | 0.011 | 0.490 | 0.531 | 0.510 | 0.801 | 0.287 | 0.265 | 282.000 | 293.000 | 1177.000 | 249.000 | -0.058 | 0.986 | 0.552 | 0.500 | 0.365 | 0.173 |
| full | xgboost | 0.763 | 0.546 | 0.161 | 0.016 | 0.493 | 0.550 | 0.520 | 0.796 | 0.296 | 0.265 | 292.000 | 300.000 | 1170.000 | 239.000 | -0.032 | 1.009 | 0.548 | 0.500 | 0.356 | 0.167 |
| physiology_plus_care | xgboost | 0.756 | 0.536 | 0.163 | 0.025 | 0.499 | 0.501 | 0.500 | 0.818 | 0.266 | 0.265 | 266.000 | 267.000 | 1203.000 | 265.000 | -0.069 | 0.982 | 0.494 | 0.508 | 0.389 | 0.097 |

## Secondary Horizons

**Appendix Table B5. Secondary horizon sensitivity analyses.**

| label | prevalence | auroc | auprc | brier | ece | precision | recall | f1 | specificity | alert_rate | tp | fp | tn | fn | calibration_intercept | calibration_slope | recall_at_primary_ppv |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| death_within_48h | 0.004 | 0.705 | 0.013 | 0.004 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | 2020.000 | 9.000 | -0.168 | 0.964 | NA |
| death_within_7d | 0.115 | 0.732 | 0.261 | 0.094 | 0.011 | 0.393 | 0.048 | 0.085 | 0.990 | 0.014 | 11.000 | 17.000 | 1753.000 | 219.000 | -0.160 | 0.891 | 0.060 |
| death_before_discharge | 0.265 | 0.763 | 0.546 | 0.161 | 0.016 | 0.493 | 0.550 | 0.520 | 0.796 | 0.296 | 292.000 | 300.000 | 1170.000 | 239.000 | -0.032 | 1.009 | 0.548 |

The secondary horizons are intentionally kept in the supplement. The 48-hour endpoint has prevalence `0.004` and produced AUROC `0.705` with no usable `PPV >= 0.50` operating point. The 7-day endpoint is more stable (AUROC `0.732`) but retains low recall at the conservative primary precision target (`0.060`), so it is treated as supporting sensitivity evidence rather than a second main result.

## ROC And Precision-Recall Benchmarks

**Appendix Figure B1. ROC and PR benchmark figure.**

![Appendix Figure B1. ROC and PR benchmarks](../../local_outputs/artifacts/roc_pr_benchmarks.png)

**Appendix Figure B2. Calibration curve for the deployed baseline.**

![Appendix Figure B2. Calibration curve](../../local_outputs/artifacts/calibration_curve.png)

**Appendix Figure B3. Decision curve for benchmark and fixed-policy comparators.**

![Appendix Figure B3. Decision curve](../../local_outputs/artifacts/decision_curve.png)

## Disagreement-Based Selective Triage Summary

**Appendix Table B6. Repeated grouped disagreement-based selective-triage summary.**

| policy_name | alert_precision_mean | alert_precision_std | alert_recall_mean | alert_recall_std | actionable_coverage_mean | actionable_coverage_std | low_risk_npv_mean | low_risk_npv_std | defer_rate_mean | defer_rate_std | actionable_error_rate_mean | actionable_error_rate_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fixed_threshold | 0.491 | 0.007 | 0.613 | 0.056 | 0.520 | 0.052 | 0.955 | 0.009 | 0.480 | 0.052 | 0.343 | 0.008 |
| selective_triage | 0.484 | 0.025 | 0.225 | 0.151 | 0.260 | 0.139 | 0.960 | 0.013 | 0.740 | 0.139 | 0.254 | 0.033 |

## Disagreement Shift Summary

**Appendix Table B7. Disagreement-based shift sensitivity summary.**

| scenario | severity | policy_name | alert_rate | low_risk_coverage | defer_rate | actionable_coverage | alert_precision | alert_recall | low_risk_npv | defer_event_rate | actionable_error_rate | high_count | low_count | defer_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| physiology_missingness_shift | 0.100 | fixed_threshold | 0.278 | 0.164 | 0.557 | 0.443 | 0.503 | 0.527 | 0.936 | 0.206 | 0.336 | 557.000 | 329.000 | 1115.000 |
| physiology_missingness_shift | 0.100 | selective_triage | 0.093 | 0.128 | 0.779 | 0.221 | 0.543 | 0.190 | 0.926 | 0.264 | 0.235 | 186.000 | 256.000 | 1559.000 |
| measurement_process_dropout | 1.000 | fixed_threshold | 0.307 | 0.132 | 0.561 | 0.439 | 0.486 | 0.563 | 0.955 | 0.196 | 0.373 | 615.000 | 264.000 | 1122.000 |
| measurement_process_dropout | 1.000 | selective_triage | 0.100 | 0.100 | 0.800 | 0.200 | 0.475 | 0.179 | 0.955 | 0.267 | 0.285 | 200.000 | 200.000 | 1601.000 |
| care_process_dropout | 1.000 | fixed_threshold | 0.296 | 0.167 | 0.537 | 0.463 | 0.493 | 0.550 | 0.952 | 0.208 | 0.341 | 592.000 | 335.000 | 1074.000 |
| care_process_dropout | 1.000 | selective_triage | 0.106 | 0.133 | 0.761 | 0.239 | 0.514 | 0.205 | 0.951 | 0.269 | 0.243 | 212.000 | 266.000 | 1523.000 |

## Clinical Score Summary

**Appendix Table B8. Repeated grouped clinical score summary.**

| model_name | auroc__mean | auroc__std | auprc__mean | auprc__std | brier__mean | brier__std | ece__mean | ece__std | recall_at_ppv_050__mean | recall_at_ppv_050__std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| apache_iii_score | 0.579 | 0.007 | 0.330 | 0.010 | 0.245 | 0.001 | 0.226 | 0.003 | 0.023 | 0.019 |
| sofa_total_24hr | 0.646 | 0.009 | 0.397 | 0.012 | 0.233 | 0.001 | 0.217 | 0.005 | 0.146 | 0.002 |

## Clinical Score Operating Points

**Appendix Table B9. Clinical score operating points on the main grouped split.**

| score_name | auroc | auprc | brier | ece | precision | recall | f1 | specificity | alert_rate | prevalence | tp | fp | tn | fn | calibration_intercept | calibration_slope | threshold_ppv_050 | precision_at_ppv_050_threshold | recall_at_ppv_050_threshold | alert_rate_at_ppv_050_threshold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sofa_total_24hr | 0.654 | 0.408 | 0.232 | 0.221 | 0.619 | 0.113 | 0.191 | 0.975 | 0.048 | 0.265 | 60.000 | 37.000 | 1433.000 | 471.000 | -1.029 | 1.098 | 0.695 | 0.500 | 0.147 | 0.078 |
| apache_iii_score | 0.586 | 0.323 | 0.244 | 0.228 | 0.480 | 0.023 | 0.043 | 0.991 | 0.012 | 0.265 | 12.000 | 13.000 | 1457.000 | 519.000 | -1.014 | 0.931 | 0.658 | 0.571 | 0.028 | 0.013 |

## Clinical Utility Summary At Key Thresholds

**Appendix Table B10. Key decision-curve net-benefit comparisons.**

| threshold | strategy | net_benefit | standardized_net_benefit | alert_rate |
| --- | --- | --- | --- | --- |
| 0.100 | apache_iii_score_continuous | 0.184 | 0.692 | 1.000 |
| 0.100 | conformal_alpha_0.05 | 0.064 | 0.242 | 0.107 |
| 0.100 | disagreement_selective | 0.045 | 0.170 | 0.099 |
| 0.100 | fixed_threshold_ppv50 | 0.129 | 0.487 | 0.296 |
| 0.100 | sofa_total_24hr_continuous | 0.184 | 0.692 | 1.000 |
| 0.100 | union_conformal_alpha_0.05 | 0.037 | 0.138 | 0.050 |
| 0.100 | xgboost_continuous | 0.194 | 0.730 | 0.821 |
| 0.200 | apache_iii_score_continuous | 0.082 | 0.308 | 1.000 |
| 0.200 | conformal_alpha_0.05 | 0.059 | 0.222 | 0.107 |
| 0.200 | disagreement_selective | 0.038 | 0.145 | 0.099 |
| 0.200 | fixed_threshold_ppv50 | 0.108 | 0.409 | 0.296 |
| 0.200 | sofa_total_24hr_continuous | 0.082 | 0.308 | 1.000 |
| 0.200 | union_conformal_alpha_0.05 | 0.035 | 0.131 | 0.050 |
| 0.200 | xgboost_continuous | 0.133 | 0.500 | 0.549 |
| 0.300 | apache_iii_score_continuous | -0.049 | -0.184 | 0.997 |
| 0.300 | conformal_alpha_0.05 | 0.052 | 0.196 | 0.107 |
| 0.300 | disagreement_selective | 0.030 | 0.112 | 0.099 |
| 0.300 | fixed_threshold_ppv50 | 0.082 | 0.308 | 0.296 |
| 0.300 | sofa_total_24hr_continuous | -0.034 | -0.130 | 0.952 |
| 0.300 | union_conformal_alpha_0.05 | 0.033 | 0.123 | 0.050 |
| 0.300 | xgboost_continuous | 0.092 | 0.345 | 0.364 |

## Fixed-Policy Clinical Utility Summary

**Appendix Table B11. Fixed-policy clinical utility summary.**

| strategy | alert_rate | alert_precision | alert_recall | net_benefit_at_0.10 | net_benefit_at_0.20 | net_benefit_at_0.30 |
| --- | --- | --- | --- | --- | --- | --- |
| treat_all | 1.000 | 0.265 | 1.000 | 0.184 | 0.082 | -0.049 |
| treat_none | 0.000 | NA | 0.000 | 0.000 | 0.000 | 0.000 |
| fixed_threshold_ppv50 | 0.296 | 0.493 | 0.550 | 0.129 | 0.108 | 0.082 |
| disagreement_selective | 0.099 | 0.510 | 0.190 | 0.045 | 0.038 | 0.030 |
| conformal_alpha_0.05 | 0.107 | 0.640 | 0.258 | 0.064 | 0.059 | 0.052 |
| conformal_alpha_0.10 | 0.157 | 0.589 | 0.348 | 0.085 | 0.076 | 0.065 |
| union_conformal_alpha_0.05 | 0.050 | 0.752 | 0.143 | 0.037 | 0.035 | 0.033 |

The continuous `xgboost` score is the primary clinical-score comparator because it retains higher net benefit against `SOFA` and `APACHE-III` across the main threshold range. The conformal and disagreement rows above should instead be read as action-restriction policies: they alert on fewer patients (`alpha=0.05` conformal alert rate `0.107`, union alert rate `0.050`) in exchange for higher alert precision and a larger defer region.

## Negative And Null Results From Probe Experiments

**Appendix Table B12. Negative and null probe results carried forward for transparency.**

| probe | auroc | auprc | recall@ppv50 |
| --- | --- | --- | --- |
| feat_missingness | 0.755 | 0.537 | 0.563 |
| feat_ratios | 0.751 | 0.526 | 0.514 |
| ensemble_stack_3 | 0.759 | 0.537 | 0.544 |
| hpo_lgbm | 0.760 | 0.540 | 0.595 |
| survival_rsf | 0.726 | NA | NA |
