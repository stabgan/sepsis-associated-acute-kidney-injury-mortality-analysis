# Appendix B: Extended Benchmarks

## Feature Ablation

| feature_set | model_name | auroc | auprc | brier | ece | precision | recall | f1 | specificity | alert_rate | prevalence | tp | fp | tn | fn | calibration_intercept | calibration_slope | recall_at_ppv_050 | precision_at_ppv_050_threshold | threshold_ppv_050 | low_risk_coverage_npv_095 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full | lightgbm | 0.766 | 0.548 | 0.161 | 0.022 | 0.489 | 0.576 | 0.529 | 0.782 | 0.313 | 0.265 | 306.000 | 320.000 | 1150.000 | 225.000 | 0.008 | 1.058 | 0.585 | 0.501 | 0.327 | 0.119 |
| physiology_plus_care | lightgbm | 0.761 | 0.545 | 0.162 | 0.023 | 0.490 | 0.582 | 0.532 | 0.781 | 0.315 | 0.265 | 309.000 | 322.000 | 1148.000 | 222.000 | 0.010 | 1.060 | 0.573 | 0.500 | 0.325 | 0.092 |
| physiology_severity | lightgbm | 0.759 | 0.543 | 0.162 | 0.030 | 0.500 | 0.535 | 0.517 | 0.807 | 0.284 | 0.265 | 284.000 | 284.000 | 1186.000 | 247.000 | 0.009 | 1.060 | 0.527 | 0.500 | 0.347 | 0.178 |

## Secondary Horizons

| label | prevalence | auroc | auprc | brier | ece | precision | recall | f1 | specificity | alert_rate | tp | fp | tn | fn | calibration_intercept | calibration_slope | recall_at_primary_ppv |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| death_within_48h | 0.004 | 0.380 | 0.018 | 0.004 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | 2020.000 | 9.000 | -0.177 | 0.962 | NA |
| death_within_7d | 0.115 | 0.751 | 0.281 | 0.094 | 0.024 | 0.425 | 0.161 | 0.233 | 0.972 | 0.043 | 37.000 | 50.000 | 1720.000 | 193.000 | 0.178 | 1.073 | 0.185 |
| death_before_discharge | 0.265 | 0.766 | 0.548 | 0.161 | 0.022 | 0.489 | 0.576 | 0.529 | 0.782 | 0.313 | 306.000 | 320.000 | 1150.000 | 225.000 | 0.008 | 1.058 | 0.585 |

## Calibration And Decision Curves

**Appendix Figure B1. Calibration curve for the deployed baseline.**

![Appendix Figure B1. Calibration curve](../../local_outputs/artifacts/calibration_curve.png)

**Appendix Figure B2. Decision curve for the deployed baseline.**

![Appendix Figure B2. Decision curve](../../local_outputs/artifacts/decision_curve.png)

## Disagreement-Based Selective Triage Summary

| policy_name | alert_precision_mean | alert_precision_std | alert_recall_mean | alert_recall_std | actionable_coverage_mean | actionable_coverage_std | low_risk_npv_mean | low_risk_npv_std | defer_rate_mean | defer_rate_std | actionable_error_rate_mean | actionable_error_rate_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fixed_threshold | 0.492 | 0.004 | 0.630 | 0.060 | 0.541 | 0.059 | 0.950 | 0.012 | 0.459 | 0.059 | 0.341 | 0.008 |
| selective_triage | 0.522 | 0.018 | 0.288 | 0.129 | 0.338 | 0.097 | 0.952 | 0.013 | 0.662 | 0.097 | 0.227 | 0.036 |

## Disagreement Shift Summary

| scenario | severity | policy_name | alert_rate | low_risk_coverage | defer_rate | actionable_coverage | alert_precision | alert_recall | low_risk_npv | defer_event_rate | actionable_error_rate | high_count | low_count | defer_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| physiology_missingness_shift | 0.100 | fixed_threshold | 0.284 | 0.181 | 0.535 | 0.465 | 0.495 | 0.529 | 0.939 | 0.213 | 0.332 | 568.000 | 363.000 | 1070.000 |
| physiology_missingness_shift | 0.100 | selective_triage | 0.143 | 0.179 | 0.678 | 0.322 | 0.493 | 0.266 | 0.942 | 0.272 | 0.257 | 286.000 | 359.000 | 1356.000 |
| measurement_process_dropout | 1.000 | fixed_threshold | 0.327 | 0.148 | 0.524 | 0.476 | 0.481 | 0.593 | 0.966 | 0.196 | 0.368 | 655.000 | 297.000 | 1049.000 |
| measurement_process_dropout | 1.000 | selective_triage | 0.165 | 0.148 | 0.687 | 0.313 | 0.535 | 0.333 | 0.966 | 0.250 | 0.262 | 331.000 | 296.000 | 1374.000 |
| care_process_dropout | 1.000 | fixed_threshold | 0.312 | 0.183 | 0.504 | 0.496 | 0.491 | 0.578 | 0.956 | 0.206 | 0.337 | 625.000 | 367.000 | 1009.000 |
| care_process_dropout | 1.000 | selective_triage | 0.161 | 0.182 | 0.656 | 0.344 | 0.545 | 0.331 | 0.956 | 0.258 | 0.237 | 323.000 | 365.000 | 1313.000 |

## Clinical Score Summary

| model_name | auroc__mean | auroc__std | auprc__mean | auprc__std | brier__mean | brier__std | ece__mean | ece__std | recall_at_ppv_050__mean | recall_at_ppv_050__std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| apache_iii_score | 0.579 | 0.007 | 0.330 | 0.010 | 0.245 | 0.001 | 0.226 | 0.003 | 0.023 | 0.019 |
| sofa_total_24hr | 0.646 | 0.009 | 0.397 | 0.012 | 0.233 | 0.001 | 0.217 | 0.005 | 0.146 | 0.002 |

## Negative And Null Results From Probe Experiments

| probe | auroc | auprc | recall@ppv50 |
| --- | --- | --- | --- |
| feat_missingness | 0.755 | 0.537 | 0.563 |
| feat_ratios | 0.751 | 0.526 | 0.514 |
| ensemble_stack_3 | 0.759 | 0.537 | 0.544 |
| hpo_lgbm | 0.760 | 0.540 | 0.595 |
| survival_rsf | 0.726 | NA | NA |
