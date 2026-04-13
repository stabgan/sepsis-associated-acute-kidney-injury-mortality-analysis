# Results

## Cohort And Data Contract

Table 1 summarizes the working cohort and the constraints that bound the manuscript claims.

**Table 1. Cohort and data-contract summary.**

| Item | Value |
| --- | --- |
| Working cohort | mimic_saaki_raw_v2.csv |
| Rows | 10036 |
| Columns | 254 |
| Outcome prevalence | 0.267 |
| Prediction time | 24 hours after ICU admission (T24). |
| Outcome | `event_observed` is the in-hospital death label; `time_to_event_hrs` is hours from ICU admission to death or last known alive/discharge censoring. |
| Evaluation | Subject-grouped holdout by `subject_id` |
| Excluded claims | No bedside-readiness claim.; No external-validation or multi-center transportability claim.; No claim that the v2 cohort exactly matches the thesis cohort description without ETL reconciliation.; No reliance on stay-level metrics for deployment-readiness claims. |

## Prediction Ceiling And Conventional Benchmarks

Figure 2 and Table 2 anchor the benchmark story. The main result is that the discrimination ceiling is bounded and the baseline model family differences are comparatively small.

**Figure 2. Prediction ceiling and benchmark summary.**

![Figure 2. Prediction ceiling](../../local_outputs/artifacts/prediction_ceiling.png)

**Table 2. Baseline discrimination and calibration benchmark.**

| model_name | auroc__mean | auroc__std | auprc__mean | auprc__std | brier__mean | ece__mean | calibration_intercept__mean | calibration_slope__mean | recall_at_ppv_050__mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| apache_iii_score | 0.579 | 0.007 | 0.330 | 0.010 | 0.245 | 0.226 | -1.003 | 0.843 | 0.023 |
| lightgbm | 0.765 | 0.002 | 0.545 | 0.004 | 0.161 | 0.021 | 0.011 | 1.031 | 0.575 |
| logistic | 0.751 | 0.003 | 0.532 | 0.008 | 0.165 | 0.026 | 0.114 | 1.140 | 0.556 |
| sofa_total_24hr | 0.646 | 0.009 | 0.397 | 0.012 | 0.233 | 0.217 | -1.008 | 1.017 | 0.146 |
| xgboost | 0.768 | 0.006 | 0.544 | 0.004 | 0.161 | 0.017 | 0.021 | 1.035 | 0.608 |

**Figure 7. Calibration and clinical-utility panel.**

![Figure 7. Calibration and clinical utility](../../local_outputs/artifacts/clinical_utility_panel.png)

The ceiling experiment is especially important. In the canonical workflow, the leak-prone configuration that adds `time_to_event_hrs` reaches AUROC `0.810`, while the honest grouped setting remains materially lower. The repeated grouped benchmark selected `xgboost` because it combined mean AUROC `0.768`, Brier `0.161`, and recall at `PPV >= 0.50` `0.608` more consistently than the competing primary baselines. The grouped Optuna pass improved LightGBM validation recall at `PPV >= 0.50` to `0.590`, but repeated grouped LightGBM still averaged AUROC `0.765` and recall `0.575`. The exploratory CatBoost comparator remained competitive (AUROC `0.768`, Brier `0.161`, recall `0.623`) but did not materially simplify or improve the downstream conformal story, so it stayed in the appendix as a diversity check rather than replacing the primary model family.

The clinical-utility panel complements that story. At decision threshold `0.20`, the continuous `xgboost` score yields net benefit `0.133` versus `0.082` for `APACHE-III`, `0.082` for `SOFA`, and `0.108` for the fixed-threshold `PPV >= 0.50` policy. The conformal `alpha=0.05` policy is deliberately more conservative, with net benefit `0.059`, alert rate `0.107`, and alert PPV `0.640`. This distinction matters: the continuous model is the main utility comparison against clinical scores, whereas the conformal and disagreement policies are complementary deployment policies that trade some net benefit for narrower, higher-confidence action sets.

## Main Conformal Triage Result

Table 3 summarizes repeated subject-grouped single-model conformal triage results. The main manuscript operating point remains the low-alpha regime, where coverage is controlled while a clinically meaningful defer region is preserved.

**Table 3. Main conformal results across repeated grouped splits.**

| alpha | n_groups | coverage_mean | coverage_ci_low | coverage_ci_high | certain_frac_mean | alert_ppv_mean | clear_npv_mean | miss_count_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.050 | 21.000 | 0.952 | 0.945 | 0.959 | 0.318 | 0.643 | 0.948 | 23.095 |
| 0.100 | 21.000 | 0.900 | 0.882 | 0.914 | 0.503 | 0.579 | 0.924 | 49.952 |
| 0.150 | 21.000 | 0.848 | 0.826 | 0.871 | 0.653 | 0.540 | 0.903 | 80.190 |
| 0.200 | 21.000 | 0.797 | 0.771 | 0.821 | 0.782 | 0.509 | 0.887 | 109.000 |

**Figure 5. Conformal operating-characteristic curve.**

![Figure 5. Conformal operating curve](../../local_outputs/artifacts/conformal_operating_curve.png)

Under the manuscript sweet-spot criterion (`NPV >= 0.90` and `PPV >= 0.55`), the best operating point occurs at `alpha=0.12` with certain-decision fraction `0.532`, alert PPV `0.556`, and clear NPV `0.912`. The repeated-seed summary in Table 3 shows that these headline estimates remain stable across `21` grouped splits.

## Multi-Model Consensus

Table 4 and Figure 6 show the consensus extension. The union of conformal prediction sets is deliberately conservative: it acts on fewer patients but produces a more reliable actionable region.

**Table 4. Multi-model consensus conformal results.**

| ensemble_name | base_model | alpha | n_groups | coverage_mean | certain_frac_mean | alert_ppv_mean | clear_npv_mean | miss_count_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| intersection | lightgbm+xgboost+logistic | 0.050 | 21 | 0.911 | 0.462 | 0.591 | 0.926 | 45.048 |
| intersection | lightgbm+xgboost+logistic | 0.100 | 21 | 0.835 | 0.683 | 0.534 | 0.901 | 86.000 |
| intersection | lightgbm+xgboost+logistic | 0.150 | 21 | 0.766 | 0.820 | 0.501 | 0.885 | 116.762 |
| intersection | lightgbm+xgboost+logistic | 0.200 | 21 | 0.699 | 0.888 | 0.486 | 0.876 | 134.000 |
| single_model | xgboost | 0.050 | 21 | 0.952 | 0.318 | 0.643 | 0.948 | 23.095 |
| single_model | xgboost | 0.100 | 21 | 0.900 | 0.503 | 0.579 | 0.924 | 49.952 |
| single_model | xgboost | 0.150 | 21 | 0.848 | 0.653 | 0.540 | 0.903 | 80.190 |
| single_model | xgboost | 0.200 | 21 | 0.797 | 0.782 | 0.509 | 0.887 | 109.000 |
| union | lightgbm+xgboost+logistic | 0.050 | 21 | 0.982 | 0.164 | 0.729 | 0.962 | 8.714 |
| union | lightgbm+xgboost+logistic | 0.100 | 21 | 0.954 | 0.308 | 0.651 | 0.949 | 21.381 |
| union | lightgbm+xgboost+logistic | 0.150 | 21 | 0.922 | 0.432 | 0.606 | 0.933 | 38.190 |
| union | lightgbm+xgboost+logistic | 0.200 | 21 | 0.885 | 0.550 | 0.568 | 0.918 | 58.286 |

**Figure 6. Multi-model consensus trade-off panel.**

![Figure 6. Consensus trade-off](../../local_outputs/artifacts/conformal_consensus_tradeoff.png)

At `alpha=0.05`, the union consensus yields alert PPV `0.729` and clear NPV `0.962`, confirming that consensus amplifies reliability at the cost of automation rate.

## Comparison With Disagreement-Based Selective Triage

The disagreement-based policy remains an important ablation because it shows that explicit deferral helps, but its guarantees are empirical and heuristic rather than finite-sample. The repeated grouped summary is included in Appendix B. In brief, disagreement triage narrows the actionable region and reduces actionable error, but conformal triage supplies the stronger methodological framing because it binds that defer region to a formal coverage objective.

## Robustness Under Distribution Shift

**Table 5. Shift robustness versus fixed-threshold policies.**

| scenario | severity | conformal_coverage | conformal_certain_frac | conformal_alert_ppv | conformal_clear_npv | fixed_precision_t050 | fixed_recall_t050 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| clean | 0.000 | 0.950 | 0.316 | 0.640 | 0.943 | 0.593 | 0.294 |
| physiology_missingness_shift | 0.100 | 0.950 | 0.297 | 0.597 | 0.935 | 0.573 | 0.252 |
| physiology_missingness_shift | 0.200 | 0.963 | 0.268 | 0.632 | 0.938 | 0.618 | 0.207 |
| physiology_missingness_shift | 0.300 | 0.970 | 0.235 | 0.663 | 0.925 | 0.619 | 0.171 |
| measurement_process_dropout | 1.000 | 0.950 | 0.281 | 0.646 | 0.943 | 0.596 | 0.309 |
| care_process_dropout | 1.000 | 0.947 | 0.320 | 0.624 | 0.945 | 0.606 | 0.301 |

**Figure 8. Robustness-under-shift panel.**

![Figure 8. Shift robustness](../../local_outputs/artifacts/conformal_shift_panel.png)

The shift experiments support the deployment argument. As extra missingness is injected, conformal coverage remains stable because the framework defers more cases. By contrast, fixed thresholds lose recall while offering no explicit warning that uncertainty has increased.

## Subgroup Heterogeneity

**Table 6. Subgroup heterogeneity summary for conformal triage at alpha=0.10.**

| subgroup | value | n | event_rate | coverage | certain_frac | alert_ppv | clear_npv |
| --- | --- | --- | --- | --- | --- | --- | --- |
| age_band | 51-65 | 556 | 0.257 | 0.903 | 0.523 | 0.571 | 0.902 |
| age_band | 66-80 | 776 | 0.277 | 0.912 | 0.448 | 0.603 | 0.926 |
| age_band | <=50 | 244 | 0.201 | 0.930 | 0.602 | 0.621 | 0.949 |
| age_band | >80 | 425 | 0.292 | 0.908 | 0.344 | 0.571 | 0.913 |
| gender | F | 954 | 0.257 | 0.921 | 0.454 | 0.620 | 0.928 |
| gender | M | 1047 | 0.273 | 0.902 | 0.477 | 0.564 | 0.914 |
| mech_vent | -1 | 916 | 0.271 | 0.904 | 0.457 | 0.558 | 0.928 |
| mech_vent | 1 | 1085 | 0.261 | 0.917 | 0.473 | 0.620 | 0.915 |
| sofa_band | High SOFA | 566 | 0.412 | 0.866 | 0.484 | 0.659 | 0.842 |
| sofa_band | Low SOFA | 855 | 0.171 | 0.949 | 0.480 | 0.469 | 0.929 |
| sofa_band | Medium SOFA | 580 | 0.262 | 0.900 | 0.428 | 0.505 | 0.952 |
| vasopressor | -1 | 1277 | 0.230 | 0.926 | 0.486 | 0.586 | 0.932 |
| vasopressor | 1 | 724 | 0.327 | 0.885 | 0.431 | 0.593 | 0.887 |

**Figure 9. Subgroup coverage and reliability forest plot.**

![Figure 9. Subgroup forest plot](../../local_outputs/artifacts/conformal_subgroup_forest.png)

Coverage and clear NPV remain high across most demographic subgroups, but high-acuity strata are predictably harder. Reporting the subgroup summaries at `alpha=0.10` exposes that trade-off explicitly: low-SOFA strata preserve high coverage with a reasonable action rate, whereas high-SOFA strata require a larger defer region. This reinforces the paper's deployment message that the model should be treated as a triage assistant with a defer option, not as a universal autonomous classifier.
