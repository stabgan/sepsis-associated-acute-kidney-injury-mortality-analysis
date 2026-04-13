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
| lightgbm | 0.772 | 0.006 | 0.554 | 0.007 | 0.160 | 0.021 | 0.085 | 1.114 | 0.630 |
| logistic | 0.751 | 0.003 | 0.532 | 0.008 | 0.165 | 0.026 | 0.114 | 1.140 | 0.556 |
| sofa_total_24hr | 0.646 | 0.009 | 0.397 | 0.012 | 0.233 | 0.217 | -1.008 | 1.017 | 0.146 |
| xgboost | 0.768 | 0.006 | 0.544 | 0.004 | 0.161 | 0.017 | 0.021 | 1.035 | 0.608 |

**Figure 7. ROC and PR benchmark figure.**

![Figure 7. ROC and PR benchmarks](../../local_outputs/artifacts/roc_pr_benchmarks.png)

The ceiling experiment is especially important. In the canonical workflow, the leak-prone configuration that adds `time_to_event_hrs` reaches AUROC `0.818`, while the honest grouped setting remains materially lower. This supports the argument that the main research opportunity is safe decision-making under bounded signal rather than marginal AUROC optimization.

## Main Conformal Triage Result

Table 3 summarizes repeated subject-grouped single-model conformal triage results. The main manuscript operating point remains the low-alpha regime, where coverage is controlled while a clinically meaningful defer region is preserved.

**Table 3. Main conformal results across repeated grouped splits.**

| alpha | coverage_mean | coverage_std | certain_frac_mean | certain_frac_std | alert_ppv_mean | alert_ppv_std | clear_npv_mean | clear_npv_std | miss_count_mean | miss_count_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.050 | 0.954 | 0.006 | 0.300 | 0.016 | 0.648 | 0.021 | 0.950 | 0.004 | 20.000 | 2.915 |
| 0.100 | 0.905 | 0.005 | 0.501 | 0.029 | 0.591 | 0.010 | 0.925 | 0.007 | 49.800 | 8.228 |
| 0.150 | 0.854 | 0.003 | 0.653 | 0.021 | 0.552 | 0.014 | 0.905 | 0.010 | 79.200 | 14.025 |
| 0.200 | 0.801 | 0.009 | 0.785 | 0.023 | 0.513 | 0.016 | 0.892 | 0.008 | 104.800 | 12.153 |

**Figure 5. Conformal operating-characteristic curve.**

![Figure 5. Conformal operating curve](../../local_outputs/artifacts/conformal_operating_curve.png)

Under the manuscript sweet-spot criterion (`NPV >= 0.90` and `PPV >= 0.55`), the best operating point occurs at `alpha=0.13` with certain-decision fraction `0.554`, alert PPV `0.556`, and clear NPV `0.918`.

## Multi-Model Consensus

Table 4 and Figure 6 show the consensus extension. The union of conformal prediction sets is deliberately conservative: it acts on fewer patients but produces a more reliable actionable region.

**Table 4. Multi-model consensus conformal results.**

| ensemble_name | base_model | alpha | coverage | certain_frac | alert_ppv | clear_npv | miss_count |
| --- | --- | --- | --- | --- | --- | --- | --- |
| intersection | lightgbm+xgboost+logistic | 0.050 | 0.918 | 0.435 | 0.602 | 0.928 | 40.600 |
| intersection | lightgbm+xgboost+logistic | 0.100 | 0.847 | 0.658 | 0.543 | 0.906 | 78.400 |
| intersection | lightgbm+xgboost+logistic | 0.150 | 0.776 | 0.810 | 0.510 | 0.885 | 116.400 |
| intersection | lightgbm+xgboost+logistic | 0.200 | 0.708 | 0.891 | 0.490 | 0.879 | 131.600 |
| single_model | lightgbm | 0.050 | 0.954 | 0.300 | 0.648 | 0.950 | 20.000 |
| single_model | lightgbm | 0.100 | 0.905 | 0.501 | 0.591 | 0.925 | 49.800 |
| single_model | lightgbm | 0.150 | 0.854 | 0.653 | 0.552 | 0.905 | 79.200 |
| single_model | lightgbm | 0.200 | 0.801 | 0.785 | 0.513 | 0.892 | 104.800 |
| union | lightgbm+xgboost+logistic | 0.050 | 0.981 | 0.169 | 0.735 | 0.953 | 11.200 |
| union | lightgbm+xgboost+logistic | 0.100 | 0.953 | 0.320 | 0.662 | 0.946 | 24.000 |
| union | lightgbm+xgboost+logistic | 0.150 | 0.920 | 0.447 | 0.613 | 0.927 | 43.600 |
| union | lightgbm+xgboost+logistic | 0.200 | 0.883 | 0.562 | 0.574 | 0.914 | 62.200 |

**Figure 6. Multi-model consensus trade-off panel.**

![Figure 6. Consensus trade-off](../../local_outputs/artifacts/conformal_consensus_tradeoff.png)

At `alpha=0.05`, the union consensus yields alert PPV `0.735` and clear NPV `0.953`, confirming that consensus amplifies reliability at the cost of automation rate.

## Comparison With Disagreement-Based Selective Triage

The disagreement-based policy remains an important ablation because it shows that explicit deferral helps, but its guarantees are empirical and heuristic rather than finite-sample. The repeated grouped summary is included in Appendix B. In brief, disagreement triage narrows the actionable region and reduces actionable error, but conformal triage supplies the stronger methodological framing because it binds that defer region to a formal coverage objective.

## Robustness Under Distribution Shift

**Table 5. Shift robustness versus fixed-threshold policies.**

| scenario | severity | conformal_coverage | conformal_certain_frac | conformal_alert_ppv | conformal_clear_npv | fixed_precision_t050 | fixed_recall_t050 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| clean | 0.000 | 0.952 | 0.291 | 0.646 | 0.955 | 0.608 | 0.339 |
| physiology_missingness_shift | 0.100 | 0.954 | 0.266 | 0.600 | 0.943 | 0.589 | 0.286 |
| physiology_missingness_shift | 0.200 | 0.964 | 0.247 | 0.615 | 0.952 | 0.591 | 0.239 |
| physiology_missingness_shift | 0.300 | 0.969 | 0.226 | 0.638 | 0.948 | 0.612 | 0.226 |
| measurement_process_dropout | 1.000 | 0.953 | 0.260 | 0.647 | 0.958 | 0.589 | 0.350 |
| care_process_dropout | 1.000 | 0.953 | 0.290 | 0.655 | 0.952 | 0.601 | 0.335 |

**Figure 8. Robustness-under-shift panel.**

![Figure 8. Shift robustness](../../local_outputs/artifacts/conformal_shift_panel.png)

The shift experiments support the deployment argument. As extra missingness is injected, conformal coverage remains stable because the framework defers more cases. By contrast, fixed thresholds lose recall while offering no explicit warning that uncertainty has increased.

## Subgroup Heterogeneity

**Table 6. Subgroup heterogeneity summary for conformal triage.**

| subgroup | value | n | event_rate | coverage | certain_frac | alert_ppv | clear_npv |
| --- | --- | --- | --- | --- | --- | --- | --- |
| age_band | 51-65 | 556 | 0.257 | 0.908 | 0.514 | 0.573 | 0.910 |
| age_band | 66-80 | 776 | 0.277 | 0.915 | 0.451 | 0.612 | 0.935 |
| age_band | <=50 | 244 | 0.201 | 0.926 | 0.627 | 0.647 | 0.950 |
| age_band | >80 | 425 | 0.292 | 0.880 | 0.374 | 0.500 | 0.890 |
| gender | F | 954 | 0.257 | 0.916 | 0.465 | 0.609 | 0.934 |
| gender | M | 1047 | 0.273 | 0.899 | 0.481 | 0.549 | 0.915 |
| mech_vent | -1 | 916 | 0.271 | 0.898 | 0.472 | 0.552 | 0.929 |
| mech_vent | 1 | 1085 | 0.261 | 0.914 | 0.476 | 0.604 | 0.920 |
| sofa_band | High SOFA | 566 | 0.412 | 0.866 | 0.486 | 0.661 | 0.860 |
| sofa_band | Low SOFA | 855 | 0.171 | 0.952 | 0.482 | 0.500 | 0.934 |
| sofa_band | Medium SOFA | 580 | 0.262 | 0.881 | 0.450 | 0.454 | 0.935 |
| vasopressor | -1 | 1277 | 0.230 | 0.920 | 0.494 | 0.572 | 0.928 |
| vasopressor | 1 | 724 | 0.327 | 0.884 | 0.438 | 0.582 | 0.912 |

**Figure 9. Subgroup coverage and reliability forest plot.**

![Figure 9. Subgroup forest plot](../../local_outputs/artifacts/conformal_subgroup_forest.png)

Coverage and clear NPV remain high across most demographic subgroups, but high-acuity strata are predictably harder. This reinforces the paper's deployment message: the model should be treated as a triage assistant with a defer option, not as a universal autonomous classifier.
