# Claim-to-Evidence Map

| Claim | Evidence | Section |
| --- | --- | --- |
| Prediction performance is bounded in this T24 SA-AKI cohort. | prediction_ceiling_results.csv; benchmark_summary.csv; probe_all_results.json | 01_introduction.md, 03_results.md |
| Grouped Optuna tuning and an added CatBoost comparator do not overturn the narrow benchmark ranking, so the manuscript should prioritize safe deployment rather than architecture churn. | canonical_hpo_summary.json; canonical_hpo_top_trials.csv; benchmark_summary.csv; repeated_group_results.csv | 01_introduction.md, 02_methods.md, 03_results.md, Appendix B |
| Conformal selective triage provides a principled Alert/Defer/Clear framework. | conformal_single_model_results.csv; conformal_operating_curve.csv | 02_methods.md, 03_results.md |
| Multi-model conformal consensus amplifies reliability while reducing automation rate. | conformal_consensus_results.csv; conformal_consensus_tradeoff.png | 03_results.md |
| Conformal triage degrades more gracefully under shift than fixed thresholds. | conformal_shift_results.csv; fixed_threshold_shift_results.csv; conformal_shift_panel.png | 03_results.md |
| The selected ML model retains better decision-curve net benefit than APACHE-III and SOFA across the main threshold range. | decision_curve_summary.csv; clinical_utility_panel.png; clinical_score_operating_points.csv | 03_results.md, Appendix B |
| Disagreement-based selective triage is a useful comparator but weaker as a main methodological story. | selective_triage_results.csv; selective_triage_shift_results.csv | 03_results.md, Appendix B |
| The paper is positioned for JAMIA with careful internal-validation claims only. | journal_positioning.md; data_contract.md; audit_report.md | 00_title_and_abstract.md, 05_limitations.md, Appendix D |
