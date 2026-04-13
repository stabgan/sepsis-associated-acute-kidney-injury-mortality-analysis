# Appendix D: Reproducibility

## Build Steps

1. Run `saaki/deployment_analysis.py` to generate the canonical artifact bundle in `local_outputs/artifacts/`.
2. Run `saaki/build_jamia_manuscript.py` to materialize the chaptered manuscript in `saaki/jamia_manuscript/`.
3. Use the generated figure and table manifests to port the markdown package into the final journal submission format.

## Key Artifact Files

- `local_outputs/artifacts/journal_positioning.md`
- `local_outputs/artifacts/benchmark_summary.csv`
- `local_outputs/artifacts/canonical_hpo_summary.json`
- `local_outputs/artifacts/canonical_hpo_top_trials.csv`
- `local_outputs/artifacts/clinical_score_operating_points.csv`
- `local_outputs/artifacts/prediction_ceiling_results.csv`
- `local_outputs/artifacts/conformal_single_model_results.csv`
- `local_outputs/artifacts/conformal_single_model_summary.csv`
- `local_outputs/artifacts/conformal_consensus_results.csv`
- `local_outputs/artifacts/conformal_consensus_summary.csv`
- `local_outputs/artifacts/conformal_shift_results.csv`
- `local_outputs/artifacts/conformal_operating_curve.csv`
- `local_outputs/artifacts/conformal_subgroup_results.csv`
- `local_outputs/artifacts/decision_curve_summary.csv`
- `local_outputs/artifacts/decision_curve_policy_metrics.csv`
- `local_outputs/artifacts/bootstrap_intervals.json`
- `local_outputs/artifacts/cluster_bootstrap_intervals.json`

## Confidence Intervals

- Row-bootstrap AUROC CI: `0.742` to `0.786`
- Cluster-bootstrap AUROC CI: `0.741` to `0.787`
- Headline conformal summaries are repeated across `21` grouped seeds.

## Selected Model Snapshot

- Selected model: `xgboost` with `isotonic` calibration.
- Mean AUROC: `0.768`.
- Mean recall at `PPV >= 0.50`: `0.608`.
- Decision-curve net benefit at threshold `0.20`: `0.133`.

## Data Availability

The source patient-level data are not redistributed with this repository. Reproducing the cohort from raw source requires independent credentialed access to MIMIC and the corresponding data-use approvals. The checked-in repository provides the analysis code, derived artifact summaries, and manuscript assets used to support the internal-validation claims in this package.

## Journal Positioning

The manuscript is written for `JAMIA`, with `npj Digital Medicine` and `Communications Medicine` treated as future stretch targets after stronger temporal or external validation.
