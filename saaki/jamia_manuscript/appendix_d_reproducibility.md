# Appendix D: Reproducibility

## Build Steps

1. Run `saaki/deployment_analysis.py` to generate the canonical artifact bundle in `local_outputs/artifacts/`.
2. Run `saaki/build_jamia_manuscript.py` to materialize the chaptered manuscript in `saaki/jamia_manuscript/`.

## Key Artifact Files

- `local_outputs/artifacts/journal_positioning.md`
- `local_outputs/artifacts/benchmark_summary.csv`
- `local_outputs/artifacts/prediction_ceiling_results.csv`
- `local_outputs/artifacts/conformal_single_model_results.csv`
- `local_outputs/artifacts/conformal_consensus_results.csv`
- `local_outputs/artifacts/conformal_shift_results.csv`
- `local_outputs/artifacts/conformal_operating_curve.csv`
- `local_outputs/artifacts/conformal_subgroup_results.csv`
- `local_outputs/artifacts/bootstrap_intervals.json`
- `local_outputs/artifacts/cluster_bootstrap_intervals.json`

## Confidence Intervals

- Row-bootstrap AUROC CI: `0.744` to `0.787`
- Cluster-bootstrap AUROC CI: `0.742` to `0.789`

## Journal Positioning

The manuscript is written for `JAMIA`, with `npj Digital Medicine` and `Communications Medicine` treated as future stretch targets after stronger temporal or external validation.
