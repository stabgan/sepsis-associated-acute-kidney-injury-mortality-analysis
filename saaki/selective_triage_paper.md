# Observation-Process-Aware Selective Triage for SA-AKI Mortality

## Working Title

Observation-Process-Aware Selective Triage for Early SA-AKI Mortality Risk Prediction

## One-Sentence Claim

In a leakage-aware SA-AKI cohort scored at T24, a disagreement-based selective triage policy reduces actionable error and improves alert precision relative to a fixed-threshold baseline, but it does so by deferring substantially more cases and sacrificing recall.

## Problem Setting

- Population: adults with sepsis-associated acute kidney injury from the working v2 cohort.
- Prediction time: 24 hours after ICU admission.
- Primary task: high-risk escalation alert for in-hospital death.
- Secondary task: low-risk rule-out under explicit NPV constraints.
- Evaluation principle: subject-grouped holdout to avoid patient-level leakage.

## Why This Paper Exists

Most prior SA-AKI mortality work focuses on AUROC and often overstates deployment readiness. This project instead treats the model as a triage policy:

- What recall survives under explicit PPV floors?
- What alert burden is required?
- When should the model defer instead of pretending to know?

The key methodological idea is that the EHR observation process itself is informative. Care-process and measurement-process features can improve prediction, but disagreement between those process-enriched predictions and a physiology-only model is also a useful uncertainty signal.

## Data Contract

The working dataset is `mimic_saaki_raw_v2.csv`.

- One row corresponds to one ICU-stay-level SA-AKI cohort record scored at T24.
- `event_observed` is the in-hospital death label.
- `time_to_event_hrs` is hours from ICU admission to death or censoring.
- Patient duplication remains present, so deployment claims must use grouped evaluation by `subject_id`.

Known constraints:

- The v2 cohort does not exactly match all thesis/documentation wording.
- The current file contains process-rich variables.
- No bedside-readiness or external-validation claim is made.

## Baseline Deployment Story

The baseline is not "best AUROC wins." The baseline is a calibrated fixed-threshold deployment policy:

- Primary objective: maximize recall under a PPV floor.
- Secondary objective: maximize low-risk coverage under an NPV floor.
- Operational reporting: include alert burden explicitly.

Fresh grouped calibration benchmark:

- `lightgbm + sigmoid`: AUROC `0.761`, recall at `PPV >= 0.50` of `0.585`
- `xgboost + isotonic/sigmoid`: AUROC about `0.756-0.757`, recall at `PPV >= 0.50` about `0.541-0.548`
- `logistic`: clearly weaker under the same grouped protocol

Fresh repeated grouped baseline summary:

- `lightgbm`: mean AUROC `0.772 +/- 0.006`, mean AUPRC `0.554 +/- 0.007`, mean recall at `PPV >= 0.50` of `0.630 +/- 0.066`
- `xgboost`: mean AUROC `0.768 +/- 0.006`, mean recall at `PPV >= 0.50` of `0.608 +/- 0.073`
- `logistic`: mean AUROC `0.751 +/- 0.003`, mean recall at `PPV >= 0.50` of `0.556 +/- 0.059`

This makes `lightgbm` the current baseline winner for the deployment-first story.

Current threshold frontier summary for the winning grouped baseline:

- Fixed alert-rate policy:
  - at alert rate about `0.11`, precision is about `0.64` and recall about `0.26`
  - at alert rate about `0.21`, precision is about `0.57` and recall about `0.45`
  - at alert rate about `0.31`, precision is about `0.51` and recall about `0.60`
- Precision-constrained policy:
  - `PPV >= 0.50` yields mean recall about `0.63`
  - `PPV >= 0.60` yields mean recall about `0.37`
  - `PPV >= 0.70` yields mean recall about `0.17`
  - `PPV >= 0.80` yields mean recall about `0.07`
- Low-risk policy:
  - `NPV >= 0.95` yields mean low-risk coverage about `0.20`
  - `NPV >= 0.98` collapses coverage toward about `0.11`

This is exactly why `PPV >= 0.99` is treated as exploratory only: it becomes clinically trivial under grouped evaluation.

## Method: Observation-Process-Aware Selective Triage

Two models are trained on the same grouped split:

1. `M_full`
- Uses the full deployable feature set, including care-process and measurement-process variables.

2. `M_phys`
- Uses physiology and severity features only.

For each patient, compute:

- `p_full`
- `p_phys`
- `disagreement = |p_full - p_phys|`

Decision policy:

- `Alert`: `p_full` exceeds the high-risk threshold and disagreement is below a validation-selected agreement threshold.
- `Clear`: `p_full` is below the low-risk threshold and disagreement is below the agreement threshold.
- `Defer`: all other cases.

Interpretation:

- If the two models agree, the prediction is actionable.
- If they disagree, the observation process is telling us the case may be atypical or unstable, so the model should defer.

Current repeated grouped comparison:

- Fixed-threshold policy:
  - alert precision about `0.492`
  - alert recall about `0.630`
  - actionable coverage about `0.541`
  - actionable error rate about `0.341`
- Selective-triage policy:
  - alert precision about `0.522`
  - alert recall about `0.288`
  - actionable coverage about `0.338`
  - actionable error rate about `0.227`

So the current empirical story is safety-oriented rather than dominance-oriented: selective triage creates a narrower but more reliable actionable region.

Current shift-stress summary:

- Under care-process dropout:
  - fixed threshold: alert precision about `0.491`, alert recall about `0.578`, actionable coverage about `0.496`
  - selective triage: alert precision about `0.545`, alert recall about `0.331`, actionable coverage about `0.344`
- Under measurement-process dropout:
  - fixed threshold: alert precision about `0.481`, alert recall about `0.593`, actionable coverage about `0.476`
  - selective triage: alert precision about `0.535`, alert recall about `0.333`, actionable coverage about `0.313`
- Under physiology missingness shift:
  - fixed threshold: alert precision about `0.495`, alert recall about `0.529`, actionable coverage about `0.465`
  - selective triage: alert precision about `0.493`, alert recall about `0.266`, actionable coverage about `0.322`

This means the selective policy is not universally superior under every perturbation. Its main strength is concentrating actionable decisions into a lower-error region, not preserving broad coverage.

## Core Experiments

1. Fixed-threshold grouped deployment frontier
- Sweep PPV floors and alert-rate budgets.
- Report recall, precision, alert rate, and low-risk coverage.

2. Feature-set ablation
- Compare `full`, `physiology_severity`, and `physiology_plus_care`.
- Test whether process features materially change deployment performance.

Current ablation readout for the winning `lightgbm` baseline:

- `full`: AUROC `0.766`, recall at `PPV >= 0.50` of `0.585`
- `physiology_plus_care`: AUROC `0.761`, recall at `PPV >= 0.50` of `0.573`
- `physiology_severity`: AUROC `0.759`, recall at `PPV >= 0.50` of `0.527`

This supports the core hypothesis that process-rich information helps the deployment policy, but the incremental gain is modest enough that disagreement between the two views may be more useful than simply adding every process feature and trusting the score.

3. Selective triage
- Compare the fixed-threshold policy against the disagreement-based selective policy.
- Primary metrics: alert precision, alert recall, low-risk NPV, actionable coverage, defer rate, actionable error rate.

4. Shift robustness
- Missingness shift in physiology variables.
- Feature-group dropout for care-process and measurement-process families.
- Compare degradation of fixed-threshold vs selective-triage policies.

5. Subgroup reliability
- Evaluate the selected policy across age bands, gender, AKI trigger, ventilation, vasopressor, RRT, and missingness quartiles.

## What We Will Claim

- A realistic, subject-grouped T24 deployment baseline for SA-AKI mortality.
- A selective triage method that uses disagreement between physiology-only and process-enriched models as an uncertainty signal.
- A concrete trade-off: lower actionable error and higher alert precision in exchange for higher deferral and lower actionable coverage.
- This can still be valuable if the intended clinical role is to make a smaller set of alerts more trustworthy rather than to maximize autonomous coverage.

## What We Will Not Claim

- Bedside readiness.
- External transportability.
- Universal robustness guarantees.
- Foundation-model superiority.
- That AUROC alone justifies deployment.
- That the selective policy dominates the fixed-threshold frontier on every metric.

## Results To Fill As Artifacts Finish

- Threshold frontier summary from `deployment_frontier.csv`
- Threshold objective memo from `threshold_objective.md`
- Feature ablation summary from `feature_ablation_results.csv`
- Selective triage summary from `selective_triage_results.csv`
- Shift robustness summary from `selective_triage_shift_results.csv`
- Final deployment policy and bootstrap intervals from `clinical_policy.json` and `bootstrap_intervals.json`

## Stretch Work Decision

The current run already supports the scoped decision to defer larger appendix ideas until later:

- Conformal prediction remains deferred.
- TabPFN remains deferred.
- SHAP coherence scoring remains deferred.

That is the correct choice for this iteration because the main paper contribution is already identifiable without them, and the current selective-triage result still needs careful interpretation rather than additional complexity.

## Discussion

The paper should be framed as a deployment paper with one methodological contribution, not a leaderboard paper. The main question is not whether AUROC is `0.77` or `0.79`; it is whether the model can be turned into a clinically legible triage policy that knows when to defer.

The current evidence suggests the selective policy is most defensible in a high-friction workflow where false actionable decisions are costly and where a larger defer region is acceptable. If the clinical partner instead values maximal high-risk recall above all else, the fixed-threshold frontier may remain the primary operational recommendation and the selective policy becomes a safety-oriented variant rather than the default.
