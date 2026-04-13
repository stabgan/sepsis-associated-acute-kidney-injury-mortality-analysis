"""Build a chaptered JAMIA-style manuscript from generated artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
ARTIFACT_DIR = REPO_ROOT / "local_outputs" / "artifacts"
PROBE_DIR = REPO_ROOT / "local_outputs" / "probes"
MANUSCRIPT_DIR = SCRIPT_DIR / "jamia_manuscript"


def read_csv(name: str) -> pd.DataFrame:
    path = ARTIFACT_DIR / name
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def write_markdown(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n")


def fmt(value: object, digits: int = 3) -> str:
    try:
        if pd.isna(value):
            return "NA"
    except Exception:
        pass
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    if isinstance(value, int):
        return str(value)
    return str(value)


def render_table(frame: pd.DataFrame, *, index: bool = False, floatfmt: str = ".3f") -> str:
    if frame.empty:
        return "_No data available._"
    working = frame.copy()
    if index:
        working = working.reset_index()
    headers = [str(column) for column in working.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in working.iterrows():
        rendered = [fmt(value) for value in row.tolist()]
        lines.append("| " + " | ".join(rendered) + " |")
    return "\n".join(lines)


def clean_markdown(text: str) -> str:
    cleaned_lines = []
    for line in text.splitlines():
        cleaned_lines.append(line[4:] if line.startswith("    ") else line)
    return "\n".join(cleaned_lines).strip()


def artifact_image(name: str) -> str:
    return f"../../local_outputs/artifacts/{name}"


def top_rows(frame: pd.DataFrame, columns: list[str], n: int | None = None) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=columns)
    subset = frame.loc[:, [column for column in columns if column in frame.columns]].copy()
    return subset.head(n) if n is not None else subset


def lookup_value(frame: pd.DataFrame, filters: dict[str, object], column: str) -> object:
    if frame.empty or column not in frame.columns:
        return float("nan")
    mask = pd.Series(True, index=frame.index)
    for key, value in filters.items():
        if key not in frame.columns:
            return float("nan")
        if isinstance(value, float):
            mask &= frame[key].astype(float).round(6) == round(value, 6)
        else:
            mask &= frame[key] == value
    subset = frame.loc[mask, column]
    if subset.empty:
        return float("nan")
    return subset.iloc[0]


def build_manuscript() -> None:
    MANUSCRIPT_DIR.mkdir(parents=True, exist_ok=True)

    audit = read_json(ARTIFACT_DIR / "audit_report.json")
    data_contract = read_json(ARTIFACT_DIR / "data_contract.json")
    clinical_policy = read_json(ARTIFACT_DIR / "clinical_policy.json")
    bootstrap_ci = read_json(ARTIFACT_DIR / "bootstrap_intervals.json")
    cluster_bootstrap_ci = read_json(ARTIFACT_DIR / "cluster_bootstrap_intervals.json")
    hpo_summary = read_json(ARTIFACT_DIR / "canonical_hpo_summary.json")
    probe_all = read_json(PROBE_DIR / "probe_all_results.json")

    benchmark_summary = read_csv("benchmark_summary.csv")
    hpo_top_trials = read_csv("canonical_hpo_top_trials.csv")
    score_summary = read_csv("clinical_score_benchmark_summary.csv")
    score_operating = read_csv("clinical_score_operating_points.csv")
    ceiling = read_csv("prediction_ceiling_results.csv")
    calibration_benchmark = read_csv("baseline_calibration_benchmark.csv")
    conformal_single = read_csv("conformal_single_model_results.csv")
    conformal_single_summary = read_csv("conformal_single_model_summary.csv")
    conformal_consensus = read_csv("conformal_consensus_results.csv")
    conformal_consensus_summary = read_csv("conformal_consensus_summary.csv")
    conformal_shift = read_csv("conformal_shift_results.csv")
    fixed_shift = read_csv("fixed_threshold_shift_results.csv")
    conformal_operating = read_csv("conformal_operating_curve.csv")
    conformal_subgroups = read_csv("conformal_subgroup_results.csv")
    disagreement = read_csv("selective_triage_results.csv")
    disagreement_shift = read_csv("selective_triage_shift_results.csv")
    decision_curve_summary = read_csv("decision_curve_summary.csv")
    decision_curve_policy_metrics = read_csv("decision_curve_policy_metrics.csv")
    feature_ablation = read_csv("feature_ablation_results.csv")
    secondary_horizons = read_csv("secondary_horizon_results.csv")

    selected_model = clinical_policy.get("selected_model", "lightgbm")
    selected_calibration = clinical_policy.get("calibration", "sigmoid")
    event_rate = audit.get("event_rate", float("nan"))
    n_rows = audit.get("current_rows", "NA")
    n_cols = audit.get("current_columns", "NA")
    conformal_seed_count = int(conformal_single["seed"].nunique()) if "seed" in conformal_single.columns and not conformal_single.empty else 0
    subgroup_alpha = (
        float(conformal_subgroups["alpha"].iloc[0])
        if not conformal_subgroups.empty and "alpha" in conformal_subgroups.columns
        else float("nan")
    )
    hpo_lightgbm = hpo_summary.get("models", {}).get("lightgbm", {}) if isinstance(hpo_summary, dict) else {}

    selected_auroc = lookup_value(benchmark_summary, {"model_name": selected_model}, "auroc__mean")
    selected_brier = lookup_value(benchmark_summary, {"model_name": selected_model}, "brier__mean")
    selected_recall_ppv50 = lookup_value(
        benchmark_summary, {"model_name": selected_model}, "recall_at_ppv_050__mean"
    )
    lightgbm_auroc = lookup_value(benchmark_summary, {"model_name": "lightgbm"}, "auroc__mean")
    lightgbm_recall_ppv50 = lookup_value(
        benchmark_summary, {"model_name": "lightgbm"}, "recall_at_ppv_050__mean"
    )
    catboost_auroc = lookup_value(benchmark_summary, {"model_name": "catboost"}, "auroc__mean")
    catboost_brier = lookup_value(benchmark_summary, {"model_name": "catboost"}, "brier__mean")
    catboost_recall_ppv50 = lookup_value(
        benchmark_summary, {"model_name": "catboost"}, "recall_at_ppv_050__mean"
    )
    tree_model_aurocs = [
        float(value)
        for value in [selected_auroc, lightgbm_auroc, catboost_auroc]
        if not pd.isna(value)
    ]
    tree_band_high = max(tree_model_aurocs) if tree_model_aurocs else float("nan")
    hpo_trials_count = hpo_lightgbm.get("n_trials", 0) if isinstance(hpo_lightgbm, dict) else 0
    hpo_best_metrics = hpo_lightgbm.get("best_metrics", {}) if isinstance(hpo_lightgbm, dict) else {}
    hpo_val_auroc = hpo_best_metrics.get("auroc", float("nan")) if isinstance(hpo_best_metrics, dict) else float("nan")
    hpo_val_recall_ppv50 = (
        hpo_best_metrics.get("recall_at_ppv_050", float("nan"))
        if isinstance(hpo_best_metrics, dict)
        else float("nan")
    )

    selected_nb_020 = lookup_value(
        decision_curve_summary,
        {"strategy": f"{selected_model}_continuous", "threshold": 0.20},
        "net_benefit",
    )
    sofa_nb_020 = lookup_value(
        decision_curve_summary,
        {"strategy": "sofa_total_24hr_continuous", "threshold": 0.20},
        "net_benefit",
    )
    apache_nb_020 = lookup_value(
        decision_curve_summary,
        {"strategy": "apache_iii_score_continuous", "threshold": 0.20},
        "net_benefit",
    )
    fixed_nb_020 = lookup_value(
        decision_curve_summary, {"strategy": "fixed_threshold_ppv50", "threshold": 0.20}, "net_benefit"
    )
    conformal_nb_020 = lookup_value(
        decision_curve_summary, {"strategy": "conformal_alpha_0.05", "threshold": 0.20}, "net_benefit"
    )
    conformal_alert_rate = lookup_value(
        decision_curve_policy_metrics, {"strategy": "conformal_alpha_0.05"}, "alert_rate"
    )
    conformal_alert_ppv = lookup_value(
        decision_curve_policy_metrics, {"strategy": "conformal_alpha_0.05"}, "alert_precision"
    )
    union_alert_rate = lookup_value(
        decision_curve_policy_metrics, {"strategy": "union_conformal_alpha_0.05"}, "alert_rate"
    )
    secondary_48h_auroc = lookup_value(
        secondary_horizons, {"label": "death_within_48h"}, "auroc"
    )
    secondary_7d_auroc = lookup_value(
        secondary_horizons, {"label": "death_within_7d"}, "auroc"
    )
    secondary_7d_recall = lookup_value(
        secondary_horizons, {"label": "death_within_7d"}, "recall_at_primary_ppv"
    )

    benchmark_main = benchmark_summary[
        benchmark_summary["model_name"].isin(
            [selected_model, "logistic", "lightgbm", "xgboost", "sofa_total_24hr", "apache_iii_score"]
        )
    ].copy() if not benchmark_summary.empty else pd.DataFrame()
    if not benchmark_main.empty:
        benchmark_main = benchmark_main[
            [
                "model_name",
                "auroc__mean",
                "auroc__std",
                "auprc__mean",
                "auprc__std",
                "brier__mean",
                "ece__mean",
                "calibration_intercept__mean",
                "calibration_slope__mean",
                "recall_at_ppv_050__mean",
            ]
        ]

    conformal_table3 = pd.DataFrame()
    if not conformal_single_summary.empty:
        conformal_table3 = top_rows(
            conformal_single_summary,
            [
                "alpha",
                "n_groups",
                "coverage_mean",
                "coverage_ci_low",
                "coverage_ci_high",
                "certain_frac_mean",
                "alert_ppv_mean",
                "clear_npv_mean",
                "miss_count_mean",
            ],
        )
    elif not conformal_single.empty:
        conformal_table3 = (
            conformal_single.groupby("alpha")[
                ["coverage", "certain_frac", "alert_ppv", "clear_npv", "miss_count"]
            ]
            .agg(["mean", "std"])
            .reset_index()
        )
        conformal_table3.columns = [
            "alpha",
            "coverage_mean",
            "coverage_std",
            "certain_frac_mean",
            "certain_frac_std",
            "alert_ppv_mean",
            "alert_ppv_std",
            "clear_npv_mean",
            "clear_npv_std",
            "miss_count_mean",
            "miss_count_std",
        ]

    consensus_table4 = pd.DataFrame()
    if not conformal_consensus_summary.empty:
        consensus_table4 = conformal_consensus_summary[
            (
                (conformal_consensus_summary["ensemble_name"] == "single_model")
                & (conformal_consensus_summary["base_model"] == selected_model)
            )
            | (conformal_consensus_summary["ensemble_name"].isin(["union", "intersection"]))
        ].copy()
        consensus_table4 = top_rows(
            consensus_table4,
            [
                "ensemble_name",
                "base_model",
                "alpha",
                "n_groups",
                "coverage_mean",
                "certain_frac_mean",
                "alert_ppv_mean",
                "clear_npv_mean",
                "miss_count_mean",
            ],
        )
    elif not conformal_consensus.empty:
        consensus_table4 = conformal_consensus[
            (
                (conformal_consensus["ensemble_name"] == "single_model")
                & (conformal_consensus["base_model"] == selected_model)
            )
            | (conformal_consensus["ensemble_name"].isin(["union", "intersection"]))
        ].copy()
        consensus_table4 = (
            consensus_table4.groupby(["ensemble_name", "base_model", "alpha"])[
                ["coverage", "certain_frac", "alert_ppv", "clear_npv", "miss_count"]
            ]
            .mean()
            .reset_index()
        )

    shift_table5 = pd.DataFrame()
    if not conformal_shift.empty:
        alpha_005 = conformal_shift[conformal_shift["alpha"] == 0.05].copy()
        fixed_05 = fixed_shift[fixed_shift["threshold_name"] == "fixed_0.50"].copy()
        shift_table5 = alpha_005.merge(
            fixed_05[["scenario", "severity", "precision", "recall"]],
            on=["scenario", "severity"],
            how="left",
        )
        shift_table5 = shift_table5.rename(
            columns={
                "coverage": "conformal_coverage",
                "certain_frac": "conformal_certain_frac",
                "alert_ppv": "conformal_alert_ppv",
                "clear_npv": "conformal_clear_npv",
                "precision": "fixed_precision_t050",
                "recall": "fixed_recall_t050",
            }
        )
        shift_table5 = shift_table5[
            [
                "scenario",
                "severity",
                "conformal_coverage",
                "conformal_certain_frac",
                "conformal_alert_ppv",
                "conformal_clear_npv",
                "fixed_precision_t050",
                "fixed_recall_t050",
            ]
        ]

    subgroup_table6 = top_rows(
        conformal_subgroups,
        ["subgroup", "value", "n", "event_rate", "coverage", "certain_frac", "alert_ppv", "clear_npv"],
    )

    disagreement_summary = pd.DataFrame()
    if not disagreement.empty:
        disagreement_summary = (
            disagreement.groupby("policy_name")[
                [
                    "alert_precision",
                    "alert_recall",
                    "actionable_coverage",
                    "low_risk_npv",
                    "defer_rate",
                    "actionable_error_rate",
                ]
            ]
            .agg(["mean", "std"])
            .reset_index()
        )
        disagreement_summary.columns = [
            "policy_name",
            "alert_precision_mean",
            "alert_precision_std",
            "alert_recall_mean",
            "alert_recall_std",
            "actionable_coverage_mean",
            "actionable_coverage_std",
            "low_risk_npv_mean",
            "low_risk_npv_std",
            "defer_rate_mean",
            "defer_rate_std",
            "actionable_error_rate_mean",
            "actionable_error_rate_std",
        ]

    best_operating = {}
    if not conformal_operating.empty:
        candidates = conformal_operating[
            (conformal_operating["clear_npv"].fillna(0) >= 0.90)
            & (conformal_operating["alert_ppv"].fillna(0) >= 0.55)
        ]
        if not candidates.empty:
            best_operating = candidates.sort_values("certain_frac", ascending=False).iloc[0].to_dict()

    best_union_005 = {}
    if not consensus_table4.empty:
        union_005 = consensus_table4[
            (consensus_table4["ensemble_name"] == "union") & (consensus_table4["alpha"] == 0.05)
        ]
        if not union_005.empty:
            best_union_005 = union_005.iloc[0].to_dict()

    utility_focus = pd.DataFrame()
    if not decision_curve_summary.empty:
        utility_focus = decision_curve_summary[
            decision_curve_summary["strategy"].isin(
                [
                    f"{selected_model}_continuous",
                    "apache_iii_score_continuous",
                    "sofa_total_24hr_continuous",
                    "fixed_threshold_ppv50",
                    "disagreement_selective",
                    "conformal_alpha_0.05",
                    "union_conformal_alpha_0.05",
                ]
            )
        ].copy()
        utility_focus = top_rows(
            utility_focus.sort_values(["threshold", "strategy"]),
            ["threshold", "strategy", "net_benefit", "standardized_net_benefit", "alert_rate"],
            n=None,
        )

    utility_policy_table = top_rows(
        decision_curve_policy_metrics,
        [
            "strategy",
            "alert_rate",
            "alert_precision",
            "alert_recall",
            "net_benefit_at_0.10",
            "net_benefit_at_0.20",
            "net_benefit_at_0.30",
        ],
        n=None,
    )

    hpo_overview_table = pd.DataFrame()
    if hpo_lightgbm:
        hpo_overview_table = pd.DataFrame(
            [
                {
                    "model_name": "lightgbm",
                    "n_trials": hpo_trials_count,
                    "validation_auroc": hpo_val_auroc,
                    "validation_recall_at_ppv_050": hpo_val_recall_ppv50,
                    "validation_brier": hpo_best_metrics.get("brier", float("nan")),
                    "validation_ece": hpo_best_metrics.get("ece", float("nan")),
                }
            ]
        )

    hpo_trials_table = top_rows(
        hpo_top_trials,
        [
            "model_name",
            "trial_number",
            "objective_value",
            "auroc",
            "brier",
            "recall_at_ppv_050",
            "param_n_estimators",
            "param_learning_rate",
            "param_num_leaves",
            "param_max_depth",
            "param_min_child_samples",
        ],
        n=5,
    )

    diversity_benchmark_table = top_rows(
        benchmark_summary[
            benchmark_summary["model_name"].isin(["lightgbm", "xgboost", "catboost"])
        ].copy()
        if not benchmark_summary.empty
        else pd.DataFrame(),
        [
            "model_name",
            "auroc__mean",
            "brier__mean",
            "ece__mean",
            "recall_at_ppv_050__mean",
        ],
        n=None,
    )

    table1 = pd.DataFrame(
        [
            {"Item": "Working cohort", "Value": data_contract.get("dataset_name", "mimic_saaki_raw_v2.csv")},
            {"Item": "Rows", "Value": n_rows},
            {"Item": "Columns", "Value": n_cols},
            {"Item": "Outcome prevalence", "Value": fmt(event_rate)},
            {"Item": "Prediction time", "Value": data_contract.get("prediction_time", "T24")},
            {"Item": "Outcome", "Value": data_contract.get("target_definition", "`event_observed`")},
            {"Item": "Evaluation", "Value": "Subject-grouped holdout by `subject_id`"},
            {"Item": "Excluded claims", "Value": "; ".join(data_contract.get("excluded_claims", []))},
        ]
    )

    title = "When to Alert, When to Defer: Conformal Selective Triage for ICU Mortality in Sepsis-Associated AKI"
    running_title = "Conformal selective triage for SA-AKI mortality"

    abstract_md = f"""
    # Title Page and Abstract

    ## Title

    {title}

    ## Running Title

    {running_title}

    ## Structured Abstract

    ### Objective

    To develop and evaluate an uncertainty-aware triage framework for early in-hospital mortality prediction in sepsis-associated acute kidney injury (SA-AKI), with primary emphasis on safe automation boundaries rather than raw discrimination alone.

    ### Materials and Methods

    We analyzed `{n_rows}` ICU-stay-level SA-AKI cohort records from `{data_contract.get("dataset_name", "mimic_saaki_raw_v2.csv")}` scored at 24 hours after ICU admission, with event prevalence `{fmt(event_rate)}`. All deployment claims used subject-grouped train/validation/test splits by `subject_id`. Conventional baselines included logistic regression, XGBoost, a grouped-Optuna-tuned LightGBM benchmark (`{hpo_trials_count}` trials), an exploratory CatBoost diversity comparator, and score-only baselines. Calibration was selected on grouped validation data. We evaluated disagreement-based selective triage as an ablation and Mondrian conformal selective triage as the main method. We summarized coverage, alert positive predictive value (PPV), clear negative predictive value (NPV), calibration, decision-curve utility, subgroup heterogeneity, and robustness under simulated distribution shift, with conformal headline estimates repeated across `{conformal_seed_count}` grouped seeds.

    ### Results

    The best conventional discriminative baseline was `{selected_model}` with `{selected_calibration}` calibration (mean AUROC `{fmt(selected_auroc)}`; recall at `PPV >= 0.50` `{fmt(selected_recall_ppv50)}`). Grouped Optuna improved LightGBM validation recall at `PPV >= 0.50` to `{fmt(hpo_val_recall_ppv50)}`, but repeated grouped benchmarking still preserved only a narrow tree-model performance band. The prediction ceiling remained limited: even when `time_to_event_hrs` was added as a leakage-like feature, AUROC reached only `{fmt(ceiling.loc[ceiling['setting'] == 'with_time_to_event', 'auroc'].iloc[0] if not ceiling.empty and (ceiling['setting'] == 'with_time_to_event').any() else float('nan'))}`. In repeated subject-grouped evaluation across `{conformal_seed_count}` seeds, single-model conformal triage achieved approximately `{fmt(conformal_table3.loc[conformal_table3['alpha'] == 0.05, 'coverage_mean'].iloc[0] if not conformal_table3.empty and (conformal_table3['alpha'] == 0.05).any() else float('nan'))}` coverage at `alpha=0.05` while retaining a clinically meaningful defer region. The multi-model union consensus at `alpha=0.05` concentrated decisions into a smaller, more reliable actionable region with alert PPV `{fmt(best_union_005.get('alert_ppv_mean', best_union_005.get('alert_ppv', float('nan'))))}` and clear NPV `{fmt(best_union_005.get('clear_npv_mean', best_union_005.get('clear_npv', float('nan'))))}`. At a decision threshold of `0.20`, the selected continuous model achieved net benefit `{fmt(selected_nb_020)}` versus `{fmt(apache_nb_020)}` for `APACHE-III` and `{fmt(sofa_nb_020)}` for `SOFA`. Under added missingness shift, conformal coverage remained stable while fixed thresholds lost recall.

    ### Discussion

    The main contribution is not a new AUROC leader; it is a deployment-oriented Alert/Defer/Clear framework that quantifies when the model should defer. The disagreement-based policy remains informative as an ablation, but conformal selective triage is more rigorous because it supplies finite-sample coverage guarantees and more graceful degradation under shift.

    ### Conclusion

    For this SA-AKI cohort, the main scientific opportunity lies in safe decision-making under bounded predictive signal. Conformal selective triage offers a realistic uncertainty-aware deployment framing for ICU mortality prediction and is a strong fit for a clinical informatics journal such as `JAMIA`.
    """

    introduction_md = f"""
    # Introduction

    Predicting mortality in critically ill patients with sepsis-associated acute kidney injury (SA-AKI) is clinically important, but retrospective machine learning studies often stop at AUROC and overstate deployment readiness. In practice, clinicians need to know not only how well a model discriminates on average, but also when it should escalate care, when it can safely clear a patient from immediate concern, and when it should defer because the case is too uncertain for automation.

    This project starts from a hard empirical observation: discrimination in this cohort is bounded. The current dataset contains `{n_rows}` T24 SA-AKI cohort rows with outcome prevalence `{fmt(event_rate)}`, and the strongest conventional models occupy a narrow performance band. The evidence package now includes grouped Optuna tuning for LightGBM and an added CatBoost diversity comparator, yet the repeated grouped benchmark still clusters the leading tree models tightly around AUROC `{fmt(selected_auroc)}` to `{fmt(tree_band_high)}` with similar Brier scores. Even a leak-prone ceiling experiment that includes `time_to_event_hrs` reaches only modestly higher discrimination. That finding changes the research objective. The key question is no longer how to squeeze another 0.01 of AUROC from the same first-24-hour features; it is how to build a clinically legible triage policy that behaves safely under bounded signal.

    Two uncertainty-aware strategies are therefore compared. The first is a disagreement-based selective-triage baseline that treats divergence between a physiology-only model and a process-enriched model as a signal to defer. The second, and primary contribution of this paper, is conformal selective triage. Conformal prediction converts probabilistic risk scores into set-valued decisions with finite-sample coverage guarantees. This naturally supports a three-way clinical interface: `Alert`, `Clear`, or `Defer`.

    The manuscript makes three contributions. First, it documents the prediction ceiling for first-24-hour SA-AKI mortality risk in a subject-grouped evaluation design. Second, it introduces conformal selective triage as the main decision framework and shows that it provides explicit safety-oriented operating points in terms of coverage, alert PPV, and clear NPV. Third, it demonstrates that multi-model conformal consensus and conformal deferral degrade more gracefully than fixed thresholds under missingness and process-feature perturbations.

    The goal is not to claim bedside readiness. Instead, the goal is to present a clinically grounded informatics study that defines a more realistic deployment boundary for ICU mortality risk models.
    """

    methods_md = f"""
    # Methods

    ## Study Design And Data Source

    This retrospective study used `{data_contract.get("dataset_name", "mimic_saaki_raw_v2.csv")}`, a T24 SA-AKI cohort derived from MIMIC. One row represents one ICU-stay-level SA-AKI record scored at 24 hours after ICU admission. The primary label is in-hospital mortality (`event_observed`), and the time-to-event variable is `time_to_event_hrs`. Because repeated patients remain in the cohort, all deployment claims use subject-grouped evaluation by `subject_id`.

    ## Cohort And Evaluation Workflow

    **Figure 1. Cohort definition and evaluation workflow.**

    ```mermaid
    flowchart LR
        rawCohort["Raw SA-AKI Cohort"] --> auditContract["Audit And Data Contract"]
        auditContract --> featureSets["Feature Set Construction"]
        featureSets --> groupedSplit["Subject-Grouped Train/Validation/Test Split"]
        groupedSplit --> baselineModels["Baseline Models And Clinical Score Comparators"]
        groupedSplit --> disagreementPolicy["Disagreement Selective Triage"]
        groupedSplit --> conformalPolicy["Conformal Selective Triage"]
        baselineModels --> manuscriptAssets["Tables, Figures, And Manuscript Assets"]
        disagreementPolicy --> manuscriptAssets
        conformalPolicy --> manuscriptAssets
    ```

    ## Baselines

    Conventional baselines included logistic regression, LightGBM, and XGBoost. Calibration was selected on grouped validation data by benchmarking `sigmoid` and `isotonic` recalibration on the held-out grouped validation split. We also retained score-only comparators based on `SOFA` and `APACHE-III` if those columns were present, treating them as single-feature logistic baselines under the same grouped protocol.

    To close the remaining methodological rigor gap from earlier drafts, we added a grouped Optuna search for the main LightGBM benchmark. The search used `{hpo_trials_count}` trials and optimized a deployment-oriented objective that combined AUROC, recall at `PPV >= 0.50`, and Brier score on grouped validation data. The tuned LightGBM configuration was then frozen and carried into the repeated grouped benchmark. We also added CatBoost as a benchmark-only diversity comparator to test whether a distinct gradient-boosting family materially changed the model-selection story.

    The disagreement-based selective-triage comparator follows the earlier project draft. A process-enriched full model and a physiology-severity model are trained on the same grouped split. Cases are actionable only when the two models agree closely enough on grouped validation data.

    ## Conformal Selective Triage

    The main method is Mondrian conformal prediction. A training subset fits the risk model, a separate calibration subset estimates nonconformity thresholds, and the test subset receives set-valued predictions. For binary mortality prediction, the prediction set can be one of three clinically interpretable outcomes:

    - `{{1}}`: Alert
    - `{{0}}`: Clear
    - `{{0,1}}`: Defer

    **Figure 3. Conformal selective-triage decision flow.**

    ```mermaid
    flowchart TD
        patientCase["Patient At T24"] --> riskModel["Risk Model"]
        riskModel --> calibrationThresholds["Mondrian Conformal Thresholds"]
        calibrationThresholds --> predSet["Prediction Set"]
        predSet -->|"{{1}}" alertNode["Alert"]
        predSet -->|"{{0}}" clearNode["Clear"]
        predSet -->|"{{0,1}}" deferNode["Defer"]
    ```

    ## Disagreement Baseline

    **Figure 4. Disagreement-based selective-triage comparator.**

    ```mermaid
    flowchart TD
        patientCase["Patient At T24"] --> fullModel["Full Feature Model"]
        patientCase --> physModel["Physiology-Only Model"]
        fullModel --> fullRisk["pFull"]
        physModel --> physRisk["pPhys"]
        fullRisk --> gapNode["Disagreement = |pFull - pPhys|"]
        physRisk --> gapNode
        gapNode --> actionGate["Agreement Threshold And Risk Thresholds"]
        actionGate --> alertNode["Alert"]
        actionGate --> clearNode["Clear"]
        actionGate --> deferNode["Defer"]
    ```

    ## Metrics And Statistical Analysis

    We reported AUROC, AUPRC, Brier score, expected calibration error, calibration intercept, calibration slope, recall at `PPV >= 0.50`, alert burden, and low-risk coverage. For conformal selective triage we reported coverage, certain-decision fraction, defer rate, alert PPV, clear NPV, and miss count. Subject-aware uncertainty was summarized with clustered bootstrap confidence intervals, and headline conformal estimates were repeated across `{conformal_seed_count}` grouped seeds. Shift robustness was assessed by random missingness injection and by dropout of measurement-process and care-process feature families. Clinical utility was summarized with decision-curve analysis against `APACHE-III`, `SOFA`, treat-all, treat-none, and fixed-policy comparators. The primary decision-curve interpretation focused on the continuous selected model versus clinical scores, while conformal and disagreement curves were treated as deployment-policy complements rather than direct substitutes for continuous ranking. Subgroup conformal summaries are reported at `alpha={fmt(subgroup_alpha, 2)}` to balance coverage with enough actionable volume for stable subgroup interpretation. Secondary horizon performance was also exported for the journal package.

    ## Ethics, Data Access, And Reproducibility

    This study is a retrospective secondary analysis of a de-identified MIMIC-derived cohort and does not involve prospective intervention or direct patient contact. The repository distributes code, derived artifact summaries, and manuscript assets rather than raw source patient data. Reproducing the cohort from source requires independent credentialed access to MIMIC and the associated data-use approvals. All manuscript claims are therefore limited to internal validation on the checked-in derivative cohort and should not be read as transportability or bedside-readiness claims.
    """

    results_md = f"""
    # Results

    ## Cohort And Data Contract

    Table 1 summarizes the working cohort and the constraints that bound the manuscript claims.

    **Table 1. Cohort and data-contract summary.**

    {render_table(table1, index=False)}

    ## Prediction Ceiling And Conventional Benchmarks

    Figure 2 and Table 2 anchor the benchmark story. The main result is that the discrimination ceiling is bounded and the baseline model family differences are comparatively small.

    **Figure 2. Prediction ceiling and benchmark summary.**

    ![Figure 2. Prediction ceiling]({artifact_image("prediction_ceiling.png")})

    **Table 2. Baseline discrimination and calibration benchmark.**

    {render_table(top_rows(benchmark_main, benchmark_main.columns.tolist(), n=None), index=False)}

    **Figure 7. Calibration and clinical-utility panel.**

    ![Figure 7. Calibration and clinical utility]({artifact_image("clinical_utility_panel.png")})

    The ceiling experiment is especially important. In the canonical workflow, the leak-prone configuration that adds `time_to_event_hrs` reaches AUROC `{fmt(ceiling.loc[ceiling['setting'] == 'with_time_to_event', 'auroc'].iloc[0] if not ceiling.empty and (ceiling['setting'] == 'with_time_to_event').any() else float('nan'))}`, while the honest grouped setting remains materially lower. The repeated grouped benchmark selected `{selected_model}` because it combined mean AUROC `{fmt(selected_auroc)}`, Brier `{fmt(selected_brier)}`, and recall at `PPV >= 0.50` `{fmt(selected_recall_ppv50)}` more consistently than the competing primary baselines. The grouped Optuna pass improved LightGBM validation recall at `PPV >= 0.50` to `{fmt(hpo_val_recall_ppv50)}`, but repeated grouped LightGBM still averaged AUROC `{fmt(lightgbm_auroc)}` and recall `{fmt(lightgbm_recall_ppv50)}`. The exploratory CatBoost comparator remained competitive (AUROC `{fmt(catboost_auroc)}`, Brier `{fmt(catboost_brier)}`, recall `{fmt(catboost_recall_ppv50)}`) but did not materially simplify or improve the downstream conformal story, so it stayed in the appendix as a diversity check rather than replacing the primary model family.

    The clinical-utility panel complements that story. At decision threshold `0.20`, the continuous `{selected_model}` score yields net benefit `{fmt(selected_nb_020)}` versus `{fmt(apache_nb_020)}` for `APACHE-III`, `{fmt(sofa_nb_020)}` for `SOFA`, and `{fmt(fixed_nb_020)}` for the fixed-threshold `PPV >= 0.50` policy. The conformal `alpha=0.05` policy is deliberately more conservative, with net benefit `{fmt(conformal_nb_020)}`, alert rate `{fmt(conformal_alert_rate)}`, and alert PPV `{fmt(conformal_alert_ppv)}`. This distinction matters: the continuous model is the main utility comparison against clinical scores, whereas the conformal and disagreement policies are complementary deployment policies that trade some net benefit for narrower, higher-confidence action sets.

    ## Main Conformal Triage Result

    Table 3 summarizes repeated subject-grouped single-model conformal triage results. The main manuscript operating point remains the low-alpha regime, where coverage is controlled while a clinically meaningful defer region is preserved.

    **Table 3. Main conformal results across repeated grouped splits.**

    {render_table(conformal_table3, index=False)}

    **Figure 5. Conformal operating-characteristic curve.**

    ![Figure 5. Conformal operating curve]({artifact_image("conformal_operating_curve.png")})

    Under the manuscript sweet-spot criterion (`NPV >= 0.90` and `PPV >= 0.55`), the best operating point occurs at `alpha={fmt(best_operating.get('alpha', float('nan')), 2)}` with certain-decision fraction `{fmt(best_operating.get('certain_frac', float('nan')))}`, alert PPV `{fmt(best_operating.get('alert_ppv', float('nan')))}`, and clear NPV `{fmt(best_operating.get('clear_npv', float('nan')))}`. The repeated-seed summary in Table 3 shows that these headline estimates remain stable across `{conformal_seed_count}` grouped splits.

    ## Multi-Model Consensus

    Table 4 and Figure 6 show the consensus extension. The union of conformal prediction sets is deliberately conservative: it acts on fewer patients but produces a more reliable actionable region.

    **Table 4. Multi-model consensus conformal results.**

    {render_table(consensus_table4, index=False)}

    **Figure 6. Multi-model consensus trade-off panel.**

    ![Figure 6. Consensus trade-off]({artifact_image("conformal_consensus_tradeoff.png")})

    At `alpha=0.05`, the union consensus yields alert PPV `{fmt(best_union_005.get('alert_ppv_mean', best_union_005.get('alert_ppv', float('nan'))))}` and clear NPV `{fmt(best_union_005.get('clear_npv_mean', best_union_005.get('clear_npv', float('nan'))))}`, confirming that consensus amplifies reliability at the cost of automation rate.

    ## Comparison With Disagreement-Based Selective Triage

    The disagreement-based policy remains an important ablation because it shows that explicit deferral helps, but its guarantees are empirical and heuristic rather than finite-sample. The repeated grouped summary is included in Appendix B. In brief, disagreement triage narrows the actionable region and reduces actionable error, but conformal triage supplies the stronger methodological framing because it binds that defer region to a formal coverage objective.

    ## Robustness Under Distribution Shift

    **Table 5. Shift robustness versus fixed-threshold policies.**

    {render_table(shift_table5, index=False)}

    **Figure 8. Robustness-under-shift panel.**

    ![Figure 8. Shift robustness]({artifact_image("conformal_shift_panel.png")})

    The shift experiments support the deployment argument. As extra missingness is injected, conformal coverage remains stable because the framework defers more cases. By contrast, fixed thresholds lose recall while offering no explicit warning that uncertainty has increased.

    ## Subgroup Heterogeneity

    **Table 6. Subgroup heterogeneity summary for conformal triage at alpha={fmt(subgroup_alpha, 2)}.**

    {render_table(subgroup_table6, index=False)}

    **Figure 9. Subgroup coverage and reliability forest plot.**

    ![Figure 9. Subgroup forest plot]({artifact_image("conformal_subgroup_forest.png")})

    Coverage and clear NPV remain high across most demographic subgroups, but high-acuity strata are predictably harder. Reporting the subgroup summaries at `alpha={fmt(subgroup_alpha, 2)}` exposes that trade-off explicitly: low-SOFA strata preserve high coverage with a reasonable action rate, whereas high-SOFA strata require a larger defer region. This reinforces the paper's deployment message that the model should be treated as a triage assistant with a defer option, not as a universal autonomous classifier.
    """

    discussion_md = f"""
    # Discussion

    This study argues that the central challenge in SA-AKI mortality modeling is not how to win a retrospective AUROC leaderboard; it is how to construct a clinically legible decision policy when the predictive ceiling is bounded. The canonical benchmark results, the grouped Optuna pass, and the added CatBoost diversity comparator all point in the same direction. Tree models remain competitive, but their gains over one another are modest and unstable enough that the real manuscript contribution comes from making uncertainty explicit rather than from claiming a dramatic architecture breakthrough.

    Conformal selective triage addresses that problem directly. Instead of forcing every patient into a binary high-risk versus low-risk decision, it partitions the cohort into `Alert`, `Clear`, and `Defer`. That defer region is not a nuisance artifact; it is the mechanism that protects coverage. In this dataset, the defer option is exactly what allows the model to preserve reliable alert PPV and clear NPV under both clean evaluation and shift stress.

    The disagreement-based comparator remains useful because it demonstrates that uncertainty-aware action restriction helps even before formal conformalization. However, the disagreement policy depends on an empirically tuned agreement threshold and is harder to justify theoretically. The conformal formulation is stronger for a journal paper because it is more principled, more general, and more transparent about what is and is not guaranteed.

    From a translational standpoint, the manuscript supports a narrow but realistic deployment framing: retrospective workflow design, threshold governance, and safe automation boundaries for ICU triage. The decision-curve results support the continuous `{selected_model}` score as the primary clinical-score comparator, while the conformal and disagreement policies should be read as downstream action-governance layers rather than as replacements for continuous ranking. The manuscript therefore supports threshold governance and workflow simulation, not bedside deployment or broad generalization claims, and it should remain explicit about that boundary.
    """

    limitations_md = f"""
    # Limitations

    This work has several important limitations.

    1. The study remains internally validated only. The current checked-in cohort does not expose a defensible calendar timestamp axis for a true temporal split, and no external cohort is included, so the manuscript supports strong internal claims and cautious deployment framing rather than transportability claims.
    2. The cohort remains affected by ETL and documentation mismatches already documented in the audit and data-contract artifacts.
    3. Repeated patients remain present in the working cohort, which is why subject-grouped evaluation is mandatory.
    4. The process-rich feature space may not transport as well as physiology-only inputs.
    5. The conformal guarantees are marginal under exchangeability and do not imply perfect subgroup-conditional coverage.
    6. The secondary horizon results, especially the 48-hour endpoint, are limited by low event counts and should be treated as sensitivity analyses rather than standalone deployment targets.
    7. The manuscript focuses on retrospective decision support and does not evaluate clinical workflow adoption, clinician behavior change, or prospective impact.
    8. The disagreement-based baseline is heuristic by design and is included as a comparison, not as the final methodological recommendation.
    """

    conclusion_md = """
    # Conclusion

    In this SA-AKI cohort, the predictive ceiling for first-24-hour ICU mortality risk appears bounded, and that changes the scientific target. The most defensible contribution is not a marginal improvement in discrimination but an uncertainty-aware triage policy that states when the model should act and when it should defer.

    Conformal selective triage provides that framing. It yields clinically interpretable Alert/Defer/Clear decisions, finite-sample coverage guarantees, stronger reliability under conservative consensus, and more graceful degradation under shift than fixed-threshold rules. For a clinical informatics venue such as `JAMIA`, this is the right story: rigorous internal validation, explicit uncertainty handling, and a careful boundary around what retrospective machine learning can responsibly claim.
    """

    appendix_a_md = f"""
    # Appendix A: Data And Cohort

    ## Audit Summary

    - Working dataset: `{data_contract.get("dataset_name", "mimic_saaki_raw_v2.csv")}`
    - Rows: `{n_rows}`
    - Columns: `{n_cols}`
    - Unique subjects: `{audit.get('unique_subjects', 'NA')}`
    - Repeated subject rows: `{audit.get('repeated_subject_rows', 'NA')}`
    - Event prevalence: `{fmt(event_rate)}`

    ## Data Contract Notes

    - Row definition: {data_contract.get("row_definition", "One ICU-stay-level SA-AKI row scored at T24.")}
    - Prediction time: {data_contract.get("prediction_time", "24 hours after ICU admission.")}
    - Target definition: {data_contract.get("target_definition", "`event_observed`.")}
    - Time anchor: {data_contract.get("time_anchor_definition", "T24 deployment framing.")}

    ## Explicitly Excluded Claims

    {chr(10).join(f"- {item}" for item in data_contract.get("excluded_claims", []))}
    """

    negative_probe_rows = []
    if probe_all:
        for key in ["feat_missingness", "feat_ratios", "ensemble_stack_3", "hpo_lgbm", "survival_rsf"]:
            if key in probe_all:
                record = probe_all[key]
                negative_probe_rows.append(
                    {
                        "probe": key,
                        "auroc": record.get("auroc"),
                        "auprc": record.get("auprc"),
                        "recall@ppv50": record.get("recall@ppv50") or record.get("recall_at_ppv050"),
                    }
                )
    negative_probe_table = pd.DataFrame(negative_probe_rows)

    appendix_b_md = f"""
    # Appendix B: Extended Benchmarks

    ## Grouped Tuning And Diversity Benchmarks

    **Appendix Table B1. Repeated grouped benchmark summary for the main tree-family comparison.**

    {render_table(diversity_benchmark_table, index=False)}

    **Appendix Table B2. Grouped Optuna summary for the LightGBM benchmark.**

    {render_table(hpo_overview_table, index=False)}

    **Appendix Table B3. Top grouped Optuna LightGBM trials.**

    {render_table(hpo_trials_table, index=False)}

    The grouped Optuna search improved the LightGBM validation operating point (`recall@PPV0.50 = {fmt(hpo_val_recall_ppv50)}`), but repeated grouped benchmarking still left XGBoost, tuned LightGBM, and CatBoost in a narrow performance band. CatBoost therefore remains an informative diversity check rather than a reason to rewrite the main conformal benchmark story.

    ## Feature Ablation

    **Appendix Table B4. Feature ablation of the selected model.**

    {render_table(feature_ablation, index=False)}

    ## Secondary Horizons

    **Appendix Table B5. Secondary horizon sensitivity analyses.**

    {render_table(secondary_horizons, index=False)}

    The secondary horizons are intentionally kept in the supplement. The 48-hour endpoint has prevalence `{fmt(lookup_value(secondary_horizons, {"label": "death_within_48h"}, "prevalence"))}` and produced AUROC `{fmt(secondary_48h_auroc)}` with no usable `PPV >= 0.50` operating point. The 7-day endpoint is more stable (AUROC `{fmt(secondary_7d_auroc)}`) but retains low recall at the conservative primary precision target (`{fmt(secondary_7d_recall)}`), so it is treated as supporting sensitivity evidence rather than a second main result.

    ## ROC And Precision-Recall Benchmarks

    **Appendix Figure B1. ROC and PR benchmark figure.**

    ![Appendix Figure B1. ROC and PR benchmarks]({artifact_image("roc_pr_benchmarks.png")})

    **Appendix Figure B2. Calibration curve for the deployed baseline.**

    ![Appendix Figure B2. Calibration curve]({artifact_image("calibration_curve.png")})

    **Appendix Figure B3. Decision curve for benchmark and fixed-policy comparators.**

    ![Appendix Figure B3. Decision curve]({artifact_image("decision_curve.png")})

    ## Disagreement-Based Selective Triage Summary

    **Appendix Table B6. Repeated grouped disagreement-based selective-triage summary.**

    {render_table(disagreement_summary, index=False)}

    ## Disagreement Shift Summary

    **Appendix Table B7. Disagreement-based shift sensitivity summary.**

    {render_table(disagreement_shift, index=False)}

    ## Clinical Score Summary

    **Appendix Table B8. Repeated grouped clinical score summary.**

    {render_table(score_summary, index=False)}

    ## Clinical Score Operating Points

    **Appendix Table B9. Clinical score operating points on the main grouped split.**

    {render_table(score_operating, index=False)}

    ## Clinical Utility Summary At Key Thresholds

    **Appendix Table B10. Key decision-curve net-benefit comparisons.**

    {render_table(utility_focus, index=False)}

    ## Fixed-Policy Clinical Utility Summary

    **Appendix Table B11. Fixed-policy clinical utility summary.**

    {render_table(utility_policy_table, index=False)}

    The continuous `{selected_model}` score is the primary clinical-score comparator because it retains higher net benefit against `SOFA` and `APACHE-III` across the main threshold range. The conformal and disagreement rows above should instead be read as action-restriction policies: they alert on fewer patients (`alpha=0.05` conformal alert rate `{fmt(conformal_alert_rate)}`, union alert rate `{fmt(union_alert_rate)}`) in exchange for higher alert precision and a larger defer region.

    ## Negative And Null Results From Probe Experiments

    **Appendix Table B12. Negative and null probe results carried forward for transparency.**

    {render_table(negative_probe_table, index=False)}
    """

    appendix_c_md = f"""
    # Appendix C: Shift And Subgroup Results

    ## Full Conformal Shift Table

    {render_table(conformal_shift, index=False)}

    ## Full Fixed-Threshold Shift Table

    {render_table(fixed_shift, index=False)}

    ## Full Conformal Subgroup Table

    {render_table(conformal_subgroups, index=False)}
    """

    appendix_d_md = f"""
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

    - Row-bootstrap AUROC CI: `{fmt(bootstrap_ci.get('auroc', [float('nan'), float('nan')])[0])}` to `{fmt(bootstrap_ci.get('auroc', [float('nan'), float('nan')])[1])}`
    - Cluster-bootstrap AUROC CI: `{fmt(cluster_bootstrap_ci.get('auroc', [float('nan'), float('nan')])[0])}` to `{fmt(cluster_bootstrap_ci.get('auroc', [float('nan'), float('nan')])[1])}`
    - Headline conformal summaries are repeated across `{conformal_seed_count}` grouped seeds.

    ## Selected Model Snapshot

    - Selected model: `{selected_model}` with `{selected_calibration}` calibration.
    - Mean AUROC: `{fmt(selected_auroc)}`.
    - Mean recall at `PPV >= 0.50`: `{fmt(selected_recall_ppv50)}`.
    - Decision-curve net benefit at threshold `0.20`: `{fmt(selected_nb_020)}`.

    ## Data Availability

    The source patient-level data are not redistributed with this repository. Reproducing the cohort from raw source requires independent credentialed access to MIMIC and the corresponding data-use approvals. The checked-in repository provides the analysis code, derived artifact summaries, and manuscript assets used to support the internal-validation claims in this package.

    ## Journal Positioning

    The manuscript is written for `JAMIA`, with `npj Digital Medicine` and `Communications Medicine` treated as future stretch targets after stronger temporal or external validation.
    """

    submission_delta_md = """
    # Submission Delta List

    ## Current Best Realistic Target: JAMIA

    - The current package is already shaped for `JAMIA`: it has a strong informatics framing, explicit deployment boundaries, rigorous internal evaluation, and an uncertainty-aware decision-support story.

    ## To Strengthen For Artificial Intelligence in Medicine

    - Expand the methodological appendix around conformal prediction and clustered uncertainty intervals.
    - Add a clearer side-by-side comparison between disagreement triage and conformal triage.
    - Add more compact sensitivity tables to foreground the methodological message.

    ## To Stretch Toward Communications Medicine

    - Add temporal validation or a more clearly separated validation cohort.
    - Strengthen the translational section on workflow integration and clinical utility.
    - Add a more explicit clinical comparator narrative.

    ## To Stretch Toward npj Digital Medicine

    - Add temporal or external validation.
    - Add stronger generalizability evidence and transportability analysis.
    - Add a broader clinical utility section with a more implementation-facing framing.
    - Tighten the discussion around what the defer region means operationally in a deployed service.
    """

    figure_manifest = pd.DataFrame(
        [
            {"Figure": "Figure 1", "Caption": "Cohort definition and evaluation workflow", "Source": "02_methods.md (mermaid)", "Path": "saaki/jamia_manuscript/02_methods.md"},
            {"Figure": "Figure 2", "Caption": "Prediction ceiling and benchmark summary", "Source": "prediction_ceiling.png", "Path": "local_outputs/artifacts/prediction_ceiling.png"},
            {"Figure": "Figure 3", "Caption": "Conformal selective-triage decision framework", "Source": "02_methods.md (mermaid)", "Path": "saaki/jamia_manuscript/02_methods.md"},
            {"Figure": "Figure 4", "Caption": "Disagreement-based selective-triage comparator", "Source": "02_methods.md (mermaid)", "Path": "saaki/jamia_manuscript/02_methods.md"},
            {"Figure": "Figure 5", "Caption": "Conformal operating-characteristic curve", "Source": "conformal_operating_curve.png", "Path": "local_outputs/artifacts/conformal_operating_curve.png"},
            {"Figure": "Figure 6", "Caption": "Multi-model conformal consensus trade-off panel", "Source": "conformal_consensus_tradeoff.png", "Path": "local_outputs/artifacts/conformal_consensus_tradeoff.png"},
            {"Figure": "Figure 7", "Caption": "Calibration and clinical-utility panel", "Source": "clinical_utility_panel.png", "Path": "local_outputs/artifacts/clinical_utility_panel.png"},
            {"Figure": "Figure 8", "Caption": "Robustness-under-shift panel", "Source": "conformal_shift_panel.png", "Path": "local_outputs/artifacts/conformal_shift_panel.png"},
            {"Figure": "Figure 9", "Caption": "Subgroup coverage and reliability forest plot", "Source": "conformal_subgroup_forest.png", "Path": "local_outputs/artifacts/conformal_subgroup_forest.png"},
            {"Figure": "Appendix Figure B1", "Caption": "ROC and PR benchmark figure", "Source": "roc_pr_benchmarks.png", "Path": "local_outputs/artifacts/roc_pr_benchmarks.png"},
            {"Figure": "Appendix Figure B2", "Caption": "Calibration curve for the deployed baseline", "Source": "calibration_curve.png", "Path": "local_outputs/artifacts/calibration_curve.png"},
            {"Figure": "Appendix Figure B3", "Caption": "Decision curve for benchmark and fixed-policy comparators", "Source": "decision_curve.png", "Path": "local_outputs/artifacts/decision_curve.png"},
        ]
    )

    table_manifest = pd.DataFrame(
        [
            {"Table": "Table 1", "Caption": "Cohort and data-contract summary", "Source": "build_jamia_manuscript.py", "Section": "03_results.md"},
            {"Table": "Table 2", "Caption": "Baseline discrimination and calibration benchmark", "Source": "benchmark_summary.csv", "Section": "03_results.md"},
            {"Table": "Table 3", "Caption": "Main conformal results across repeated grouped splits", "Source": "conformal_single_model_results.csv", "Section": "03_results.md"},
            {"Table": "Table 4", "Caption": "Multi-model consensus conformal results", "Source": "conformal_consensus_results.csv", "Section": "03_results.md"},
            {"Table": "Table 5", "Caption": "Shift robustness versus fixed-threshold policies", "Source": "conformal_shift_results.csv + fixed_threshold_shift_results.csv", "Section": "03_results.md"},
            {"Table": "Table 6", "Caption": "Subgroup heterogeneity summary at alpha=0.10", "Source": "conformal_subgroup_results.csv", "Section": "03_results.md"},
            {"Table": "Appendix Table B1", "Caption": "Repeated grouped benchmark summary for the main tree-family comparison", "Source": "benchmark_summary.csv", "Section": "appendix_b_extended_benchmarks.md"},
            {"Table": "Appendix Table B2", "Caption": "Grouped Optuna summary for the LightGBM benchmark", "Source": "canonical_hpo_summary.json", "Section": "appendix_b_extended_benchmarks.md"},
            {"Table": "Appendix Table B3", "Caption": "Top grouped Optuna LightGBM trials", "Source": "canonical_hpo_top_trials.csv", "Section": "appendix_b_extended_benchmarks.md"},
            {"Table": "Appendix Table B4", "Caption": "Feature ablation of the selected model", "Source": "feature_ablation_results.csv", "Section": "appendix_b_extended_benchmarks.md"},
            {"Table": "Appendix Table B5", "Caption": "Secondary horizon sensitivity analyses", "Source": "secondary_horizon_results.csv", "Section": "appendix_b_extended_benchmarks.md"},
            {"Table": "Appendix Table B6", "Caption": "Disagreement-based selective triage summary", "Source": "selective_triage_results.csv", "Section": "appendix_b_extended_benchmarks.md"},
            {"Table": "Appendix Table B7", "Caption": "Disagreement-based shift sensitivity summary", "Source": "selective_triage_shift_results.csv", "Section": "appendix_b_extended_benchmarks.md"},
            {"Table": "Appendix Table B8", "Caption": "Repeated grouped clinical score summary", "Source": "clinical_score_benchmark_summary.csv", "Section": "appendix_b_extended_benchmarks.md"},
            {"Table": "Appendix Table B9", "Caption": "Clinical score operating points on the main grouped split", "Source": "clinical_score_operating_points.csv", "Section": "appendix_b_extended_benchmarks.md"},
            {"Table": "Appendix Table B10", "Caption": "Key decision-curve net-benefit comparisons", "Source": "decision_curve_summary.csv", "Section": "appendix_b_extended_benchmarks.md"},
            {"Table": "Appendix Table B11", "Caption": "Fixed-policy clinical utility summary", "Source": "decision_curve_policy_metrics.csv", "Section": "appendix_b_extended_benchmarks.md"},
            {"Table": "Appendix Table B12", "Caption": "Negative and null probe results", "Source": "probe_all_results.json", "Section": "appendix_b_extended_benchmarks.md"},
        ]
    )

    claim_map = pd.DataFrame(
        [
            {"Claim": "Prediction performance is bounded in this T24 SA-AKI cohort.", "Evidence": "prediction_ceiling_results.csv; benchmark_summary.csv; probe_all_results.json", "Section": "01_introduction.md, 03_results.md"},
            {"Claim": "Grouped Optuna tuning and an added CatBoost comparator do not overturn the narrow benchmark ranking, so the manuscript should prioritize safe deployment rather than architecture churn.", "Evidence": "canonical_hpo_summary.json; canonical_hpo_top_trials.csv; benchmark_summary.csv; repeated_group_results.csv", "Section": "01_introduction.md, 02_methods.md, 03_results.md, Appendix B"},
            {"Claim": "Conformal selective triage provides a principled Alert/Defer/Clear framework.", "Evidence": "conformal_single_model_results.csv; conformal_operating_curve.csv", "Section": "02_methods.md, 03_results.md"},
            {"Claim": "Multi-model conformal consensus amplifies reliability while reducing automation rate.", "Evidence": "conformal_consensus_results.csv; conformal_consensus_tradeoff.png", "Section": "03_results.md"},
            {"Claim": "Conformal triage degrades more gracefully under shift than fixed thresholds.", "Evidence": "conformal_shift_results.csv; fixed_threshold_shift_results.csv; conformal_shift_panel.png", "Section": "03_results.md"},
            {"Claim": "The selected ML model retains better decision-curve net benefit than APACHE-III and SOFA across the main threshold range.", "Evidence": "decision_curve_summary.csv; clinical_utility_panel.png; clinical_score_operating_points.csv", "Section": "03_results.md, Appendix B"},
            {"Claim": "Disagreement-based selective triage is a useful comparator but weaker as a main methodological story.", "Evidence": "selective_triage_results.csv; selective_triage_shift_results.csv", "Section": "03_results.md, Appendix B"},
            {"Claim": "The paper is positioned for JAMIA with careful internal-validation claims only.", "Evidence": "journal_positioning.md; data_contract.md; audit_report.md", "Section": "00_title_and_abstract.md, 05_limitations.md, Appendix D"},
        ]
    )

    readme_md = """
    # JAMIA Manuscript Package

    This folder contains the chaptered markdown manuscript generated from the canonical artifact bundle in `local_outputs/artifacts/`.

    ## File Order

    1. `00_title_and_abstract.md`
    2. `01_introduction.md`
    3. `02_methods.md`
    4. `03_results.md`
    5. `04_discussion.md`
    6. `05_limitations.md`
    7. `06_conclusion.md`
    8. Appendices
    9. `figure_manifest.md`
    10. `table_manifest.md`
    11. `claim_evidence_map.md`
    12. `submission_delta_list.md`
    """

    write_markdown(MANUSCRIPT_DIR / "README.md", clean_markdown(readme_md))
    write_markdown(MANUSCRIPT_DIR / "00_title_and_abstract.md", clean_markdown(abstract_md))
    write_markdown(MANUSCRIPT_DIR / "01_introduction.md", clean_markdown(introduction_md))
    write_markdown(MANUSCRIPT_DIR / "02_methods.md", clean_markdown(methods_md))
    write_markdown(MANUSCRIPT_DIR / "03_results.md", clean_markdown(results_md))
    write_markdown(MANUSCRIPT_DIR / "04_discussion.md", clean_markdown(discussion_md))
    write_markdown(MANUSCRIPT_DIR / "05_limitations.md", clean_markdown(limitations_md))
    write_markdown(MANUSCRIPT_DIR / "06_conclusion.md", clean_markdown(conclusion_md))
    write_markdown(MANUSCRIPT_DIR / "appendix_a_data_and_cohort.md", clean_markdown(appendix_a_md))
    write_markdown(MANUSCRIPT_DIR / "appendix_b_extended_benchmarks.md", clean_markdown(appendix_b_md))
    write_markdown(MANUSCRIPT_DIR / "appendix_c_shift_and_subgroup_results.md", clean_markdown(appendix_c_md))
    write_markdown(MANUSCRIPT_DIR / "appendix_d_reproducibility.md", clean_markdown(appendix_d_md))
    write_markdown(MANUSCRIPT_DIR / "submission_delta_list.md", clean_markdown(submission_delta_md))
    write_markdown(ARTIFACT_DIR / "submission_delta_list.md", clean_markdown(submission_delta_md))
    write_markdown(MANUSCRIPT_DIR / "figure_manifest.md", "# Figure Manifest\n\n" + render_table(figure_manifest, index=False))
    write_markdown(MANUSCRIPT_DIR / "table_manifest.md", "# Table Manifest\n\n" + render_table(table_manifest, index=False))
    write_markdown(MANUSCRIPT_DIR / "claim_evidence_map.md", "# Claim-to-Evidence Map\n\n" + render_table(claim_map, index=False))


def main() -> None:
    build_manuscript()


if __name__ == "__main__":
    main()
