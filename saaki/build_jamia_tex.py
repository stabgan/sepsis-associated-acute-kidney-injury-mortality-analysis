from __future__ import annotations

import shutil
import sys
from pathlib import Path
from string import Template

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from saaki.build_jamia_manuscript import fmt, lookup_value, read_csv, read_json, top_rows

TEX_DIR = REPO_ROOT / "jamia_tex"
FIGURES_DIR = TEX_DIR / "figures"
TABLES_DIR = TEX_DIR / "tables"
UPSTREAM_DIR = TEX_DIR / "upstream" / "official"
ARTIFACT_DIR = REPO_ROOT / "local_outputs" / "artifacts"
PROBE_DIR = REPO_ROOT / "local_outputs" / "probes"
MANUSCRIPT_DIR = REPO_ROOT / "saaki" / "jamia_manuscript"

DISPLAY_NAME = {
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "logistic": "Logistic regression",
    "catboost": "CatBoost",
    "sofa_total_24hr": "SOFA",
    "apache_iii_score": "APACHE-III",
}

FIGURE_MAP = {
    "prediction_ceiling.png": "figure2_prediction_ceiling.png",
    "conformal_operating_curve.png": "figure5_conformal_operating_curve.png",
    "conformal_consensus_tradeoff.png": "figure6_consensus_tradeoff.png",
    "clinical_utility_panel.png": "figure7_clinical_utility_panel.png",
    "conformal_shift_panel.png": "figure8_shift_panel.png",
    "conformal_subgroup_forest.png": "figure9_subgroup_forest.png",
    "roc_pr_benchmarks.png": "appendix_b1_roc_pr.png",
    "calibration_curve.png": "appendix_b2_calibration_curve.png",
    "decision_curve.png": "appendix_b3_decision_curve.png",
}

MAIN_TABLE_FILES = [
    "table1_cohort.tex",
    "table2_benchmark.tex",
    "table3_conformal.tex",
    "table4_consensus.tex",
    "table5_shift.tex",
    "table6_subgroups.tex",
]

APPENDIX_TABLE_FILES = [
    "appendix_b1_tree_benchmark.tex",
    "appendix_b2_hpo_overview.tex",
    "appendix_b3_hpo_trials.tex",
    "appendix_b4_feature_ablation.tex",
    "appendix_b5_secondary_horizons.tex",
    "appendix_b6_disagreement_summary.tex",
    "appendix_b7_disagreement_shift.tex",
    "appendix_b8_score_summary.tex",
    "appendix_b9_score_operating.tex",
    "appendix_b10_utility_focus.tex",
    "appendix_b11_utility_policy.tex",
    "appendix_b12_negative_probes.tex",
]


class TexTemplate(Template):
    delimiter = "@"


def ensure_dirs() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)


def tex_escape(value: object) -> str:
    text = fmt(value)
    for old, new in [
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
        ("~", r"\textasciitilde{}"),
        ("^", r"\textasciicircum{}"),
        ("\n", " "),
    ]:
        text = text.replace(old, new)
    return text


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n")


def dataframe_to_tabular(frame: pd.DataFrame) -> str:
    if frame.empty:
        return (
            r"\begin{tabular}{@{}l@{}}" + "\n"
            r"\toprule" + "\n"
            r"No data available\\" + "\n"
            r"\bottomrule" + "\n"
            r"\end{tabular}" + "\n"
        )
    align = "l" + "c" * (len(frame.columns) - 1)
    lines = [
        rf"\begin{{tabular}}{{@{{}}{align}@{{}}}}",
        r"\toprule",
        " & ".join(tex_escape(col) for col in frame.columns) + r"\\",
        r"\midrule",
    ]
    for _, row in frame.iterrows():
        lines.append(" & ".join(tex_escape(v) for v in row.tolist()) + r"\\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def write_table_snippet(filename: str, frame: pd.DataFrame) -> None:
    write_text(TABLES_DIR / filename, dataframe_to_tabular(frame))


def copy_template_files() -> None:
    required = [
        "oup-authoring-template.cls",
        "oup-plain.bst",
        "oup-abbrvnat.bst",
    ]
    for name in required:
        src = UPSTREAM_DIR / name
        dst = TEX_DIR / name
        if not src.exists():
            raise FileNotFoundError(f"Missing upstream template file: {src}")
        shutil.copy2(src, dst)


def copy_data_figures() -> None:
    for src_name, dst_name in FIGURE_MAP.items():
        src = ARTIFACT_DIR / src_name
        dst = FIGURES_DIR / dst_name
        if not src.exists():
            raise FileNotFoundError(f"Missing figure artifact: {src}")
        shutil.copy2(src, dst)


def draw_flowchart(
    path_stem: str,
    nodes: list[dict[str, object]],
    arrows: list[tuple[str, str]],
    figsize: tuple[float, float],
) -> None:
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    lookup: dict[str, tuple[float, float, float, float]] = {}
    for node in nodes:
        x = float(node["x"])
        y = float(node["y"])
        w = float(node["w"])
        h = float(node["h"])
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.02,rounding_size=0.02",
                facecolor="#eaf3fb",
                edgecolor="#23455a",
                linewidth=1.4,
            )
        )
        ax.text(
            x + w / 2,
            y + h / 2,
            str(node["label"]),
            ha="center",
            va="center",
            fontsize=10,
            wrap=True,
        )
        lookup[str(node["id"])] = (x, y, w, h)

    for src, dst in arrows:
        sx, sy, sw, sh = lookup[src]
        tx, ty, tw, th = lookup[dst]
        start = (sx + sw, sy + sh / 2)
        end = (tx, ty + th / 2)
        if start[0] > end[0]:
            start = (sx + sw / 2, sy)
            end = (tx + tw / 2, ty + th)
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=14,
                linewidth=1.2,
                color="#23455a",
            )
        )

    fig.tight_layout(pad=0.5)
    fig.savefig(FIGURES_DIR / f"{path_stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / f"{path_stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def render_workflow_figures() -> None:
    draw_flowchart(
        "figure1_workflow",
        [
            {"id": "raw", "label": "Raw SA-AKI cohort", "x": 0.02, "y": 0.38, "w": 0.12, "h": 0.20},
            {"id": "audit", "label": "Audit and data contract", "x": 0.19, "y": 0.38, "w": 0.15, "h": 0.20},
            {"id": "features", "label": "Feature-set construction", "x": 0.39, "y": 0.38, "w": 0.15, "h": 0.20},
            {"id": "split", "label": "Subject-grouped train/validation/test split", "x": 0.59, "y": 0.38, "w": 0.18, "h": 0.20},
            {"id": "baseline", "label": "Baseline models and clinical scores", "x": 0.81, "y": 0.63, "w": 0.16, "h": 0.18},
            {"id": "disagreement", "label": "Disagreement selective triage", "x": 0.81, "y": 0.38, "w": 0.16, "h": 0.18},
            {"id": "conformal", "label": "Conformal selective triage", "x": 0.81, "y": 0.13, "w": 0.16, "h": 0.18},
            {"id": "assets", "label": "Tables, figures, and manuscript assets", "x": 0.59, "y": 0.05, "w": 0.18, "h": 0.20},
        ],
        [
            ("raw", "audit"),
            ("audit", "features"),
            ("features", "split"),
            ("split", "baseline"),
            ("split", "disagreement"),
            ("split", "conformal"),
            ("baseline", "assets"),
            ("disagreement", "assets"),
            ("conformal", "assets"),
        ],
        (12, 3.6),
    )

    draw_flowchart(
        "figure3_conformal_flow",
        [
            {"id": "patient", "label": "Patient at T24", "x": 0.08, "y": 0.40, "w": 0.16, "h": 0.18},
            {"id": "model", "label": "Risk model", "x": 0.33, "y": 0.40, "w": 0.16, "h": 0.18},
            {"id": "thresholds", "label": "Mondrian conformal thresholds", "x": 0.58, "y": 0.40, "w": 0.20, "h": 0.18},
            {"id": "pred", "label": "Prediction set", "x": 0.36, "y": 0.08, "w": 0.16, "h": 0.16},
            {"id": "alert", "label": "{1} Alert", "x": 0.72, "y": 0.72, "w": 0.16, "h": 0.16},
            {"id": "clear", "label": "{0} Clear", "x": 0.72, "y": 0.40, "w": 0.16, "h": 0.16},
            {"id": "defer", "label": "{0,1} Defer", "x": 0.72, "y": 0.08, "w": 0.16, "h": 0.16},
        ],
        [
            ("patient", "model"),
            ("model", "thresholds"),
            ("thresholds", "alert"),
            ("thresholds", "clear"),
            ("thresholds", "defer"),
            ("thresholds", "pred"),
        ],
        (11, 4.0),
    )

    draw_flowchart(
        "figure4_disagreement_flow",
        [
            {"id": "patient", "label": "Patient at T24", "x": 0.05, "y": 0.42, "w": 0.16, "h": 0.16},
            {"id": "full", "label": "Full feature model", "x": 0.30, "y": 0.67, "w": 0.18, "h": 0.16},
            {"id": "phys", "label": "Physiology-only model", "x": 0.30, "y": 0.17, "w": 0.18, "h": 0.16},
            {"id": "gap", "label": "Disagreement = |pFull - pPhys|", "x": 0.58, "y": 0.42, "w": 0.23, "h": 0.18},
            {"id": "gate", "label": "Agreement threshold and risk thresholds", "x": 0.58, "y": 0.12, "w": 0.23, "h": 0.18},
            {"id": "alert", "label": "Alert", "x": 0.86, "y": 0.67, "w": 0.10, "h": 0.16},
            {"id": "clear", "label": "Clear", "x": 0.86, "y": 0.42, "w": 0.10, "h": 0.16},
            {"id": "defer", "label": "Defer", "x": 0.86, "y": 0.17, "w": 0.10, "h": 0.16},
        ],
        [
            ("patient", "full"),
            ("patient", "phys"),
            ("full", "gap"),
            ("phys", "gap"),
            ("gap", "gate"),
            ("gate", "alert"),
            ("gate", "clear"),
            ("gate", "defer"),
        ],
        (11.5, 3.8),
    )


def load_artifacts() -> dict[str, object]:
    return {
        "audit": read_json(ARTIFACT_DIR / "audit_report.json"),
        "data_contract": read_json(ARTIFACT_DIR / "data_contract.json"),
        "clinical_policy": read_json(ARTIFACT_DIR / "clinical_policy.json"),
        "hpo_summary": read_json(ARTIFACT_DIR / "canonical_hpo_summary.json"),
        "probe_all": read_json(PROBE_DIR / "probe_all_results.json"),
        "benchmark_summary": read_csv("benchmark_summary.csv"),
        "hpo_top_trials": read_csv("canonical_hpo_top_trials.csv"),
        "score_summary": read_csv("clinical_score_benchmark_summary.csv"),
        "score_operating": read_csv("clinical_score_operating_points.csv"),
        "conformal_single_summary": read_csv("conformal_single_model_summary.csv"),
        "conformal_consensus_summary": read_csv("conformal_consensus_summary.csv"),
        "conformal_shift": read_csv("conformal_shift_results.csv"),
        "fixed_shift": read_csv("fixed_threshold_shift_results.csv"),
        "conformal_subgroups": read_csv("conformal_subgroup_results.csv"),
        "disagreement": read_csv("selective_triage_results.csv"),
        "disagreement_shift": read_csv("selective_triage_shift_results.csv"),
        "decision_curve_summary": read_csv("decision_curve_summary.csv"),
        "decision_curve_policy_metrics": read_csv("decision_curve_policy_metrics.csv"),
        "feature_ablation": read_csv("feature_ablation_results.csv"),
        "secondary_horizons": read_csv("secondary_horizon_results.csv"),
        "prediction_ceiling": pd.read_csv(ARTIFACT_DIR / "prediction_ceiling_results.csv"),
        "conformal_operating": pd.read_csv(ARTIFACT_DIR / "conformal_operating_curve.csv"),
    }


def float_match(frame: pd.DataFrame, column: str, value: float) -> pd.DataFrame:
    return frame[frame[column].round(6) == round(value, 6)]


def write_tables(payload: dict[str, object]) -> None:
    audit = payload["audit"]
    data_contract = payload["data_contract"]
    clinical_policy = payload["clinical_policy"]
    hpo_summary = payload["hpo_summary"]
    probe_all = payload["probe_all"]
    benchmark_summary = payload["benchmark_summary"]
    hpo_top_trials = payload["hpo_top_trials"]
    score_summary = payload["score_summary"]
    score_operating = payload["score_operating"]
    conformal_single_summary = payload["conformal_single_summary"]
    conformal_consensus_summary = payload["conformal_consensus_summary"]
    conformal_shift = payload["conformal_shift"]
    fixed_shift = payload["fixed_shift"]
    conformal_subgroups = payload["conformal_subgroups"]
    disagreement = payload["disagreement"]
    disagreement_shift = payload["disagreement_shift"]
    decision_curve_summary = payload["decision_curve_summary"]
    decision_curve_policy_metrics = payload["decision_curve_policy_metrics"]
    feature_ablation = payload["feature_ablation"]
    secondary_horizons = payload["secondary_horizons"]

    selected_model = clinical_policy.get("selected_model", "xgboost")
    hpo_lightgbm = hpo_summary.get("models", {}).get("lightgbm", {})
    hpo_best = hpo_lightgbm.get("best_metrics", {})

    table1 = pd.DataFrame(
        [
            {"Item": "Working cohort", "Value": data_contract.get("dataset_name", "mimic_saaki_raw_v2.csv")},
            {"Item": "Rows", "Value": audit.get("current_rows", "NA")},
            {"Item": "Columns", "Value": audit.get("current_columns", "NA")},
            {"Item": "Outcome prevalence", "Value": audit.get("event_rate", float("nan"))},
            {"Item": "Prediction time", "Value": data_contract.get("prediction_time", "24 hours after ICU admission (T24).")},
            {"Item": "Outcome", "Value": data_contract.get("target_definition", "In-hospital mortality")},
            {"Item": "Evaluation", "Value": "Subject-grouped holdout by subject_id"},
            {"Item": "Explicitly excluded claims", "Value": "; ".join(data_contract.get("excluded_claims", []))},
        ]
    )

    benchmark_main = benchmark_summary[
        benchmark_summary["model_name"].isin(
            [selected_model, "logistic", "lightgbm", "xgboost", "sofa_total_24hr", "apache_iii_score"]
        )
    ].copy()
    benchmark_main = benchmark_main[
        [
            "model_name",
            "auroc__mean",
            "auprc__mean",
            "brier__mean",
            "ece__mean",
            "calibration_slope__mean",
            "recall_at_ppv_050__mean",
        ]
    ].rename(
        columns={
            "model_name": "Model",
            "auroc__mean": "AUROC",
            "auprc__mean": "AUPRC",
            "brier__mean": "Brier",
            "ece__mean": "ECE",
            "calibration_slope__mean": "Calibration slope",
            "recall_at_ppv_050__mean": "Recall @ PPV>=0.50",
        }
    )

    table3 = top_rows(
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
        n=None,
    ).rename(
        columns={
            "alpha": "alpha",
            "n_groups": "Seeds",
            "coverage_mean": "Coverage mean",
            "coverage_ci_low": "Coverage 2.5%",
            "coverage_ci_high": "Coverage 97.5%",
            "certain_frac_mean": "Certain fraction",
            "alert_ppv_mean": "Alert PPV",
            "clear_npv_mean": "Clear NPV",
            "miss_count_mean": "Miss count",
        }
    )

    table4 = conformal_consensus_summary[
        (
            (conformal_consensus_summary["ensemble_name"] == "single_model")
            & (conformal_consensus_summary["base_model"] == selected_model)
        )
        | conformal_consensus_summary["ensemble_name"].isin(["union", "intersection"])
    ].copy()
    table4 = top_rows(
        table4,
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
        n=None,
    ).rename(
        columns={
            "ensemble_name": "Ensemble",
            "base_model": "Base model(s)",
            "alpha": "alpha",
            "n_groups": "Seeds",
            "coverage_mean": "Coverage mean",
            "certain_frac_mean": "Certain fraction",
            "alert_ppv_mean": "Alert PPV",
            "clear_npv_mean": "Clear NPV",
            "miss_count_mean": "Miss count",
        }
    )

    alpha_005 = float_match(conformal_shift, "alpha", 0.05)
    fixed_05 = fixed_shift[fixed_shift["threshold_name"] == "fixed_0.50"].copy()
    table5 = alpha_005.merge(
        fixed_05[["scenario", "severity", "precision", "recall"]],
        on=["scenario", "severity"],
        how="left",
    )
    table5 = table5.rename(
        columns={
            "scenario": "Scenario",
            "severity": "Severity",
            "coverage": "Conformal coverage",
            "certain_frac": "Conformal certain fraction",
            "alert_ppv": "Conformal alert PPV",
            "clear_npv": "Conformal clear NPV",
            "precision": "Fixed PPV @0.50",
            "recall": "Fixed recall @0.50",
        }
    )[
        [
            "Scenario",
            "Severity",
            "Conformal coverage",
            "Conformal certain fraction",
            "Conformal alert PPV",
            "Conformal clear NPV",
            "Fixed PPV @0.50",
            "Fixed recall @0.50",
        ]
    ]

    table6 = top_rows(
        conformal_subgroups,
        ["subgroup", "value", "n", "event_rate", "coverage", "certain_frac", "alert_ppv", "clear_npv"],
        n=None,
    ).rename(
        columns={
            "subgroup": "Subgroup",
            "value": "Level",
            "n": "n",
            "event_rate": "Event rate",
            "coverage": "Coverage",
            "certain_frac": "Certain fraction",
            "alert_ppv": "Alert PPV",
            "clear_npv": "Clear NPV",
        }
    )

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
        "Policy",
        "Alert precision mean",
        "Alert precision std",
        "Alert recall mean",
        "Alert recall std",
        "Actionable coverage mean",
        "Actionable coverage std",
        "Low-risk NPV mean",
        "Low-risk NPV std",
        "Defer rate mean",
        "Defer rate std",
        "Actionable error mean",
        "Actionable error std",
    ]

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
    ).rename(
        columns={
            "threshold": "Threshold",
            "strategy": "Strategy",
            "net_benefit": "Net benefit",
            "standardized_net_benefit": "Standardized net benefit",
            "alert_rate": "Alert rate",
        }
    )

    utility_policy = top_rows(
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
    ).rename(
        columns={
            "strategy": "Strategy",
            "alert_rate": "Alert rate",
            "alert_precision": "Alert precision",
            "alert_recall": "Alert recall",
            "net_benefit_at_0.10": "Net benefit @0.10",
            "net_benefit_at_0.20": "Net benefit @0.20",
            "net_benefit_at_0.30": "Net benefit @0.30",
        }
    )

    hpo_overview = pd.DataFrame(
        [
            {
                "Model": "lightgbm",
                "Trials": hpo_lightgbm.get("n_trials", 0),
                "Validation AUROC": hpo_best.get("auroc", float("nan")),
                "Validation recall @ PPV>=0.50": hpo_best.get("recall_at_ppv_050", float("nan")),
                "Validation Brier": hpo_best.get("brier", float("nan")),
                "Validation ECE": hpo_best.get("ece", float("nan")),
            }
        ]
    )

    hpo_trials = top_rows(
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
    ).rename(
        columns={
            "model_name": "Model",
            "trial_number": "Trial",
            "objective_value": "Objective",
            "auroc": "AUROC",
            "brier": "Brier",
            "recall_at_ppv_050": "Recall @ PPV>=0.50",
            "param_n_estimators": "Trees",
            "param_learning_rate": "Learning rate",
            "param_num_leaves": "Leaves",
            "param_max_depth": "Max depth",
            "param_min_child_samples": "Min child samples",
        }
    )

    diversity = benchmark_summary[benchmark_summary["model_name"].isin(["lightgbm", "xgboost", "catboost"])].copy()
    diversity = top_rows(
        diversity,
        ["model_name", "auroc__mean", "brier__mean", "ece__mean", "recall_at_ppv_050__mean"],
        n=None,
    ).rename(
        columns={
            "model_name": "Model",
            "auroc__mean": "AUROC",
            "brier__mean": "Brier",
            "ece__mean": "ECE",
            "recall_at_ppv_050__mean": "Recall @ PPV>=0.50",
        }
    )

    negative_rows: list[dict[str, object]] = []
    for key in ["feat_missingness", "feat_ratios", "ensemble_stack_3", "hpo_lgbm", "survival_rsf"]:
        if key not in probe_all:
            continue
        record = probe_all[key]
        negative_rows.append(
            {
                "Probe": key,
                "AUROC": record.get("auroc"),
                "AUPRC": record.get("auprc"),
                "Recall @ PPV>=0.50": record.get("recall@ppv50") or record.get("recall_at_ppv050"),
            }
        )
    negative_probe_table = pd.DataFrame(negative_rows)

    write_table_snippet("table1_cohort.tex", table1)
    write_table_snippet("table2_benchmark.tex", benchmark_main)
    write_table_snippet("table3_conformal.tex", table3)
    write_table_snippet("table4_consensus.tex", table4)
    write_table_snippet("table5_shift.tex", table5)
    write_table_snippet("table6_subgroups.tex", table6)
    write_table_snippet("appendix_b1_tree_benchmark.tex", diversity)
    write_table_snippet("appendix_b2_hpo_overview.tex", hpo_overview)
    write_table_snippet("appendix_b3_hpo_trials.tex", hpo_trials)
    write_table_snippet("appendix_b4_feature_ablation.tex", feature_ablation)
    write_table_snippet("appendix_b5_secondary_horizons.tex", secondary_horizons)
    write_table_snippet("appendix_b6_disagreement_summary.tex", disagreement_summary)
    write_table_snippet("appendix_b7_disagreement_shift.tex", disagreement_shift)
    write_table_snippet("appendix_b8_score_summary.tex", score_summary)
    write_table_snippet("appendix_b9_score_operating.tex", score_operating)
    write_table_snippet("appendix_b10_utility_focus.tex", utility_focus)
    write_table_snippet("appendix_b11_utility_policy.tex", utility_policy)
    write_table_snippet("appendix_b12_negative_probes.tex", negative_probe_table)


def float_lookup(frame: pd.DataFrame, column: str, value: float) -> pd.Series:
    matches = float_match(frame, column, value)
    if matches.empty:
        raise ValueError(f"Could not find {column}={value}")
    return matches.iloc[0]


def build_context(payload: dict[str, object]) -> dict[str, str]:
    audit = payload["audit"]
    clinical_policy = payload["clinical_policy"]
    hpo_summary = payload["hpo_summary"]
    benchmark_summary = payload["benchmark_summary"]
    conformal_single_summary = payload["conformal_single_summary"]
    conformal_consensus_summary = payload["conformal_consensus_summary"]
    decision_curve_summary = payload["decision_curve_summary"]
    conformal_subgroups = payload["conformal_subgroups"]
    prediction_ceiling = payload["prediction_ceiling"]
    conformal_operating = payload["conformal_operating"]

    selected_model = clinical_policy.get("selected_model", "xgboost")
    selected_display = DISPLAY_NAME.get(selected_model, selected_model)
    selected_calibration = clinical_policy.get("calibration", "isotonic")

    selected_auroc = lookup_value(benchmark_summary, {"model_name": selected_model}, "auroc__mean")
    selected_auprc = lookup_value(benchmark_summary, {"model_name": selected_model}, "auprc__mean")
    selected_brier = lookup_value(benchmark_summary, {"model_name": selected_model}, "brier__mean")
    selected_recall = lookup_value(benchmark_summary, {"model_name": selected_model}, "recall_at_ppv_050__mean")
    lightgbm_auroc = lookup_value(benchmark_summary, {"model_name": "lightgbm"}, "auroc__mean")
    lightgbm_recall = lookup_value(benchmark_summary, {"model_name": "lightgbm"}, "recall_at_ppv_050__mean")
    catboost_auroc = lookup_value(benchmark_summary, {"model_name": "catboost"}, "auroc__mean")
    catboost_brier = lookup_value(benchmark_summary, {"model_name": "catboost"}, "brier__mean")
    catboost_recall = lookup_value(benchmark_summary, {"model_name": "catboost"}, "recall_at_ppv_050__mean")

    honest_row = prediction_ceiling[prediction_ceiling["setting"] == "honest_grouped"].iloc[0]
    ceiling_row = prediction_ceiling[prediction_ceiling["setting"] == "with_time_to_event"].iloc[0]
    single_alpha005 = float_lookup(conformal_single_summary, "alpha", 0.05)
    union_alpha005 = conformal_consensus_summary[
        (conformal_consensus_summary["ensemble_name"] == "union")
        & (conformal_consensus_summary["alpha"].round(6) == round(0.05, 6))
    ].iloc[0]
    sweet_spot = (
        conformal_operating[
            (conformal_operating["alert_ppv"] >= 0.55) & (conformal_operating["clear_npv"] >= 0.90)
        ]
        .sort_values(["certain_frac", "coverage"], ascending=[False, False])
        .iloc[0]
    )

    def decision_value(strategy: str, threshold: float) -> str:
        row = decision_curve_summary[
            (decision_curve_summary["strategy"] == strategy)
            & (decision_curve_summary["threshold"].round(6) == round(threshold, 6))
        ].iloc[0]
        return fmt(row["net_benefit"])

    hpo_lightgbm = hpo_summary.get("models", {}).get("lightgbm", {})
    hpo_best = hpo_lightgbm.get("best_metrics", {})
    subgroup_alpha = fmt(conformal_subgroups["alpha"].iloc[0]) if not conformal_subgroups.empty else "0.10"

    return {
        "rows": str(audit.get("current_rows", "NA")),
        "columns": str(audit.get("current_columns", "NA")),
        "unique_subjects": str(audit.get("unique_subjects", "NA")),
        "repeated_subject_rows": str(audit.get("repeated_subject_rows", "NA")),
        "event_rate": fmt(audit.get("event_rate", float("nan"))),
        "selected_model": selected_model,
        "selected_display": selected_display,
        "selected_calibration": selected_calibration,
        "selected_auroc": fmt(selected_auroc),
        "selected_auprc": fmt(selected_auprc),
        "selected_brier": fmt(selected_brier),
        "selected_recall": fmt(selected_recall),
        "lightgbm_auroc": fmt(lightgbm_auroc),
        "lightgbm_recall": fmt(lightgbm_recall),
        "catboost_auroc": fmt(catboost_auroc),
        "catboost_brier": fmt(catboost_brier),
        "catboost_recall": fmt(catboost_recall),
        "ceiling_honest_auroc": fmt(honest_row["auroc"]),
        "ceiling_with_time_auroc": fmt(ceiling_row["auroc"]),
        "hpo_trials": str(hpo_lightgbm.get("n_trials", 0)),
        "hpo_val_auroc": fmt(hpo_best.get("auroc", float("nan"))),
        "hpo_val_recall": fmt(hpo_best.get("recall_at_ppv_050", float("nan"))),
        "hpo_val_brier": fmt(hpo_best.get("brier", float("nan"))),
        "single_coverage": fmt(single_alpha005["coverage_mean"]),
        "single_certain_frac": fmt(single_alpha005["certain_frac_mean"]),
        "single_alert_ppv": fmt(single_alpha005["alert_ppv_mean"]),
        "single_clear_npv": fmt(single_alpha005["clear_npv_mean"]),
        "union_coverage": fmt(union_alpha005["coverage_mean"]),
        "union_alert_ppv": fmt(union_alpha005["alert_ppv_mean"]),
        "union_clear_npv": fmt(union_alpha005["clear_npv_mean"]),
        "sweet_alpha": fmt(sweet_spot["alpha"]),
        "sweet_certain_frac": fmt(sweet_spot["certain_frac"]),
        "sweet_alert_ppv": fmt(sweet_spot["alert_ppv"]),
        "sweet_clear_npv": fmt(sweet_spot["clear_npv"]),
        "decision_nb_010": decision_value(f"{selected_model}_continuous", 0.10),
        "decision_nb_020": decision_value(f"{selected_model}_continuous", 0.20),
        "decision_nb_030": decision_value(f"{selected_model}_continuous", 0.30),
        "decision_apache_020": decision_value("apache_iii_score_continuous", 0.20),
        "decision_sofa_020": decision_value("sofa_total_24hr_continuous", 0.20),
        "decision_fixed_020": decision_value("fixed_threshold_ppv50", 0.20),
        "subgroup_alpha": subgroup_alpha,
    }


def table_block(filename: str, caption: str, label: str, *, wide: bool = False, size: str = "small") -> str:
    env = "table*" if wide else "table"
    max_w = r"\textwidth" if wide else r"\linewidth"
    return (
        rf"\begin{{{env}}}[!t]" + "\n"
        r"\centering" + "\n"
        + rf"\{size}" + "\n"
        + rf"\caption{{{caption}}}" + "\n"
        + rf"\label{{{label}}}" + "\n"
        + rf"\begin{{adjustbox}}{{max width={max_w}}}" + "\n"
        + rf"\input{{tables/{filename}}}" + "\n"
        + r"\end{adjustbox}" + "\n"
        + rf"\end{{{env}}}" + "\n"
    )


def figure_block(filename: str, caption: str, label: str, *, width: str = r"\linewidth") -> str:
    return (
        r"\begin{figure}[!t]" + "\n"
        r"\centering" + "\n"
        + rf"\includegraphics[width={width}]{{{filename}}}" + "\n"
        + rf"\caption{{{caption}}}" + "\n"
        + rf"\label{{{label}}}" + "\n"
        + r"\end{figure}" + "\n"
    )


REFS_BIB = """@article{johnson2016mimic,
  title={MIMIC-III, a freely accessible critical care database},
  author={Johnson, Alistair E. W. and Pollard, Tom J. and Shen, Lu and Li-wei, H. Lehman and Feng, Mengling and Ghassemi, Mohammad and Moody, Benjamin and Szolovits, Peter and Celi, Leo Anthony and Mark, Roger G.},
  journal={Scientific Data},
  volume={3},
  pages={160035},
  year={2016}
}

@inproceedings{chen2016xgboost,
  title={XGBoost: A scalable tree boosting system},
  author={Chen, Tianqi and Guestrin, Carlos},
  booktitle={Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining},
  pages={785--794},
  year={2016}
}

@inproceedings{ke2017lightgbm,
  title={LightGBM: A highly efficient gradient boosting decision tree},
  author={Ke, Guolin and Meng, Qi and Finley, Thomas and Wang, Taifeng and Chen, Wei and Ma, Weidong and Ye, Qiwei and Liu, Tie-Yan},
  booktitle={Advances in Neural Information Processing Systems},
  volume={30},
  year={2017}
}

@inproceedings{prokhorenkova2018catboost,
  title={CatBoost: Unbiased boosting with categorical features},
  author={Prokhorenkova, Liudmila and Gusev, Gleb and Vorobev, Aleksandr and Dorogush, Anna Veronika and Gulin, Andrey},
  booktitle={Advances in Neural Information Processing Systems},
  volume={31},
  year={2018}
}

@inproceedings{akiba2019optuna,
  title={Optuna: A next-generation hyperparameter optimization framework},
  author={Akiba, Takuya and Sano, Shotaro and Yanase, Toshihiko and Ohta, Takeru and Koyama, Masanori},
  booktitle={Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining},
  pages={2623--2631},
  year={2019}
}

@book{vovk2005algorithmic,
  title={Algorithmic Learning in a Random World},
  author={Vovk, Vladimir and Gammerman, Alex and Shafer, Glenn},
  publisher={Springer},
  year={2005}
}

@article{angelopoulos2021gentle,
  title={A gentle introduction to conformal prediction and distribution-free uncertainty quantification},
  author={Angelopoulos, Anastasios N. and Bates, Stephen},
  journal={arXiv preprint arXiv:2107.07511},
  year={2021}
}

@article{vickers2006decision,
  title={Decision curve analysis: A novel method for evaluating prediction models},
  author={Vickers, Andrew J. and Elkin, Ethan B.},
  journal={Medical Decision Making},
  volume={26},
  number={6},
  pages={565--574},
  year={2006}
}

@article{knaus1991apache,
  title={The APACHE III prognostic system. Risk prediction of hospital mortality for critically ill hospitalized adults},
  author={Knaus, William A. and Wagner, Douglas P. and Draper, Elizabeth A. and Zimmerman, Jack E. and Bergner, Marilyn and Bastos, Patricia G. and Sirio, Christopher A. and Murphy, Donald J. and Lotring, Terry and Damiano, Anne and Harrell, Frank E.},
  journal={Chest},
  volume={100},
  number={6},
  pages={1619--1636},
  year={1991}
}

@article{vincent1996sofa,
  title={The SOFA (Sepsis-related Organ Failure Assessment) score to describe organ dysfunction/failure},
  author={Vincent, Jean-Louis and Moreno, Rui and Takala, Jukka and Willatts, Sheila and De Mendon{\\c{c}}a, Arnaldo and Bruining, Hans and Reinhart, Christian K. and Suter, Peter M. and Thijs, L. G.},
  journal={Intensive Care Medicine},
  volume={22},
  number={7},
  pages={707--710},
  year={1996}
}
"""


def write_refs_bib() -> None:
    write_text(TEX_DIR / "refs.bib", REFS_BIB)


def write_main_tex(context: dict[str, str]) -> None:
    template = TexTemplate(
        r"""\documentclass[unnumsec,webpdf,contemporary,large,numbered,nocrop]{oup-authoring-template}
\usepackage{booktabs}
\usepackage{adjustbox}
\usepackage{array}
\usepackage{microtype}
\graphicspath{{figures/}}
% OUP class defines \state and \country but not \city (used in placeholder affiliations).
\providecommand{\city}[1]{#1}
% webpdf enables CropBox pdfmarks via PostScript \special; pdfTeX ignores them and warns.
\makeatletter
\AtBeginDocument{%
  \let\shipout@@PageObjects\@@empty
  \let\rest@@dvi@@pages\@@empty
}%
\makeatother
\emergencystretch=2em
% Contemporary opening header uses nested \hbox to \textwidth (OUP class); ~11.4pt overfull is intrinsic.
\hfuzz=12pt

\begin{document}

\journaltitle{Journal of the American Medical Informatics Association}
\DOI{DOI added during production}
\copyrightyear{2026}
\pubyear{2026}
\vol{XX}
\issue{X}
\access{Advance Access publication date TBD}
\appnotes{Original Article}
\firstpage{1}

\title[Conformal selective triage for SA-AKI mortality]{When to Alert, When to Defer: Conformal Selective Triage for ICU Mortality in Sepsis-Associated AKI}

\author[1,$\ast$]{Author Name}
\address[1]{\orgdiv{Department}, \orgname{Institution}, \orgaddress{\city{City}, \state{State}, \country{Country}}}
\corresp[$\ast$]{Corresponding author. \href{mailto:author@@example.com}{author@@example.com}}

\received{Date}{0}{Year}
\revised{Date}{0}{Year}
\accepted{Date}{0}{Year}

\abstract{\textbf{Objective:} To develop and evaluate an uncertainty-aware triage framework for early in-hospital mortality prediction in sepsis-associated acute kidney injury (SA-AKI), with primary emphasis on safe automation boundaries rather than raw discrimination alone.\\
\textbf{Materials and Methods:} We analyzed @rows ICU-stay-level SA-AKI cohort records from \texttt{mimic\_saaki\_raw\_v2.csv} scored at 24 hours after ICU admission, with event prevalence @event_rate. All deployment claims used subject-grouped train/validation/test splits by \texttt{subject\_id}. Conventional baselines included logistic regression, XGBoost, a grouped-Optuna-tuned LightGBM benchmark (@hpo_trials trials), an exploratory CatBoost diversity comparator, and score-only baselines. Calibration was selected on grouped validation data. We evaluated disagreement-based selective triage as an ablation and Mondrian conformal selective triage as the main method. We summarized coverage, alert positive predictive value (PPV), clear negative predictive value (NPV), calibration, decision-curve utility, subgroup heterogeneity, and robustness under simulated distribution shift, with headline conformal estimates repeated across 21 grouped seeds.\\
\textbf{Results:} The best conventional discriminative baseline was @selected_display with @selected_calibration calibration (mean AUROC @selected_auroc; recall at PPV $\geq 0.50$ of @selected_recall). Grouped Optuna improved LightGBM validation recall at PPV $\geq 0.50$ to @hpo_val_recall, but repeated grouped benchmarking still preserved only a narrow tree-model performance band. The prediction ceiling remained limited: even when \texttt{time\_to\_event\_hrs} was added as a leakage-like feature, AUROC reached only @ceiling_with_time_auroc. Single-model conformal triage achieved coverage @single_coverage at $\alpha=0.05$ while retaining a clinically meaningful defer region. The multi-model union consensus at $\alpha=0.05$ concentrated decisions into a smaller, more reliable actionable region with alert PPV @union_alert_ppv and clear NPV @union_clear_npv. At a decision threshold of 0.20, the selected continuous model achieved net benefit @decision_nb_020 versus @decision_apache_020 for APACHE-III and @decision_sofa_020 for SOFA.\\
\textbf{Discussion and Conclusion:} The main contribution is not a new AUROC leader; it is a deployment-oriented Alert/Defer/Clear framework that quantifies when the model should defer. Conformal selective triage is a stronger primary story than disagreement-based deferral because it supplies finite-sample coverage guarantees and more graceful degradation under shift.}

\keywords{conformal prediction, sepsis-associated acute kidney injury, mortality prediction, intensive care, uncertainty quantification, deployment}
\keywords[Abbreviations]{AKI, APACHE-III, AUROC, ICU, MIMIC, NPV, PPV, SA-AKI, SOFA}

\boxedtext{Key Messages}{
\begin{itemize}
\item First-24-hour SA-AKI mortality discrimination is bounded even after grouped Optuna tuning and an added CatBoost benchmark.
\item Conformal selective triage reframes ICU risk prediction into Alert/Defer/Clear actions with coverage guarantees.
\item Conservative multi-model consensus and explicit deferral remain more reliable under simulated shift than fixed-threshold policies.
\end{itemize}}

\maketitle

\section{Introduction}
Predicting mortality in critically ill patients with sepsis-associated acute kidney injury is clinically important, but retrospective machine-learning studies often stop at aggregate discrimination and overstate deployment readiness. In practice, clinicians need to know not only how well a model ranks patients on average, but also when it should escalate care, when it can safely clear a patient from immediate concern, and when it should defer because the case is too uncertain for automation. The working cohort in this study is a MIMIC-derived SA-AKI dataset with @rows T24 rows, outcome prevalence @event_rate, and subject-level duplication that makes grouped evaluation mandatory.\cite{johnson2016mimic}

This project starts from a hard empirical observation: discrimination in this cohort is bounded. The strongest conventional models occupy a narrow performance band, with @selected_display at AUROC @selected_auroc, LightGBM at @lightgbm_auroc, and CatBoost at @catboost_auroc. Even a leak-prone ceiling experiment that includes \texttt{time\_to\_event\_hrs} reaches only AUROC @ceiling_with_time_auroc, compared with @ceiling_honest_auroc in the honest grouped setting. That finding changes the research objective. The key question is no longer how to squeeze another 0.01 of AUROC from the same first-24-hour features; it is how to build a clinically legible triage policy that behaves safely under bounded signal.

Two uncertainty-aware strategies are therefore compared. The first is a disagreement-based selective-triage baseline that treats divergence between a physiology-only model and a process-enriched model as a signal to defer. The second, and primary contribution of this paper, is conformal selective triage. Conformal prediction converts probabilistic risk scores into set-valued decisions with finite-sample coverage guarantees, which naturally supports a three-way clinical interface: Alert, Clear, or Defer.\cite{vovk2005algorithmic,angelopoulos2021gentle}

The manuscript makes three contributions. First, it documents the prediction ceiling for first-24-hour SA-AKI mortality risk in a subject-grouped evaluation design. Second, it introduces conformal selective triage as the main decision framework and shows that it provides explicit safety-oriented operating points in terms of coverage, alert PPV, and clear NPV. Third, it demonstrates that multi-model conformal consensus and conformal deferral degrade more gracefully than fixed thresholds under missingness and process-feature perturbations.

\section{Methods}
\subsection{Study Design and Data Source}
This retrospective study used \texttt{mimic\_saaki\_raw\_v2.csv}, a T24 SA-AKI cohort derived from MIMIC.\cite{johnson2016mimic} One row represents one ICU-stay-level SA-AKI record scored at 24 hours after ICU admission. The primary label is in-hospital mortality (\texttt{event\_observed}), and the time-to-event variable is \texttt{time\_to\_event\_hrs}. Because repeated patients remain in the cohort, all deployment claims use subject-grouped evaluation by \texttt{subject\_id}.

\subsection{Cohort and Evaluation Workflow}
The analysis pipeline began with an audit and data-contract pass, followed by feature-set construction, subject-grouped train/validation/test splitting, conventional benchmark modeling, disagreement-based selective triage, conformal selective triage, and manuscript-asset generation.

@figure1

\subsection{Baselines}
Conventional baselines included logistic regression, LightGBM,\cite{ke2017lightgbm} and XGBoost.\cite{chen2016xgboost} Calibration was selected on grouped validation data by benchmarking sigmoid and isotonic recalibration on a held-out grouped validation split. We also retained score-only comparators based on SOFA and APACHE-III where those variables were available, treating them as single-feature logistic baselines under the same grouped protocol.\cite{knaus1991apache,vincent1996sofa}

To close the remaining methodological rigor gap from earlier drafts, we added a grouped Optuna search for the main LightGBM benchmark.\cite{akiba2019optuna} The search used @hpo_trials trials and optimized a deployment-oriented objective that combined AUROC, recall at PPV $\geq 0.50$, and Brier score on grouped validation data. The tuned LightGBM configuration was then frozen and carried into the repeated grouped benchmark. We also added CatBoost as a benchmark-only diversity comparator to test whether a distinct gradient-boosting family materially changed the model-selection story.\cite{prokhorenkova2018catboost}

The disagreement-based selective-triage comparator follows the earlier project draft. A process-enriched full model and a physiology-severity model were trained on the same grouped split. Cases were considered actionable only when the two models agreed closely enough on grouped validation data.

\subsection{Conformal Selective Triage}
The main method is Mondrian conformal prediction.\cite{vovk2005algorithmic,angelopoulos2021gentle} A training subset fits the risk model, a separate calibration subset estimates nonconformity thresholds, and the test subset receives set-valued predictions. For binary mortality prediction, the prediction set can be one of three clinically interpretable outcomes:
\begin{itemize}
\item \{1\}: Alert
\item \{0\}: Clear
\item \{0,1\}: Defer
\end{itemize}

@figure3

\subsection{Disagreement Baseline}
The disagreement baseline translates model disagreement into an empirical defer region. Unlike conformal prediction, this policy is heuristic and does not provide finite-sample coverage guarantees.

@figure4

\subsection{Metrics and Statistical Analysis}
We reported AUROC, AUPRC, Brier score, expected calibration error, calibration intercept, calibration slope, recall at PPV $\geq 0.50$, alert burden, and low-risk coverage. For conformal selective triage we reported coverage, certain-decision fraction, defer rate, alert PPV, clear NPV, and miss count. Subject-aware uncertainty was summarized with clustered bootstrap confidence intervals, and headline conformal estimates were repeated across 21 grouped seeds. Shift robustness was assessed by random missingness injection and by dropout of measurement-process and care-process feature families. Clinical utility was summarized with decision-curve analysis against APACHE-III, SOFA, treat-all, treat-none, and fixed-policy comparators.\cite{vickers2006decision} The primary decision-curve interpretation focused on the continuous selected model versus clinical scores, while conformal and disagreement curves were treated as deployment-policy complements rather than direct substitutes for continuous ranking.

\subsection{Ethics, Data Access, and Reproducibility}
This study is a retrospective secondary analysis of a de-identified MIMIC-derived cohort and does not involve prospective intervention or direct patient contact. The repository distributes code, derived artifact summaries, and manuscript assets rather than raw source patient data. Reproducing the cohort from source requires independent credentialed access to MIMIC and the associated data-use approvals. All manuscript claims are therefore limited to internal validation on the checked-in derivative cohort and should not be read as transportability or bedside-readiness claims.

\section{Results}
\subsection{Cohort and Data Contract}
Table~\ref{tab:cohort} summarizes the working cohort and the constraints that bound the manuscript claims.

@table1

\subsection{Prediction Ceiling and Conventional Benchmarks}
Figure~\ref{fig:ceiling} and Table~\ref{tab:benchmark} anchor the benchmark story. The main result is that the discrimination ceiling is bounded and the baseline model-family differences are comparatively small.

@figure2

@table2

@figure7

The ceiling experiment is especially important. In the canonical workflow, the leak-prone configuration that adds \texttt{time\_to\_event\_hrs} reaches AUROC @ceiling_with_time_auroc, while the honest grouped setting remains materially lower at @ceiling_honest_auroc. The repeated grouped benchmark selected @selected_display because it combined mean AUROC @selected_auroc, Brier @selected_brier, and recall at PPV $\geq 0.50$ of @selected_recall more consistently than the competing primary baselines. The grouped Optuna pass improved LightGBM validation recall at PPV $\geq 0.50$ to @hpo_val_recall, but repeated grouped LightGBM still averaged AUROC @lightgbm_auroc and recall @lightgbm_recall. The exploratory CatBoost comparator remained competitive (AUROC @catboost_auroc, Brier @catboost_brier, recall @catboost_recall) but did not materially simplify or improve the downstream conformal story, so it stayed in the appendix as a diversity check rather than replacing the primary model family.

The clinical-utility panel complements that story. At decision threshold 0.20, the continuous @selected_model score yields net benefit @decision_nb_020 versus @decision_apache_020 for APACHE-III, @decision_sofa_020 for SOFA, and @decision_fixed_020 for the fixed-threshold PPV $\geq 0.50$ policy. The conformal $\alpha=0.05$ policy is deliberately more conservative, with lower alert rate but higher alert precision. This distinction matters: the continuous model is the main utility comparison against clinical scores, whereas the conformal and disagreement policies are complementary deployment policies that trade some net benefit for narrower, higher-confidence action sets.

\subsection{Main Conformal Triage Result}
Table~\ref{tab:conformal-main} summarizes repeated subject-grouped single-model conformal triage results. The main manuscript operating point remains the low-$\alpha$ regime, where coverage is controlled while a clinically meaningful defer region is preserved.

@table3

@figure5

At $\alpha=0.05$, single-model conformal triage achieved coverage @single_coverage with certain-decision fraction @single_certain_frac, alert PPV @single_alert_ppv, and clear NPV @single_clear_npv. Under the manuscript sweet-spot criterion (NPV $\geq 0.90$ and PPV $\geq 0.55$), the best operating point occurs at $\alpha=@sweet_alpha$ with certain-decision fraction @sweet_certain_frac, alert PPV @sweet_alert_ppv, and clear NPV @sweet_clear_npv.

\subsection{Multi-Model Consensus}
Table~\ref{tab:conformal-consensus} and Figure~\ref{fig:consensus} show the consensus extension. The union of conformal prediction sets is deliberately conservative: it acts on fewer patients but produces a more reliable actionable region.

@table4

@figure6

At $\alpha=0.05$, the union consensus yields coverage @union_coverage, alert PPV @union_alert_ppv, and clear NPV @union_clear_npv, confirming that consensus amplifies reliability at the cost of automation rate.

\subsection{Comparison With Disagreement-Based Selective Triage}
The disagreement-based policy remains an important ablation because it shows that explicit deferral helps, but its guarantees are empirical and heuristic rather than finite-sample. The repeated grouped summary is included in Appendix~\ref{app:benchmarks}. In brief, disagreement triage narrows the actionable region and reduces actionable error, but conformal triage supplies the stronger methodological framing because it binds that defer region to a formal coverage objective.

\subsection{Robustness Under Distribution Shift}
Table~\ref{tab:shift} and Figure~\ref{fig:shift} summarize robustness under simulated shift.

@table5

@figure8

As extra missingness is injected, conformal coverage remains stable because the framework defers more cases. By contrast, fixed thresholds lose recall while offering no explicit warning that uncertainty has increased.

\subsection{Subgroup Heterogeneity}
Table~\ref{tab:subgroups} and Figure~\ref{fig:subgroups} summarize subgroup heterogeneity for conformal triage at $\alpha=@subgroup_alpha$.

@table6

@figure9

Coverage and clear NPV remain high across most demographic subgroups, but high-acuity strata are predictably harder. Reporting subgroup summaries at $\alpha=@subgroup_alpha$ exposes that trade-off explicitly: low-SOFA strata preserve high coverage with a reasonable action rate, whereas high-SOFA strata require a larger defer region.

\section{Discussion}
This study argues that the central challenge in SA-AKI mortality modeling is not how to win a retrospective AUROC leaderboard; it is how to construct a clinically legible decision policy when the predictive ceiling is bounded. The canonical benchmark results, the grouped Optuna pass, and the added CatBoost diversity comparator all point in the same direction. Tree models remain competitive, but their gains over one another are modest and unstable enough that the real manuscript contribution comes from making uncertainty explicit rather than from claiming a dramatic architecture breakthrough.

Conformal selective triage addresses that problem directly. Instead of forcing every patient into a binary high-risk versus low-risk decision, it partitions the cohort into Alert, Clear, and Defer. That defer region is not a nuisance artifact; it is the mechanism that protects coverage. In this dataset, the defer option is exactly what allows the model to preserve reliable alert PPV and clear NPV under both clean evaluation and shift stress.

The disagreement-based comparator remains useful because it demonstrates that uncertainty-aware action restriction helps even before formal conformalization. However, the disagreement policy depends on an empirically tuned agreement threshold and is harder to justify theoretically. The conformal formulation is stronger for a journal paper because it is more principled, more general, and more transparent about what is and is not guaranteed.

From a translational standpoint, the manuscript supports a narrow but realistic deployment framing: retrospective workflow design, threshold governance, and safe automation boundaries for ICU triage. The decision-curve results support the continuous @selected_display score as the primary clinical-score comparator, while conformal and disagreement policies should be read as downstream action-governance layers rather than as replacements for continuous ranking.

\section{Limitations}
This work has several important limitations.
\begin{enumerate}
\item The study remains internally validated only. The current checked-in cohort does not expose a defensible calendar timestamp axis for a true temporal split, and no external cohort is included.
\item The cohort remains affected by ETL and documentation mismatches already documented in the audit and data-contract artifacts.
\item Repeated patients remain present in the working cohort, which is why subject-grouped evaluation is mandatory.
\item The process-rich feature space may not transport as well as physiology-only inputs.
\item The conformal guarantees are marginal under exchangeability and do not imply perfect subgroup-conditional coverage.
\item Secondary horizon results, especially the 48-hour endpoint, are limited by low event counts and should be treated as sensitivity analyses rather than standalone deployment targets.
\item The manuscript focuses on retrospective decision support and does not evaluate clinical workflow adoption, clinician behavior change, or prospective impact.
\item The disagreement-based baseline is heuristic by design and is included as a comparison rather than as the final methodological recommendation.
\end{enumerate}

\section{Conclusion}
In this SA-AKI cohort, the predictive ceiling for first-24-hour ICU mortality risk appears bounded, and that changes the scientific target. The most defensible contribution is not a marginal improvement in discrimination but an uncertainty-aware triage policy that states when the model should act and when it should defer.

Conformal selective triage provides that framing. It yields clinically interpretable Alert/Defer/Clear decisions, finite-sample coverage guarantees, stronger reliability under conservative consensus, and more graceful degradation under shift than fixed-threshold rules. For a clinical informatics venue such as JAMIA, this is the right story: rigorous internal validation, explicit uncertainty handling, and a careful boundary around what retrospective machine learning can responsibly claim.

\begin{appendices}

\section{Data and Cohort}\label{app:data}
The audit pass identified @unique_subjects unique subjects among @rows working rows, with @repeated_subject_rows repeated-subject rows that make stay-level random splitting inappropriate for deployment claims. The package therefore treats subject-grouped evaluation as a hard design constraint rather than an optional robustness check.

The explicit data contract for the journal package is intentionally conservative: one row equals one ICU-stay-level SA-AKI record scored at T24; all predictor features must be available by T24; post-T24 information is excluded from deployment claims even if it exists elsewhere in the repository; and bedside-readiness, transportability, and exact thesis-cohort equivalence are not claimed.

\section{Extended Benchmarks}\label{app:benchmarks}
Appendix~\ref{app:benchmarks} gathers the benchmark tables that support the main story without overloading the primary narrative.

@appendix_b1

@appendix_b2

@appendix_b3

Grouped Optuna improved the LightGBM validation operating point (validation AUROC @hpo_val_auroc; recall at PPV $\geq 0.50$ of @hpo_val_recall), but the repeated grouped benchmark still left XGBoost, tuned LightGBM, and CatBoost in a narrow performance band.

@appendix_b4

@appendix_b5

@appendix_fig_b1

@appendix_fig_b2

@appendix_fig_b3

@appendix_b6

@appendix_b7

@appendix_b8

@appendix_b9

@appendix_b10

@appendix_b11

@appendix_b12

\section{Shift and Subgroup Notes}\label{app:shift}
The shift and subgroup appendices are intentionally framed as deployment notes rather than as new performance claims. Missingness injection, care-process dropout, and measurement-process dropout all reinforce the same interpretation as the main text: conformal policies preserve coverage by deferring a larger share of difficult cases, whereas fixed thresholds lose recall without exposing uncertainty directly.

Subgroup analyses likewise reinforce a governance-oriented interpretation. Low-acuity strata support broader automation, while high-acuity strata require larger defer regions to retain reliability. These patterns do not invalidate the framework; they define where a cautious deployment policy should be most conservative.

\section{Reproducibility}\label{app:repro}
The package is generated from a fixed artifact bundle. The recommended build order is:
\begin{enumerate}
\item Run \texttt{saaki/deployment\_analysis.py} to generate the canonical artifact bundle in \texttt{local\_outputs/artifacts/}.
\item Run \texttt{saaki/build\_jamia\_manuscript.py} to materialize the chaptered markdown manuscript in \texttt{saaki/jamia\_manuscript/}.
\item Run \texttt{saaki/build\_jamia\_tex.py} to copy the official OUP files, regenerate the figure/table assets, and write \texttt{main.tex}, \texttt{refs.bib}, and the package README.
\end{enumerate}

The selected model snapshot preserved in the artifact bundle is @selected_display with @selected_calibration calibration, mean AUROC @selected_auroc, mean AUPRC @selected_auprc, mean recall at PPV $\geq 0.50$ of @selected_recall, and decision-curve net benefit @decision_nb_020 at threshold 0.20. The source patient-level data are not redistributed with this repository; reproducing the cohort from raw source requires independent credentialed access to MIMIC and the corresponding data-use approvals.

\bibliographystyle{oup-plain}
\bibliography{refs}

\end{appendices}
\end{document}
"""
    )

    substitutions = {
        **context,
        "figure1": figure_block(
            "figure1_workflow.pdf",
            "Cohort definition and evaluation workflow.",
            "fig:workflow",
        ),
        "figure2": figure_block(
            "figure2_prediction_ceiling.png",
            "Prediction ceiling and benchmark summary.",
            "fig:ceiling",
        ),
        "figure3": figure_block(
            "figure3_conformal_flow.pdf",
            "Conformal selective-triage decision flow.",
            "fig:conformal-flow",
        ),
        "figure4": figure_block(
            "figure4_disagreement_flow.pdf",
            "Disagreement-based selective-triage comparator.",
            "fig:disagreement-flow",
        ),
        "figure5": figure_block(
            "figure5_conformal_operating_curve.png",
            "Conformal operating-characteristic curve.",
            "fig:conformal-curve",
        ),
        "figure6": figure_block(
            "figure6_consensus_tradeoff.png",
            "Multi-model consensus trade-off panel.",
            "fig:consensus",
        ),
        "figure7": figure_block(
            "figure7_clinical_utility_panel.png",
            "Calibration and clinical-utility panel.",
            "fig:utility-panel",
        ),
        "figure8": figure_block(
            "figure8_shift_panel.png",
            "Robustness-under-shift panel.",
            "fig:shift",
        ),
        "figure9": figure_block(
            "figure9_subgroup_forest.png",
            "Subgroup coverage and reliability forest plot.",
            "fig:subgroups",
        ),
        "table1": table_block(
            "table1_cohort.tex",
            "Cohort and data-contract summary.",
            "tab:cohort",
        ),
        "table2": table_block(
            "table2_benchmark.tex",
            "Baseline discrimination and calibration benchmark.",
            "tab:benchmark",
            wide=True,
            size="scriptsize",
        ),
        "table3": table_block(
            "table3_conformal.tex",
            "Main conformal results across repeated grouped splits.",
            "tab:conformal-main",
            wide=True,
        ),
        "table4": table_block(
            "table4_consensus.tex",
            "Multi-model consensus conformal results.",
            "tab:conformal-consensus",
            wide=True,
            size="scriptsize",
        ),
        "table5": table_block(
            "table5_shift.tex",
            "Shift robustness versus fixed-threshold policies.",
            "tab:shift",
            wide=True,
        ),
        "table6": table_block(
            "table6_subgroups.tex",
            "Subgroup heterogeneity summary for conformal triage at $\\alpha=0.10$.",
            "tab:subgroups",
            wide=True,
            size="scriptsize",
        ),
        "appendix_b1": table_block(
            "appendix_b1_tree_benchmark.tex",
            "Appendix Table B1. Repeated grouped benchmark summary for the main tree-family comparison.",
            "tab:app-b1",
        ),
        "appendix_b2": table_block(
            "appendix_b2_hpo_overview.tex",
            "Appendix Table B2. Grouped Optuna summary for the LightGBM benchmark.",
            "tab:app-b2",
            wide=True,
        ),
        "appendix_b3": table_block(
            "appendix_b3_hpo_trials.tex",
            "Appendix Table B3. Top grouped Optuna LightGBM trials.",
            "tab:app-b3",
            wide=True,
            size="scriptsize",
        ),
        "appendix_b4": table_block(
            "appendix_b4_feature_ablation.tex",
            "Appendix Table B4. Feature ablation of the selected model.",
            "tab:app-b4",
            wide=True,
            size="scriptsize",
        ),
        "appendix_b5": table_block(
            "appendix_b5_secondary_horizons.tex",
            "Appendix Table B5. Secondary horizon sensitivity analyses.",
            "tab:app-b5",
            wide=True,
            size="scriptsize",
        ),
        "appendix_b6": table_block(
            "appendix_b6_disagreement_summary.tex",
            "Appendix Table B6. Repeated grouped disagreement-based selective-triage summary.",
            "tab:app-b6",
            wide=True,
            size="scriptsize",
        ),
        "appendix_b7": table_block(
            "appendix_b7_disagreement_shift.tex",
            "Appendix Table B7. Disagreement-based shift sensitivity summary.",
            "tab:app-b7",
            wide=True,
            size="scriptsize",
        ),
        "appendix_b8": table_block(
            "appendix_b8_score_summary.tex",
            "Appendix Table B8. Repeated grouped clinical score summary.",
            "tab:app-b8",
            wide=True,
            size="scriptsize",
        ),
        "appendix_b9": table_block(
            "appendix_b9_score_operating.tex",
            "Appendix Table B9. Clinical score operating points on the main grouped split.",
            "tab:app-b9",
            wide=True,
            size="scriptsize",
        ),
        "appendix_b10": table_block(
            "appendix_b10_utility_focus.tex",
            "Appendix Table B10. Key decision-curve net-benefit comparisons.",
            "tab:app-b10",
            wide=True,
            size="scriptsize",
        ),
        "appendix_b11": table_block(
            "appendix_b11_utility_policy.tex",
            "Appendix Table B11. Fixed-policy clinical utility summary.",
            "tab:app-b11",
            wide=True,
            size="scriptsize",
        ),
        "appendix_b12": table_block(
            "appendix_b12_negative_probes.tex",
            "Appendix Table B12. Negative and null probe results carried forward for transparency.",
            "tab:app-b12",
            wide=True,
        ),
        "appendix_fig_b1": figure_block(
            "appendix_b1_roc_pr.png",
            "Appendix Figure B1. ROC and PR benchmark figure.",
            "fig:app-b1",
        ),
        "appendix_fig_b2": figure_block(
            "appendix_b2_calibration_curve.png",
            "Appendix Figure B2. Calibration curve for the deployed baseline.",
            "fig:app-b2",
        ),
        "appendix_fig_b3": figure_block(
            "appendix_b3_decision_curve.png",
            "Appendix Figure B3. Decision curve for benchmark and fixed-policy comparators.",
            "fig:app-b3",
        ),
    }
    write_text(TEX_DIR / "main.tex", template.substitute(substitutions))


def write_readme() -> None:
    readme = """# JAMIA TeX Package

This folder contains a submission-oriented TeX package for the SA-AKI conformal selective-triage manuscript, formatted against the official Oxford University Press authoring template files.

## Contents

- `main.tex`: primary JAMIA manuscript source
- `refs.bib`: paper-specific bibliography
- `oup-authoring-template.cls`, `oup-plain.bst`, `oup-abbrvnat.bst`: official OUP template assets
- `figures/`: numbered main-text and appendix figures
- `tables/`: generated TeX tabular snippets
- `upstream/official/`: retained copy of the downloaded official template inputs

## Regeneration

From the repository root:

```bash
/opt/ml-venv/bin/python saaki/build_jamia_tex.py
```

This script:

1. Copies the official OUP class and bibliography files into `jamia_tex/`.
2. Copies artifact figures and re-renders the workflow diagrams as static PNG/PDF files.
3. Regenerates all table snippets from the canonical artifact bundle.
4. Writes `refs.bib`, `main.tex`, and this README.

## Suggested Build Sequence

If a local TeX engine is available:

```bash
cd jamia_tex
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

`latexmk -pdf main.tex` is also reasonable if installed.

## Notes

- The author, affiliation, and correspondence fields in `main.tex` are placeholders and should be replaced with submission metadata.
- The TeX package is built from the chaptered markdown manuscript in `saaki/jamia_manuscript/` plus the canonical artifacts in `local_outputs/artifacts/`.
- Local PDF compilation was not validated in this environment because a TeX engine was not available.
"""
    write_text(TEX_DIR / "README.md", readme)


def validate_package() -> None:
    required_paths = [
        TEX_DIR / "main.tex",
        TEX_DIR / "refs.bib",
        TEX_DIR / "README.md",
        TEX_DIR / "oup-authoring-template.cls",
        TEX_DIR / "oup-plain.bst",
        TEX_DIR / "oup-abbrvnat.bst",
        FIGURES_DIR / "figure1_workflow.pdf",
        FIGURES_DIR / "figure2_prediction_ceiling.png",
        FIGURES_DIR / "figure9_subgroup_forest.png",
    ]
    required_paths.extend(TABLES_DIR / name for name in MAIN_TABLE_FILES + APPENDIX_TABLE_FILES)
    for section_file in [
        "00_title_and_abstract.md",
        "01_introduction.md",
        "02_methods.md",
        "03_results.md",
        "04_discussion.md",
        "05_limitations.md",
        "06_conclusion.md",
    ]:
        required_paths.append(MANUSCRIPT_DIR / section_file)
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing TeX package assets:\n" + "\n".join(missing))


def main() -> None:
    ensure_dirs()
    copy_template_files()
    copy_data_figures()
    render_workflow_figures()
    payload = load_artifacts()
    write_tables(payload)
    context = build_context(payload)
    write_refs_bib()
    write_main_tex(context)
    write_readme()
    validate_package()
    print("jamia_tex_ready")


if __name__ == "__main__":
    main()
