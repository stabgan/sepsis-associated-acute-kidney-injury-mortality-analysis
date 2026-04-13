"""Deployment-first SA-AKI modeling workflow.

This module implements the strategy captured in the attached plan:
1. audit the current datasets and reconcile schema mismatches;
2. reproduce a leakage-aware baseline on the current v2 cohort;
3. evaluate deployment-oriented operating points with calibration;
4. compare robust tabular baselines and a monotonic guardrail model;
5. package a three-tier clinical policy with subgroup and shift checks.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from scipy.stats import chi2_contingency
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

log = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
LOCAL_OUTPUT_DIR = REPO_ROOT / "local_outputs"
ARTIFACT_DIR = LOCAL_OUTPUT_DIR / "artifacts"
CACHE_DIR = ARTIFACT_DIR / "cache"
CATBOOST_OUTPUT_DIR = LOCAL_OUTPUT_DIR / "catboost_info"

DEFAULT_DATA_PATH = REPO_ROOT / "mimic_saaki_raw_v2.csv"
LEGACY_DATA_PATH = SCRIPT_DIR / "data" / "mimic_saaki_final.csv"

TARGET_COL = "event_observed"
TIME_COL = "time_to_event_hrs"
STAY_ID_COL = "stay_id"
SUBJECT_ID_COL = "subject_id"
HADM_ID_COL = "hadm_id"

PRIMARY_PRECISION_TARGET = 0.50
SECONDARY_NPV_TARGET = 0.95
HIGH_PRECISION_TARGETS = [0.50, 0.60, 0.70, 0.80]
EXPLORATORY_PRECISION_TARGETS = [0.99]
LOW_RISK_TARGETS = [0.95, 0.98]
FIXED_ALERT_RATES = [0.10, 0.20, 0.30]
REPEATED_SEEDS = [42, 123, 404]
OUTER_TEST_SIZE = 0.20
VAL_SIZE_WITHIN_TRAIN = 0.20
BOOTSTRAP_SAMPLES = 200
SELECTIVE_AGREEMENT_QUANTILES = [0.25, 0.50, 0.75]
SHIFT_LEVELS = [0.10]
PERMUTATION_IMPORTANCE_REPEATS = 2
EVALUATE_SECONDARY_HORIZONS = True
FEATURE_ABLATION_ORDER = ["full", "physiology_severity", "physiology_plus_care"]
CONFORMAL_ALPHAS = [0.05, 0.10, 0.15, 0.20]
CONFORMAL_OPERATING_ALPHAS = [round(x, 2) for x in np.arange(0.01, 0.51, 0.01)]
CONFORMAL_REPEATED_SEEDS = [
    42,
    77,
    123,
    202,
    404,
    505,
    606,
    707,
    808,
    909,
    1001,
    1103,
    1207,
    1301,
    1409,
    1511,
    1601,
    1709,
    1801,
    1907,
    2003,
]
CONFORMAL_SHIFT_LEVELS = [0.10, 0.20, 0.30]
CONFORMAL_SUBGROUP_ALPHA = 0.10
JOURNAL_TARGET = "JAMIA"
CLINICAL_UTILITY_THRESHOLDS = [0.10, 0.20, 0.30]

CORRELATION_THRESHOLD = 0.99
MISSINGNESS_DROP_THRESHOLD = 0.99
MISSING_INDICATOR_P_THRESHOLD = 0.05
CALIBRATION_CV_FOLDS = 3
CPU_COUNT = os.cpu_count() or 1

COHORT_TIME_COLS = ["aki_onset_delta_hrs", "sepsis_onset_delta_hrs"]
CARE_PROCESS_COLS = [
    "mechanical_ventilation_24hr_flag",
    "vasopressor_24hr_flag",
    "rrt_24hr_flag",
    "mechvent_time_to_start_hrs",
    "vasopressor_time_to_start_hrs",
]
BASELINE_PROCESS_COLS = ["baseline_scr_estimated_flag"]
PRIMARY_MODELS = ["logistic", "lightgbm", "xgboost"]
ROBUST_MODELS = ["logistic", "lightgbm", "xgboost"]
CALIBRATION_METHODS = ["sigmoid", "isotonic"]
MONOTONIC_FEATURES = {
    "age": 1,
    "apache_iii_score": 1,
    "sofa_total_24hr": 1,
    "bun_median": 1,
    "creatinine_max": 1,
    "lactate_max": 1,
    "fluid_balance_24hr_ml_perkg": 1,
    "urine_output_total_24hr_perkg": -1,
    "pf_min": -1,
    "meanbp_median": -1,
    "spo2_median": -1,
}


@dataclass
class DatasetAudit:
    dataset_path: str
    preferred_dataset: str
    current_rows: int
    current_columns: int
    legacy_rows: int | None
    legacy_columns: int | None
    event_rate: float
    unique_stays: int
    unique_subjects: int
    repeated_subject_rows: int
    repeated_subjects: int
    duplicate_hadm_rows: int
    rrt_prevalence: float
    features_after_gt99_drop: int
    features_after_gt99_and_cohort_time_drop: int
    features_after_obs_count_drop: int
    doc_mismatches: list[str]


@dataclass
class DataContract:
    dataset_name: str
    row_definition: str
    prediction_time: str
    target_definition: str
    time_anchor_definition: str
    identifier_columns: list[str]
    process_feature_families: list[str]
    key_schema_notes: list[str]
    excluded_claims: list[str]


@dataclass
class FeatureSpace:
    original_feature_cols: list[str]
    missing_indicator_sources: list[str]
    correlation_drop_cols: list[str]
    final_feature_cols: list[str]
    numeric_cols: list[str]
    categorical_cols: list[str]
    preprocessor: ColumnTransformer

    def build_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        work = frame.loc[:, self.original_feature_cols].copy()
        for source_col in self.missing_indicator_sources:
            work[f"{source_col}__missing"] = work[source_col].isna().astype(int)
        work = work.drop(columns=self.correlation_drop_cols, errors="ignore")
        return work.loc[:, self.final_feature_cols]

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        built = self.build_frame(frame)
        return self.preprocessor.transform(built)

    def transform_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        built = self.build_frame(frame)
        transformed = self.preprocessor.transform(built)
        return pd.DataFrame(
            transformed,
            columns=self.transformed_feature_names(),
            index=frame.index,
        )

    def transformed_feature_names(self) -> list[str]:
        return list(self.preprocessor.get_feature_names_out())


@dataclass
class ModelSelectionRecord:
    split_name: str
    feature_set: str
    model_name: str
    calibration_method: str
    auroc: float
    auprc: float
    brier: float
    ece: float
    recall_at_ppv_045: float
    recall_at_ppv_050: float
    recall_at_ppv_060: float
    low_risk_coverage_npv_095: float
    low_risk_coverage_npv_098: float


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )


def ensure_artifact_dir() -> None:
    LOCAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    CATBOOST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n")


def write_markdown(path: Path, content: str) -> None:
    path.write_text(content.rstrip() + "\n")


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.int64)):
        return int(value)
    if isinstance(value, (np.floating, np.float64)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if pd.isna(value):
        return None
    raise TypeError(f"Object of type {type(value)!r} is not JSON serializable")


def flatten_columns(frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame.columns, pd.MultiIndex):
        return frame
    frame = frame.copy()
    frame.columns = [
        "__".join(str(part) for part in column if str(part) != "")
        for column in frame.columns.to_flat_index()
    ]
    return frame


def render_table(frame: pd.DataFrame, *, index: bool = True, floatfmt: str = ".3f") -> str:
    try:
        return frame.to_markdown(index=index, floatfmt=floatfmt)
    except ImportError:
        return frame.to_string(index=index)


def resolve_preferred_dataset() -> Path:
    if DEFAULT_DATA_PATH.exists():
        return DEFAULT_DATA_PATH
    if LEGACY_DATA_PATH.exists():
        return LEGACY_DATA_PATH
    raise FileNotFoundError("No SA-AKI CSV found in the repository.")


def cached_parquet_path(csv_path: Path) -> Path:
    return CACHE_DIR / f"{csv_path.stem}.parquet"


def normalize_bool_token(value: Any) -> float | Any:
    if pd.isna(value):
        return np.nan
    token = str(value).strip().lower()
    mapping = {"true": 1.0, "false": 0.0, "1": 1.0, "0": 0.0}
    return mapping.get(token, value)


def coerce_boolean_like_columns(frame: pd.DataFrame) -> pd.DataFrame:
    for col in frame.columns:
        series = frame[col]
        if pd.api.types.is_bool_dtype(series):
            frame[col] = series.astype(float)
            continue
        if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            values = {str(v).strip().lower() for v in series.dropna().unique()}
            if values and values.issubset({"true", "false", "1", "0"}):
                frame[col] = series.map(normalize_bool_token).astype(float)
    return frame


def canonicalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "subject_id_x": SUBJECT_ID_COL,
        "hadm_id_x": HADM_ID_COL,
        "death": TARGET_COL,
        "time_to_death": TIME_COL,
        "wbc_n_y": "wbc_n",
        "sodium_n_y": "sodium_n",
        "potassium_n_y": "potassium_n",
        "rdw_n_y": "rdw_n",
    }
    frame = frame.rename(columns={k: v for k, v in rename_map.items() if k in frame.columns})
    drop_cols = [
        "subject_id_y",
        "hadm_id_y",
        "wbc_n_x",
        "sodium_n_x",
        "potassium_n_x",
        "rdw_n_x",
        "pf_n.1",
    ]
    frame = frame.drop(columns=[c for c in drop_cols if c in frame.columns], errors="ignore")
    frame = coerce_boolean_like_columns(frame)
    return frame


@lru_cache(maxsize=4)
def load_dataset(path: Path) -> pd.DataFrame:
    parquet_path = cached_parquet_path(path)
    if parquet_path.exists() and parquet_path.stat().st_mtime >= path.stat().st_mtime:
        frame = pd.read_parquet(parquet_path)
        return canonicalize_frame(frame)

    frame = pd.read_csv(path)
    frame = canonicalize_frame(frame)
    if TARGET_COL not in frame.columns or TIME_COL not in frame.columns:
        raise ValueError(f"{path} does not expose canonical outcome columns.")
    frame.to_parquet(parquet_path, index=False)
    return frame


def add_secondary_horizon_labels(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["death_within_48h"] = (
        (frame[TARGET_COL] == 1) & (frame[TIME_COL] <= 48.0)
    ).astype(int)
    frame["death_within_7d"] = (
        (frame[TARGET_COL] == 1) & (frame[TIME_COL] <= 168.0)
    ).astype(int)
    frame["death_before_discharge"] = frame[TARGET_COL].astype(int)
    return frame


def build_dataset_audit(current: pd.DataFrame) -> DatasetAudit:
    legacy_rows = None
    legacy_columns = None
    if LEGACY_DATA_PATH.exists():
        legacy_head = pd.read_csv(LEGACY_DATA_PATH, nrows=3)
        legacy_rows = int(pd.read_csv(LEGACY_DATA_PATH, usecols=[0]).shape[0])
        legacy_columns = int(len(legacy_head.columns))

    feature_cols = candidate_feature_columns(current, include_process=True)
    gt99_drop = columns_above_missingness(current, feature_cols, MISSINGNESS_DROP_THRESHOLD)
    after_gt99 = [c for c in feature_cols if c not in gt99_drop]
    after_gt99_and_times = [c for c in after_gt99 if c not in COHORT_TIME_COLS]
    after_obs_count = [c for c in after_gt99_and_times if not is_measurement_process_col(c)]

    vc = current[SUBJECT_ID_COL].value_counts()
    doc_mismatches = [
        "The repo contains two different cohort files: root v2 (10036 rows) and legacy saaki/data/mimic_saaki_final.csv (11661 rows).",
        "The root documentation.md describes renamed target columns (death, time_to_death), but the working v2 CSV still uses event_observed and time_to_event_hrs.",
        "The root documentation says p=162, while the current v2 CSV has 258 raw columns and 223 candidate features after dropping IDs/outcomes and >99% missingness.",
        "The thesis methodology says first-24h RRT patients were excluded, but the v2 CSV has rrt_24hr_flag prevalence around 69%.",
        "The thesis says first ICU stay only, but subject_id repeats across rows in the current v2 cohort, so subject-grouped evaluation is required.",
    ]
    return DatasetAudit(
        dataset_path=str(DEFAULT_DATA_PATH if DEFAULT_DATA_PATH.exists() else resolve_preferred_dataset()),
        preferred_dataset=DEFAULT_DATA_PATH.name if DEFAULT_DATA_PATH.exists() else LEGACY_DATA_PATH.name,
        current_rows=int(current.shape[0]),
        current_columns=int(current.shape[1]),
        legacy_rows=legacy_rows,
        legacy_columns=legacy_columns,
        event_rate=float(current[TARGET_COL].mean()),
        unique_stays=int(current[STAY_ID_COL].nunique()),
        unique_subjects=int(current[SUBJECT_ID_COL].nunique()),
        repeated_subject_rows=int(current[SUBJECT_ID_COL].duplicated().sum()),
        repeated_subjects=int((vc > 1).sum()),
        duplicate_hadm_rows=int(current[HADM_ID_COL].duplicated().sum()),
        rrt_prevalence=float(current["rrt_24hr_flag"].fillna(0).astype(float).mean()),
        features_after_gt99_drop=int(len(after_gt99)),
        features_after_gt99_and_cohort_time_drop=int(len(after_gt99_and_times)),
        features_after_obs_count_drop=int(len(after_obs_count)),
        doc_mismatches=doc_mismatches,
    )


def build_data_contract(audit: DatasetAudit) -> DataContract:
    return DataContract(
        dataset_name=audit.preferred_dataset,
        row_definition=(
            "One ICU-stay-level SA-AKI cohort row scored at T24 using features derived from the "
            "first 24 hours of ICU stay; repeated patients may still appear across multiple rows."
        ),
        prediction_time="24 hours after ICU admission (T24).",
        target_definition=(
            "`event_observed` is the in-hospital death label; `time_to_event_hrs` is hours from ICU "
            "admission to death or last known alive/discharge censoring."
        ),
        time_anchor_definition=(
            "All predictor features must be available by T24; post-T24 information is excluded from "
            "deployment claims even if present elsewhere in project documents."
        ),
        identifier_columns=[STAY_ID_COL, SUBJECT_ID_COL, HADM_ID_COL],
        process_feature_families=[
            "Care-process features such as ventilation, vasopressor, and RRT flags/times.",
            "Measurement-process features such as `*_n`, `*_abn_frac`, `*_stat_frac`, and `*_outofref_frac`.",
        ],
        key_schema_notes=[
            "The working v2 CSV canonicalizes `death` -> `event_observed` and `time_to_death` -> `time_to_event_hrs`.",
            "The v2 cohort is the working source of truth for deployment analysis; the legacy CSV remains historical context only.",
            "Patient-level duplication remains present, so grouped evaluation by `subject_id` is mandatory for deployment claims.",
            "The v2 cohort contains process-rich variables and does not fully match the thesis wording around first-24h RRT exclusion.",
        ],
        excluded_claims=[
            "No bedside-readiness claim.",
            "No external-validation or multi-center transportability claim.",
            "No claim that the v2 cohort exactly matches the thesis cohort description without ETL reconciliation.",
            "No reliance on stay-level metrics for deployment-readiness claims.",
        ],
    )


def candidate_feature_columns(frame: pd.DataFrame, include_process: bool = True) -> list[str]:
    protected = {
        TARGET_COL,
        TIME_COL,
        STAY_ID_COL,
        SUBJECT_ID_COL,
        HADM_ID_COL,
        "death_within_48h",
        "death_within_7d",
        "death_before_discharge",
    }
    cols = [c for c in frame.columns if c not in protected]
    if not include_process:
        cols = [c for c in cols if c not in COHORT_TIME_COLS]
    return cols


def split_frame(
    frame: pd.DataFrame,
    *,
    label_col: str,
    random_state: int,
    use_group_split: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if use_group_split:
        train_outer, test_outer = stratified_group_split(
            frame,
            group_col=SUBJECT_ID_COL,
            label_col=label_col,
            test_size=OUTER_TEST_SIZE,
            random_state=random_state,
        )
        train_model, val_model = stratified_group_split(
            train_outer,
            group_col=SUBJECT_ID_COL,
            label_col=label_col,
            test_size=VAL_SIZE_WITHIN_TRAIN,
            random_state=random_state,
        )
        return train_outer, test_outer, train_model, val_model

    train_outer, test_outer = stay_level_split(
        frame,
        label_col=label_col,
        test_size=OUTER_TEST_SIZE,
        random_state=random_state,
    )
    train_model, val_model = stay_level_split(
        train_outer,
        label_col=label_col,
        test_size=VAL_SIZE_WITHIN_TRAIN,
        random_state=random_state,
    )
    return train_outer, test_outer, train_model, val_model


def columns_above_missingness(
    frame: pd.DataFrame, columns: list[str], threshold: float
) -> list[str]:
    missing = frame.loc[:, columns].isna().mean()
    return missing[missing > threshold].index.tolist()


def is_measurement_process_col(column: str) -> bool:
    return (
        column in BASELINE_PROCESS_COLS
        or column.endswith("_n")
        or column.endswith("_abn_frac")
        or column.endswith("_stat_frac")
        or column.endswith("_outofref_frac")
    )


def build_feature_sets(frame: pd.DataFrame) -> dict[str, list[str]]:
    all_features = candidate_feature_columns(frame, include_process=True)
    gt99_drop = set(columns_above_missingness(frame, all_features, MISSINGNESS_DROP_THRESHOLD))
    usable = [c for c in all_features if c not in gt99_drop and c not in COHORT_TIME_COLS]

    physiology_severity = [
        c
        for c in usable
        if c not in CARE_PROCESS_COLS and not is_measurement_process_col(c)
    ]
    physiology_plus_care = [c for c in usable if not is_measurement_process_col(c)]
    physiology_plus_measurement = [c for c in usable if c not in CARE_PROCESS_COLS]
    return {
        "full": usable,
        "physiology_severity": physiology_severity,
        "physiology_plus_care": physiology_plus_care,
        "physiology_plus_measurement": physiology_plus_measurement,
    }


def stratified_group_split(
    frame: pd.DataFrame,
    *,
    group_col: str,
    label_col: str,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    grouped = frame.groupby(group_col, as_index=False)[label_col].max()
    train_groups, test_groups = train_test_split(
        grouped[group_col],
        test_size=test_size,
        random_state=random_state,
        stratify=grouped[label_col],
    )
    train = frame[frame[group_col].isin(train_groups)].copy()
    test = frame[frame[group_col].isin(test_groups)].copy()
    return train, test


def stay_level_split(
    frame: pd.DataFrame,
    *,
    label_col: str,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_idx, test_idx = train_test_split(
        frame.index,
        test_size=test_size,
        random_state=random_state,
        stratify=frame[label_col],
    )
    return frame.loc[train_idx].copy(), frame.loc[test_idx].copy()


def select_missing_indicator_sources(
    frame: pd.DataFrame, y: pd.Series, columns: list[str]
) -> list[str]:
    selected: list[str] = []
    for col in columns:
        indicator = frame[col].isna().astype(int)
        if indicator.sum() == 0 or indicator.sum() == len(indicator):
            continue
        contingency = pd.crosstab(indicator, y)
        if contingency.shape != (2, 2):
            continue
        _, p_value, _, _ = chi2_contingency(contingency)
        if p_value < MISSING_INDICATOR_P_THRESHOLD:
            selected.append(col)
    return selected


def correlation_drop_columns(frame: pd.DataFrame, numeric_cols: list[str]) -> list[str]:
    if len(numeric_cols) < 2:
        return []
    filled = frame.loc[:, numeric_cols].copy()
    medians = filled.median(numeric_only=True)
    filled = filled.fillna(medians)
    corr = filled.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    return [col for col in upper.columns if any(upper[col] > CORRELATION_THRESHOLD)]


def fit_feature_space(train: pd.DataFrame, feature_cols: list[str], y: pd.Series) -> FeatureSpace:
    raw = train.loc[:, feature_cols].copy()
    missing_sources = select_missing_indicator_sources(raw, y, feature_cols)
    for source_col in missing_sources:
        raw[f"{source_col}__missing"] = raw[source_col].isna().astype(int)

    numeric_raw_cols = raw.select_dtypes(include=[np.number]).columns.tolist()
    correlation_drop = correlation_drop_columns(
        raw, [c for c in numeric_raw_cols if not c.endswith("__missing")]
    )
    final_cols = [c for c in raw.columns if c not in correlation_drop]
    numeric_cols = raw.loc[:, final_cols].select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in final_cols if c not in numeric_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric_cols,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical_cols,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )
    preprocessor.fit(raw.loc[:, final_cols])
    return FeatureSpace(
        original_feature_cols=feature_cols,
        missing_indicator_sources=missing_sources,
        correlation_drop_cols=correlation_drop,
        final_feature_cols=final_cols,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        preprocessor=preprocessor,
    )


def safe_auroc(y_true: pd.Series | np.ndarray, probs: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, probs))


def safe_auprc(y_true: pd.Series | np.ndarray, probs: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, probs))


def expected_calibration_error(
    y_true: pd.Series | np.ndarray, probs: np.ndarray, n_bins: int = 10
) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(probs, bins[1:-1], right=True)
    total = len(probs)
    error = 0.0
    for bin_id in range(n_bins):
        mask = bin_ids == bin_id
        if not mask.any():
            continue
        error += mask.mean() * abs(probs[mask].mean() - np.asarray(y_true)[mask].mean())
    return float(error)


def calibration_intercept_slope(
    y_true: pd.Series | np.ndarray, probs: np.ndarray
) -> dict[str, float]:
    y_array = np.asarray(y_true).astype(int)
    p_array = np.clip(np.asarray(probs, dtype=float), 1e-6, 1.0 - 1e-6)
    if len(np.unique(y_array)) < 2:
        return {"calibration_intercept": float("nan"), "calibration_slope": float("nan")}
    logits = np.log(p_array / (1.0 - p_array)).reshape(-1, 1)
    model = LogisticRegression(
        max_iter=5000,
        solver="lbfgs",
        C=1e6,
        fit_intercept=True,
    )
    model.fit(logits, y_array)
    return {
        "calibration_intercept": float(model.intercept_[0]),
        "calibration_slope": float(model.coef_[0][0]),
    }


def binary_metrics(y_true: pd.Series | np.ndarray, probs: np.ndarray, threshold: float) -> dict[str, float]:
    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    metrics = {
        "auroc": safe_auroc(y_true, probs),
        "auprc": safe_auprc(y_true, probs),
        "brier": float(brier_score_loss(y_true, probs)),
        "ece": expected_calibration_error(y_true, probs),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "specificity": float(specificity),
        "alert_rate": float(preds.mean()),
        "prevalence": float(np.asarray(y_true).mean()),
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
    }
    metrics.update(calibration_intercept_slope(y_true, probs))
    return metrics


def low_risk_metrics(
    y_true: pd.Series | np.ndarray, probs: np.ndarray, threshold: float
) -> dict[str, float]:
    low_preds = np.asarray(probs) <= threshold
    coverage = float(low_preds.mean())
    if coverage == 0.0:
        return {"coverage": 0.0, "npv": float("nan"), "miss_rate": float("nan")}
    y_array = np.asarray(y_true)
    npv = float((y_array[low_preds] == 0).mean())
    miss_rate = float((y_array[low_preds] == 1).mean())
    return {"coverage": coverage, "npv": npv, "miss_rate": miss_rate}


def threshold_search(
    y_true: pd.Series | np.ndarray,
    probs: np.ndarray,
    *,
    precision_target: float | None = None,
    npv_target: float | None = None,
    alert_rate_target: float | None = None,
) -> dict[str, float]:
    thresholds = np.unique(np.clip(np.asarray(probs), 0.0, 1.0))
    best: dict[str, float] | None = None
    for threshold in thresholds:
        high_preds = probs >= threshold
        low_preds = probs <= threshold
        if precision_target is not None:
            precision = precision_score(y_true, high_preds, zero_division=0)
            recall = recall_score(y_true, high_preds, zero_division=0)
            if precision < precision_target:
                continue
            record = {
                "threshold": float(threshold),
                "precision": float(precision),
                "recall": float(recall),
                "alert_rate": float(high_preds.mean()),
            }
            if best is None or record["recall"] > best["recall"] or (
                math.isclose(record["recall"], best["recall"]) and record["alert_rate"] < best["alert_rate"]
            ):
                best = record
        elif npv_target is not None:
            coverage = float(low_preds.mean())
            if coverage == 0.0:
                continue
            npv = float((np.asarray(y_true)[low_preds] == 0).mean())
            if npv < npv_target:
                continue
            record = {"threshold": float(threshold), "npv": float(npv), "coverage": coverage}
            if best is None or record["coverage"] > best["coverage"]:
                best = record
        elif alert_rate_target is not None:
            rate = float(high_preds.mean())
            precision = precision_score(y_true, high_preds, zero_division=0)
            recall = recall_score(y_true, high_preds, zero_division=0)
            distance = abs(rate - alert_rate_target)
            record = {
                "threshold": float(threshold),
                "precision": float(precision),
                "recall": float(recall),
                "alert_rate": rate,
                "distance": float(distance),
            }
            if best is None or record["distance"] < best["distance"]:
                best = record
    if best is None:
        key = (
            "precision_target"
            if precision_target is not None
            else "npv_target"
            if npv_target is not None
            else "alert_rate_target"
        )
        return {"threshold": float("nan"), key: float("nan")}
    return best


def frontier_precision_targets() -> list[float]:
    return HIGH_PRECISION_TARGETS + EXPLORATORY_PRECISION_TARGETS


def flatten_frontier_records(
    *,
    result: dict[str, Any],
    seed: int,
    model_name: str,
    calibration_method: str,
    policy_name: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in result["precision_frontier"]:
        row = {
            "seed": seed,
            "model_name": model_name,
            "calibration_method": calibration_method,
            "policy_name": policy_name,
            "frontier_type": "precision",
            "target": float(item["target"]),
        }
        row.update(item.get("validation", {}))
        row.update({f"test_{k}": v for k, v in item.get("test", {}).items()})
        rows.append(row)
    for item in result["low_risk_frontier"]:
        row = {
            "seed": seed,
            "model_name": model_name,
            "calibration_method": calibration_method,
            "policy_name": policy_name,
            "frontier_type": "low_risk",
            "target": float(item["target"]),
        }
        row.update(item.get("validation", {}))
        row.update({f"test_{k}": v for k, v in item.get("test", {}).items()})
        rows.append(row)
    for item in result["alert_rate_frontier"]:
        row = {
            "seed": seed,
            "model_name": model_name,
            "calibration_method": calibration_method,
            "policy_name": policy_name,
            "frontier_type": "alert_rate",
            "target": float(item["target"]),
        }
        row.update(item.get("validation", {}))
        row.update({f"test_{k}": v for k, v in item.get("test", {}).items()})
        rows.append(row)
    return rows


def decision_curve(y_true: pd.Series | np.ndarray, probs: np.ndarray) -> pd.DataFrame:
    thresholds = np.arange(0.05, 0.51, 0.01)
    prevalence = float(np.asarray(y_true).mean())
    rows: list[dict[str, float]] = []
    for threshold in thresholds:
        preds = probs >= threshold
        tp = np.logical_and(preds, np.asarray(y_true) == 1).sum()
        fp = np.logical_and(preds, np.asarray(y_true) == 0).sum()
        n = len(y_true)
        harm = threshold / (1.0 - threshold)
        rows.append(
            {
                "threshold": float(threshold),
                "model": float(tp / n - fp / n * harm),
                "treat_all": float(prevalence - (1.0 - prevalence) * harm),
                "treat_none": 0.0,
            }
        )
    return pd.DataFrame(rows)


def net_benefit(y_true: pd.Series | np.ndarray, alert_mask: np.ndarray, threshold: float) -> dict[str, float]:
    y_array = np.asarray(y_true).astype(int)
    alerts = np.asarray(alert_mask, dtype=bool)
    n = len(y_array)
    tp = int(np.logical_and(alerts, y_array == 1).sum())
    fp = int(np.logical_and(alerts, y_array == 0).sum())
    harm = threshold / (1.0 - threshold)
    prevalence = float(y_array.mean())
    net_benefit_value = float(tp / n - fp / n * harm)
    standardized = float(net_benefit_value / prevalence) if prevalence > 0 else float("nan")
    return {
        "net_benefit": net_benefit_value,
        "standardized_net_benefit": standardized,
        "alert_rate": float(alerts.mean()),
        "tp": float(tp),
        "fp": float(fp),
    }


def decision_curve_from_probs(
    y_true: pd.Series | np.ndarray,
    probs: np.ndarray,
    *,
    strategy_name: str,
    thresholds: list[float] | np.ndarray | None = None,
) -> pd.DataFrame:
    threshold_grid = np.asarray(thresholds if thresholds is not None else np.arange(0.05, 0.51, 0.01), dtype=float)
    rows: list[dict[str, float | str]] = []
    prevalence = float(np.asarray(y_true).mean())
    for threshold in threshold_grid:
        metrics = net_benefit(y_true, np.asarray(probs) >= threshold, float(threshold))
        rows.append(
            {
                "strategy": strategy_name,
                "strategy_type": "continuous_risk",
                "threshold": float(threshold),
                "prevalence": prevalence,
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def decision_curve_from_alert_mask(
    y_true: pd.Series | np.ndarray,
    alert_mask: np.ndarray,
    *,
    strategy_name: str,
    thresholds: list[float] | np.ndarray | None = None,
) -> pd.DataFrame:
    threshold_grid = np.asarray(thresholds if thresholds is not None else np.arange(0.05, 0.51, 0.01), dtype=float)
    rows: list[dict[str, float | str]] = []
    prevalence = float(np.asarray(y_true).mean())
    for threshold in threshold_grid:
        metrics = net_benefit(y_true, alert_mask, float(threshold))
        rows.append(
            {
                "strategy": strategy_name,
                "strategy_type": "fixed_policy",
                "threshold": float(threshold),
                "prevalence": prevalence,
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def summarize_metric_intervals(
    frame: pd.DataFrame,
    *,
    group_cols: list[str],
    value_cols: list[str],
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=group_cols + value_cols)
    rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {column: key for column, key in zip(group_cols, keys)}
        row["n_groups"] = int(len(group))
        for value_col in value_cols:
            values = pd.Series(group[value_col]).dropna().astype(float)
            if values.empty:
                row[f"{value_col}_mean"] = float("nan")
                row[f"{value_col}_std"] = float("nan")
                row[f"{value_col}_ci_low"] = float("nan")
                row[f"{value_col}_ci_high"] = float("nan")
                continue
            row[f"{value_col}_mean"] = float(values.mean())
            row[f"{value_col}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            row[f"{value_col}_ci_low"] = float(values.quantile(0.025))
            row[f"{value_col}_ci_high"] = float(values.quantile(0.975))
        rows.append(row)
    return pd.DataFrame(rows).reset_index(drop=True)


def build_estimator(model_name: str, y_train: pd.Series, random_state: int) -> Any:
    positives = max(float(y_train.sum()), 1.0)
    negatives = max(float(len(y_train) - y_train.sum()), 1.0)
    scale_pos_weight = negatives / positives

    if model_name == "logistic":
        return LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=random_state,
        )
    if model_name == "lightgbm":
        return LGBMClassifier(
            objective="binary",
            n_estimators=350,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_samples=20,
            reg_alpha=0.1,
            reg_lambda=1.0,
            class_weight="balanced",
            n_jobs=CPU_COUNT,
            random_state=random_state,
            verbose=-1,
        )
    if model_name == "xgboost":
        return XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            n_estimators=350,
            learning_rate=0.05,
            max_depth=4,
            min_child_weight=1.0,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.0,
            reg_lambda=1.0,
            scale_pos_weight=scale_pos_weight,
            tree_method="hist",
            n_jobs=CPU_COUNT,
            random_state=random_state,
        )
    if model_name == "catboost":
        return CatBoostClassifier(
            iterations=500,
            learning_rate=0.04,
            depth=6,
            loss_function="Logloss",
            eval_metric="AUC",
            verbose=False,
            random_seed=random_state,
            auto_class_weights="Balanced",
            thread_count=-1,
            train_dir=str(CATBOOST_OUTPUT_DIR),
        )
    raise ValueError(f"Unsupported model: {model_name}")


def build_monotonic_model(
    random_state: int, feature_names: list[str] | None = None
) -> HistGradientBoostingClassifier:
    active_features = feature_names or list(MONOTONIC_FEATURES)
    monotonic_constraints = [MONOTONIC_FEATURES[name] for name in active_features]
    return HistGradientBoostingClassifier(
        max_depth=3,
        max_iter=350,
        learning_rate=0.05,
        min_samples_leaf=20,
        l2_regularization=0.1,
        monotonic_cst=monotonic_constraints,
        class_weight="balanced",
        random_state=random_state,
    )


def fit_calibrated_model(
    model_name: str,
    calibration_method: str,
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series,
    random_state: int,
) -> Any:
    if model_name == "monotonic_hgbt":
        feature_names = list(X_train.columns) if isinstance(X_train, pd.DataFrame) else None
        estimator = build_monotonic_model(random_state, feature_names=feature_names)
    else:
        estimator = build_estimator(model_name, y_train, random_state)
    if calibration_method == "none":
        estimator.fit(X_train, y_train)
        return estimator
    calibrated = CalibratedClassifierCV(
        estimator=estimator,
        method=calibration_method,
        cv=CALIBRATION_CV_FOLDS,
    )
    calibrated.fit(X_train, y_train)
    return calibrated


def predict_probabilities(model: Any, X: np.ndarray) -> np.ndarray:
    probs = model.predict_proba(X)
    if probs.ndim == 2:
        return np.asarray(probs[:, 1], dtype=float)
    return np.asarray(probs, dtype=float)


def fit_and_score_candidate(
    *,
    model_name: str,
    calibration_method: str,
    X_train: np.ndarray,
    y_train: pd.Series,
    X_eval: np.ndarray,
    y_eval: pd.Series,
    random_state: int,
    split_name: str,
    feature_set: str,
) -> ModelSelectionRecord:
    fitted = fit_calibrated_model(model_name, calibration_method, X_train, y_train, random_state)
    eval_probs = predict_probabilities(fitted, X_eval)
    metrics = binary_metrics(y_eval, eval_probs, threshold=0.50)
    rec_050 = threshold_search(y_eval, eval_probs, precision_target=0.50)
    rec_060 = threshold_search(y_eval, eval_probs, precision_target=0.60)
    low_095 = threshold_search(y_eval, eval_probs, npv_target=0.95)
    low_098 = threshold_search(y_eval, eval_probs, npv_target=0.98)
    return ModelSelectionRecord(
        split_name=split_name,
        feature_set=feature_set,
        model_name=model_name,
        calibration_method=calibration_method,
        auroc=metrics["auroc"],
        auprc=metrics["auprc"],
        brier=metrics["brier"],
        ece=metrics["ece"],
        recall_at_ppv_045=float("nan"),
        recall_at_ppv_050=float(rec_050.get("recall", float("nan"))),
        recall_at_ppv_060=float(rec_060.get("recall", float("nan"))),
        low_risk_coverage_npv_095=float(low_095.get("coverage", float("nan"))),
        low_risk_coverage_npv_098=float(low_098.get("coverage", float("nan"))),
    )


def monotonic_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    available = [name for name in MONOTONIC_FEATURES if name in frame.columns]
    matrix = frame.loc[:, available].copy()
    return matrix.fillna(matrix.median(numeric_only=True))


def choose_best_calibration_per_model(
    frame: pd.DataFrame,
    feature_cols: list[str],
    *,
    split_name: str,
    random_state: int,
    use_group_split: bool,
) -> pd.DataFrame:
    train_outer, test_outer, train_model, val_model = split_frame(
        frame,
        label_col=TARGET_COL,
        random_state=random_state,
        use_group_split=use_group_split,
    )

    feature_space = fit_feature_space(train_model, feature_cols, train_model[TARGET_COL])
    X_train = feature_space.transform_frame(train_model)
    X_val = feature_space.transform_frame(val_model)

    records: list[ModelSelectionRecord] = []
    for model_name in PRIMARY_MODELS:
        for calibration_method in CALIBRATION_METHODS:
            record = fit_and_score_candidate(
                model_name=model_name,
                calibration_method=calibration_method,
                X_train=X_train,
                y_train=train_model[TARGET_COL],
                X_eval=X_val,
                y_eval=val_model[TARGET_COL],
                random_state=random_state,
                split_name=split_name,
                feature_set="full",
            )
            records.append(record)

    benchmark = pd.DataFrame([asdict(record) for record in records]).sort_values(
        ["model_name", "brier", "auroc"], ascending=[True, True, False]
    )
    return benchmark.reset_index(drop=True)


def select_calibration_map(benchmark: pd.DataFrame) -> dict[str, str]:
    chosen: dict[str, str] = {}
    for model_name, group in benchmark.groupby("model_name"):
        group = group.sort_values(["brier", "auroc"], ascending=[True, False])
        chosen[model_name] = str(group.iloc[0]["calibration_method"])
    return chosen


def evaluate_on_outer_split(
    frame: pd.DataFrame,
    feature_cols: list[str],
    *,
    model_name: str,
    calibration_method: str,
    random_state: int,
    use_group_split: bool,
    label_col: str = TARGET_COL,
) -> dict[str, Any]:
    train_outer, test_outer, train_model, val_model = split_frame(
        frame,
        label_col=label_col,
        random_state=random_state,
        use_group_split=use_group_split,
    )

    if model_name == "monotonic_hgbt":
        X_train_model = monotonic_matrix(train_model)
        X_val_model = monotonic_matrix(val_model)
        X_train_outer = monotonic_matrix(train_outer)
        X_test_outer = monotonic_matrix(test_outer)
        fitted_for_thresholds = fit_calibrated_model(
            model_name, calibration_method, X_train_model.to_numpy(), train_model[label_col], random_state
        )
        val_probs = predict_probabilities(fitted_for_thresholds, X_val_model.to_numpy())
        fitted_final = fit_calibrated_model(
            model_name, calibration_method, X_train_outer.to_numpy(), train_outer[label_col], random_state
        )
        test_probs = predict_probabilities(fitted_final, X_test_outer.to_numpy())
        feature_space = None
        transformed_names = list(MONOTONIC_FEATURES)
        X_test_for_importance = X_test_outer.to_numpy()
    else:
        feature_space = fit_feature_space(train_model, feature_cols, train_model[label_col])
        X_train_model = feature_space.transform_frame(train_model)
        X_val_model = feature_space.transform_frame(val_model)
        fitted_for_thresholds = fit_calibrated_model(
            model_name, calibration_method, X_train_model, train_model[label_col], random_state
        )
        val_probs = predict_probabilities(fitted_for_thresholds, X_val_model)

        final_feature_space = fit_feature_space(train_outer, feature_cols, train_outer[label_col])
        X_train_outer = final_feature_space.transform_frame(train_outer)
        X_test_outer = final_feature_space.transform_frame(test_outer)
        fitted_final = fit_calibrated_model(
            model_name, calibration_method, X_train_outer, train_outer[label_col], random_state
        )
        test_probs = predict_probabilities(fitted_final, X_test_outer)
        feature_space = final_feature_space
        transformed_names = feature_space.transformed_feature_names()
        X_test_for_importance = X_test_outer

    primary_threshold = threshold_search(
        val_model[label_col], val_probs, precision_target=PRIMARY_PRECISION_TARGET
    )
    secondary_threshold = threshold_search(
        val_model[label_col], val_probs, npv_target=SECONDARY_NPV_TARGET
    )

    if math.isnan(primary_threshold.get("threshold", float("nan"))):
        primary_threshold["threshold"] = 0.50
    if math.isnan(secondary_threshold.get("threshold", float("nan"))):
        secondary_threshold["threshold"] = 0.10

    default_metrics = binary_metrics(test_outer[label_col], test_probs, threshold=primary_threshold["threshold"])
    precision_frontier = [
        {
            "target": target,
            "validation": threshold_search(val_model[label_col], val_probs, precision_target=target),
        }
        for target in frontier_precision_targets()
    ]
    for item in precision_frontier:
        threshold = item["validation"].get("threshold", float("nan"))
        if not math.isnan(threshold):
            item["test"] = binary_metrics(test_outer[label_col], test_probs, threshold=threshold)
        else:
            item["test"] = {}

    low_risk_frontier = [
        {
            "target": target,
            "validation": threshold_search(val_model[label_col], val_probs, npv_target=target),
        }
        for target in LOW_RISK_TARGETS
    ]
    for item in low_risk_frontier:
        threshold = item["validation"].get("threshold", float("nan"))
        if not math.isnan(threshold):
            low_preds = test_probs <= threshold
            coverage = float(low_preds.mean())
            npv = float((np.asarray(test_outer[label_col])[low_preds] == 0).mean()) if coverage else float("nan")
            item["test"] = {"coverage": coverage, "npv": npv}
        else:
            item["test"] = {}

    alert_rate_frontier = [
        {
            "target": target,
            "validation": threshold_search(val_model[label_col], val_probs, alert_rate_target=target),
        }
        for target in FIXED_ALERT_RATES
    ]
    for item in alert_rate_frontier:
        threshold = item["validation"].get("threshold", float("nan"))
        if not math.isnan(threshold):
            item["test"] = binary_metrics(test_outer[label_col], test_probs, threshold=threshold)
        else:
            item["test"] = {}

    low_threshold = secondary_threshold["threshold"]
    high_threshold = primary_threshold["threshold"]
    if high_threshold <= low_threshold:
        low_threshold = min(low_threshold, high_threshold - 1e-6)

    tiers = np.where(
        test_probs >= high_threshold,
        "High risk",
        np.where(test_probs <= low_threshold, "Low risk", "Uncertain"),
    )
    policy_rows = []
    for tier in ["High risk", "Uncertain", "Low risk"]:
        mask = tiers == tier
        if not mask.any():
            policy_rows.append({"tier": tier, "coverage": 0.0, "event_rate": float("nan")})
            continue
        policy_rows.append(
            {
                "tier": tier,
                "coverage": float(mask.mean()),
                "event_rate": float(test_outer.loc[mask, label_col].mean()),
                "count": int(mask.sum()),
            }
        )

    return {
        "train_outer": train_outer,
        "test_outer": test_outer,
        "val_model": val_model,
        "test_probs": test_probs,
        "val_probs": val_probs,
        "model": fitted_final,
        "feature_space": feature_space,
        "transformed_feature_names": transformed_names,
        "X_test_for_importance": X_test_for_importance,
        "metrics": default_metrics,
        "primary_threshold": primary_threshold,
        "secondary_threshold": secondary_threshold,
        "precision_frontier": precision_frontier,
        "low_risk_frontier": low_risk_frontier,
        "alert_rate_frontier": alert_rate_frontier,
        "policy_rows": policy_rows,
        "tiers": tiers,
    }


def evaluate_score_baseline_on_outer_split(
    frame: pd.DataFrame,
    score_col: str,
    *,
    random_state: int,
    use_group_split: bool,
    label_col: str = TARGET_COL,
) -> dict[str, Any]:
    train_outer, test_outer, train_model, val_model = split_frame(
        frame,
        label_col=label_col,
        random_state=random_state,
        use_group_split=use_group_split,
    )
    train_fill = float(train_model[score_col].median())
    outer_fill = float(train_outer[score_col].median())

    threshold_model = LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=random_state,
    )
    X_train_model = train_model[[score_col]].fillna(train_fill)
    X_val_model = val_model[[score_col]].fillna(train_fill)
    threshold_model.fit(X_train_model, train_model[label_col])
    val_probs = predict_probabilities(threshold_model, X_val_model)

    final_model = LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=random_state,
    )
    X_train_outer = train_outer[[score_col]].fillna(outer_fill)
    X_test_outer = test_outer[[score_col]].fillna(outer_fill)
    final_model.fit(X_train_outer, train_outer[label_col])
    test_probs = predict_probabilities(final_model, X_test_outer)

    primary_threshold = threshold_search(
        val_model[label_col], val_probs, precision_target=PRIMARY_PRECISION_TARGET
    )
    if math.isnan(primary_threshold.get("threshold", float("nan"))):
        primary_threshold["threshold"] = 0.50
    metrics = binary_metrics(test_outer[label_col], test_probs, threshold=primary_threshold["threshold"])
    return {
        "score_name": score_col,
        "train_outer": train_outer,
        "test_outer": test_outer,
        "val_probs": val_probs,
        "test_probs": test_probs,
        "model": final_model,
        "metrics": metrics,
        "primary_threshold": primary_threshold,
    }


def fit_feature_model_bundle(
    *,
    train_model: pd.DataFrame,
    val_model: pd.DataFrame,
    train_outer: pd.DataFrame,
    test_outer: pd.DataFrame,
    feature_cols: list[str],
    model_name: str,
    calibration_method: str,
    label_col: str,
    random_state: int,
) -> dict[str, Any]:
    feature_space = fit_feature_space(train_model, feature_cols, train_model[label_col])
    X_train_model = feature_space.transform_frame(train_model)
    X_val_model = feature_space.transform_frame(val_model)
    fitted_for_thresholds = fit_calibrated_model(
        model_name, calibration_method, X_train_model, train_model[label_col], random_state
    )
    val_probs = predict_probabilities(fitted_for_thresholds, X_val_model)

    final_feature_space = fit_feature_space(train_outer, feature_cols, train_outer[label_col])
    X_train_outer = final_feature_space.transform_frame(train_outer)
    X_test_outer = final_feature_space.transform_frame(test_outer)
    fitted_final = fit_calibrated_model(
        model_name, calibration_method, X_train_outer, train_outer[label_col], random_state
    )
    test_probs = predict_probabilities(fitted_final, X_test_outer)
    return {
        "validation_model": fitted_for_thresholds,
        "final_model": fitted_final,
        "validation_probs": val_probs,
        "test_probs": test_probs,
        "validation_features": X_val_model,
        "test_features": X_test_outer,
        "feature_space": final_feature_space,
    }


def repeated_clinical_score_baselines(
    frame: pd.DataFrame,
    score_cols: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for score_col in score_cols:
        if score_col not in frame.columns:
            continue
        for seed in REPEATED_SEEDS:
            result = evaluate_score_baseline_on_outer_split(
                frame,
                score_col,
                random_state=seed,
                use_group_split=True,
            )
            rows.append(
                {
                    "seed": seed,
                    "model_name": score_col,
                    "model_family": "clinical_score",
                    **result["metrics"],
                    "recall_at_ppv_050": float(result["primary_threshold"].get("recall", float("nan"))),
                    "precision_at_ppv_050_threshold": float(result["primary_threshold"].get("precision", float("nan"))),
                    "threshold_ppv_050": float(result["primary_threshold"].get("threshold", float("nan"))),
                }
            )
    return pd.DataFrame(rows)


def clinical_score_operating_points(
    frame: pd.DataFrame,
    score_cols: list[str],
    *,
    random_state: int = 42,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for score_col in score_cols:
        if score_col not in frame.columns:
            continue
        result = evaluate_score_baseline_on_outer_split(
            frame,
            score_col,
            random_state=random_state,
            use_group_split=True,
        )
        rows.append(
            {
                "score_name": score_col,
                **result["metrics"],
                "threshold_ppv_050": float(result["primary_threshold"].get("threshold", float("nan"))),
                "precision_at_ppv_050_threshold": float(result["primary_threshold"].get("precision", float("nan"))),
                "recall_at_ppv_050_threshold": float(result["primary_threshold"].get("recall", float("nan"))),
                "alert_rate_at_ppv_050_threshold": float(result["primary_threshold"].get("alert_rate", float("nan"))),
            }
        )
    return pd.DataFrame(rows)


def evaluate_policy_masks(
    y_true: pd.Series | np.ndarray, high_mask: np.ndarray, low_mask: np.ndarray
) -> dict[str, float]:
    y_array = np.asarray(y_true).astype(int)
    high_mask = np.asarray(high_mask, dtype=bool)
    low_mask = np.asarray(low_mask, dtype=bool)
    low_mask = np.logical_and(low_mask, ~high_mask)
    defer_mask = ~(high_mask | low_mask)

    high_count = int(high_mask.sum())
    low_count = int(low_mask.sum())
    actionable_count = high_count + low_count
    actionable_coverage = actionable_count / len(y_array)

    true_positive_alerts = int(np.logical_and(high_mask, y_array == 1).sum())
    false_positive_alerts = int(np.logical_and(high_mask, y_array == 0).sum())
    false_negative_clears = int(np.logical_and(low_mask, y_array == 1).sum())

    alert_precision = (
        true_positive_alerts / high_count if high_count else float("nan")
    )
    alert_recall = (
        true_positive_alerts / int((y_array == 1).sum()) if int((y_array == 1).sum()) else float("nan")
    )
    clear_npv = (
        float((y_array[low_mask] == 0).mean()) if low_count else float("nan")
    )

    errors_on_actionable = false_positive_alerts + false_negative_clears
    actionable_error_rate = (
        errors_on_actionable / actionable_count if actionable_count else float("nan")
    )
    return {
        "alert_rate": float(high_mask.mean()),
        "low_risk_coverage": float(low_mask.mean()),
        "defer_rate": float(defer_mask.mean()),
        "actionable_coverage": float(actionable_coverage),
        "alert_precision": float(alert_precision),
        "alert_recall": float(alert_recall),
        "low_risk_npv": float(clear_npv),
        "defer_event_rate": float(y_array[defer_mask].mean()) if defer_mask.any() else float("nan"),
        "actionable_error_rate": float(actionable_error_rate),
        "high_count": float(high_count),
        "low_count": float(low_count),
        "defer_count": float(defer_mask.sum()),
    }


def evaluate_selective_triage_on_split(
    frame: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    *,
    model_name: str,
    calibration_method: str,
    random_state: int,
    label_col: str = TARGET_COL,
) -> dict[str, Any]:
    train_outer, test_outer, train_model, val_model = split_frame(
        frame,
        label_col=label_col,
        random_state=random_state,
        use_group_split=True,
    )

    full_bundle = fit_feature_model_bundle(
        train_model=train_model,
        val_model=val_model,
        train_outer=train_outer,
        test_outer=test_outer,
        feature_cols=feature_sets["full"],
        model_name=model_name,
        calibration_method=calibration_method,
        label_col=label_col,
        random_state=random_state,
    )
    phys_bundle = fit_feature_model_bundle(
        train_model=train_model,
        val_model=val_model,
        train_outer=train_outer,
        test_outer=test_outer,
        feature_cols=feature_sets["physiology_severity"],
        model_name=model_name,
        calibration_method=calibration_method,
        label_col=label_col,
        random_state=random_state,
    )

    primary_threshold = threshold_search(
        val_model[label_col], full_bundle["validation_probs"], precision_target=PRIMARY_PRECISION_TARGET
    )
    secondary_threshold = threshold_search(
        val_model[label_col], full_bundle["validation_probs"], npv_target=SECONDARY_NPV_TARGET
    )
    if math.isnan(primary_threshold.get("threshold", float("nan"))):
        primary_threshold["threshold"] = 0.50
    if math.isnan(secondary_threshold.get("threshold", float("nan"))):
        secondary_threshold["threshold"] = 0.10

    high_threshold = float(primary_threshold["threshold"])
    low_threshold = float(secondary_threshold["threshold"])
    if high_threshold <= low_threshold:
        low_threshold = min(low_threshold, high_threshold - 1e-6)

    val_disagreement = np.abs(full_bundle["validation_probs"] - phys_bundle["validation_probs"])
    fixed_val_high = full_bundle["validation_probs"] >= high_threshold
    fixed_val_low = full_bundle["validation_probs"] <= low_threshold

    selected_policy: dict[str, float] | None = None
    policy_grid: list[dict[str, float]] = []
    for quantile in SELECTIVE_AGREEMENT_QUANTILES:
        tau = float(np.quantile(val_disagreement, quantile))
        agreement_mask = val_disagreement <= tau
        selective_metrics = evaluate_policy_masks(
            val_model[label_col],
            np.logical_and(fixed_val_high, agreement_mask),
            np.logical_and(fixed_val_low, agreement_mask),
        )
        record = {
            "agreement_quantile": float(quantile),
            "agreement_threshold": tau,
            **selective_metrics,
        }
        policy_grid.append(record)
        feasible = (
            not math.isnan(record["alert_precision"])
            and record["alert_precision"] >= PRIMARY_PRECISION_TARGET
            and (math.isnan(record["low_risk_npv"]) or record["low_risk_npv"] >= SECONDARY_NPV_TARGET)
        )
        if not feasible:
            continue
        if selected_policy is None or (
            record["actionable_coverage"],
            record["alert_recall"],
        ) > (
            selected_policy["actionable_coverage"],
            selected_policy["alert_recall"],
        ):
            selected_policy = record

    if selected_policy is None:
        selected_policy = max(policy_grid, key=lambda row: row["actionable_coverage"])

    test_disagreement = np.abs(full_bundle["test_probs"] - phys_bundle["test_probs"])
    agreement_test = test_disagreement <= selected_policy["agreement_threshold"]
    fixed_metrics = evaluate_policy_masks(
        test_outer[label_col],
        full_bundle["test_probs"] >= high_threshold,
        full_bundle["test_probs"] <= low_threshold,
    )
    selective_metrics = evaluate_policy_masks(
        test_outer[label_col],
        np.logical_and(full_bundle["test_probs"] >= high_threshold, agreement_test),
        np.logical_and(full_bundle["test_probs"] <= low_threshold, agreement_test),
    )
    return {
        "train_outer": train_outer,
        "test_outer": test_outer,
        "val_model": val_model,
        "full_bundle": full_bundle,
        "phys_bundle": phys_bundle,
        "primary_threshold": primary_threshold,
        "secondary_threshold": secondary_threshold,
        "fixed_policy_metrics": fixed_metrics,
        "selective_policy_metrics": selective_metrics,
        "agreement_threshold": float(selected_policy["agreement_threshold"]),
        "agreement_quantile": float(selected_policy["agreement_quantile"]),
        "validation_policy_grid": pd.DataFrame(policy_grid),
        "test_disagreement": test_disagreement,
    }


def repeated_group_model_comparison(
    frame: pd.DataFrame, feature_sets: dict[str, list[str]], calibration_map: dict[str, str]
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for seed in REPEATED_SEEDS:
        for model_name in ROBUST_MODELS:
            calibration = calibration_map.get(model_name, "sigmoid")
            result = evaluate_on_outer_split(
                frame,
                feature_sets["full"],
                model_name=model_name,
                calibration_method=calibration,
                random_state=seed,
                use_group_split=True,
            )
            record = {
                "seed": seed,
                "model_name": model_name,
                "calibration_method": calibration,
                **result["metrics"],
                "recall_at_ppv_050": float(result["primary_threshold"].get("recall", float("nan"))),
                "precision_at_ppv_050_threshold": float(result["primary_threshold"].get("precision", float("nan"))),
                "threshold_ppv_050": float(result["primary_threshold"].get("threshold", float("nan"))),
            }
            records.append(record)
            log.info(
                "Repeated split seed=%s model=%s AUROC=%.3f recall@PPV0.50=%.3f",
                seed,
                model_name,
                record["auroc"],
                record["recall_at_ppv_050"],
            )
    return pd.DataFrame(records)


def repeated_group_frontier(
    frame: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    *,
    model_name: str,
    calibration_method: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for seed in REPEATED_SEEDS:
        result = evaluate_on_outer_split(
            frame,
            feature_sets["full"],
            model_name=model_name,
            calibration_method=calibration_method,
            random_state=seed,
            use_group_split=True,
        )
        rows.extend(
            flatten_frontier_records(
                result=result,
                seed=seed,
                model_name=model_name,
                calibration_method=calibration_method,
                policy_name="fixed_threshold",
            )
        )
    return pd.DataFrame(rows)


def repeated_selective_triage(
    frame: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    *,
    model_name: str,
    calibration_method: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for seed in REPEATED_SEEDS:
        result = evaluate_selective_triage_on_split(
            frame,
            feature_sets,
            model_name=model_name,
            calibration_method=calibration_method,
            random_state=seed,
        )
        rows.append(
            {
                "seed": seed,
                "model_name": model_name,
                "calibration_method": calibration_method,
                "policy_name": "fixed_threshold",
                **result["fixed_policy_metrics"],
                "agreement_quantile": 1.0,
                "agreement_threshold": float("inf"),
            }
        )
        rows.append(
            {
                "seed": seed,
                "model_name": model_name,
                "calibration_method": calibration_method,
                "policy_name": "selective_triage",
                **result["selective_policy_metrics"],
                "agreement_quantile": result["agreement_quantile"],
                "agreement_threshold": result["agreement_threshold"],
            }
        )
    return pd.DataFrame(rows)


def fit_conformal_bundle(
    train_model: pd.DataFrame,
    cal_model: pd.DataFrame,
    test_outer: pd.DataFrame,
    *,
    feature_cols: list[str],
    model_name: str,
    calibration_method: str,
    label_col: str,
    random_state: int,
) -> dict[str, Any]:
    feature_space = fit_feature_space(train_model, feature_cols, train_model[label_col])
    X_train = feature_space.transform_frame(train_model)
    X_cal = feature_space.transform_frame(cal_model)
    X_test = feature_space.transform_frame(test_outer)
    fitted_model = fit_calibrated_model(
        model_name,
        calibration_method,
        X_train,
        train_model[label_col],
        random_state,
    )
    return {
        "model": fitted_model,
        "feature_space": feature_space,
        "train_frame": train_model,
        "calibration_frame": cal_model,
        "test_frame": test_outer,
        "calibration_probs": predict_probabilities(fitted_model, X_cal),
        "test_probs": predict_probabilities(fitted_model, X_test),
        "train_features": X_train,
        "calibration_features": X_cal,
        "test_features": X_test,
    }


def split_conformal_prediction_sets(
    calibration_probs: np.ndarray,
    calibration_labels: pd.Series | np.ndarray,
    test_probs: np.ndarray,
    alpha: float,
) -> tuple[list[set[int]], dict[str, float]]:
    cal_y = np.asarray(calibration_labels).astype(int)
    n_cal = len(cal_y)
    scores = np.where(cal_y == 1, 1.0 - calibration_probs, calibration_probs)
    q_level = min(np.ceil((n_cal + 1) * (1.0 - alpha)) / n_cal, 1.0)
    q_hat = float(np.quantile(scores, q_level))
    pred_sets: list[set[int]] = []
    for prob in np.asarray(test_probs, dtype=float):
        label_set: set[int] = set()
        if prob <= q_hat:
            label_set.add(0)
        if (1.0 - prob) <= q_hat:
            label_set.add(1)
        pred_sets.append(label_set)
    return pred_sets, {"q_hat": q_hat}


def mondrian_conformal_prediction_sets(
    calibration_probs: np.ndarray,
    calibration_labels: pd.Series | np.ndarray,
    test_probs: np.ndarray,
    alpha: float,
) -> tuple[list[set[int]], dict[str, float]]:
    cal_y = np.asarray(calibration_labels).astype(int)
    scores_0 = calibration_probs[cal_y == 0]
    scores_1 = 1.0 - calibration_probs[cal_y == 1]
    n0 = max(len(scores_0), 1)
    n1 = max(len(scores_1), 1)
    q0 = float(np.quantile(scores_0, min(np.ceil((n0 + 1) * (1.0 - alpha)) / n0, 1.0))) if len(scores_0) else float("nan")
    q1 = float(np.quantile(scores_1, min(np.ceil((n1 + 1) * (1.0 - alpha)) / n1, 1.0))) if len(scores_1) else float("nan")
    pred_sets: list[set[int]] = []
    for prob in np.asarray(test_probs, dtype=float):
        label_set: set[int] = set()
        if not math.isnan(q0) and prob <= q0:
            label_set.add(0)
        if not math.isnan(q1) and (1.0 - prob) <= q1:
            label_set.add(1)
        pred_sets.append(label_set)
    return pred_sets, {"q0": q0, "q1": q1}


def evaluate_conformal_sets(
    pred_sets: list[set[int]],
    y_true: pd.Series | np.ndarray,
) -> dict[str, float]:
    y_array = np.asarray(y_true).astype(int)
    certain_clear_idx = [idx for idx, label_set in enumerate(pred_sets) if label_set == {0}]
    certain_alert_idx = [idx for idx, label_set in enumerate(pred_sets) if label_set == {1}]
    uncertain_idx = [idx for idx, label_set in enumerate(pred_sets) if len(label_set) == 2]
    empty_idx = [idx for idx, label_set in enumerate(pred_sets) if len(label_set) == 0]
    coverage = float(np.mean([y_array[idx] in label_set for idx, label_set in enumerate(pred_sets)]))
    alert_ppv = float(y_array[certain_alert_idx].mean()) if certain_alert_idx else float("nan")
    clear_npv = float((1 - y_array[certain_clear_idx]).mean()) if certain_clear_idx else float("nan")
    missed_events = int(y_array[certain_clear_idx].sum()) if certain_clear_idx else 0
    false_alerts = int((1 - y_array[certain_alert_idx]).sum()) if certain_alert_idx else 0
    total_events = int(y_array.sum())
    return {
        "coverage": coverage,
        "alert_count": float(len(certain_alert_idx)),
        "clear_count": float(len(certain_clear_idx)),
        "defer_count": float(len(uncertain_idx)),
        "empty_count": float(len(empty_idx)),
        "certain_frac": float((len(certain_alert_idx) + len(certain_clear_idx)) / len(y_array)),
        "defer_rate": float(len(uncertain_idx) / len(y_array)),
        "alert_burden": float(len(certain_alert_idx) / len(y_array)),
        "alert_ppv": float(alert_ppv),
        "clear_npv": float(clear_npv),
        "miss_count": float(missed_events),
        "false_alert_count": float(false_alerts),
        "miss_rate": float(missed_events / max(len(certain_clear_idx), 1)),
        "recall_decided": float((int(y_array[certain_alert_idx].sum()) / total_events) if total_events else float("nan")),
        "total_events": float(total_events),
        "event_rate": float(y_array.mean()),
    }


def build_conformal_prediction_sets(
    calibration_probs: np.ndarray,
    calibration_labels: pd.Series | np.ndarray,
    test_probs: np.ndarray,
    *,
    alpha: float,
    method: str,
) -> tuple[list[set[int]], dict[str, float]]:
    if method == "split":
        return split_conformal_prediction_sets(calibration_probs, calibration_labels, test_probs, alpha)
    if method == "mondrian":
        return mondrian_conformal_prediction_sets(calibration_probs, calibration_labels, test_probs, alpha)
    raise ValueError(f"Unsupported conformal method: {method}")


def repeated_conformal_triage(
    frame: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    *,
    model_name: str,
    calibration_method: str,
    method: str = "mondrian",
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for seed in CONFORMAL_REPEATED_SEEDS:
        train_outer, test_outer, train_model, cal_model = split_frame(
            frame,
            label_col=TARGET_COL,
            random_state=seed,
            use_group_split=True,
        )
        bundle = fit_conformal_bundle(
            train_model,
            cal_model,
            test_outer,
            feature_cols=feature_sets["full"],
            model_name=model_name,
            calibration_method=calibration_method,
            label_col=TARGET_COL,
            random_state=seed,
        )
        for alpha in CONFORMAL_ALPHAS:
            pred_sets, thresholds = build_conformal_prediction_sets(
                bundle["calibration_probs"],
                cal_model[TARGET_COL],
                bundle["test_probs"],
                alpha=alpha,
                method=method,
            )
            metrics = evaluate_conformal_sets(pred_sets, test_outer[TARGET_COL])
            rows.append(
                {
                    "seed": seed,
                    "model_name": model_name,
                    "calibration_method": calibration_method,
                    "policy_name": f"{method}_conformal",
                    "alpha": alpha,
                    **thresholds,
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def repeated_conformal_consensus(
    frame: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    calibration_map: dict[str, str],
    *,
    method: str = "mondrian",
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    consensus_models = ["lightgbm", "xgboost", "logistic"]
    for seed in CONFORMAL_REPEATED_SEEDS:
        train_outer, test_outer, train_model, cal_model = split_frame(
            frame,
            label_col=TARGET_COL,
            random_state=seed,
            use_group_split=True,
        )
        bundles: dict[str, dict[str, Any]] = {}
        for model_name in consensus_models:
            bundles[model_name] = fit_conformal_bundle(
                train_model,
                cal_model,
                test_outer,
                feature_cols=feature_sets["full"],
                model_name=model_name,
                calibration_method=calibration_map.get(model_name, "sigmoid"),
                label_col=TARGET_COL,
                random_state=seed,
            )

        for alpha in CONFORMAL_ALPHAS:
            per_model_sets: dict[str, list[set[int]]] = {}
            for model_name in consensus_models:
                pred_sets, thresholds = build_conformal_prediction_sets(
                    bundles[model_name]["calibration_probs"],
                    cal_model[TARGET_COL],
                    bundles[model_name]["test_probs"],
                    alpha=alpha,
                    method=method,
                )
                per_model_sets[model_name] = pred_sets
                metrics = evaluate_conformal_sets(pred_sets, test_outer[TARGET_COL])
                rows.append(
                    {
                        "seed": seed,
                        "alpha": alpha,
                        "ensemble_name": "single_model",
                        "base_model": model_name,
                        **thresholds,
                        **metrics,
                    }
                )

            intersection_sets: list[set[int]] = []
            union_sets: list[set[int]] = []
            n_test = len(test_outer)
            for idx in range(n_test):
                intersection_sets.append(
                    per_model_sets["lightgbm"][idx]
                    & per_model_sets["xgboost"][idx]
                    & per_model_sets["logistic"][idx]
                )
                union_sets.append(
                    per_model_sets["lightgbm"][idx]
                    | per_model_sets["xgboost"][idx]
                    | per_model_sets["logistic"][idx]
                )
            for ensemble_name, pred_sets in {
                "intersection": intersection_sets,
                "union": union_sets,
            }.items():
                metrics = evaluate_conformal_sets(pred_sets, test_outer[TARGET_COL])
                rows.append(
                    {
                        "seed": seed,
                        "alpha": alpha,
                        "ensemble_name": ensemble_name,
                        "base_model": "lightgbm+xgboost+logistic",
                        **metrics,
                    }
                )
    return pd.DataFrame(rows)


def inject_missingness_shift(
    frame: pd.DataFrame, columns: list[str], fraction: float, random_state: int
) -> pd.DataFrame:
    shifted = frame.copy()
    rng = np.random.default_rng(random_state)
    for column in columns:
        if column not in shifted.columns:
            continue
        non_missing_idx = shifted.index[shifted[column].notna()].to_numpy()
        if len(non_missing_idx) == 0:
            continue
        sample_size = int(len(non_missing_idx) * fraction)
        if sample_size <= 0:
            continue
        sampled_idx = rng.choice(non_missing_idx, size=sample_size, replace=False)
        shifted.loc[sampled_idx, column] = np.nan
    return shifted


def drop_feature_group(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    shifted = frame.copy()
    for column in columns:
        if column in shifted.columns:
            if pd.api.types.is_numeric_dtype(shifted[column]):
                shifted[column] = shifted[column].astype(float)
            else:
                shifted[column] = shifted[column].astype(object)
            shifted.loc[:, column] = np.nan
    return shifted


def evaluate_shifted_selective_policy(
    selective_result: dict[str, Any], shifted_test_frame: pd.DataFrame, *, label_col: str = TARGET_COL
) -> dict[str, dict[str, float]]:
    full_bundle = selective_result["full_bundle"]
    phys_bundle = selective_result["phys_bundle"]
    full_probs = predict_probabilities(
        full_bundle["final_model"],
        full_bundle["feature_space"].transform_frame(shifted_test_frame),
    )
    phys_probs = predict_probabilities(
        phys_bundle["final_model"],
        phys_bundle["feature_space"].transform_frame(shifted_test_frame),
    )
    high_threshold = float(selective_result["primary_threshold"]["threshold"])
    low_threshold = float(selective_result["secondary_threshold"]["threshold"])
    if high_threshold <= low_threshold:
        low_threshold = min(low_threshold, high_threshold - 1e-6)
    fixed_metrics = evaluate_policy_masks(
        shifted_test_frame[label_col],
        full_probs >= high_threshold,
        full_probs <= low_threshold,
    )
    disagreement = np.abs(full_probs - phys_probs)
    selective_metrics = evaluate_policy_masks(
        shifted_test_frame[label_col],
        np.logical_and(full_probs >= high_threshold, disagreement <= selective_result["agreement_threshold"]),
        np.logical_and(full_probs <= low_threshold, disagreement <= selective_result["agreement_threshold"]),
    )
    return {"fixed_threshold": fixed_metrics, "selective_triage": selective_metrics}


def run_shift_stress_tests(
    frame: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    *,
    model_name: str,
    calibration_method: str,
) -> pd.DataFrame:
    selective_result = evaluate_selective_triage_on_split(
        frame,
        feature_sets,
        model_name=model_name,
        calibration_method=calibration_method,
        random_state=42,
    )
    test_frame = selective_result["test_outer"]
    physiology_numeric_cols = [
        column
        for column in feature_sets["physiology_severity"]
        if column in test_frame.columns and pd.api.types.is_numeric_dtype(test_frame[column])
    ]
    measurement_cols = [
        column for column in feature_sets["full"] if column in test_frame.columns and is_measurement_process_col(column)
    ]
    care_cols = [column for column in CARE_PROCESS_COLS if column in test_frame.columns]

    rows: list[dict[str, Any]] = []
    for fraction in SHIFT_LEVELS:
        shifted = inject_missingness_shift(
            test_frame,
            physiology_numeric_cols,
            fraction,
            random_state=int(1000 * fraction) + 42,
        )
        metrics = evaluate_shifted_selective_policy(selective_result, shifted)
        for policy_name, policy_metrics in metrics.items():
            rows.append(
                {
                    "scenario": "physiology_missingness_shift",
                    "severity": fraction,
                    "policy_name": policy_name,
                    **policy_metrics,
                }
            )

    for scenario_name, columns in {
        "measurement_process_dropout": measurement_cols,
        "care_process_dropout": care_cols,
    }.items():
        shifted = drop_feature_group(test_frame, columns)
        metrics = evaluate_shifted_selective_policy(selective_result, shifted)
        for policy_name, policy_metrics in metrics.items():
            rows.append(
                {
                    "scenario": scenario_name,
                    "severity": 1.0,
                    "policy_name": policy_name,
                    **policy_metrics,
                }
            )
    return pd.DataFrame(rows)


def conformal_shift_stress_tests(
    frame: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    *,
    model_name: str,
    calibration_method: str,
    method: str = "mondrian",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    _train_outer, test_outer, train_model, cal_model = split_frame(
        frame,
        label_col=TARGET_COL,
        random_state=42,
        use_group_split=True,
    )
    bundle = fit_conformal_bundle(
        train_model,
        cal_model,
        test_outer,
        feature_cols=feature_sets["full"],
        model_name=model_name,
        calibration_method=calibration_method,
        label_col=TARGET_COL,
        random_state=42,
    )
    physiology_numeric_cols = [
        column
        for column in feature_sets["physiology_severity"]
        if column in test_outer.columns and pd.api.types.is_numeric_dtype(test_outer[column])
    ]
    measurement_cols = [
        column for column in feature_sets["full"] if column in test_outer.columns and is_measurement_process_col(column)
    ]
    care_cols = [column for column in CARE_PROCESS_COLS if column in test_outer.columns]

    conformal_rows: list[dict[str, Any]] = []
    fixed_rows: list[dict[str, Any]] = []
    deployed_threshold = float(
        threshold_search(cal_model[TARGET_COL], bundle["calibration_probs"], precision_target=PRIMARY_PRECISION_TARGET).get(
            "threshold",
            0.50,
        )
    )
    fixed_thresholds = {"deployed_ppv50": deployed_threshold, "fixed_0.50": 0.50}
    shift_scenarios: list[tuple[str, float, pd.DataFrame]] = [("clean", 0.0, test_outer)]
    for fraction in CONFORMAL_SHIFT_LEVELS:
        shift_scenarios.append(
            (
                "physiology_missingness_shift",
                fraction,
                inject_missingness_shift(
                    test_outer,
                    physiology_numeric_cols,
                    fraction,
                    random_state=int(1000 * fraction) + 42,
                ),
            )
        )
    shift_scenarios.extend(
        [
            ("measurement_process_dropout", 1.0, drop_feature_group(test_outer, measurement_cols)),
            ("care_process_dropout", 1.0, drop_feature_group(test_outer, care_cols)),
        ]
    )

    for scenario, severity, shifted_frame in shift_scenarios:
        shifted_probs = predict_probabilities(
            bundle["model"],
            bundle["feature_space"].transform_frame(shifted_frame),
        )
        for alpha in [0.05, 0.10]:
            pred_sets, thresholds = build_conformal_prediction_sets(
                bundle["calibration_probs"],
                cal_model[TARGET_COL],
                shifted_probs,
                alpha=alpha,
                method=method,
            )
            metrics = evaluate_conformal_sets(pred_sets, shifted_frame[TARGET_COL])
            conformal_rows.append(
                {
                    "scenario": scenario,
                    "severity": severity,
                    "alpha": alpha,
                    "policy_name": f"{method}_conformal",
                    **thresholds,
                    **metrics,
                }
            )
        for threshold_name, threshold_value in fixed_thresholds.items():
            metrics = binary_metrics(shifted_frame[TARGET_COL], shifted_probs, threshold_value)
            fixed_rows.append(
                {
                    "scenario": scenario,
                    "severity": severity,
                    "threshold_name": threshold_name,
                    "threshold": threshold_value,
                    **metrics,
                }
            )
    return pd.DataFrame(conformal_rows), pd.DataFrame(fixed_rows)


def conformal_operating_curve(
    frame: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    *,
    model_name: str,
    calibration_method: str,
    method: str = "mondrian",
) -> pd.DataFrame:
    train_outer, test_outer, train_model, cal_model = split_frame(
        frame,
        label_col=TARGET_COL,
        random_state=42,
        use_group_split=True,
    )
    bundle = fit_conformal_bundle(
        train_model,
        cal_model,
        test_outer,
        feature_cols=feature_sets["full"],
        model_name=model_name,
        calibration_method=calibration_method,
        label_col=TARGET_COL,
        random_state=42,
    )
    rows: list[dict[str, Any]] = []
    for alpha in CONFORMAL_OPERATING_ALPHAS:
        pred_sets, thresholds = build_conformal_prediction_sets(
            bundle["calibration_probs"],
            cal_model[TARGET_COL],
            bundle["test_probs"],
            alpha=alpha,
            method=method,
        )
        metrics = evaluate_conformal_sets(pred_sets, test_outer[TARGET_COL])
        rows.append({"alpha": alpha, **thresholds, **metrics})
    return pd.DataFrame(rows)


def conformal_subgroup_metrics(
    frame: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    *,
    model_name: str,
    calibration_method: str,
    alpha: float = CONFORMAL_SUBGROUP_ALPHA,
    method: str = "mondrian",
) -> pd.DataFrame:
    train_outer, test_outer, train_model, cal_model = split_frame(
        frame,
        label_col=TARGET_COL,
        random_state=42,
        use_group_split=True,
    )
    bundle = fit_conformal_bundle(
        train_model,
        cal_model,
        test_outer,
        feature_cols=feature_sets["full"],
        model_name=model_name,
        calibration_method=calibration_method,
        label_col=TARGET_COL,
        random_state=42,
    )
    pred_sets, _ = build_conformal_prediction_sets(
        bundle["calibration_probs"],
        cal_model[TARGET_COL],
        bundle["test_probs"],
        alpha=alpha,
        method=method,
    )
    subgroup_definitions: dict[str, pd.Series] = {
        "gender": test_outer["gender"].astype(str),
        "age_band": pd.cut(
            test_outer["age"],
            bins=[0, 50, 65, 80, 120],
            labels=["<=50", "51-65", "66-80", ">80"],
            include_lowest=True,
        ).astype(str),
    }
    if "sofa_total_24hr" in test_outer.columns:
        subgroup_definitions["sofa_band"] = pd.qcut(
            test_outer["sofa_total_24hr"],
            q=3,
            labels=["Low SOFA", "Medium SOFA", "High SOFA"],
            duplicates="drop",
        ).astype(str)
    if "mechanical_ventilation_24hr_flag" in test_outer.columns:
        subgroup_definitions["mech_vent"] = (
            test_outer["mechanical_ventilation_24hr_flag"].fillna(-1).astype(int).astype(str)
        )
    if "vasopressor_24hr_flag" in test_outer.columns:
        subgroup_definitions["vasopressor"] = (
            test_outer["vasopressor_24hr_flag"].fillna(-1).astype(int).astype(str)
        )

    rows: list[dict[str, Any]] = []
    for subgroup_name, subgroup_values in subgroup_definitions.items():
        for value in sorted(pd.Series(subgroup_values).dropna().unique()):
            mask = subgroup_values == value
            if mask.sum() < 50:
                continue
            positions = np.flatnonzero(mask.to_numpy())
            subset_sets = [pred_sets[idx] for idx in positions]
            subset_metrics = evaluate_conformal_sets(subset_sets, test_outer.iloc[positions][TARGET_COL])
            rows.append(
                {
                    "alpha": alpha,
                    "subgroup": subgroup_name,
                    "value": value,
                    "n": int(mask.sum()),
                    "event_rate": float(test_outer.loc[mask, TARGET_COL].mean()),
                    **subset_metrics,
                }
            )
    return pd.DataFrame(rows).sort_values(["subgroup", "value"]).reset_index(drop=True)


def alert_policy_metrics(
    y_true: pd.Series | np.ndarray,
    alert_mask: np.ndarray,
    *,
    strategy_name: str,
) -> dict[str, float | str]:
    y_array = np.asarray(y_true).astype(int)
    alerts = np.asarray(alert_mask, dtype=bool)
    tp = int(np.logical_and(alerts, y_array == 1).sum())
    fp = int(np.logical_and(alerts, y_array == 0).sum())
    total_events = int(y_array.sum())
    alert_count = int(alerts.sum())
    return {
        "strategy": strategy_name,
        "alert_rate": float(alerts.mean()),
        "alert_precision": float(tp / alert_count) if alert_count else float("nan"),
        "alert_recall": float(tp / total_events) if total_events else float("nan"),
        "tp": float(tp),
        "fp": float(fp),
    }


def build_clinical_utility_artifacts(
    frame: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    *,
    selected_model: str,
    selected_calibration: str,
    calibration_map: dict[str, str],
    final_result: dict[str, Any],
    score_comparison_results: list[dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    y_true = final_result["test_outer"][TARGET_COL]
    continuous_frames = [
        decision_curve_from_probs(
            y_true,
            final_result["test_probs"],
            strategy_name=f"{selected_model}_continuous",
        )
    ]
    for score_result in score_comparison_results:
        continuous_frames.append(
            decision_curve_from_probs(
                score_result["test_outer"][TARGET_COL],
                score_result["test_probs"],
                strategy_name=f"{score_result['score_name']}_continuous",
            )
        )
    continuous_curve = pd.concat(continuous_frames, ignore_index=True)

    selective_result = evaluate_selective_triage_on_split(
        frame,
        feature_sets,
        model_name=selected_model,
        calibration_method=selected_calibration,
        random_state=42,
    )
    high_threshold = float(selective_result["primary_threshold"]["threshold"])
    selective_alert_mask = np.logical_and(
        selective_result["full_bundle"]["test_probs"] >= high_threshold,
        selective_result["test_disagreement"] <= selective_result["agreement_threshold"],
    )

    train_outer, test_outer, train_model, cal_model = split_frame(
        frame,
        label_col=TARGET_COL,
        random_state=42,
        use_group_split=True,
    )
    conformal_bundle = fit_conformal_bundle(
        train_model,
        cal_model,
        test_outer,
        feature_cols=feature_sets["full"],
        model_name=selected_model,
        calibration_method=selected_calibration,
        label_col=TARGET_COL,
        random_state=42,
    )
    policy_frames = [
        decision_curve_from_alert_mask(
            y_true,
            np.ones(len(y_true), dtype=bool),
            strategy_name="treat_all",
        ),
        decision_curve_from_alert_mask(
            y_true,
            np.zeros(len(y_true), dtype=bool),
            strategy_name="treat_none",
        ),
        decision_curve_from_alert_mask(
            y_true,
            final_result["test_probs"] >= float(final_result["primary_threshold"]["threshold"]),
            strategy_name="fixed_threshold_ppv50",
        ),
        decision_curve_from_alert_mask(
            y_true,
            selective_alert_mask,
            strategy_name="disagreement_selective",
        ),
    ]
    policy_metric_rows = [
        alert_policy_metrics(
            y_true,
            np.ones(len(y_true), dtype=bool),
            strategy_name="treat_all",
        ),
        alert_policy_metrics(
            y_true,
            np.zeros(len(y_true), dtype=bool),
            strategy_name="treat_none",
        ),
        alert_policy_metrics(
            y_true,
            final_result["test_probs"] >= float(final_result["primary_threshold"]["threshold"]),
            strategy_name="fixed_threshold_ppv50",
        ),
        alert_policy_metrics(
            y_true,
            selective_alert_mask,
            strategy_name="disagreement_selective",
        ),
    ]

    for alpha in [0.05, 0.10]:
        pred_sets, _ = build_conformal_prediction_sets(
            conformal_bundle["calibration_probs"],
            cal_model[TARGET_COL],
            conformal_bundle["test_probs"],
            alpha=alpha,
            method="mondrian",
        )
        alert_mask = np.asarray([label_set == {1} for label_set in pred_sets], dtype=bool)
        strategy_name = f"conformal_alpha_{alpha:.2f}"
        policy_frames.append(
            decision_curve_from_alert_mask(y_true, alert_mask, strategy_name=strategy_name)
        )
        policy_metric_rows.append(alert_policy_metrics(y_true, alert_mask, strategy_name=strategy_name))

    consensus_models = ["lightgbm", "xgboost", "logistic"]
    consensus_bundles: dict[str, dict[str, Any]] = {}
    for model_name in consensus_models:
        consensus_bundles[model_name] = fit_conformal_bundle(
            train_model,
            cal_model,
            test_outer,
            feature_cols=feature_sets["full"],
            model_name=model_name,
            calibration_method=calibration_map.get(model_name, "sigmoid"),
            label_col=TARGET_COL,
            random_state=42,
        )
    per_model_sets: dict[str, list[set[int]]] = {}
    for model_name in consensus_models:
        pred_sets, _ = build_conformal_prediction_sets(
            consensus_bundles[model_name]["calibration_probs"],
            cal_model[TARGET_COL],
            consensus_bundles[model_name]["test_probs"],
            alpha=0.05,
            method="mondrian",
        )
        per_model_sets[model_name] = pred_sets
    union_sets = [
        per_model_sets["lightgbm"][idx]
        | per_model_sets["xgboost"][idx]
        | per_model_sets["logistic"][idx]
        for idx in range(len(test_outer))
    ]
    union_alert_mask = np.asarray([label_set == {1} for label_set in union_sets], dtype=bool)
    policy_frames.append(
        decision_curve_from_alert_mask(y_true, union_alert_mask, strategy_name="union_conformal_alpha_0.05")
    )
    policy_metric_rows.append(
        alert_policy_metrics(y_true, union_alert_mask, strategy_name="union_conformal_alpha_0.05")
    )

    policy_curve = pd.concat(policy_frames, ignore_index=True)
    summary = pd.concat([continuous_curve, policy_curve], ignore_index=True)
    summary["threshold_round"] = summary["threshold"].round(2)
    summary = summary[summary["threshold_round"].isin(CLINICAL_UTILITY_THRESHOLDS)].drop(columns=["threshold_round"])

    policy_metrics = pd.DataFrame(policy_metric_rows)
    for threshold in CLINICAL_UTILITY_THRESHOLDS:
        curve_slice = policy_curve[np.isclose(policy_curve["threshold"], threshold)].set_index("strategy")
        policy_metrics[f"net_benefit_at_{threshold:.2f}"] = policy_metrics["strategy"].map(
            curve_slice["net_benefit"].to_dict()
        )
    return continuous_curve, policy_curve, summary, policy_metrics


def prediction_ceiling_analysis(
    frame: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    *,
    model_name: str,
    calibration_method: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    honest_result = evaluate_on_outer_split(
        frame,
        feature_sets["full"],
        model_name=model_name,
        calibration_method=calibration_method,
        random_state=42,
        use_group_split=True,
    )
    rows.append({"setting": "honest_grouped", **honest_result["metrics"]})

    cheat_features = feature_sets["full"] + [TIME_COL]
    cheat_result = evaluate_on_outer_split(
        frame,
        cheat_features,
        model_name=model_name,
        calibration_method=calibration_method,
        random_state=42,
        use_group_split=True,
    )
    rows.append({"setting": "with_time_to_event", **cheat_result["metrics"]})

    feature_space = fit_feature_space(frame, feature_sets["full"], frame[TARGET_COL])
    X_full = feature_space.transform_frame(frame)
    overfit_model = fit_calibrated_model(model_name, calibration_method, X_full, frame[TARGET_COL], 42)
    overfit_probs = predict_probabilities(overfit_model, X_full)
    rows.append({"setting": "overfit_train_equals_test", **binary_metrics(frame[TARGET_COL], overfit_probs, 0.50)})

    train_outer, test_outer, _, _ = split_frame(
        frame,
        label_col=TARGET_COL,
        random_state=42,
        use_group_split=True,
    )
    shuffled_y = pd.Series(
        np.random.default_rng(42).permutation(train_outer[TARGET_COL].to_numpy()),
        index=train_outer.index,
    )
    shuffled_space = fit_feature_space(train_outer, feature_sets["full"], shuffled_y)
    X_train_shuffled = shuffled_space.transform_frame(train_outer)
    X_test_shuffled = shuffled_space.transform_frame(test_outer)
    shuffled_model = fit_calibrated_model(
        model_name,
        calibration_method,
        X_train_shuffled,
        shuffled_y,
        42,
    )
    shuffled_probs = predict_probabilities(shuffled_model, X_test_shuffled)
    rows.append({"setting": "shuffled_label_noise_floor", **binary_metrics(test_outer[TARGET_COL], shuffled_probs, 0.50)})
    return pd.DataFrame(rows)


def select_best_model(repeated_results: pd.DataFrame) -> str:
    summary = (
        repeated_results.groupby("model_name")[
            ["recall_at_ppv_050", "brier", "auroc", "auprc"]
        ]
        .agg(["mean", "std"])
        .sort_values(
            by=[("recall_at_ppv_050", "mean"), ("brier", "mean"), ("auroc", "mean")],
            ascending=[False, True, False],
        )
    )
    write_markdown(
        ARTIFACT_DIR / "repeated_split_summary.md",
        render_table(summary, floatfmt=".3f"),
    )
    deployable_summary = summary.loc[
        [index for index in summary.index if index != "monotonic_hgbt"]
    ]
    return str(deployable_summary.index[0])


def feature_ablation_comparison(
    frame: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    *,
    model_name: str,
    calibration_method: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for feature_set_name in FEATURE_ABLATION_ORDER:
        feature_cols = feature_sets[feature_set_name]
        result = evaluate_on_outer_split(
            frame,
            feature_cols,
            model_name=model_name,
            calibration_method=calibration_method,
            random_state=42,
            use_group_split=True,
        )
        rows.append(
            {
                "feature_set": feature_set_name,
                "model_name": model_name,
                **result["metrics"],
                "recall_at_ppv_050": float(result["primary_threshold"].get("recall", float("nan"))),
                "precision_at_ppv_050_threshold": float(result["primary_threshold"].get("precision", float("nan"))),
                "threshold_ppv_050": float(result["primary_threshold"].get("threshold", float("nan"))),
                "low_risk_coverage_npv_095": float(result["secondary_threshold"].get("coverage", float("nan"))),
            }
        )
    return pd.DataFrame(rows).sort_values(["recall_at_ppv_050", "brier"], ascending=[False, True])


def bootstrap_intervals(
    y_true: pd.Series | np.ndarray,
    probs: np.ndarray,
    threshold: float,
    *,
    n_boot: int = BOOTSTRAP_SAMPLES,
    random_state: int = 42,
) -> dict[str, list[float]]:
    rng = np.random.default_rng(random_state)
    metrics: dict[str, list[float]] = {
        "auroc": [],
        "auprc": [],
        "brier": [],
        "precision": [],
        "recall": [],
        "alert_rate": [],
    }
    y_array = np.asarray(y_true)
    for _ in range(n_boot):
        sample_idx = rng.integers(0, len(y_array), len(y_array))
        y_boot = y_array[sample_idx]
        p_boot = probs[sample_idx]
        if len(np.unique(y_boot)) < 2:
            continue
        boot_metrics = binary_metrics(y_boot, p_boot, threshold)
        for key in metrics:
            metrics[key].append(float(boot_metrics[key]))

    intervals: dict[str, list[float]] = {}
    for key, values in metrics.items():
        if not values:
            intervals[key] = [float("nan"), float("nan")]
            continue
        intervals[key] = [
            float(np.quantile(values, 0.025)),
            float(np.quantile(values, 0.975)),
        ]
    return intervals


def cluster_bootstrap_intervals(
    y_true: pd.Series | np.ndarray,
    probs: np.ndarray,
    threshold: float,
    groups: pd.Series | np.ndarray,
    *,
    n_boot: int = BOOTSTRAP_SAMPLES,
    random_state: int = 42,
) -> dict[str, list[float]]:
    rng = np.random.default_rng(random_state)
    y_array = np.asarray(y_true)
    p_array = np.asarray(probs, dtype=float)
    group_array = np.asarray(groups)
    unique_groups = pd.Series(group_array).dropna().unique()
    if len(unique_groups) == 0:
        return bootstrap_intervals(y_array, p_array, threshold, n_boot=n_boot, random_state=random_state)

    group_indices = {
        group: np.flatnonzero(group_array == group)
        for group in unique_groups
    }
    metrics: dict[str, list[float]] = {
        "auroc": [],
        "auprc": [],
        "brier": [],
        "precision": [],
        "recall": [],
        "alert_rate": [],
        "calibration_intercept": [],
        "calibration_slope": [],
    }
    for _ in range(n_boot):
        sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        sample_idx = np.concatenate([group_indices[group] for group in sampled_groups])
        y_boot = y_array[sample_idx]
        p_boot = p_array[sample_idx]
        if len(np.unique(y_boot)) < 2:
            continue
        boot_metrics = binary_metrics(y_boot, p_boot, threshold)
        for key in metrics:
            metrics[key].append(float(boot_metrics[key]))

    intervals: dict[str, list[float]] = {}
    for key, values in metrics.items():
        if not values:
            intervals[key] = [float("nan"), float("nan")]
            continue
        intervals[key] = [
            float(np.quantile(values, 0.025)),
            float(np.quantile(values, 0.975)),
        ]
    return intervals


def subgroup_metrics(
    frame: pd.DataFrame,
    y_true: pd.Series,
    probs: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    subgroup_definitions: dict[str, pd.Series] = {
        "gender": frame["gender"].astype(str),
        "aki_trigger": frame["aki_trigger"].astype(str),
        "mech_vent": frame["mechanical_ventilation_24hr_flag"].fillna(-1).astype(int).astype(str),
        "vasopressor": frame["vasopressor_24hr_flag"].fillna(-1).astype(int).astype(str),
        "rrt": frame["rrt_24hr_flag"].fillna(-1).astype(int).astype(str),
        "age_band": pd.cut(
            frame["age"],
            bins=[0, 50, 65, 80, 120],
            labels=["<=50", "51-65", "66-80", ">80"],
            include_lowest=True,
        ).astype(str),
    }

    missing_burden = frame.isna().sum(axis=1)
    subgroup_definitions["missingness_quartile"] = pd.qcut(
        missing_burden, q=4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop"
    ).astype(str)

    rows: list[dict[str, Any]] = []
    for subgroup_name, subgroup_values in subgroup_definitions.items():
        for value in sorted(pd.Series(subgroup_values).dropna().unique()):
            mask = subgroup_values == value
            if mask.sum() < 50 or len(np.unique(y_true[mask])) < 2:
                continue
            metrics = binary_metrics(y_true[mask], probs[mask], threshold)
            rows.append(
                {
                    "subgroup": subgroup_name,
                    "value": value,
                    "n": int(mask.sum()),
                    "event_rate": float(y_true[mask].mean()),
                    "mean_predicted_risk": float(probs[mask].mean()),
                    **metrics,
                }
            )
    return pd.DataFrame(rows).sort_values(["subgroup", "value"]).reset_index(drop=True)


def aggregation_map_for_transformed_features(
    transformed_names: list[str], raw_numeric_cols: list[str], raw_categorical_cols: list[str]
) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for name in transformed_names:
        if name.startswith("num__"):
            mapping[name] = name.split("num__", 1)[1]
            continue
        if name.startswith("cat__"):
            encoded = name.split("cat__", 1)[1]
            raw_name = encoded
            for cat_col in raw_categorical_cols:
                prefix = f"{cat_col}_"
                if encoded.startswith(prefix):
                    raw_name = cat_col
                    break
            mapping[name] = raw_name
            continue
        mapping[name] = name
    return mapping


def top_feature_importance_table(
    model: Any,
    X_test: pd.DataFrame | np.ndarray,
    y_test: pd.Series,
    transformed_names: list[str],
    feature_space: FeatureSpace | None,
) -> pd.DataFrame:
    native_importances: np.ndarray | None = None
    if hasattr(model, "feature_importances_"):
        native_importances = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        native_importances = np.abs(np.asarray(model.coef_)).ravel().astype(float)
    elif hasattr(model, "estimator") and hasattr(model.estimator, "feature_importances_"):
        native_importances = np.asarray(model.estimator.feature_importances_, dtype=float)
    elif hasattr(model, "estimator") and hasattr(model.estimator, "coef_"):
        native_importances = np.abs(np.asarray(model.estimator.coef_)).ravel().astype(float)

    if feature_space is None:
        if native_importances is None or len(native_importances) != len(transformed_names):
            transformed_df = (
                X_test.copy()
                if isinstance(X_test, pd.DataFrame)
                else pd.DataFrame(X_test, columns=transformed_names)
            )
            importances = permutation_importance(
                model,
                transformed_df,
                y_test,
                scoring="roc_auc",
                n_repeats=PERMUTATION_IMPORTANCE_REPEATS,
                random_state=42,
                n_jobs=CPU_COUNT,
            )
            native_importances = importances.importances_mean
        table = pd.DataFrame({"feature": transformed_names, "importance": native_importances})
        return table.sort_values("importance", ascending=False).reset_index(drop=True)

    if native_importances is None or len(native_importances) != len(transformed_names):
        transformed_df = (
            X_test.copy()
            if isinstance(X_test, pd.DataFrame)
            else pd.DataFrame(X_test, columns=transformed_names)
        )
        importances = permutation_importance(
            model,
            transformed_df,
            y_test,
            scoring="roc_auc",
            n_repeats=PERMUTATION_IMPORTANCE_REPEATS,
            random_state=42,
            n_jobs=CPU_COUNT,
        )
        native_importances = importances.importances_mean
    raw_mapping = aggregation_map_for_transformed_features(
        transformed_names, feature_space.numeric_cols, feature_space.categorical_cols
    )
    table = pd.DataFrame(
        {
            "transformed_feature": transformed_names,
            "raw_feature": [raw_mapping[name] for name in transformed_names],
            "importance": native_importances,
        }
    )
    return (
        table.groupby("raw_feature", as_index=False)["importance"]
        .sum()
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def tier_feature_summary(
    raw_test: pd.DataFrame, tiers: np.ndarray, top_features: list[str]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for feature in top_features:
        if feature not in raw_test.columns:
            continue
        for tier in ["High risk", "Uncertain", "Low risk"]:
            mask = tiers == tier
            series = raw_test.loc[mask, feature]
            rows.append(
                {
                    "feature": feature,
                    "tier": tier,
                    "mean": float(series.mean()) if pd.api.types.is_numeric_dtype(series) else np.nan,
                    "median": float(series.median()) if pd.api.types.is_numeric_dtype(series) else np.nan,
                    "mode": series.mode(dropna=True).iloc[0] if not series.mode(dropna=True).empty else None,
                }
            )
    return pd.DataFrame(rows)


def plot_precision_frontier(frontier: list[dict[str, Any]], path: Path) -> None:
    targets = []
    recalls = []
    alert_rates = []
    for item in frontier:
        if not item["test"]:
            continue
        targets.append(item["target"])
        recalls.append(item["test"]["recall"])
        alert_rates.append(item["test"]["alert_rate"])

    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.plot(targets, recalls, marker="o", label="Test recall")
    ax1.set_xlabel("Validation PPV target")
    ax1.set_ylabel("Test recall")
    ax1.set_ylim(0, 1)
    ax2 = ax1.twinx()
    ax2.plot(targets, alert_rates, marker="s", color="tab:orange", label="Alert rate")
    ax2.set_ylabel("Alert rate")
    ax2.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_calibration_curve(y_true: pd.Series, probs: np.ndarray, path: Path) -> None:
    frac_pos, mean_pred = calibration_curve(y_true, probs, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Perfect calibration")
    ax.plot(mean_pred, frac_pos, marker="o", label="Model")
    ax.set_xlabel("Mean predicted risk")
    ax.set_ylabel("Observed event rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_decision_curve(curve: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    if {"strategy", "net_benefit"}.issubset(curve.columns):
        for strategy_name, group in curve.groupby("strategy"):
            ax.plot(group["threshold"], group["net_benefit"], label=strategy_name)
    else:
        ax.plot(curve["threshold"], curve["model"], label="Model")
        ax.plot(curve["threshold"], curve["treat_all"], label="Treat all")
        ax.plot(curve["threshold"], curve["treat_none"], label="Treat none")
    ax.set_xlabel("Threshold probability")
    ax.set_ylabel("Net benefit")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_clinical_utility_panel(
    y_true: pd.Series | np.ndarray,
    probs: np.ndarray,
    decision_curve_table: pd.DataFrame,
    path: Path,
) -> None:
    frac_pos, mean_pred = calibration_curve(y_true, probs, n_bins=10, strategy="quantile")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    calib_ax, util_ax = axes
    calib_ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Perfect calibration")
    calib_ax.plot(mean_pred, frac_pos, marker="o", label="Selected model")
    calib_ax.set_xlabel("Mean predicted risk")
    calib_ax.set_ylabel("Observed event rate")
    calib_ax.set_xlim(0, 1)
    calib_ax.set_ylim(0, 1)
    calib_ax.set_title("Calibration")
    calib_ax.legend(fontsize=8)

    for strategy_name, group in decision_curve_table.groupby("strategy"):
        util_ax.plot(group["threshold"], group["net_benefit"], label=strategy_name)
    util_ax.set_xlabel("Threshold probability")
    util_ax.set_ylabel("Net benefit")
    util_ax.set_title("Clinical utility")
    util_ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_roc_pr_benchmarks(curves: list[dict[str, Any]], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    roc_ax, pr_ax = axes
    for curve in curves:
        y_true = np.asarray(curve["y_true"]).astype(int)
        probs = np.asarray(curve["probs"], dtype=float)
        if len(np.unique(y_true)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true, probs)
        precision, recall, _ = precision_recall_curve(y_true, probs)
        roc_ax.plot(fpr, tpr, label=f"{curve['label']} (AUROC={safe_auroc(y_true, probs):.3f})")
        pr_ax.plot(recall, precision, label=f"{curve['label']} (AUPRC={safe_auprc(y_true, probs):.3f})")
    roc_ax.plot([0, 1], [0, 1], linestyle="--", color="grey")
    roc_ax.set_xlabel("False positive rate")
    roc_ax.set_ylabel("True positive rate")
    roc_ax.set_title("ROC")
    pr_ax.set_xlabel("Recall")
    pr_ax.set_ylabel("Precision")
    pr_ax.set_title("Precision-recall")
    roc_ax.legend(fontsize=8)
    pr_ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_prediction_ceiling(ceiling_table: pd.DataFrame, path: Path) -> None:
    plot_frame = ceiling_table.loc[:, ["setting", "auroc", "auprc"]].copy()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    axes[0].barh(plot_frame["setting"], plot_frame["auroc"])
    axes[0].set_xlabel("AUROC")
    axes[0].set_xlim(0, 1)
    axes[0].set_title("Prediction ceiling")
    axes[1].barh(plot_frame["setting"], plot_frame["auprc"])
    axes[1].set_xlabel("AUPRC")
    axes[1].set_xlim(0, 1)
    axes[1].set_title("AUPRC")
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_conformal_operating_curve(operating_curve: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    left_ax, right_ax = axes
    left_ax.plot(operating_curve["alpha"], operating_curve["certain_frac"], label="Certain fraction")
    left_ax.plot(operating_curve["alpha"], operating_curve["defer_rate"], label="Defer rate")
    left_ax.set_xlabel("Alpha")
    left_ax.set_ylabel("Fraction of cohort")
    left_ax.set_ylim(0, 1)
    left_ax.set_title("Coverage vs defer")
    left_ax.legend()

    right_ax.plot(operating_curve["alpha"], operating_curve["alert_ppv"], label="Alert PPV")
    right_ax.plot(operating_curve["alpha"], operating_curve["clear_npv"], label="Clear NPV")
    right_ax.plot(operating_curve["alpha"], operating_curve["coverage"], label="Coverage")
    right_ax.set_xlabel("Alpha")
    right_ax.set_ylabel("Metric value")
    right_ax.set_ylim(0, 1)
    right_ax.set_title("Conformal operating characteristics")
    right_ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_conformal_consensus_tradeoff(consensus_table: pd.DataFrame, path: Path) -> None:
    summary = (
        consensus_table.groupby(["ensemble_name", "base_model", "alpha"])[
            ["certain_frac", "alert_ppv", "clear_npv", "coverage"]
        ]
        .mean()
        .reset_index()
    )
    union = summary[summary["ensemble_name"] == "union"]
    single_lgbm = summary[
        (summary["ensemble_name"] == "single_model") & (summary["base_model"] == "lightgbm")
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    axes[0].plot(single_lgbm["alpha"], single_lgbm["certain_frac"], marker="o", label="Single model certain")
    axes[0].plot(union["alpha"], union["certain_frac"], marker="o", label="Union certain")
    axes[0].plot(single_lgbm["alpha"], single_lgbm["coverage"], marker="s", label="Single model coverage")
    axes[0].plot(union["alpha"], union["coverage"], marker="s", label="Union coverage")
    axes[0].set_xlabel("Alpha")
    axes[0].set_ylabel("Fraction")
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Coverage / automation trade-off")
    axes[0].legend(fontsize=8)

    axes[1].plot(single_lgbm["alpha"], single_lgbm["alert_ppv"], marker="o", label="Single model alert PPV")
    axes[1].plot(union["alpha"], union["alert_ppv"], marker="o", label="Union alert PPV")
    axes[1].plot(single_lgbm["alpha"], single_lgbm["clear_npv"], marker="s", label="Single model clear NPV")
    axes[1].plot(union["alpha"], union["clear_npv"], marker="s", label="Union clear NPV")
    axes[1].set_xlabel("Alpha")
    axes[1].set_ylabel("Metric value")
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Consensus reliability trade-off")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_conformal_shift_panel(
    conformal_shift_table: pd.DataFrame,
    fixed_shift_table: pd.DataFrame,
    path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    left_ax, right_ax = axes

    missing = conformal_shift_table[
        conformal_shift_table["scenario"] == "physiology_missingness_shift"
    ].sort_values(["alpha", "severity"])
    for alpha, group in missing.groupby("alpha"):
        left_ax.plot(group["severity"], group["coverage"], marker="o", label=f"Coverage α={alpha:.2f}")
        left_ax.plot(group["severity"], group["certain_frac"], marker="s", linestyle="--", label=f"Certain α={alpha:.2f}")
    left_ax.set_xlabel("Injected missingness fraction")
    left_ax.set_ylabel("Fraction")
    left_ax.set_ylim(0, 1)
    left_ax.set_title("Conformal under missingness shift")
    left_ax.legend(fontsize=8)

    fixed_missing = fixed_shift_table[
        fixed_shift_table["scenario"] == "physiology_missingness_shift"
    ].sort_values(["threshold_name", "severity"])
    for threshold_name, group in fixed_missing.groupby("threshold_name"):
        right_ax.plot(group["severity"], group["recall"], marker="o", label=f"Recall {threshold_name}")
        right_ax.plot(group["severity"], group["precision"], marker="s", linestyle="--", label=f"Precision {threshold_name}")
    right_ax.set_xlabel("Injected missingness fraction")
    right_ax.set_ylabel("Metric value")
    right_ax.set_ylim(0, 1)
    right_ax.set_title("Fixed thresholds under shift")
    right_ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_conformal_subgroup_forest(subgroup_table: pd.DataFrame, path: Path) -> None:
    plot_frame = subgroup_table.copy()
    plot_frame["label"] = plot_frame["subgroup"] + ": " + plot_frame["value"]
    fig, axes = plt.subplots(1, 3, figsize=(13, max(5, 0.32 * len(plot_frame))))
    metrics = [("coverage", "Coverage"), ("alert_ppv", "Alert PPV"), ("clear_npv", "Clear NPV")]
    for axis, (column, title) in zip(axes, metrics):
        valid = plot_frame.dropna(subset=[column]).sort_values(column)
        axis.scatter(valid[column], valid["label"])
        axis.set_xlim(0, 1)
        axis.set_xlabel(title)
        axis.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def search_external_validation_assets() -> list[str]:
    patterns = ["*eicu*", "*amsterdam*", "*hirid*"]
    matches: list[str] = []
    for pattern in patterns:
        for match in REPO_ROOT.rglob(pattern):
            matches.append(str(match))
    return sorted(set(matches))


def evaluate_secondary_horizons(
    frame: pd.DataFrame,
    feature_cols: list[str],
    *,
    model_name: str,
    calibration_method: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for label_col in ["death_within_48h", "death_within_7d", "death_before_discharge"]:
        result = evaluate_on_outer_split(
            frame,
            feature_cols,
            model_name=model_name,
            calibration_method=calibration_method,
            random_state=42,
            use_group_split=True,
            label_col=label_col,
        )
        rows.append(
            {
                "label": label_col,
                "prevalence": float(frame[label_col].mean()),
                **result["metrics"],
                "recall_at_primary_ppv": float(result["primary_threshold"].get("recall", float("nan"))),
            }
        )
    return pd.DataFrame(rows)


def write_audit_report(audit: DatasetAudit, feature_sets: dict[str, list[str]]) -> None:
    lines = [
        "# SA-AKI audit report",
        "",
        f"- Preferred dataset: `{audit.preferred_dataset}`",
        f"- Current cohort size: `{audit.current_rows}` rows x `{audit.current_columns}` columns",
        f"- Legacy cohort size: `{audit.legacy_rows}` rows x `{audit.legacy_columns}` columns",
        f"- Event prevalence: `{audit.event_rate:.3f}`",
        f"- Unique stays: `{audit.unique_stays}`",
        f"- Unique subjects: `{audit.unique_subjects}`",
        f"- Repeated subject rows: `{audit.repeated_subject_rows}` across `{audit.repeated_subjects}` subjects",
        f"- Duplicate hospital admission rows: `{audit.duplicate_hadm_rows}`",
        f"- RRT prevalence in current v2 cohort: `{audit.rrt_prevalence:.3f}`",
        f"- Features after >99% missing drop: `{audit.features_after_gt99_drop}`",
        f"- Features after removing cohort times: `{audit.features_after_gt99_and_cohort_time_drop}`",
        f"- Features after removing measurement-process counts/QC fractions: `{audit.features_after_obs_count_drop}`",
        "",
        "## Feature sets",
        "",
    ]
    for name, cols in feature_sets.items():
        lines.append(f"- `{name}`: `{len(cols)}` columns")
    lines.extend(["", "## Known mismatches", ""])
    lines.extend([f"- {item}" for item in audit.doc_mismatches])
    write_markdown(ARTIFACT_DIR / "audit_report.md", "\n".join(lines))
    write_json(ARTIFACT_DIR / "audit_report.json", asdict(audit))


def write_data_contract(contract: DataContract) -> None:
    lines = [
        "# SA-AKI deployment data contract",
        "",
        f"- Working dataset: `{contract.dataset_name}`",
        f"- Row definition: {contract.row_definition}",
        f"- Prediction time: {contract.prediction_time}",
        f"- Target definition: {contract.target_definition}",
        f"- Time anchor: {contract.time_anchor_definition}",
        "",
        "## Identifier columns",
        "",
    ]
    lines.extend([f"- `{column}`" for column in contract.identifier_columns])
    lines.extend(["", "## Process feature families", ""])
    lines.extend([f"- {item}" for item in contract.process_feature_families])
    lines.extend(["", "## Key schema notes", ""])
    lines.extend([f"- {item}" for item in contract.key_schema_notes])
    lines.extend(["", "## Claims explicitly excluded", ""])
    lines.extend([f"- {item}" for item in contract.excluded_claims])
    write_markdown(ARTIFACT_DIR / "data_contract.md", "\n".join(lines))
    write_json(ARTIFACT_DIR / "data_contract.json", asdict(contract))


def summarize_frontier(frontier_table: pd.DataFrame) -> pd.DataFrame:
    if frontier_table.empty:
        return pd.DataFrame()
    value_cols = [
        column
        for column in ["test_precision", "test_recall", "test_alert_rate", "test_coverage", "test_npv"]
        if column in frontier_table.columns
    ]
    summary = (
        frontier_table.groupby(["frontier_type", "target"])[value_cols]
        .agg(["mean", "std"])
        .reset_index()
    )
    return flatten_columns(summary)


def write_threshold_objective_report(frontier_table: pd.DataFrame) -> None:
    summary = summarize_frontier(frontier_table)
    exploratory_row = frontier_table[
        (frontier_table["frontier_type"] == "precision")
        & (frontier_table["target"].isin(EXPLORATORY_PRECISION_TARGETS))
    ]
    extreme_precision_viable = False
    if not exploratory_row.empty and "test_alert_rate" in exploratory_row.columns:
        extreme_precision_viable = bool((exploratory_row["test_alert_rate"].fillna(0) >= 0.01).any())

    lines = [
        "# Threshold objective memo",
        "",
        f"- Primary deployment objective: maximize recall under `PPV >= {PRIMARY_PRECISION_TARGET:.2f}`.",
        f"- Secondary objective: maximize low-risk coverage under `NPV >= {SECONDARY_NPV_TARGET:.2f}`.",
        "- Report alert burden explicitly through fixed alert-rate frontiers.",
        (
            "- `PPV >= 0.99` remained non-trivial on at least one repeated split, so it can stay as an exploratory extreme-alert tier."
            if extreme_precision_viable
            else "- `PPV >= 0.99` collapsed to trivial alert volume, so it is documented as exploratory only and excluded from the main deployment objective."
        ),
        "",
        "## Frontier summary",
        "",
        render_table(summary, index=False, floatfmt=".3f") if not summary.empty else "No frontier summary available.",
    ]
    write_markdown(ARTIFACT_DIR / "threshold_objective.md", "\n".join(lines))


def write_selective_triage_report(
    selective_table: pd.DataFrame, shift_table: pd.DataFrame
) -> None:
    summary = flatten_columns(
        selective_table.groupby("policy_name")[
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
    shift_summary = (
        shift_table.groupby(["scenario", "policy_name"])[
            ["alert_precision", "alert_recall", "actionable_coverage", "low_risk_npv", "defer_rate"]
        ]
        .mean()
        .reset_index()
    )
    lines = [
        "# Observation-process selective triage",
        "",
        "- `fixed_threshold` uses the calibrated full model with high- and low-risk cutoffs only.",
        "- `selective_triage` defers cases where physiology-only and full-model risks disagree beyond the validation-selected agreement threshold.",
        "",
        "## Repeated grouped summary",
        "",
        render_table(summary, index=False, floatfmt=".3f"),
        "",
        "## Shift stress summary",
        "",
        render_table(shift_summary, index=False, floatfmt=".3f"),
    ]
    write_markdown(ARTIFACT_DIR / "selective_triage_report.md", "\n".join(lines))


def write_journal_positioning() -> None:
    lines = [
        f"# {JOURNAL_TARGET} manuscript positioning",
        "",
        f"- Primary target journal: `{JOURNAL_TARGET}`",
        "- Primary story: conformal selective triage with formal coverage guarantees and graceful degradation under shift.",
        "- Comparator/ablation: disagreement-based selective triage using physiology-only versus process-enriched models.",
        "- Foundational result: the prediction ceiling shows first-24h SA-AKI mortality discrimination is bounded, so the paper shifts from raw accuracy to safe automation.",
        "",
        "## Claims we will make",
        "",
        "- This is a clinically grounded informatics paper about uncertainty-aware deployment, not a leaderboard paper.",
        "- The main contribution is an Alert/Defer/Clear framework with finite-sample conformal guarantees under subject-grouped evaluation.",
        "- The disagreement-based policy remains in scope as a baseline/ablation to show why the conformal formulation is stronger.",
        "",
        "## Claims we will not make",
        "",
        "- No bedside-readiness claim.",
        "- No external transportability claim without temporal or external validation.",
        "- No universal claim that conformal or disagreement triage dominates every operating point.",
        "",
        "## Submission ladder",
        "",
        "- Primary: `JAMIA`",
        "- Stretch after stronger external or temporal validation: `npj Digital Medicine`, `Communications Medicine`",
        "- Fallbacks: `Artificial Intelligence in Medicine`, `BMJ Health & Care Informatics`",
    ]
    write_markdown(ARTIFACT_DIR / "journal_positioning.md", "\n".join(lines))


def write_benchmark_report(
    benchmark_summary: pd.DataFrame,
    ceiling_table: pd.DataFrame,
    clinical_score_table: pd.DataFrame,
    clinical_score_operating_table: pd.DataFrame | None = None,
) -> None:
    lines = [
        "# Benchmark and ceiling report",
        "",
        "## Prediction ceiling",
        "",
        render_table(ceiling_table, index=False, floatfmt=".3f"),
        "",
        "## Repeated grouped benchmark summary",
        "",
        render_table(benchmark_summary, index=False, floatfmt=".3f"),
    ]
    if not clinical_score_table.empty:
        lines.extend(
            [
                "",
                "## Clinical score baselines",
                "",
                render_table(clinical_score_table, index=False, floatfmt=".3f"),
            ]
        )
    if clinical_score_operating_table is not None and not clinical_score_operating_table.empty:
        lines.extend(
            [
                "",
                "## Clinical score operating points",
                "",
                render_table(clinical_score_operating_table, index=False, floatfmt=".3f"),
            ]
        )
    write_markdown(ARTIFACT_DIR / "benchmark_report.md", "\n".join(lines))


def write_clinical_utility_report(
    continuous_curve_table: pd.DataFrame,
    policy_curve_table: pd.DataFrame,
    summary_table: pd.DataFrame,
    policy_metrics_table: pd.DataFrame,
) -> None:
    lines = [
        "# Clinical utility report",
        "",
        "## Decision-curve summary at key thresholds",
        "",
        render_table(summary_table.sort_values(["threshold", "strategy"]), index=False, floatfmt=".4f"),
        "",
        "## Fixed-policy operating summaries",
        "",
        render_table(policy_metrics_table, index=False, floatfmt=".4f"),
        "",
        "## Continuous benchmark curves",
        "",
        render_table(continuous_curve_table, index=False, floatfmt=".4f"),
        "",
        "## Fixed-policy curves",
        "",
        render_table(policy_curve_table, index=False, floatfmt=".4f"),
    ]
    write_markdown(ARTIFACT_DIR / "clinical_utility_report.md", "\n".join(lines))


def write_conformal_report(
    single_model_table: pd.DataFrame,
    consensus_table: pd.DataFrame,
    shift_table: pd.DataFrame,
    fixed_shift_table: pd.DataFrame,
    operating_curve: pd.DataFrame,
    subgroup_table: pd.DataFrame,
) -> None:
    single_summary = summarize_metric_intervals(
        single_model_table,
        group_cols=["alpha"],
        value_cols=["coverage", "certain_frac", "alert_ppv", "clear_npv", "miss_count"],
    )
    consensus_summary = summarize_metric_intervals(
        consensus_table,
        group_cols=["ensemble_name", "base_model", "alpha"],
        value_cols=["coverage", "certain_frac", "alert_ppv", "clear_npv", "miss_count"],
    )
    sweet_spot = operating_curve[
        operating_curve["clear_npv"].fillna(0) >= 0.90
    ]
    sweet_spot = sweet_spot[sweet_spot["alert_ppv"].fillna(0) >= 0.55]
    best_row = (
        sweet_spot.sort_values("certain_frac", ascending=False).iloc[0]
        if not sweet_spot.empty
        else None
    )
    lines = [
        "# Conformal triage report",
        "",
        "- Main contribution: single-model Mondrian conformal triage with explicit coverage and defer-region reporting.",
        "- Consensus extension: union and intersection of conformal prediction sets across LightGBM, XGBoost, and logistic regression.",
        f"- Repeated grouped seeds for conformal headline estimates: `{single_model_table['seed'].nunique() if 'seed' in single_model_table.columns else 0}`.",
        "",
        "## Single-model conformal summary",
        "",
        render_table(single_summary, index=False, floatfmt=".3f"),
        "",
        "## Consensus summary",
        "",
        render_table(consensus_summary, index=False, floatfmt=".3f"),
        "",
        "## Shift robustness (conformal)",
        "",
        render_table(shift_table, index=False, floatfmt=".3f"),
        "",
        "## Shift robustness (fixed thresholds)",
        "",
        render_table(fixed_shift_table, index=False, floatfmt=".3f"),
        "",
        "## Operating curve best point",
        "",
    ]
    if best_row is None:
        lines.append("- No operating point met the manuscript sweet-spot constraints.")
    else:
        lines.extend(
            [
                f"- Best alpha under `NPV >= 0.90` and `PPV >= 0.55`: `{best_row['alpha']:.2f}`",
                f"- Certain fraction: `{best_row['certain_frac']:.3f}`",
                f"- Alert PPV: `{best_row['alert_ppv']:.3f}`",
                f"- Clear NPV: `{best_row['clear_npv']:.3f}`",
                f"- Misses / events: `{int(best_row['miss_count'])}` / `{int(best_row['total_events'])}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Subgroup summary",
            "",
            render_table(subgroup_table, index=False, floatfmt=".3f"),
        ]
    )
    write_markdown(ARTIFACT_DIR / "conformal_report.md", "\n".join(lines))


def write_stretch_assessment(core_signal_is_strong: bool) -> None:
    lines = [
        "# Stretch work assessment",
        "",
        (
            "- The core selective-triage signal was strong enough to justify future appendix work on conformal selection, TabPFN, or coherence scoring."
            if core_signal_is_strong
            else "- The core selective-triage signal is not yet strong enough to justify appendix-level stretch work, so conformal prediction, TabPFN, and SHAP coherence remain deferred."
        ),
        "- This artifact intentionally documents the decision instead of silently expanding scope.",
    ]
    write_markdown(ARTIFACT_DIR / "stretch_assessment.md", "\n".join(lines))


def write_deployment_report(
    *,
    audit: DatasetAudit,
    calibration_benchmark: pd.DataFrame,
    repeated_results: pd.DataFrame,
    feature_ablation: pd.DataFrame,
    selected_model: str,
    selected_calibration: str,
    final_result: dict[str, Any],
    subgroup_table: pd.DataFrame,
    horizon_table: pd.DataFrame,
    top_features: pd.DataFrame,
    external_assets: list[str],
) -> None:
    primary_metrics = final_result["metrics"]
    policy_table = pd.DataFrame(final_result["policy_rows"])
    lines = [
        "# SA-AKI deployment-first summary",
        "",
        "## Locked clinical objective",
        "",
        f"- Primary use case: T24 high-risk escalation alert maximizing recall under `PPV >= {PRIMARY_PRECISION_TARGET:.2f}`",
        f"- Secondary use case: low-risk rule-out maximizing coverage under `NPV >= {SECONDARY_NPV_TARGET:.2f}`",
        f"- Dataset used for deployment workflow: `{audit.preferred_dataset}`",
        "",
        "## Selected deployment model",
        "",
        f"- Model: `{selected_model}`",
        f"- Calibration: `{selected_calibration}`",
        f"- Group-aware split: `yes` (subject-level holdout)",
        f"- Test AUROC: `{primary_metrics['auroc']:.3f}`",
        f"- Test AUPRC: `{primary_metrics['auprc']:.3f}`",
        f"- Test Brier: `{primary_metrics['brier']:.3f}`",
        f"- Calibration intercept: `{primary_metrics['calibration_intercept']:.3f}`",
        f"- Calibration slope: `{primary_metrics['calibration_slope']:.3f}`",
        f"- Primary-policy precision: `{final_result['primary_threshold'].get('precision', float('nan')):.3f}`",
        f"- Primary-policy recall: `{final_result['primary_threshold'].get('recall', float('nan')):.3f}`",
        f"- Primary-policy alert rate: `{final_result['primary_threshold'].get('alert_rate', float('nan')):.3f}`",
        f"- Low-risk coverage at NPV target: `{final_result['secondary_threshold'].get('coverage', float('nan')):.3f}`",
        "",
        "## Three-tier policy",
        "",
        render_table(policy_table, index=False, floatfmt=".3f"),
        "",
        "## Calibration benchmark",
        "",
        render_table(calibration_benchmark, index=False, floatfmt=".3f"),
        "",
        "## Repeated group-split robustness",
        "",
        (
            render_table(
                repeated_results.groupby("model_name")[["auroc", "auprc", "brier", "recall_at_ppv_050"]]
                .agg(["mean", "std"]),
                floatfmt=".3f",
            )
        ),
        "",
        "## Feature-set ablation",
        "",
        render_table(feature_ablation, index=False, floatfmt=".3f"),
        "",
        "## Secondary horizons",
        "",
        render_table(horizon_table, index=False, floatfmt=".3f"),
        "",
        "## Top global features",
        "",
        render_table(top_features.head(12), index=False, floatfmt=".4f"),
        "",
        "## External validation availability",
        "",
    ]
    if external_assets:
        lines.extend([f"- Found candidate external assets: `{item}`" for item in external_assets])
    else:
        lines.append("- No external ICU validation dataset was found inside the local repository/workspace, so robustness evidence remains internal.")
    lines.extend(
        [
            "",
            "## Deployment decision memo",
            "",
            "- Strengths: calibration-aware thresholding, group-aware patient holdout, threshold frontiers, feature ablations, subgroup diagnostics.",
            "- Limits: single-center MIMIC-derived cohort, unresolved ETL/documentation mismatches, no external validation dataset available locally.",
            "- Clinical stance: usable as a retrospective triage prototype, not ready for bedside deployment without external validation and schema reconciliation at the ETL layer.",
            "",
            "## Residual risks",
            "",
            "- The v2 cohort violates the documented first-ICU-stay assumption at the patient level, so patient-grouped evaluation is mandatory.",
            "- The current file contains process-rich variables that may not transport as well as physiology-only features.",
            "- RRT handling remains inconsistent between the dataset and the thesis methods text.",
            "",
            "## Subgroup table",
            "",
            render_table(subgroup_table, index=False, floatfmt=".3f"),
        ]
    )
    write_markdown(ARTIFACT_DIR / "deployment_report.md", "\n".join(lines))


def write_decision_memo(
    final_result: dict[str, Any],
    bootstrap_ci: dict[str, list[float]],
    cluster_bootstrap_ci: dict[str, list[float]] | None = None,
) -> None:
    lines = [
        "# Deployment decision memo",
        "",
        f"- Default high-risk threshold: `{final_result['primary_threshold'].get('threshold', float('nan')):.3f}`",
        f"- Default low-risk threshold: `{final_result['secondary_threshold'].get('threshold', float('nan')):.3f}`",
        f"- AUROC 95% bootstrap CI: `{bootstrap_ci['auroc'][0]:.3f}` to `{bootstrap_ci['auroc'][1]:.3f}`",
        f"- AUPRC 95% bootstrap CI: `{bootstrap_ci['auprc'][0]:.3f}` to `{bootstrap_ci['auprc'][1]:.3f}`",
        f"- Precision 95% bootstrap CI at deployed threshold: `{bootstrap_ci['precision'][0]:.3f}` to `{bootstrap_ci['precision'][1]:.3f}`",
        f"- Recall 95% bootstrap CI at deployed threshold: `{bootstrap_ci['recall'][0]:.3f}` to `{bootstrap_ci['recall'][1]:.3f}`",
        "",
    ]
    if cluster_bootstrap_ci is not None:
        lines.extend(
            [
                "## Subject-clustered uncertainty",
                "",
                f"- AUROC 95% clustered bootstrap CI: `{cluster_bootstrap_ci['auroc'][0]:.3f}` to `{cluster_bootstrap_ci['auroc'][1]:.3f}`",
                f"- Precision 95% clustered bootstrap CI: `{cluster_bootstrap_ci['precision'][0]:.3f}` to `{cluster_bootstrap_ci['precision'][1]:.3f}`",
                f"- Recall 95% clustered bootstrap CI: `{cluster_bootstrap_ci['recall'][0]:.3f}` to `{cluster_bootstrap_ci['recall'][1]:.3f}`",
                "",
            ]
        )
    lines.extend(
        [
        "## Recommendation",
        "",
        "- Treat the current model as an internally validated triage prototype.",
        "- Use the three-tier policy for retrospective simulation and workflow design only.",
        "- Do not claim bedside readiness until external validation and ETL/schema reconciliation are complete.",
        ]
    )
    write_markdown(ARTIFACT_DIR / "decision_memo.md", "\n".join(lines))


def run_workflow() -> None:
    ensure_artifact_dir()
    frame = add_secondary_horizon_labels(load_dataset(resolve_preferred_dataset()))
    audit = build_dataset_audit(frame)
    data_contract = build_data_contract(audit)
    feature_sets = build_feature_sets(frame)
    write_audit_report(audit, feature_sets)
    write_data_contract(data_contract)
    write_journal_positioning()

    calibration_benchmark = choose_best_calibration_per_model(
        frame,
        feature_sets["full"],
        split_name="subject_group_baseline",
        random_state=42,
        use_group_split=True,
    )
    calibration_benchmark.to_csv(ARTIFACT_DIR / "baseline_calibration_benchmark.csv", index=False)
    calibration_map = select_calibration_map(calibration_benchmark)

    repeated_results = repeated_group_model_comparison(frame, feature_sets, calibration_map)
    repeated_results.to_csv(ARTIFACT_DIR / "repeated_group_results.csv", index=False)
    selected_model = select_best_model(repeated_results)
    selected_calibration = calibration_map.get(selected_model, "sigmoid")

    score_cols = [column for column in ["sofa_total_24hr", "apache_iii_score"] if column in frame.columns]
    score_benchmarks = repeated_clinical_score_baselines(frame, score_cols)
    score_benchmarks.to_csv(ARTIFACT_DIR / "clinical_score_benchmarks.csv", index=False)
    score_operating_points = clinical_score_operating_points(frame, score_cols, random_state=42)
    score_operating_points.to_csv(ARTIFACT_DIR / "clinical_score_operating_points.csv", index=False)
    benchmark_inputs = [repeated_results.assign(model_family="ml_model")]
    if not score_benchmarks.empty:
        benchmark_inputs.append(score_benchmarks)
    benchmark_full = pd.concat(benchmark_inputs, ignore_index=True)
    benchmark_summary = flatten_columns(
        benchmark_full.groupby("model_name")[
            [
                "auroc",
                "auprc",
                "brier",
                "ece",
                "calibration_intercept",
                "calibration_slope",
                "recall_at_ppv_050",
            ]
        ]
        .agg(["mean", "std"])
        .reset_index()
    )
    benchmark_summary.to_csv(ARTIFACT_DIR / "benchmark_summary.csv", index=False)

    ceiling_table = prediction_ceiling_analysis(
        frame,
        feature_sets,
        model_name=selected_model,
        calibration_method=selected_calibration,
    )
    ceiling_table.to_csv(ARTIFACT_DIR / "prediction_ceiling_results.csv", index=False)
    plot_prediction_ceiling(ceiling_table, ARTIFACT_DIR / "prediction_ceiling.png")

    frontier_table = repeated_group_frontier(
        frame,
        feature_sets,
        model_name=selected_model,
        calibration_method=selected_calibration,
    )
    frontier_table.to_csv(ARTIFACT_DIR / "deployment_frontier.csv", index=False)
    write_threshold_objective_report(frontier_table)

    feature_ablation = feature_ablation_comparison(
        frame,
        feature_sets,
        model_name=selected_model,
        calibration_method=selected_calibration,
    )
    feature_ablation.to_csv(ARTIFACT_DIR / "feature_ablation_results.csv", index=False)

    selective_triage_table = repeated_selective_triage(
        frame,
        feature_sets,
        model_name=selected_model,
        calibration_method=selected_calibration,
    )
    selective_triage_table.to_csv(ARTIFACT_DIR / "selective_triage_results.csv", index=False)
    shift_table = run_shift_stress_tests(
        frame,
        feature_sets,
        model_name=selected_model,
        calibration_method=selected_calibration,
    )
    shift_table.to_csv(ARTIFACT_DIR / "selective_triage_shift_results.csv", index=False)
    write_selective_triage_report(selective_triage_table, shift_table)
    fixed_error = selective_triage_table.loc[
        selective_triage_table["policy_name"] == "fixed_threshold",
        "actionable_error_rate",
    ].mean()
    selective_error = selective_triage_table.loc[
        selective_triage_table["policy_name"] == "selective_triage",
        "actionable_error_rate",
    ].mean()
    core_signal_is_strong = bool(
        np.isfinite(fixed_error) and np.isfinite(selective_error) and selective_error < fixed_error
    )
    write_stretch_assessment(core_signal_is_strong)

    final_result = evaluate_on_outer_split(
        frame,
        feature_sets["full"],
        model_name=selected_model,
        calibration_method=selected_calibration,
        random_state=42,
        use_group_split=True,
    )

    comparison_results: dict[str, dict[str, Any]] = {selected_model: final_result}
    for model_name in ["logistic", "lightgbm", "xgboost"]:
        if model_name in comparison_results:
            continue
        comparison_results[model_name] = evaluate_on_outer_split(
            frame,
            feature_sets["full"],
            model_name=model_name,
            calibration_method=calibration_map.get(model_name, "sigmoid"),
            random_state=42,
            use_group_split=True,
        )
    score_comparison_results: list[dict[str, Any]] = []
    for score_col in score_cols:
        score_comparison_results.append(
            evaluate_score_baseline_on_outer_split(
                frame,
                score_col,
                random_state=42,
                use_group_split=True,
            )
        )

    write_json(
        ARTIFACT_DIR / "clinical_policy.json",
        {
            "selected_model": selected_model,
            "calibration": selected_calibration,
            "primary_threshold": final_result["primary_threshold"],
            "secondary_threshold": final_result["secondary_threshold"],
            "policy_rows": final_result["policy_rows"],
        },
    )

    bootstrap_ci = bootstrap_intervals(
        final_result["test_outer"][TARGET_COL],
        final_result["test_probs"],
        final_result["primary_threshold"]["threshold"],
    )
    write_json(ARTIFACT_DIR / "bootstrap_intervals.json", bootstrap_ci)
    cluster_bootstrap_ci = cluster_bootstrap_intervals(
        final_result["test_outer"][TARGET_COL],
        final_result["test_probs"],
        final_result["primary_threshold"]["threshold"],
        final_result["test_outer"][SUBJECT_ID_COL],
    )
    write_json(ARTIFACT_DIR / "cluster_bootstrap_intervals.json", cluster_bootstrap_ci)

    subgroup_table = subgroup_metrics(
        final_result["test_outer"],
        final_result["test_outer"][TARGET_COL],
        final_result["test_probs"],
        final_result["primary_threshold"]["threshold"],
    )
    subgroup_table.to_csv(ARTIFACT_DIR / "subgroup_metrics.csv", index=False)

    if EVALUATE_SECONDARY_HORIZONS:
        horizon_table = evaluate_secondary_horizons(
            frame,
            feature_sets["full"],
            model_name=selected_model,
            calibration_method=selected_calibration,
        )
    else:
        horizon_table = pd.DataFrame(
            [
                {
                    "label": "deferred_for_speed",
                    "prevalence": float("nan"),
                    "auroc": float("nan"),
                    "auprc": float("nan"),
                    "brier": float("nan"),
                    "ece": float("nan"),
                    "recall_at_primary_ppv": float("nan"),
                }
            ]
        )
    horizon_table.to_csv(ARTIFACT_DIR / "secondary_horizon_results.csv", index=False)

    top_features = top_feature_importance_table(
        final_result["model"],
        final_result["X_test_for_importance"],
        final_result["test_outer"][TARGET_COL],
        final_result["transformed_feature_names"],
        final_result["feature_space"],
    )
    top_features.to_csv(ARTIFACT_DIR / "feature_importance.csv", index=False)

    raw_test_frame = (
        final_result["feature_space"].build_frame(final_result["test_outer"])
        if final_result["feature_space"] is not None
        else monotonic_matrix(final_result["test_outer"])
    )
    tier_table = tier_feature_summary(
        raw_test_frame,
        final_result["tiers"],
        top_features.head(10).iloc[:, 0].tolist(),
    )
    tier_table.to_csv(ARTIFACT_DIR / "tier_feature_summary.csv", index=False)

    plot_precision_frontier(
        final_result["precision_frontier"],
        ARTIFACT_DIR / "precision_recall_frontier.png",
    )
    plot_calibration_curve(
        final_result["test_outer"][TARGET_COL],
        final_result["test_probs"],
        ARTIFACT_DIR / "calibration_curve.png",
    )
    roc_pr_curves = [
        {
            "label": selected_model,
            "y_true": final_result["test_outer"][TARGET_COL],
            "probs": final_result["test_probs"],
        }
    ]
    for model_name in ["logistic", "lightgbm", "xgboost"]:
        if model_name == selected_model:
            continue
        result = comparison_results[model_name]
        roc_pr_curves.append(
            {
                "label": model_name,
                "y_true": result["test_outer"][TARGET_COL],
                "probs": result["test_probs"],
            }
        )
    for result in score_comparison_results:
        roc_pr_curves.append(
            {
                "label": result["score_name"],
                "y_true": result["test_outer"][TARGET_COL],
                "probs": result["test_probs"],
            }
        )
    plot_roc_pr_benchmarks(roc_pr_curves, ARTIFACT_DIR / "roc_pr_benchmarks.png")
    utility_continuous, utility_policies, utility_summary, utility_policy_metrics = (
        build_clinical_utility_artifacts(
            frame,
            feature_sets,
            selected_model=selected_model,
            selected_calibration=selected_calibration,
            calibration_map=calibration_map,
            final_result=final_result,
            score_comparison_results=score_comparison_results,
        )
    )
    utility_continuous.to_csv(ARTIFACT_DIR / "decision_curve_continuous.csv", index=False)
    utility_policies.to_csv(ARTIFACT_DIR / "decision_curve_policies.csv", index=False)
    utility_summary.to_csv(ARTIFACT_DIR / "decision_curve_summary.csv", index=False)
    utility_policy_metrics.to_csv(ARTIFACT_DIR / "decision_curve_policy_metrics.csv", index=False)
    plotted_utility = pd.concat(
        [
            utility_continuous[
                utility_continuous["strategy"].isin(
                    [
                        f"{selected_model}_continuous",
                        "apache_iii_score_continuous",
                        "sofa_total_24hr_continuous",
                    ]
                )
            ],
            utility_policies[utility_policies["strategy"].isin(["treat_all", "treat_none"])],
        ],
        ignore_index=True,
    )
    plot_decision_curve(plotted_utility, ARTIFACT_DIR / "decision_curve.png")
    plot_clinical_utility_panel(
        final_result["test_outer"][TARGET_COL],
        final_result["test_probs"],
        plotted_utility,
        ARTIFACT_DIR / "clinical_utility_panel.png",
    )

    conformal_single_table = repeated_conformal_triage(
        frame,
        feature_sets,
        model_name=selected_model,
        calibration_method=selected_calibration,
    )
    conformal_single_table.to_csv(ARTIFACT_DIR / "conformal_single_model_results.csv", index=False)
    conformal_single_summary = summarize_metric_intervals(
        conformal_single_table,
        group_cols=["alpha"],
        value_cols=["coverage", "certain_frac", "alert_ppv", "clear_npv", "miss_count"],
    )
    conformal_single_summary.to_csv(ARTIFACT_DIR / "conformal_single_model_summary.csv", index=False)
    conformal_consensus_table = repeated_conformal_consensus(
        frame,
        feature_sets,
        calibration_map,
    )
    conformal_consensus_table.to_csv(ARTIFACT_DIR / "conformal_consensus_results.csv", index=False)
    conformal_consensus_summary = summarize_metric_intervals(
        conformal_consensus_table,
        group_cols=["ensemble_name", "base_model", "alpha"],
        value_cols=["coverage", "certain_frac", "alert_ppv", "clear_npv", "miss_count"],
    )
    conformal_consensus_summary.to_csv(ARTIFACT_DIR / "conformal_consensus_summary.csv", index=False)
    conformal_shift_table, fixed_threshold_shift_table = conformal_shift_stress_tests(
        frame,
        feature_sets,
        model_name=selected_model,
        calibration_method=selected_calibration,
    )
    conformal_shift_table.to_csv(ARTIFACT_DIR / "conformal_shift_results.csv", index=False)
    fixed_threshold_shift_table.to_csv(ARTIFACT_DIR / "fixed_threshold_shift_results.csv", index=False)
    conformal_operating_table = conformal_operating_curve(
        frame,
        feature_sets,
        model_name=selected_model,
        calibration_method=selected_calibration,
    )
    conformal_operating_table.to_csv(ARTIFACT_DIR / "conformal_operating_curve.csv", index=False)
    conformal_subgroup_table = conformal_subgroup_metrics(
        frame,
        feature_sets,
        model_name=selected_model,
        calibration_method=selected_calibration,
    )
    conformal_subgroup_table.to_csv(ARTIFACT_DIR / "conformal_subgroup_results.csv", index=False)
    plot_conformal_operating_curve(
        conformal_operating_table,
        ARTIFACT_DIR / "conformal_operating_curve.png",
    )
    plot_conformal_consensus_tradeoff(
        conformal_consensus_table,
        ARTIFACT_DIR / "conformal_consensus_tradeoff.png",
    )
    plot_conformal_shift_panel(
        conformal_shift_table,
        fixed_threshold_shift_table,
        ARTIFACT_DIR / "conformal_shift_panel.png",
    )
    if not conformal_subgroup_table.empty:
        plot_conformal_subgroup_forest(
            conformal_subgroup_table,
            ARTIFACT_DIR / "conformal_subgroup_forest.png",
        )
    write_conformal_report(
        conformal_single_table,
        conformal_consensus_table,
        conformal_shift_table,
        fixed_threshold_shift_table,
        conformal_operating_table,
        conformal_subgroup_table,
    )

    external_assets = search_external_validation_assets()
    clinical_score_summary = pd.DataFrame()
    if not score_benchmarks.empty:
        clinical_score_summary = flatten_columns(
            score_benchmarks.groupby("model_name")[
                ["auroc", "auprc", "brier", "ece", "recall_at_ppv_050"]
            ]
            .agg(["mean", "std"])
            .reset_index()
        )
        clinical_score_summary.to_csv(ARTIFACT_DIR / "clinical_score_benchmark_summary.csv", index=False)
    write_benchmark_report(
        benchmark_summary,
        ceiling_table,
        clinical_score_summary,
        score_operating_points,
    )
    write_clinical_utility_report(
        utility_continuous,
        utility_policies,
        utility_summary,
        utility_policy_metrics,
    )
    write_deployment_report(
        audit=audit,
        calibration_benchmark=calibration_benchmark,
        repeated_results=repeated_results,
        feature_ablation=feature_ablation,
        selected_model=selected_model,
        selected_calibration=selected_calibration,
        final_result=final_result,
        subgroup_table=subgroup_table,
        horizon_table=horizon_table,
        top_features=top_features,
        external_assets=external_assets,
    )
    write_decision_memo(final_result, bootstrap_ci, cluster_bootstrap_ci)

    log.info(
        "Completed deployment workflow with model=%s calibration=%s AUROC=%.3f AUPRC=%.3f",
        selected_model,
        selected_calibration,
        final_result["metrics"]["auroc"],
        final_result["metrics"]["auprc"],
    )


def main() -> None:
    configure_logging()
    run_workflow()


if __name__ == "__main__":
    main()
