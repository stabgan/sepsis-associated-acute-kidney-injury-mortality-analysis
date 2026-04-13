"""Probe 3: Feature engineering — missingness fingerprints, ratios, interactions.

Test whether engineered features can push the ceiling above 0.77.
"""
import os, sys, time, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "96"
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deployment_analysis import (
    load_dataset, resolve_preferred_dataset, add_secondary_horizon_labels,
    build_feature_sets, stratified_group_split, candidate_feature_columns,
    columns_above_missingness, COHORT_TIME_COLS,
    TARGET_COL, SUBJECT_ID_COL, safe_auroc, safe_auprc, threshold_search,
    MISSINGNESS_DROP_THRESHOLD,
)

from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

OUT = Path(__file__).resolve().parent.parent.parent / "local_outputs" / "probes"
OUT.mkdir(parents=True, exist_ok=True)

frame = add_secondary_horizon_labels(load_dataset(resolve_preferred_dataset()))

train, test = stratified_group_split(
    frame, group_col=SUBJECT_ID_COL, label_col=TARGET_COL,
    test_size=0.20, random_state=42,
)
y_train = train[TARGET_COL].astype(int)
y_test = test[TARGET_COL].astype(int)

all_features = candidate_feature_columns(frame, include_process=True)
gt99_drop = set(columns_above_missingness(frame, all_features, MISSINGNESS_DROP_THRESHOLD))
usable = [c for c in all_features if c not in gt99_drop and c not in COHORT_TIME_COLS]
numeric_cols = [c for c in usable if pd.api.types.is_numeric_dtype(frame[c])]

def prepare_simple(train_df, test_df, cols):
    imp = SimpleImputer(strategy="median")
    X_tr = pd.DataFrame(imp.fit_transform(train_df[cols]), columns=cols, index=train_df.index)
    X_te = pd.DataFrame(imp.transform(test_df[cols]), columns=cols, index=test_df.index)
    return X_tr, X_te

def add_missingness_fingerprint(X_tr, X_te, original_train, original_test, cols):
    for c in cols:
        X_tr[f"{c}__miss"] = original_train[c].isna().astype(int).values
        X_te[f"{c}__miss"] = original_test[c].isna().astype(int).values
    miss_cols_tr = [c for c in cols if c in original_train.columns]
    X_tr["total_missing"] = original_train[miss_cols_tr].isna().sum(axis=1).values
    X_te["total_missing"] = original_test[miss_cols_tr].isna().sum(axis=1).values
    X_tr["missing_frac"] = X_tr["total_missing"] / len(miss_cols_tr)
    X_te["missing_frac"] = X_te["total_missing"] / len(miss_cols_tr)
    return X_tr, X_te

def add_clinical_ratios(X_tr, X_te):
    ratio_pairs = [
        ("bun_median", "creatinine_max", "bun_cr_ratio"),
        ("lactate_max", "bicarbonate_min", "lactate_bicarb_ratio"),
        ("sofa_total_24hr", "apache_iii_score", "sofa_apache_ratio"),
        ("urine_output_total_24hr", "fluid_balance_24hr_ml", "uo_fluid_ratio"),
        ("plateletcount_min", "wbc_max_10e9l", "plt_wbc_ratio"),
        ("ast_max", "alt_median", "ast_alt_ratio"),
        ("inr_max", "plateletcount_min", "inr_plt_ratio"),
    ]
    for num, den, name in ratio_pairs:
        if num in X_tr.columns and den in X_tr.columns:
            X_tr[name] = X_tr[num] / (X_tr[den] + 1e-6)
            X_te[name] = X_te[num] / (X_te[den] + 1e-6)
    return X_tr, X_te

def add_delta_velocity(X_tr, X_te):
    base_names = ["heartrate", "sysbp", "meanbp", "resprate", "spo2", "creatinine", "lactate"]
    for base in base_names:
        first_col = f"{base}_first"
        last_col = f"{base}_last"
        median_col = f"{base}_median"
        if first_col in X_tr.columns and last_col in X_tr.columns:
            X_tr[f"{base}_abs_delta"] = (X_tr[last_col] - X_tr[first_col]).abs()
            X_te[f"{base}_abs_delta"] = (X_te[last_col] - X_te[first_col]).abs()
        if median_col in X_tr.columns and first_col in X_tr.columns:
            X_tr[f"{base}_deviation"] = (X_tr[median_col] - X_tr[first_col]).abs()
            X_te[f"{base}_deviation"] = (X_te[median_col] - X_te[first_col]).abs()
    return X_tr, X_te

def add_organ_dysfunction_composite(X_tr, X_te):
    sofa_cols = [c for c in X_tr.columns if c.startswith("sofa_") and c.endswith("_24hr") and c != "sofa_total_24hr"]
    if sofa_cols:
        X_tr["sofa_max_component"] = X_tr[sofa_cols].max(axis=1)
        X_te["sofa_max_component"] = X_te[sofa_cols].max(axis=1)
        X_tr["sofa_nonzero_count"] = (X_tr[sofa_cols] > 0).sum(axis=1)
        X_te["sofa_nonzero_count"] = (X_te[sofa_cols] > 0).sum(axis=1)
    charlson_cols = [c for c in X_tr.columns if c.startswith("charlson_")]
    if charlson_cols:
        X_tr["charlson_total"] = X_tr[charlson_cols].sum(axis=1)
        X_te["charlson_total"] = X_te[charlson_cols].sum(axis=1)
    return X_tr, X_te

def eval_lgbm(X_tr, X_te, y_tr, y_te, label):
    clf = LGBMClassifier(
        objective="binary", n_estimators=350, learning_rate=0.05,
        num_leaves=31, class_weight="balanced", n_jobs=96,
        random_state=42, verbose=-1,
    )
    t0 = time.time()
    clf.fit(X_tr.values, y_tr.values)
    probs = clf.predict_proba(X_te.values)[:, 1]
    elapsed = time.time() - t0
    auroc = safe_auroc(y_te, probs)
    auprc = safe_auprc(y_te, probs)
    ppv50 = threshold_search(y_te, probs, precision_target=0.50)
    ppv60 = threshold_search(y_te, probs, precision_target=0.60)
    print(f"{label}: AUROC={auroc:.4f}  AUPRC={auprc:.4f}  recall@PPV50={ppv50.get('recall', float('nan')):.4f}  n_features={X_tr.shape[1]}  time={elapsed:.1f}s")
    return {
        "auroc": auroc, "auprc": auprc,
        "recall_at_ppv050": ppv50.get("recall", float("nan")),
        "recall_at_ppv060": ppv60.get("recall", float("nan")),
        "n_features": X_tr.shape[1],
        "elapsed_s": round(elapsed, 1),
    }

results = {}

X_tr_base, X_te_base = prepare_simple(train, test, numeric_cols)
results["baseline_numeric"] = eval_lgbm(X_tr_base, X_te_base, y_train, y_test, "Baseline (numeric only)")

X_tr_miss, X_te_miss = X_tr_base.copy(), X_te_base.copy()
X_tr_miss, X_te_miss = add_missingness_fingerprint(X_tr_miss, X_te_miss, train, test, numeric_cols)
results["with_missingness"] = eval_lgbm(X_tr_miss, X_te_miss, y_train, y_test, "+ Missingness fingerprints")

X_tr_ratio, X_te_ratio = X_tr_miss.copy(), X_te_miss.copy()
X_tr_ratio, X_te_ratio = add_clinical_ratios(X_tr_ratio, X_te_ratio)
results["with_ratios"] = eval_lgbm(X_tr_ratio, X_te_ratio, y_train, y_test, "+ Clinical ratios")

X_tr_delta, X_te_delta = X_tr_ratio.copy(), X_te_ratio.copy()
X_tr_delta, X_te_delta = add_delta_velocity(X_tr_delta, X_te_delta)
results["with_deltas"] = eval_lgbm(X_tr_delta, X_te_delta, y_train, y_test, "+ Delta velocity")

X_tr_organ, X_te_organ = X_tr_delta.copy(), X_te_delta.copy()
X_tr_organ, X_te_organ = add_organ_dysfunction_composite(X_tr_organ, X_te_organ)
results["with_organ_composites"] = eval_lgbm(X_tr_organ, X_te_organ, y_train, y_test, "+ Organ composites (FULL)")

with open(OUT / "probe_feature_engineering.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nSaved to {OUT / 'probe_feature_engineering.json'}")
