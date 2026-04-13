"""Probe 6: Survival-aware modeling.

time_to_event_hrs is available — can we use a survival model (CoxPH, 
Random Survival Forest) and derive risk scores that beat flat classification?
Also: multi-horizon prediction (24h, 48h, 72h, 168h, full-stay mortality).
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
    build_feature_sets, stratified_group_split, fit_feature_space,
    TARGET_COL, TIME_COL, SUBJECT_ID_COL, safe_auroc,
    candidate_feature_columns, columns_above_missingness,
    MISSINGNESS_DROP_THRESHOLD, COHORT_TIME_COLS,
)

from sklearn.impute import SimpleImputer

OUT = Path(__file__).resolve().parent.parent.parent / "local_outputs" / "probes"
OUT.mkdir(parents=True, exist_ok=True)

frame = add_secondary_horizon_labels(load_dataset(resolve_preferred_dataset()))
all_features = candidate_feature_columns(frame, include_process=True)
gt99_drop = set(columns_above_missingness(frame, all_features, MISSINGNESS_DROP_THRESHOLD))
numeric_cols = [c for c in all_features if c not in gt99_drop and c not in COHORT_TIME_COLS and pd.api.types.is_numeric_dtype(frame[c])]

train, test = stratified_group_split(
    frame, group_col=SUBJECT_ID_COL, label_col=TARGET_COL,
    test_size=0.20, random_state=42,
)

imp = SimpleImputer(strategy="median")
X_train = pd.DataFrame(imp.fit_transform(train[numeric_cols]), columns=numeric_cols, index=train.index)
X_test = pd.DataFrame(imp.transform(test[numeric_cols]), columns=numeric_cols, index=test.index)
y_train = train[TARGET_COL].astype(int)
y_test = test[TARGET_COL].astype(int)
time_train = train[TIME_COL].values
time_test = test[TIME_COL].values

results = {}

# --- scikit-survival CoxPH + RSF ---
try:
    from sksurv.ensemble import RandomSurvivalForest
    from sksurv.linear_model import CoxPHSurvivalAnalysis
    from sksurv.metrics import concordance_index_censored

    y_surv_train = np.array(
        [(bool(e), float(t)) for e, t in zip(y_train.values, time_train)],
        dtype=[("event", bool), ("time", float)],
    )
    y_surv_test = np.array(
        [(bool(e), float(t)) for e, t in zip(y_test.values, time_test)],
        dtype=[("event", bool), ("time", float)],
    )

    # CoxPH (penalized — alpha selection)
    t0 = time.time()
    cox = CoxPHSurvivalAnalysis(alpha=0.1, n_iter=300)
    cox.fit(X_train.values, y_surv_train)
    risk_cox = cox.predict(X_test.values)
    ci_cox = concordance_index_censored(y_surv_test["event"], y_surv_test["time"], risk_cox)
    auroc_cox = safe_auroc(y_test, risk_cox)
    elapsed = time.time() - t0
    results["cox_ph"] = {
        "c_index": ci_cox[0],
        "auroc_vs_binary": auroc_cox,
        "elapsed_s": round(elapsed, 1),
    }
    print(f"CoxPH: C-index={ci_cox[0]:.4f}  AUROC(binary)={auroc_cox:.4f}  time={elapsed:.1f}s")

    # Random Survival Forest
    t0 = time.time()
    rsf = RandomSurvivalForest(
        n_estimators=200, max_depth=6, min_samples_leaf=10,
        n_jobs=96, random_state=42,
    )
    rsf.fit(X_train.values, y_surv_train)
    risk_rsf = rsf.predict(X_test.values)
    ci_rsf = concordance_index_censored(y_surv_test["event"], y_surv_test["time"], risk_rsf)
    auroc_rsf = safe_auroc(y_test, risk_rsf)
    elapsed = time.time() - t0
    results["rsf"] = {
        "c_index": ci_rsf[0],
        "auroc_vs_binary": auroc_rsf,
        "elapsed_s": round(elapsed, 1),
    }
    print(f"RSF: C-index={ci_rsf[0]:.4f}  AUROC(binary)={auroc_rsf:.4f}  time={elapsed:.1f}s")

    # Multi-horizon: survival function predictions at specific timepoints
    horizons_hrs = [24, 48, 72, 168]
    surv_fn = rsf.predict_survival_function(X_test.values)
    for h in horizons_hrs:
        mortality_at_h = np.array([1.0 - fn(h) if h <= fn.x[-1] else 1.0 for fn in surv_fn])
        y_actual = (y_test.values == 1) & (time_test <= h)
        auroc_h = safe_auroc(y_actual.astype(int), mortality_at_h)
        results[f"rsf_horizon_{h}h"] = {"auroc": auroc_h, "n_events": int(y_actual.sum())}
        print(f"RSF @ {h}h: AUROC={auroc_h:.4f}  n_events={y_actual.sum()}")

except ImportError as e:
    results["survival_error"] = str(e)
    print(f"scikit-survival not available: {e}")
except Exception as e:
    results["survival_error"] = str(e)
    print(f"Survival analysis error: {e}")

# --- Multi-horizon classification (direct LightGBM) ---
from lightgbm import LGBMClassifier

for h in [48, 72, 168]:
    col = f"event_observed_{h}h"
    if col in frame.columns:
        y_tr_h = train[col].astype(int)
        y_te_h = test[col].astype(int)
        clf = LGBMClassifier(
            objective="binary", n_estimators=350, learning_rate=0.05,
            num_leaves=31, class_weight="balanced", n_jobs=96,
            random_state=42, verbose=-1,
        )
        clf.fit(X_train.values, y_tr_h.values)
        probs = clf.predict_proba(X_test.values)[:, 1]
        auroc = safe_auroc(y_te_h, probs)
        prevalence = y_te_h.mean()
        results[f"lgbm_horizon_{h}h"] = {"auroc": auroc, "prevalence": prevalence}
        print(f"LightGBM @ {h}h: AUROC={auroc:.4f}  prevalence={prevalence:.3f}")

with open(OUT / "probe_survival_aware.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nSaved to {OUT / 'probe_survival_aware.json'}")
