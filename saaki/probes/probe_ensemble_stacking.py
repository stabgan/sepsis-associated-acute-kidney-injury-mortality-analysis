"""Probe 4: Stacked ensemble — can we break the 0.80 AUROC barrier?

Stack LightGBM, XGBoost, Logistic, and optionally TabPFN predictions
using a meta-learner.
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
    TARGET_COL, SUBJECT_ID_COL, safe_auroc, safe_auprc, threshold_search,
)

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.calibration import CalibratedClassifierCV

OUT = Path(__file__).resolve().parent.parent.parent / "local_outputs" / "probes"
OUT.mkdir(parents=True, exist_ok=True)

frame = add_secondary_horizon_labels(load_dataset(resolve_preferred_dataset()))
feature_sets = build_feature_sets(frame)

train_outer, test_outer = stratified_group_split(
    frame, group_col=SUBJECT_ID_COL, label_col=TARGET_COL,
    test_size=0.20, random_state=42,
)

feature_space = fit_feature_space(train_outer, feature_sets["full"], train_outer[TARGET_COL])
X_train = feature_space.transform_frame(train_outer).values
X_test = feature_space.transform_frame(test_outer).values
y_train = train_outer[TARGET_COL].astype(int).values
y_test = test_outer[TARGET_COL].astype(int).values

results = {}

estimators = [
    ("lgbm", LGBMClassifier(
        objective="binary", n_estimators=350, learning_rate=0.05,
        num_leaves=31, class_weight="balanced", n_jobs=32,
        random_state=42, verbose=-1,
    )),
    ("xgb", XGBClassifier(
        objective="binary:logistic", n_estimators=350, learning_rate=0.05,
        max_depth=4, scale_pos_weight=2.7, tree_method="hist",
        n_jobs=32, random_state=42,
    )),
    ("lr", LogisticRegression(
        max_iter=3000, class_weight="balanced", solver="lbfgs", random_state=42,
    )),
]

t0 = time.time()
stacker = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(max_iter=1000, solver="lbfgs"),
    cv=3,
    stack_method="predict_proba",
    n_jobs=3,
)
stacker.fit(X_train, y_train)
probs = stacker.predict_proba(X_test)[:, 1]
elapsed = time.time() - t0
auroc = safe_auroc(y_test, probs)
auprc = safe_auprc(y_test, probs)
ppv50 = threshold_search(y_test, probs, precision_target=0.50)
ppv60 = threshold_search(y_test, probs, precision_target=0.60)
results["stacked_3model"] = {
    "auroc": auroc, "auprc": auprc,
    "recall_at_ppv050": ppv50.get("recall", float("nan")),
    "recall_at_ppv060": ppv60.get("recall", float("nan")),
    "elapsed_s": round(elapsed, 1),
}
print(f"Stacked (3-model): AUROC={auroc:.4f}  AUPRC={auprc:.4f}  recall@PPV50={ppv50.get('recall', float('nan')):.4f}  time={elapsed:.1f}s")

# Also try individual models for comparison
for name, est in estimators:
    t0 = time.time()
    est.fit(X_train, y_train)
    probs_i = est.predict_proba(X_test)[:, 1]
    elapsed = time.time() - t0
    auroc_i = safe_auroc(y_test, probs_i)
    ppv50_i = threshold_search(y_test, probs_i, precision_target=0.50)
    results[f"individual_{name}"] = {
        "auroc": auroc_i,
        "recall_at_ppv050": ppv50_i.get("recall", float("nan")),
        "elapsed_s": round(elapsed, 1),
    }
    print(f"Individual {name}: AUROC={auroc_i:.4f}  recall@PPV50={ppv50_i.get('recall', float('nan')):.4f}")

with open(OUT / "probe_ensemble_stacking.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nSaved to {OUT / 'probe_ensemble_stacking.json'}")
