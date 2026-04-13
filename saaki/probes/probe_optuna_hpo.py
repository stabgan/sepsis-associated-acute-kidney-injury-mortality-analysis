"""Probe 2: Optuna HPO on LightGBM with PostgreSQL storage.

The current script uses hand-picked hyperparameters. With 96 cores and
Optuna, we can run 200+ trials in minutes. If HPO pushes AUROC from
0.77 to 0.80+, that changes the paper story.
"""
import os, sys, time, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "4"
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deployment_analysis import (
    load_dataset, resolve_preferred_dataset, add_secondary_horizon_labels,
    build_feature_sets, stratified_group_split, fit_feature_space,
    TARGET_COL, SUBJECT_ID_COL, safe_auroc, threshold_search,
)

import optuna
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV

OUT = Path(__file__).resolve().parent.parent.parent / "local_outputs" / "probes"
OUT.mkdir(parents=True, exist_ok=True)

frame = add_secondary_horizon_labels(load_dataset(resolve_preferred_dataset()))
feature_sets = build_feature_sets(frame)

train_outer, test_outer = stratified_group_split(
    frame, group_col=SUBJECT_ID_COL, label_col=TARGET_COL,
    test_size=0.20, random_state=42,
)
train_model, val_model = stratified_group_split(
    train_outer, group_col=SUBJECT_ID_COL, label_col=TARGET_COL,
    test_size=0.20, random_state=42,
)

feature_space = fit_feature_space(train_model, feature_sets["full"], train_model[TARGET_COL])
X_train = feature_space.transform_frame(train_model).values
X_val = feature_space.transform_frame(val_model).values
X_train_outer = fit_feature_space(train_outer, feature_sets["full"], train_outer[TARGET_COL])
X_test_fs = X_train_outer.transform_frame(test_outer).values
X_train_outer_vals = X_train_outer.transform_frame(train_outer).values
y_train = train_model[TARGET_COL].astype(int).values
y_val = val_model[TARGET_COL].astype(int).values
y_train_outer = train_outer[TARGET_COL].astype(int).values
y_test = test_outer[TARGET_COL].astype(int).values

def objective(trial):
    params = {
        "objective": "binary",
        "n_estimators": trial.suggest_int("n_estimators", 200, 1200),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "class_weight": "balanced",
        "n_jobs": 4,
        "random_state": 42,
        "verbose": -1,
    }
    clf = LGBMClassifier(**params)
    clf.fit(X_train, y_train)
    probs = clf.predict_proba(X_val)[:, 1]
    auroc = safe_auroc(y_val, probs)
    ppv50 = threshold_search(y_val, probs, precision_target=0.50)
    recall = ppv50.get("recall", 0.0)
    return auroc + 0.3 * recall

storage_url = "postgresql://optuna:optuna@127.0.0.1/optuna"
study_name = "saaki_lgbm_hpo_v1"

try:
    optuna.delete_study(study_name=study_name, storage=storage_url)
except:
    pass

study = optuna.create_study(
    study_name=study_name,
    storage=storage_url,
    direction="maximize",
    load_if_exists=False,
)

t0 = time.time()
study.optimize(objective, n_trials=200, n_jobs=24, show_progress_bar=False)
elapsed = time.time() - t0

best = study.best_trial
print(f"\nOptuna done: {len(study.trials)} trials in {elapsed:.1f}s")
print(f"Best objective: {best.value:.4f}")
print(f"Best params: {best.params}")

best_params = {
    **best.params,
    "objective": "binary",
    "class_weight": "balanced",
    "n_jobs": 96,
    "random_state": 42,
    "verbose": -1,
}
final_clf = LGBMClassifier(**best_params)
final_clf.fit(X_train_outer_vals, y_train_outer)
test_probs = final_clf.predict_proba(X_test_fs)[:, 1]
test_auroc = safe_auroc(y_test, test_probs)
ppv50_test = threshold_search(y_test, test_probs, precision_target=0.50)

cal_clf = CalibratedClassifierCV(LGBMClassifier(**best_params), method="sigmoid", cv=3)
cal_clf.fit(X_train_outer_vals, y_train_outer)
cal_probs = cal_clf.predict_proba(X_test_fs)[:, 1]
cal_auroc = safe_auroc(y_test, cal_probs)
cal_ppv50 = threshold_search(y_test, cal_probs, precision_target=0.50)

results = {
    "n_trials": len(study.trials),
    "elapsed_s": round(elapsed, 1),
    "best_objective": best.value,
    "best_params": best.params,
    "test_auroc_uncalibrated": test_auroc,
    "test_recall_at_ppv050_uncalibrated": ppv50_test.get("recall", float("nan")),
    "test_auroc_calibrated": cal_auroc,
    "test_recall_at_ppv050_calibrated": cal_ppv50.get("recall", float("nan")),
}
print(f"Test AUROC (uncal): {test_auroc:.4f}  recall@PPV50: {ppv50_test.get('recall', float('nan')):.4f}")
print(f"Test AUROC (cal):   {cal_auroc:.4f}  recall@PPV50: {cal_ppv50.get('recall', float('nan')):.4f}")

with open(OUT / "probe_optuna_hpo.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"Saved to {OUT / 'probe_optuna_hpo.json'}")
