"""Probe 7: Conformal prediction — distribution-free coverage guarantees.

If we can say "when the model says Alert, it's correct >= 95% of the time
with finite-sample guarantees," that's a real contribution.
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
    TARGET_COL, SUBJECT_ID_COL, safe_auroc,
)

from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV

OUT = Path(__file__).resolve().parent.parent.parent / "local_outputs" / "probes"
OUT.mkdir(parents=True, exist_ok=True)

frame = add_secondary_horizon_labels(load_dataset(resolve_preferred_dataset()))
feature_sets = build_feature_sets(frame)

train_full, test = stratified_group_split(
    frame, group_col=SUBJECT_ID_COL, label_col=TARGET_COL,
    test_size=0.20, random_state=42,
)
train, cal = stratified_group_split(
    train_full, group_col=SUBJECT_ID_COL, label_col=TARGET_COL,
    test_size=0.25, random_state=42,
)

feature_space = fit_feature_space(train, feature_sets["full"], train[TARGET_COL])
X_train = feature_space.transform_frame(train).values
X_cal = feature_space.transform_frame(cal).values
X_test = feature_space.transform_frame(test).values
y_train = train[TARGET_COL].astype(int).values
y_cal = cal[TARGET_COL].astype(int).values
y_test = test[TARGET_COL].astype(int).values

clf = LGBMClassifier(
    objective="binary", n_estimators=350, learning_rate=0.05,
    num_leaves=31, class_weight="balanced", n_jobs=96,
    random_state=42, verbose=-1,
)
clf.fit(X_train, y_train)

results = {}

# --- LAC (Least Ambiguous set-valued Classifier) / Simple split conformal ---
def split_conformal_classification(cal_probs, cal_labels, test_probs, alpha=0.10):
    """
    Adaptive prediction sets for binary classification.
    Returns set predictions: {0}, {1}, or {0,1} for each test point.
    """
    n_cal = len(cal_labels)
    scores_cal = np.zeros(n_cal)
    for i in range(n_cal):
        true_label = cal_labels[i]
        scores_cal[i] = 1.0 - (cal_probs[i] if true_label == 1 else (1 - cal_probs[i]))

    q_level = np.ceil((n_cal + 1) * (1 - alpha)) / n_cal
    q_hat = np.quantile(scores_cal, min(q_level, 1.0))

    sets = []
    for p in test_probs:
        s = set()
        if (1 - p) <= q_hat:  # score for class 0
            s.add(0)
        if (1 - (1 - p)) <= q_hat:  # score for class 1, i.e., p <= q_hat? No...
            pass
        s_0 = 1.0 - (1.0 - p)  # nonconformity if true class = 0
        s_1 = 1.0 - p  # nonconformity if true class = 1
        s = set()
        if s_0 <= q_hat:
            s.add(0)
        if s_1 <= q_hat:
            s.add(1)
        sets.append(s)
    return sets, q_hat

cal_probs = clf.predict_proba(X_cal)[:, 1]
test_probs = clf.predict_proba(X_test)[:, 1]

for alpha in [0.05, 0.10, 0.15, 0.20]:
    pred_sets, q_hat = split_conformal_classification(cal_probs, y_cal, test_probs, alpha)
    
    coverage = np.mean([y_test[i] in s for i, s in enumerate(pred_sets)])
    avg_size = np.mean([len(s) for s in pred_sets])
    n_certain_0 = sum(1 for s in pred_sets if s == {0})
    n_certain_1 = sum(1 for s in pred_sets if s == {1})
    n_uncertain = sum(1 for s in pred_sets if len(s) == 2)
    n_empty = sum(1 for s in pred_sets if len(s) == 0)
    
    # Among patients assigned {1} (Alert), what fraction actually died?
    alert_idx = [i for i, s in enumerate(pred_sets) if s == {1}]
    if alert_idx:
        alert_precision = y_test[alert_idx].mean()
    else:
        alert_precision = float("nan")
    
    # Among patients assigned {0} (Clear), what fraction survived?
    clear_idx = [i for i, s in enumerate(pred_sets) if s == {0}]
    if clear_idx:
        clear_npv = (1 - y_test[clear_idx]).mean()
    else:
        clear_npv = float("nan")
    
    results[f"conformal_alpha_{alpha}"] = {
        "alpha": alpha,
        "coverage": round(coverage, 4),
        "avg_set_size": round(avg_size, 4),
        "n_certain_death": n_certain_1,
        "n_certain_survive": n_certain_0,
        "n_uncertain": n_uncertain,
        "n_empty": n_empty,
        "alert_precision": round(alert_precision, 4) if not np.isnan(alert_precision) else None,
        "clear_npv": round(clear_npv, 4) if not np.isnan(clear_npv) else None,
        "q_hat": round(q_hat, 4),
    }
    print(f"alpha={alpha}: coverage={coverage:.4f}  avg_size={avg_size:.2f}  "
          f"certain_death={n_certain_1}  certain_survive={n_certain_0}  "
          f"uncertain={n_uncertain}  empty={n_empty}  "
          f"alert_PPV={alert_precision:.4f}  clear_NPV={clear_npv:.4f}")

# --- Mondrian conformal (class-conditional) ---
def mondrian_conformal(cal_probs, cal_labels, test_probs, alpha=0.10):
    """Class-conditional conformal for better per-class coverage."""
    scores_0 = 1.0 - (1.0 - cal_probs[cal_labels == 0])
    scores_1 = 1.0 - cal_probs[cal_labels == 1]
    
    n0 = len(scores_0)
    n1 = len(scores_1)
    q0 = np.quantile(scores_0, min(np.ceil((n0 + 1) * (1 - alpha)) / n0, 1.0))
    q1 = np.quantile(scores_1, min(np.ceil((n1 + 1) * (1 - alpha)) / n1, 1.0))
    
    sets = []
    for p in test_probs:
        s = set()
        if (1.0 - (1.0 - p)) <= q0:  # nonconformity for class 0
            s.add(0)
        if (1.0 - p) <= q1:  # nonconformity for class 1
            s.add(1)
        sets.append(s)
    return sets, q0, q1

for alpha in [0.05, 0.10, 0.15]:
    pred_sets, q0, q1 = mondrian_conformal(cal_probs, y_cal, test_probs, alpha)
    coverage = np.mean([y_test[i] in s for i, s in enumerate(pred_sets)])
    n_certain_0 = sum(1 for s in pred_sets if s == {0})
    n_certain_1 = sum(1 for s in pred_sets if s == {1})
    n_uncertain = sum(1 for s in pred_sets if len(s) == 2)
    
    alert_idx = [i for i, s in enumerate(pred_sets) if s == {1}]
    alert_precision = y_test[alert_idx].mean() if alert_idx else float("nan")
    clear_idx = [i for i, s in enumerate(pred_sets) if s == {0}]
    clear_npv = (1 - y_test[clear_idx]).mean() if clear_idx else float("nan")
    
    results[f"mondrian_alpha_{alpha}"] = {
        "coverage": round(coverage, 4),
        "n_certain_death": n_certain_1,
        "n_certain_survive": n_certain_0,
        "n_uncertain": n_uncertain,
        "alert_precision": round(alert_precision, 4) if not np.isnan(alert_precision) else None,
        "clear_npv": round(clear_npv, 4) if not np.isnan(clear_npv) else None,
    }
    print(f"Mondrian alpha={alpha}: coverage={coverage:.4f}  "
          f"certain_death={n_certain_1}  certain_survive={n_certain_0}  "
          f"uncertain={n_uncertain}  alert_PPV={alert_precision:.4f}  clear_NPV={clear_npv:.4f}")

# Also: base model AUROC for reference
auroc_base = safe_auroc(y_test, test_probs)
results["base_auroc"] = auroc_base
print(f"\nBase model AUROC: {auroc_base:.4f}")

with open(OUT / "probe_conformal.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nSaved to {OUT / 'probe_conformal.json'}")
