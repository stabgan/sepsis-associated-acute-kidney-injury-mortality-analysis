"""Probe 1: TabPFN zero-shot vs trained LightGBM on the same grouped split.

If a foundation model with NO training on this distribution gets close to
a trained GBDT, that's a headline finding. If it's far behind, we know
the ceiling requires domain-specific training.
"""
import os, sys, time, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "96"
os.environ["OPENBLAS_NUM_THREADS"] = "96"
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deployment_analysis import (
    load_dataset, resolve_preferred_dataset, add_secondary_horizon_labels,
    build_feature_sets, stratified_group_split, fit_feature_space,
    TARGET_COL, SUBJECT_ID_COL, safe_auroc, safe_auprc,
    binary_metrics, threshold_search,
)

OUT = Path(__file__).resolve().parent.parent.parent / "local_outputs" / "probes"
OUT.mkdir(parents=True, exist_ok=True)

frame = add_secondary_horizon_labels(load_dataset(resolve_preferred_dataset()))
feature_sets = build_feature_sets(frame)

train, test = stratified_group_split(
    frame, group_col=SUBJECT_ID_COL, label_col=TARGET_COL,
    test_size=0.20, random_state=42,
)

feature_space = fit_feature_space(train, feature_sets["full"], train[TARGET_COL])
X_train = feature_space.transform_frame(train)
X_test = feature_space.transform_frame(test)
y_train = train[TARGET_COL].astype(int)
y_test = test[TARGET_COL].astype(int)

results = {}

# --- TabPFN ---
t0 = time.time()
try:
    from tabpfn import TabPFNClassifier
    n_train = min(len(X_train), 10000)
    n_features = min(X_train.shape[1], 500)
    idx = np.random.default_rng(42).choice(len(X_train), n_train, replace=False)
    feat_idx = np.argsort(X_train.iloc[idx].var())[-n_features:]
    X_tr_sub = X_train.iloc[idx, feat_idx].values
    X_te_sub = X_test.iloc[:, feat_idx].values
    y_tr_sub = y_train.iloc[idx].values

    clf = TabPFNClassifier(device="cpu", n_estimators=8)
    clf.fit(X_tr_sub, y_tr_sub)
    probs = clf.predict_proba(X_te_sub)[:, 1]
    elapsed = time.time() - t0
    results["tabpfn"] = {
        "auroc": safe_auroc(y_test, probs),
        "auprc": safe_auprc(y_test, probs),
        **binary_metrics(y_test, probs, 0.5),
        "elapsed_s": round(elapsed, 1),
        "n_train": n_train,
        "n_features": n_features,
    }
    ppv50 = threshold_search(y_test, probs, precision_target=0.50)
    results["tabpfn"]["recall_at_ppv050"] = ppv50.get("recall", float("nan"))
    print(f"TabPFN: AUROC={results['tabpfn']['auroc']:.4f}  AUPRC={results['tabpfn']['auprc']:.4f}  recall@PPV50={results['tabpfn']['recall_at_ppv050']:.4f}  time={elapsed:.1f}s")
except Exception as e:
    results["tabpfn"] = {"error": str(e)}
    print(f"TabPFN FAILED: {e}")

# --- LightGBM baseline (same split) ---
from lightgbm import LGBMClassifier
t0 = time.time()
lgbm = LGBMClassifier(
    objective="binary", n_estimators=350, learning_rate=0.05,
    num_leaves=31, class_weight="balanced", n_jobs=96,
    random_state=42, verbose=-1,
)
lgbm.fit(X_train.values, y_train.values)
probs_lgbm = lgbm.predict_proba(X_test.values)[:, 1]
elapsed = time.time() - t0
results["lightgbm_default"] = {
    "auroc": safe_auroc(y_test, probs_lgbm),
    "auprc": safe_auprc(y_test, probs_lgbm),
    **binary_metrics(y_test, probs_lgbm, 0.5),
    "elapsed_s": round(elapsed, 1),
}
ppv50 = threshold_search(y_test, probs_lgbm, precision_target=0.50)
results["lightgbm_default"]["recall_at_ppv050"] = ppv50.get("recall", float("nan"))
print(f"LightGBM: AUROC={results['lightgbm_default']['auroc']:.4f}  recall@PPV50={results['lightgbm_default']['recall_at_ppv050']:.4f}  time={elapsed:.1f}s")

with open(OUT / "probe_tabpfn.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nSaved to {OUT / 'probe_tabpfn.json'}")
